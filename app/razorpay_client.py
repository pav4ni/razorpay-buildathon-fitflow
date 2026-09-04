"""
razorpay_client.py — thin wrapper around the Razorpay Python SDK (test mode).

Two jobs, and deliberately nothing else:
  1. Convert rupees to paise correctly. Razorpay's API takes the *smallest*
     currency unit, so ₹2799 must be sent as 279900. Getting this wrong is the
     classic payments bug (100x overcharge), so the conversion lives in exactly
     one place and uses Decimal, not float.
  2. Never let a network/API failure crash the agent. Every SDK call is wrapped
     and returns {"success": False, "error": "..."} instead of raising, so the
     agent loop can turn a failed payment into a sentence the user understands.

Credentials come from the environment only — RAZORPAY_KEY_ID and
RAZORPAY_KEY_SECRET, same pattern as ANTHROPIC_API_KEY. Nothing is hardcoded and
no .env file is created by this module.
"""

import hashlib
import os
from decimal import Decimal, ROUND_HALF_UP

import razorpay
from razorpay.errors import BadRequestError, GatewayError, ServerError

KEY_ID_ENV = "RAZORPAY_KEY_ID"
KEY_SECRET_ENV = "RAZORPAY_KEY_SECRET"

# Opt-in offline mode. Lets the whole agent loop be demoed end-to-end before real
# test keys exist. It is OFF unless explicitly requested, and every order it
# returns is stamped with "mock": True so a fake order can never be mistaken for
# a real one in the audit log or on stage.
MOCK_ENV = "RAZORPAY_MOCK"

_client = None


def is_mock_mode():
    return os.environ.get(MOCK_ENV, "").lower() in ("1", "true", "yes")


def credentials_available():
    """True if both env vars are set. Used to give a clear, early diagnosis."""
    return bool(os.environ.get(KEY_ID_ENV)) and bool(os.environ.get(KEY_SECRET_ENV))


def get_client():
    """Build (and cache) the Razorpay client.

    Raises RuntimeError with an actionable message if credentials are missing —
    the caller catches this and turns it into a normal error result. That's much
    friendlier than the SDK's own failure mode, which is an opaque 401 at request
    time or a TypeError on a None key.
    """
    global _client
    if _client is not None:
        return _client

    key_id = os.environ.get(KEY_ID_ENV)
    key_secret = os.environ.get(KEY_SECRET_ENV)

    if not key_id or not key_secret:
        missing = [name for name in (KEY_ID_ENV, KEY_SECRET_ENV) if not os.environ.get(name)]
        raise RuntimeError(
            f"Razorpay credentials are not configured: {', '.join(missing)} not set. "
            f"Get test-mode keys from the Razorpay Dashboard (Settings → API Keys) and "
            f"export them, e.g.  export {KEY_ID_ENV}=rzp_test_xxx  "
            f"export {KEY_SECRET_ENV}=yyy   "
            f"(or set {MOCK_ENV}=1 to run the agent offline against a stub)."
        )

    _client = razorpay.Client(auth=(key_id, key_secret))
    return _client


def rupees_to_paise(amount_in_rupees):
    """₹ → paise, the only place this conversion happens.

    Decimal + ROUND_HALF_UP rather than float arithmetic: int(2799.35 * 100) is
    279934 on a float, which is a real (if small) money bug.
    """
    amount = Decimal(str(amount_in_rupees))
    if amount <= 0:
        raise ValueError(f"Amount must be positive, got {amount_in_rupees}")
    paise = (amount * 100).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
    return int(paise)


def _mock_order(amount_in_paise, currency, receipt, notes):
    """Stub order shaped like Razorpay's real response, clearly flagged as fake."""
    return {
        "id": f"order_MOCK{receipt or 'X'}",
        "entity": "order",
        "amount": amount_in_paise,
        "amount_paid": amount_in_paise,
        "amount_due": 0,
        "currency": currency,
        "receipt": receipt,
        "status": "paid",
        "notes": notes or {},
        # Mock orders come back already "paid" with a payment id attached, so the
        # refund flow is demoable offline. A REAL order has no payment_id at
        # creation time — the customer produces one by paying through Razorpay
        # Checkout, which this agent does not host. See create_refund().
        "payment_id": f"pay_MOCK{receipt or 'X'}",
        "mock": True,
        "warning": "MOCK ORDER — RAZORPAY_MOCK is set, no real API call was made.",
    }


def _api_failure(exc, action_phrase, reassurance):
    """Map any SDK/network exception to the uniform failure dict.

    Every failure carries two messages: `error` is the full technical detail and
    goes into the audit log; `user_message` is what's safe to say to a customer.
    Backend credential problems are our embarrassment, not the shopper's.

    Args:
        action_phrase: verb phrase for the customer message, e.g. "place the order"
        reassurance: the "nothing bad happened" clause, e.g. "Nothing has been charged."
    """
    if isinstance(exc, RuntimeError):
        # Missing credentials — our own clear message from get_client().
        return {"success": False, "error": str(exc), "error_type": "missing_credentials",
                "user_message": f"Our payment system is temporarily unavailable, so I couldn't "
                                f"{action_phrase}. {reassurance}"}
    if isinstance(exc, BadRequestError):
        # Razorpay rejected the request (bad amount, bad id, auth failure).
        return {"success": False, "error": f"Razorpay rejected the request: {exc}",
                "error_type": "bad_request",
                "user_message": f"The payment provider couldn't {action_phrase}. {reassurance}"}
    if isinstance(exc, (GatewayError, ServerError)):
        # Razorpay's side is unhealthy — worth telling the user to retry.
        return {"success": False, "error": f"Razorpay is temporarily unavailable: {exc}",
                "error_type": "gateway_error",
                "user_message": f"The payment provider is having trouble right now, so I couldn't "
                                f"{action_phrase}. {reassurance} It's worth trying again shortly."}
    # Network timeouts, DNS failures, anything unforeseen. Still no crash.
    return {"success": False, "error": f"Could not reach Razorpay: {exc}",
            "error_type": "connection_error",
            "user_message": f"I couldn't reach the payment provider, so I couldn't "
                            f"{action_phrase}. {reassurance} Please try again in a moment."}


def create_order(amount_in_rupees, currency="INR", receipt=None, notes=None, customer_id=None):
    """Create a Razorpay order.

    Args:
        amount_in_rupees: price in rupees (converted to paise internally)
        currency: ISO currency code, defaults to INR
        receipt: your own reference id for this order (Razorpay caps this at 40 chars)
        notes: dict of arbitrary key/value metadata stored on the order
        customer_id: Razorpay Customer id to associate the order with, if the
            session is linked to one. Recorded in notes, since the Orders API has
            no first-class customer field.

    Returns:
        On success: the order dict from Razorpay, with "success": True added.
        On failure: {"success": False, "error": "...", "error_type": "...",
                     "user_message": "..."}.

    This function never raises. A payment failure is an outcome the agent has to
    explain to the user, not an exception that kills the conversation.
    """
    try:
        amount_in_paise = rupees_to_paise(amount_in_rupees)
    except (ValueError, ArithmeticError) as exc:
        return {"success": False, "error": f"Invalid amount: {exc}", "error_type": "invalid_amount",
                "user_message": "Something was wrong with the price on that order, so I didn't "
                                "place it. Nothing has been charged."}

    # Razorpay rejects receipts longer than 40 characters.
    # Truncate with a hash suffix to preserve uniqueness.
    if receipt is not None:
        receipt = str(receipt)
        if len(receipt) > 40:
            digest = hashlib.sha256(receipt.encode()).hexdigest()[:8]
            receipt = receipt[:31] + "-" + digest

    notes = dict(notes or {})
    if customer_id:
        notes["customer_id"] = customer_id

    payload = {"amount": amount_in_paise, "currency": currency}
    if receipt:
        payload["receipt"] = receipt
    if notes:
        # Notes values must be strings on Razorpay's side.
        payload["notes"] = {str(k): str(v) for k, v in notes.items()}

    if is_mock_mode():
        order = _mock_order(amount_in_paise, currency, receipt, payload.get("notes"))
        order["success"] = True
        return order

    try:
        client = get_client()
        order = client.order.create(data=payload)
        order["success"] = True
        return order
    except Exception as exc:
        return _api_failure(exc, "place the order", "Nothing has been charged.")


# ---------------------------------------------------------------------------
# Refunds API
# ---------------------------------------------------------------------------

def _mock_refund(payment_id, amount_in_paise, notes, speed="normal"):
    """Stub refund shaped like Razorpay's real response, clearly flagged as fake."""
    return {
        "id": f"rfnd_MOCK{str(payment_id).replace('pay_MOCK', '')[:24]}",
        "entity": "refund",
        "amount": amount_in_paise,
        "currency": "INR",
        "payment_id": payment_id,
        "notes": notes or {},
        "receipt": None,
        "status": "processed",
        "speed_processed": speed,
        "mock": True,
        "warning": "MOCK REFUND — RAZORPAY_MOCK is set, no real API call was made.",
    }


def create_refund(payment_id, amount_in_rupees=None, notes=None):
    """Refund a payment, fully or partially.

    Args:
        payment_id: the Razorpay *payment* id (pay_...) to refund. Note this is a
            payment id, not an order id — Razorpay refunds money against the
            payment that captured it, not against the order that requested it.
        amount_in_rupees: None for a full refund; a rupee amount for a partial one
        notes: dict of metadata to attach to the refund

    Returns:
        On success: the refund dict, with "success": True added.
        On failure: the same uniform failure dict create_order returns.

    Never raises, for the same reason create_order doesn't.
    """
    if not payment_id:
        return {
            "success": False,
            "error": "create_refund called without a payment_id.",
            "error_type": "missing_payment_id",
            "user_message": "I couldn't find a completed payment for that order, so there's "
                            "nothing to refund yet.",
        }

    payload = {}
    amount_in_paise = None
    if amount_in_rupees is not None:
        try:
            amount_in_paise = rupees_to_paise(amount_in_rupees)
        except (ValueError, ArithmeticError) as exc:
            return {"success": False, "error": f"Invalid refund amount: {exc}",
                    "error_type": "invalid_amount",
                    "user_message": "That refund amount didn't look right, so I haven't "
                                    "processed it. No money has been moved."}
        payload["amount"] = amount_in_paise

    if notes:
        payload["notes"] = {str(k): str(v) for k, v in notes.items()}

    if is_mock_mode():
        refund = _mock_refund(payment_id, amount_in_paise, payload.get("notes"))
        refund["success"] = True
        refund["full_refund"] = amount_in_rupees is None
        return refund

    try:
        client = get_client()
        # POST /payments/:id/refund — the canonical refund endpoint. An empty
        # payload means "refund the whole captured amount".
        refund = client.payment.refund(payment_id, payload)
        refund["success"] = True
        refund["full_refund"] = amount_in_rupees is None
        return refund
    except Exception as exc:
        return _api_failure(exc, "process the refund", "No money has been moved.")


# ---------------------------------------------------------------------------
# Customers API
# ---------------------------------------------------------------------------

def _mock_customer(name, email, contact):
    """Stub customer, deterministic on contact so repeat lookups return the same id."""
    handle = "".join(ch for ch in str(contact) if ch.isalnum())[-10:] or "anon"
    return {
        "id": f"cust_MOCK{handle}",
        "entity": "customer",
        "name": name,
        "email": email,
        "contact": contact,
        "gstin": None,
        "notes": {},
        "mock": True,
        "warning": "MOCK CUSTOMER — RAZORPAY_MOCK is set, no real API call was made.",
    }


def get_or_create_customer(name, email, contact):
    """Fetch the customer if Razorpay already has them, otherwise create them.

    Razorpay gives us this for free: passing fail_existing=0 to the create call
    means "if a customer with this contact already exists, return them instead of
    erroring". That's a single round trip rather than a list-and-scan, and it has
    no race between the check and the create.

    Returns:
        On success: the customer dict with "success": True and "created" (bool).
        On failure: the uniform failure dict.
    """
    if not contact and not email:
        return {
            "success": False,
            "error": "get_or_create_customer needs at least a contact or an email.",
            "error_type": "missing_identity",
            "user_message": "I don't have enough details to look up your account.",
        }

    if is_mock_mode():
        customer = _mock_customer(name, email, contact)
        customer["success"] = True
        customer["created"] = False
        return customer

    payload = {
        "name": name,
        "email": email,
        "contact": contact,
        # 0 = return the existing customer rather than failing. Razorpay's SDK
        # sends this through as-is; it is the documented upsert switch.
        "fail_existing": "0",
    }

    try:
        client = get_client()
        customer = client.customer.create(data=payload)
        customer["success"] = True
        # Razorpay doesn't tell us which branch it took, and it doesn't matter to
        # the caller — either way we now hold a valid customer id.
        customer["created"] = None
        return customer
    except Exception as exc:
        return _api_failure(
            exc,
            "look up your customer record",
            "You can still shop as a guest.",
        )


if __name__ == "__main__":
    print("=== rupee → paise conversion ===")
    for rupees in (2799, 1099.50, 0.5):
        print(f"  ₹{rupees} -> {rupees_to_paise(rupees)} paise")

    print("\n=== credentials ===")
    print(f"  {KEY_ID_ENV} set: {bool(os.environ.get(KEY_ID_ENV))}")
    print(f"  {KEY_SECRET_ENV} set: {bool(os.environ.get(KEY_SECRET_ENV))}")
    print(f"  mock mode: {is_mock_mode()}")

    print("\n=== create_order (graceful either way) ===")
    order = create_order(2799, receipt="test_receipt_001", notes={"product_id": "P001"})
    print(f"  {order}")

    print("\n=== create_refund ===")
    print(f"  full   : {create_refund(order.get('payment_id'), notes={'reason': 'demo'})}")
    print(f"  partial: {create_refund(order.get('payment_id'), amount_in_rupees=500)}")
    print(f"  no id  : {create_refund(None)}")

    print("\n=== get_or_create_customer ===")
    print(f"  {get_or_create_customer('Demo User', 'demo@example.com', '+919999999999')}")
