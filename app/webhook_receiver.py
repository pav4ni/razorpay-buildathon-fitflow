"""
webhook_receiver.py — the inbound half of the agent. Razorpay tells *us* when
money actually moved, and we record it against the conversation that caused it.

Why this matters for this project: everything else in the codebase is the agent
acting outward (search, gate, create order). A webhook is the one place the
outside world talks back. Payment capture is asynchronous — the agent creates an
order and the conversation ends, and the customer pays seconds or minutes later
through Razorpay Checkout. Without this file, the audit trail stops at
"order_created" and never learns whether the money arrived.

Design, in the same spirit as the rest of the build:

  1. THE LOGIC IS NOT THE SERVER. Signature verification and event handling are
     plain functions over bytes and dicts. Flask is a thin transport on top. That
     means the whole thing is unit-testable locally with no server running, no
     port bound, and no ngrok tunnel — see test_stretch.py.
  2. SIGNATURE FIRST, ALWAYS. An unverified webhook is an anonymous stranger
     claiming a payment succeeded. Nothing is logged as a payment event until the
     HMAC matches, and comparison is constant-time.
  3. WEBHOOKS REJOIN THEIR CONVERSATION. create_order writes session_id into the
     order's `notes`, and Razorpay echoes notes back on the payment. So an
     asynchronous payment event can be filed against the exact session that
     produced it, and `/audit` on that session shows the full story end to end.
  4. A FAILED PAYMENT IS NOT A DEAD END. payment.failed is recorded with its
     reason and an explicit retry disposition, so the agent can explain it rather
     than silently dropping the customer.

Environment:
    RAZORPAY_WEBHOOK_SECRET — the secret you set when creating the webhook in the
    Razorpay Dashboard. This is NOT your API key secret; it is a separate value
    chosen per webhook endpoint.
"""

import hashlib
import hmac
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import audit

from idempotency import IdempotencyManager

_idempotency = IdempotencyManager()  # or pass your audit.db path

WEBHOOK_SECRET_ENV = "RAZORPAY_WEBHOOK_SECRET"
SIGNATURE_HEADER = "X-Razorpay-Signature"

# Events we understand. Anything else is still acknowledged (so Razorpay stops
# retrying it) but recorded as unhandled rather than silently dropped.
#
# subscription.charged is wired here even though the Subscriptions API itself is
# not enabled on this account yet — the webhook side needs no add-on, so when
# Subscriptions is switched on this endpoint already knows what to do with it.
HANDLED_EVENTS = ("payment.captured", "payment.failed", "subscription.charged")

# event name -> audit action_type
_ACTION_TYPE = {
    "payment.captured": "payment_captured",
    "payment.failed": "payment_failed",
    "subscription.charged": "subscription_charged",
}

# Webhook events that can't be tied back to a conversation get filed here, so
# they're still queryable instead of being attached to a random session.
UNATTRIBUTED_SESSION = "webhook-unattributed"


# ---------------------------------------------------------------------------
# Signature verification
# ---------------------------------------------------------------------------

def get_webhook_secret():
    """Read the webhook secret, or None if it isn't configured."""
    return os.environ.get(WEBHOOK_SECRET_ENV) or None


def compute_signature(raw_body, secret):
    """The signature Razorpay would send for this exact body.

    HMAC-SHA256 over the raw request bytes, hex-encoded. Exposed separately
    because the test suite and buyer_agent.py both need to *produce* a valid
    signature, not just check one.
    """
    if isinstance(raw_body, str):
        raw_body = raw_body.encode("utf-8")
    if isinstance(secret, str):
        secret = secret.encode("utf-8")
    return hmac.new(secret, raw_body, hashlib.sha256).hexdigest()


def verify_webhook_signature(raw_body, signature_header, secret=None):
    """True if `signature_header` is a valid signature for `raw_body`.

    Two details that are easy to get wrong and both matter:

    - It hashes the RAW BYTES as received. Parsing the JSON and re-serializing it
      would change whitespace and key order and break every signature, so the
      caller must hand us the untouched body.
    - It compares with hmac.compare_digest, not ==. String equality short-circuits
      on the first differing byte, which leaks how much of a guess was correct
      and makes forging a signature tractable given enough attempts.

    Returns False rather than raising on a missing secret or header — an
    unverifiable webhook is simply not verified.
    """
    secret = secret or get_webhook_secret()
    if not secret or not signature_header:
        return False
    expected = compute_signature(raw_body, secret)
    return hmac.compare_digest(expected, str(signature_header))


# ---------------------------------------------------------------------------
# Event handling
# ---------------------------------------------------------------------------

def _entity(payload, entity_name):
    """Dig an entity out of Razorpay's envelope.

    Razorpay nests entities as payload.payload.<name>.entity — e.g. a payment
    lives at payload["payload"]["payment"]["entity"]. Returns {} rather than
    raising, so a malformed webhook degrades to an empty summary.
    """
    try:
        return payload["payload"][entity_name]["entity"] or {}
    except (KeyError, TypeError):
        return {}


def _summarize(event, payload):
    """Pull the handful of fields worth logging and saying out loud.

    Deliberately small: the full payload goes into the audit row anyway, so this
    is just what a human needs to read in a console during a demo.
    """
    if event == "subscription.charged":
        subscription = _entity(payload, "subscription")
        payment = _entity(payload, "payment")
        return {
            "subscription_id": subscription.get("id"),
            "payment_id": payment.get("id"),
            "amount_rupees": (payment.get("amount") or 0) / 100,
            "currency": payment.get("currency"),
            "paid_count": subscription.get("paid_count"),
            "total_count": subscription.get("total_count"),
            "notes": subscription.get("notes") or payment.get("notes") or {},
        }

    payment = _entity(payload, "payment")
    summary = {
        "payment_id": payment.get("id"),
        "order_id": payment.get("order_id"),
        "amount_rupees": (payment.get("amount") or 0) / 100,
        "currency": payment.get("currency"),
        "method": payment.get("method"),
        "notes": payment.get("notes") or {},
    }
    if event == "payment.failed":
        # Razorpay puts the human-readable cause in error_description; the
        # machine-readable one in error_code / error_reason.
        summary.update(
            {
                "error_code": payment.get("error_code"),
                "error_reason": payment.get("error_reason"),
                "error_description": payment.get("error_description"),
            }
        )
    return summary


def _retry_disposition(summary):
    """What should happen after a failed payment.

    Razorpay does not retry a one-off payment for us — the customer simply
    didn't pay. So "retry" here means the agent should re-offer, not that any
    money will move on its own. Saying that explicitly in the log is the point:
    a failed payment that nobody follows up on is a lost customer, and this is
    the record that it wasn't silently dropped.
    """
    reason = summary.get("error_reason") or summary.get("error_code") or "unknown"
    # These are the ones where asking the customer to try again is reasonable.
    retryable = reason in (
        "payment_failed", "gateway_error", "server_error", "network_error", "unknown",
    )
    return {
        "retryable": retryable,
        "next_step": (
            "Re-offer the same order to the customer; the order is still open and unpaid."
            if retryable
            else "Do not auto-retry; the customer needs to use a different payment method."
        ),
    }


def _session_id_from(summary):
    """Recover the conversation this event belongs to.

    create_order stamps session_id into the order's notes, and Razorpay echoes
    notes onto the payment — so an asynchronous event can be filed against the
    conversation that caused it.
    """
    notes = summary.get("notes") or {}
    return notes.get("session_id") or UNATTRIBUTED_SESSION


def handle_webhook_event(payload, echo=True):
    """Process one *already-verified* webhook payload.

    Separated from signature checking so the dispatch logic can be tested on its
    own, and so process_webhook below reads as the two steps it actually is.

    Returns a dict describing what was done — the same thing the HTTP layer
    returns as its JSON body.
    """
    event = (payload or {}).get("event", "")
        # --- IDEMPOTENCY CHECK ---
    event_id = (payload or {}).get("id", "")
    if event_id and not _idempotency.check_and_claim(event_id, event):
        return {"status": "skipped", "reason": "duplicate_event"}
    # --- END IDEMPOTENCY ---

    summary = _summarize(event, payload or {})
    session_id = _session_id_from(summary)
    handled = event in HANDLED_EVENTS

    action_type = _ACTION_TYPE.get(event, "webhook_unhandled")
    reasoning = (
        f"Razorpay reported {event}; recording it against the session that created the order."
        if handled
        else f"Received webhook event {event!r}, which this endpoint does not handle. "
             f"Acknowledged so Razorpay stops retrying, and logged for visibility."
    )

    result = {"event": event, "handled": handled, **summary}
    if event == "payment.failed":
        result["retry"] = _retry_disposition(summary)

    audit.log_event(
        session_id=session_id,
        user_query=None,
        agent_reasoning=reasoning,
        action_type=action_type,
        action_params={"event": event, "source": "razorpay_webhook", "payload": payload},
        result=result,
        customer_id=(summary.get("notes") or {}).get("customer_id"),
    )

    if echo:
        _print_event(event, handled, summary, result, session_id)

    return {
        "status": "ok",
        "event": event,
        "handled": handled,
        "session_id": session_id,
        "audit_action": action_type,
        "summary": result,
    }


def _print_event(event, handled, summary, result, session_id):
    """Loud console output — this is what a judge watches during the demo."""
    marker = {
        "payment.captured": "[PAID]     ",
        "payment.failed": "[FAILED]   ",
        "subscription.charged": "[RENEWED]  ",
    }.get(event, "[UNHANDLED]")

    print(f"\n  {marker} webhook: {event}")
    print(f"      session      : {session_id}")
    if summary.get("payment_id"):
        print(f"      payment      : {summary['payment_id']}")
    if summary.get("order_id"):
        print(f"      order        : {summary['order_id']}")
    if summary.get("subscription_id"):
        print(f"      subscription : {summary['subscription_id']} "
              f"(cycle {summary.get('paid_count')}/{summary.get('total_count')})")
    if summary.get("amount_rupees"):
        print(f"      amount       : ₹{summary['amount_rupees']}")
    if event == "payment.failed":
        print(f"      reason       : {summary.get('error_description') or summary.get('error_code')}")
        print(f"      retry        : {result['retry']['next_step']}")
    if not handled:
        print("      note         : event acknowledged but not handled by this endpoint")
    # flush explicitly: when the server's output is piped to a file or a log
    # (which it is, whenever it's run in the background for a demo), Python
    # block-buffers stdout and these lines would otherwise not appear until the
    # process exits — exactly when they're least useful.
    print(flush=True)


def process_webhook(raw_body, signature_header, secret=None, echo=True):
    """Verify, then handle. The whole endpoint in one testable function.

    Returns (http_status, response_dict). Status codes are chosen for how
    Razorpay reacts to them:
        200 — accepted (Razorpay stops retrying)
        400 — malformed JSON (retrying won't help)
        401 — bad or missing signature (we refuse to trust it)
        503 — we have no secret configured, so we *cannot* verify; Razorpay
              should retry once the endpoint is configured properly.
    """
    secret = secret or get_webhook_secret()
    if not secret:
        # Fail closed. An endpoint that processes unverified webhooks because it
        # was misconfigured is worse than one that's temporarily unavailable.
        audit.log_event(
            session_id=UNATTRIBUTED_SESSION,
            user_query=None,
            agent_reasoning="Webhook arrived but no signing secret is configured, so it cannot be verified. Refusing it.",
            action_type="webhook_rejected",
            action_params={"reason": "no_secret_configured"},
            result={"accepted": False, "env_var": WEBHOOK_SECRET_ENV},
        )
        if echo:
            print(f"\n  [REJECTED] webhook refused — {WEBHOOK_SECRET_ENV} is not set.\n", flush=True)
        return 503, {
            "status": "error",
            "error": f"{WEBHOOK_SECRET_ENV} is not configured; cannot verify webhook signatures.",
        }

    if not verify_webhook_signature(raw_body, signature_header, secret):
        # Log the rejection but NOT the body — an unverified payload is
        # attacker-controlled and shouldn't be persisted as if it were fact.
        audit.log_event(
            session_id=UNATTRIBUTED_SESSION,
            user_query=None,
            agent_reasoning="Webhook signature did not match the expected HMAC. Rejected without processing.",
            action_type="webhook_rejected",
            action_params={
                "reason": "invalid_signature",
                "signature_present": bool(signature_header),
                "body_bytes": len(raw_body or b""),
            },
            result={"accepted": False},
        )
        if echo:
            print("\n  [REJECTED] webhook signature invalid — payload discarded, "
                  "nothing logged as a payment.\n", flush=True)
        return 401, {"status": "error", "error": "Invalid webhook signature."}

    try:
        payload = json.loads(raw_body)
    except (ValueError, TypeError) as exc:
        return 400, {"status": "error", "error": f"Body was not valid JSON: {exc}"}

    return 200, handle_webhook_event(payload, echo=echo)


# ---------------------------------------------------------------------------
# HTTP transport (Flask) — deliberately thin
# ---------------------------------------------------------------------------

def register_routes(app):
    """Attach the webhook route to an existing Flask app.

    Kept separate from create_app so server.py can serve the webhook, the
    catalog manifest and the agent chat endpoint from one process on one port —
    which is one ngrok tunnel instead of three during a demo.
    """
    from flask import request, jsonify

    @app.route("/webhook", methods=["POST"])
    def razorpay_webhook():
        # request.get_data() gives the raw bytes. Using request.json here instead
        # would silently break every signature check.
        status, body = process_webhook(
            raw_body=request.get_data(),
            signature_header=request.headers.get(SIGNATURE_HEADER),
        )
        return jsonify(body), status

    return app


def create_app():
    """Standalone Flask app serving only the webhook endpoint."""
    from flask import Flask

    app = Flask(__name__)
    return register_routes(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    print("=" * 70)
    print("  Razorpay webhook receiver")
    print("=" * 70)
    print(f"  POST http://127.0.0.1:{port}/webhook")
    print(f"  secret configured : {bool(get_webhook_secret())}  (env {WEBHOOK_SECRET_ENV})")
    print(f"  handled events    : {', '.join(HANDLED_EVENTS)}")
    if not get_webhook_secret():
        print(f"\n  WARNING: {WEBHOOK_SECRET_ENV} is not set — every webhook will be refused with 503.")
    print("=" * 70)
    create_app().run(port=port, debug=False)
