"""
buyer_agent.py — the other side of the conversation. No human in the loop.

This is a standalone shopper: it has a goal in natural language, it talks to the
merchant agent over HTTP, it reads the replies, and when an order is created it
completes payment. Nobody types anything. That's the whole point of the
demo — two agents transacting, with the merchant's safety gate and audit trail
doing their job regardless of whether the buyer is a person or a program.

It is deliberately NOT an LLM. A model on this side would make the demo
non-deterministic and would prove nothing extra: what's being demonstrated is
that the merchant is *transactable by software*. So the buyer is a scripted
shopper with hardcoded intents, which makes the demo repeatable.

Flow per scenario:

    1. POST /agent/chat with a natural-language want
    2. read the reply and the machine-readable session state
    3. keep talking until an order exists or the script runs out
    4. simulate the buyer paying — sign a payment.captured (or payment.failed)
       webhook exactly the way Razorpay would, and POST it to /webhook
    5. the merchant's webhook receiver verifies the signature and files the
       payment against the same session that created the order

Step 4 requires mock mode. Real payment capture happens inside Razorpay's hosted
Checkout, which this project intentionally doesn't integrate — so a real order
stays `created` with no payment_id and there is nothing to capture. With
RAZORPAY_MOCK=1 the order comes back already carrying a pay_MOCK... id, which is
what makes an end-to-end payment event demonstrable offline.

Run it:
    # terminal 1
    RAZORPAY_MOCK=1 RAZORPAY_WEBHOOK_SECRET=whsec_demo python3 app/server.py
    # terminal 2
    RAZORPAY_MOCK=1 RAZORPAY_WEBHOOK_SECRET=whsec_demo python3 app/buyer_agent.py
"""

import json
import os
import sys
import time
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import negotiation
import webhook_receiver

# Port 5050, not 5000 — macOS Control Centre's AirPlay Receiver holds 5000 and
# replies 403, which looks exactly like a merchant bug. Keep in sync with
# server.py's default.
DEFAULT_BASE_URL = os.environ.get("MERCHANT_URL", "http://127.0.0.1:5050")


# ---------------------------------------------------------------------------
# Scenarios — what this buyer wants, in its own words
# ---------------------------------------------------------------------------
#
# `utterances` are sent in order. The buyer stops early once an order exists, so
# a scenario that closes in two turns doesn't keep talking.
# `payment_outcome` decides which webhook gets fired afterwards.

SCENARIOS = [
    {
        "name": "Restock protein",
        "goal": "Buy a tub of whey protein without being fussy about flavour.",
        "utterances": [
            "I need to restock my whey protein, chocolate if you have it. Budget is about 2500.",
            "Yes, order one of those please.",
        ],
        "payment_outcome": "captured",
    },
    {
        "name": "Kit out a home yoga setup",
        "goal": "Find a yoga mat and accept a sensible cross-sell.",
        "utterances": [
            "I'm setting up a home yoga corner. I want a good non-slip mat, nothing over 1500.",
            "That sounds right, go ahead and buy it.",
        ],
        "payment_outcome": "captured",
    },
    {
        "name": "Card declines on a resistance band set",
        "goal": "Buy resistance bands, but the buyer's card fails at capture.",
        "utterances": [
            "Looking for a set of resistance bands for home workouts, under 1200.",
            "The PowerBand Resistance Band Set is the one I want — please order it.",
        ],
        "payment_outcome": "failed",
    },
]

# ---------------------------------------------------------------------------
# Negotiation scenarios — what this buyer wants, and what it will really pay
# ---------------------------------------------------------------------------
#
# `true_max` is the buyer's private ceiling and is never sent to the merchant.
# `product_id` pins the item the buyer is tracking so the merchant's quotes can
# be read out of the structured response rather than parsed out of prose.
#
# The two scenarios are chosen to end differently on purpose. The first closes,
# because the item carries enough margin for the policy cap to be the only thing
# in the way. The second cannot close at any permitted price, because the buyer's
# real ceiling sits below the item's margin floor — and the interesting assertion
# is that the merchant does NOT chase the sale.

NEGOTIATION_SCENARIOS = [
    {
        "name": "Haggle over a compression base layer",
        # Named exactly, and this matters. An earlier version asked for "a
        # lightweight training t-shirt", and the merchant's semantic search
        # quite correctly returned a Rs.799 tee that was already inside the
        # buyer's opening budget — so there was nothing to negotiate and the
        # buyer sat there restating itself. A negotiation scenario has to steer
        # the merchant to the item it is actually about.
        "product_hint": "the CoolCore Compression Base Layer",
        "product_id": "P008",       # Rs.1199 retail, Rs.525 cost -> 20% cap binds
        "true_max": 1000,           # merchant's best (Rs.959.20 at the cap) clears this
        "max_rounds": 3,
        "payment_outcome": "captured",
        "expect": "deal",
    },
    {
        "name": "Lowball a thin-margin supplement",
        "product_hint": "the CreaPure Creatine Monohydrate tub",
        "product_id": "P032",       # Rs.899 retail, Rs.700 cost -> Rs.770 margin floor
        "true_max": 700,            # BELOW the margin floor: no legal price can reach it
        "max_rounds": 3,
        "expect": "walk_away",
    },
]


# Sent once if the scripted utterances run out without an order existing.
#
# This is not a workaround — it's the buyer handling the merchant doing its job.
# The merchant is built to ask ONE clarifying question when a request is
# ambiguous (two comparable band sets, say), and a "place the order" that
# doesn't name a product will correctly get a "which one?" back. A human would
# answer that; an autonomous buyer has to be able to as well.
DISAMBIGUATION_FALLBACK = (
    "Pick whichever one you'd recommend and place the order for a single unit."
)


# ---------------------------------------------------------------------------
# Transports — how this buyer reaches the merchant
# ---------------------------------------------------------------------------

class HttpTransport:
    """Talk to a merchant server over the network."""

    def __init__(self, base_url=DEFAULT_BASE_URL):
        self.base_url = base_url.rstrip("/")

    def post(self, path, body, headers=None):
        import requests

        response = requests.post(
            f"{self.base_url}{path}", json=body, headers=headers or {}, timeout=120
        )
        return response.status_code, _safe_json(response.text)

    def post_raw(self, path, raw_body, headers=None):
        import requests

        response = requests.post(
            f"{self.base_url}{path}", data=raw_body, headers=headers or {}, timeout=60
        )
        return response.status_code, _safe_json(response.text)

    def get(self, path):
        import requests

        response = requests.get(f"{self.base_url}{path}", timeout=60)
        return response.status_code, _safe_json(response.text)


class InProcessTransport:
    """Talk to a merchant Flask app directly, with no port bound.

    This is what lets test_stretch.py run the buyer agent end to end without
    starting a server — same code path, no networking.
    """

    def __init__(self, app=None):
        if app is None:
            import server

            app = server.create_app()
        self.client = app.test_client()

    def post(self, path, body, headers=None):
        response = self.client.post(path, json=body, headers=headers or {})
        return response.status_code, _safe_json(response.get_data(as_text=True))

    def post_raw(self, path, raw_body, headers=None):
        response = self.client.post(path, data=raw_body, headers=headers or {})
        return response.status_code, _safe_json(response.get_data(as_text=True))

    def get(self, path):
        response = self.client.get(path)
        return response.status_code, _safe_json(response.get_data(as_text=True))


def _safe_json(text):
    try:
        return json.loads(text)
    except (ValueError, TypeError):
        return {"raw": text}


# ---------------------------------------------------------------------------
# Webhook payload construction — impersonating Razorpay, correctly
# ---------------------------------------------------------------------------

def build_payment_payload(event, order, session_id, customer_id=None):
    """A Razorpay webhook envelope for a payment on `order`.

    The nesting (payload -> payment -> entity) mirrors Razorpay's real shape,
    because webhook_receiver parses that shape. `notes` carries session_id — the
    same notes create_order wrote onto the order — which is how the merchant
    files this event against the right conversation.
    """
    amount_paise = int(round(float(order.get("amount_rupees", 0)) * 100))
    notes = {
        "session_id": session_id,
        "product_id": order.get("product_id"),
        "product_name": order.get("product_name"),
    }
    if customer_id:
        notes["customer_id"] = customer_id

    payment = {
        "id": order.get("payment_id") or f"pay_SIM{order.get('order_id', 'X')[-10:]}",
        "entity": "payment",
        "amount": amount_paise,
        "currency": "INR",
        "order_id": order.get("order_id"),
        "method": "card",
        "notes": notes,
    }

    if event == "payment.captured":
        payment["status"] = "captured"
        payment["captured"] = True
    else:
        payment.update(
            {
                "status": "failed",
                "captured": False,
                "error_code": "BAD_REQUEST_ERROR",
                "error_reason": "payment_failed",
                "error_description": "The card issuer declined the transaction.",
            }
        )

    return {
        "entity": "event",
        "account_id": "acc_SIMULATED",
        "event": event,
        "contains": ["payment"],
        "payload": {"payment": {"entity": payment}},
        "created_at": int(time.time()),
    }


def send_signed_webhook(transport, payload, secret, path="/webhook"):
    """Sign a payload the way Razorpay does and POST it.

    Signs the exact bytes that get sent. Serializing once into `raw` and reusing
    it for both the HMAC and the request body is the whole trick — re-encoding
    between signing and sending is the classic way to break signature checks.
    """
    raw = json.dumps(payload).encode("utf-8")
    signature = webhook_receiver.compute_signature(raw, secret)
    return transport.post_raw(
        path,
        raw,
        headers={
            "Content-Type": "application/json",
            webhook_receiver.SIGNATURE_HEADER: signature,
        },
    )


# ---------------------------------------------------------------------------
# The buyer
# ---------------------------------------------------------------------------

class BuyerAgent:
    """A shopper made of code."""

    def __init__(self, transport=None, webhook_secret=None, verbose=True):
        self.transport = transport or HttpTransport()
        self.webhook_secret = webhook_secret or webhook_receiver.get_webhook_secret()
        self.verbose = verbose

    def _say(self, text):
        if self.verbose:
            print(text)

    def run_scenario(self, scenario):
        """Play out one scenario end to end, returning a result summary."""
        self._say("\n" + "=" * 70)
        self._say(f"  BUYER SCENARIO: {scenario['name']}")
        self._say(f"  goal: {scenario['goal']}")
        self._say("=" * 70)

        session_id = None
        state = {}
        orders_before = 0

        # The scripted lines, then at most one fallback if the merchant asked a
        # clarifying question and the script had nothing left to say.
        script = list(scenario["utterances"])
        script.append(scenario.get("fallback", DISAMBIGUATION_FALLBACK))

        for utterance in script:
            self._say(f"\n  buyer    > {utterance}")
            status, state = self.transport.post(
                "/agent/chat", {"message": utterance, "session_id": session_id}
            )
            if status != 200:
                self._say(f"  !! merchant returned {status}: {state}")
                return {"scenario": scenario["name"], "ok": False, "error": state}

            session_id = state.get("session_id")
            self._say(f"  merchant > {state.get('reply', '')}")

            # Stop as soon as the merchant has actually created an order — no
            # point reading from the script once the goal is met.
            if len(state.get("orders", [])) > orders_before:
                break

        orders = state.get("orders", [])
        if not orders:
            self._say("\n  (no order was created in this scenario)")
            return {
                "scenario": scenario["name"],
                "ok": True,
                "session_id": session_id,
                "orders": [],
                "webhook": None,
            }

        order = orders[-1]
        self._say(
            f"\n  [order]  {order.get('order_id')}  {order.get('product_name')} "
            f"x{order.get('quantity')}  ₹{order.get('amount_rupees')}"
        )

        webhook_result = self._complete_payment(scenario, order, session_id, state)
        return {
            "scenario": scenario["name"],
            "ok": True,
            "session_id": session_id,
            "orders": orders,
            "webhook": webhook_result,
        }

    def _complete_payment(self, scenario, order, session_id, state):
        """Simulate the buyer paying (or failing to), and fire the webhook."""
        if not self.webhook_secret:
            self._say(
                f"\n  (skipping payment simulation — {webhook_receiver.WEBHOOK_SECRET_ENV} is not set)"
            )
            return None

        if not order.get("payment_id"):
            # Real (non-mock) mode: the order is genuine but unpaid, because
            # capture only happens inside Razorpay Checkout.
            self._say(
                "\n  (order has no payment_id — this is real-mode behaviour; "
                "run with RAZORPAY_MOCK=1 to simulate capture)"
            )
            return None

        event = (
            "payment.captured"
            if scenario.get("payment_outcome", "captured") == "captured"
            else "payment.failed"
        )
        self._say(f"\n  [buyer pays] simulating {event} for {order['payment_id']}")

        payload = build_payment_payload(event, order, session_id, state.get("customer_id"))
        status, body = send_signed_webhook(self.transport, payload, self.webhook_secret)
        self._say(f"  [webhook] merchant responded {status}: handled={body.get('handled')}")
        return {"event": event, "status": status, "body": body}

    # -- negotiation ------------------------------------------------------

    @staticmethod
    def _merchant_quote(state, product_id):
        """The merchant's best price for the tracked item, from THIS turn.

        Read from the structured response rather than from the reply text. Two
        sources, in order of authority:

          1. `discount_offers` — the merchant actually costed a discount. This is
             a real negotiating position and carries `capped_by`, which says
             whether it can go lower.
          2. `products` — the merchant only showed the item. Sticker price, no
             offer made yet.

        Returns None when the merchant named no price at all this turn, which the
        buyer treats as "ask again" rather than as a refusal.
        """
        for offer in state.get("discount_offers") or []:
            if offer.get("product_id") == product_id and offer.get("price") is not None:
                return {
                    "price": float(offer["price"]),
                    "discount_percent": offer.get("discount_percent"),
                    "sticker_price": offer.get("sticker_price"),
                    "capped_by": offer.get("capped_by"),
                    "source": "discount_offer",
                }

        for product in state.get("products") or []:
            if product.get("id") == product_id and product.get("price") is not None:
                return {
                    "price": float(product["price"]),
                    "discount_percent": 0,
                    "sticker_price": float(product["price"]),
                    "capped_by": None,
                    "source": "list_price",
                }
        return None

    def run_negotiation(self, scenario):
        """Play out one negotiation, logging both sides against one session.

        The buyer mints the session id itself rather than adopting whatever the
        merchant returns. That is not cosmetic: it lets the opening move be
        logged BEFORE the merchant's first reply, so the audit trail reads in the
        order the exchange actually happened instead of opening with the
        merchant's answer to a question that appears later.
        """
        policy = negotiation.BuyerPolicy(
            product_hint=scenario["product_hint"],
            true_max=scenario["true_max"],
            max_rounds=scenario.get("max_rounds", negotiation.DEFAULT_MAX_ROUNDS),
        )
        session_id = scenario.get("session_id") or f"negotiate-{uuid.uuid4().hex[:10]}"
        product_id = scenario["product_id"]

        self._say("\n" + "=" * 70)
        self._say(f"  NEGOTIATION: {scenario['name']}")
        self._say(f"  session      : {session_id}")
        self._say(f"  buyer wants  : {policy.product_hint}")
        self._say(f"  hidden ceiling: Rs.{policy.true_max:g}   "
                  f"(opens claiming Rs.{policy.opening_budget:g} — never states the real one)")
        self._say("=" * 70)

        negotiation.log_opened(session_id, policy)
        utterance = policy.opening_utterance()
        state, customer_id = {}, None

        # Defence in depth. The policy is responsible for terminating, and it is
        # tested to do so — but this loop spends real money on every lap, so it
        # does not take that on trust. The cap is deliberately just above
        # max_rounds: if it ever trips, the policy has a bug.
        hard_cap = policy.max_rounds + 2
        laps = 0

        while policy.outcome is None:
            laps += 1
            if laps > hard_cap:
                self._say(f"\n  !! ABORT: {laps} laps exceeds the {hard_cap}-lap cap — "
                          f"the buyer policy failed to terminate. This is a bug.")
                return {"scenario": scenario["name"], "ok": False,
                        "error": f"policy did not terminate within {hard_cap} laps",
                        "session_id": session_id, "history": policy.history}
            self._say(f"\n  buyer    > {utterance}")
            status, state = self.transport.post(
                "/agent/chat", {"message": utterance, "session_id": session_id}
            )
            if status != 200:
                self._say(f"  !! merchant returned {status}: {state}")
                return {"scenario": scenario["name"], "ok": False, "error": state,
                        "session_id": session_id}

            customer_id = state.get("customer_id")
            self._say(f"  merchant > {state.get('reply', '')}")

            quote = self._merchant_quote(state, product_id)
            negotiation.log_merchant_position(session_id, policy.round + 1, quote, customer_id)
            if quote:
                capped = f"  [cannot go lower: {quote['capped_by']}]" if quote.get("capped_by") else ""
                self._say(f"  [quote]  Rs.{quote['price']:g}"
                          f" ({quote.get('discount_percent') or 0:g}% off){capped}")

            record = policy.decide(quote["price"] if quote else None)

            if record["decision"] == negotiation.ACCEPT:
                utterance = policy.accept_utterance()
            elif record["decision"] == negotiation.WALK_AWAY:
                utterance = policy.walk_away_utterance()
            elif quote is None:
                # The merchant asked something rather than quoting. Answer it,
                # otherwise it will simply ask again.
                utterance = policy.clarify_utterance()
            else:
                utterance = policy.counter_utterance()

            negotiation.log_round(session_id, record, utterance, customer_id)
            self._say(f"  [buyer]  round {record['round']}: {record['decision'].upper()}"
                      f" — {record['reasoning']}")

        # Deliver the final decision so the merchant can act on it: place the
        # order on an accept, or close the conversation gracefully on a walk-away.
        self._say(f"\n  buyer    > {utterance}")
        status, state = self.transport.post(
            "/agent/chat", {"message": utterance, "session_id": session_id}
        )
        if status == 200:
            self._say(f"  merchant > {state.get('reply', '')}")

        orders = state.get("orders", []) if status == 200 else []
        order = orders[-1] if orders else None
        final_price = order.get("amount_rupees") if order else None
        summary = negotiation.summarize(
            policy, final_price=final_price,
            order_id=order.get("order_id") if order else None,
        )

        webhook_result = None
        if order and policy.outcome == negotiation.ACCEPT:
            webhook_result = self._complete_payment(scenario, order, session_id, state)

        self._say("\n  " + "-" * 66)
        self._say(f"  OUTCOME  : {policy.outcome.upper()} after {policy.round} round(s)")
        if policy.outcome == negotiation.ACCEPT:
            self._say(f"  best quote: Rs.{policy.best_price:g}   "
                      f"(hidden ceiling was Rs.{policy.true_max:g})")
            self._say(f"  order     : {summary['order_id'] or 'NONE — merchant did not place it'}")
            if final_price is not None:
                self._say(f"  paid      : Rs.{final_price:g}")
        elif policy.best_price is not None:
            self._say(f"  merchant held at Rs.{policy.best_price:g}; buyer would pay at most "
                      f"Rs.{policy.true_max:g}. No order placed.")
        else:
            # best_price stays None when the merchant never quoted anything —
            # formatting it with :g raised a TypeError and took the whole run
            # down after the negotiation had already finished cleanly.
            self._say("  the merchant never quoted a price for this item. No order placed.")
        self._say("  " + "-" * 66)

        return {
            "scenario": scenario["name"],
            "ok": True,
            "session_id": session_id,
            "summary": summary,
            "history": policy.history,
            "orders": orders,
            "webhook": webhook_result,
        }

    def negotiate_all(self, scenarios=None):
        return [self.run_negotiation(s) for s in (scenarios or NEGOTIATION_SCENARIOS)]

    def run_all(self, scenarios=None):
        return [self.run_scenario(s) for s in (scenarios or SCENARIOS)]


def main():
    secret = webhook_receiver.get_webhook_secret()
    base = DEFAULT_BASE_URL

    print("=" * 70)
    print("  Buyer agent — autonomous shopper, no human in the loop")
    print("=" * 70)
    print(f"  merchant       : {base}")
    print(f"  webhook secret : {'set' if secret else 'NOT SET — payment simulation will be skipped'}")
    print(f"  mock mode      : {os.environ.get('RAZORPAY_MOCK', '(unset)')}")
    if not os.environ.get("RAZORPAY_MOCK"):
        print("\n  NOTE: without RAZORPAY_MOCK=1 the merchant creates real orders that have")
        print("        no payment_id, so there is nothing to 'capture' and no webhook fires.")
    print("=" * 70)

    transport = HttpTransport(base)
    status, health = transport.get("/health")
    if status != 200:
        print(f"\n  Could not reach the merchant at {base} (status {status}).")
        print("  Start it first:  python3 app/server.py")
        return 1
    print(f"\n  merchant health: {health}")

    buyer = BuyerAgent(transport=transport, webhook_secret=secret)

    # --negotiate runs the two-sided negotiation scenarios instead of the
    # scripted purchases. Kept as a flag rather than replacing the default so the
    # original agent-to-agent demo still runs unchanged.
    if "--negotiate" in sys.argv:
        results = buyer.negotiate_all()
        print("\n" + "=" * 70)
        print("  NEGOTIATION SUMMARY")
        print("=" * 70)
        for result in results:
            summary = result.get("summary") or {}
            outcome = (summary.get("outcome") or "error").upper()
            best = summary.get("best_price_offered")
            best_text = f"best=Rs.{best:g}" if best is not None else "best=none quoted"
            print(f"  {result['scenario']:36s} {outcome:11s} "
                  f"rounds={summary.get('rounds_used', '-')}  {best_text}")
            print(f"      hidden ceiling Rs.{summary.get('hidden_true_max', '-')}, "
                  f"opened at Rs.{summary.get('opening_budget', '-')}, "
                  f"order={summary.get('order_id') or 'none'}")
            print(f"      replay: /api/audit/{result.get('session_id')}")
        print()
        return 0

    results = buyer.run_all()

    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    for result in results:
        orders = result.get("orders") or []
        webhook = result.get("webhook") or {}
        print(
            f"  {result['scenario']:38s} orders={len(orders)} "
            f"webhook={webhook.get('event', '-')} ({webhook.get('status', '-')})"
        )
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
