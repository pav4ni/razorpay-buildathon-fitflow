"""
test_tier2.py — exercise of the Tier 2 features: discounts, budget negotiation,
upsell/attach rate, refunds and customers.

Run it:

    export ANTHROPIC_API_KEY=...
    RAZORPAY_MOCK=1 python3 test_tier2.py              # everything
    RAZORPAY_MOCK=1 python3 test_tier2.py --no-llm     # deterministic part only, no API key needed
    RAZORPAY_MOCK=1 python3 test_tier2.py --fresh-db   # against a throwaway audit db

The suite is in two halves, and the split is deliberate.

PART A drives agent.execute_tool directly, with no model in the loop. That is the
same code path a real tool call takes — same handlers, same gate, same audit
writes — minus the one component whose output is not reproducible. It means the
money assertions ("the charge is ₹2491.11, not ₹2799") are exact rather than
approximate, and a red test always means the Python is wrong rather than that the
model phrased something differently today.

PART B runs real conversations through the model and asserts on outcomes recorded in
the audit log, not on wording. It's what proves the model actually reaches for
these tools when a customer talks to it; Part A is what proves the tools are
correct when it does.

Nothing here relaxes gate.py. Where a scenario is supposed to be refused, the
test asserts that it WAS refused and that the refusal was explained.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

import audit

# --fresh-db must be applied before anything reads the database.
if "--fresh-db" in sys.argv:
    import tempfile
    audit.DB_PATH = os.path.join(tempfile.mkdtemp(prefix="tier2-audit-"), "audit.db")

import agent
import catalog
import discount
import gate
import metrics
import razorpay_client

failures = []

# Catalog constants the scenarios are built on. Pinned here so a catalog edit
# fails loudly in one place instead of quietly changing what the tests mean.
SHOES = "P001"          # CloudRunner Mesh Running Shoes, ₹2799, stock 14
SOCKS = "P002"          # AirFlex socks, ₹349 — a complementary product of P001
YOGA_MAT = "P011"       # ZenFlow Yoga Mat, ₹999, stock 33
NO_SUCH_PRODUCT = "P999"


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def approx(a, b, tol=0.005):
    return a is not None and b is not None and abs(a - b) <= tol


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


def new_session():
    """A session with a linked customer, exactly as the CLI creates one."""
    session = agent.ShoppingSession()
    agent.link_customer(session)
    return session


def tool(session, name, **params):
    """Invoke one tool through the real execution layer, so it is audited."""
    params.setdefault("reasoning", f"test harness exercising {name}")
    return agent.execute_tool(session, name, params)


def action_types(session):
    return [e["action_type"] for e in audit.get_session_log(session.session_id)]


def rows_of_type(session, action_type):
    return [e for e in audit.get_session_log(session.session_id)
            if e["action_type"] == action_type]


def say(session, message):
    print(f"\n  you   > {message}")
    reply = agent.run_turn(session, message)
    print(f"  agent > {reply}")
    return reply


# ===========================================================================
# PART A — deterministic, no model in the loop
# ===========================================================================

def a1_discount_bridges_the_gap():
    """The headline Tier 2 path: a budget gap closed by the smallest legal
    discount, and — the part that was missing before — a charge that actually
    reflects it."""
    header("A1 — discount bridges a budget gap, and the CHARGE reflects it")
    session = new_session()

    search = tool(session, "search_catalog",
                  query="lightweight running shoes for daily jogging", max_price=2500)
    negotiation = search.get("budget_negotiation")

    print("\n  budget-aware negotiation:")
    check("price ceiling triggered the negotiation block", negotiation is not None,
          f"reason={negotiation.get('reason') if negotiation else None}")
    if negotiation:
        closest = negotiation.get("closest_option", {})
        check("it surfaced the shoes the budget excluded", closest.get("id") == SHOES,
              f"closest={closest.get('name')} ₹{closest.get('price')}")
        check("it reported the gap as negotiable", closest.get("gap_is_negotiable") is True)
        check("it pre-computed the discount needed",
              closest.get("minimum_discount_needed_percent") == 11,
              f"{closest.get('minimum_discount_needed_percent')}%")
        check("guidance tells the agent not to dead-end",
              bool(negotiation.get("guidance")))

    print("\n  discount arithmetic:")
    quote = tool(session, "compute_discount", product_id=SHOES, customer_budget=2500)
    check("a discount is sufficient to close the gap", quote.get("sufficient") is True)
    check("it is the SMALLEST whole percent that works", quote.get("discount_percent") == 11,
          f"{quote.get('discount_percent')}% (exact need {quote.get('required_percent')}%)")
    check("discounted price lands at or under the stated budget",
          quote.get("discounted_price", 1e9) <= 2500,
          f"₹{quote.get('discounted_price')} vs budget ₹2500")
    check("one percent less would NOT have been enough",
          discount.apply_discount(2799, 10) > 2500,
          f"10% -> ₹{discount.apply_discount(2799, 10)}")

    print("\n  gate:")
    decision = tool(session, "check_gate", product_id=SHOES, discount_percent=11)
    check("gate approved the discounted purchase", decision.get("allowed") is True,
          decision.get("explanation"))
    check("the gate checked the DISCOUNTED amount, not the sticker",
          approx(decision.get("amount_checked"), 2491.11),
          f"checked ₹{decision.get('amount_checked')}, sticker ₹{decision.get('sticker_amount')}")

    print("\n  the charge:")
    order = tool(session, "create_order", product_id=SHOES)
    check("order was created", order.get("success") is True, order.get("error"))
    check("charged the discounted price", approx(order.get("amount_rupees"), 2491.11),
          f"₹{order.get('amount_rupees')}")
    check("did NOT charge the sticker price", not approx(order.get("amount_rupees"), 2799),
          "this is the bug Tier 2 was written to fix")
    check("amount sent to Razorpay is the discounted figure in paise",
          order.get("amount") == 249111,
          f"{order.get('amount')} paise (sticker would be 279900)")
    check("savings recorded on the order", approx(order.get("discount_savings"), 307.89),
          f"₹{order.get('discount_savings')}")
    check("session cart total tracks the discounted amount",
          approx(session.cart_total_so_far, 2491.11), f"₹{session.cart_total_so_far}")

    print("\n  audit:")
    discount_rows = rows_of_type(session, "discount_computed")
    check("the discount decision was logged", len(discount_rows) == 1)
    if discount_rows:
        row = discount_rows[0]
        check("the logged decision carries its reasoning", bool(row["agent_reasoning"]))
        check("the logged decision records the percentage offered",
              row["result"].get("discount_percent") == 11)
    return session


def a2_discount_over_the_cap_is_denied():
    """A gap too wide for the ceiling. compute_discount must be honest about it,
    and the gate must refuse an over-cap discount outright."""
    header("A2 — a discount beyond the 20% ceiling is refused and explained")
    session = new_session()

    quote = tool(session, "compute_discount", product_id=SHOES, customer_budget=1500)
    print("\n  arithmetic is honest about being unable to help:")
    check("flagged as insufficient", quote.get("sufficient") is False)
    check("still returns the best permitted offer",
          quote.get("discount_percent") == gate.DEFAULT_MAX_DISCOUNT_PERCENT,
          f"{quote.get('discount_percent')}%")
    check("reports the discount that WOULD have been needed",
          approx(quote.get("required_percent"), 46.41, tol=0.05),
          f"{quote.get('required_percent')}%")
    check("reports the shortfall in rupees", approx(quote.get("gap"), 739.2),
          f"₹{quote.get('gap')} still over budget")
    check("explanation is plain language, safe to say out loud",
          "20%" in quote.get("explanation", ""), quote.get("explanation", "")[:80])
    check("tells the agent not to promise it",
          "do not promise" in quote.get("next_step", "").lower())

    print("\n  the gate refuses an over-cap discount:")
    denied = tool(session, "check_gate", product_id=SHOES, discount_percent=46)
    check("gate denied", denied.get("allowed") is False)
    check("denial reason is exceeds_discount_limit",
          denied.get("reason") == "exceeds_discount_limit", denied.get("reason"))
    check("denial explains the actual ceiling", "20%" in denied.get("explanation", ""),
          denied.get("explanation", "")[:90])

    print("\n  and the refusal actually blocks the sale:")
    order = tool(session, "create_order", product_id=SHOES)
    check("order refused for want of an approval", order.get("success") is False)
    check("refused because the gate never approved it",
          order.get("error_type") == "gate_not_checked", order.get("error_type"))
    check("nothing was charged", session.cart_total_so_far == 0,
          f"₹{session.cart_total_so_far}")

    print("\n  the ceiling boundary is exact:")
    at_cap = tool(session, "check_gate", product_id=SHOES, discount_percent=20)
    check("20% is allowed", at_cap.get("allowed") is True)
    over_cap = tool(session, "check_gate", product_id=SHOES, discount_percent=21)
    check("21% is denied", over_cap.get("allowed") is False)
    check("a denial revokes the approval the previous check granted",
          SHOES not in session.pending_approvals,
          f"pending={list(session.pending_approvals)}")

    print("\n  audit:")
    check("the denial was recorded as a rejection",
          any(e["gate_decision"] == "denied" for e in rows_of_type(session, "rejection")))
    return session


def a3_upsell_offered_and_accepted():
    header("A3 — one upsell offered after an order, and accepted")
    session = new_session()

    tool(session, "check_gate", product_id=SHOES)
    order = tool(session, "create_order", product_id=SHOES)
    check("the first order succeeded", order.get("success") is True, order.get("error"))

    comps = tool(session, "get_complementary_products", product_id=SHOES)
    print("\n  the offer:")
    check("a genuine offer was made", comps.get("upsell_offer") is True)
    check("the socks were among the products offered",
          SOCKS in [p["id"] for p in comps.get("products", [])],
          f"offered={[p['id'] for p in comps.get('products', [])]}")
    check("logged as upsell_offered (the attach-rate denominator)",
          len(rows_of_type(session, "upsell_offered")) == 1)

    print("\n  the customer takes it:")
    tool(session, "check_gate", product_id=SOCKS)
    upsell_order = tool(session, "create_order", product_id=SOCKS)
    check("the suggested item was ordered", upsell_order.get("success") is True)
    check("recognised as an accepted upsell — by code, not by the model",
          upsell_order.get("upsell_accepted") is True)
    check("logged as upsell_accepted (the numerator)",
          len(rows_of_type(session, "upsell_accepted")) == 1)
    check("cart total is the sum of both orders",
          approx(session.cart_total_so_far, 2799 + 349), f"₹{session.cart_total_so_far}")

    outcome = agent.finalize_session(session)
    check("finalizing does not downgrade an acceptance to a decline",
          outcome == "accepted", f"outcome={outcome}")
    check("no phantom decline was written",
          len(rows_of_type(session, "upsell_declined")) == 0)
    return session


def a4_upsell_declined_and_never_repeated():
    header("A4 — an upsell declined, and never offered a second time")
    session = new_session()

    tool(session, "check_gate", product_id=YOGA_MAT)
    check("order succeeded", tool(session, "create_order", product_id=YOGA_MAT).get("success"))

    first = tool(session, "get_complementary_products", product_id=YOGA_MAT)
    check("the one offer was made", first.get("upsell_offer") is True)

    print("\n  a second attempt is refused in code, not discouraged in a prompt:")
    second = tool(session, "get_complementary_products", product_id=YOGA_MAT)
    check("second lookup refused", second.get("refused") is True)
    check("refusal reason is the session limit",
          second.get("reason") == "upsell_limit_reached", second.get("reason"))
    check("refusal returns no products to tempt the model with",
          second.get("products") == [])

    third = tool(session, "get_complementary_products", product_id=SHOES)
    check("the limit is per SESSION, not per product", third.get("refused") is True,
          "a different product must not buy a second offer")

    print("\n  the decline is recorded when the session closes:")
    outcome = agent.finalize_session(session)
    check("resolved as declined", outcome == "declined", f"outcome={outcome}")
    check("exactly one offer in the log",
          len(rows_of_type(session, "upsell_offered")) == 1)
    check("exactly one decline in the log",
          len(rows_of_type(session, "upsell_declined")) == 1)
    check("no acceptance was recorded",
          len(rows_of_type(session, "upsell_accepted")) == 0)

    agent.finalize_session(session)
    check("finalizing twice does not double-count the decline",
          len(rows_of_type(session, "upsell_declined")) == 1,
          "the metric must not drift on a repeated close")
    return session


def a5_refund_a_valid_order():
    header("A5 — refunding a real order")
    session = new_session()

    tool(session, "check_gate", product_id=SHOES)
    order = tool(session, "create_order", product_id=SHOES)
    check("order to refund was created", order.get("success") is True)

    print("\n  a refund larger than the order is refused:")
    too_big = tool(session, "refund_order", product_id=SHOES, amount_in_rupees=5000,
                   reason="testing the upper bound")
    check("refused", too_big.get("success") is False)
    check("reason is refund_exceeds_order",
          too_big.get("error_type") == "refund_exceeds_order", too_big.get("error_type"))
    check("customer-safe message names the real amount",
          "2799" in too_big.get("user_message", "").replace(",", ""),
          too_big.get("user_message", "")[:80])

    print("\n  the real refund:")
    refund = tool(session, "refund_order", product_id=SHOES, reason="wrong size")
    check("refund succeeded", refund.get("success") is True, refund.get("error"))
    check("it was a full refund", refund.get("refund_type") == "full")
    check("the full order value came back", approx(refund.get("amount_rupees"), 2799),
          f"₹{refund.get('amount_rupees')}")
    check("it refunded against the payment, not the order",
          str(refund.get("payment_id", "")).startswith("pay_"),
          f"payment_id={refund.get('payment_id')}")
    check("refunding returned the spending headroom too",
          session.cart_total_so_far == 0, f"₹{session.cart_total_so_far}")
    if razorpay_client.is_mock_mode():
        check("the mock refund is stamped as a mock", refund.get("mock") is True,
              "a stub must never be mistakable for real money moving")

    print("\n  the same order cannot be refunded twice:")
    again = tool(session, "refund_order", product_id=SHOES, reason="trying it on")
    check("second refund refused", again.get("success") is False)
    check("reason is already_refunded",
          again.get("error_type") == "already_refunded", again.get("error_type"))

    print("\n  audit:")
    check("exactly one refund recorded",
          len(rows_of_type(session, "refund_created")) == 1)
    check("the two refusals were recorded as rejections",
          len([e for e in rows_of_type(session, "rejection")
               if e["action_params"].get("tool") == "refund_order"]) == 2)
    return session


def a6_refund_with_no_matching_order():
    header("A6 — a refund with nothing to refund, failing gracefully")
    session = new_session()

    missing = tool(session, "refund_order", product_id=NO_SUCH_PRODUCT,
                   reason="I want to return this")
    print("\n  no such order:")
    check("refused rather than raised", missing.get("success") is False)
    check("reason is no_matching_order",
          missing.get("error_type") == "no_matching_order", missing.get("error_type"))
    check("the customer gets an explanation and a way forward",
          "couldn't find" in missing.get("user_message", "").lower()
          and "?" in missing.get("user_message", ""),
          missing.get("user_message", "")[:90])
    check("no internal detail leaked into the customer message",
          "product_id=" not in missing.get("user_message", ""),
          "the raw lookup keys belong in `error`, not in front of a shopper")
    check("the technical detail IS kept for the log",
          NO_SUCH_PRODUCT in missing.get("error", ""))

    print("\n  no order identified at all:")
    vague = tool(session, "refund_order", reason="cancel my thing")
    check("refused", vague.get("success") is False)
    check("reason is no_order_specified",
          vague.get("error_type") == "no_order_specified", vague.get("error_type"))
    check("the agent is told to ask which order",
          "which order" in vague.get("user_message", "").lower(),
          vague.get("user_message", "")[:80])

    print("\n  audit:")
    check("both failures were recorded",
          len([e for e in rows_of_type(session, "rejection")
               if e["action_params"].get("tool") == "refund_order"]) == 2)
    check("no refund event was written",
          len(rows_of_type(session, "refund_created")) == 0,
          "a failed refund must never look like a successful one in the log")
    return session


def a7_customer_and_purchase_history():
    header("A7 — customer record, and history that survives the session")
    session = new_session()

    print("\n  customer linkage:")
    check("a customer was attached at session start",
          bool(session.customer_id), f"customer_id={session.customer_id}")
    check("the linkage was audited",
          len(rows_of_type(session, "customer_linked")) == 1)
    if razorpay_client.is_mock_mode():
        check("the mock customer is stamped as a mock",
              session.customer.get("mock") is True)
        check("mock customer ids are stable across sessions",
              new_session().customer_id == session.customer_id,
              "history lookups depend on the same shopper resolving the same way")

    tool(session, "check_gate", product_id=YOGA_MAT)
    order = tool(session, "create_order", product_id=YOGA_MAT)
    check("an order was placed to become history", order.get("success") is True)

    print("\n  a later, separate session can see it:")
    later = new_session()
    check("it is genuinely a different session",
          later.session_id != session.session_id)
    history = tool(later, "get_past_orders", limit=10)
    order_ids = [o.get("order_id") for o in history.get("orders", [])]
    check("past orders were found", history.get("count", 0) > 0,
          f"{history.get('count')} orders")
    check("the order just placed is in the history",
          order.get("id") in order_ids, f"looking for {order.get('id')}")
    check("history is attributed to the customer",
          history.get("customer_id") == session.customer_id)
    check("the lookup was audited",
          len(rows_of_type(later, "past_orders_lookup")) == 1)

    print("\n  and a refund can reach across sessions:")
    cross = tool(later, "refund_order", order_id=order.get("id"),
                 reason="changed my mind since last time")
    check("the past-session order was refundable", cross.get("success") is True,
          cross.get("error"))
    check("it was found in history, not in this session's cart",
          cross.get("order_source") == "past_session", cross.get("order_source"))
    return session


def a8_attach_rate_metric(before):
    """The pitch-video number. Asserted as a delta, because the audit log is
    cumulative across every run and an absolute figure would mean nothing."""
    header("A8 — attach rate, computed from the audit log alone")
    after = metrics.compute_attach_rate()

    offered = after["offered"] - before["offered"]
    accepted = after["accepted"] - before["accepted"]
    declined = after["declined"] - before["declined"]

    print(f"\n  this run contributed: offered {offered}, accepted {accepted}, declined {declined}")
    check("both upsell scenarios registered an offer", offered == 2, f"{offered}")
    check("A3's acceptance was counted", accepted == 1, f"{accepted}")
    check("A4's decline was counted", declined == 1, f"{declined}")
    check("every offer this run resolved one way or the other",
          accepted + declined == offered)

    print(f"\n  cumulative attach rate: {after['attach_rate_percent']}%  "
          f"({after['accepted']}/{after['offered']})")
    check("attach rate is a clean number in [0, 1]",
          isinstance(after["attach_rate"], float) and 0.0 <= after["attach_rate"] <= 1.0,
          f"{after['attach_rate']}")
    check("it is accepted / offered",
          approx(after["attach_rate"], after["accepted"] / after["offered"], tol=1e-4)
          if after["offered"] else True)
    check("unresolved offers are shown, not hidden in the denominator",
          "unresolved" in after)
    return after


# ===========================================================================
# PART B — real conversations through the model
# ===========================================================================

def b1_negotiate_then_buy_at_a_discount():
    header("B1 (live model) — a budget gap talked through and closed")
    session = new_session()

    reply = say(session, "I'm after some running shoes for daily jogging, but I can only spend 2500.")
    log = audit.get_session_log(session.session_id)
    searches = [e for e in log if e["action_type"] == "search"]
    negotiated = any(isinstance(e["result"], dict) and "budget_negotiation" in e["result"]
                     for e in searches)

    print("\n  assertions:")
    check("the agent searched with the stated budget",
          any(e["action_params"].get("max_price") for e in searches),
          f"{len(searches)} search(es)")
    check("the budget-negotiation path fired", negotiated)
    check("it did not dead-end with 'nothing found'",
          "nothing" not in reply.lower() or "2799" in reply.replace(",", ""),
          "the closest item should be named with its price")

    say(session, "Yes please, see if a discount can bridge that.")
    say(session, "Perfect, order them.")

    log = audit.get_session_log(session.session_id)
    orders = [e["result"] for e in log if e["action_type"] == "order_created"]
    discounts = [e for e in log if e["action_type"] == "discount_computed"]
    gate_rows = [e for e in log if e["action_type"] == "gate_check"]

    check("a discount was actually computed", len(discounts) >= 1)
    check("an order was created", len(orders) == 1, f"{len(orders)} order(s)")
    if orders:
        order = orders[0]
        check("the order carries a discount", (order.get("discount_percent") or 0) > 0,
              f"{order.get('discount_percent')}%")
        check("the customer was charged less than the sticker price",
              order.get("amount_rupees", 1e9) < order.get("sticker_amount", 0),
              f"₹{order.get('amount_rupees')} vs sticker ₹{order.get('sticker_amount')}")
        check("the charge is within the budget the customer stated",
              order.get("amount_rupees", 1e9) <= 2500, f"₹{order.get('amount_rupees')}")
        check("the discount stayed inside the ceiling",
              (order.get("discount_percent") or 0) <= gate.DEFAULT_MAX_DISCOUNT_PERCENT)
        check("the charge matches what the gate approved",
              any(approx(e["result"].get("amount_checked"), order.get("amount_rupees"))
                  for e in gate_rows),
              "the money charged must be the money that was authorised")
    return session


def b2_buy_then_decline_upsell_then_refund():
    header("B2 (live model) — buy, one suggestion declined, then a refund")
    session = new_session()

    say(session, "I'd like the ZenFlow Yoga Mat, product P011. Just one, please order it.")

    log = audit.get_session_log(session.session_id)
    orders = [e["result"] for e in log if e["action_type"] == "order_created"]
    offers = [e for e in log if e["action_type"] == "upsell_offered"]

    print("\n  assertions:")
    check("the order went through", len(orders) == 1, f"{len(orders)} order(s)")
    check("one complementary suggestion was made after the order", len(offers) == 1,
          f"{len(offers)} offer(s)")

    reply = say(session, "No thanks, nothing else for me.")
    check("the agent accepted the no briefly", len(reply) < 400, f"{len(reply)} chars")
    check("it did not try for a second suggestion",
          len([e for e in audit.get_session_log(session.session_id)
               if e["action_type"] == "upsell_offered"]) == 1)

    say(session, "Actually — I've changed my mind about the mat. Can you cancel it and refund me?")

    log = audit.get_session_log(session.session_id)
    refunds = [e["result"] for e in log if e["action_type"] == "refund_created"]
    check("the refund was processed", len(refunds) == 1, f"{len(refunds)} refund(s)")
    if refunds and orders:
        check("the full order value was returned",
              approx(refunds[0].get("amount_rupees"), orders[0].get("amount_rupees")),
              f"₹{refunds[0].get('amount_rupees')} of ₹{orders[0].get('amount_rupees')}")
    check("the refund freed the spending headroom", session.cart_total_so_far == 0,
          f"₹{session.cart_total_so_far}")

    outcome = agent.finalize_session(session)
    check("the unaccepted suggestion was booked as declined", outcome == "declined",
          f"outcome={outcome}")
    return session


def b3_refund_something_never_bought():
    header("B3 (live model) — asking to return something that was never ordered")
    session = agent.ShoppingSession()  # no customer link, so no history to find
    agent.link_customer(session, customer_details={
        "name": "History-Free Tester",
        "email": "no-history@example.com",
        "contact": "+910000000001",
    })

    reply = say(session, "I want to return the trekking shoes I bought, they don't fit.")

    log = audit.get_session_log(session.session_id)
    attempts = [e for e in log if e["action_params"].get("tool") == "refund_order"]
    refunds = [e for e in log if e["action_type"] == "refund_created"]

    print("\n  assertions:")
    check("the agent tried to look the order up", len(attempts) >= 1,
          f"{len(attempts)} attempt(s)")
    check("no refund was issued", len(refunds) == 0)
    check("the failed attempt was still audited",
          any(e["action_type"] == "rejection" for e in attempts))
    check("the agent did not claim a refund happened",
          not any(word in reply.lower() for word in ("refunded", "processed your refund")),
          reply[:100])
    check("it asked the customer to identify the order", "?" in reply)
    return session


# ===========================================================================

def main():
    run_llm = "--no-llm" not in sys.argv

    print(f"razorpay  : {'MOCK MODE — orders, refunds and customers are stubs' if razorpay_client.is_mock_mode() else ('test keys found' if razorpay_client.credentials_available() else 'NO CREDENTIALS — order creation will fail gracefully')}")
    print(f"audit db  : {os.path.abspath(audit.DB_PATH)}")
    print(f"live model: {'yes — ' + agent.MODEL if run_llm else 'no (--no-llm)'}")

    if run_llm and not os.environ.get("ANTHROPIC_API_KEY"):
        print("\nANTHROPIC_API_KEY is not set — running the deterministic half only.")
        run_llm = False

    # Warm the embedding model before the first scenario so its load time doesn't
    # look like part of a test.
    catalog.search_catalog("warmup", top_k=1)

    attach_before = metrics.compute_attach_rate()

    sessions = [
        a1_discount_bridges_the_gap(),
        a2_discount_over_the_cap_is_denied(),
        a3_upsell_offered_and_accepted(),
        a4_upsell_declined_and_never_repeated(),
        a5_refund_a_valid_order(),
        a6_refund_with_no_matching_order(),
        a7_customer_and_purchase_history(),
    ]
    a8_attach_rate_metric(attach_before)

    if run_llm:
        sessions += [
            b1_negotiate_then_buy_at_a_discount(),
            b2_buy_then_decline_upsell_then_refund(),
            b3_refund_something_never_bought(),
        ]

    # Close every session's books before reporting metrics. finalize_session is
    # idempotent, so the scenarios that already called it are unaffected; this
    # catches the ones (like B1) that end with an offer still hanging, which
    # would otherwise sit in "unresolved" forever and quietly distort the rate.
    for session in sessions:
        agent.finalize_session(session)

    header("AUDIT TRAIL — A1 (the discounted purchase), in full")
    print(audit.format_session_log(sessions[0].session_id))

    header("METRICS")
    print(metrics.format_summary())

    header("RESULT")
    if failures:
        print(f"  {len(failures)} assertion(s) failed:")
        for name in failures:
            print(f"    - {name}")
        return 1
    print("  All assertions passed.")
    print("\n  Session ids for manual audit replay:")
    for session in sessions:
        print(f"    {session.session_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
