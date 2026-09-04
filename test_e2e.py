"""
test_e2e.py — end-to-end exercise of the Core tier.

Runs four real conversations against the real agent (real model calls, real
catalog, real gate, real audit writes) and then asserts against the audit log
that the right things happened for the right recorded reasons.

    export ANTHROPIC_API_KEY=...
    RAZORPAY_MOCK=1 python3 test_e2e.py          # offline, stubbed orders
    RAZORPAY_KEY_ID=... RAZORPAY_KEY_SECRET=... python3 test_e2e.py   # real test-mode orders

Scenarios:
  1. Vague request          -> agent asks ONE clarifying question, does not search yet
  2. Search and buy         -> gate approves, Razorpay order created, cart total advances
  3. Over the item limit    -> gate denies (exceeds_single_item_limit), explained in plain language
  4. The split-purchase exploit -> two affordable items whose *sum* breaks the session
                                   limit; the second is denied on cumulative total

Nothing here relaxes gate.py. Scenarios 3 and 4 are built from real catalog items
(the priciest item is ₹5499, under the ₹6000 per-item limit) by buying a quantity
of two, and by chaining two purchases — which is exactly the behaviour the
cumulative limit exists to catch.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

import agent
import audit
import razorpay_client

failures = []


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def say(session, message):
    print(f"\n  you   > {message}")
    reply = agent.run_turn(session, message)
    print(f"  agent > {reply}")
    return reply


def actions(session):
    """(action_type, gate_decision) for every audited action in the session."""
    return [(e["action_type"], e["gate_decision"]) for e in audit.get_session_log(session.session_id)]


def tools_used(session):
    return [e["action_params"].get("tool") for e in audit.get_session_log(session.session_id)]


def header(title):
    print("\n" + "=" * 72)
    print(f"  {title}")
    print("=" * 72)


# ---------------------------------------------------------------------------
# 1. Vague request -> one clarifying question, no premature search
# ---------------------------------------------------------------------------

def scenario_vague_query():
    header("SCENARIO 1 — vague request should trigger ONE clarifying question")
    session = agent.ShoppingSession()

    reply = say(session, "hey, I want to get into fitness")

    print("\n  assertions:")
    check("agent asked a question", "?" in reply, f"reply length {len(reply)} chars")
    check("agent did not search before clarifying", "search" not in [a for a, _ in actions(session)],
          f"actions={actions(session)}")
    check("no order created", "order_created" not in [a for a, _ in actions(session)])
    return session


# ---------------------------------------------------------------------------
# 2. Normal search-and-purchase -> gate allows, order created
# ---------------------------------------------------------------------------

def scenario_successful_purchase():
    header("SCENARIO 2 — specific request, search, gate approval, real order")
    session = agent.ShoppingSession()

    say(session, "I need lightweight running shoes for daily jogging, budget around 3000")
    say(session, "The CloudRunner Mesh ones sound right. Buy them for me, just one pair.")

    recorded = actions(session)
    used = tools_used(session)
    log = audit.get_session_log(session.session_id)

    print("\n  assertions:")
    check("catalog was searched", "search_catalog" in used, f"tools={used}")
    check("gate was checked", "check_gate" in used)
    check("gate approved", ("gate_check", "allowed") in recorded)
    check("order was created", ("order_created", "allowed") in recorded, f"actions={recorded}")
    check("gate_check precedes create_order",
          used.index("check_gate") < used.index("create_order") if
          ("check_gate" in used and "create_order" in used) else False)
    check("cart total advanced", session.cart_total_so_far > 0,
          f"₹{session.cart_total_so_far}")
    check("approval was consumed", session.pending_approvals == {},
          f"pending={session.pending_approvals}")
    check("every audited action recorded reasoning",
          all(e["agent_reasoning"] and e["agent_reasoning"] != "(no reasoning given)" for e in log))

    order_rows = [e for e in log if e["action_type"] == "order_created"]
    if order_rows:
        order = order_rows[0]["result"]
        expected_paise = razorpay_client.rupees_to_paise(order["amount_rupees"])
        check("amount sent to Razorpay is in paise", order["amount"] == expected_paise,
              f"₹{order['amount_rupees']} -> {order['amount']} paise")
        print(f"    order id: {order.get('id')}  mock={order.get('mock', False)}")
    return session


# ---------------------------------------------------------------------------
# 3. Over the per-item limit -> explained rejection
# ---------------------------------------------------------------------------

def scenario_gate_rejection_single_item():
    header("SCENARIO 3 — purchase over the ₹6000 per-item limit is denied and explained")
    session = agent.ShoppingSession()

    # 2 x PulseTrack Pro Smartwatch (₹5499) = ₹10998, over the ₹6000 item limit.
    say(session, "I want to buy two PulseTrack Pro Smartwatches, one for me and one for my sister.")
    reply = say(session, "Yes, go ahead and order both.")

    recorded = actions(session)
    log = audit.get_session_log(session.session_id)
    denials = [e for e in log if e["gate_decision"] == "denied"]

    print("\n  assertions:")
    check("gate denied the purchase", ("rejection", "denied") in recorded, f"actions={recorded}")
    check("denial reason is exceeds_single_item_limit",
          any(e["result"].get("reason") == "exceeds_single_item_limit" for e in denials),
          f"reasons={[e['result'].get('reason') for e in denials]}")
    check("no order was created", "order_created" not in [a for a, _ in recorded])
    check("cart total unchanged", session.cart_total_so_far == 0)
    check("agent explained the limit, not just refused",
          any(token in reply for token in ("6,000", "6000", "₹6")),
          "reply should surface the actual limit")
    return session


# ---------------------------------------------------------------------------
# 4. The split-purchase exploit -> cumulative limit catches it
# ---------------------------------------------------------------------------

def scenario_cumulative_limit():
    header("SCENARIO 4 — two individually-allowed items whose sum breaks the ₹10000 session limit")
    session = agent.ShoppingSession()

    # ₹5499 + ₹4999 = ₹10498. Each clears the ₹6000 item limit on its own;
    # together they exceed the ₹10000 session cap.
    say(session, "Buy me the PulseTrack Pro Smartwatch, product P024. Just order it.")
    print(f"\n  [cart total after first purchase: ₹{session.cart_total_so_far}]")
    reply = say(session, "Great. Now also order the IronCore Adjustable Dumbbell Set, P013.")

    log = audit.get_session_log(session.session_id)
    recorded = actions(session)
    orders = [e for e in log if e["action_type"] == "order_created"]
    denials = [e for e in log if e["gate_decision"] == "denied"]

    print("\n  assertions:")
    check("first purchase succeeded", len(orders) >= 1, f"orders={len(orders)}")
    check("second purchase was denied", ("rejection", "denied") in recorded)
    check("denial reason is exceeds_cart_total_limit",
          any(e["result"].get("reason") == "exceeds_cart_total_limit" for e in denials),
          f"reasons={[e['result'].get('reason') for e in denials]}")
    check("exactly one order created", len(orders) == 1, f"orders={len(orders)}")
    check("cart total reflects only the first order", session.cart_total_so_far == 5499,
          f"₹{session.cart_total_so_far}")
    check("gate saw the real running total",
          any(e["result"].get("cart_total_so_far") == 5499 for e in denials),
          "the cumulative total is read from session state, not from the model")
    check("agent explained the session limit",
          any(token in reply for token in ("10,000", "10000", "₹10")))
    return session


# ---------------------------------------------------------------------------

def main():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("ANTHROPIC_API_KEY is not set — cannot run the end-to-end test.")
        return 1

    print(f"model     : {agent.MODEL}")
    print(f"razorpay  : {'MOCK MODE' if razorpay_client.is_mock_mode() else ('test keys found' if razorpay_client.credentials_available() else 'NO CREDENTIALS — scenarios 2 and 4 will fail at the payment step')}")

    sessions = [
        scenario_vague_query(),
        scenario_successful_purchase(),
        scenario_gate_rejection_single_item(),
        scenario_cumulative_limit(),
    ]

    # Close each session's books. Scenario 2 ends with the agent having offered a
    # complementary product that nobody bought — without finalizing, that offer
    # stays forever "unresolved" and drags the attach-rate denominator around
    # without ever landing in either column.
    for session in sessions:
        agent.finalize_session(session)

    header("AUDIT TRAIL — scenario 4 (the cumulative-limit session), in full")
    print(audit.format_session_log(sessions[3].session_id))

    header("RESULT")
    if failures:
        print(f"  {len(failures)} assertion(s) failed:")
        for name in failures:
            print(f"    - {name}")
        return 1
    print("  All assertions passed.")
    print("\n  Session ids for manual audit replay:")
    for i, session in enumerate(sessions, start=1):
        print(f"    scenario {i}: {session.session_id}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
