"""
test_revenue.py — revenue attribution.

    python3 test_revenue.py

Entirely OFFLINE. Builds a session with a known mix of events, then asserts the
reported numbers are exactly what those events imply.

The thing worth testing here is not the addition — it is what is deliberately
NOT added. Upsell revenue and price-watch conversions are subsets of realised
revenue, not additions to it, so a summary that adds all four components
together would overstate the headline. Group C is entirely about that.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")

os.environ.setdefault("RAZORPAY_MOCK", "1")

import agent
import audit
import catalog
import discount
import metrics

failures = []
RUN_ID = uuid.uuid4().hex[:8]
SESSION = f"test-revenue-{RUN_ID}"
CUSTOMER = f"cust_test_revenue_{RUN_ID}"

THIN = "P032"    # Rs.899, cost Rs.700 -> margin floor binds at 14%
SOCKS = "P002"   # Rs.349 — the classic upsell


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def approx(a, b, tol=0.005):
    return abs(float(a) - float(b)) <= tol


def build_session():
    """One session with a normal order, a margin refusal, and an accepted upsell.

    Written through the real audit layer rather than by poking the database, so
    the shapes the reporter reads are the shapes the agent actually produces.
    """
    # 1. a real order
    audit.log_event(
        session_id=SESSION, user_query="I'll take the creatine",
        agent_reasoning="Gate approved; creating the order.",
        action_type="order_created",
        action_params={"tool": "create_order", "product_id": THIN},
        result={"id": "order_TEST001", "amount_rupees": 899.0, "product_name": "CreaPure",
                "quantity_ordered": 1, "notes": {"product_id": THIN}},
        gate_decision="allowed", customer_id=CUSTOMER,
    )

    # 2. a margin-capped discount — the merchant refused to go past 14%
    session = agent.ShoppingSession(session_id=SESSION)
    session.customer_id = CUSTOMER
    agent.execute_tool(session, "compute_discount", {
        "product_id": THIN, "customer_budget": 600,
        "reasoning": "checking whether a discount closes this gap",
    })

    # 3. an accepted upsell
    audit.log_event(
        session_id=SESSION, user_query=None,
        agent_reasoning="Customer bought the socks that were suggested to them.",
        action_type="upsell_accepted",
        action_params={"product_id": SOCKS},
        result={"product_name": "AirFlex Socks", "amount_rupees": 349.0,
                "order_id": "order_TEST002"},
        customer_id=CUSTOMER,
    )
    audit.log_event(
        session_id=SESSION, user_query=None,
        agent_reasoning="Order for the suggested socks.",
        action_type="order_created",
        action_params={"tool": "create_order", "product_id": SOCKS},
        result={"id": "order_TEST002", "amount_rupees": 349.0, "product_name": "AirFlex Socks",
                "quantity_ordered": 1, "notes": {"product_id": SOCKS}},
        gate_decision="allowed", customer_id=CUSTOMER,
    )


def a1_the_components_are_right():
    header("A1. Each component matches the events that produced it")

    r = metrics.get_revenue_impact_summary(SESSION)
    c = r["components"]

    check("scope is the session under test", r["scope"] == SESSION)
    check("two orders counted", c["orders_count"] == 2, str(c["orders_count"]))
    check("gross is 899 + 349 = 1248", approx(c["gross_orders_rupees"], 1248.0),
          str(c["gross_orders_rupees"]))
    check("nothing refunded", approx(c["refunded_rupees"], 0.0))
    check("realised revenue equals gross when there are no refunds",
          approx(r["realised_revenue_rupees"], 1248.0), str(r["realised_revenue_rupees"]))

    check("one margin protection event", c["margin_protection_events"] == 1,
          str(c["margin_protection_events"]))
    # 20% would charge Rs.719.20; the 14% margin ceiling charges Rs.773.14.
    check("margin protected is Rs.53.94", approx(r["margin_protected_rupees"], 53.94),
          f"Rs.{r['margin_protected_rupees']}")

    check("one accepted upsell", c["upsells_accepted"] == 1)
    check("upsell revenue is Rs.349", approx(c["upsell_revenue_rupees"], 349.0),
          str(c["upsell_revenue_rupees"]))


def a2_the_headline_adds_up():
    header("A2. The headline is realised revenue plus protected margin")

    r = metrics.get_revenue_impact_summary(SESSION)
    expected = r["realised_revenue_rupees"] + r["margin_protected_rupees"]
    check("attributed impact = realised + protected",
          approx(r["attributed_impact_rupees"], expected),
          f"Rs.{r['attributed_impact_rupees']} vs Rs.{round(expected, 2)}")
    check("which is Rs.1301.94", approx(r["attributed_impact_rupees"], 1301.94),
          f"Rs.{r['attributed_impact_rupees']}")


def a3_refunds_reduce_it():
    header("A3. A refund reduces realised revenue")

    before = metrics.get_revenue_impact_summary(SESSION)["realised_revenue_rupees"]
    audit.log_event(
        session_id=SESSION, user_query="refund the socks please",
        agent_reasoning="Customer asked to return the socks.",
        action_type="refund_created",
        action_params={"tool": "refund_order", "product_id": SOCKS},
        result={"success": True, "amount_rupees": 349.0, "order_id": "order_TEST002"},
        customer_id=CUSTOMER,
    )
    after = metrics.get_revenue_impact_summary(SESSION)
    check("realised revenue drops by the refunded amount",
          approx(before - after["realised_revenue_rupees"], 349.0),
          f"Rs.{before} -> Rs.{after['realised_revenue_rupees']}")
    check("the refund is counted", after["components"]["refunds_count"] == 1)
    check("margin protection is unaffected by a refund",
          approx(after["margin_protected_rupees"], 53.94))

    # A failed refund must not reduce anything — no money moved.
    audit.log_event(
        session_id=SESSION, user_query=None, agent_reasoning="Refund attempt failed.",
        action_type="refund_created", action_params={"tool": "refund_order"},
        result={"success": False, "amount_rupees": 899.0, "error_type": "gateway_error"},
        customer_id=CUSTOMER,
    )
    check("a FAILED refund does not reduce revenue",
          approx(metrics.get_revenue_impact_summary(SESSION)["realised_revenue_rupees"],
                 after["realised_revenue_rupees"]),
          "only successful refunds count")


# ---------------------------------------------------------------------------
# B. Price-watch conversions
# ---------------------------------------------------------------------------

def b1_watch_conversion_needs_both_halves():
    header("B1. A watch converts only when the same customer later buys it")

    session_id = f"test-revenue-watch-{RUN_ID}"
    customer = f"cust_test_watch_{RUN_ID}"

    # A watch fires, and nobody buys anything.
    audit.log_event(
        session_id=session_id, user_query=None,
        agent_reasoning="Price watch condition met.",
        action_type="price_watch_triggered",
        action_params={"watch_id": 1, "product_id": THIN},
        result={"current_price": 799.0}, customer_id=customer,
    )
    r = metrics.get_revenue_impact_summary(session_id)
    check("a triggered watch alone is not revenue",
          r["components"]["price_watch_conversions"] == 0,
          str(r["components"]["price_watch_conversions"]))
    check("...but the trigger is still counted",
          r["components"]["price_watches_triggered"] == 1)

    # Now the customer comes back and buys that product.
    audit.log_event(
        session_id=session_id, user_query="ok I'll take it now",
        agent_reasoning="Customer returned after the price drop.",
        action_type="order_created",
        action_params={"tool": "create_order", "product_id": THIN},
        result={"id": "order_TEST003", "amount_rupees": 799.0,
                "notes": {"product_id": THIN}},
        gate_decision="allowed", customer_id=customer,
    )
    r = metrics.get_revenue_impact_summary(session_id)
    check("the watch now counts as converted",
          r["components"]["price_watch_conversions"] == 1)
    check("and carries the order value",
          approx(r["components"]["price_watch_revenue_rupees"], 799.0),
          str(r["components"]["price_watch_revenue_rupees"]))

    # An order for a DIFFERENT product must not be attributed to the watch.
    other = f"test-revenue-watch-b-{RUN_ID}"
    audit.log_event(
        session_id=other, user_query=None, agent_reasoning="Watch fired.",
        action_type="price_watch_triggered",
        action_params={"watch_id": 2, "product_id": THIN},
        result={"current_price": 799.0}, customer_id=customer + "-x",
    )
    audit.log_event(
        session_id=other, user_query=None, agent_reasoning="Bought something else.",
        action_type="order_created", action_params={"tool": "create_order"},
        result={"id": "order_TEST004", "amount_rupees": 349.0,
                "notes": {"product_id": SOCKS}},
        gate_decision="allowed", customer_id=customer + "-x",
    )
    r = metrics.get_revenue_impact_summary(other)
    check("buying a different product does not convert the watch",
          r["components"]["price_watch_conversions"] == 0,
          str(r["components"]["price_watch_conversions"]))


# ---------------------------------------------------------------------------
# C. No double counting — the part that keeps the headline honest
# ---------------------------------------------------------------------------

def c1_subsets_are_not_added_twice():
    header("C1. Upsell and watch revenue are subsets, not additions")

    r = metrics.get_revenue_impact_summary(SESSION)
    c = r["components"]

    naive = (r["realised_revenue_rupees"] + r["margin_protected_rupees"]
             + c["upsell_revenue_rupees"] + c["price_watch_revenue_rupees"])
    check("the headline is NOT the naive sum of all four components",
          not approx(r["attributed_impact_rupees"], naive),
          f"headline Rs.{r['attributed_impact_rupees']} vs naive Rs.{round(naive, 2)}")
    check("upsell revenue never exceeds realised revenue",
          c["upsell_revenue_rupees"] <= c["gross_orders_rupees"],
          f"Rs.{c['upsell_revenue_rupees']} <= Rs.{c['gross_orders_rupees']}")


def c2_empty_scope_is_all_zeros():
    header("C2. A session with no events reports zeros, not an error")

    r = metrics.get_revenue_impact_summary(f"nonexistent-{RUN_ID}")
    check("attributed impact is 0", approx(r["attributed_impact_rupees"], 0.0))
    check("realised revenue is 0", approx(r["realised_revenue_rupees"], 0.0))
    check("no divide-by-zero on close rate",
          r["negotiation"]["close_rate_percent"] == 0.0)


def c3_negotiation_outcomes_are_reported():
    header("C3. Negotiation outcomes appear in the summary")

    session_id = f"test-revenue-neg-{RUN_ID}"
    for action in ("negotiation_accepted", "negotiation_accepted", "negotiation_walked_away"):
        audit.log_event(
            session_id=session_id, user_query=None,
            agent_reasoning="negotiation outcome fixture",
            action_type=action, action_params={"actor": "buyer_agent"},
            result={"decision": action}, customer_id=CUSTOMER,
        )
    n = metrics.get_revenue_impact_summary(session_id)["negotiation"]
    check("closed count", n["closed"] == 2, str(n["closed"]))
    check("walked-away count", n["walked_away"] == 1, str(n["walked_away"]))
    check("close rate is 66.7%", approx(n["close_rate_percent"], 66.7, tol=0.05),
          str(n["close_rate_percent"]))


def c4_the_printed_block_is_readable():
    header("C4. The pitch-video block renders")

    text = metrics.format_revenue_impact(SESSION)
    check("names the headline", "ATTRIBUTED IMPACT" in text)
    check("shows margin protected", "margin protected" in text)
    check("labels the subsets explicitly",
          "breakdown, not additions" in text, text[:0])
    check("carries the session scope", SESSION in text)
    check("all-sessions scope also renders",
          "all sessions" in metrics.format_revenue_impact())
    print("\n" + "\n".join("      " + l for l in text.splitlines()))


def main():
    print("=" * 70)
    print("  REVENUE ATTRIBUTION")
    print("=" * 70)
    print(f"  run id: {RUN_ID}")
    build_session()

    a1_the_components_are_right()
    a2_the_headline_adds_up()
    a3_refunds_reduce_it()
    b1_watch_conversion_needs_both_halves()
    c1_subsets_are_not_added_twice()
    c2_empty_scope_is_all_zeros()
    c3_negotiation_outcomes_are_reported()
    c4_the_printed_block_is_readable()

    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    if failures:
        print(f"\n  {len(failures)} assertion(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("\n  All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
