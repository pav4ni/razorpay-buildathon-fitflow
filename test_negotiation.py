"""
test_negotiation.py — two-sided agent-to-agent negotiation.

    python3 test_negotiation.py              # groups A-C, fully offline, no API key
    python3 test_negotiation.py --live       # adds group D: one real negotiation

Groups A-C cost NOTHING to run: the buyer is a deterministic policy, and the
merchant's bounds can be driven through its own tool handlers with no model in
the loop. That is deliberate — the expensive part of an agent-to-agent demo is
the merchant's LLM, and none of the safety claims need it.

  A. BUYER POLICY     the hidden ceiling, the concession schedule, and the three
                      decisions. Includes the assertion that matters most: the
                      buyer's true_max never appears in anything it SAYS.
  B. THE EXCHANGE     both scenarios simulated end to end against the real
                      margin-aware discount logic — one closes, one does not.
  C. SAFETY           the point of the whole exercise: a negotiated price is
                      still subject to gate.py. Better haggling must not buy a
                      cheaper "yes" than the gate allows.
  D. LIVE (opt-in)    one real negotiation against the real merchant agent, to
                      confirm the loop works and the transcript reads correctly.
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

os.environ.setdefault("RAZORPAY_MOCK", "1")

import agent
import audit
import catalog
import discount
import gate
import negotiation

failures = []
skipped = []
RUN_ID = uuid.uuid4().hex[:8]

TEE = "P008"          # Rs.1199, cost Rs.525  -> 20% policy cap binds
CREATINE = "P032"     # Rs.899,  cost Rs.700  -> Rs.770 margin floor binds


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def simulate(sticker, cost, true_max, max_rounds=3):
    """Run a whole negotiation against the real discount logic, no model.

    The merchant's move is exactly what agent.py would compute for a stated
    budget: discount.compute_minimum_discount with the item's real cost. So this
    exercises the actual pricing rule, only without paying for the prose around it.
    """
    policy = negotiation.BuyerPolicy("the item", true_max, max_rounds=max_rounds)
    transcript = []
    while policy.outcome is None:
        stated = policy.stated_budget
        analysis = discount.compute_minimum_discount(stated, sticker, cost=cost)
        record = policy.decide(analysis["discounted_price"])
        transcript.append({
            "round": record["round"],
            "buyer_said": stated,
            "merchant_quoted": analysis["discounted_price"],
            "discount_percent": analysis["discount_percent"],
            "capped_by": analysis["capped_by"],
            "decision": record["decision"],
        })
    return policy, transcript


# ---------------------------------------------------------------------------
# A. Buyer policy
# ---------------------------------------------------------------------------

def a1_hidden_ceiling():
    header("A1. The buyer opens below its true ceiling and never states it")

    policy = negotiation.BuyerPolicy("a training tee", 1000, opening_ratio=0.80)
    check("opening claim is below the true ceiling",
          policy.opening_budget < policy.true_max,
          f"Rs.{policy.opening_budget:g} < Rs.{policy.true_max:g}")
    check("opening claim is 80% of the ceiling", policy.opening_budget == 800,
          str(policy.opening_budget))

    # THE assertion. Everything the buyer says is sent to the merchant's model;
    # if the real ceiling appears in any of it, the negotiation is theatre.
    said = " ".join([
        policy.opening_utterance(),
        policy.counter_utterance(),
        policy.accept_utterance(959.2),
        policy.walk_away_utterance(),
    ])
    check("the true ceiling never appears in anything the buyer SAYS",
          "1000" not in said, said[:120])
    check("the opening claim does appear", "800" in said)

    check("rejects a non-positive ceiling",
          _raises(lambda: negotiation.BuyerPolicy("x", 0)))
    check("rejects an out-of-range opening ratio",
          _raises(lambda: negotiation.BuyerPolicy("x", 100, opening_ratio=1.5)))


def _raises(fn):
    try:
        fn()
        return False
    except ValueError:
        return True


def a2_concession_schedule():
    header("A2. Concessions move toward the ceiling and stop there")

    policy = negotiation.BuyerPolicy("a tee", 1000, opening_ratio=0.80, max_rounds=3)
    budgets = [policy.budget_for_round(r) for r in (1, 2, 3)]
    check("schedule is 800 -> 885 -> 970", budgets == [800, 885, 970], str(budgets))
    check("the final concession stops SHORT of the true ceiling",
          max(budgets) < policy.true_max,
          f"max stated Rs.{max(budgets):g} < ceiling Rs.{policy.true_max:g}")
    check("no round, however far out, ever reaches the ceiling",
          policy.budget_for_round(99) < policy.true_max,
          f"round 99 would be Rs.{policy.budget_for_round(99):g}")
    check("concessions are monotonic", budgets == sorted(budgets))


def a3_the_three_decisions():
    header("A3. Accept, counter, walk away")

    # Merchant meets the stated ask -> accept immediately.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=3)
    r = p.decide(750)
    check("meets the stated budget -> ACCEPT", r["decision"] == negotiation.ACCEPT, r["decision"])

    # Above the stated ask with rounds left -> counter.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=3)
    r = p.decide(1200)
    check("above the ask, rounds left -> COUNTER", r["decision"] == negotiation.COUNTER)
    check("...and the stated budget rises, but stays under the ceiling",
          p.stated_budget == 885 and p.stated_budget < p.true_max, str(p.stated_budget))

    # Out of rounds, still within the true ceiling -> concede and accept.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=1)
    r = p.decide(980)
    check("out of rounds, within ceiling -> ACCEPT", r["decision"] == negotiation.ACCEPT)

    # Out of rounds, above the true ceiling -> walk.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=1)
    r = p.decide(1100)
    check("out of rounds, above ceiling -> WALK_AWAY", r["decision"] == negotiation.WALK_AWAY)

    # No price quoted at all is not a refusal.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=3)
    r = p.decide(None)
    check("no quote yet -> COUNTER, not a walk-away", r["decision"] == negotiation.COUNTER)


def a4_holds_merchant_to_its_best_quote():
    header("A4. The buyer holds the merchant to its BEST quote, not its latest")

    # This is the subtle one. The merchant offers the smallest discount that
    # clears the stated budget, so its quote gets WORSE as the buyer concedes.
    p = negotiation.BuyerPolicy("x", 1000, max_rounds=3)
    p.decide(959.20)     # round 1: good quote against a lowball ask
    p.decide(975.00)     # round 2: worse quote, because the ask went up
    r = p.decide(995.17)  # round 3: worse still — but out of rounds

    check("accepts on the final round", r["decision"] == negotiation.ACCEPT)
    check("best price seen is the round-1 quote, not the round-3 one",
          p.best_price == 959.20, f"Rs.{p.best_price}")
    check("the acceptance names the best quote, not the latest",
          "959.2" in p.accept_utterance(), p.accept_utterance())
    check("a naive buyer would have paid Rs.35.97 more",
          round(995.17 - p.best_price, 2) == 35.97)


# ---------------------------------------------------------------------------
# B. The exchange, against the real pricing logic
# ---------------------------------------------------------------------------

def b1_deal_closes_on_a_healthy_margin_item():
    header("B1. A tee with room to discount: the negotiation closes")

    product = catalog.get_product_by_id(TEE)
    policy, transcript = simulate(product["price"], product["cost"], true_max=1000)

    for row in transcript:
        print(f"      R{row['round']}  buyer Rs.{row['buyer_said']:<6g} -> merchant "
              f"Rs.{row['merchant_quoted']:<8g} ({row['discount_percent']}% off)"
              f"  -> {row['decision']}")

    check("ends in a deal", policy.outcome == negotiation.ACCEPT, str(policy.outcome))
    check("it took more than one round", policy.round > 1, f"{policy.round} rounds")
    check("it stayed within the round limit", policy.round <= 3, f"{policy.round} rounds")
    check("agreed price is at or under the hidden ceiling",
          policy.best_price <= policy.true_max,
          f"Rs.{policy.best_price:g} <= Rs.{policy.true_max:g}")
    check("the buyer beat its own ceiling by Rs.40.80",
          round(policy.true_max - policy.best_price, 2) == 40.80,
          f"Rs.{round(policy.true_max - policy.best_price, 2)}")
    check("the merchant never exceeded the 20% policy cap",
          all(r["discount_percent"] <= 20 for r in transcript))


def b2_merchant_holds_the_margin_floor():
    header("B2. A thin-margin supplement: the merchant refuses to chase the sale")

    product = catalog.get_product_by_id(CREATINE)
    floor = discount.margin_floor_price(product["cost"])
    policy, transcript = simulate(product["price"], product["cost"], true_max=700)

    for row in transcript:
        cap = f" [{row['capped_by']}]" if row["capped_by"] else ""
        print(f"      R{row['round']}  buyer Rs.{row['buyer_said']:<6g} -> merchant "
              f"Rs.{row['merchant_quoted']:<8g} ({row['discount_percent']}% off){cap}"
              f"  -> {row['decision']}")

    check("the buyer walks away", policy.outcome == negotiation.WALK_AWAY, str(policy.outcome))
    check("the merchant was pushed for the full 3 rounds", policy.round == 3, str(policy.round))

    # The whole point: pressure did not move the floor.
    check("every quote stayed at or above the margin floor",
          all(r["merchant_quoted"] >= floor for r in transcript),
          f"floor Rs.{floor:g}, quotes {[r['merchant_quoted'] for r in transcript]}")
    check("the merchant never improved its offer under pressure",
          len({r["merchant_quoted"] for r in transcript}) == 1,
          str({r["merchant_quoted"] for r in transcript}))
    check("every round reported the margin floor as the binding constraint",
          all(r["capped_by"] == discount.CAPPED_BY_MARGIN_FLOOR for r in transcript))
    check("the buyer's ceiling really was below the floor (an impossible deal)",
          policy.true_max < floor, f"Rs.{policy.true_max:g} < Rs.{floor:g}")


def b3a_never_loops_when_the_merchant_wont_quote():
    header("B3a. A merchant that never quotes a price must not loop forever")

    # REGRESSION. The first version of decide() returned COUNTER unconditionally
    # when no price had been quoted, ignoring the round limit. Against a merchant
    # that kept asking a clarifying question instead of pricing anything, the
    # buyer re-stated its budget forever — and every lap was a paid API call.
    # It ran to 198 rounds before it was killed by hand.
    policy = negotiation.BuyerPolicy("a training tee", 1000, max_rounds=3)

    laps = 0
    while policy.outcome is None:
        laps += 1
        if laps > 20:
            break                      # the bug looked exactly like this
        policy.decide(None)            # merchant quotes nothing, every time

    check("it terminates instead of looping", policy.outcome is not None,
          f"still undecided after {laps} laps")
    check("it terminates within max_rounds", policy.round <= 3, f"{policy.round} rounds")
    check("no quote ever means WALK_AWAY, not a purchase",
          policy.outcome == negotiation.WALK_AWAY, str(policy.outcome))
    check("the reason given is honest about why",
          "never quoted a price" in policy.history[-1]["reasoning"],
          policy.history[-1]["reasoning"][:80])

    # And the buyer must have something useful to SAY in that situation, or it
    # invites the same question again — which is what caused the loop.
    said = policy.clarify_utterance()
    check("the clarification answers rather than repeating the budget",
          "whichever you'd recommend" in said, said[:90])
    check("...and still never states the true ceiling", "1000" not in said, said[:90])


def b3b_hard_cap_backstops_a_broken_policy():
    header("B3b. buyer_agent has a hard lap cap even if the policy misbehaves")

    import buyer_agent

    class NeverTerminates(negotiation.BuyerPolicy):
        """A deliberately broken policy: it never sets an outcome."""
        def decide(self, merchant_price):
            record = super().decide(merchant_price)
            self.outcome = None        # refuse to finish
            return record

    class SilentTransport:
        """A merchant that answers 200 but never quotes anything."""
        def __init__(self):
            self.calls = 0
        def post(self, path, body, headers=None):
            self.calls += 1
            return 200, {"session_id": body.get("session_id"), "reply": "Men's or women's?",
                         "products": [], "discount_offers": [], "orders": []}

    transport = SilentTransport()
    buyer = buyer_agent.BuyerAgent(transport=transport, webhook_secret=None, verbose=False)

    scenario = {"name": "broken policy", "product_hint": "a tee", "product_id": TEE,
                "true_max": 1000, "max_rounds": 3, "session_id": f"test-neg-cap-{RUN_ID}"}

    original = negotiation.BuyerPolicy
    negotiation.BuyerPolicy = NeverTerminates
    try:
        result = buyer.run_negotiation(scenario)
    finally:
        negotiation.BuyerPolicy = original

    check("the run aborts instead of hanging", result["ok"] is False, str(result.get("ok")))
    check("it says the policy failed to terminate",
          "did not terminate" in str(result.get("error")), str(result.get("error")))
    check("it stopped after max_rounds + 2 calls at most", transport.calls <= 5,
          f"{transport.calls} merchant calls")


def b3_negotiation_is_bounded():
    header("B3. A negotiation always terminates inside its round limit")

    product = catalog.get_product_by_id(CREATINE)
    for rounds in (1, 2, 3, 5):
        policy, _ = simulate(product["price"], product["cost"], true_max=500, max_rounds=rounds)
        check(f"max_rounds={rounds} terminates in exactly {rounds} round(s)",
              policy.round == rounds and policy.outcome is not None,
              f"{policy.round} rounds, outcome {policy.outcome}")


# ---------------------------------------------------------------------------
# C. Safety — negotiation happens inside the gate, never around it
# ---------------------------------------------------------------------------

def c1_negotiated_price_still_faces_the_gate():
    header("C1. A negotiated discount is still subject to check_gate")

    session = agent.ShoppingSession(session_id=f"test-neg-gate-{RUN_ID}")
    session.customer_id = f"cust_test_neg_{RUN_ID}"

    # Simulate the buyer having "won" a 20% discount on the thin-margin item —
    # the deepest the POLICY cap allows. The margin floor must still refuse it.
    result = agent.execute_tool(session, "check_gate", {
        "product_id": CREATINE, "quantity": 1, "discount_percent": 20,
        "reasoning": "buyer negotiated hard for the maximum discount",
    })
    check("a hard-won 20% is still DENIED on margin", result["allowed"] is False)
    check("reason is below_margin_floor", result["reason"] == "below_margin_floor",
          str(result["reason"]))
    check("no approval was granted", CREATINE not in session.pending_approvals)

    order = agent.execute_tool(session, "create_order", {
        "product_id": CREATINE, "quantity": 1,
        "reasoning": "buyer agreed, placing the order",
    })
    check("create_order refuses without an approval", order["success"] is False)
    check("...for the gate_not_checked reason",
          order["error_type"] == "gate_not_checked", str(order.get("error_type")))


def c2_negotiation_cannot_beat_the_spending_limits():
    header("C2. Negotiation does not widen the per-item or session limits")

    session = agent.ShoppingSession(session_id=f"test-neg-limits-{RUN_ID}")
    session.customer_id = f"cust_test_neg_{RUN_ID}"

    # Even at the deepest legal discount, two dumbbell sets exceed the per-item cap.
    result = agent.execute_tool(session, "check_gate", {
        "product_id": "P013", "quantity": 2, "discount_percent": 20,
        "reasoning": "buyer negotiated a bulk discount",
    })
    check("bulk buy at max discount still hits the per-item limit",
          result["allowed"] is False and result["reason"] == "exceeds_single_item_limit",
          str(result.get("reason")))

    # And the cumulative session limit is unmoved by a good negotiation.
    session.cart_total_so_far = 9500
    result = agent.execute_tool(session, "check_gate", {
        "product_id": TEE, "quantity": 1, "discount_percent": 20,
        "reasoning": "buyer negotiated a discount on a second item",
    })
    check("a discounted item still hits the session cap",
          result["allowed"] is False and result["reason"] == "exceeds_cart_total_limit",
          str(result.get("reason")))


def c3_both_sides_reasoning_reaches_one_transcript():
    header("C3. Both sides' reasoning lands in ONE audit trail")

    session_id = f"test-neg-audit-{RUN_ID}"
    policy = negotiation.BuyerPolicy("a training tee", 1000, max_rounds=2)

    negotiation.log_opened(session_id, policy)
    rec = policy.decide(1150.0)
    negotiation.log_merchant_position(session_id, 1, {
        "price": 1150.0, "discount_percent": 4, "capped_by": None})
    negotiation.log_round(session_id, rec, policy.counter_utterance())
    rec = policy.decide(959.2)
    negotiation.log_round(session_id, rec, policy.accept_utterance())

    events = audit.get_session_log(session_id)
    types = [e["action_type"] for e in events]

    check("the opening move is logged", "negotiation_opened" in types, str(types))
    check("the merchant's position is logged", "merchant_counter_offer" in types)
    check("the buyer's counter is logged", "buyer_counter_offer" in types)
    check("the outcome is logged", "negotiation_accepted" in types)
    check("every row carries reasoning a human can read",
          all(e["agent_reasoning"] for e in events),
          f"{sum(1 for e in events if not e['agent_reasoning'])} rows without it")

    actors = {e["action_params"].get("actor") for e in events
              if isinstance(e["action_params"], dict)}
    check("the transcript names both actors", actors == {"buyer_agent", "merchant_agent"},
          str(actors))

    opened = [e for e in events if e["action_type"] == "negotiation_opened"][0]
    check("the hidden ceiling IS in the audit log, for the merchant to review",
          opened["result"]["hidden_true_max"] == 1000.0)
    check("...but never in anything the buyer said",
          all("1000" not in (e["user_query"] or "") for e in events),
          str([e["user_query"] for e in events if e["user_query"]])[:120])


# ---------------------------------------------------------------------------
# D. Live — one real negotiation (opt-in, costs money)
# ---------------------------------------------------------------------------

def d1_live_negotiation():
    header("D1. LIVE: a real negotiation against the real merchant agent")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("    [SKIP] ANTHROPIC_API_KEY not set.")
        skipped.append("live negotiation")
        return

    import buyer_agent

    scenario = dict(buyer_agent.NEGOTIATION_SCENARIOS[1])  # the walk-away case
    scenario["session_id"] = f"test-neg-live-{RUN_ID}"

    buyer = buyer_agent.BuyerAgent(
        transport=buyer_agent.InProcessTransport(),
        webhook_secret=os.environ.get("RAZORPAY_WEBHOOK_SECRET"),
        verbose=True,
    )
    result = buyer.run_negotiation(scenario)

    check("the negotiation ran", result["ok"] is True, str(result.get("error"))[:100])
    summary = result["summary"]
    check("it took more than one round — a negotiation, not a request",
          summary["rounds_used"] > 1, f"{summary['rounds_used']} rounds")
    check("the merchant held its line and the buyer walked",
          summary["outcome"] == negotiation.WALK_AWAY, str(summary["outcome"]))
    check("no order was created", not result["orders"], str(result["orders"]))

    floor = discount.margin_floor_price(catalog.get_product_by_id(CREATINE)["cost"])
    check("the merchant's best offer never went below the margin floor",
          summary["best_price_offered"] is None or summary["best_price_offered"] >= floor,
          f"best Rs.{summary['best_price_offered']} vs floor Rs.{floor:g}")

    events = audit.get_session_log(result["session_id"])
    types = [e["action_type"] for e in events]
    check("the audit trail holds both sides",
          "negotiation_opened" in types and "negotiation_walked_away" in types,
          str(sorted(set(types))))
    check("the merchant's own pricing reasoning is in the same trail",
          any(t in types for t in ("discount_computed", "margin_protection", "search")),
          str(sorted(set(types))))
    print(f"\n      replay this transcript: /api/audit/{result['session_id']}")


def main():
    live = "--live" in sys.argv
    print("=" * 70)
    print("  TWO-SIDED AGENT-TO-AGENT NEGOTIATION")
    print("=" * 70)
    print(f"  policy cap   : {gate.DEFAULT_MAX_DISCOUNT_PERCENT}%")
    print(f"  margin floor : cost + {gate.DEFAULT_MIN_MARGIN_PERCENT}%")
    print(f"  mode         : {'OFFLINE + LIVE' if live else 'OFFLINE only (pass --live to add group D)'}")
    print(f"  run id       : {RUN_ID}")

    a1_hidden_ceiling()
    a2_concession_schedule()
    a3_the_three_decisions()
    a4_holds_merchant_to_its_best_quote()

    b1_deal_closes_on_a_healthy_margin_item()
    b2_merchant_holds_the_margin_floor()
    b3a_never_loops_when_the_merchant_wont_quote()
    b3b_hard_cap_backstops_a_broken_policy()
    b3_negotiation_is_bounded()

    c1_negotiated_price_still_faces_the_gate()
    c2_negotiation_cannot_beat_the_spending_limits()
    c3_both_sides_reasoning_reaches_one_transcript()

    if live:
        d1_live_negotiation()

    print("\n" + "=" * 70)
    print("  RESULT")
    print("=" * 70)
    if skipped:
        for s in skipped:
            print(f"  SKIPPED: {s}")
    if failures:
        print(f"\n  {len(failures)} assertion(s) FAILED:")
        for f in failures:
            print(f"    - {f}")
        return 1
    print("\n  All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
