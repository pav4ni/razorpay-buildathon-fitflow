"""
test_margin.py — margin-aware discounting.

    python3 test_margin.py

Entirely OFFLINE. No ANTHROPIC_API_KEY, no Razorpay credentials, no network.
Everything under test is deterministic arithmetic plus the agent's own tool
handlers, driven directly.

The claim being tested, in one line: a discount is bounded by the tighter of two
independent ceilings — the 20% policy cap and the "never sell below cost + 10%"
margin floor — and when the margin floor is the tighter one, it wins.

Four groups:

  A. discount.py     the arithmetic, including the back-compat guarantee that
                     omitting `cost` reproduces the pre-margin behaviour exactly
  B. gate.py         the ENFORCEMENT. discount.py computes and can be skipped;
                     the gate is what a proposed discount cannot get around
  C. agent.py        the wiring, driven through execute_tool with no model in
                     the loop, including that cost never reaches the model
  D. catalog.json    every product actually has a plausible cost
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

import agent
import audit
import catalog
import discount
import gate

failures = []

RUN_ID = uuid.uuid4().hex[:8]

# Products the scenarios are pinned to, chosen because they sit on opposite
# sides of the two-ceiling disagreement. Asserted in group D so a catalog edit
# fails loudly here rather than quietly changing what these tests mean.
THIN_MARGIN = "P032"   # supplement,  Rs.899 retail / Rs.700 cost -> margin binds
FAT_MARGIN = "P008"    # apparel,    Rs.1199 retail / Rs.525 cost -> 20% cap binds


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


def new_session(name):
    session = agent.ShoppingSession(session_id=f"test-margin-{name}-{RUN_ID}")
    session.customer_id = f"cust_test_margin_{RUN_ID}"
    return session


def rows_of_type(session, action_type):
    return [e for e in audit.get_session_log(session.session_id)
            if e["action_type"] == action_type]


# ---------------------------------------------------------------------------
# A. discount.py — the arithmetic
# ---------------------------------------------------------------------------

def a1_back_compat_without_cost():
    header("A1. Omitting `cost` reproduces the pre-margin behaviour exactly")

    # These three are lifted from discount.py's own pre-existing self-test, so a
    # regression here means the margin work changed something it shouldn't have.
    r = discount.compute_minimum_discount(3000, 2799)
    check("already affordable -> 0%", r["discount_percent"] == 0 and r["sufficient"])

    r = discount.compute_minimum_discount(2500, 2799)
    check("realistic gap -> 11%", r["discount_percent"] == 11, str(r["discount_percent"]))
    check("charged Rs.2491.11", approx(r["discounted_price"], 2491.11), str(r["discounted_price"]))
    check("sufficient", r["sufficient"] is True)

    r = discount.compute_minimum_discount(1500, 2799)
    check("gap too wide -> capped at 20%", r["discount_percent"] == 20)
    check("not sufficient", r["sufficient"] is False)
    check("capped_by names the policy cap", r["capped_by"] == discount.CAPPED_BY_DISCOUNT_CEILING,
          str(r["capped_by"]))
    check("no cost fields leak into a costless call",
          "cost" not in r and "margin_floor_price" not in r)


def a2_margin_floor_arithmetic():
    header("A2. Margin floor arithmetic, and the deliberate rounding direction")

    check("floor is cost + 10%", approx(discount.margin_floor_price(700), 770.0),
          str(discount.margin_floor_price(700)))

    # Rs.899 at Rs.700 cost: the exact ceiling is 14.35%, which MUST floor to 14.
    # Rounding it up to 15% would sell at Rs.764.15 — under the Rs.770 floor.
    ceiling = discount.max_discount_for_margin(899, 700)
    check("ceiling rounds DOWN to 14%, not up to 15%", ceiling == 14, str(ceiling))
    check("14% clears the floor", discount.apply_discount(899, 14) >= 770.0,
          f"Rs.{discount.apply_discount(899, 14)} vs floor Rs.770")
    check("15% would have breached it", discount.apply_discount(899, 15) < 770.0,
          f"Rs.{discount.apply_discount(899, 15)} vs floor Rs.770")

    # An item already at or below its floor has no room at all.
    check("no room when cost+10% already exceeds price",
          discount.max_discount_for_margin(100, 95) == 0.0)


def a3_the_two_ceilings_disagree():
    header("A3. The case that matters: the two ceilings give different answers")

    thin_price, thin_cost = 899, 700
    fat_price, fat_cost = 1199, 525

    thin_ceiling, thin_why = discount.effective_discount_ceiling(thin_price, cost=thin_cost)
    fat_ceiling, fat_why = discount.effective_discount_ceiling(fat_price, cost=fat_cost)

    check("thin-margin item is capped at 14%, below the 20% policy cap",
          thin_ceiling == 14, str(thin_ceiling))
    check("...and reports the margin floor as the reason",
          thin_why == discount.CAPPED_BY_MARGIN_FLOOR, thin_why)

    check("fat-margin item is capped at 20% by policy", fat_ceiling == 20, str(fat_ceiling))
    check("...and reports the policy cap as the reason",
          fat_why == discount.CAPPED_BY_DISCOUNT_CEILING, fat_why)

    check("the two ceilings genuinely differ on this catalog",
          thin_ceiling != fat_ceiling, f"{thin_ceiling}% vs {fat_ceiling}%")

    # The money question: what the old, margin-blind code would have done.
    would_have_charged = discount.apply_discount(thin_price, 20)
    actually_charges = discount.apply_discount(thin_price, thin_ceiling)
    floor = discount.margin_floor_price(thin_cost)
    check("a margin-blind 20% would have sold BELOW cost+10%",
          would_have_charged < floor, f"Rs.{would_have_charged} < floor Rs.{floor}")
    check("the margin-aware price clears the floor",
          actually_charges >= floor, f"Rs.{actually_charges} >= floor Rs.{floor}")
    check("margin protection is worth Rs.53.94 on this one line",
          approx(actually_charges - would_have_charged, 53.94),
          f"Rs.{round(actually_charges - would_have_charged, 2)}")


def a4_capped_by_distinguishes_the_two_reasons():
    header("A4. `capped_by` distinguishes 'hit the cap' from 'would be unprofitable'")

    # Budget far under a thin-margin item -> margin floor is what stops us.
    thin = discount.compute_minimum_discount(600, 899, cost=700)
    check("thin-margin: not sufficient", thin["sufficient"] is False)
    check("thin-margin: capped_by == margin_floor",
          thin["capped_by"] == discount.CAPPED_BY_MARGIN_FLOOR, str(thin["capped_by"]))
    check("thin-margin: offers 14%, not 20%", thin["discount_percent"] == 14,
          str(thin["discount_percent"]))
    check("thin-margin: explanation names the margin floor, for the log",
          "margin floor" in thin["explanation"], thin["explanation"][:80])

    # Budget far under a fat-margin item -> the policy cap is what stops us.
    fat = discount.compute_minimum_discount(600, 1199, cost=525)
    check("fat-margin: not sufficient", fat["sufficient"] is False)
    check("fat-margin: capped_by == discount_cap",
          fat["capped_by"] == discount.CAPPED_BY_DISCOUNT_CEILING, str(fat["capped_by"]))
    check("fat-margin: offers the full 20%", fat["discount_percent"] == 20)

    # And when the budget IS met, nothing capped anything.
    fine = discount.compute_minimum_discount(850, 899, cost=700)
    check("budget met -> capped_by is None", fine["capped_by"] is None, str(fine["capped_by"]))
    check("budget met -> sufficient", fine["sufficient"] is True)
    check("budget met -> 6% offered", fine["discount_percent"] == 6, str(fine["discount_percent"]))


# ---------------------------------------------------------------------------
# B. gate.py — the enforcement, which is the part that actually binds
# ---------------------------------------------------------------------------

def b1_gate_refuses_below_the_floor():
    header("B1. The gate refuses a sale below the margin floor")

    # This is the bypass the gate exists to close: a discount of 20% is within
    # the policy cap, so nothing in discount.py has to have been called at all.
    denied = gate.check_gate(amount=719.20, cart_total_so_far=0, item_in_stock=True,
                             discount_percent=20, cost_basis=700)
    check("20% on a thin-margin item is DENIED", denied["allowed"] is False)
    check("reason is below_margin_floor", denied["reason"] == "below_margin_floor",
          str(denied["reason"]))
    check("explanation leaks no cost, margin or profit figure",
          not any(w in denied["explanation"].lower()
                  for w in ("cost", "margin", "profit", "wholesale", "700", "770")),
          denied["explanation"])

    allowed = gate.check_gate(amount=773.14, cart_total_so_far=0, item_in_stock=True,
                              discount_percent=14, cost_basis=700)
    check("14% on the same item is ALLOWED", allowed["allowed"] is True)


def b2_gate_boundary_is_exact():
    header("B2. The floor is an exact boundary, not a rounded one")

    def verdict(amount):
        return gate.check_gate(amount=amount, cart_total_so_far=0, item_in_stock=True,
                               discount_percent=10, cost_basis=700)["allowed"]

    check("Rs.769.99 (a paisa under) is denied", verdict(769.99) is False)
    check("Rs.770.00 (exactly the floor) is allowed", verdict(770.00) is True)
    check("Rs.770.01 is allowed", verdict(770.01) is True)
    # The pre-existing int rounding would have let 769.60 through as 770.
    check("Rs.769.60 is denied, not rounded up to the floor", verdict(769.60) is False)


def b3_margin_check_is_purely_additive():
    header("B3. The margin bound is additive — it weakens no existing check")

    # Omitting cost_basis must reproduce the old behaviour exactly.
    r = gate.check_gate(amount=719.20, cart_total_so_far=0, item_in_stock=True,
                        discount_percent=20)
    check("no cost_basis -> margin check does not run", r["allowed"] is True)

    # Every pre-existing rejection still fires, and still fires FIRST, even when
    # the margin floor would also have been breached.
    r = gate.check_gate(amount=100, cart_total_so_far=0, item_in_stock=False,
                        discount_percent=0, cost_basis=1000)
    check("out_of_stock still takes precedence", r["reason"] == "out_of_stock", str(r["reason"]))

    r = gate.check_gate(amount=7000, cart_total_so_far=0, item_in_stock=True,
                        discount_percent=0, cost_basis=100000)
    check("exceeds_single_item_limit still takes precedence",
          r["reason"] == "exceeds_single_item_limit", str(r["reason"]))

    r = gate.check_gate(amount=3000, cart_total_so_far=8500, item_in_stock=True,
                        discount_percent=0, cost_basis=100000)
    check("exceeds_cart_total_limit still takes precedence",
          r["reason"] == "exceeds_cart_total_limit", str(r["reason"]))

    r = gate.check_gate(amount=1000, cart_total_so_far=0, item_in_stock=True,
                        discount_percent=35, cost_basis=100000)
    check("exceeds_discount_limit still takes precedence",
          r["reason"] == "exceeds_discount_limit", str(r["reason"]))

    # And the ordinary happy path is untouched.
    r = gate.check_gate(amount=2799, cart_total_so_far=0, item_in_stock=True)
    check("an ordinary purchase is still approved", r["allowed"] is True)


# ---------------------------------------------------------------------------
# C. agent.py — the wiring, no model in the loop
# ---------------------------------------------------------------------------

def c1_compute_discount_is_margin_aware():
    header("C1. compute_discount through the agent respects the margin floor")

    session = new_session("compute")
    product = catalog.get_product_by_id(THIN_MARGIN)

    result = agent.execute_tool(session, "compute_discount", {
        "product_id": THIN_MARGIN,
        "customer_budget": 600,
        "reasoning": "customer named a budget well under this item's price",
    })

    check("not sufficient", result["sufficient"] is False)
    check("capped_by == margin_floor",
          result["capped_by"] == discount.CAPPED_BY_MARGIN_FLOOR, str(result.get("capped_by")))
    check("offered discount is below the 20% policy cap",
          result["discount_percent"] < gate.DEFAULT_MAX_DISCOUNT_PERCENT,
          f"{result['discount_percent']}% < 20%")
    check("next_step tells the model not to explain the reason",
          "cost or margin" in result["next_step"], result["next_step"][:90])

    # THE redaction assertion: the merchant's cost must not be in what the model reads.
    blob = json.dumps(result)
    check("tool result contains no `cost` field", '"cost"' not in blob)
    check("tool result contains no margin_floor_price", "margin_floor_price" not in blob)
    check("tool result contains no literal cost value", str(product["cost"]) not in blob,
          f"cost is Rs.{product['cost']}")


def c2_margin_protection_is_audited():
    header("C2. A margin-capped discount writes its own audit row, with the money")

    session = new_session("audit")
    agent.execute_tool(session, "compute_discount", {
        "product_id": THIN_MARGIN,
        "customer_budget": 600,
        "reasoning": "checking whether a discount closes this gap",
    })

    rows = rows_of_type(session, "margin_protection")
    check("exactly one margin_protection row written", len(rows) == 1, f"{len(rows)} rows")
    if not rows:
        return

    row = rows[0]["result"]
    check("row records the policy cap that would have applied",
          row["policy_cap_percent"] == 20, str(row["policy_cap_percent"]))
    check("row records the tighter margin ceiling",
          row["margin_ceiling_percent"] == 14, str(row["margin_ceiling_percent"]))
    check("row records the cost basis (merchant-side, not model-side)",
          row["cost_basis"] == 700, str(row["cost_basis"]))
    check("revenue_protected is the gap between the two prices",
          approx(row["revenue_protected"],
                 row["price_at_margin_ceiling"] - row["price_at_policy_cap"]),
          f"Rs.{row['revenue_protected']}")
    check("revenue_protected is positive", row["revenue_protected"] > 0,
          f"Rs.{row['revenue_protected']}")

    # A fat-margin item must NOT write one — the policy cap bound it, not margin.
    fat_session = new_session("audit-fat")
    agent.execute_tool(fat_session, "compute_discount", {
        "product_id": FAT_MARGIN, "customer_budget": 600,
        "reasoning": "same probe against an item with room to discount",
    })
    check("a policy-capped discount writes NO margin_protection row",
          len(rows_of_type(fat_session, "margin_protection")) == 0)


def c3_gate_blocks_the_model_bypass():
    header("C3. The model cannot route around the floor by skipping compute_discount")

    session = new_session("bypass")

    # The model asks the gate directly for the full policy-permitted 20%, never
    # having called compute_discount. Nothing in the prompt stops it. The gate does.
    result = agent.execute_tool(session, "check_gate", {
        "product_id": THIN_MARGIN,
        "quantity": 1,
        "discount_percent": 20,
        "reasoning": "customer pushed hard, going straight for the maximum discount",
    })

    check("gate DENIES the 20% discount", result["allowed"] is False)
    check("reason is below_margin_floor", result["reason"] == "below_margin_floor",
          str(result["reason"]))
    check("no approval was recorded for it",
          THIN_MARGIN not in session.pending_approvals,
          str(list(session.pending_approvals)))

    # create_order must then refuse, because there is no approval to spend.
    order = agent.execute_tool(session, "create_order", {
        "product_id": THIN_MARGIN, "quantity": 1,
        "reasoning": "trying to place the order anyway",
    })
    check("create_order refuses without an approval", order["success"] is False)
    check("...for the gate_not_checked reason", order["error_type"] == "gate_not_checked",
          str(order.get("error_type")))

    # The permitted discount goes through, and the charge clears the floor.
    ok = agent.execute_tool(session, "check_gate", {
        "product_id": THIN_MARGIN, "quantity": 1, "discount_percent": 14,
        "reasoning": "offering the deepest discount this item can actually take",
    })
    check("gate ALLOWS the margin-safe 14%", ok["allowed"] is True, str(ok.get("reason")))
    floor = discount.margin_floor_price(catalog.get_product_by_id(THIN_MARGIN)["cost"])
    check("the approved amount clears the margin floor",
          ok["amount_checked"] >= floor, f"Rs.{ok['amount_checked']} >= Rs.{floor}")


def c4_quantity_scales_the_floor():
    header("C4. The floor scales with quantity — it is a line-item rule, not a unit rule")

    session = new_session("qty")
    result = agent.execute_tool(session, "check_gate", {
        "product_id": THIN_MARGIN, "quantity": 3, "discount_percent": 20,
        "reasoning": "three units at the maximum policy discount",
    })
    check("3 units at 20% is still denied on margin", result["allowed"] is False)
    check("reason is below_margin_floor", result["reason"] == "below_margin_floor",
          str(result["reason"]))

    ok = agent.execute_tool(session, "check_gate", {
        "product_id": THIN_MARGIN, "quantity": 3, "discount_percent": 14,
        "reasoning": "three units at the margin-safe discount",
    })
    check("3 units at 14% is allowed", ok["allowed"] is True, str(ok.get("reason")))
    product = catalog.get_product_by_id(THIN_MARGIN)
    check("the amount checked is the 3-unit discounted total",
          approx(ok["amount_checked"], discount.apply_discount(product["price"] * 3, 14)),
          f"Rs.{ok['amount_checked']}")


# ---------------------------------------------------------------------------
# D. catalog.json — the data the whole feature rests on
# ---------------------------------------------------------------------------

def d1_every_product_has_a_plausible_cost():
    header("D1. Every product carries a plausible cost")

    products = catalog._load_catalog_raw()
    missing = [p["id"] for p in products if p.get("cost") is None]
    check("every product has a cost", not missing, f"missing on {missing[:5]}")

    nonpositive = [p["id"] for p in products if p.get("cost", 0) <= 0]
    check("every cost is positive", not nonpositive, str(nonpositive[:5]))

    not_below_price = [p["id"] for p in products if p.get("cost", 0) >= p["price"]]
    check("every cost is below its retail price", not not_below_price, str(not_below_price[:5]))

    # A product that can't take even a 1% discount without breaching its floor
    # would make the whole feature untestable in a demo.
    ratios = {p["id"]: p["cost"] / p["price"] for p in products}
    check("no product costs more than 90% of retail",
          max(ratios.values()) <= 0.90, f"max {max(ratios.values()):.0%}")

    binding = [pid for pid, r in ratios.items() if (1 - 1.10 * r) * 100 < 20]
    check("some products have margin as the binding constraint",
          len(binding) >= 5, f"{len(binding)} of {len(products)}")
    check("...and most do not, so both branches are demoable",
          len(binding) < len(products), f"{len(products) - len(binding)} policy-capped")

    # The two products the scenarios above are pinned to.
    thin = catalog.get_product_by_id(THIN_MARGIN)
    fat = catalog.get_product_by_id(FAT_MARGIN)
    check(f"{THIN_MARGIN} is still Rs.899 / cost Rs.700",
          thin["price"] == 899 and thin["cost"] == 700,
          f"Rs.{thin['price']} / Rs.{thin['cost']}")
    check(f"{FAT_MARGIN} is still Rs.1199 / cost Rs.525",
          fat["price"] == 1199 and fat["cost"] == 525,
          f"Rs.{fat['price']} / Rs.{fat['cost']}")


def main():
    print("=" * 70)
    print("  MARGIN-AWARE DISCOUNTING")
    print("=" * 70)
    print(f"  policy cap   : {gate.DEFAULT_MAX_DISCOUNT_PERCENT}% off retail")
    print(f"  margin floor : cost + {gate.DEFAULT_MIN_MARGIN_PERCENT}%")
    print(f"  run id       : {RUN_ID}")

    a1_back_compat_without_cost()
    a2_margin_floor_arithmetic()
    a3_the_two_ceilings_disagree()
    a4_capped_by_distinguishes_the_two_reasons()

    b1_gate_refuses_below_the_floor()
    b2_gate_boundary_is_exact()
    b3_margin_check_is_purely_additive()

    c1_compute_discount_is_margin_aware()
    c2_margin_protection_is_audited()
    c3_gate_blocks_the_model_bypass()
    c4_quantity_scales_the_floor()

    d1_every_product_has_a_plausible_cost()

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
