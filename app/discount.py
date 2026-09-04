"""
discount.py — budget-gap arithmetic for the negotiation feature.

The question this module answers is narrow and deliberately boring: "the customer
said ₹2500 and the item costs ₹2799 — what is the *smallest* discount that closes
that gap, and are we even allowed to offer it?"

Two design decisions worth defending:

  1. This module computes; it does not approve. It will happily tell you that a
     45% discount would be needed. Whether a discount may actually be granted is
     gate.py's call, and gate.py enforces the ceiling independently. Keeping the
     arithmetic and the authority in separate files means a bug in the maths can
     never widen the discount ceiling.

  2. The required discount is rounded UP to the next whole percent, not to the
     nearest. Rounding to the nearest would sometimes produce a discount that
     leaves the price a rupee or two *above* the stated budget — which fails the
     one job the function has. Rounding up costs the merchant at most 1% and
     always lands at or under budget.

  3. A discount is bounded by TWO independent ceilings, and which one bites
     matters to the customer. The price ceiling ("never more than 20% off") is a
     policy. The margin floor ("never sell below cost + 10%") is arithmetic about
     whether the sale is worth making at all. On this catalog they genuinely
     disagree: a ₹899 whey protein tub costs ₹700, so 20% off would sell it at
     ₹719 — below the ₹770 floor and barely above cost. The same 20% on a ₹1199
     training tee (cost ₹525) is nowhere near its floor.

     So the two are computed separately and the tighter one wins, and the result
     says WHICH one bound it. "I've hit the most I'm allowed to discount" and
     "any deeper and this sale stops being worth making" are different facts,
     and a merchant deserves to know which is which.

     Note the rounding asymmetry that falls out of this: the required discount
     rounds UP (design note 2), but the margin ceiling rounds DOWN. Rounding the
     ceiling up by even one percent would step straight through the floor it
     exists to protect.
"""

import math
from decimal import Decimal, ROUND_HALF_UP

# Mirrors gate.DEFAULT_MAX_DISCOUNT_PERCENT. Passed in explicitly by callers so
# this module never has to import the gate — see design note 1 above.
DEFAULT_MAX_DISCOUNT_PERCENT = 20

# How far above a stated budget an item can sit before we stop treating it as a
# "close match worth negotiating on". 20% mirrors the discount ceiling: beyond
# that, no permitted discount could close the gap anyway.
NEGOTIABLE_GAP_PERCENT = 20

# The thinnest margin over cost this merchant will accept on a discounted sale.
# 10% is a judgement call, and it is the number that stops the agent from
# discounting a low-margin item into a sale that loses money once the cost of
# actually fulfilling it is counted.
DEFAULT_MIN_MARGIN_PERCENT = 10

# Reason codes for which ceiling bound a discount. Returned rather than a bare
# boolean because the agent says something different to the customer in each
# case — see the DISCOUNTS section of agent.py's system prompt.
CAPPED_BY_DISCOUNT_CEILING = "discount_cap"
CAPPED_BY_MARGIN_FLOOR = "margin_floor"


def apply_discount(item_price, discount_percent):
    """Return the payable price after a discount, to paise precision.

    Decimal rather than float, for the same reason rupees_to_paise uses it: this
    number becomes a real charge.
    """
    price = Decimal(str(item_price))
    factor = Decimal(1) - (Decimal(str(discount_percent)) / Decimal(100))
    payable = (price * factor).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
    return float(payable)


def _whole(value):
    """Narrow a whole-numbered float to int, so 20.0 renders as "20%" not "20.0%".

    Both discount ceilings are whole percents by construction, and this value
    ends up in prompt text, audit rows and JSON — where 20.0 reads as a bug.
    """
    number = float(value)
    return int(number) if number.is_integer() else number


def margin_floor_price(cost, min_margin_percent=DEFAULT_MIN_MARGIN_PERCENT):
    """The lowest price this item can be sold at and still be worth selling.

    Decimal for the same reason apply_discount uses it: this number decides
    whether a real charge is allowed to happen.
    """
    floor = Decimal(str(cost)) * (Decimal(1) + Decimal(str(min_margin_percent)) / Decimal(100))
    return float(floor.quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def max_discount_for_margin(item_price, cost, min_margin_percent=DEFAULT_MIN_MARGIN_PERCENT):
    """The deepest whole-percent discount that still clears the margin floor.

    Rounded DOWN, deliberately — see design note 3. A 16.4% ceiling becomes 16%,
    not 17%, because 17% would sell below the floor this function exists to
    defend.

    Returns 0.0 when the item is already at or under its floor at full price,
    which is the honest answer: there is no room to discount it at all.
    """
    price = float(item_price)
    floor = margin_floor_price(cost, min_margin_percent)
    if floor >= price:
        return 0.0
    return float(math.floor((1 - floor / price) * 100))


def effective_discount_ceiling(item_price, cost=None,
                               max_discount_percent=DEFAULT_MAX_DISCOUNT_PERCENT,
                               min_margin_percent=DEFAULT_MIN_MARGIN_PERCENT):
    """The real ceiling on a discount: the tighter of policy and profitability.

    Returns (ceiling_percent, limiting_constraint). limiting_constraint is
    CAPPED_BY_MARGIN_FLOOR only when margin is strictly tighter than policy — a
    tie is reported as the policy cap, because that is the rule that would still
    apply if the item's cost changed.

    With cost=None this is just the policy cap, which is what keeps every
    caller written before margins existed behaving exactly as it did.
    """
    if cost is None:
        return _whole(max_discount_percent), CAPPED_BY_DISCOUNT_CEILING

    margin_ceiling = max_discount_for_margin(item_price, cost, min_margin_percent)
    if margin_ceiling < max_discount_percent:
        return _whole(margin_ceiling), CAPPED_BY_MARGIN_FLOOR
    return _whole(max_discount_percent), CAPPED_BY_DISCOUNT_CEILING


def compute_minimum_discount(customer_budget, item_price,
                             max_discount_percent=DEFAULT_MAX_DISCOUNT_PERCENT,
                             cost=None,
                             min_margin_percent=DEFAULT_MIN_MARGIN_PERCENT):
    """Smallest whole-percent discount that brings item_price to or under budget.

    Args:
        customer_budget: what the customer said they want to spend, in rupees
        item_price: sticker price of the item, in rupees
        max_discount_percent: policy ceiling we are permitted to offer
        cost: what this item cost the merchant, in rupees. Optional — omit it
            and the result is bit-for-bit what it was before margins existed.
            Supply it and the discount is additionally bounded so the sale never
            drops below cost + min_margin_percent.
        min_margin_percent: the thinnest margin over cost worth selling at

    Returns:
        dict with:
            discount_percent: whole percent to offer (0 if none needed)
            discounted_price: what the customer would actually pay
            sufficient: True if discounted_price <= customer_budget
            required_percent: the exact (unrounded, uncapped) discount the gap
                needs — useful in the audit log for showing how far off we were
            gap: rupees still outstanding after the discount (0 when sufficient)
            limiting_constraint: which ceiling was in force — "discount_cap" or
                "margin_floor". Present whether or not it actually bit.
            capped_by: the constraint that ACTUALLY stopped us, or None when the
                budget was met. This is the field to branch on.
            max_discount_allowed_percent: the effective ceiling that applied
            margin_floor_price / cost: present only when cost was supplied
            explanation: plain-language summary, safe to show the customer
    """
    budget = float(customer_budget)
    price = float(item_price)

    if price <= 0:
        raise ValueError(f"item_price must be positive, got {item_price}")
    if budget <= 0:
        raise ValueError(f"customer_budget must be positive, got {customer_budget}")

    ceiling, limiting_constraint = effective_discount_ceiling(
        price, cost=cost,
        max_discount_percent=max_discount_percent,
        min_margin_percent=min_margin_percent,
    )

    # Margin facts travel with every result so the audit row can show the
    # merchant's reasoning, even on the calls where margin changed nothing.
    margin_fields = {
        "limiting_constraint": limiting_constraint,
        "max_discount_allowed_percent": ceiling,
    }
    if cost is not None:
        margin_fields["cost"] = float(cost)
        margin_fields["margin_floor_price"] = margin_floor_price(cost, min_margin_percent)

    # Already affordable — don't discount something that doesn't need it.
    if price <= budget:
        return {
            "discount_percent": 0,
            "discounted_price": price,
            "sufficient": True,
            "required_percent": 0.0,
            "gap": 0.0,
            "capped_by": None,
            **margin_fields,
            "explanation": f"₹{price:g} is already within the ₹{budget:g} budget — no discount needed.",
        }

    required_percent = (1 - budget / price) * 100

    # Round UP to the next whole percent so the result genuinely lands at or
    # under budget (see design note 2 in the module docstring).
    minimum_percent = math.ceil(required_percent)

    if minimum_percent <= ceiling:
        discounted_price = apply_discount(price, minimum_percent)
        return {
            "discount_percent": minimum_percent,
            "discounted_price": discounted_price,
            "sufficient": True,
            "required_percent": round(required_percent, 2),
            "gap": 0.0,
            "capped_by": None,
            **margin_fields,
            "explanation": (
                f"A {minimum_percent}% discount brings ₹{price:g} down to ₹{discounted_price:g}, "
                f"which fits the ₹{budget:g} budget."
            ),
        }

    # The gap is wider than we're allowed to close. Return the best permitted
    # offer anyway — a partial discount plus an honest shortfall is more useful
    # to the customer than a flat "no".
    best_price = apply_discount(price, ceiling)
    gap = round(best_price - budget, 2)

    if limiting_constraint == CAPPED_BY_MARGIN_FLOOR:
        # State the merchant-side fact plainly here. agent.py's system prompt is
        # what stops this reaching the customer as a raw cost figure — this
        # string is for the log and for the model to reason from, and the model
        # is told to translate it into "that's the best I can do on this one".
        explanation = (
            f"Closing this gap would take a {required_percent:.1f}% discount. The policy "
            f"ceiling is {max_discount_percent:g}%, but this item cannot go past "
            f"{ceiling:g}% without selling below its ₹{margin_fields['margin_floor_price']:g} "
            f"margin floor — so {ceiling:g}% is the real limit. At {ceiling:g}% the price is "
            f"₹{best_price:g}, still ₹{gap:g} over the ₹{budget:g} budget."
        )
    else:
        explanation = (
            f"Closing this gap would take a {required_percent:.1f}% discount, but "
            f"{ceiling:g}% is the most that can be offered. Even at "
            f"{ceiling:g}% the price is ₹{best_price:g}, still ₹{gap:g} over "
            f"the ₹{budget:g} budget."
        )

    return {
        "discount_percent": ceiling,
        "discounted_price": best_price,
        "sufficient": False,
        "required_percent": round(required_percent, 2),
        "gap": gap,
        "capped_by": limiting_constraint,
        **margin_fields,
        "explanation": explanation,
    }


def is_negotiable_gap(customer_budget, item_price, threshold_percent=NEGOTIABLE_GAP_PERCENT):
    """True if the item is above budget but close enough to be worth negotiating on.

    Used by the budget-aware negotiation flow to decide whether an
    slightly-over-budget search result is worth surfacing to the customer at all.
    """
    budget = float(customer_budget)
    price = float(item_price)
    if price <= budget:
        return False
    over_by_percent = (price - budget) / price * 100
    return over_by_percent <= threshold_percent


if __name__ == "__main__":
    cases = [
        ("already affordable",        3000, 2799),
        ("small gap, 1% closes it",   2780, 2799),
        ("realistic gap",             2500, 2799),
        ("exactly at the 20% cap",    2240, 2799),  # needs 19.97% -> rounds up to 20%
        ("gap too wide to close",     1500, 2799),
    ]
    for label, budget, price in cases:
        result = compute_minimum_discount(budget, price)
        flag = "OK " if result["sufficient"] else "CAP"
        print(f"[{flag}] {label:28} budget ₹{budget} vs ₹{price}")
        print(f"       needs {result['required_percent']}% -> offering {result['discount_percent']}% "
              f"-> pay ₹{result['discounted_price']}")
        print(f"       {result['explanation']}\n")

    print("negotiable-gap check (is this worth offering at all?):")
    for budget, price in ((2500, 2799), (1500, 2799), (3000, 2799)):
        print(f"  budget ₹{budget} vs ₹{price}: {is_negotiable_gap(budget, price)}")
