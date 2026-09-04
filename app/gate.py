"""
gate.py — the safety layer. Every action that touches money passes through here
BEFORE it's allowed to execute. This is plain Python logic, not a prompt instruction
to the LLM — the agent can *request* an action, but this file decides whether it's
actually allowed to happen.

Design principle: the LLM proposes, the gate disposes. Nothing in here trusts the
model's judgment about what's safe.
"""

# Default bounds — in a real product these might come from a merchant config or
# per-user settings. Hard-coded here so the logic is transparent and testable.
DEFAULT_MAX_SINGLE_ITEM = 6000       # no single item purchase above this without extra confirmation
DEFAULT_MAX_CART_TOTAL = 10000       # cumulative cart total per conversation
DEFAULT_MAX_DISCOUNT_PERCENT = 20    # ceiling for discount optimization feature

# The thinnest margin over cost a discounted sale may leave. This is a SECOND,
# independent ceiling on discounting, and it exists because the percentage cap
# above is blind to what an item actually cost us: 20% off a fat-margin tee is
# fine, 20% off a thin-margin supplement sells it for less than it is worth
# stocking.
#
# It lives here, next to the other bounds, rather than only in discount.py, for
# the reason this whole file exists: discount.py *computes*, and a computation
# can be skipped. The model can propose a discount without ever calling
# compute_discount, and the only thing standing between that proposal and a real
# charge is check_gate. A margin rule the gate does not enforce is not a rule.
DEFAULT_MIN_MARGIN_PERCENT = 10


def check_gate(amount, cart_total_so_far, item_in_stock, discount_percent=0,
                max_single_item=DEFAULT_MAX_SINGLE_ITEM,
                max_cart_total=DEFAULT_MAX_CART_TOTAL,
                max_discount_percent=DEFAULT_MAX_DISCOUNT_PERCENT,
                cost_basis=None,
                min_margin_percent=DEFAULT_MIN_MARGIN_PERCENT):
    """
    The core gate check. Called before ANY create_order or similar money-moving
    tool call is actually executed.

    Args:
        amount: price of the item being purchased right now
        cart_total_so_far: sum of everything already approved in this conversation
                            (this is what stops someone splitting a big purchase
                            into several small ones to sneak under the per-item limit)
        item_in_stock: bool, result of a prior check_stock() call
        discount_percent: discount being applied to this item, if any
        max_single_item / max_cart_total / max_discount_percent: override bounds,
            useful for testing different merchant configs
        cost_basis: what this line item cost the merchant, for the SAME quantity
            `amount` covers (i.e. unit cost x quantity). Optional: pass None and
            this check does not run, which is why every caller written before
            margins existed behaves exactly as it did.
        min_margin_percent: thinnest acceptable margin over cost_basis

    Returns:
        dict with:
            allowed: bool
            reason: short machine-readable code (only present if not allowed)
            explanation: plain-language reason, safe to show directly to the user
    """
    # --- input validation ---
    if not isinstance(amount, (int, float)) or amount < 0:
        return {
            "allowed": False,
            "reason": "invalid_amount",
            "explanation": "Invalid amount provided.",
        }
    if not isinstance(discount_percent, (int, float)) or discount_percent < 0:
        return {
            "allowed": False,
            "reason": "invalid_discount",
            "explanation": "Invalid discount percentage.",
        }
    # The exact charge, kept before the rounding below. The margin floor is
    # compared against this rather than the rounded figure: rounding 769.60 up
    # to 770 would let a sale through at a fraction under its floor. Every other
    # bound keeps using the rounded value, exactly as before.
    exact_amount = float(amount)

    # Ensure amount is clean integer paise
    amount = int(round(amount))

    if not item_in_stock:
        return {
            "allowed": False,
            "reason": "out_of_stock",
            "explanation": "This item is currently out of stock, so I can't complete this purchase.",
        }

    if amount > max_single_item:
        return {
            "allowed": False,
            "reason": "exceeds_single_item_limit",
            "explanation": (
                f"This item is priced at ₹{amount}, which is above the ₹{max_single_item} "
                f"limit I can approve on my own. This would need manual confirmation."
            ),
        }

    projected_total = cart_total_so_far + amount
    if projected_total > max_cart_total:
        return {
            "allowed": False,
            "reason": "exceeds_cart_total_limit",
            "explanation": (
                f"Adding this ₹{amount} item would bring your total to ₹{projected_total}, "
                f"which is above the ₹{max_cart_total} limit for this session. "
                f"You're currently at ₹{cart_total_so_far}."
            ),
        }

    if discount_percent > max_discount_percent:
        return {
            "allowed": False,
            "reason": "exceeds_discount_limit",
            "explanation": (
                f"A {discount_percent}% discount exceeds the maximum {max_discount_percent}% "
                f"I'm allowed to offer, so I can't apply that."
            ),
        }

    # --- margin floor: the last bound, and the only one that looks at cost ---
    #
    # Checked against `amount`, which is the DISCOUNTED total actually being
    # charged, against cost for the same quantity. Ordered last so that every
    # rejection precedence above is untouched.
    #
    # The explanation deliberately states no cost, no margin and no percentage.
    # This string is shown to the shopper, and what our supplier charges us is
    # not their business — "this is the best I can do" is both true and
    # sufficient.
    if cost_basis is not None:
        floor = round(float(cost_basis) * (1 + float(min_margin_percent) / 100.0), 2)
        if exact_amount < floor:
            return {
                "allowed": False,
                "reason": "below_margin_floor",
                "explanation": (
                    "I can't bring the price down that far on this particular item — "
                    "that's the best I'm able to do on it."
                ),
            }

    return {
        "allowed": True,
        "reason": None,
        "explanation": "Within all bounds — approved.",
    }


if __name__ == "__main__":
    # Manual sanity tests covering each rejection path
    print("=== Normal purchase, should be allowed ===")
    print(check_gate(amount=2799, cart_total_so_far=0, item_in_stock=True))

    print("\n=== Out of stock ===")
    print(check_gate(amount=2799, cart_total_so_far=0, item_in_stock=False))

    print("\n=== Single item over limit ===")
    print(check_gate(amount=7000, cart_total_so_far=0, item_in_stock=True))

    print("\n=== Cart total exceeded (the 'split into small purchases' exploit) ===")
    print(check_gate(amount=3000, cart_total_so_far=8500, item_in_stock=True))

    print("\n=== Discount too high ===")
    print(check_gate(amount=1000, cart_total_so_far=0, item_in_stock=True, discount_percent=35))

    print("\n=== Within the 20% cap but BELOW the margin floor ===")
    # A thin-margin supplement: Rs.899 retail, Rs.700 cost. 20% off is Rs.719.20,
    # which the discount cap permits and the margin floor (Rs.770) does not.
    print(check_gate(amount=719.20, cart_total_so_far=0, item_in_stock=True,
                     discount_percent=20, cost_basis=700))

    print("\n=== Same item at 14%, which clears the floor ===")
    print(check_gate(amount=773.14, cart_total_so_far=0, item_in_stock=True,
                     discount_percent=14, cost_basis=700))

    print("\n=== No cost_basis supplied -> margin check does not run (back-compat) ===")
    print(check_gate(amount=719.20, cart_total_so_far=0, item_in_stock=True, discount_percent=20))
