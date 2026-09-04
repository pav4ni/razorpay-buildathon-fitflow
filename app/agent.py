"""
agent.py — the conversational checkout agent: an LLM with tool use, behind a hard
safety gate and an audit trail wrapped around every money-moving action.

The architecture in one paragraph:

    The model decides *what* to do and says why. This file decides whether that's
    allowed and actually does it. Nothing the model says is trusted as fact —
    prices come from the catalog, stock comes from the catalog, the running cart
    total comes from session state, and permission comes from gate.py. The model
    supplies intent and language; the Python layer supplies truth.

Enforcement points that live at the tool-execution layer, not in the prompt,
because a prompt is a request and code is a rule:

  1. PRICES ARE LOOKED UP, NEVER QUOTED. The model passes a product_id and a
     quantity. It cannot pass an amount, so it cannot understate a price to slip
     an item past the gate.
  2. CART TOTAL IS SERVER-SIDE. check_gate needs the cumulative session total to
     catch the "split one big purchase into several small ones" exploit. That
     number is read from ShoppingSession, not from the model's message.
  3. NO ORDER WITHOUT A FRESH GATE APPROVAL. create_order refuses unless this
     exact product, quantity and discount was approved by check_gate earlier in
     this session, and each approval is consumed on use.
  4. DISCOUNTS ARE PROPOSED BY THE MODEL, PRICED BY US, AND CAPPED BY THE GATE.
     compute_discount only does arithmetic. The percentage it suggests still has
     to survive check_gate's 20% ceiling, and the amount finally charged is
     recomputed here from the catalog price and the *approved* percentage —
     never from anything the model asserts.
  5. ONE UPSELL PER SESSION, COUNTED IN CODE. The complementary-product lookup
     is refused after the first offer, and offered/accepted/declined are written
     by Python, so the attach-rate metric can't be inflated by the model.

Every tool call is written to audit.py before its result goes back to the model,
including the model's own stated reasoning for the call.
"""

import hashlib
import json
import os
import sys
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import anthropic

import audit
import catalog
import discount
import gate
import preferences
import razorpay_client
import watches

MODEL = "claude-sonnet-4-6"
MAX_TOKENS = 16000

# Safety valve on the tool loop itself. A model stuck in a search→search→search
# cycle would otherwise burn tokens indefinitely.
MAX_TOOL_ITERATIONS = 12

# Below this blended match score, a price-filtered search counts as "nothing good
# found" at all — a genuine dead end rather than a budget problem.
WEAK_MATCH_THRESHOLD = 0.35
#
# That absolute floor is necessary but nowhere near sufficient, and the reason is
# worth stating because it is the non-obvious part of this feature. Embeddings
# score *topic*, not product type: "ProGrip Insole Support Pads" scores 0.61
# against "running shoes", and "TrailBlazer Trekking Socks" scores 0.70 against
# "trekking shoes". So a budget that excludes every actual shoe still returns a
# confident-looking 0.6 accessory, sails past any absolute threshold, and the
# customer is shown insoles when they asked for shoes.
#
# The reliable signal is relative, not absolute: run the search again without the
# price ceiling and ask whether the ceiling is what's costing them the better
# match. Measured across the catalog, that margin is a clean separator — 0.000
# when the budget is genuinely fine (the same product wins either way), and
# +0.06 to +0.21 when the budget is the thing standing in the way.
RELATIVE_MATCH_MARGIN = 0.05

# There is no authentication in this build. A real deployment would resolve the
# shopper from a logged-in session or a checkout form; here one demo identity is
# hardcoded so the Customers API and cross-session purchase history can be
# exercised end to end.
# LIMITATION, stated plainly: every session is the same person. Purchase history
# is therefore shared across all demo runs, which is exactly why it's useful for
# a demo and exactly why it is not production behaviour.
DEMO_CUSTOMER = {
    "name": "Demo User",
    "email": "demo.user@example.com",
    "contact": "+919876543210",
}


SYSTEM_PROMPT = """You are a shopping assistant for an Indian fitness and athleisure store. You help customers find products and complete purchases, and all prices are in Indian Rupees (₹).

HOW TO HANDLE A REQUEST

If the customer's request is vague — for example "I need something for the gym", "I want to get fit", "show me some shoes" — ask exactly ONE short clarifying question before searching. Pick the single question that most narrows the search: usually budget, or the specific activity, or who it's for. Ask one question, not a list. Then search once you have the answer.

If the request is already specific enough to search on (a product type, or a product type plus a budget or use case), search immediately. Do not ask a clarifying question just to be thorough — that wastes the customer's time.

When you present results, describe two or three of the best matches in a sentence or two each, including the price and why it fits what they asked for. Do not dump the whole result list.

WHEN THE BUDGET IS IN THE WAY

If search_catalog comes back with a `budget_negotiation` block, the customer's price ceiling is costing them the match they actually asked for. The search was automatically re-run without the ceiling, and the block contains the closest option above it. Read the block's `guidance` field — it tells you which of these two situations you're in.

If `has_in_budget_options` is false, nothing decent was found under their budget. Don't say "nothing found" and stop. Say honestly that nothing fits the budget exactly, name the closest item and its price, and offer them the choice: you can check whether a discount closes the gap, or you can show other options.

If `has_in_budget_options` is true, there are real options under their budget — lead with the best of those, because that's what they asked for. Then mention that the closest match to their description sits just above budget, name it and its price, and offer to check whether a discount bridges the gap.

Either way: if the block says the gap is negotiable, say so. If it says the gap is too wide, be straight about that instead of raising hopes you can't deliver on.

DISCOUNTS

When a customer's stated budget is close to but under an item's price, use compute_discount to find the smallest discount that would close the gap. Only do this when there is an actual budget to work against — never volunteer a discount to someone who hasn't mentioned price.

compute_discount only calculates; it does not approve anything. If it says `sufficient` is true, offer that exact percentage to the customer. If it says `sufficient` is false, you have run out of room — tell them the best you can do and what it would still cost them, and offer to find something cheaper instead.

When `sufficient` is false, read `capped_by`. It says WHY you ran out of room, and the two reasons call for different words:

- `discount_cap` means you reached the store's flat discount ceiling. You can name that limit plainly: "20% is the most I can take off."

- `margin_floor` means this particular item cannot be discounted that deep and still be worth the store selling it. Say only that you have reached your limit on this item — "that's the best I can do on this one", or "I can't go any lower on this particular product". Then offer a cheaper alternative that would actually fit their budget.

  On a `margin_floor` cap, never mention cost, margin, profit, wholesale, what the store paid, or any reason the limit sits where it does. What a shop pays its suppliers is confidential and is not the customer's business. "That's the best I can do on this one" is true, sufficient, and the whole of what you should say. Do not imply the item is low quality or being discontinued either — invent no reason at all.

To actually sell at a discount: pass the agreed `discount_percent` to check_gate. If the gate approves it, create_order charges the discounted price automatically — you do not pass a price anywhere. Never promise a discount you have not put through check_gate.

BUYING SOMETHING

You must call check_gate before you call create_order. Every single time, with no exceptions. check_gate is the store's spending-safety check: it verifies stock, the per-item limit, the cumulative session limit, and any discount ceiling. create_order will refuse to run if the gate has not approved that exact item.

If check_gate returns allowed = false, tell the customer what happened using the plain-language text in the `explanation` field. Never just say "I can't do that." Explain the actual reason, then offer a genuinely useful alternative — a cheaper item in the same category, or splitting the purchase across sessions, or removing something from the cart. The customer should understand the limit, not feel stonewalled.

Only call create_order after the customer has clearly confirmed they want to buy that specific item. Do not buy anything on a maybe.

When an order is created successfully, give the customer the order id and the amount, and tell them the payment link will be sent to complete the payment — you create the order, they complete the payment through Razorpay.

SUGGESTING ONE MORE THING

After an order succeeds — and only then — you may make ONE product suggestion for the whole conversation. Call get_complementary_products for the item they just bought and offer the single best fit in one short sentence, with its price: "Want to add the AirFlex socks for ₹349?"

One offer per conversation, total. The tool will refuse a second lookup, and that refusal is correct — do not work around it, and do not re-raise a suggestion the customer has already passed on. If they say no, accept it in a few words and move on. A suggestion that annoys the customer costs more than it earns.

IF THEY'D RATHER WAIT FOR A BETTER PRICE

Use create_price_watch when the customer would rather be told about a price drop than buy now. Two moments call for it:

- They say something like "let me know if it gets cheaper" or "tell me when it goes on sale". Take them at their word and set the watch.
- A budget conversation has run out of road: nothing fits their budget, and either the discount isn't enough or they don't want one. Before you let that conversation end with nothing, offer the watch in one sentence: "Want me to keep an eye on it and let you know if it drops below ₹2500?"

Pass target_price when they name a number they'd buy at. Leave it out when they just want to know about any drop. Offer a watch once per item — if they say no, drop it.

Be straight about what a watch is: the store records it and checks prices, and in this build a triggered watch is logged rather than emailed. Don't promise them an email or a text.

RETURNS AND REFUNDS

If a customer wants to cancel or return something — "actually cancel that", "I want to return the shoes", "refund my order" — use refund_order. It finds their order and refunds it. For a partial refund, pass the rupee amount; leave it out for a full refund.

If no matching order is found, say so plainly and ask which order they mean. If the order exists but hasn't been paid for yet, explain that there's nothing to refund because the payment was never completed, and offer to help another way. Never claim a refund succeeded unless the tool says it did.

RETURNING CUSTOMERS

You can call get_past_orders to see what this customer has bought before. It's a nice touch when it's relevant — someone restocking supplements, or asking "what did I order last time?". Use it when it genuinely helps them. Don't open every conversation by reciting their purchase history.

WHEN THINGS FAIL

If a tool returns an error — payment gateway unreachable, item out of stock, product not found — say so plainly and calmly, explain what it means for their purchase, and suggest a next step. Never pretend an order succeeded when it did not.

When a failed result includes a `user_message` field, base your reply on that. The `error` field next to it is internal detail for the store's engineers: never repeat API credentials, configuration problems, stack traces or internal system names to the customer. "Our payment system is temporarily unavailable" is what they need to hear; the rest is our problem to fix.

REASONING

Every tool takes a `reasoning` parameter. Write one honest sentence explaining why you are making that specific call right now. This is recorded in the store's audit log and is read by humans, so make it useful: "checking stock before proposing this item to the customer" is good, "calling a tool" is not.

Keep your replies short and conversational. You are a helpful shop assistant, not a brochure."""


# ---------------------------------------------------------------------------
# Tool schemas
#
# Note what is deliberately ABSENT from these schemas: there is no `amount`
# parameter and no `cart_total_so_far` parameter anywhere. Those values are
# resolved from the catalog and from session state at execution time. If the
# model could supply them, the gate would be checking numbers the model chose.
# ---------------------------------------------------------------------------

REASONING_PARAM = {
    "type": "string",
    "description": "One sentence: why you are making this call right now. Recorded in the audit log.",
}

TOOLS = [
    {
        "name": "search_catalog",
        "description": (
            "Semantic search over the store's product catalog. Use a natural language "
            "description of what the customer wants. Returns matching products with their "
            "id, name, price in rupees, stock level, rating and description. "
            "If you pass max_price and nothing good is found under it, the response also "
            "includes a `budget_negotiation` block with the closest options above budget."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language description of the product, e.g. 'lightweight running shoes for women'.",
                },
                "max_price": {
                    "type": "number",
                    "description": "Optional maximum price in rupees. Use this when the customer states a budget.",
                },
                "category": {
                    "type": "string",
                    "enum": ["footwear", "apparel", "gear", "supplements", "accessories", "electronics"],
                    "description": "Optional category filter. Only use it when the customer is clearly asking within one category.",
                },
                "top_k": {
                    "type": "integer",
                    "description": "How many results to return. Defaults to 5.",
                },
                "reasoning": REASONING_PARAM,
            },
            "required": ["query", "reasoning"],
        },
    },
    {
        "name": "check_stock",
        "description": (
            "Check whether a product has enough stock for the quantity the customer wants. "
            "Call this before proposing a specific item for purchase."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product id, e.g. 'P001'."},
                "quantity": {"type": "integer", "description": "Units the customer wants. Defaults to 1."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "reasoning"],
        },
    },
    {
        "name": "compute_discount",
        "description": (
            "Work out the smallest discount that would bring an item within the customer's "
            "stated budget. Use this when they've named a budget that's a bit under the price. "
            "This only calculates — it does not approve or apply anything. To actually sell at "
            "the discount, pass the returned discount_percent to check_gate, which enforces the "
            "store's ceiling."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product the customer is interested in."},
                "customer_budget": {
                    "type": "number",
                    "description": "The budget the customer actually stated, in rupees. Use their number, not an estimate.",
                },
                "quantity": {"type": "integer", "description": "Units they want. Defaults to 1."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "customer_budget", "reasoning"],
        },
    },
    {
        "name": "check_gate",
        "description": (
            "The store's spending-safety check. You MUST call this and get allowed = true "
            "before calling create_order — create_order will refuse otherwise. "
            "It checks stock, the per-item spending limit, the cumulative limit for this "
            "conversation, and the discount ceiling. "
            "You do not pass the price or the cart total: the store looks both up itself. "
            "If it returns allowed = false, relay the `explanation` field to the customer in your own words."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product id the customer wants to buy."},
                "quantity": {"type": "integer", "description": "Units to buy. Defaults to 1."},
                "discount_percent": {
                    "type": "number",
                    "description": (
                        "Only set this if a discount has actually been agreed with the customer, "
                        "normally one that compute_discount recommended. Otherwise leave it at 0. "
                        "Anything above the store's ceiling will be refused."
                    ),
                },
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "reasoning"],
        },
    },
    {
        "name": "create_order",
        "description": (
            "Create a real Razorpay order for a product the customer has confirmed they want. "
            "This is the money-moving action. It requires a prior check_gate approval for this "
            "exact product and quantity in this conversation, and each approval can only be used once. "
            "The amount charged is computed from the catalog price and whatever discount the gate "
            "approved — you never pass a price. "
            "Returns the order details on success, or an error you must explain to the customer on failure."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product id to purchase."},
                "quantity": {"type": "integer", "description": "Units to buy. Defaults to 1. Must match what the gate approved."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "reasoning"],
        },
    },
    {
        "name": "get_complementary_products",
        "description": (
            "Get products that pair naturally with something the customer just bought, so you can "
            "make ONE suggestion. This works once per conversation — a second call is refused. "
            "Use your one call after a successful order, on the item they actually bought."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product to find companions for — normally the one just purchased."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "reasoning"],
        },
    },
    {
        "name": "refund_order",
        "description": (
            "Refund an order the customer wants to cancel or return. Identify the order by "
            "product_id or order_id — the store looks up their order history itself. "
            "Leave amount_in_rupees out for a full refund; set it for a partial one. "
            "Returns the refund details on success, or an explained failure if there's no "
            "matching order or the order was never paid for."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product the customer wants to return, e.g. 'P001'."},
                "order_id": {"type": "string", "description": "Razorpay order id, if the customer gave you one."},
                "amount_in_rupees": {
                    "type": "number",
                    "description": "Partial refund amount. Omit entirely for a full refund.",
                },
                "reason": {"type": "string", "description": "Why the customer is returning it, in their words."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["reasoning"],
        },
    },
    {
        "name": "create_price_watch",
        "description": (
            "Record a price-drop watch on a product, for a customer who'd rather wait than "
            "buy now — 'tell me if it gets cheaper', or a budget conversation that ended with "
            "nothing affordable. Set target_price to the price they'd buy at; leave it out to "
            "watch for any drop from today's price. The store checks watches and logs a "
            "notification when one trips — do not promise the customer an email or SMS."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "product_id": {"type": "string", "description": "Product to watch, e.g. 'P001'."},
                "target_price": {
                    "type": "number",
                    "description": (
                        "The price in rupees the customer would buy at, if they named one. "
                        "Omit it entirely to watch for any drop below the current price."
                    ),
                },
                "reasoning": REASONING_PARAM,
            },
            "required": ["product_id", "reasoning"],
        },
    },
    {
        "name": "get_past_orders",
        "description": (
            "Look up what this customer has bought before, newest first. Useful for reorders "
            "and for answering 'what did I order last time?'. Reads the store's own records."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "How many past orders to return. Defaults to 5."},
                "reasoning": REASONING_PARAM,
            },
            "required": ["reasoning"],
        },
    },
]


class ShoppingSession:
    """One conversation. Holds the message history and — critically — the state
    the gate and the metrics need, which is kept here rather than anywhere the
    model can reach."""

    def __init__(self, session_id=None):
        self.session_id = session_id or f"sess-{uuid.uuid4().hex[:12]}"
        self.messages = []

        # Cumulative rupee value of orders successfully created in this session.
        # This is what makes the "split it into small purchases" exploit fail.
        self.cart_total_so_far = 0

        # Gate approvals awaiting use:
        #   product_id -> {"amount", "quantity", "discount_percent", "sticker_amount"}
        # Popped when consumed by create_order, so one approval buys one order.
        self.pending_approvals = {}

        # Orders actually created, for the CLI summary and refund lookups.
        self.orders = []

        # Razorpay Customer, attached by link_customer().
        self.customer = None
        self.customer_id = None

        # Upsell state. `offered_ids` are the products we put in front of the
        # customer; `resolved` becomes "accepted" or "declined" exactly once.
        # Enforced here rather than in the prompt so attach rate is a real metric.
        self.upsell = {"offered": False, "offered_ids": [], "resolved": None}

        # Orders already refunded in this session, so a double refund is refused.
        self.refunded_order_ids = set()

        # The user message currently being handled, so audit rows can record the
        # utterance that triggered each action.
        self.current_user_query = None


# ---------------------------------------------------------------------------
# Tool execution
#
# Each handler returns a plain dict that becomes the tool_result content. Every
# one of them is audited by execute_tool below.
# ---------------------------------------------------------------------------

def _line_amount(product, quantity):
    """Sticker price for a line item: catalog price x quantity, no discount."""
    return product["price"] * quantity


def _cost_basis(product, quantity):
    """What this line item cost the merchant, or None if the catalog has no cost.

    Returning None rather than 0 matters: 0 would read as "free to us", which
    would make every margin check trivially pass. None means "unknown", and both
    gate.check_gate and discount.compute_minimum_discount treat unknown as
    "don't run the margin check" rather than "the margin is fine".
    """
    cost = product.get("cost")
    if cost is None:
        return None
    return cost * quantity


def _log_margin_protection(session, product, quantity, analysis):
    """Record a discount the margin floor refused to go past, and what it saved.

    `revenue_protected` is the difference between what the item would have sold
    for at the plain 20% policy cap and what the margin floor actually permitted.
    That is the number the revenue-impact report sums, and it is computed here —
    at the moment the decision is made, from the values that drove it — rather
    than reconstructed later from prices that may since have changed.
    """
    sticker = _line_amount(product, quantity)
    price_at_policy_cap = discount.apply_discount(sticker, gate.DEFAULT_MAX_DISCOUNT_PERCENT)
    price_at_margin_ceiling = analysis["discounted_price"]

    return _log_side_event(
        session,
        action_type="margin_protection",
        action_params={
            "product_id": product["id"],
            "product_name": product["name"],
            "quantity": quantity,
            "sticker_price": sticker,
        },
        result={
            "capped_by": analysis["capped_by"],
            "policy_cap_percent": gate.DEFAULT_MAX_DISCOUNT_PERCENT,
            "margin_ceiling_percent": analysis["discount_percent"],
            "cost_basis": analysis.get("cost"),
            "margin_floor_price": analysis.get("margin_floor_price"),
            "price_at_policy_cap": price_at_policy_cap,
            "price_at_margin_ceiling": price_at_margin_ceiling,
            "revenue_protected": round(price_at_margin_ceiling - price_at_policy_cap, 2),
            "rule": (
                f"Never sell below cost + {gate.DEFAULT_MIN_MARGIN_PERCENT}%. The "
                f"{gate.DEFAULT_MAX_DISCOUNT_PERCENT}% policy cap would have allowed a deeper "
                f"discount than this item can profitably take."
            ),
        },
        reasoning=(
            f"A discount on {product['name']} was capped at {analysis['discount_percent']}% "
            f"by the margin floor rather than by the {gate.DEFAULT_MAX_DISCOUNT_PERCENT}% "
            f"policy ceiling."
        ),
    )


def _payable_amount(product, quantity, discount_percent=0):
    """What the customer actually pays after an approved discount.

    Computed here from the catalog price and the approved percentage — the two
    values the model cannot forge — so the charge always matches what the gate
    signed off on.
    """
    sticker = _line_amount(product, quantity)
    if not discount_percent:
        return round(float(sticker), 2)
    return round(float(discount.apply_discount(sticker, discount_percent)), 2)


def _log_side_event(session, action_type, action_params, result, reasoning, gate_decision=None):
    """Write an extra audit row for a business event that isn't itself a tool call.

    Used for upsell accepted/declined: those are consequences of other actions,
    but they're the events the attach-rate metric counts, so they get their own
    rows rather than being buried inside an order result.
    """
    return audit.log_event(
        session_id=session.session_id,
        user_query=session.current_user_query,
        agent_reasoning=reasoning,
        action_type=action_type,
        action_params=action_params,
        result=result,
        gate_decision=gate_decision,
        customer_id=session.customer_id,
    )


def _slim_product(product, include_score=False):
    slim = {
        "id": product["id"],
        "name": product["name"],
        "category": product["category"],
        "price": product["price"],
        "stock": product["stock"],
        "rating": product["rating"],
        "num_reviews": product["num_reviews"],
        "description": product["description"],
    }
    if include_score and "match_score" in product:
        slim["match_score"] = product["match_score"]
        if product.get("preference_boost"):
            slim["preference_boost"] = product["preference_boost"]
            slim["preference_matched"] = product.get("preference_matched", [])
    return slim


def _negotiation_reason(results, unfiltered, max_price):
    """Should this price-filtered search carry a negotiation block, and why?

    Three triggers, in descending order of severity. The reason code is returned
    rather than a bare bool because the three cases need genuinely different
    things said to the customer — "there's nothing here" and "there are options,
    but the one you actually described costs a bit more" are not the same
    conversation.

    Returns a reason code, or None when the budget is doing no harm.
    """
    if not results:
        return "no_matches_under_budget"

    top_score = results[0]["match_score"]
    if top_score < WEAK_MATCH_THRESHOLD:
        return "only_weak_matches_under_budget"

    # The relative test — see RELATIVE_MATCH_MARGIN. Only the top unfiltered hit
    # matters: if the single best answer to the query is affordable, the budget
    # isn't the problem, however many pricier things also exist.
    if unfiltered:
        best = unfiltered[0]
        if best["price"] > max_price and best["match_score"] - top_score >= RELATIVE_MATCH_MARGIN:
            return "better_match_above_budget"

    return None


def _preference_boost_for(session):
    """The preference nudge to apply to this session's searches, if any.

    Wrapped in a try/except on purpose: preference memory is a nicety, and a
    problem reading it must never be the reason a customer can't search. Failing
    to a plain unboosted search is the correct degradation.
    """
    try:
        return preferences.build_boost_map(session.customer_id)
    except Exception:
        return {}


def _handle_search_catalog(session, params):
    """Catalog search, with the budget-aware negotiation fallback built in.

    The fallback is deliberately automatic rather than left to the model: when a
    price ceiling is costing the customer the match they actually asked for,
    re-running the search without it is always the right next move, so it happens
    here in one round trip instead of relying on the model to notice and retry.
    """
    query = params["query"]
    max_price = params.get("max_price")
    category = params.get("category")
    top_k = params.get("top_k", 5)

    # Preference memory, if this customer has earned any. The same boost map is
    # used for the unfiltered re-run below, so the two score sets stay
    # comparable — the negotiation test compares them directly.
    boost_map = _preference_boost_for(session)

    results = catalog.search_catalog(
        query=query, max_price=max_price, category=category, top_k=top_k,
        preference_boost=boost_map,
    )

    response = {
        "count": len(results),
        "products": [_slim_product(p, include_score=True) for p in results],
    }

    if boost_map:
        # Surfaced in the tool result (and therefore in the audit row) so a
        # nudged ranking can always be accounted for after the fact.
        response["personalization"] = {
            "applied": True,
            "signals": boost_map,
            "explanation": preferences.describe(session.customer_id),
            "note": (
                "Small tie-breaker boost from this customer's past purchases. "
                "It cannot outrank a better match — don't mention it unless asked."
            ),
        }

    if max_price is None:
        return response

    # Re-run without the ceiling. This is the comparison the relative test needs,
    # and it's cheap — 35 products and a cached embedding matrix.
    unfiltered = catalog.search_catalog(
        query=query, max_price=None, category=category, top_k=top_k,
        preference_boost=boost_map,
    )

    reason = _negotiation_reason(results, unfiltered, max_price)
    if reason is None:
        return response

    above_budget = [p for p in unfiltered if p["price"] > max_price]
    # "There are options under budget" changes the advice, so state it explicitly
    # rather than making the model infer it from the length of a list.
    has_in_budget_options = reason == "better_match_above_budget"

    negotiation = {
        "triggered": True,
        "stated_budget": max_price,
        "reason": reason,
        "has_in_budget_options": has_in_budget_options,
        "alternatives_above_budget": [_slim_product(p, include_score=True) for p in above_budget[:3]],
    }

    if above_budget:
        closest = above_budget[0]
        gap_analysis = discount.compute_minimum_discount(
            max_price, closest["price"],
            cost=closest.get("cost"),
            min_margin_percent=gate.DEFAULT_MIN_MARGIN_PERCENT,
        )
        negotiation["closest_option"] = {
            **_slim_product(closest, include_score=True),
            "over_budget_by": round(closest["price"] - max_price, 2),
            "gap_is_negotiable": discount.is_negotiable_gap(max_price, closest["price"]),
            "minimum_discount_needed_percent": gap_analysis["discount_percent"],
            "discount_would_be_enough": gap_analysis["sufficient"],
        }

        if has_in_budget_options and gap_analysis["sufficient"]:
            negotiation["guidance"] = (
                f"There ARE options under ₹{max_price:g} — lead with the best of those. But the "
                f"closest match to what they actually described is {closest['name']} at "
                f"₹{closest['price']:g}, ₹{negotiation['closest_option']['over_budget_by']:g} over. "
                f"Mention it as well, and offer to check whether a "
                f"{gap_analysis['discount_percent']}% discount can bridge the gap."
            )
        elif has_in_budget_options:
            negotiation["guidance"] = (
                f"There ARE options under ₹{max_price:g} — lead with the best of those. The closer "
                f"match ({closest['name']} at ₹{closest['price']:g}) is too far over budget for any "
                f"permitted discount to reach, so mention it only as a stretch option and don't "
                f"suggest a discount could cover it."
            )
        elif gap_analysis["sufficient"]:
            negotiation["guidance"] = (
                f"Nothing good fits ₹{max_price:g}. The closest is {closest['name']} at "
                f"₹{closest['price']:g}. A {gap_analysis['discount_percent']}% discount would "
                f"bring it into budget — offer to check whether that discount can be applied, "
                f"or to show other options."
            )
        else:
            negotiation["guidance"] = (
                f"Nothing good fits ₹{max_price:g}, and the closest option "
                f"({closest['name']} at ₹{closest['price']:g}) is too far above budget for any "
                f"permitted discount to close. Be honest about that and offer to look at a "
                f"different kind of product or a higher budget."
            )
    else:
        negotiation["guidance"] = (
            f"Nothing matched well under ₹{max_price:g} and there's nothing close above it "
            f"either. Ask the customer to describe what they want differently."
        )

    response["budget_negotiation"] = negotiation
    return response


def _handle_check_stock(session, params):
    return catalog.check_stock(params["product_id"], params.get("quantity", 1))


def _handle_compute_discount(session, params):
    """Budget-gap arithmetic. Calculates only — grants nothing.

    The customer's budget is model-supplied, because only the model heard what
    they said. That's safe: a fabricated budget could at most justify a larger
    discount, and check_gate independently caps every discount at the store's
    ceiling regardless of what this function recommended.
    """
    product_id = params["product_id"]
    quantity = params.get("quantity", 1)
    customer_budget = params["customer_budget"]

    product = catalog.get_product_by_id(product_id)
    if product is None:
        return {
            "error": f"Product {product_id} does not exist.",
            "error_type": "product_not_found",
        }

    sticker = _line_amount(product, quantity)
    cost_basis = _cost_basis(product, quantity)
    try:
        analysis = discount.compute_minimum_discount(
            customer_budget=customer_budget,
            item_price=sticker,
            max_discount_percent=gate.DEFAULT_MAX_DISCOUNT_PERCENT,
            cost=cost_basis,
            min_margin_percent=gate.DEFAULT_MIN_MARGIN_PERCENT,
        )
    except ValueError as exc:
        return {"error": str(exc), "error_type": "invalid_input"}

    # When the margin floor is what stopped us, record the merchant-side
    # arithmetic in its own audit row — including the money that decision
    # protected. It goes in the log rather than in the tool result because the
    # log is read by the merchant and the tool result is read by the model.
    if analysis.get("capped_by") == discount.CAPPED_BY_MARGIN_FLOOR:
        _log_margin_protection(session, product, quantity, analysis)

    # ---- redaction, and the reason it is done in code ----
    # cost and margin_floor_price never reach the model. The system prompt does
    # tell it to say "that's the best I can do" rather than quoting a cost, but
    # a prompt is a request; not putting the number in its context in the first
    # place is a rule. The model still gets `capped_by`, which is all it needs
    # to pick the right sentence.
    model_safe = {k: v for k, v in analysis.items()
                  if k not in ("cost", "margin_floor_price")}

    if analysis["sufficient"] and analysis["discount_percent"] > 0:
        next_step = (
            f"Offer {analysis['discount_percent']}% to the customer, then pass "
            f"discount_percent={analysis['discount_percent']} to check_gate."
        )
    elif analysis["discount_percent"] == 0 and analysis["sufficient"]:
        next_step = "No discount needed — go straight to check_gate."
    elif analysis.get("capped_by") == discount.CAPPED_BY_MARGIN_FLOOR:
        next_step = (
            f"{analysis['discount_percent']}% is the deepest discount this particular item "
            f"can take. Offer it if it helps, but do not promise more, and do not explain "
            f"why in terms of cost or margin — say it's the best you can do on this one, "
            f"then offer a cheaper alternative."
        )
    else:
        next_step = ("This gap is wider than the store allows. Do not promise this discount; "
                     "offer a cheaper alternative instead.")

    return {
        **model_safe,
        "product_id": product_id,
        "product_name": product["name"],
        "quantity": quantity,
        "sticker_price": sticker,
        "customer_budget": customer_budget,
        "next_step": next_step,
    }


def _handle_check_gate(session, params):
    """Run the safety gate. This is where model-supplied intent meets real numbers.

    The model gives us a product, a quantity and (optionally) a discount it has
    agreed with the customer. We supply the price, the stock status and the
    running cart total ourselves, then hand all of it to the unmodified
    gate.check_gate().

    The amount checked is the DISCOUNTED price — the money that will actually
    leave the customer's account — while discount_percent goes to the gate
    separately so the ceiling is enforced on its own terms.
    """
    product_id = params["product_id"]
    quantity = params.get("quantity", 1)
    discount_percent = params.get("discount_percent", 0)

    # --- SECURITY FIX: quantity must be positive integer ---
    if not isinstance(quantity, int) or quantity < 1 or quantity > 999:
        return {
            "success": False,
            "error": f"Invalid quantity: {quantity}. Must be an integer between 1 and 999.",
            "error_type": "invalid_quantity",
        }

    product = catalog.get_product_by_id(product_id)
    if product is None:
        return {
            "allowed": False,
            "reason": "product_not_found",
            "explanation": f"I couldn't find product {product_id} in the catalog, so I can't approve a purchase for it.",
        }

    stock_result = catalog.check_stock(product_id, quantity)
    sticker_amount = _line_amount(product, quantity)
    payable_amount = _payable_amount(product, quantity, discount_percent)

    decision = gate.check_gate(
        amount=payable_amount,
        cart_total_so_far=session.cart_total_so_far,      # server-side, not model-supplied
        item_in_stock=stock_result["available"],
        discount_percent=discount_percent,
        # Cost comes from the catalog, like the price does — the model cannot
        # supply it, so it cannot argue its way under the margin floor.
        cost_basis=_cost_basis(product, quantity),
        min_margin_percent=gate.DEFAULT_MIN_MARGIN_PERCENT,
    )

    if decision["allowed"]:
        # Record the approval so create_order can verify it later. The discount
        # is part of the approval: changing it invalidates the approval.
        session.pending_approvals[product_id] = {
            "amount": payable_amount,
            "quantity": quantity,
            "discount_percent": discount_percent,
            "sticker_amount": sticker_amount,
        }
    else:
        # A denial revokes any approval this product already had.
        #
        # Without this, asking the gate for 20% (approved), then 21% (denied),
        # leaves the 20% approval sitting in the session — and create_order will
        # happily spend it. No money moves outside the gate's authority either
        # way, since the stale approval was itself legitimately granted. But
        # "the gate said no and an order was placed anyway" is indefensible in
        # an audit trail, and it lets the model tell the customer it secured
        # terms the store actually refused.
        #
        # Note this lives here, not in gate.py: the gate's bound checks are
        # untouched: this is the execution layer declining to reuse a decision
        # the conversation has moved past. Re-running check_gate is cheap.
        session.pending_approvals.pop(product_id, None)

    # Echo the numbers the gate actually used, so the model (and the audit log)
    # can see the basis for the decision rather than guessing at it.
    return {
        **decision,
        "product_id": product_id,
        "product_name": product["name"],
        "quantity": quantity,
        "sticker_amount": sticker_amount,
        "discount_percent": discount_percent,
        "amount_checked": payable_amount,
        "cart_total_so_far": session.cart_total_so_far,
        "in_stock": stock_result["available"],
    }


def _handle_create_order(session, params):
    """Create the Razorpay order — but only behind a valid, unused gate approval."""
    product_id = params["product_id"]
    quantity = params.get("quantity", 1)

        # --- input validation ---
    if not isinstance(quantity, int) or quantity < 1 or quantity > 999:
        return {
            "success": False,
            "error": f"Invalid quantity: {quantity}. Must be an integer between 1 and 999.",
            "error_type": "invalid_quantity",
        }


    product = catalog.get_product_by_id(product_id)
    if product is None:
        return {
            "success": False,
            "error": f"Product {product_id} does not exist.",
            "error_type": "product_not_found",
        }

    approval = session.pending_approvals.get(product_id)

    # ---- enforcement point: no order without a matching, fresh gate approval ----
    if approval is None:
        return {
            "success": False,
            "error": (
                f"Refused: check_gate has not approved a purchase of {product_id} in this "
                f"conversation. Call check_gate first."
            ),
            "error_type": "gate_not_checked",
        }
    if approval["quantity"] != quantity:
        return {
            "success": False,
            "error": (
                f"Refused: the gate approved {approval['quantity']} unit(s) of {product_id}, "
                f"but this order is for {quantity}. Re-run check_gate for the quantity you "
                f"actually want."
            ),
            "error_type": "gate_approval_mismatch",
        }

    # The charged amount is recomputed from the catalog price and the APPROVED
    # discount, then checked against what the gate signed off on. Belt and
    # braces: if these ever disagree, something is wrong and nothing is charged.
    amount = approval["amount"]
    discount_percent = approval.get("discount_percent", 0) or 0

    # Stock can change between the gate check and the order in a real store, so
    # re-check rather than trusting a stale approval.
    stock_result = catalog.check_stock(product_id, quantity)
    if not stock_result["available"]:
        session.pending_approvals.pop(product_id, None)
        return {
            "success": False,
            "error": f"{product['name']} went out of stock before the order could be placed.",
            "error_type": "out_of_stock",
            "stock_detail": stock_result,
        }

    notes = {
        "product_id": product_id,
        "product_name": product["name"],
        "quantity": quantity,
        "session_id": session.session_id,
    }
    if discount_percent:
        notes["discount_percent"] = discount_percent
        notes["sticker_amount"] = approval["sticker_amount"]

    # Razorpay caps receipts at 40 chars. Truncate and hash to keep
    # uniqueness if the combined id exceeds that limit.
    raw_receipt = f"{session.session_id}-{product_id}"
    if len(raw_receipt) > 40:
        digest = hashlib.sha256(raw_receipt.encode()).hexdigest()[:8]
        receipt = raw_receipt[:31] + "-" + digest
    else:
        receipt = raw_receipt

    order = razorpay_client.create_order(
        amount_in_rupees=amount,
        currency="INR",
        receipt=receipt,
        notes=notes,
        customer_id=session.customer_id,
    )

    if order.get("success"):
        # Consume the approval and advance the session total. Both matter: the
        # next purchase is gated against the new, higher total.
        session.pending_approvals.pop(product_id, None)
        session.cart_total_so_far += amount

        order_record = {
            "order_id": order.get("id"),
            "payment_id": order.get("payment_id"),
            "product_id": product_id,
            "product_name": product["name"],
            "quantity": quantity,
            "amount_rupees": amount,
            "discount_percent": discount_percent,
        }
        session.orders.append(order_record)

        order = {
            **order,
            "product_name": product["name"],
            "amount_rupees": amount,
            "sticker_amount": approval["sticker_amount"],
            "discount_percent": discount_percent,
            "discount_savings": round(approval["sticker_amount"] - amount, 2),
            "quantity_ordered": quantity,
            "new_cart_total": session.cart_total_so_far,
        }

        # ---- affinity memory: learn one or two plain facts from this purchase ----
        _learn_from_purchase(session, product, order)

        # ---- attach-rate bookkeeping, done in code, not by the model ----
        # If this product was the one we suggested, that's an accepted upsell.
        if (session.upsell["offered"]
                and session.upsell["resolved"] is None
                and product_id in session.upsell["offered_ids"]):
            session.upsell["resolved"] = "accepted"
            order["upsell_accepted"] = True
            _log_side_event(
                session,
                action_type="upsell_accepted",
                action_params={"product_id": product_id, "offered_ids": session.upsell["offered_ids"]},
                result={
                    "product_name": product["name"],
                    "amount_rupees": amount,
                    "order_id": order.get("id"),
                },
                reasoning=f"Customer bought {product['name']}, which was the complementary item suggested to them.",
            )

    return order


def _handle_get_complementary_products(session, params):
    """The one upsell per session.

    The once-only rule is enforced here rather than trusted to the prompt, for
    two reasons: a pushy agent is a bad shopping experience, and attach rate is
    only meaningful if the denominator is a real count of offers made.
    """
    if session.upsell["offered"]:
        return {
            "refused": True,
            "reason": "upsell_limit_reached",
            "explanation": (
                "You've already made your one product suggestion this conversation. "
                "Don't suggest anything else — just help with what the customer asked."
            ),
            "previously_offered_ids": session.upsell["offered_ids"],
            "products": [],
        }

    products = catalog.get_complementary_products(params["product_id"])
    in_stock = [p for p in products if p["stock"] > 0]

    if not in_stock:
        # Nothing to offer, so this doesn't count as an offer — leaving the
        # session's one chance unspent rather than logging a phantom upsell.
        return {
            "count": 0,
            "products": [],
            "upsell_offer": False,
            "note": "No complementary products in stock. Don't suggest anything.",
        }

    session.upsell["offered"] = True
    session.upsell["offered_ids"] = [p["id"] for p in in_stock]

    return {
        "count": len(in_stock),
        "products": [_slim_product(p) for p in in_stock],
        "upsell_offer": True,
        "instruction": (
            "This is your one suggestion for this conversation. Offer the single best fit "
            "in one short sentence with its price, and accept a 'no' gracefully."
        ),
    }


def _find_order_for_refund(session, product_id=None, order_id=None):
    """Locate the order the customer wants refunded.

    Looks in this session first, then falls back to the customer's history in the
    audit log — "cancel my order" in a fresh conversation should still work.
    """
    for order in reversed(session.orders):
        if order_id and order.get("order_id") == order_id:
            return order, "current_session"
        if product_id and order.get("product_id") == product_id:
            return order, "current_session"

    for order in audit.get_customer_orders(session.customer_id, limit=25):
        if order_id and order.get("order_id") == order_id:
            return order, "past_session"
        if product_id and order.get("product_id") == product_id:
            return order, "past_session"

    return None, None


def _handle_refund_order(session, params):
    """Refund a past order, with the same never-raise contract as create_order."""
    product_id = params.get("product_id")
    order_id = params.get("order_id")
    amount_in_rupees = params.get("amount_in_rupees")
    reason = params.get("reason", "customer requested")

    if not product_id and not order_id:
        return {
            "success": False,
            "error": "refund_order called without a product_id or an order_id.",
            "error_type": "no_order_specified",
            "user_message": "Which order would you like me to refund? I can look it up by product.",
        }

    order, source = _find_order_for_refund(session, product_id, order_id)

    if order is None:
        return {
            "success": False,
            "error": f"No order found for product_id={product_id} order_id={order_id}.",
            "error_type": "no_matching_order",
            "user_message": (
                "I couldn't find an order for that in your history, so there's nothing for me "
                "to refund. Could you tell me which item you mean, or the order id?"
            ),
        }

    resolved_order_id = order.get("order_id")

    # Refusing a second refund on the same order is this layer's job, not the
    # model's — it's the refund-side equivalent of the single-use gate approval.
    if resolved_order_id in session.refunded_order_ids:
        return {
            "success": False,
            "error": f"Order {resolved_order_id} was already refunded in this session.",
            "error_type": "already_refunded",
            "user_message": f"That order has already been refunded — nothing further is owed on it.",
        }

    payment_id = order.get("payment_id")
    if not payment_id:
        # The honest case in real test mode: we create orders, but the customer
        # completes payment through Razorpay Checkout, which this agent doesn't
        # host. No payment means no captured money and nothing to refund.
        return {
            "success": False,
            "error": (
                f"Order {resolved_order_id} has no associated payment_id — the order was created "
                f"but never paid, so there is nothing to refund."
            ),
            "error_type": "order_not_paid",
            "user_message": (
                f"That order was created but the payment was never completed, so there's nothing "
                f"to refund — you haven't been charged. I can cancel it for you instead."
            ),
        }

    refund_amount = amount_in_rupees
    order_amount = order.get("amount_rupees") or 0
    if refund_amount is not None and refund_amount > order_amount:
        return {
            "success": False,
            "error": f"Requested refund ₹{refund_amount} exceeds the order amount ₹{order_amount}.",
            "error_type": "refund_exceeds_order",
            "user_message": (
                f"That's more than the ₹{order_amount:g} you paid for that order, so I can't "
                f"refund it. The most I can return is ₹{order_amount:g}."
            ),
        }

    refund = razorpay_client.create_refund(
        payment_id=payment_id,
        amount_in_rupees=refund_amount,
        notes={
            "order_id": resolved_order_id,
            "product_id": order.get("product_id"),
            "reason": reason,
            "session_id": session.session_id,
        },
    )

    if refund.get("success"):
        refunded_rupees = refund_amount if refund_amount is not None else order_amount
        session.refunded_order_ids.add(resolved_order_id)

        # Returning the money returns the spending headroom with it. Not doing
        # this would lock a customer out of the session limit for money they no
        # longer have with us — and net exposure stays bounded either way.
        if source == "current_session":
            session.cart_total_so_far = max(0, session.cart_total_so_far - refunded_rupees)

        refund = {
            **refund,
            "order_id": resolved_order_id,
            "product_name": order.get("product_name"),
            "amount_rupees": refunded_rupees,
            "order_amount_rupees": order_amount,
            "refund_type": "full" if refund_amount is None else "partial",
            "order_source": source,
            "new_cart_total": session.cart_total_so_far,
        }

    return refund


def _handle_create_price_watch(session, params):
    """Record a price-drop watch — the "I'd rather wait" alternative to a sale.

    Two things are refused here rather than left to the prompt: watching a
    product that doesn't exist, and watching at a price the item is already at
    or below (which is not a watch, it's a purchase the customer should just be
    told about). Duplicate watches on the same product collapse into the
    existing one, so a customer who asks twice doesn't get notified twice.
    """
    product_id = params["product_id"]
    target_price = params.get("target_price")

    product = catalog.get_product_by_id(product_id)
    if product is None:
        return {
            "created": False,
            "error": f"Product {product_id} does not exist.",
            "error_type": "product_not_found",
        }

    current_price = product["price"]

    if target_price is not None and current_price <= target_price:
        return {
            "created": False,
            "reason": "already_at_target",
            "product_id": product_id,
            "product_name": product["name"],
            "current_price": current_price,
            "target_price": target_price,
            "user_message": (
                f"{product['name']} is already ₹{current_price:g}, at or below the "
                f"₹{target_price:g} they'd pay — tell them that and offer to buy it now "
                f"instead of setting a watch."
            ),
        }

    existing = watches.get_active_watches(
        customer_id=session.customer_id, product_id=product_id
    )
    if existing:
        watch = existing[0]
        return {
            "created": False,
            "reason": "watch_already_active",
            "watch_id": watch["id"],
            "product_id": product_id,
            "product_name": product["name"],
            "target_price": watch["target_price"],
            "current_price": current_price,
            "user_message": (
                f"There's already an active watch on {product['name']}"
                + (f" at ₹{watch['target_price']:g}" if watch["target_price"] is not None else "")
                + ". Reassure them it's still in place rather than setting a second one."
            ),
        }

    watch = watches.create_watch(
        session_id=session.session_id,
        product_id=product_id,
        price_at_creation=current_price,
        target_price=target_price,
        customer_id=session.customer_id,
        product_name=product["name"],
    )

    return {
        "created": True,
        "watch_id": watch["id"],
        "product_id": product_id,
        "product_name": product["name"],
        "current_price": current_price,
        "target_price": target_price,
        "watching_for": (
            f"price at or below ₹{target_price:g}" if target_price is not None
            else f"any drop below ₹{current_price:g}"
        ),
        "note": (
            "Confirm the watch in one short sentence with the product and the price you're "
            "watching for. The store logs a notification when it trips — do not promise "
            "an email or a text message."
        ),
    }


def _learn_from_purchase(session, product, order):
    """Update this customer's affinity memory after a completed order.

    Called from the create_order success path and wrapped so it can never take
    an order down with it: the money has already moved, and a failure to write a
    preference row is not a reason to hand the customer an error.

    Note the DEMO_CUSTOMER caveat at the top of this file — with one shared
    identity, these signals accumulate across every demo run. That's what makes
    the feature visible in a demo, and it is not how it would behave with real
    per-user auth.
    """
    if not session.customer_id:
        return []

    try:
        recorded = preferences.record_purchase(session.customer_id, product)
    except Exception as exc:
        _log_side_event(
            session,
            action_type="error",
            action_params={"stage": "preference_inference", "product_id": product["id"]},
            result={"error": str(exc)},
            reasoning="Preference inference failed after a successful order; the order itself is unaffected.",
        )
        return []

    if recorded:
        _log_side_event(
            session,
            action_type="preference_signal",
            action_params={
                "product_id": product["id"],
                "product_name": product["name"],
                "category": product["category"],
                "price": product["price"],
                "order_id": order.get("id"),
            },
            result={
                "signals": [
                    {
                        "signal_type": s["signal_type"],
                        "signal_value": s["signal_value"],
                        "count": s["count"],
                        "confidence": s["confidence"],
                    }
                    for s in recorded
                ],
                "rule": (
                    "price_tier from the catalog median; category_affinity from the product's "
                    "category. Rules only — no model."
                ),
            },
            reasoning=f"Customer bought {product['name']}; recording what that suggests about their preferences.",
        )
    return recorded


def _handle_get_past_orders(session, params):
    """Purchase history, read from our own audit log rather than from Razorpay."""
    limit = params.get("limit", 5)
    orders = audit.get_customer_orders(session.customer_id, limit=limit)
    return {
        "customer_id": session.customer_id,
        "count": len(orders),
        "orders": orders,
        "note": "Ordered newest first. Timestamps are UTC." if orders
                else "No previous orders on record for this customer.",
    }


TOOL_HANDLERS = {
    "search_catalog": _handle_search_catalog,
    "check_stock": _handle_check_stock,
    "compute_discount": _handle_compute_discount,
    "check_gate": _handle_check_gate,
    "create_order": _handle_create_order,
    "get_complementary_products": _handle_get_complementary_products,
    "refund_order": _handle_refund_order,
    "get_past_orders": _handle_get_past_orders,
    "create_price_watch": _handle_create_price_watch,
}

# Maps a tool name to the audit action_type recorded for it. Tools whose outcome
# changes the meaning of the event (an order that was refused isn't an order) are
# resolved dynamically in _audit_fields_for.
AUDIT_ACTION_TYPES = {
    "search_catalog": "search",
    "check_stock": "stock_check",
    "compute_discount": "discount_computed",
    "check_gate": "gate_check",
    "get_past_orders": "past_orders_lookup",
}


def _audit_fields_for(tool_name, result):
    """Decide the action_type and gate_decision recorded for a completed call."""
    if tool_name == "check_gate":
        allowed = bool(result.get("allowed"))
        return ("gate_check" if allowed else "rejection", "allowed" if allowed else "denied")

    if tool_name == "create_order":
        if result.get("success"):
            return ("order_created", "allowed")
        # A gate-related refusal is a denial; a gateway failure is just an error.
        if result.get("error_type") in ("gate_not_checked", "gate_approval_mismatch"):
            return ("rejection", "denied")
        return ("rejection", None)

    if tool_name == "get_complementary_products":
        # Only a real offer counts toward the attach-rate denominator.
        return ("upsell_offered" if result.get("upsell_offer") else "complementary_lookup", None)

    if tool_name == "refund_order":
        return ("refund_created" if result.get("success") else "rejection", None)

    if tool_name == "create_price_watch":
        # A watch that wasn't created — bad product, already at target, already
        # watched — is this layer declining the action, which is what
        # "rejection" means everywhere else in this log.
        return ("price_watch_created" if result.get("created") else "rejection", None)

    return (AUDIT_ACTION_TYPES.get(tool_name, "search"), None)


def execute_tool(session, tool_name, tool_input):
    """Run one tool call, audit it, and return the result dict for the model.

    The audit write happens here — at the execution layer — precisely so it is
    not something the model can forget to do.
    """
    # `reasoning` is for the log, not for the handler.
    reasoning = tool_input.get("reasoning", "(no reasoning given)")
    params = {k: v for k, v in tool_input.items() if k != "reasoning"}

    handler = TOOL_HANDLERS.get(tool_name)
    if handler is None:
        result = {"error": f"Unknown tool: {tool_name}"}
        audit.log_event(
            session_id=session.session_id,
            user_query=session.current_user_query,
            agent_reasoning=reasoning,
            action_type="error",
            action_params={"tool": tool_name, **params},
            result=result,
            customer_id=session.customer_id,
        )
        return result

    try:
        result = handler(session, params)
        action_type, gate_decision = _audit_fields_for(tool_name, result)
    except Exception as exc:
        # A tool blowing up must not kill the conversation — hand the model an
        # error it can explain, and record it.
        result = {"error": f"{tool_name} failed: {exc}", "error_type": "tool_exception"}
        action_type, gate_decision = "error", None

    audit.log_event(
        session_id=session.session_id,
        user_query=session.current_user_query,
        agent_reasoning=reasoning,
        action_type=action_type,
        action_params={"tool": tool_name, **params},
        result=result,
        gate_decision=gate_decision,
        customer_id=session.customer_id,
    )
    return result


# ---------------------------------------------------------------------------
# Session lifecycle
# ---------------------------------------------------------------------------

def link_customer(session, customer_details=None):
    """Attach a Razorpay Customer to the session.

    See DEMO_CUSTOMER above for the authentication caveat: with no login system,
    every session resolves to the same demo shopper. The Razorpay call and the
    audit linkage are real; only the identity is stubbed.
    """
    details = customer_details or DEMO_CUSTOMER
    customer = razorpay_client.get_or_create_customer(
        name=details["name"], email=details["email"], contact=details["contact"]
    )

    if customer.get("success"):
        session.customer = customer
        session.customer_id = customer.get("id")

    audit.log_event(
        session_id=session.session_id,
        user_query=None,
        agent_reasoning="Session started; resolving the Razorpay Customer record so orders and history attach to it.",
        action_type="customer_linked" if customer.get("success") else "error",
        action_params={"name": details["name"], "email": details["email"]},
        result={
            "customer_id": customer.get("id"),
            "success": customer.get("success"),
            "mock": customer.get("mock", False),
            "error": customer.get("error"),
        },
        customer_id=session.customer_id,
    )
    return customer


def finalize_session(session):
    """Close out a session's open bookkeeping.

    Right now that means one thing: an upsell that was offered and never taken up
    is recorded as declined. Without this the attach-rate denominator would
    include offers that are neither accepted nor declined, and the rate would
    drift upward for no reason other than sessions ending.

    Safe to call more than once.
    """
    if session.upsell["offered"] and session.upsell["resolved"] is None:
        session.upsell["resolved"] = "declined"
        _log_side_event(
            session,
            action_type="upsell_declined",
            action_params={"offered_ids": session.upsell["offered_ids"]},
            result={"outcome": "declined", "reason": "session ended without the suggested item being bought"},
            reasoning="Session ended and the suggested complementary product was not purchased.",
        )
        return "declined"
    return session.upsell["resolved"]


# ---------------------------------------------------------------------------
# The agent loop
# ---------------------------------------------------------------------------

def _get_anthropic_client():
    # Strip whitespace from env vars — hosted secret stores and shell exports
    # sometimes introduce leading/trailing spaces that break HTTP headers.
    api_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Export your Anthropic API key before running the agent."
        )
    # Overwrite the possibly-dirty env var so the SDK reads the clean value.
    os.environ["ANTHROPIC_API_KEY"] = api_key

    extra_headers = {}
    workspace_id = (os.environ.get("ANTHROPIC_WORKSPACE_ID") or "").strip()
    if workspace_id:
        extra_headers["anthropic-workspace-id"] = workspace_id
    return anthropic.Anthropic(default_headers=extra_headers if extra_headers else None)


def _response_text(response):
    """Concatenate the text blocks of a response into the reply for the user."""
    return "\n".join(block.text for block in response.content if block.type == "text").strip()


def run_turn(session, user_message, client=None, verbose=False):
    """Handle one user message end to end.

    Sends the conversation to the model; while it asks for tools, executes them
    for real, audits each one, and feeds the results back; returns the final text.
    """
    client = client or _get_anthropic_client()
    session.current_user_query = user_message
    session.messages.append({"role": "user", "content": user_message})

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=session.messages,
            )
        except anthropic.APIError as exc:
            # The model itself is unreachable. Log it and degrade to a sentence
            # rather than a stack trace in the customer's face.
            audit.log_event(
                session_id=session.session_id,
                user_query=user_message,
                agent_reasoning="Anthropic API call failed while handling this turn.",
                action_type="error",
                action_params={"model": MODEL, "iteration": iteration},
                result={"error": str(exc)},
                customer_id=session.customer_id,
            )
            return "Sorry — I'm having trouble reaching my assistant service right now. Please try again in a moment."

        session.messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason != "tool_use":
            return _response_text(response)

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            if verbose:
                print(f"    [tool] {block.name}({json.dumps(block.input, default=str)})")
            result = execute_tool(session, block.name, block.input)
            if verbose:
                print(f"    [ -> ] {json.dumps(result, default=str)[:300]}")
            tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": json.dumps(result, default=str),
                }
            )

        # All results from one assistant turn go back in a single user message.
        session.messages.append({"role": "user", "content": tool_results})

    # Ran out of iterations — the model is looping. Stop it and say so.
    audit.log_event(
        session_id=session.session_id,
        user_query=user_message,
        agent_reasoning="Tool loop exceeded the iteration limit; aborting the turn.",
        action_type="error",
        action_params={"max_iterations": MAX_TOOL_ITERATIONS},
        result={"error": "max_tool_iterations_exceeded"},
        customer_id=session.customer_id,
    )
    return "Sorry — I got stuck working on that. Could you rephrase what you're looking for?"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_banner(session):
    print("=" * 70)
    print("  Razorpay Shopping Agent — fitness & athleisure")
    print("=" * 70)
    print(f"  session   : {session.session_id}")
    print(f"  model     : {MODEL}")
    print(f"  limits    : ₹{gate.DEFAULT_MAX_SINGLE_ITEM} per item, "
          f"₹{gate.DEFAULT_MAX_CART_TOTAL} per session, "
          f"{gate.DEFAULT_MAX_DISCOUNT_PERCENT}% max discount")
    if razorpay_client.is_mock_mode():
        print("  razorpay  : MOCK MODE — orders, refunds and customers are stubs")
    elif razorpay_client.credentials_available():
        print("  razorpay  : test-mode credentials found")
    else:
        print("  razorpay  : no credentials set — order creation will fail gracefully")
    print(f"  customer  : {session.customer_id or '(not linked)'}")
    print()
    print("  commands  : /audit  /cart  /metrics  /watches  /prefs  /reset  /quit   (verbose: -v)")
    print("=" * 70)
    print()


def _new_cli_session():
    session = ShoppingSession()
    link_customer(session)
    return session


def main():
    # The catalog's cosine-similarity matmul emits harmless RuntimeWarnings on
    # macOS/Accelerate. Silence them so the demo transcript stays readable.
    import numpy as np
    np.seterr(all="ignore")

    import metrics

    verbose = "-v" in sys.argv or "--verbose" in sys.argv

    try:
        client = _get_anthropic_client()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return 1

    session = _new_cli_session()
    _print_banner(session)

    while True:
        try:
            user_input = input("you > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input in ("/quit", "/exit"):
            break
        if user_input == "/audit":
            print(audit.format_session_log(session.session_id))
            print()
            continue
        if user_input == "/metrics":
            # finalize first so this session's upsell outcome is counted
            finalize_session(session)
            print(metrics.format_summary())
            print()
            continue
        if user_input == "/cart":
            print(f"\n  cart total this session : ₹{session.cart_total_so_far}")
            print(f"  headroom remaining      : ₹{gate.DEFAULT_MAX_CART_TOTAL - session.cart_total_so_far}")
            if session.orders:
                print("  orders:")
                for order in session.orders:
                    discount_note = (f"  (-{order['discount_percent']}%)"
                                     if order.get("discount_percent") else "")
                    print(f"    - {order['order_id']}  {order['product_name']} "
                          f"x{order['quantity']}  ₹{order['amount_rupees']}{discount_note}")
            else:
                print("  orders: none yet")
            print(f"  upsell                  : offered={session.upsell['offered']} "
                  f"resolved={session.upsell['resolved']}")
            print()
            continue
        if user_input == "/watches":
            active = watches.get_active_watches(customer_id=session.customer_id)
            print(f"\n  active price watches for {session.customer_id or '(no customer)'}: {len(active)}")
            for watch in active:
                target = (f"target ₹{watch['target_price']:g}"
                          if watch["target_price"] is not None else "any drop")
                print(f"    #{watch['id']}  {watch['product_name']}  ({target}, "
                      f"was ₹{watch['price_at_creation']:g})")
            print("\n  check them with:  python3 app/check_price_watches.py --simulate-drop 20\n")
            continue
        if user_input == "/prefs":
            print(f"\n  preference memory for {session.customer_id or '(no customer)'}")
            print(f"    confident signals : {preferences.describe(session.customer_id)}")
            all_signals = preferences.get_signals(session.customer_id)
            for signal in all_signals:
                applied = "applied" if signal["confidence"] >= preferences.MIN_CONFIDENCE_TO_APPLY else "too weak to apply"
                print(f"    {signal['signal_type']}={signal['signal_value']}  "
                      f"count {signal['count']}, confidence {signal['confidence']:g}  ({applied})")
            if not all_signals:
                print("    (nothing learned yet — buy something and check again)")
            print("\n  NOTE: no auth in this build, so every session is the same customer.\n")
            continue
        if user_input == "/reset":
            finalize_session(session)
            session = _new_cli_session()
            print(f"\n  new session: {session.session_id}\n")
            continue

        reply = run_turn(session, user_input, client=client, verbose=verbose)
        print(f"\nagent > {reply}\n")

    finalize_session(session)
    print(f"\nSession {session.session_id} ended. "
          f"Cart total ₹{session.cart_total_so_far}. "
          f"Run  python3 -c \"import sys; sys.path.insert(0,'app'); import audit; "
          f"print(audit.format_session_log('{session.session_id}'))\"  to replay the audit trail.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
