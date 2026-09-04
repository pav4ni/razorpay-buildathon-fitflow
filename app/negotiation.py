"""
negotiation.py — the buyer's side of a two-sided negotiation.

The merchant agent already negotiates: tell it a budget and it works out the
smallest discount that closes the gap, bounded by the policy cap and the margin
floor. What was missing was an opponent — something with its own incentives that
does not simply accept the first number it is given.

This module is that opponent, and it is deliberately NOT a language model, for
the reason buyer_agent.py has always given: a model on this side would make the
demo non-deterministic and would prove nothing extra. What is being demonstrated
is that the merchant holds its bounds under pressure, and pressure is more
convincing when it is reproducible. So the buyer is a policy — a few rules about
when to concede and when to leave.

THE ONE THING THAT MAKES THIS A NEGOTIATION RATHER THAN A REQUEST

The buyer has a `true_max`: the most it would actually pay. It never says that
number out loud. It opens below it, concedes toward it across rounds, and walks
away if the merchant never gets there. That asymmetry — one side holding a
number the other cannot see — is what makes the exchange a negotiation, and it
is why the buyer's stated budget and its real ceiling are two separate fields
here.

`true_max` IS written to the audit log, on purpose. The log is read by the
merchant and by a judge after the fact, and the interesting question — "did the
merchant leave money on the table, or hold a line it was right to hold?" — is
unanswerable without it. It reaches the log; it never reaches the merchant's
model, because the only thing sent over the wire is `utterance`.

WHAT THIS MODULE CANNOT DO

It cannot buy anything. It produces sentences and decisions. Every order still
goes through the merchant's own check_gate, which enforces the spending limits
and the margin floor exactly as it does for a human shopper. A buyer that
negotiates well simply gets told "no" more politely; it does not get told "yes"
more often. test_negotiation.py asserts that directly.
"""

import audit

# What the buyer decided to do at the end of a round.
ACCEPT = "accept"
COUNTER = "counter"
WALK_AWAY = "walk_away"

# How far below its true ceiling the buyer opens. 0.80 is a plausible haggle:
# low enough to leave visible room to concede, high enough that the merchant
# does not dismiss it outright.
DEFAULT_OPENING_RATIO = 0.80

# Rounds of back-and-forth before the buyer commits or leaves. Three keeps a
# demo watchable and bounds the token spend of the merchant's side.
DEFAULT_MAX_ROUNDS = 3

# The buyer concedes toward its true ceiling but deliberately stops short of it.
#
# This exists because of a real hole found in testing: a schedule that reaches
# true_max on the final round makes the buyer SAY its true ceiling out loud, and
# a ceiling you have stated is not a hidden one. Stopping at 97% keeps the number
# private for the entire exchange.
#
# It costs the buyer nothing. If the merchant lands between the final stated
# budget and the real ceiling, the out-of-rounds branch in decide() accepts it
# anyway — the buyer still pays up to true_max, it just never announces that it
# would.
STATED_CEILING_RATIO = 0.97


class BuyerPolicy:
    """A shopper with a hidden ceiling and a concession schedule.

    Pure: it holds no connection, writes no log and sends no message. It is
    handed the best price the merchant has offered so far and returns a
    decision. That is what makes every branch of it testable without spending a
    penny on the merchant's model.
    """

    def __init__(self, product_hint, true_max, opening_ratio=DEFAULT_OPENING_RATIO,
                 max_rounds=DEFAULT_MAX_ROUNDS, walk_away_below=None):
        if true_max <= 0:
            raise ValueError(f"true_max must be positive, got {true_max}")
        if not 0 < opening_ratio <= 1:
            raise ValueError(f"opening_ratio must be in (0, 1], got {opening_ratio}")

        self.product_hint = product_hint
        self.true_max = float(true_max)
        self.opening_ratio = opening_ratio
        self.max_rounds = max_rounds

        # The buyer's opening claim about its budget. Rounded to a whole rupee
        # because people negotiate in round numbers, not in paise.
        self.opening_budget = float(round(self.true_max * opening_ratio))
        self.stated_budget = self.opening_budget

        # The highest figure the buyer will ever say out loud — strictly below
        # its real ceiling. See STATED_CEILING_RATIO.
        self.max_stated_budget = float(round(self.true_max * STATED_CEILING_RATIO))
        if self.max_stated_budget < self.opening_budget:
            # A very high opening_ratio could otherwise make the buyer concede
            # downward, which is not a concession.
            self.max_stated_budget = self.opening_budget

        self.round = 0
        self.history = []          # one record per round, for the transcript
        self.outcome = None

        # The lowest price the merchant has named at ANY point, not just the
        # most recent one. This matters, and it is counter-intuitive: because
        # the merchant offers the SMALLEST discount that clears the stated
        # budget, its quote gets WORSE as the buyer concedes. A buyer that only
        # remembered the latest number would haggle its way to a higher price
        # than it was offered in round one. So the buyer holds the merchant to
        # the best figure it has actually put on the table.
        self.best_price = None

    # -- the concession schedule -------------------------------------------

    def budget_for_round(self, round_number):
        """The budget the buyer is willing to *admit to* at a given round.

        Moves linearly from the opening claim toward `max_stated_budget`, which
        sits just below the true ceiling — so the buyer concedes visibly without
        ever naming the number it would actually pay. Conceding past your own
        ceiling is not a negotiating position; conceding exactly to it hands the
        other side your whole hand.
        """
        if round_number <= 1:
            return self.opening_budget
        if self.max_rounds <= 1:
            return self.max_stated_budget
        progress = min(1.0, (round_number - 1) / float(self.max_rounds - 1))
        budget = self.opening_budget + (self.max_stated_budget - self.opening_budget) * progress
        return float(round(min(budget, self.max_stated_budget)))

    # -- the decision -------------------------------------------------------

    def decide(self, merchant_price):
        """What to do about the best price the merchant has put on the table.

        Args:
            merchant_price: best price offered so far for the target product,
                or None if the merchant has not named one yet.

        Returns a dict with `decision`, `reasoning`, and the numbers behind it.
        The reasoning is written here rather than by the caller so the buyer's
        stated motive and its actual rule can never drift apart in the log.
        """
        self.round += 1
        rounds_remaining = self.max_rounds - self.round

        if merchant_price is not None:
            self.best_price = (merchant_price if self.best_price is None
                               else min(self.best_price, merchant_price))
        best = self.best_price

        if merchant_price is None and best is None:
            # Nothing quotable yet — the merchant is still asking questions
            # rather than pricing anything.
            #
            # This branch MUST honour the round limit. An earlier version
            # returned COUNTER unconditionally here, and against a merchant that
            # kept asking a clarifying question it never terminated: the buyer
            # re-stated its budget forever, and every lap was a paid API call.
            # A negotiation that cannot end is not a negotiation, and "the other
            # side never quoted a price" is a perfectly good reason to leave.
            if rounds_remaining > 0:
                decision, reasoning = COUNTER, (
                    "The merchant has not named a price for this item yet, so there is "
                    "nothing to evaluate. Answering their question and restating what I want."
                )
                next_budget = self.budget_for_round(self.round + 1)
            else:
                decision, reasoning = WALK_AWAY, (
                    f"Out of rounds and the merchant never quoted a price for "
                    f"{self.product_hint}. Nothing to accept, so I'm leaving."
                )
                next_budget = self.stated_budget
        elif best <= self.stated_budget:
            decision, reasoning = ACCEPT, (
                f"The merchant met the ₹{self.stated_budget:g} I asked for — ₹{best:g} "
                f"is at or under it. Accepting rather than pushing further; there is nothing "
                f"left to win here."
            )
            next_budget = self.stated_budget
        elif rounds_remaining > 0:
            next_budget = self.budget_for_round(self.round + 1)
            decision, reasoning = COUNTER, (
                f"₹{best:g} is above the ₹{self.stated_budget:g} I've admitted to, "
                f"and I have {rounds_remaining} round(s) left. Conceding to ₹{next_budget:g} — "
                f"still below my real ceiling, so I keep room to move."
            )
        elif best <= self.true_max:
            decision, reasoning = ACCEPT, (
                f"Out of rounds. The best the merchant has offered is ₹{best:g} — more than I "
                f"opened at, but at or under the ₹{self.true_max:g} I was actually prepared to "
                f"pay. Taking it at that price."
            )
            next_budget = self.stated_budget
        else:
            decision, reasoning = WALK_AWAY, (
                f"Out of rounds and the best offer, ₹{best:g}, is still above the "
                f"₹{self.true_max:g} I was prepared to pay. Walking away — the merchant held "
                f"its line and I hold mine."
            )
            next_budget = self.stated_budget

        record = {
            "round": self.round,
            "stated_budget": self.stated_budget,
            "true_max": self.true_max,
            "merchant_price": merchant_price,
            "best_price_seen": best,
            "decision": decision,
            "reasoning": reasoning,
            "rounds_remaining": rounds_remaining,
        }
        self.history.append(record)

        if decision == COUNTER:
            self.stated_budget = next_budget
        else:
            self.outcome = decision

        return record

    # -- what the buyer actually says --------------------------------------

    def opening_utterance(self):
        """The buyer's first message. States a budget; states a lowballed one."""
        return (
            f"I'm looking for {self.product_hint}. My budget is about "
            f"₹{self.opening_budget:g} — what can you do?"
        )

    def clarify_utterance(self):
        """What to say when the merchant asked a question instead of quoting.

        The merchant is built to ask ONE clarifying question when a request is
        ambiguous, which is correct behaviour and which an autonomous buyer has
        to be able to answer. Repeating the budget — as the plain counter does —
        answers nothing and invites the same question again.
        """
        return (
            f"Either option is fine — whichever you'd recommend. I'm after "
            f"{self.product_hint}, and I can spend about ₹{self.stated_budget:g}. "
            f"What's your best price?"
        )

    def counter_utterance(self):
        """A concession, phrased as a stretch rather than as a surrender."""
        return (
            f"That's more than I wanted to spend. I could stretch to "
            f"₹{self.stated_budget:g} — can you make that work?"
        )

    def accept_utterance(self, merchant_price=None):
        """Accept at the best price the merchant actually quoted.

        Naming the figure back to the merchant matters: by the final round the
        merchant's most recent quote may be higher than its best one, because a
        higher stated budget needs a smaller discount. Referring to the price it
        already offered is both what a real buyer would do and what keeps the
        agreed number unambiguous.
        """
        price = merchant_price if merchant_price is not None else self.best_price
        return (
            f"You quoted ₹{price:g} for it earlier — I'll take it at that price. "
            f"Please go ahead and place the order."
        )

    def walk_away_utterance(self):
        return (
            "That's still over what I can justify spending, so I'll leave it for now. "
            "Thanks for looking."
        )


# ---------------------------------------------------------------------------
# Audit logging — the buyer's half of the transcript
#
# The merchant's moves already write themselves (search, discount_computed,
# margin_protection, gate_check, order_created). These functions add the other
# half, against the SAME session_id, so one /audit read replays both sides of
# the conversation interleaved in the order they actually happened. That single
# combined transcript is the point of the feature.
# ---------------------------------------------------------------------------

def log_opened(session_id, policy, customer_id=None):
    return audit.log_event(
        session_id=session_id,
        user_query=policy.opening_utterance(),
        agent_reasoning=(
            f"Opening at ₹{policy.opening_budget:g}, which is "
            f"{policy.opening_ratio:.0%} of the ₹{policy.true_max:g} I would actually pay. "
            f"Opening at my true ceiling would leave me nothing to concede."
        ),
        action_type="negotiation_opened",
        action_params={
            "actor": "buyer_agent",
            "product_hint": policy.product_hint,
            "opening_budget": policy.opening_budget,
            "max_rounds": policy.max_rounds,
        },
        result={
            # Recorded for the merchant's own post-hoc analysis and for a judge.
            # The merchant's MODEL never receives this — only `utterance` is sent.
            "hidden_true_max": policy.true_max,
            "note": (
                "true_max is the buyer's private ceiling. It is written here so the "
                "negotiation can be audited after the fact; it is never sent to the "
                "merchant agent."
            ),
        },
        customer_id=customer_id,
    )


def log_round(session_id, record, utterance, customer_id=None):
    """One buyer move: what it saw, what it decided, and why."""
    return audit.log_event(
        session_id=session_id,
        user_query=utterance,
        agent_reasoning=record["reasoning"],
        action_type={
            ACCEPT: "negotiation_accepted",
            WALK_AWAY: "negotiation_walked_away",
            COUNTER: "buyer_counter_offer",
        }[record["decision"]],
        action_params={
            "actor": "buyer_agent",
            "round": record["round"],
            "stated_budget": record["stated_budget"],
            "merchant_price": record["merchant_price"],
            "rounds_remaining": record["rounds_remaining"],
        },
        result={
            "decision": record["decision"],
            "hidden_true_max": record["true_max"],
            "merchant_price": record["merchant_price"],
            "within_true_max": (
                record["merchant_price"] is not None
                and record["merchant_price"] <= record["true_max"]
            ),
        },
        customer_id=customer_id,
    )


def log_merchant_position(session_id, round_number, offer, customer_id=None):
    """The merchant's move, restated as a negotiating position.

    The merchant already logs *why* it priced something the way it did
    (discount_computed, and margin_protection when the floor bit). This row adds
    the thing those rows do not carry: that the number was an offer, in a
    negotiation, at a particular round. It is what lets the transcript read as an
    exchange rather than as two unrelated logs.
    """
    return audit.log_event(
        session_id=session_id,
        user_query=None,
        agent_reasoning=(
            f"Merchant's position at round {round_number}: "
            + (
                f"best price ₹{offer['price']:g}"
                + (f" at {offer['discount_percent']:g}% off" if offer.get("discount_percent") else "")
                + (
                    f", and it cannot go further because {offer['capped_by']} is binding"
                    if offer.get("capped_by") else ""
                )
                if offer else "no price named yet"
            )
        ),
        action_type="merchant_counter_offer",
        action_params={"actor": "merchant_agent", "round": round_number},
        result=offer or {"price": None, "note": "merchant did not quote a price this round"},
        customer_id=customer_id,
    )


def summarize(policy, final_price=None, order_id=None):
    """The one-paragraph outcome, for the end of a demo run."""
    return {
        "product_hint": policy.product_hint,
        "hidden_true_max": policy.true_max,
        "opening_budget": policy.opening_budget,
        "final_stated_budget": policy.stated_budget,
        "rounds_used": policy.round,
        "max_rounds": policy.max_rounds,
        "outcome": policy.outcome,
        "best_price_offered": policy.best_price,
        "final_price": final_price,
        "order_id": order_id,
        "saved_vs_true_max": (
            round(policy.true_max - final_price, 2)
            if final_price is not None and policy.outcome == ACCEPT else None
        ),
    }
