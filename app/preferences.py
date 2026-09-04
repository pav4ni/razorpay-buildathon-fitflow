"""
preferences.py — lightweight affinity memory: what we've noticed about a shopper.

The idea in one line: after someone buys something, write down one or two plain
facts about what they bought, and let those facts break ties in future searches.

Deliberately NOT a model:
- The inference is two if-statements (see infer_signals_from_purchase). A judge
  can read it in ten seconds and predict exactly what it will do, which is worth
  more here than any accuracy a learned ranker could add over 35 products.
- Confidence is just a count, capped. Buy footwear three times and the footwear
  signal is at full strength; buy it once and it barely registers.
- The boost is a tie-breaker, not a re-ranking. See PREFERENCE_MAX_BOOST below
  for the arithmetic showing it can't outweigh relevance.

LIMITATION, and it matters for how this demos — agent.py has no authentication,
so every session resolves to the same DEMO_CUSTOMER. Preference memory therefore
accumulates across all demo runs against one identity, which is what makes it
visible in a demo at all, and is also exactly why it wouldn't survive contact
with real multi-user traffic. Nothing here is per-user-safe until there's a real
login; the storage is keyed on customer_id, so that's the only piece that would
need to change.
"""

import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit.db")

# Signal types this build understands. brand_affinity is listed because it's the
# obvious third signal, but the catalog has no brand field, so nothing infers it
# and nothing boosts on it — better an honest gap than a fake column.
SIGNAL_TYPES = (
    "price_tier",         # "budget" or "premium", relative to the catalog median
    "category_affinity",  # a catalog category they keep coming back to
    "brand_affinity",     # storable, not inferred — no brand data in the catalog
)

# Signal types where the values are mutually exclusive: a shopper is a budget
# shopper or a premium one, not both, so only their dominant value is applied at
# ranking time. Without this a customer with history on both sides gets a boost
# on every product in the catalog, which is a constant offset — it changes no
# ordering and explains nothing. Category affinity is deliberately NOT in here:
# liking both footwear and supplements is a real thing to like.
EXCLUSIVE_SIGNAL_TYPES = ("price_tier",)

# Observations needed for a signal to count as fully confident. Three is a
# judgement call: one purchase is noise, three is a habit.
FULL_CONFIDENCE_COUNT = 3

# Below this confidence a signal is ignored at ranking time. At
# FULL_CONFIDENCE_COUNT = 3 this means "seen at least twice" — one stray purchase
# never moves anyone's search results.
MIN_CONFIDENCE_TO_APPLY = 0.6

# Ceiling on what preference can add to a match score, per signal type.
#
# Sense of scale, against the existing blend in catalog.py
# (0.8 * semantic + 0.2 * rating):
#   - the rating term spans about 0.056 across the catalog's real ratings (3.5-4.9)
#   - adjacent search results typically differ by 0.01-0.05 of semantic score
# So 0.02 per signal, 0.04 if both a category and a price-tier signal match, sits
# below a genuine relevance difference and above a dead heat. It nudges; it can't
# pull an irrelevant product to the top.
PREFERENCE_WEIGHT_PER_SIGNAL = 0.02
PREFERENCE_MAX_BOOST = 0.04


def _connect():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS customer_preferences (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            customer_id  TEXT NOT NULL,
            signal_type  TEXT NOT NULL,
            signal_value TEXT NOT NULL,
            count        INTEGER NOT NULL DEFAULT 0,
            confidence   REAL NOT NULL DEFAULT 0,
            last_updated TEXT NOT NULL
        )
        """
    )
    # One row per (customer, signal_type, signal_value) — the count is the
    # history, so duplicate rows would be a bug, not extra data.
    conn.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS idx_pref_unique
        ON customer_preferences (customer_id, signal_type, signal_value)
        """
    )
    conn.commit()
    return conn


def _confidence_for(count):
    """Count -> confidence in [0, 1]. Linear, capped, explainable."""
    return round(min(1.0, count / FULL_CONFIDENCE_COUNT), 4)


def record_signal(customer_id, signal_type, signal_value, increment=1):
    """Add one observation of a signal, creating the row if it's new.

    Returns the updated signal as a dict, or None if there's no customer to
    attribute it to (an unlinked session records nothing rather than piling up
    orphan rows).
    """
    if not customer_id or not signal_value:
        return None

    now = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        # UPSERT on the unique index: first observation inserts, later ones bump
        # the count. Keeps "how many times have we seen this" in one place.
        # (ON CONFLICT needs SQLite 3.24+, from 2018 — fine everywhere this runs.)
        conn.execute(
            """
            INSERT INTO customer_preferences (
                customer_id, signal_type, signal_value, count, confidence, last_updated
            ) VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (customer_id, signal_type, signal_value) DO UPDATE SET
                count = count + excluded.count,
                last_updated = excluded.last_updated
            """,
            (customer_id, signal_type, signal_value, increment, _confidence_for(increment), now),
        )
        # Confidence is derived from the count, so recompute it after the bump
        # rather than trying to express the cap inside the upsert.
        conn.execute(
            """
            UPDATE customer_preferences
            SET confidence = MIN(1.0, CAST(count AS REAL) / ?)
            WHERE customer_id = ? AND signal_type = ? AND signal_value = ?
            """,
            (float(FULL_CONFIDENCE_COUNT), customer_id, signal_type, signal_value),
        )
        conn.commit()
        row = conn.execute(
            """
            SELECT * FROM customer_preferences
            WHERE customer_id = ? AND signal_type = ? AND signal_value = ?
            """,
            (customer_id, signal_type, signal_value),
        ).fetchone()
    finally:
        conn.close()

    return dict(row) if row else None


def get_signals(customer_id, signal_type=None, min_confidence=0.0):
    """Everything we've noticed about one customer, strongest first."""
    if not customer_id:
        return []

    query = "SELECT * FROM customer_preferences WHERE customer_id = ? AND confidence >= ?"
    args = [customer_id, min_confidence]
    if signal_type is not None:
        query += " AND signal_type = ?"
        args.append(signal_type)
    query += " ORDER BY confidence DESC, count DESC, signal_value ASC"

    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(query, args).fetchall()]
    finally:
        conn.close()


def clear_signals(customer_id):
    """Forget a customer. Used by the tests; also the honest answer to "can you
    delete what you know about me?"."""
    conn = _connect()
    try:
        conn.execute("DELETE FROM customer_preferences WHERE customer_id = ?", (customer_id,))
        conn.commit()
    finally:
        conn.close()


def infer_signals_from_purchase(product, median_price=None):
    """The whole inference layer. Two rules, no model.

    Args:
        product: the catalog product that was bought.
        median_price: catalog median, passed in so this stays a pure function
            (the tests pin a median instead of depending on catalog.json drifting).

    Returns a list of (signal_type, signal_value) pairs — nothing is written here.
    """
    if median_price is None:
        import catalog
        median_price = catalog.median_price()

    signals = []

    # Rule 1: which half of the catalog do they shop in?
    # <= median rather than < so the median item itself counts as budget; with an
    # odd-length catalog the median IS a real product's price, and calling that
    # one "premium" would be arbitrary.
    signals.append(("price_tier", "budget" if product["price"] <= median_price else "premium"))

    # Rule 2: what do they keep buying?
    if product.get("category"):
        signals.append(("category_affinity", product["category"]))

    return signals


def record_purchase(customer_id, product, median_price=None):
    """Infer and store signals for one completed purchase.

    Returns the list of updated signal dicts, so the caller can put them in the
    audit trail — "we learned this from that order" should be reconstructable.
    """
    if not customer_id:
        return []

    recorded = []
    for signal_type, signal_value in infer_signals_from_purchase(product, median_price):
        signal = record_signal(customer_id, signal_type, signal_value)
        if signal:
            recorded.append(signal)
    return recorded


def build_boost_map(customer_id, min_confidence=MIN_CONFIDENCE_TO_APPLY):
    """Turn stored signals into the boost map catalog.search_catalog understands.

    Shape: {signal_type: {signal_value: confidence}}. Only confident signals get
    in — see MIN_CONFIDENCE_TO_APPLY. Returns {} when there's nothing to say,
    which makes search behave exactly as it did before this feature existed.
    """
    # get_signals returns strongest first, so for an exclusive type the first
    # value seen is the dominant one.
    boost_map = {}
    exclusive_best = {}
    exclusive_tied = set()

    for signal in get_signals(customer_id, min_confidence=min_confidence):
        signal_type = signal["signal_type"]
        if signal_type not in ("category_affinity", "price_tier"):
            continue  # storable but not actionable — see SIGNAL_TYPES

        if signal_type in EXCLUSIVE_SIGNAL_TYPES:
            best = exclusive_best.get(signal_type)
            if best is None:
                exclusive_best[signal_type] = signal
            elif (best["confidence"], best["count"]) == (signal["confidence"], signal["count"]):
                # A dead heat is not a preference. Someone who buys equally from
                # both halves of the catalog has told us nothing about price.
                exclusive_tied.add(signal_type)
            continue

        boost_map.setdefault(signal_type, {})[signal["signal_value"]] = signal["confidence"]

    for signal_type, signal in exclusive_best.items():
        if signal_type not in exclusive_tied:
            boost_map[signal_type] = {signal["signal_value"]: signal["confidence"]}

    return boost_map


def describe(customer_id):
    """One-line summary of what we think we know. For the CLI and for the audit
    row on a boosted search — a boost nobody can explain is a boost nobody
    should ship.

    Built from the boost map rather than the raw rows on purpose: this string
    has to describe what actually affected the ranking, not everything we've
    ever written down. A signal that lost the exclusive tie-break, or sits below
    the confidence floor, changed nothing and shouldn't be claimed as a reason.
    """
    boost_map = build_boost_map(customer_id)
    if not boost_map:
        return "no confident preference signals yet"

    counts = {
        (s["signal_type"], s["signal_value"]): s["count"]
        for s in get_signals(customer_id)
    }
    parts = []
    for signal_type in sorted(boost_map):
        for value, confidence in sorted(boost_map[signal_type].items()):
            seen = counts.get((signal_type, value), 0)
            parts.append(f"{signal_type}={value} (seen {seen}x, conf {confidence:g})")
    return "; ".join(parts)


if __name__ == "__main__":
    # Smoke test against a throwaway customer id, cleaned up at the end.
    demo_customer = "cust_preferences_selftest"
    clear_signals(demo_customer)

    fake_budget_shoe = {"id": "P001", "price": 500, "category": "footwear"}
    print("median pinned at 899 for this self-test")
    for _ in range(3):
        record_purchase(demo_customer, fake_budget_shoe, median_price=899)

    print("signals :", describe(demo_customer))
    print("boosts  :", build_boost_map(demo_customer))

    clear_signals(demo_customer)
    print("cleaned up.")
