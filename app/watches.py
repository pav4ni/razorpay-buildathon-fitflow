"""
watches.py — price-drop watches: "don't buy it now, tell me if it gets cheaper".

Why this exists: the budget-negotiation path in agent.py can end in a genuine
dead end — nothing fits, and the customer doesn't want the discount that would
make it fit. Today that conversation just stops. A watch turns it into a
deferred sale instead of a lost one, and it costs the customer nothing.

Design notes, in the same spirit as audit.py:
- Same SQLite file as the audit log (data/audit.db). One database file to carry
  around for a demo, and the watch rows sit next to the audit rows that explain
  why each one was created.
- The table is mutable, unlike audit_log — a watch has a lifecycle
  (active -> triggered / cancelled). The audit trail is what stays append-only:
  every status change here also writes an audit row.
- There is NO background scheduler. A real deployment would run the check on a
  cron or a queue; here check_watches() is a plain function called by
  app/check_price_watches.py, which Pavani runs by hand. The evaluation logic is
  the part worth building and testing; the trigger mechanism is infrastructure.
- No notification transport either. When a watch trips we log it and print a
  clear "this would notify the customer" line. Email/SMS is out of scope and
  pretending otherwise would just be a stub with a nicer name.
"""

import os
import sqlite3
from datetime import datetime, timezone

import audit

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit.db")

STATUS_ACTIVE = "active"
STATUS_TRIGGERED = "triggered"
STATUS_CANCELLED = "cancelled"


def _connect():
    """Open a connection and make sure the table exists — same per-call pattern
    as audit._connect, so there's no separate init step to forget before a demo."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS price_watches (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            created_at        TEXT NOT NULL,
            session_id        TEXT NOT NULL,
            customer_id       TEXT,
            product_id        TEXT NOT NULL,
            product_name      TEXT,
            price_at_creation REAL NOT NULL,
            target_price      REAL,
            status            TEXT NOT NULL,
            triggered_at      TEXT,
            triggered_price   REAL
        )
        """
    )
    # "which watches are still live" is the only hot read — the check script does
    # it on every run.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_watch_status ON price_watches (status, id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_watch_customer ON price_watches (customer_id, product_id)"
    )
    conn.commit()
    return conn


def _now():
    return datetime.now(timezone.utc).isoformat()


def create_watch(session_id, product_id, price_at_creation, target_price=None,
                 customer_id=None, product_name=None):
    """Record a watch. target_price=None means "any drop from today's price".

    price_at_creation is stored because it is the baseline the "any drop" case is
    measured against — without it, "cheaper" has nothing to be cheaper *than*.
    It is passed in (from the catalog, by the caller) rather than looked up here
    so this module never has to import the catalog on the write path.

    Returns the created watch as a dict.
    """
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            INSERT INTO price_watches (
                created_at, session_id, customer_id, product_id, product_name,
                price_at_creation, target_price, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _now(),
                session_id,
                customer_id,
                product_id,
                product_name,
                price_at_creation,
                target_price,
                STATUS_ACTIVE,
            ),
        )
        conn.commit()
        watch_id = cursor.lastrowid
    finally:
        conn.close()

    return get_watch(watch_id)


def get_watch(watch_id):
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM price_watches WHERE id = ?", (watch_id,)
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def get_active_watches(customer_id=None, product_id=None):
    """Live watches, oldest first. Both filters are optional."""
    query = "SELECT * FROM price_watches WHERE status = ?"
    args = [STATUS_ACTIVE]
    if customer_id is not None:
        query += " AND customer_id = ?"
        args.append(customer_id)
    if product_id is not None:
        query += " AND product_id = ?"
        args.append(product_id)
    query += " ORDER BY id ASC"

    conn = _connect()
    try:
        return [dict(row) for row in conn.execute(query, args).fetchall()]
    finally:
        conn.close()


def get_watches_for_session(session_id):
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM price_watches WHERE session_id = ? ORDER BY id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(row) for row in rows]


def evaluate_watch(watch, current_price):
    """Would this watch fire at current_price? Pure function — no DB, no logging.

    Kept separate from check_watches so the rule can be tested directly, and so
    a caller can ask "what would happen if..." without touching any row.

    Two cases, deliberately both supported:
      - target_price set  -> fires when the price is at or below the target
      - target_price None -> fires on ANY drop below the price when the watch
        was created ("just tell me if it gets cheaper")

    Returns a dict describing the decision, always with the same keys, so the
    check script and the tests can read it uniformly.
    """
    baseline = watch["price_at_creation"]
    target = watch["target_price"]

    if target is not None:
        would_trigger = current_price <= target
        return {
            "watch_id": watch["id"],
            "product_id": watch["product_id"],
            "mode": "target_price",
            "target_price": target,
            "price_at_creation": baseline,
            "current_price": current_price,
            "would_trigger": would_trigger,
            "drop_from_creation": round(baseline - current_price, 2),
            "reason": (
                f"₹{current_price:g} is at or below the ₹{target:g} target"
                if would_trigger
                else f"₹{current_price:g} is still above the ₹{target:g} target"
            ),
        }

    would_trigger = current_price < baseline
    return {
        "watch_id": watch["id"],
        "product_id": watch["product_id"],
        "mode": "any_drop",
        "target_price": None,
        "price_at_creation": baseline,
        "current_price": current_price,
        "would_trigger": would_trigger,
        "drop_from_creation": round(baseline - current_price, 2),
        "reason": (
            f"price dropped from ₹{baseline:g} to ₹{current_price:g}"
            if would_trigger
            else f"price is unchanged at ₹{current_price:g}"
        ),
    }


def mark_triggered(watch_id, current_price):
    """Move a watch to `triggered`. Only ever applied to an active watch, so a
    second check run can't re-fire (and re-log) a watch that already went off."""
    conn = _connect()
    try:
        conn.execute(
            """
            UPDATE price_watches
            SET status = ?, triggered_at = ?, triggered_price = ?
            WHERE id = ? AND status = ?
            """,
            (STATUS_TRIGGERED, _now(), current_price, watch_id, STATUS_ACTIVE),
        )
        conn.commit()
    finally:
        conn.close()
    return get_watch(watch_id)


def cancel_watch(watch_id):
    conn = _connect()
    try:
        conn.execute(
            "UPDATE price_watches SET status = ? WHERE id = ? AND status = ?",
            (STATUS_CANCELLED, watch_id, STATUS_ACTIVE),
        )
        conn.commit()
    finally:
        conn.close()
    return get_watch(watch_id)


def _catalog_price_lookup(product_id):
    """Default price source: today's catalog.

    Imported lazily because catalog.py loads sentence-transformers at import
    time. The check script and the tests only need a price, and shouldn't pay
    for an embedding model to get one.
    """
    import catalog

    product = catalog.get_product_by_id(product_id)
    return product["price"] if product else None


def check_watches(price_lookup=None, mark=True, customer_id=None):
    """Evaluate every active watch against current prices.

    This is the whole "background job", called by hand. Args:
        price_lookup: product_id -> current price (or None if the product is
            gone). Defaults to the catalog. Injectable so a test — or a demo —
            can simulate a price drop without editing catalog.json.
        mark: write the status change and the audit row for triggered watches.
            Pass False for a dry run.
        customer_id: optionally scope the run to one customer.

    Returns a list of evaluation dicts, one per active watch, each with an
    added `triggered` (did we actually act) and `notification` (the message a
    real notifier would send) field.
    """
    lookup = price_lookup or _catalog_price_lookup
    report = []

    for watch in get_active_watches(customer_id=customer_id):
        current_price = lookup(watch["product_id"])

        if current_price is None:
            # The product left the catalog. Not a trigger, and not an error the
            # customer should hear about — record it and move on.
            report.append(
                {
                    "watch_id": watch["id"],
                    "product_id": watch["product_id"],
                    "mode": "target_price" if watch["target_price"] is not None else "any_drop",
                    "target_price": watch["target_price"],
                    "price_at_creation": watch["price_at_creation"],
                    "current_price": None,
                    "would_trigger": False,
                    "drop_from_creation": None,
                    "reason": "product is no longer in the catalog",
                    "triggered": False,
                    "notification": None,
                }
            )
            continue

        evaluation = evaluate_watch(watch, current_price)
        evaluation["product_name"] = watch["product_name"]
        evaluation["session_id"] = watch["session_id"]
        evaluation["customer_id"] = watch["customer_id"]
        evaluation["triggered"] = False
        evaluation["notification"] = None

        if evaluation["would_trigger"]:
            evaluation["notification"] = (
                f"{watch['product_name'] or watch['product_id']} is now "
                f"₹{current_price:g} — down from ₹{watch['price_at_creation']:g}. "
                f"This is where the customer would be notified."
            )
            if mark:
                mark_triggered(watch["id"], current_price)
                evaluation["triggered"] = True
                # Same audit pattern as every other decision in this system: the
                # row is filed against the session that created the watch, so
                # replaying that conversation shows the follow-up too.
                audit.log_event(
                    session_id=watch["session_id"],
                    user_query=None,
                    agent_reasoning=(
                        "Scheduled price-watch check found the watched price condition met; "
                        "in production this is the point where the customer would be notified."
                    ),
                    action_type="price_watch_triggered",
                    action_params={
                        "watch_id": watch["id"],
                        "product_id": watch["product_id"],
                        "target_price": watch["target_price"],
                        "price_at_creation": watch["price_at_creation"],
                    },
                    result={
                        "current_price": current_price,
                        "reason": evaluation["reason"],
                        "notification": evaluation["notification"],
                        "notification_sent": False,
                        "note": "No notification transport in this build — logged only.",
                    },
                    customer_id=watch["customer_id"],
                )

        report.append(evaluation)

    return report


if __name__ == "__main__":
    # Smoke test: create two watches against a fake product, check them under a
    # simulated price drop, and print what would happen. Uses a throwaway
    # session id so it doesn't pollute a demo session's audit trail.
    demo_session = "watches-selftest"

    target_watch = create_watch(
        session_id=demo_session, product_id="P999", price_at_creation=2000,
        target_price=1500, product_name="Fake Test Product",
    )
    any_drop_watch = create_watch(
        session_id=demo_session, product_id="P999", price_at_creation=2000,
        target_price=None, product_name="Fake Test Product",
    )

    print("no drop yet:")
    for row in (target_watch, any_drop_watch):
        print("   ", evaluate_watch(row, 2000)["reason"])

    print("price falls to 1499:")
    for row in (target_watch, any_drop_watch):
        print("   ", evaluate_watch(row, 1499)["reason"])

    cancel_watch(target_watch["id"])
    cancel_watch(any_drop_watch["id"])
    print("self-test watches cancelled.")
