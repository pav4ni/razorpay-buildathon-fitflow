"""
audit.py — the audit trail. Every decision the agent makes gets one row here.

Why this exists: an agent that spends money needs to be *accountable*, not just
correct. If a judge (or a customer, or a compliance team) asks "why did the agent
buy that?", the answer has to be reconstructable from a log, not from re-running
the model and hoping it behaves the same way.

Design notes:
- SQLite, single file on disk. No server, no extra dependency (sqlite3 ships with
  Python), and the file is inspectable with any SQLite browser during a demo.
- One row per *action*, not per conversation turn. A single user message can
  produce a search, a stock check, a gate check and an order — that's four rows.
- Writes are append-only. Nothing in this module updates or deletes rows.
- `agent_reasoning` is captured from the model at the moment it requests a tool,
  so the log records *why* it wanted the action, not just that the action happened.
"""

import json
import os
import sqlite3
from datetime import datetime, timezone

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "audit.db")

# The action_type values the agent actually emits. Kept as a plain tuple for
# documentation — it's intentionally not enforced as a DB constraint, so adding a
# new action type later doesn't require a migration.
ACTION_TYPES = (
    "search",              # catalog search
    "stock_check",         # availability lookup
    "gate_check",          # safety gate evaluated
    "order_created",       # money actually moved (Razorpay order created)
    "rejection",           # an action was refused, by the gate or by this layer
    "complementary_lookup",  # cross-sell suggestion lookup
    "error",               # a tool blew up; recorded rather than swallowed
    # --- Tier 2 ---
    "discount_computed",   # budget gap evaluated, discount proposed
    "upsell_offered",      # a complementary product was put in front of the customer
    "upsell_accepted",     # they bought the thing we suggested
    "upsell_declined",     # they didn't — recorded so attach rate has a denominator
    "refund_created",      # money moved back
    "customer_linked",     # a Razorpay Customer record was attached to this session
    "past_orders_lookup",  # purchase history read for a returning customer
    # --- Stretch: inbound webhook events ---
    # These are the only rows not written by the agent's own tool loop — they
    # arrive from Razorpay after the fact and are filed against the session that
    # created the order, via session_id in the order's notes.
    "payment_captured",           # money actually arrived
    "payment_failed",             # capture failed; row carries the retry disposition
    "subscription_charged",       # a recurring cycle billed successfully
    "webhook_unhandled",          # verified webhook for an event we don't act on
    "webhook_rejected",           # signature invalid or no secret configured
    # --- Stretch: price watches and preference memory ---
    "price_watch_created",        # customer asked to be told if a price drops
    "price_watch_triggered",      # a manual check found the condition met
    "preference_signal",          # something was inferred about this customer
    # --- margin-aware discounting ---
    "margin_protection",          # a discount was capped by the margin floor rather
                                  # than by the policy ceiling; row carries the
                                  # revenue that decision protected
    # --- two-sided negotiation ---
    # These are the only rows written by the BUYER rather than the merchant.
    # They share the merchant's session_id on purpose, so one /audit read
    # replays both sides of the exchange interleaved.
    "negotiation_opened",         # buyer stated an (understated) opening budget
    "buyer_counter_offer",        # buyer conceded and restated a higher budget
    "merchant_counter_offer",     # merchant's best price at a given round
    "negotiation_accepted",       # buyer took the deal
    "negotiation_walked_away",    # buyer left; the merchant held its line
)


def _connect():
    """Open a connection and make sure the table exists.

    Cheap enough to do per call at this scale, and it means no separate
    'initialize the database' step to forget before a demo.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS audit_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            session_id      TEXT NOT NULL,
            user_query      TEXT,
            agent_reasoning TEXT,
            action_type     TEXT NOT NULL,
            action_params   TEXT,
            result          TEXT,
            gate_decision   TEXT
        )
        """
    )
    # Sessions are the main read pattern (replay one conversation), so index them.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log (session_id, id)"
    )

    # Tier 2 added customer_id. SQLite has no ADD COLUMN IF NOT EXISTS, so check
    # the existing schema and migrate in place — this keeps audit databases
    # written by the Core-tier build readable instead of forcing a wipe.
    existing_columns = {row["name"] for row in conn.execute("PRAGMA table_info(audit_log)")}
    if "customer_id" not in existing_columns:
        conn.execute("ALTER TABLE audit_log ADD COLUMN customer_id TEXT")

    # "What has this customer bought before?" is the second read pattern.
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_audit_customer ON audit_log (customer_id, action_type)"
    )
    conn.commit()
    return conn


def _to_json(value):
    """Serialize params/results for storage.

    default=str keeps this from ever raising on an unexpected object — an audit
    log that crashes the agent because it couldn't serialize a field would be a
    worse failure than a slightly lossy log entry.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, default=str)


def log_event(session_id, user_query, agent_reasoning, action_type,
              action_params, result, gate_decision=None, customer_id=None):
    """Write one row to the audit log.

    Args:
        session_id: groups all actions from one conversation
        user_query: what the user said that led to this action (may be None for
            follow-up actions the agent chained on its own)
        agent_reasoning: short text — the agent's stated reason for this step
        action_type: one of ACTION_TYPES
        action_params: dict of the inputs to the action (JSON-serialized here)
        result: dict/list of the outcome (JSON-serialized here)
        gate_decision: "allowed" / "denied", or None when the gate wasn't involved
        customer_id: Razorpay Customer id, when the session is linked to one.
            Lets purchase history be reconstructed across sessions from the audit
            log alone, with no extra Razorpay call.

    Returns:
        The id of the row just written.
    """
    timestamp = datetime.now(timezone.utc).isoformat()
    conn = _connect()
    try:
        cursor = conn.execute(
            """
            INSERT INTO audit_log (
                timestamp, session_id, user_query, agent_reasoning,
                action_type, action_params, result, gate_decision, customer_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                timestamp,
                session_id,
                user_query,
                agent_reasoning,
                action_type,
                _to_json(action_params),
                _to_json(result),
                gate_decision,
                customer_id,
            ),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_session_log(session_id):
    """Return every row for one session, oldest first.

    Ordered by (timestamp, id) — id breaks ties when two actions land in the same
    ISO timestamp, which happens easily with parallel tool calls.

    Returns:
        List of dicts. action_params and result are parsed back into Python
        objects so callers don't each have to json.loads them.
    """
    conn = _connect()
    try:
        rows = conn.execute(
            "SELECT * FROM audit_log WHERE session_id = ? ORDER BY timestamp ASC, id ASC",
            (session_id,),
        ).fetchall()
    finally:
        conn.close()

    return _rows_to_events(rows)


def _rows_to_events(rows):
    """Turn sqlite rows into dicts with action_params/result parsed back to objects."""
    events = []
    for row in rows:
        event = dict(row)
        for field in ("action_params", "result"):
            if event.get(field):
                try:
                    event[field] = json.loads(event[field])
                except (ValueError, TypeError):
                    pass  # leave it as the raw string if it wasn't JSON
        events.append(event)
    return events


def get_events_by_type(action_type, session_id=None):
    """Every event of one action_type, optionally scoped to a session.

    Used by metrics.py to count upsell offers and acceptances across all
    sessions, which is what makes attach rate computable from the log alone.
    """
    query = "SELECT * FROM audit_log WHERE action_type = ?"
    args = [action_type]
    if session_id is not None:
        query += " AND session_id = ?"
        args.append(session_id)
    query += " ORDER BY timestamp ASC, id ASC"

    conn = _connect()
    try:
        return _rows_to_events(conn.execute(query, args).fetchall())
    finally:
        conn.close()


def get_customer_orders(customer_id, limit=10):
    """Past successful orders for one customer, newest first.

    This is the "you bought X last week, want to reorder?" lookup. It reads our
    own audit log rather than calling Razorpay — the log already has everything
    needed (product, amount, when), and it keeps a conversational nicety off the
    critical path of the payments API.
    """
    if not customer_id:
        return []
    conn = _connect()
    try:
        rows = conn.execute(
            """
            SELECT * FROM audit_log
            WHERE customer_id = ? AND action_type = 'order_created'
            ORDER BY timestamp DESC, id DESC
            LIMIT ?
            """,
            (customer_id, limit),
        ).fetchall()
    finally:
        conn.close()

    orders = []
    for event in _rows_to_events(rows):
        result = event["result"] if isinstance(event["result"], dict) else {}
        orders.append(
            {
                "order_id": result.get("id"),
                "product_id": result.get("notes", {}).get("product_id")
                if isinstance(result.get("notes"), dict) else None,
                "product_name": result.get("product_name"),
                "amount_rupees": result.get("amount_rupees"),
                "quantity": result.get("quantity_ordered"),
                "payment_id": result.get("payment_id"),
                "timestamp": event["timestamp"],
                "session_id": event["session_id"],
            }
        )
    return orders


def format_session_log(session_id):
    """Human-readable dump of a session — used by the CLI's /audit command and
    handy for showing the trail to a judge without opening a SQLite browser."""
    events = get_session_log(session_id)
    if not events:
        return f"(no audit events for session {session_id})"

    lines = [f"=== Audit trail for session {session_id} ({len(events)} events) ==="]
    for event in events:
        gate = f" [gate: {event['gate_decision']}]" if event["gate_decision"] else ""
        lines.append(f"\n#{event['id']}  {event['timestamp']}  {event['action_type']}{gate}")
        if event["user_query"]:
            lines.append(f"    user said : {event['user_query']}")
        if event["agent_reasoning"]:
            lines.append(f"    reasoning : {event['agent_reasoning']}")
        lines.append(f"    params    : {json.dumps(event['action_params'], default=str)}")
        lines.append(f"    result    : {json.dumps(event['result'], default=str)}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Standalone smoke test: write three fake events for one session, read them back.
    demo_session = "demo-session-001"

    log_event(
        session_id=demo_session,
        user_query="I need running shoes under 3000",
        agent_reasoning="User gave a clear product intent and a budget, so search the catalog directly.",
        action_type="search",
        action_params={"query": "running shoes", "max_price": 3000},
        result={"num_results": 2, "top_result": "CloudRunner Mesh Running Shoes"},
    )

    log_event(
        session_id=demo_session,
        user_query=None,
        agent_reasoning="User picked P001. Must clear the safety gate before creating any order.",
        action_type="gate_check",
        action_params={"amount": 2799, "cart_total_so_far": 0, "item_in_stock": True},
        result={"allowed": True, "reason": None, "explanation": "Within all bounds — approved."},
        gate_decision="allowed",
    )

    log_event(
        session_id=demo_session,
        user_query=None,
        agent_reasoning="Gate approved, so create the Razorpay order for the confirmed item.",
        action_type="order_created",
        action_params={"amount_in_rupees": 2799, "product_id": "P001"},
        result={"id": "order_FAKE123", "amount": 279900, "currency": "INR", "status": "created"},
        gate_decision="allowed",
    )

    print(format_session_log(demo_session))
    print(f"\nDatabase file: {os.path.abspath(DB_PATH)}")
