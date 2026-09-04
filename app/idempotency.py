"""
FitFlow Idempotency Manager — prevents double-processing of webhooks
and race conditions on order creation.
"""
import os
import sqlite3
from typing import Optional

# Default lives next to the audit log in data/, not in whatever directory the
# process happened to be started from. The old CWD-relative default scattered
# idempotency.db (plus its -shm/-wal sidecars) across the repo root.
DEFAULT_DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "data", "idempotency.db"
)


class IdempotencyManager:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DEFAULT_DB_PATH
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.execute("PRAGMA journal_mode=WAL;")
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS idempotency_keys (
                    event_id TEXT PRIMARY KEY,
                    event_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS processed_orders (
                    order_id TEXT PRIMARY KEY,
                    status TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

    def check_and_claim(self, event_id: str, event_type: str) -> bool:
        """
        Atomic check-and-set.
        Returns True  → first time seeing this event (process it).
        Returns False → duplicate event (skip it).
        """
        with self._get_conn() as conn:
            try:
                cursor = conn.execute(
                    "INSERT OR IGNORE INTO idempotency_keys (event_id, event_type) VALUES (?, ?)",
                    (event_id, event_type),
                )
                return cursor.rowcount == 1
            except sqlite3.Error:
                return False

    def is_order_processed(self, order_id: str) -> bool:
        with self._get_conn() as conn:
            cursor = conn.execute(
                "SELECT 1 FROM processed_orders WHERE order_id = ?", (order_id,)
            )
            return cursor.fetchone() is not None

    def mark_order_processed(self, order_id: str, status: str):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO processed_orders (order_id, status) VALUES (?, ?)",
                (order_id, status),
            )


if __name__ == "__main__":
    m = IdempotencyManager(":memory:")
    print("Claim event_1:", m.check_and_claim("evt_1", "payment.captured"))  # True
    print("Claim event_1 again:", m.check_and_claim("evt_1", "payment.captured"))  # False
    m.mark_order_processed("ord_123", "paid")
    print("Is ord_123 processed?", m.is_order_processed("ord_123"))  # True
