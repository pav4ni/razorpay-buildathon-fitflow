"""
test_stretch.py — Stretch tier tests.

    RAZORPAY_MOCK=1 RAZORPAY_WEBHOOK_SECRET=whsec_test python3 test_stretch.py

Two groups:

  A. OFFLINE (no network, no API keys, no server, no ngrok). Webhook signature
     verification, event dispatch, retry logging, and the catalog manifest. These
     run anywhere, which is the point — the webhook logic must be verifiable
     without a public tunnel.

  B. AGENT-TO-AGENT (needs ANTHROPIC_API_KEY; skipped without it). Runs
     buyer_agent.py end to end against a mock-mode merchant using an in-process
     Flask test client — real model, real gate, real audit writes, no port bound.

SCOPE NOTE — Subscriptions and Invoices are not covered here because they are not
built. Razorpay returns 401 on /v1/plans and /v1/subscriptions for this account
(the Subscriptions add-on isn't activated), so per the build decision those two
features were deferred rather than written blind. The tests that would cover them
(pause/resume authority rules, failed recurring charge retry) are listed as
DEFERRED at the end of the run so they aren't quietly forgotten.

The subscription.charged *webhook* path IS built and IS tested below — receiving
that event needs no add-on, so it's ready the moment Subscriptions is enabled.
"""

import json
import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

# Mock mode must be set before the merchant app builds any orders, so that
# create_order returns a payment_id the buyer can "capture".
os.environ.setdefault("RAZORPAY_MOCK", "1")
TEST_SECRET = os.environ.setdefault("RAZORPAY_WEBHOOK_SECRET", "whsec_test_stretch")

import audit
import buyer_agent
import catalog_api
import webhook_receiver

failures = []
skipped = []

# The audit log is a real, persistent SQLite database — the same one the agent
# writes to. Fixture session ids are therefore scoped to this run, so re-running
# the suite doesn't accumulate rows under a shared id and turn assertions like
# "exactly one payment row was written" into false failures.
RUN_ID = uuid.uuid4().hex[:8]


def fixture_session(name):
    return f"test-stretch-{name}-{RUN_ID}"


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def rows_for(session_id):
    return audit.get_session_log(session_id)


def signed(payload, secret=TEST_SECRET):
    """(raw_bytes, valid_signature) for a payload — signing the exact bytes sent."""
    raw = json.dumps(payload).encode("utf-8")
    return raw, webhook_receiver.compute_signature(raw, secret)


class no_secret_configured:
    """Temporarily remove RAZORPAY_WEBHOOK_SECRET from the environment.

    Passing secret=None to the webhook functions does NOT simulate an
    unconfigured endpoint — None means "fall back to the env var", which is the
    correct production behaviour. To actually exercise the fail-closed path the
    env var has to be genuinely absent, which is what this does.
    """

    def __enter__(self):
        self._saved = os.environ.pop(webhook_receiver.WEBHOOK_SECRET_ENV, None)
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            os.environ[webhook_receiver.WEBHOOK_SECRET_ENV] = self._saved
        return False


# ---------------------------------------------------------------------------
# A1. Signature verification — the valid case AND the invalid cases
# ---------------------------------------------------------------------------

def test_signature_verification():
    header("A1. Webhook signature verification")

    raw = b'{"event":"payment.captured","payload":{}}'
    good = webhook_receiver.compute_signature(raw, TEST_SECRET)

    check(
        "valid signature verifies",
        webhook_receiver.verify_webhook_signature(raw, good, TEST_SECRET),
    )
    check(
        "invalid signature is rejected",
        not webhook_receiver.verify_webhook_signature(raw, "deadbeef" * 8, TEST_SECRET),
    )
    check(
        "signature from the wrong secret is rejected",
        not webhook_receiver.verify_webhook_signature(
            raw, webhook_receiver.compute_signature(raw, "whsec_attacker"), TEST_SECRET
        ),
    )
    check(
        "tampered body invalidates a previously-valid signature",
        not webhook_receiver.verify_webhook_signature(
            raw.replace(b"captured", b"failed  "), good, TEST_SECRET
        ),
        "amount/event tampering is exactly what the signature exists to catch",
    )
    check(
        "missing signature header is rejected",
        not webhook_receiver.verify_webhook_signature(raw, None, TEST_SECRET),
    )
    check(
        "empty signature header is rejected",
        not webhook_receiver.verify_webhook_signature(raw, "", TEST_SECRET),
    )
    with no_secret_configured():
        check(
            "no secret configured means nothing verifies",
            not webhook_receiver.verify_webhook_signature(raw, good, None),
            "even a genuinely-valid signature can't be checked without the secret",
        )
    check(
        "secret=None falls back to the env var when it is set",
        webhook_receiver.verify_webhook_signature(raw, good, None),
        "production callers rely on this fallback",
    )
    check(
        "signature is stable for identical bytes",
        webhook_receiver.compute_signature(raw, TEST_SECRET) == good,
    )
    check(
        "str and bytes bodies sign identically",
        webhook_receiver.compute_signature(raw.decode(), TEST_SECRET) == good,
    )


# ---------------------------------------------------------------------------
# A2. An invalid webhook must not be processed as a payment
# ---------------------------------------------------------------------------

def test_invalid_webhook_is_not_processed():
    header("A2. Rejected webhooks change nothing")

    session_id = fixture_session("forged")
    payload = {
        "entity": "event",
        "event": "payment.captured",
        "contains": ["payment"],
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_FORGED",
                    "amount": 9999900,
                    "currency": "INR",
                    "order_id": "order_FORGED",
                    "notes": {"session_id": session_id},
                }
            }
        },
    }
    raw = json.dumps(payload).encode("utf-8")

    status, body = webhook_receiver.process_webhook(raw, "not-a-real-signature", TEST_SECRET, echo=False)

    check("forged webhook returns 401", status == 401, f"got {status}")
    check("response reports an error", body.get("status") == "error", str(body))
    check(
        "no payment row was written for the forged session",
        rows_for(session_id) == [],
        "a rejected webhook must never land in the log as a payment",
    )

    rejections = audit.get_events_by_type("webhook_rejected", webhook_receiver.UNATTRIBUTED_SESSION)
    check("the rejection itself is audited", len(rejections) > 0, f"{len(rejections)} rejection row(s)")
    if rejections:
        params = rejections[-1]["action_params"]
        check(
            "rejection row records the reason but not the payload body",
            params.get("reason") == "invalid_signature" and "payload" not in params,
            "attacker-controlled bodies aren't persisted as fact",
        )

    # Fail closed when misconfigured. The env var must be genuinely absent here —
    # an endpoint that can't verify must refuse, not process the webhook anyway.
    with no_secret_configured():
        raw_valid, signature_valid = signed(payload)
        status_no_secret, body_no_secret = webhook_receiver.process_webhook(
            raw_valid, signature_valid, echo=False
        )
    check(
        "missing webhook secret fails closed with 503",
        status_no_secret == 503,
        f"got {status_no_secret}",
    )
    check(
        "503 response names the missing env var",
        webhook_receiver.WEBHOOK_SECRET_ENV in (body_no_secret.get("error") or ""),
        body_no_secret.get("error", ""),
    )
    check(
        "an otherwise-valid webhook is still refused when we cannot verify it",
        rows_for(session_id) == [],
        "fail closed, not open",
    )

    # Malformed JSON, correctly signed.
    bad_raw = b"{not json at all"
    status_bad, _ = webhook_receiver.process_webhook(
        bad_raw, webhook_receiver.compute_signature(bad_raw, TEST_SECRET), TEST_SECRET, echo=False
    )
    check("valid signature over malformed JSON returns 400", status_bad == 400, f"got {status_bad}")


# ---------------------------------------------------------------------------
# A3. Event handling — captured, failed (+ retry), subscription, unknown
# ---------------------------------------------------------------------------

def test_payment_captured():
    header("A3. payment.captured is filed against its conversation")

    session_id = fixture_session("captured")
    order = {
        "order_id": "order_TESTCAP",
        "payment_id": "pay_TESTCAP",
        "product_id": "P016",
        "product_name": "WhaySource Whey Protein (1kg, Chocolate)",
        "amount_rupees": 2199,
        "quantity": 1,
    }
    payload = buyer_agent.build_payment_payload(
        "payment.captured", order, session_id, customer_id="cust_TEST"
    )
    raw, signature = signed(payload)

    status, body = webhook_receiver.process_webhook(raw, signature, TEST_SECRET, echo=True)

    check("verified webhook returns 200", status == 200, f"got {status}")
    check("event reported as handled", body.get("handled") is True)
    check(
        "event routed back to the originating session",
        body.get("session_id") == session_id,
        "session_id travels in the order's notes",
    )
    check("audit action is payment_captured", body.get("audit_action") == "payment_captured")

    rows = [r for r in rows_for(session_id) if r["action_type"] == "payment_captured"]
    check("exactly one payment_captured row written", len(rows) == 1, f"{len(rows)} row(s)")
    if rows:
        result = rows[0]["result"]
        check("row records the payment id", result.get("payment_id") == "pay_TESTCAP")
        check("row records the order id", result.get("order_id") == "order_TESTCAP")
        check(
            "amount converted back to rupees correctly",
            result.get("amount_rupees") == 2199,
            f"got {result.get('amount_rupees')} (paise->rupees)",
        )
        check("customer id carried onto the row", rows[0]["customer_id"] == "cust_TEST")


def test_payment_failed_logs_retry():
    header("A4. payment.failed is explained and not silently dropped")

    session_id = fixture_session("failed")
    order = {
        "order_id": "order_TESTFAIL",
        "payment_id": "pay_TESTFAIL",
        "product_id": "P019",
        "product_name": "AminoBoost BCAA Powder (300g)",
        "amount_rupees": 1299,
        "quantity": 1,
    }
    payload = buyer_agent.build_payment_payload("payment.failed", order, session_id)
    raw, signature = signed(payload)

    status, body = webhook_receiver.process_webhook(raw, signature, TEST_SECRET, echo=True)

    check("failed payment webhook accepted (200)", status == 200, f"got {status}")
    check("audit action is payment_failed", body.get("audit_action") == "payment_failed")

    summary = body.get("summary", {})
    check(
        "failure reason captured in plain language",
        "declined" in (summary.get("error_description") or "").lower(),
        summary.get("error_description", ""),
    )

    retry = summary.get("retry") or {}
    check("retry disposition present", bool(retry), str(retry))
    check("a declined card is marked retryable", retry.get("retryable") is True)
    check(
        "retry row states the next step rather than giving up",
        "re-offer" in (retry.get("next_step") or "").lower(),
        retry.get("next_step", ""),
    )

    rows = [r for r in rows_for(session_id) if r["action_type"] == "payment_failed"]
    check("failure written to the audit trail", len(rows) == 1, f"{len(rows)} row(s)")
    if rows:
        check(
            "retry decision is persisted, not just returned",
            bool((rows[0]["result"].get("retry") or {}).get("next_step")),
        )


def test_subscription_charged():
    header("A5. subscription.charged (webhook path ready ahead of the API)")

    session_id = fixture_session("subscription")
    payload = {
        "entity": "event",
        "event": "subscription.charged",
        "contains": ["subscription", "payment"],
        "payload": {
            "subscription": {
                "entity": {
                    "id": "sub_TEST123",
                    "status": "active",
                    "paid_count": 3,
                    "total_count": 12,
                    "notes": {"session_id": session_id, "product_id": "P016"},
                }
            },
            "payment": {
                "entity": {
                    "id": "pay_SUBCYCLE3",
                    "amount": 219900,
                    "currency": "INR",
                    "notes": {"session_id": session_id},
                }
            },
        },
    }
    raw, signature = signed(payload)
    status, body = webhook_receiver.process_webhook(raw, signature, TEST_SECRET, echo=True)

    check("subscription.charged accepted", status == 200, f"got {status}")
    check("treated as a handled event", body.get("handled") is True)
    check("audit action is subscription_charged", body.get("audit_action") == "subscription_charged")

    summary = body.get("summary", {})
    check("subscription id captured", summary.get("subscription_id") == "sub_TEST123")
    check("billing cycle captured", summary.get("paid_count") == 3 and summary.get("total_count") == 12,
          f"cycle {summary.get('paid_count')}/{summary.get('total_count')}")
    check("cycle amount in rupees", summary.get("amount_rupees") == 2199)

    rows = [r for r in rows_for(session_id) if r["action_type"] == "subscription_charged"]
    check("cycle written to the audit trail", len(rows) == 1, f"{len(rows)} row(s)")


def test_unknown_event():
    header("A6. Unknown events are acknowledged, not crashed on")

    payload = {"entity": "event", "event": "refund.speed_changed", "payload": {}}
    raw, signature = signed(payload)
    status, body = webhook_receiver.process_webhook(raw, signature, TEST_SECRET, echo=False)

    check("unknown event still returns 200", status == 200, f"got {status}")
    check("marked as unhandled", body.get("handled") is False)
    check("logged as webhook_unhandled", body.get("audit_action") == "webhook_unhandled")
    check(
        "unattributable event filed under the fallback session",
        body.get("session_id") == webhook_receiver.UNATTRIBUTED_SESSION,
    )


# ---------------------------------------------------------------------------
# A7. Machine-readable catalog
# ---------------------------------------------------------------------------

def test_catalog_manifest():
    header("A7. Machine-readable catalog manifest")

    manifest = catalog_api.build_manifest()

    check("schema_version present", manifest.get("schema_version") == "1.0")
    check("product_count matches products length",
          manifest["product_count"] == len(manifest["products"]),
          f"{manifest['product_count']} products")
    check("catalog is non-empty", manifest["product_count"] == 35, f"{manifest['product_count']}")

    required = {"id", "name", "category", "price", "stock", "currency", "in_stock"}
    missing = [p.get("id") for p in manifest["products"] if not required.issubset(p)]
    check("every product carries the required schema fields", not missing, f"missing on {missing[:3]}")

    check(
        "purchase limits are advertised so a buyer agent knows the bounds up front",
        manifest["purchase_limits"]["max_single_item"] == 6000
        and manifest["purchase_limits"]["max_cart_total_per_session"] == 10000,
        str(manifest["purchase_limits"]),
    )
    check("endpoints advertised", set(manifest["endpoints"]) == {"manifest", "chat", "webhook"})
    check(
        "in_stock is consistent with stock",
        all(p["in_stock"] == (p["stock"] > 0) for p in manifest["products"]),
    )
    check("manifest is JSON-serializable", bool(json.dumps(manifest)))

    # And over HTTP.
    app = catalog_api.create_app()
    client = app.test_client()
    for route in ("/agent-manifest", "/catalog.json"):
        response = client.get(route)
        check(f"GET {route} returns 200", response.status_code == 200, f"got {response.status_code}")
        check(f"GET {route} returns the catalog",
              response.get_json().get("product_count") == manifest["product_count"])


# ---------------------------------------------------------------------------
# B. Agent-to-agent — buyer_agent against a mock-mode merchant, in process
# ---------------------------------------------------------------------------

def test_buyer_agent_end_to_end():
    header("B1. Buyer agent -> merchant agent -> webhook (no human, no server)")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("    [SKIP] ANTHROPIC_API_KEY not set — skipping the live agent-to-agent run.")
        skipped.append("buyer agent end-to-end (no ANTHROPIC_API_KEY)")
        return

    import razorpay_client

    check("running in mock mode", razorpay_client.is_mock_mode(),
          "required: real orders have no payment_id to capture")

    transport = buyer_agent.InProcessTransport()

    status, health = transport.get("/health")
    check("merchant /health responds", status == 200, str(health))
    check("merchant reports mock mode", health.get("mock_mode") is True)
    check("merchant reports webhook secret configured", health.get("webhook_secret_configured") is True)

    status, manifest = transport.get("/agent-manifest")
    check("merchant serves the manifest on the same port", status == 200)
    check("manifest reachable by the buyer", manifest.get("product_count") == 35)

    buyer = buyer_agent.BuyerAgent(transport=transport, webhook_secret=TEST_SECRET, verbose=True)

    # Scenario 1 (successful capture) and scenario 3 (declined card). Two is
    # enough to prove both webhook paths without three full LLM conversations.
    results = buyer.run_all([buyer_agent.SCENARIOS[0], buyer_agent.SCENARIOS[2]])

    for result in results:
        name = result["scenario"]
        check(f"[{name}] conversation completed", result.get("ok") is True, str(result.get("error", "")))

        orders = result.get("orders") or []
        check(f"[{name}] merchant created an order", len(orders) >= 1, f"{len(orders)} order(s)")
        if not orders:
            continue

        order = orders[-1]
        check(f"[{name}] order is a mock order", str(order.get("order_id", "")).startswith("order_MOCK"),
              order.get("order_id"))
        check(f"[{name}] order carries a payment id to capture", bool(order.get("payment_id")))

        webhook = result.get("webhook") or {}
        check(f"[{name}] webhook was accepted", webhook.get("status") == 200, str(webhook.get("status")))
        check(f"[{name}] webhook was verified and handled",
              (webhook.get("body") or {}).get("handled") is True)

        # The audit trail must show the whole story in one session.
        session_id = result["session_id"]
        types = [r["action_type"] for r in rows_for(session_id)]
        check(f"[{name}] audit shows the gate ran", "gate_check" in types, str(types))
        check(f"[{name}] audit shows the order", "order_created" in types, str(types))

        expected_event = webhook["event"].replace("payment.", "payment_")
        check(f"[{name}] audit shows the async payment event ({expected_event})",
              expected_event in types,
              "order and its payment land in the same session trail")

    # The declined-card scenario must have recorded a retry decision.
    declined = results[-1]
    if declined.get("session_id"):
        failed_rows = [r for r in rows_for(declined["session_id"]) if r["action_type"] == "payment_failed"]
        check("declined payment logged a retry decision",
              bool(failed_rows) and bool((failed_rows[0]["result"].get("retry") or {}).get("next_step")),
              "the agent knows to re-offer rather than dropping the sale")


# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("  STRETCH TIER TESTS")
    print("=" * 70)
    print(f"  RAZORPAY_MOCK           : {os.environ.get('RAZORPAY_MOCK')}")
    print(f"  RAZORPAY_WEBHOOK_SECRET : {'set' if TEST_SECRET else 'NOT SET'}")
    print(f"  ANTHROPIC_API_KEY       : {'set' if os.environ.get('ANTHROPIC_API_KEY') else 'NOT SET (group B skips)'}")

    test_signature_verification()
    test_invalid_webhook_is_not_processed()
    test_payment_captured()
    test_payment_failed_logs_retry()
    test_subscription_charged()
    test_unknown_event()
    test_catalog_manifest()
    test_buyer_agent_end_to_end()

    header("DEFERRED — not built this session, so not tested")
    print("  Razorpay returns 401 on /v1/plans and /v1/subscriptions for this account,")
    print("  so Subscriptions + Invoices were deferred rather than written unverified.")
    print("  Still owed once the add-on is activated:")
    print("    - subscription create -> pause -> resume")
    print("    - pause/resume authority (business-paused vs customer-paused)")
    print("    - failed recurring charge with retry logging at the subscription level")
    print("    - whether invoices auto-generate per billing cycle")

    header("RESULT")
    if skipped:
        print(f"  {len(skipped)} group(s) skipped:")
        for name in skipped:
            print(f"    - {name}")
    if failures:
        print(f"\n  {len(failures)} assertion(s) FAILED:")
        for name in failures:
            print(f"    - {name}")
        return 1
    print("\n  All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
