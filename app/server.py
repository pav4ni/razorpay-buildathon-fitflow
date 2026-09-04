"""
server.py — the single Flask process behind the FitFlow demo.

There is exactly one API layer in this project and this is it. It replaces two
near-duplicate wrappers (api.py and merchant_api.py) that an earlier rewrite of
the interface layer left behind.

What it serves, all on one port:

    GET  /                      the built React UI (frontend/dist)
    GET  /api/health            liveness + a preflight report on the API keys
    GET  /api/config            the *public* Razorpay key id, for Checkout.js
    POST /api/chat              one conversational turn through agent.run_turn
    GET  /api/session/<id>      cart total, orders, customer for a session
    GET  /api/audit/<id>        the session's audit trail
    POST /api/payment/verify    Checkout.js callback — verify + attach payment_id
    POST /webhook               Razorpay webhooks (from webhook_receiver)
    GET  /agent-manifest        machine-readable catalog (from catalog_api)
    POST /agent/chat            alias of /api/chat, for buyer_agent.py
    GET  /health                alias of /api/health, for buyer_agent.py

DESIGN NOTE — the UI reads the audit log, not the agent.

The one thing worth defending here: /api/chat does not reach inside the agent to
find out what happened during a turn. It records the highest audit row id before
the turn, runs the turn, and then reads back every row written since. Product
cards, gate rejections and orders are all derived from those rows.

That means the UI is a *consumer* of the audit trail rather than a second,
parallel account of events — if the panel shows a product card, an audit row
exists that proves the search happened. It also keeps agent.py untouched: no
tool handler, no gate call and no session field had to change to make the
interface work.
"""

import hashlib
import hmac
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

import agent
import audit
import catalog_api
import gate
import razorpay_client
import webhook_receiver

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(REPO_ROOT, "frontend", "dist")
ENV_FILE = os.path.join(REPO_ROOT, ".env")


# Credentials where .env is authoritative and a pre-existing environment
# variable is NOT allowed to win.
#
# This is deliberately the opposite of the usual "environment beats dotfile"
# rule, and it is worth explaining. A stale `export RAZORPAY_KEY_ID=...` in a
# shell profile silently shadows the project's own .env in every terminal, and
# the failure it produces — "Authentication failed" from Razorpay — looks like
# a broken integration rather than a shell config problem. The project's .env
# is the file someone edits when they mean to change this project's keys, so
# that is the file that wins. Any override is reported at startup rather than
# applied quietly.
ENV_FILE_AUTHORITATIVE = (
    "ANTHROPIC_API_KEY",
    "RAZORPAY_KEY_ID",
    "RAZORPAY_KEY_SECRET",
    "RAZORPAY_WEBHOOK_SECRET",
)


def load_env_file(path=ENV_FILE):
    """Read .env into os.environ, without adding a dependency.

    python-dotenv is one more thing to install and one more thing to have not
    installed; this is a few dozen lines and does what this project needs.

    Precedence, and the reasoning is in ENV_FILE_AUTHORITATIVE above:
      - credentials      -> .env wins over the ambient environment
      - everything else  -> the ambient environment wins, so
                            `RAZORPAY_MOCK=1 python3 app/server.py` and the test
                            suites can still override behaviour flags.

    Values are stripped: a trailing space on an API key becomes an invalid HTTP
    header, which surfaces as a baffling 401 rather than as a config error.

    Returns (loaded, overrides) — `overrides` names any credential where .env
    disagreed with an already-set environment variable.
    """
    if not os.path.isfile(path):
        return [], []

    loaded, overrides = [], []
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'").strip()
            if not key:
                continue

            current = os.environ.get(key)
            if current is None:
                os.environ[key] = value
                loaded.append(key)
            elif key in ENV_FILE_AUTHORITATIVE and current != value:
                os.environ[key] = value
                loaded.append(key)
                overrides.append(key)

    return loaded, overrides

# Load .env at import time so every entry point — python3 app/server.py, the
# test suites, and buyer_agent's in-process transport — sees the same config.
_LOADED_ENV_KEYS, _ENV_OVERRIDES = load_env_file()

# session_id -> ShoppingSession. In-memory on purpose: this is a demo, and a
# restart should give a clean cart. The audit log on disk is the durable record.
_sessions = {}


# ---------------------------------------------------------------------------
# Session handling
# ---------------------------------------------------------------------------

def _get_or_create_session(session_id):
    """Fetch the live session, creating and customer-linking it on first use."""
    existing = _sessions.get(session_id)
    if existing is not None:
        return existing

    session = agent.ShoppingSession(session_id=session_id)
    # Attaches a Razorpay Customer so orders and cross-session history hang off
    # a real customer id. Never fatal — link_customer already logs its own
    # failure and leaves customer_id as None.
    agent.link_customer(session)
    _sessions[session_id] = session
    return session


def _last_audit_id(session_id):
    """Highest audit row id for this session, or 0 if it has no rows yet.

    max() rather than the last element: get_session_log orders by
    (timestamp, id), so the final row is not guaranteed to hold the highest id
    if two writes land in the same ISO timestamp. Getting this wrong would
    replay already-seen rows into the next turn's digest.
    """
    events = audit.get_session_log(session_id)
    return max((e["id"] for e in events), default=0)


def _session_state(session):
    """The machine-readable half of a chat response.

    `orders` is returned as-is from session.orders because buyer_agent.py reads
    order_id / product_name / amount_rupees / payment_id straight off it.
    """
    return {
        "session_id": session.session_id,
        "customer_id": session.customer_id,
        "cart_total": session.cart_total_so_far,
        "cart_headroom": max(0, gate.DEFAULT_MAX_CART_TOTAL - session.cart_total_so_far),
        "orders": session.orders,
    }


# ---------------------------------------------------------------------------
# Reading a turn back out of the audit log
# ---------------------------------------------------------------------------

def _turn_digest(session_id, since_id):
    """Everything the UI needs about one turn, read from rows written during it.

    Returns products (from the last search), gate decisions (so a rejection can
    be shown as a distinct event rather than buried in prose), and any orders
    created.
    """
    events = [e for e in audit.get_session_log(session_id) if e["id"] > since_id]

    products = []
    negotiation = None
    gate_events = []
    new_orders = []
    discount_offers = []

    for event in events:
        result = event["result"] if isinstance(event["result"], dict) else {}
        params = event["action_params"] if isinstance(event["action_params"], dict) else {}
        action = event["action_type"]

        # The most recent search in the turn is the one whose results are on
        # screen — an agent that searches twice is refining, not accumulating.
        if action == "search" and result.get("products"):
            products = result["products"]
            negotiation = result.get("budget_negotiation")

        # Both halves of the gate's verdict are surfaced. A denial is the whole
        # point of the safety story, so it gets a first-class field rather than
        # only existing inside the model's reply text.
        #
        # Two things this must NOT do, both learned the hard way:
        #   - it must not treat every row carrying a gate_decision as a gate
        #     event. An `order_created` row is stamped gate_decision="allowed"
        #     (the gate did approve it), but its result is the Razorpay order,
        #     which has no `allowed` key — reading one off it yields False and
        #     renders a successful purchase as a red DENIED box.
        #   - it must therefore take the verdict from gate_decision, which the
        #     audit layer sets deliberately, rather than inferring it from the
        #     shape of a result dict that differs per tool.
        is_gate_check = params.get("tool") == "check_gate"
        is_denial = event["gate_decision"] == "denied"
        if is_gate_check or is_denial:
            gate_events.append({
                "audit_id": event["id"],
                "allowed": event["gate_decision"] == "allowed",
                "decision": event["gate_decision"],
                "reason": result.get("reason") or result.get("error_type"),
                "explanation": result.get("explanation") or result.get("error"),
                "product_id": result.get("product_id") or params.get("product_id"),
                "product_name": result.get("product_name"),
                "amount_checked": result.get("amount_checked"),
                "cart_total_so_far": result.get("cart_total_so_far"),
            })

        # The merchant's negotiating position, made machine-readable.
        #
        # Without this a buyer agent would have to parse a price out of English
        # prose, which is exactly the kind of brittleness that makes an
        # agent-to-agent demo fall over on stage. The number is taken from the
        # discount_computed audit row — the same row that records why the
        # merchant priced it that way — so the offer and its justification can
        # never disagree.
        if action == "discount_computed" and result.get("product_id"):
            discount_offers.append({
                "product_id": result.get("product_id"),
                "product_name": result.get("product_name"),
                "sticker_price": result.get("sticker_price"),
                "discount_percent": result.get("discount_percent"),
                "price": result.get("discounted_price"),
                "sufficient": result.get("sufficient"),
                # "discount_cap" or "margin_floor" — which ceiling stopped the
                # merchant going lower. Note this carries no cost figure; the
                # redaction in agent.py applies to this path too.
                "capped_by": result.get("capped_by"),
            })

        if action == "order_created":
            new_orders.append({
                "order_id": result.get("id"),
                "product_id": (result.get("notes") or {}).get("product_id"),
                "product_name": result.get("product_name"),
                "amount_rupees": result.get("amount_rupees"),
                "sticker_amount": result.get("sticker_amount"),
                "discount_percent": result.get("discount_percent") or 0,
                "discount_savings": result.get("discount_savings") or 0,
                "quantity": result.get("quantity_ordered"),
                "currency": result.get("currency", "INR"),
                "amount_paise": result.get("amount"),
                "mock": bool(result.get("mock")),
            })

    return {
        "products": products,
        "budget_negotiation": negotiation,
        "gate_events": gate_events,
        "new_orders": new_orders,
        "discount_offers": discount_offers,
        "audit_events_this_turn": len(events),
    }


# ---------------------------------------------------------------------------
# Preflight — say loudly at boot whether the keys actually work
# ---------------------------------------------------------------------------

def preflight():
    """Report on credentials without making the process fail to start.

    A demo that boots and then 401s on the first message is much harder to
    diagnose under time pressure than one that says what's wrong at startup.
    """
    anthropic_key = (os.environ.get("ANTHROPIC_API_KEY") or "").strip()
    mock_mode = razorpay_client.is_mock_mode()
    webhook_secret = bool(webhook_receiver.get_webhook_secret())

    report = {
        "anthropic_key_present": bool(anthropic_key),
        "razorpay_credentials_present": razorpay_client.credentials_available(),
        "razorpay_mock_mode": mock_mode,
        "razorpay_key_id": os.environ.get("RAZORPAY_KEY_ID") or None,
        "webhook_secret_present": webhook_secret,
        # The original /health contract, which test_stretch.py asserts on and
        # which an earlier rewrite dropped. Kept as the names the buyer agent
        # and the suite already expect rather than renaming the test to fit a
        # newer spelling.
        "mock_mode": mock_mode,
        "webhook_secret_configured": webhook_secret,
        "frontend_built": os.path.exists(os.path.join(DIST_DIR, "index.html")),
        "model": agent.MODEL,
        "limits": {
            "max_single_item": gate.DEFAULT_MAX_SINGLE_ITEM,
            "max_cart_total": gate.DEFAULT_MAX_CART_TOTAL,
            "max_discount_percent": gate.DEFAULT_MAX_DISCOUNT_PERCENT,
        },
    }
    return report


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

def create_app():
    # static_folder=None because the SPA routes below do the serving; leaving
    # Flask's own /static handler registered would shadow them.
    app = Flask(__name__, static_folder=None)
    CORS(app)

    webhook_receiver.register_routes(app)   # POST /webhook
    catalog_api.register_routes(app)        # GET /agent-manifest, /catalog.json

    # ---- health & config ----

    @app.route("/api/health", methods=["GET"])
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", **preflight()})

    @app.route("/api/config", methods=["GET"])
    def config():
        """Public config for the browser.

        Only the Razorpay *key id* is exposed. It is a publishable value —
        Checkout.js needs it client-side — and the key SECRET never leaves the
        server.
        """
        return jsonify({
            "razorpay_key_id": os.environ.get("RAZORPAY_KEY_ID") or None,
            "razorpay_mock_mode": razorpay_client.is_mock_mode(),
            "limits": preflight()["limits"],
        })

    # ---- chat ----

    @app.route("/api/chat", methods=["POST"])
    @app.route("/agent/chat", methods=["POST"])
    def chat():
        data = request.get_json(force=True, silent=True) or {}
        message = (data.get("message") or "").strip()
        session_id = data.get("session_id") or None

        if not message:
            return jsonify({"error": "message is required"}), 400

        # buyer_agent.py opens a conversation with session_id=None and adopts
        # whatever id comes back, so mint one rather than rejecting the call.
        session = _get_or_create_session(session_id or agent.ShoppingSession().session_id)

        since_id = _last_audit_id(session.session_id)
        reply = agent.run_turn(session, message)
        digest = _turn_digest(session.session_id, since_id)

        return jsonify({
            "reply": reply,
            **_session_state(session),
            **digest,
        })

    # ---- session & audit ----

    @app.route("/api/session/<session_id>", methods=["GET"])
    def session_state(session_id):
        session = _sessions.get(session_id)
        if session is None:
            return jsonify({"session_id": session_id, "known": False,
                            "cart_total": 0, "orders": []})
        return jsonify({"known": True, **_session_state(session)})

    @app.route("/api/audit/<session_id>", methods=["GET"])
    def audit_trail(session_id):
        events = audit.get_session_log(session_id)
        return jsonify({
            "session_id": session_id,
            "count": len(events),
            "events": events,
        })

    # ---- Checkout.js callback ----

    @app.route("/api/payment/verify", methods=["POST"])
    def verify_payment():
        """Verify a completed Razorpay Checkout payment and record it.

        Razorpay's Checkout handler hands the browser three values. The
        signature is HMAC-SHA256 of "<order_id>|<payment_id>" keyed on the API
        key secret, so a browser cannot forge one — which is exactly why the
        payment_id is only trusted after this check passes.

        On success the payment_id is attached to the matching order in the live
        session. That is what makes a *real* refund demonstrable: create_order
        in real mode returns an order with no payment_id (capture happens inside
        Checkout, not here), and agent.refund_order correctly refuses to refund
        an order it can't find a payment for.
        """
        data = request.get_json(force=True, silent=True) or {}
        session_id = data.get("session_id")
        order_id = data.get("razorpay_order_id")
        payment_id = data.get("razorpay_payment_id")
        signature = data.get("razorpay_signature")

        if not all([session_id, order_id, payment_id]):
            return jsonify({
                "success": False,
                "error": "session_id, razorpay_order_id and razorpay_payment_id are required",
            }), 400

        session = _sessions.get(session_id)
        if session is None:
            return jsonify({"success": False, "error": f"Unknown session {session_id}"}), 404

        verified = False
        verification_note = None

        if razorpay_client.is_mock_mode():
            # Nothing real was signed, so there is nothing real to verify.
            # Recorded explicitly rather than quietly treated as verified.
            verification_note = "RAZORPAY_MOCK is set — signature not checked."
        else:
            secret = os.environ.get(razorpay_client.KEY_SECRET_ENV) or ""
            if not secret:
                return jsonify({"success": False,
                                "error": "RAZORPAY_KEY_SECRET is not configured."}), 503
            expected = hmac.new(
                secret.encode("utf-8"),
                f"{order_id}|{payment_id}".encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            # compare_digest, not ==, for the same timing reason as the webhook.
            verified = hmac.compare_digest(expected, str(signature or ""))
            if not verified:
                audit.log_event(
                    session_id=session_id,
                    user_query=None,
                    agent_reasoning="Checkout reported a payment but the signature did not match. Refused.",
                    action_type="webhook_rejected",
                    action_params={"source": "checkout_handler", "order_id": order_id},
                    result={"accepted": False, "reason": "invalid_signature"},
                    customer_id=session.customer_id,
                )
                return jsonify({"success": False,
                                "error": "Payment signature verification failed."}), 401

        # Attach the payment to the order it paid for.
        matched = None
        for order in session.orders:
            if order.get("order_id") == order_id:
                order["payment_id"] = payment_id
                matched = order
                break

        audit.log_event(
            session_id=session_id,
            user_query=None,
            agent_reasoning=(
                "Customer completed payment through Razorpay Checkout; attaching the "
                "payment id to the order so a refund can be issued against it."
            ),
            action_type="payment_captured",
            action_params={
                "source": "checkout_handler",
                "order_id": order_id,
                "payment_id": payment_id,
            },
            result={
                "payment_id": payment_id,
                "order_id": order_id,
                "signature_verified": verified,
                "note": verification_note,
                "product_name": (matched or {}).get("product_name"),
                "amount_rupees": (matched or {}).get("amount_rupees"),
                "matched_session_order": matched is not None,
            },
            customer_id=session.customer_id,
        )

        return jsonify({
            "success": True,
            "signature_verified": verified,
            "note": verification_note,
            "payment_id": payment_id,
            "order_id": order_id,
            "matched_session_order": matched is not None,
            **_session_state(session),
        })

    # ---- the built React app ----
    # Registered last so the API rules above always win the routing match.

    @app.route("/", defaults={"path": ""})
    @app.route("/<path:path>")
    def spa(path):
        if path.startswith(("api/", "agent/", "webhook")):
            return jsonify({"error": f"No such endpoint: /{path}"}), 404

        if path and os.path.isfile(os.path.join(DIST_DIR, path)):
            return send_from_directory(DIST_DIR, path)

        index = os.path.join(DIST_DIR, "index.html")
        if os.path.isfile(index):
            return send_from_directory(DIST_DIR, "index.html")

        return jsonify({
            "error": "The frontend has not been built yet.",
            "fix": "cd frontend && npm install && npm run build",
        }), 503

    return app


def main():
    # catalog.py's cosine-similarity matmul emits harmless RuntimeWarnings on
    # macOS/Accelerate. Silence them so the server log stays readable.
    import numpy as np
    np.seterr(all="ignore")

    port = int(os.environ.get("PORT", 5050))
    report = preflight()

    print("=" * 70)
    print("  FitFlow — conversational checkout agent")
    print("=" * 70)
    print(f"  url          : http://127.0.0.1:{port}")
    print(f"  .env         : {'loaded ' + ', '.join(_LOADED_ENV_KEYS) if _LOADED_ENV_KEYS else 'not loaded (using the ambient environment)'}")
    for name in _ENV_OVERRIDES:
        print(f"                 NOTE: .env's {name} overrode a different value "
              f"already exported in your shell (check ~/.zshrc).")
    print(f"  model        : {report['model']}")
    print(f"  limits       : Rs.{report['limits']['max_single_item']} per item, "
          f"Rs.{report['limits']['max_cart_total']} per session, "
          f"{report['limits']['max_discount_percent']}% max discount")
    print(f"  anthropic key: {'set' if report['anthropic_key_present'] else 'MISSING'}")
    if report["razorpay_mock_mode"]:
        print("  razorpay     : MOCK MODE (RAZORPAY_MOCK=1) — orders and refunds are stubs")
    elif report["razorpay_credentials_present"]:
        print(f"  razorpay     : LIVE test-mode keys ({report['razorpay_key_id']})")
    else:
        print("  razorpay     : NO CREDENTIALS — order creation will fail gracefully")
    print(f"  webhook sec  : {'set' if report['webhook_secret_present'] else 'not set'}")
    if not report["frontend_built"]:
        print("\n  WARNING: frontend/dist is missing. Build it with:")
        print("      cd frontend && npm install && npm run build")
    if not report["anthropic_key_present"]:
        print("\n  WARNING: ANTHROPIC_API_KEY is not set — every message will fail.")
    print("=" * 70)
    print(flush=True)

    # threaded=True so fetching the audit panel doesn't block behind a long
    # agent turn.
    create_app().run(host="127.0.0.1", port=port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
