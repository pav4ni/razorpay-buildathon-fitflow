# FitFlow — conversational checkout agent

A shopping agent for a fitness & athleisure store. You describe what you want in
plain language; it searches the catalog, negotiates on budget, and completes a
real Razorpay purchase — with **every money-moving decision bounded by a safety
gate and written to an append-only audit trail**.

Razorpay AI Buildathon entry (AI Growth & Agentic Commerce).

---

## Run it

Prerequisites: **Python 3.9+** and **Node 18+** (both already present on macOS
if you've run this before). From a completely fresh terminal:

```bash
cd ~/razorpay-buildathon-fitflow
git pull

# 1. Python deps (one time). The venv is already built; this just tops it up.
venv/bin/pip install -r requirements.txt

# 2. Frontend deps + build (one time, ~30s)
cd frontend && npm install && npm run build && cd ..

# 3. Start everything — one process, one port
./run.sh
```

Then open **<http://127.0.0.1:5050>**.

That's it. Flask serves both the API and the built React UI from the same
origin, so there is no second server and no CORS to configure.

### Configuration

Copy `.env.example` to `.env` and fill it in. `.env` is gitignored.

| Variable | What it does |
|---|---|
| `ANTHROPIC_API_KEY` | Powers the agent's tool-use loop. Required. |
| `RAZORPAY_KEY_ID` / `RAZORPAY_KEY_SECRET` | Test-mode keys. The key id is public (the browser needs it for Checkout); the secret never leaves the server. |
| `RAZORPAY_MOCK` | `1` stubs every Razorpay call (offline demo). `0` creates **real** test-mode orders you can pay with a test card. |
| `RAZORPAY_WEBHOOK_SECRET` | Only needed if you expose `/webhook` to Razorpay via ngrok. |
| `PORT` | Defaults to 5050. Not 5000 — macOS AirPlay Receiver holds that port and replies 403. |

> **Credential precedence.** `.env` wins over your shell environment for the four
> credential variables above. This is deliberate: a stale
> `export RAZORPAY_KEY_ID=...` in `~/.zshrc` otherwise silently shadows this
> project's keys in every terminal, and the resulting "Authentication failed"
> looks like a broken integration rather than a shell-config problem. The server
> prints a NOTE at startup whenever it overrides one.

### Completing a test payment

With `RAZORPAY_MOCK=0` the agent creates a genuine Razorpay order, but a real
order has **no `payment_id`** until someone actually pays — capture happens
inside Razorpay Checkout, not in this app. To produce one:

1. Buy something in the chat. An **ORDER CREATED** card appears.
2. Click **Complete payment**. Razorpay Checkout opens.
3. Pay with test card `4111 1111 1111 1111`, any future expiry, any CVV, OTP `1234`.
4. The browser posts the result to `/api/payment/verify`, which recomputes the
   HMAC server-side and only then attaches the `payment_id` to the order.
5. Now ask the agent to **refund** it — it has a real payment to refund against.

---

## Demo script (5 minutes)

| Say this | What to point at |
|---|---|
| "hey, I want to get into yoga" | It asks **one** clarifying question, not five. |
| "just starting out, budget under 1500 for a mat" | Product cards, with match scores. |
| "the ZenFlow one, order it" | Real `order_...` id. Cart total updates. |
| "also add three of the IronCore dumbbell sets" | **GATE · DENIED** — `exceeds_single_item_limit` at ₹14,997. The agent explains the limit and offers a way forward. |
| *Toggle* **Show audit trail** | Every step above, with the agent's own stated reason for it. |

---

## Architecture

```
                    ┌─────────────────────────────────┐
   browser ───────► │  app/server.py   (Flask)        │
   React SPA        │  · serves frontend/dist         │
                    │  · /api/chat  /api/audit        │
                    │  · /api/payment/verify          │
                    │  · /webhook   /agent-manifest   │
                    └───────────────┬─────────────────┘
                                    │
                    ┌───────────────▼─────────────────┐
                    │  app/agent.py — LLM + tool use  │
                    │  the LLM proposes …             │
                    └───────────────┬─────────────────┘
                                    │
              ┌─────────────────────┼─────────────────────┐
              ▼                     ▼                     ▼
      app/gate.py           app/catalog.py       app/razorpay_client.py
      … the gate disposes   prices & stock       rupees → paise, never raises
              │                     │                     │
              └─────────────────────┼─────────────────────┘
                                    ▼
                            app/audit.py
                    one append-only row per decision
```

### The safety design

Three things are enforced in Python at the tool-execution layer, not requested
in a prompt — because a prompt is a request and code is a rule:

1. **No `amount` parameter exists on any tool.** The model passes a
   `product_id` and a quantity; the price is looked up from the catalog. It
   cannot understate a price to slip an item past the gate.
2. **The cart total is server-side.** It lives on `ShoppingSession`, not in the
   model's message — which is what defeats the "split one big purchase into
   several small ones" exploit.
3. **No order without a fresh, matching gate approval.** `create_order` refuses
   unless `check_gate` approved that exact product and quantity in this session,
   and each approval is consumed on use. A later denial revokes a stale approval.

Bounds live in `app/gate.py`: ₹6,000 per item, ₹10,000 per session, 20% max
discount.

### The audit trail

`app/audit.py` writes one append-only SQLite row per action — including the
model's own stated reasoning, captured at the moment it requests the tool.

The UI is a *consumer* of that log, not a parallel account of events:
`/api/chat` records the highest audit row id before the turn and reads back
every row written since. So if a product card is on screen, an audit row exists
that proves the search happened.

---

## Tests

```bash
set -a; source .env; set +a
export RAZORPAY_MOCK=1 RAZORPAY_WEBHOOK_SECRET=whsec_test_local

venv/bin/python3 test_watches_prefs.py   # 92 assertions — fully offline
venv/bin/python3 test_stretch.py         # 84 — webhooks, manifest, buyer agent
venv/bin/python3 test_e2e.py             # 24 — core tier, real model calls
venv/bin/python3 test_tier2.py           # 123 — discounts, upsell, refunds
```

`RAZORPAY_MOCK=1` is required: mock orders come back with a `pay_MOCK...` id, so
the refund and payment-capture paths are exercisable offline.

## Other entry points

```bash
venv/bin/python3 app/agent.py                          # CLI agent (/audit /cart /metrics)
venv/bin/python3 app/buyer_agent.py                    # autonomous buyer, no human
venv/bin/python3 app/check_price_watches.py --simulate-drop 20
venv/bin/python3 -c "import sys;sys.path.insert(0,'app');import metrics;print(metrics.format_summary())"
```

## Known limitations

Stated plainly rather than hidden:

- **No authentication.** Every session resolves to one hardcoded `DEMO_CUSTOMER`,
  so purchase history and preference memory accumulate across all demo runs.
- **No scheduler.** Price watches are checked by running
  `app/check_price_watches.py` by hand. The evaluation rule is tested; the cron
  entry is not built.
- **No notification transport.** A triggered watch logs and prints; the audit row
  says `notification_sent: false`.
- **Subscriptions and Invoices are not built.** Razorpay returns 401 on
  `/v1/plans` for this account (add-on not activated), so they were deferred
  rather than written blind. The `subscription.charged` *webhook* path is built
  and tested — receiving it needs no add-on.
- **Cross-session refunds of real payments.** A `payment_id` attached via
  Checkout lives on the in-memory session. Refund it in the same session; a
  server restart loses the linkage.
