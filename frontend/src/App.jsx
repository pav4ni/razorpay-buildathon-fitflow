import { useCallback, useEffect, useRef, useState } from 'react'
import AuditPanel from './AuditPanel.jsx'
import ProductCard from './ProductCard.jsx'
import { openCheckout } from './checkout.js'

/** Minimal **bold** renderer — the agent's replies use it and nothing else. */
function renderText(text) {
  if (!text) return null
  return String(text)
    .split(/(\*\*[^*]+\*\*)/g)
    .map((part, i) =>
      part.startsWith('**') && part.endsWith('**')
        ? <strong key={i}>{part.slice(2, -2)}</strong>
        : <span key={i}>{part}</span>,
    )
}

const rupees = (n) =>
  '₹' + Number(n ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })

const newSessionId = () =>
  'sess-' + Math.random().toString(36).slice(2, 10) + Date.now().toString(36).slice(-4)

const GREETING = {
  role: 'assistant',
  content:
    "Hi — I'm your shopping assistant for fitness and athleisure gear. " +
    'Tell me what you\'re after and I\'ll find it. Try **"I need running shoes under 3000"**.',
}

export default function App() {
  const [sessionId, setSessionId] = useState(newSessionId)
  const [messages, setMessages] = useState([GREETING])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  const [cartTotal, setCartTotal] = useState(0)
  const [headroom, setHeadroom] = useState(null)
  const [orders, setOrders] = useState([])

  const [config, setConfig] = useState(null)
  const [health, setHealth] = useState(null)

  const [showAudit, setShowAudit] = useState(false)
  const [auditEvents, setAuditEvents] = useState([])
  const [auditLoading, setAuditLoading] = useState(false)

  const [payingOrderId, setPayingOrderId] = useState(null)
  const [banner, setBanner] = useState(null)

  const endRef = useRef(null)
  const inputRef = useRef(null)

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, loading])

  // One-time boot: the public Razorpay key id for Checkout, and a health probe
  // so a bad API key shows up as a banner instead of a mystery failure later.
  useEffect(() => {
    fetch('/api/config').then((r) => r.json()).then(setConfig).catch(() => {})
    fetch('/api/health')
      .then((r) => r.json())
      .then((h) => {
        setHealth(h)
        if (!h.anthropic_key_present) {
          setBanner({
            kind: 'error',
            text: 'ANTHROPIC_API_KEY is not set on the server — chat will fail until it is.',
          })
        }
      })
      .catch(() =>
        setBanner({ kind: 'error', text: 'Cannot reach the backend on port 5050.' }),
      )
  }, [])

  const refreshAudit = useCallback(async (sid) => {
    setAuditLoading(true)
    try {
      const res = await fetch(`/api/audit/${encodeURIComponent(sid)}`)
      const data = await res.json()
      setAuditEvents(data.events || [])
    } catch {
      setAuditEvents([])
    } finally {
      setAuditLoading(false)
    }
  }, [])

  // Keep the panel live while it's open, so a purchase made mid-demo shows up
  // in the trail without anyone having to close and reopen it.
  useEffect(() => {
    if (showAudit) refreshAudit(sessionId)
  }, [showAudit, sessionId, messages, refreshAudit])

  const applyState = (data) => {
    if (typeof data.cart_total === 'number') setCartTotal(data.cart_total)
    if (typeof data.cart_headroom === 'number') setHeadroom(data.cart_headroom)
    if (Array.isArray(data.orders)) setOrders(data.orders)
  }

  const sendMessage = async (text) => {
    const msg = (text ?? input).trim()
    if (!msg || loading) return

    setInput('')
    setBanner(null)
    setMessages((prev) => [...prev, { role: 'user', content: msg }])
    setLoading(true)

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, message: msg }),
      })
      const data = await res.json()

      if (!res.ok) {
        setMessages((prev) => [
          ...prev,
          { role: 'error', content: data.error || `Request failed (${res.status}).` },
        ])
        return
      }

      if (data.session_id && data.session_id !== sessionId) setSessionId(data.session_id)
      applyState(data)

      setMessages((prev) => [
        ...prev,
        {
          role: 'assistant',
          content: data.reply,
          products: data.products || [],
          negotiation: data.budget_negotiation || null,
          // Only denials are surfaced as their own block; an approval is
          // already implied by the order that follows it.
          rejections: (data.gate_events || []).filter((g) => !g.allowed),
          newOrders: data.new_orders || [],
          auditCount: data.audit_events_this_turn,
        },
      ])
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: 'error', content: `Could not reach the server: ${err.message}` },
      ])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }

  const payForOrder = async (order) => {
    if (!config?.razorpay_key_id) {
      setBanner({ kind: 'error', text: 'No Razorpay key id configured on the server.' })
      return
    }
    setPayingOrderId(order.order_id)
    setBanner(null)
    try {
      const result = await openCheckout({
        keyId: config.razorpay_key_id,
        order,
        sessionId,
      })
      applyState(result)
      setBanner({
        kind: 'success',
        text:
          `Payment ${result.payment_id} captured for ${order.order_id}` +
          (result.signature_verified ? ' (signature verified).' : '.') +
          ' You can now ask the agent to refund it.',
      })
      refreshAudit(sessionId)
    } catch (err) {
      if (err.dismissed) {
        setBanner({ kind: 'info', text: 'Checkout closed — no payment was made.' })
      } else {
        setBanner({ kind: 'error', text: `Payment failed: ${err.message}` })
      }
    } finally {
      setPayingOrderId(null)
    }
  }

  const resetSession = () => {
    const sid = newSessionId()
    setSessionId(sid)
    setMessages([GREETING])
    setCartTotal(0)
    setHeadroom(null)
    setOrders([])
    setAuditEvents([])
    setBanner(null)
  }

  const unpaidOrders = orders.filter((o) => !o.payment_id)
  // Orders in the transcript are frozen at the moment they were created, so a
  // payment made afterwards has to be reflected from live session state —
  // otherwise the "Complete payment" button on an old message stays clickable
  // and reopens Checkout for an order that is already paid.
  const paidOrderIds = new Set(orders.filter((o) => o.payment_id).map((o) => o.order_id))
  const limits = config?.limits

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand">
          <span className="logo">◈</span>
          <div>
            <h1>FitFlow</h1>
            <p>conversational checkout · fitness &amp; athleisure</p>
          </div>
        </div>

        <div className="topbar-right">
          <div className="cart">
            <span className="cart-label">Cart total</span>
            <span className="cart-value">{rupees(cartTotal)}</span>
            {headroom !== null && (
              <span className="cart-sub">{rupees(headroom)} left this session</span>
            )}
          </div>
          <button
            className={'ghost' + (showAudit ? ' active' : '')}
            onClick={() => setShowAudit((v) => !v)}
          >
            {showAudit ? 'Hide' : 'Show'} audit trail
            {auditEvents.length > 0 && <span className="pill">{auditEvents.length}</span>}
          </button>
          <button className="ghost" onClick={resetSession} title="Start a fresh session">
            New session
          </button>
        </div>
      </header>

      {banner && (
        <div className={`banner banner-${banner.kind}`}>
          <span>{banner.text}</span>
          <button onClick={() => setBanner(null)}>×</button>
        </div>
      )}

      <div className={'body' + (showAudit ? ' with-audit' : '')}>
        <main className="chat">
          <div className="messages">
            {messages.map((m, i) => (
              <Message
                key={i}
                message={m}
                onPay={payForOrder}
                payingOrderId={payingOrderId}
                paidOrderIds={paidOrderIds}
              />
            ))}

            {loading && (
              <div className="msg assistant">
                <div className="bubble thinking">
                  <span className="dot" /><span className="dot" /><span className="dot" />
                </div>
              </div>
            )}
            <div ref={endRef} />
          </div>

          {unpaidOrders.length > 0 && (
            <div className="pending-pay">
              {unpaidOrders.map((o) => (
                <button
                  key={o.order_id}
                  className="pay"
                  disabled={payingOrderId === o.order_id}
                  onClick={() => payForOrder(o)}
                >
                  {payingOrderId === o.order_id
                    ? 'Opening Razorpay…'
                    : `Pay ${rupees(o.amount_rupees)} for ${o.product_name}`}
                </button>
              ))}
            </div>
          )}

          <form
            className="composer"
            onSubmit={(e) => {
              e.preventDefault()
              sendMessage()
            }}
          >
            <input
              ref={inputRef}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask for something — “I need running shoes under 3000”"
              disabled={loading}
              autoFocus
            />
            <button type="submit" disabled={loading || !input.trim()}>
              Send
            </button>
          </form>

          <footer className="statusline">
            <span>session <code>{sessionId}</code></span>
            {limits && (
              <span>
                limits: {rupees(limits.max_single_item)}/item ·{' '}
                {rupees(limits.max_cart_total)}/session · {limits.max_discount_percent}% max discount
              </span>
            )}
            <span className={config?.razorpay_mock_mode ? 'warn' : 'ok'}>
              razorpay: {config?.razorpay_mock_mode ? 'MOCK' : 'live test mode'}
            </span>
            {health?.model && <span>model: <code>{health.model}</code></span>}
          </footer>
        </main>

        {showAudit && (
          <AuditPanel
            events={auditEvents}
            loading={auditLoading}
            sessionId={sessionId}
            onRefresh={() => refreshAudit(sessionId)}
          />
        )}
      </div>
    </div>
  )
}

function Message({ message, onPay, payingOrderId, paidOrderIds }) {
  const { role } = message

  if (role === 'user') {
    return (
      <div className="msg user">
        <div className="bubble">{message.content}</div>
      </div>
    )
  }

  if (role === 'error') {
    return (
      <div className="msg assistant">
        <div className="bubble error">{message.content}</div>
      </div>
    )
  }

  return (
    <div className="msg assistant">
      <div className="bubble">{renderText(message.content)}</div>

      {message.rejections?.length > 0 &&
        message.rejections.map((r) => (
          <div className="gate-reject" key={r.audit_id}>
            <div className="gate-head">
              <span className="gate-badge">GATE · DENIED</span>
              <code>{r.reason}</code>
            </div>
            <p>{r.explanation}</p>
            {r.amount_checked != null && (
              <p className="gate-meta">
                checked {rupees(r.amount_checked)}
                {r.product_name ? ` for ${r.product_name}` : ''}
                {r.cart_total_so_far != null
                  ? ` · cart was ${rupees(r.cart_total_so_far)}`
                  : ''}
              </p>
            )}
          </div>
        ))}

      {message.products?.length > 0 && (
        <div className="cards">
          {message.products.map((p) => (
            <ProductCard key={p.id} product={p} />
          ))}
        </div>
      )}

      {message.negotiation?.closest_option && (
        <div className="negotiation">
          <span className="neg-badge">OVER BUDGET</span>
          <p>
            Closest match <strong>{message.negotiation.closest_option.name}</strong> is{' '}
            {rupees(message.negotiation.closest_option.over_budget_by)} over the{' '}
            {rupees(message.negotiation.stated_budget)} budget
            {message.negotiation.closest_option.discount_would_be_enough
              ? ` — a ${message.negotiation.closest_option.minimum_discount_needed_percent}% discount would close it.`
              : ' — too wide for any permitted discount.'}
          </p>
        </div>
      )}

      {message.newOrders?.map((o) => (
        <div className="order-card" key={o.order_id}>
          <div className="order-head">
            <span className="order-badge">ORDER CREATED</span>
            {o.mock && <span className="mock-badge">MOCK</span>}
          </div>
          <div className="order-id">{o.order_id}</div>
          <div className="order-line">
            {o.product_name} × {o.quantity} — <strong>{rupees(o.amount_rupees)}</strong>
            {o.discount_percent > 0 && (
              <span className="saving">
                {' '}
                (−{o.discount_percent}%, saved {rupees(o.discount_savings)})
              </span>
            )}
          </div>
          {paidOrderIds?.has(o.order_id) ? (
            <div className="paid-note">✓ Paid — you can now ask for a refund.</div>
          ) : (
            <button
              className="pay small"
              disabled={payingOrderId === o.order_id}
              onClick={() => onPay(o)}
            >
              {payingOrderId === o.order_id ? 'Opening Razorpay…' : 'Complete payment'}
            </button>
          )}
        </div>
      ))}
    </div>
  )
}
