import { useState } from 'react'

// Colour/label per audit action_type. Anything unmapped still renders, just
// without a colour — a new action type must never break the panel.
const KIND = {
  search: ['search', 'k-blue'],
  stock_check: ['stock', 'k-grey'],
  gate_check: ['gate ok', 'k-green'],
  rejection: ['rejected', 'k-red'],
  order_created: ['order', 'k-gold'],
  refund_created: ['refund', 'k-purple'],
  discount_computed: ['discount', 'k-teal'],
  upsell_offered: ['upsell', 'k-teal'],
  upsell_accepted: ['upsell ✓', 'k-green'],
  upsell_declined: ['upsell ✗', 'k-grey'],
  customer_linked: ['customer', 'k-grey'],
  past_orders_lookup: ['history', 'k-grey'],
  payment_captured: ['paid', 'k-green'],
  payment_failed: ['payment failed', 'k-red'],
  price_watch_created: ['watch', 'k-teal'],
  price_watch_triggered: ['watch fired', 'k-gold'],
  preference_signal: ['learned', 'k-purple'],
  webhook_rejected: ['webhook denied', 'k-red'],
  webhook_unhandled: ['webhook', 'k-grey'],
  error: ['error', 'k-red'],
}

function time(ts) {
  try {
    return new Date(ts).toLocaleTimeString('en-GB', { hour12: false })
  } catch {
    return ts
  }
}

function Event({ event }) {
  const [open, setOpen] = useState(false)
  const [label, cls] = KIND[event.action_type] || [event.action_type, 'k-grey']

  return (
    <li className="audit-event">
      <div className="ae-head" onClick={() => setOpen((v) => !v)}>
        <span className="ae-id">#{event.id}</span>
        <span className={`ae-kind ${cls}`}>{label}</span>
        <span className="ae-time">{time(event.timestamp)}</span>
        {event.gate_decision && (
          <span className={'ae-gate ' + (event.gate_decision === 'allowed' ? 'ok' : 'no')}>
            {event.gate_decision}
          </span>
        )}
        <span className="ae-chev">{open ? '▾' : '▸'}</span>
      </div>

      {event.agent_reasoning && <p className="ae-why">{event.agent_reasoning}</p>}

      {open && (
        <div className="ae-detail">
          {event.user_query && (
            <>
              <span className="ae-key">user said</span>
              <div className="ae-quote">{event.user_query}</div>
            </>
          )}
          <span className="ae-key">params</span>
          <pre>{JSON.stringify(event.action_params, null, 2)}</pre>
          <span className="ae-key">result</span>
          <pre>{JSON.stringify(event.result, null, 2)}</pre>
        </div>
      )}
    </li>
  )
}

export default function AuditPanel({ events, loading, sessionId, onRefresh }) {
  return (
    <aside className="audit">
      <div className="audit-head">
        <div>
          <h2>Audit trail</h2>
          <p>
            {events.length} event{events.length === 1 ? '' : 's'} ·{' '}
            <code>{sessionId}</code>
          </p>
        </div>
        <button className="ghost small" onClick={onRefresh} disabled={loading}>
          {loading ? '…' : 'Refresh'}
        </button>
      </div>

      <p className="audit-note">
        Every money-touching decision writes one append-only row, including the
        agent's own stated reason for it. Click a row for the full params and result.
      </p>

      {events.length === 0 && !loading && (
        <p className="audit-empty">
          No events yet. Ask the agent for something and they'll appear here.
        </p>
      )}

      <ul className="audit-list">
        {events.map((e) => (
          <Event key={e.id} event={e} />
        ))}
      </ul>
    </aside>
  )
}
