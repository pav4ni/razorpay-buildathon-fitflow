/**
 * checkout.js — Razorpay Checkout, and the verification round trip after it.
 *
 * The important part is what happens *after* Razorpay's handler fires. The
 * browser is handed a payment_id, but the browser is not trusted with it: it
 * goes straight to POST /api/payment/verify, which recomputes the
 * HMAC-SHA256 of "<order_id>|<payment_id>" with the key secret it holds
 * server-side. Only if that matches is the payment attached to the order.
 *
 * That attachment is what makes a real refund demoable — a real Razorpay order
 * is created with no payment_id, and the agent correctly refuses to refund an
 * order it can't find a payment for.
 */

export function openCheckout({ keyId, order, sessionId }) {
  return new Promise((resolve, reject) => {
    if (typeof window.Razorpay !== 'function') {
      reject(new Error('Razorpay Checkout script did not load. Check your connection.'))
      return
    }

    let settled = false

    const rzp = new window.Razorpay({
      key: keyId,
      // Amount and currency are display values here — the real charge is
      // whatever the server-created order says it is.
      amount: Math.round(Number(order.amount_rupees) * 100),
      currency: 'INR',
      name: 'FitFlow',
      description: `${order.product_name} × ${order.quantity ?? 1}`,
      order_id: order.order_id,
      prefill: {
        name: 'Demo User',
        email: 'demo.user@example.com',
        contact: '+919876543210',
      },
      theme: { color: '#4f8cff' },

      handler: async (response) => {
        settled = true
        try {
          const res = await fetch('/api/payment/verify', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              session_id: sessionId,
              razorpay_order_id: response.razorpay_order_id,
              razorpay_payment_id: response.razorpay_payment_id,
              razorpay_signature: response.razorpay_signature,
            }),
          })
          const data = await res.json()
          if (!res.ok || !data.success) {
            reject(new Error(data.error || 'Server rejected the payment.'))
            return
          }
          resolve(data)
        } catch (err) {
          reject(err)
        }
      },

      modal: {
        ondismiss: () => {
          if (settled) return
          const err = new Error('Checkout dismissed')
          err.dismissed = true
          reject(err)
        },
      },
    })

    rzp.on('payment.failed', (response) => {
      settled = true
      const desc = response?.error?.description || 'Payment failed at the gateway.'
      reject(new Error(desc))
    })

    rzp.open()
  })
}
