const rupees = (n) =>
  '₹' + Number(n ?? 0).toLocaleString('en-IN', { maximumFractionDigits: 2 })

export default function ProductCard({ product }) {
  const outOfStock = (product.stock ?? 0) <= 0
  const low = !outOfStock && product.stock <= 5

  return (
    <div className={'card' + (outOfStock ? ' out' : '')}>
      <div className="card-top">
        <span className="card-id">{product.id}</span>
        <span className="card-cat">{product.category}</span>
      </div>

      <h3>{product.name}</h3>
      <p className="card-desc">{product.description}</p>

      <div className="card-meta">
        <span className="price">{rupees(product.price)}</span>
        <span className="rating">
          ★ {product.rating} <em>({product.num_reviews})</em>
        </span>
      </div>

      <div className="card-foot">
        <span className={outOfStock ? 'stock out' : low ? 'stock low' : 'stock'}>
          {outOfStock ? 'Out of stock' : `${product.stock} in stock`}
        </span>
        {product.match_score != null && (
          <span className="score" title="blended semantic + rating match score">
            match {product.match_score.toFixed(3)}
          </span>
        )}
      </div>

      {/* Surfaced because a nudged ranking that nobody can account for has no
          business being in a product recommendation. */}
      {product.preference_boost > 0 && (
        <div className="boost" title={(product.preference_matched || []).join(', ')}>
          +{product.preference_boost.toFixed(3)} from your past purchases
        </div>
      )}
    </div>
  )
}
