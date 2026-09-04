"""
catalog.py — RAG-based product search over the fitness & athleisure catalog.

Design notes:
- Embeddings are computed once at startup and cached in memory (fine for 35 products;
  no need for a vector DB at this scale).
- Ranking blends semantic similarity with rating, so a well-reviewed product edges out
  a barely-relevant one with the same similarity score (Tier 2 feature: rating-weighted
  ranking, built in from the start since it's nearly free here).
- All money/stock fields stay untouched by RAG — this file only answers "what's relevant,"
  never "is this allowed." That decision belongs to gate.py.
"""

import json
import os
import statistics
from sentence_transformers import SentenceTransformer
import numpy as np

CATALOG_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "catalog.json")

# How much a confident preference signal can add to a match score, per signal.
# Mirrors (and is capped well below) the rating term in the blend: rating spans
# ~0.056 across the catalog, so 0.02 per signal is a tie-breaker, not a re-rank.
# The authoritative discussion of these numbers is in preferences.py; they're
# duplicated as constants here so this module stays importable on its own.
PREFERENCE_WEIGHT_PER_SIGNAL = 0.02
PREFERENCE_MAX_BOOST = 0.04

_model = None
_catalog = None
_embeddings = None


def _load_model():
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def _load_catalog_raw():
    """Load just the JSON catalog — no model, no embeddings. Used by lookups
    that don't need semantic search (get_product_by_id, check_stock, etc.)."""
    global _catalog
    if _catalog is None:
        with open(CATALOG_PATH, "r") as f:
            _catalog = json.load(f)
    return _catalog


def _load_catalog():
    """Load catalog + compute embeddings. Only called by search_catalog,
    since that's the only function that actually needs semantic similarity."""
    global _embeddings
    catalog = _load_catalog_raw()
    if _embeddings is None:
        model = _load_model()
        texts = [
            f"{p['name']}. {p['description']} Tags: {', '.join(p['tags'])}"
            for p in catalog
        ]
        _embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return catalog, _embeddings


def _cosine_sim(query_vec, matrix):
    # matrix rows are already normalized, so dot product == cosine similarity
    return matrix @ query_vec


def median_price():
    """Median catalog price. Used to split the catalog into a budget half and a
    premium half for the price_tier preference signal."""
    catalog = _load_catalog_raw()
    return statistics.median(p["price"] for p in catalog)


def _preference_boost(product, preference_boost):
    """How much preference memory adds to one product's score.

    preference_boost is {signal_type: {signal_value: confidence}} — built by
    preferences.build_boost_map(). Only two signal types do anything:

        category_affinity : the product's own category
        price_tier        : "budget" if the product is at or below the catalog
                            median, "premium" otherwise

    Each matching signal contributes PREFERENCE_WEIGHT_PER_SIGNAL scaled by its
    confidence, and the total is capped at PREFERENCE_MAX_BOOST so no amount of
    accumulated history can outrank a genuinely better match.

    Returns (boost, matched_signals) — the second is for explainability; a
    ranking nudge nobody can account for later has no business being here.
    """
    if not preference_boost:
        return 0.0, []

    matched = []
    boost = 0.0

    category_signals = preference_boost.get("category_affinity") or {}
    confidence = category_signals.get(product["category"])
    if confidence:
        boost += PREFERENCE_WEIGHT_PER_SIGNAL * confidence
        matched.append(f"category_affinity={product['category']}")

    tier_signals = preference_boost.get("price_tier") or {}
    if tier_signals:
        tier = "budget" if product["price"] <= median_price() else "premium"
        confidence = tier_signals.get(tier)
        if confidence:
            boost += PREFERENCE_WEIGHT_PER_SIGNAL * confidence
            matched.append(f"price_tier={tier}")

    return min(boost, PREFERENCE_MAX_BOOST), matched


def search_catalog(query: str, max_price: float = None, category: str = None, top_k: int = 5,
                   preference_boost: dict = None):
    """
    Search the catalog with a natural language query.

    Args:
        query: natural language product description, e.g. "running shoes for women"
        max_price: optional upper price bound
        category: optional category filter (footwear, apparel, gear, supplements, accessories, electronics)
        top_k: number of results to return
        preference_boost: optional {signal_type: {signal_value: confidence}} map
            from preferences.build_boost_map(). Omit it (the default) and scoring
            is bit-for-bit what it was before preference memory existed.

    Returns:
        List of product dicts, each with an added "match_score" field (0-1 range,
        blended semantic similarity + rating), sorted best-first. When a
        preference boost applied, the product also carries "preference_boost"
        and "preference_matched" so the nudge is inspectable.
    """
    catalog, embeddings = _load_catalog()
    model = _load_model()

    query_vec = model.encode(query, convert_to_numpy=True, normalize_embeddings=True)
    sims = _cosine_sim(query_vec, embeddings)

    candidates = []
    for i, product in enumerate(catalog):
        if max_price is not None and product["price"] > max_price:
            continue
        if category is not None and product["category"] != category:
            continue
        semantic_score = float(sims[i])
        rating_score = product["rating"] / 5.0
        # 80% semantic relevance, 20% rating — relevance should dominate,
        # rating just breaks ties and nudges quality upward
        blended = 0.8 * semantic_score + 0.2 * rating_score

        # Preference memory rides on top of that blend rather than inside it, so
        # the relevance/rating weights keep meaning exactly what they meant
        # before, and a customer with no history gets the original score.
        boost, matched = _preference_boost(product, preference_boost)
        scored = {**product, "match_score": round(blended + boost, 4)}
        if boost:
            scored["preference_boost"] = round(boost, 4)
            scored["preference_matched"] = matched
        candidates.append(scored)

    candidates.sort(key=lambda p: p["match_score"], reverse=True)
    return candidates[:top_k]


def get_product_by_id(product_id: str):
    catalog = _load_catalog_raw()
    for p in catalog:
        if p["id"] == product_id:
            return p
    return None


def get_complementary_products(product_id: str):
    """Used by the upsell/cross-sell feature."""
    product = get_product_by_id(product_id)
    if not product or not product.get("complementary_ids"):
        return []
    catalog = _load_catalog_raw()
    comp_ids = set(product["complementary_ids"])
    return [p for p in catalog if p["id"] in comp_ids]


def check_stock(product_id: str, quantity: int = 1):
    """Used by the graceful-failure feature (out of stock mid-conversation)."""
    product = get_product_by_id(product_id)
    if not product:
        return {"available": False, "reason": "product_not_found"}
    if product["stock"] < quantity:
        return {
            "available": False,
            "reason": "insufficient_stock",
            "in_stock": product["stock"],
            "requested": quantity,
        }
    return {"available": True, "in_stock": product["stock"]}


if __name__ == "__main__":
    # Quick manual test
    results = search_catalog("running shoes under 3000 for women", max_price=3000)
    for r in results:
        print(f"{r['match_score']:.3f}  {r['name']}  (₹{r['price']}, rating {r['rating']})")
