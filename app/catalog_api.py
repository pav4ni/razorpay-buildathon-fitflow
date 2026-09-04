"""
catalog_api.py — the machine-readable storefront.

The premise of agentic commerce is that the buyer might not be a human with a
browser. If another agent is going to transact with this merchant, it needs to
discover what's for sale without scraping HTML or guessing at an API. So: one
GET, no auth, a stable schema, and the checkout contract stated in the response
itself.

Deliberately simple — this demonstrates the discoverability principle, it is not
a full product-feed spec.

Note it calls catalog._load_catalog_raw(), not search_catalog: serving the
manifest must not drag the sentence-transformers model into memory.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import catalog
import gate

SCHEMA_VERSION = "1.0"


def build_manifest():
    """The full catalog as a stable, parseable dict.

    Pure function, no Flask — so it's testable directly and reusable by any
    non-HTTP caller.
    """
    products = [
        {
            "id": p["id"],
            "name": p["name"],
            "category": p["category"],
            "price": p["price"],
            "currency": "INR",
            "stock": p["stock"],
            "in_stock": p["stock"] > 0,
            "rating": p["rating"],
            "num_reviews": p["num_reviews"],
            "description": p["description"],
            "tags": p["tags"],
            "subscription_eligible": p.get("subscription_eligible", False),
        }
        for p in catalog._load_catalog_raw()
    ]

    return {
        "schema_version": SCHEMA_VERSION,
        "merchant": "Razorpay Buildathon — fitness & athleisure store",
        "currency": "INR",
        "product_count": len(products),
        # An agent reading this knows the bounds before it tries to buy, rather
        # than discovering them by getting rejected.
        "purchase_limits": {
            "max_single_item": gate.DEFAULT_MAX_SINGLE_ITEM,
            "max_cart_total_per_session": gate.DEFAULT_MAX_CART_TOTAL,
            "max_discount_percent": gate.DEFAULT_MAX_DISCOUNT_PERCENT,
        },
        "endpoints": {
            "manifest": "/agent-manifest",
            "chat": "/agent/chat",
            "webhook": "/webhook",
        },
        "payment_provider": "razorpay",
        "products": products,
    }


def register_routes(app):
    """Attach the manifest routes to an existing Flask app."""
    from flask import jsonify

    @app.route("/agent-manifest", methods=["GET"])
    @app.route("/catalog.json", methods=["GET"])
    def agent_manifest():
        return jsonify(build_manifest())

    return app


def create_app():
    from flask import Flask

    app = Flask(__name__)
    return register_routes(app)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5002))
    print(f"  catalog manifest: http://127.0.0.1:{port}/agent-manifest")
    create_app().run(port=port, debug=False)
