"""
check_price_watches.py — the "background job", run by hand.

    python3 app/check_price_watches.py                 # check against real catalog prices
    python3 app/check_price_watches.py --dry-run       # evaluate, change nothing
    python3 app/check_price_watches.py --simulate-drop 20
                                                       # pretend every price fell 20%

There is no scheduler in this build and that is a deliberate scope decision, not
an oversight: a cron entry or a queue worker is infrastructure, and the part
worth demonstrating is the evaluation rule and the audit trail it writes. This
script is what a scheduler would call once an hour.

--simulate-drop exists because a demo needs a price drop to actually happen, and
editing data/catalog.json mid-demo to fake one is worse than saying out loud
that we're simulating it. With the flag, prices are discounted in memory only —
catalog.json is never written to.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import watches


def _price_lookup_with_drop(drop_percent):
    """Catalog prices, reduced by drop_percent. In memory only."""
    import catalog

    def lookup(product_id):
        product = catalog.get_product_by_id(product_id)
        if product is None:
            return None
        return round(product["price"] * (1 - drop_percent / 100.0), 2)

    return lookup


def main():
    parser = argparse.ArgumentParser(description="Check active price-drop watches.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Evaluate and print, but don't mark watches triggered or write audit rows.",
    )
    parser.add_argument(
        "--simulate-drop", type=float, default=None, metavar="PERCENT",
        help="Pretend every catalog price dropped by this percent (in memory only).",
    )
    parser.add_argument(
        "--customer", default=None,
        help="Only check watches belonging to this customer id.",
    )
    args = parser.parse_args()

    price_lookup = None
    if args.simulate_drop is not None:
        price_lookup = _price_lookup_with_drop(args.simulate_drop)

    print("=" * 70)
    print("  PRICE WATCH CHECK")
    print("=" * 70)
    print(f"  mode      : {'DRY RUN — nothing will be updated' if args.dry_run else 'live'}")
    print(f"  prices    : {'catalog' if price_lookup is None else f'catalog minus {args.simulate_drop:g}% (simulated)'}")
    if args.customer:
        print(f"  customer  : {args.customer}")
    print()

    report = watches.check_watches(
        price_lookup=price_lookup, mark=not args.dry_run, customer_id=args.customer
    )

    if not report:
        print("  No active watches. Ask the agent to watch something first:")
        print('    "let me know if the CloudRunner shoes drop below 2500"')
        print()
        return 0

    fired = 0
    for row in report:
        status = "TRIGGER" if row["would_trigger"] else "  hold "
        target = f"target ₹{row['target_price']:g}" if row["target_price"] is not None else "any drop"
        name = row.get("product_name") or row["product_id"]
        current = f"₹{row['current_price']:g}" if row["current_price"] is not None else "n/a"
        print(f"  [{status}] watch #{row['watch_id']}  {name}  ({target})")
        print(f"            was ₹{row['price_at_creation']:g} -> now {current}  — {row['reason']}")
        if row["would_trigger"]:
            fired += 1
            print(f"            NOTIFY: {row['notification']}")
            if row["triggered"]:
                print("            status -> triggered, audit row written")
            else:
                print("            (dry run — status left active, nothing logged)")
        print()

    print("-" * 70)
    print(f"  {len(report)} active watch(es) checked, {fired} would notify the customer.")
    if fired and args.dry_run:
        print("  Dry run: no statuses changed and no audit rows written.")
    print("  Note: this build logs and prints notifications. There is no email/SMS")
    print("  transport — that's out of scope for the demo, not a missing piece of logic.")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
