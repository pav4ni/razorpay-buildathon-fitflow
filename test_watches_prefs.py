"""
test_watches_prefs.py — the two optional Stretch features: price-drop watches
and preference/affinity memory.

    python3 test_watches_prefs.py

Entirely OFFLINE. No ANTHROPIC_API_KEY, no Razorpay credentials, no server, no
network. Everything under test is deterministic Python plus SQLite:

  A. PRICE WATCHES — creation through the agent's own tool handler, the trigger
     rule for both watch modes, the manual check run (including the audit row it
     writes), and the refusals: unknown product, price already at target,
     duplicate watch, and re-firing a watch that already went off.

  B. PREFERENCE MEMORY — the two inference rules, count-to-confidence, the
     confidence floor that keeps a single purchase from moving anyone's results,
     and the end-to-end claim that matters: buy twice, then a later relevant
     search measurably ranks the matching category higher — WITHOUT displacing
     the genuinely most relevant result.

Both groups use their own test customer id and their own session ids, so they
never read or mutate the demo customer's real watches and preferences, and
re-running the suite is safe.
"""

import os
import sys
import uuid

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "app"))

import numpy as np
np.seterr(all="ignore")  # silence harmless macOS matmul warnings from catalog.py

import agent
import audit
import catalog
import preferences
import watches

failures = []

RUN_ID = uuid.uuid4().hex[:8]
TEST_CUSTOMER = f"cust_test_prefs_{RUN_ID}"

# Products used by the fixtures, pinned by id so a catalog edit fails loudly
# here rather than quietly changing what the tests mean.
WATCHED_PRODUCT = "P001"      # CloudRunner Mesh Running Shoes, ₹2799, footwear
BOUGHT_SUPPLEMENT = "P020"    # MultiFuel Daily Multivitamin, ₹649, supplements
PREFERRED_IN_SEARCH = "P033"  # SleepWell Recovery Magnesium, ₹599, supplements

# The search used for the ranking test. Chosen because its top result
# (ThermaFlex Compression Sleeves, an accessory) is NOT in the preferred
# category — so the test can assert the nudge happened and that it didn't
# override relevance, in the same query.
RANKING_QUERY = "recovery after workout"

_created_watch_ids = []


def fixture_session(name):
    """A ShoppingSession wired to the test customer, with no Razorpay call.

    link_customer() is deliberately not used: it would hit the Customers API and
    bind these rows to the shared demo identity, which is exactly what these
    tests need to stay away from.
    """
    session = agent.ShoppingSession(session_id=f"test-watch-{name}-{RUN_ID}")
    session.customer_id = TEST_CUSTOMER
    session.current_user_query = f"(fixture: {name})"
    return session


def check(label, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"    [{status}] {label}" + (f"  — {detail}" if detail else ""))
    if not condition:
        failures.append(label)


def header(title):
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def rows_for(session_id, action_type=None):
    events = audit.get_session_log(session_id)
    if action_type:
        events = [e for e in events if e["action_type"] == action_type]
    return events


def call_tool(session, tool_name, **tool_input):
    """Invoke a tool exactly the way run_turn would, so the audit write is
    covered too — these tests exercise the real execution path, not the handler
    in isolation."""
    tool_input.setdefault("reasoning", "test fixture call")
    return agent.execute_tool(session, tool_name, tool_input)


def price_lookup_fixed(prices):
    """A stand-in for the catalog price source: {product_id: price}."""
    return lambda product_id: prices.get(product_id)


# ---------------------------------------------------------------------------
# A. PRICE WATCHES
# ---------------------------------------------------------------------------

def test_watch_creation():
    header("A1. Creating a price watch through the agent's tool layer")
    session = fixture_session("create")

    product = catalog.get_product_by_id(WATCHED_PRODUCT)
    target = product["price"] - 300

    result = call_tool(session, "create_price_watch",
                       product_id=WATCHED_PRODUCT, target_price=target)

    check("tool reports the watch was created", result.get("created") is True, str(result)[:120])
    check("watch id returned", bool(result.get("watch_id")))
    check("current price came from the catalog, not the model",
          result.get("current_price") == product["price"],
          f"₹{result.get('current_price')} vs catalog ₹{product['price']}")

    if not result.get("watch_id"):
        return None
    _created_watch_ids.append(result["watch_id"])

    stored = watches.get_watch(result["watch_id"])
    check("row persisted as active", stored["status"] == watches.STATUS_ACTIVE, stored["status"])
    check("target price stored", stored["target_price"] == target)
    check("baseline price stored for the drop comparison",
          stored["price_at_creation"] == product["price"])
    check("watch is attached to the customer", stored["customer_id"] == TEST_CUSTOMER)
    check("watch is attached to the session", stored["session_id"] == session.session_id)

    audit_rows = rows_for(session.session_id, "price_watch_created")
    check("audit trail records the watch", len(audit_rows) == 1, f"{len(audit_rows)} row(s)")
    if audit_rows:
        check("audit row carries the agent's reasoning",
              bool(audit_rows[0]["agent_reasoning"]))
        check("audit row carries the customer id",
              audit_rows[0]["customer_id"] == TEST_CUSTOMER)

    return stored


def test_trigger_rule():
    header("A2. The trigger rule, both watch modes (pure function, no DB)")

    target_watch = {"id": 1, "product_id": "PX", "price_at_creation": 2799, "target_price": 2499}
    any_drop_watch = {"id": 2, "product_id": "PX", "price_at_creation": 2799, "target_price": None}

    check("target watch holds at the original price",
          evaluate(target_watch, 2799) is False)
    check("target watch holds on a drop that misses the target",
          evaluate(target_watch, 2600) is False, "₹2600 > ₹2499 target")
    check("target watch fires exactly at the target",
          evaluate(target_watch, 2499) is True, "at-or-below, not strictly-below")
    check("target watch fires below the target",
          evaluate(target_watch, 2300) is True)

    check("any-drop watch holds at an unchanged price",
          evaluate(any_drop_watch, 2799) is False)
    check("any-drop watch fires on a small drop",
          evaluate(any_drop_watch, 2798) is True, "'any drop' means any drop")
    check("neither watch fires on a price INCREASE",
          evaluate(target_watch, 3200) is False and evaluate(any_drop_watch, 3200) is False)

    detail = watches.evaluate_watch(any_drop_watch, 2500)
    check("evaluation reports the size of the drop",
          detail["drop_from_creation"] == 299, str(detail["drop_from_creation"]))
    check("evaluation explains itself in plain language",
          "2799" in detail["reason"] and "2500" in detail["reason"], detail["reason"])


def evaluate(watch, price):
    return watches.evaluate_watch(watch, price)["would_trigger"]


def test_check_run_triggers(stored_watch):
    header("A3. The manual check run: a simulated drop flags the watch")
    if stored_watch is None:
        check("watch fixture available", False, "A1 did not create a watch")
        return

    session_id = stored_watch["session_id"]
    target = stored_watch["target_price"]

    # First: a check where the price hasn't moved. Nothing should happen.
    report = watches.check_watches(
        price_lookup=price_lookup_fixed({WATCHED_PRODUCT: stored_watch["price_at_creation"]}),
        customer_id=TEST_CUSTOMER,
    )
    check("unchanged price does not trigger", all(not r["would_trigger"] for r in report),
          f"{len(report)} watch(es) checked")
    check("watch stays active after a no-op check",
          watches.get_watch(stored_watch["id"])["status"] == watches.STATUS_ACTIVE)
    check("no premature audit row",
          len(rows_for(session_id, "price_watch_triggered")) == 0)

    # Now the drop.
    dropped_price = target - 100
    report = watches.check_watches(
        price_lookup=price_lookup_fixed({WATCHED_PRODUCT: dropped_price}),
        customer_id=TEST_CUSTOMER,
    )
    fired = [r for r in report if r["would_trigger"]]
    check("the dropped price triggers the watch", len(fired) == 1, f"{len(fired)} fired")

    if fired:
        row = fired[0]
        check("check run acted on it (status written)", row["triggered"] is True)
        check("a notification message was produced", bool(row["notification"]))
        # SQLite hands prices back as floats, so format the expected price the
        # same way the notification does rather than str()-ing 2399.0.
        check("the notification names the product and the new price",
              f"{dropped_price:g}" in (row["notification"] or "")
              and "notified" in (row["notification"] or ""),
              (row["notification"] or "")[:120])

    check("watch is now marked triggered",
          watches.get_watch(stored_watch["id"])["status"] == watches.STATUS_TRIGGERED)
    check("triggered price recorded",
          watches.get_watch(stored_watch["id"])["triggered_price"] == dropped_price)

    trigger_rows = rows_for(session_id, "price_watch_triggered")
    check("the trigger is in the audit trail", len(trigger_rows) == 1, f"{len(trigger_rows)} row(s)")
    if trigger_rows:
        result = trigger_rows[0]["result"]
        check("audit row records the price that fired it",
              result.get("current_price") == dropped_price)
        check("audit row is honest that nothing was actually sent",
              result.get("notification_sent") is False,
              "logging is the demo scope; email/SMS is not built")

    # A second run must not re-fire a watch that already went off.
    report = watches.check_watches(
        price_lookup=price_lookup_fixed({WATCHED_PRODUCT: dropped_price}),
        customer_id=TEST_CUSTOMER,
    )
    check("a triggered watch is not re-checked",
          all(r["watch_id"] != stored_watch["id"] for r in report),
          "no duplicate notifications on repeated runs")
    check("no duplicate audit row",
          len(rows_for(session_id, "price_watch_triggered")) == 1)


def test_dry_run():
    header("A4. --dry-run evaluates without changing anything")
    session = fixture_session("dryrun")
    product = catalog.get_product_by_id(WATCHED_PRODUCT)

    result = call_tool(session, "create_price_watch", product_id=WATCHED_PRODUCT,
                       target_price=product["price"] - 500)
    if not result.get("created"):
        check("dry-run fixture watch created", False, str(result)[:120])
        return
    watch_id = result["watch_id"]
    _created_watch_ids.append(watch_id)

    report = watches.check_watches(
        price_lookup=price_lookup_fixed({WATCHED_PRODUCT: 100}),
        mark=False,
        customer_id=TEST_CUSTOMER,
    )
    mine = [r for r in report if r["watch_id"] == watch_id]
    check("dry run still evaluates the condition",
          bool(mine) and mine[0]["would_trigger"] is True)
    check("dry run does not mark it triggered", bool(mine) and mine[0]["triggered"] is False)
    check("status left active",
          watches.get_watch(watch_id)["status"] == watches.STATUS_ACTIVE)
    check("dry run writes no audit row",
          len(rows_for(session.session_id, "price_watch_triggered")) == 0)

    watches.cancel_watch(watch_id)
    check("a cancelled watch drops out of the active set",
          all(w["id"] != watch_id for w in watches.get_active_watches(customer_id=TEST_CUSTOMER)))


def test_watch_refusals():
    header("A5. Refusals: unknown product, already at target, already watching")
    session = fixture_session("refusals")

    result = call_tool(session, "create_price_watch", product_id="P999")
    check("unknown product is refused", result.get("created") is False)
    check("refusal names the problem", result.get("error_type") == "product_not_found")

    product = catalog.get_product_by_id(WATCHED_PRODUCT)
    result = call_tool(session, "create_price_watch", product_id=WATCHED_PRODUCT,
                       target_price=product["price"] + 500)
    check("a target the price already meets does not become a watch",
          result.get("created") is False, str(result.get("reason")))
    check("agent is told to offer the purchase instead",
          result.get("reason") == "already_at_target" and "buy it now" in (result.get("user_message") or ""),
          (result.get("user_message") or "")[:80])

    # A duplicate on a product this customer is already watching.
    first = call_tool(session, "create_price_watch", product_id=BOUGHT_SUPPLEMENT,
                      target_price=400)
    if first.get("created"):
        _created_watch_ids.append(first["watch_id"])
    second = call_tool(session, "create_price_watch", product_id=BOUGHT_SUPPLEMENT,
                       target_price=350)
    check("a second watch on the same product is refused",
          second.get("created") is False and second.get("reason") == "watch_already_active",
          str(second.get("reason")))
    check("the refusal points at the existing watch",
          second.get("watch_id") == first.get("watch_id"))
    check("only one active watch exists for that product",
          len(watches.get_active_watches(customer_id=TEST_CUSTOMER,
                                         product_id=BOUGHT_SUPPLEMENT)) == 1)

    rejections = rows_for(session.session_id, "rejection")
    check("every refusal is in the audit trail", len(rejections) == 3, f"{len(rejections)} row(s)")


# ---------------------------------------------------------------------------
# B. PREFERENCE MEMORY
# ---------------------------------------------------------------------------

def test_inference_rules():
    header("B1. The inference rules — two if-statements, no model")
    median = catalog.median_price()
    check("catalog median is computable", median > 0, f"₹{median:g}")

    cheap = {"id": "PX", "price": median - 100, "category": "supplements"}
    dear = {"id": "PY", "price": median + 100, "category": "footwear"}
    at_median = {"id": "PZ", "price": median, "category": "gear"}

    check("below the median reads as budget",
          ("price_tier", "budget") in preferences.infer_signals_from_purchase(cheap, median))
    check("above the median reads as premium",
          ("price_tier", "premium") in preferences.infer_signals_from_purchase(dear, median))
    check("the median item itself counts as budget",
          ("price_tier", "budget") in preferences.infer_signals_from_purchase(at_median, median),
          "an odd-length catalog makes the median a real product's price")
    check("the product's category becomes the affinity signal",
          ("category_affinity", "supplements") in preferences.infer_signals_from_purchase(cheap, median))
    check("exactly two signals per purchase",
          len(preferences.infer_signals_from_purchase(cheap, median)) == 2)


def test_confidence_and_floor():
    header("B2. Confidence grows with repetition, and one purchase isn't enough")
    scratch = f"{TEST_CUSTOMER}_confidence"
    preferences.clear_signals(scratch)

    product = catalog.get_product_by_id(BOUGHT_SUPPLEMENT)

    preferences.record_purchase(scratch, product)
    signals = preferences.get_signals(scratch)
    check("first purchase records both signals", len(signals) == 2, f"{len(signals)} signal(s)")
    check("first purchase has count 1", all(s["count"] == 1 for s in signals))
    check("one purchase is below the apply threshold",
          all(s["confidence"] < preferences.MIN_CONFIDENCE_TO_APPLY for s in signals),
          f"conf {signals[0]['confidence']:g} < {preferences.MIN_CONFIDENCE_TO_APPLY}")
    check("so nothing is boosted yet", preferences.build_boost_map(scratch) == {},
          "a single stray purchase must not move anyone's search results")

    preferences.record_purchase(scratch, product)
    signals = preferences.get_signals(scratch)
    check("second purchase increments rather than duplicating",
          len(signals) == 2 and all(s["count"] == 2 for s in signals),
          f"{len(signals)} row(s), counts {[s['count'] for s in signals]}")
    check("two purchases clear the threshold",
          all(s["confidence"] >= preferences.MIN_CONFIDENCE_TO_APPLY for s in signals),
          f"conf {signals[0]['confidence']:g}")

    boost_map = preferences.build_boost_map(scratch)
    check("boost map now carries the category affinity",
          boost_map.get("category_affinity", {}).get("supplements", 0) > 0, str(boost_map))
    check("boost map now carries the price tier",
          boost_map.get("price_tier", {}).get("budget", 0) > 0, str(boost_map))

    for _ in range(5):
        preferences.record_purchase(scratch, product)
    signals = preferences.get_signals(scratch)
    check("confidence is capped at 1.0 however many times they buy",
          all(s["confidence"] == 1.0 for s in signals),
          f"count {signals[0]['count']}, conf {signals[0]['confidence']}")

    preferences.clear_signals(scratch)
    check("signals can be deleted", preferences.get_signals(scratch) == [])


def test_exclusive_price_tier():
    header("B2b. Price tier is exclusive — a shopper isn't budget AND premium")
    scratch = f"{TEST_CUSTOMER}_exclusive"
    preferences.clear_signals(scratch)

    # Three premium purchases, two budget ones. Both clear the confidence floor.
    for _ in range(3):
        preferences.record_signal(scratch, "price_tier", "premium")
    for _ in range(2):
        preferences.record_signal(scratch, "price_tier", "budget")

    stored = preferences.get_signals(scratch)
    check("both tiers are stored — nothing is thrown away",
          len(stored) == 2, f"{len(stored)} row(s)")
    check("both tiers are individually confident enough to apply",
          all(s["confidence"] >= preferences.MIN_CONFIDENCE_TO_APPLY for s in stored))

    boost_map = preferences.build_boost_map(scratch)
    check("but only the dominant tier is applied",
          boost_map.get("price_tier") == {"premium": 1.0}, str(boost_map.get("price_tier")))
    check("the explanation only claims what actually moved the ranking",
          "premium" in preferences.describe(scratch) and "budget" not in preferences.describe(scratch),
          preferences.describe(scratch))

    # Category affinity is NOT exclusive — liking two categories is a real thing.
    preferences.clear_signals(scratch)
    for _ in range(2):
        preferences.record_signal(scratch, "category_affinity", "footwear")
        preferences.record_signal(scratch, "category_affinity", "supplements")
    check("two category affinities can both apply",
          len(preferences.build_boost_map(scratch).get("category_affinity", {})) == 2,
          str(preferences.build_boost_map(scratch).get("category_affinity")))

    # A dead heat between tiers is no evidence at all.
    preferences.clear_signals(scratch)
    for _ in range(2):
        preferences.record_signal(scratch, "price_tier", "premium")
        preferences.record_signal(scratch, "price_tier", "budget")
    check("an exact tie between tiers applies neither",
          "price_tier" not in preferences.build_boost_map(scratch),
          str(preferences.build_boost_map(scratch)))

    preferences.clear_signals(scratch)


def test_signal_recorded_after_purchase():
    header("B3. A completed order writes a signal, through the agent's own path")
    preferences.clear_signals(TEST_CUSTOMER)
    session = fixture_session("learn")
    product = catalog.get_product_by_id(BOUGHT_SUPPLEMENT)

    # The success branch of _handle_create_order calls this. Calling it directly
    # keeps the test offline — no Razorpay, mock or otherwise — while still
    # covering the real inference, storage and audit code.
    recorded = agent._learn_from_purchase(session, product, {"id": "order_TEST123"})

    check("the purchase produced signals", len(recorded) == 2, f"{len(recorded)} signal(s)")
    stored = {(s["signal_type"], s["signal_value"]) for s in preferences.get_signals(TEST_CUSTOMER)}
    check("category affinity stored", ("category_affinity", "supplements") in stored, str(stored))
    check("price tier stored", ("price_tier", "budget") in stored,
          f"₹{product['price']} vs median ₹{catalog.median_price():g}")

    signal_rows = rows_for(session.session_id, "preference_signal")
    check("the inference is in the audit trail", len(signal_rows) == 1, f"{len(signal_rows)} row(s)")
    if signal_rows:
        result = signal_rows[0]["result"]
        check("audit row lists what was inferred", len(result.get("signals", [])) == 2)
        check("audit row states the rule used, so it can be defended",
              "median" in (result.get("rule") or ""), (result.get("rule") or "")[:60])
        check("audit row links back to the order",
              signal_rows[0]["action_params"].get("order_id") == "order_TEST123")

    # Second purchase of the same kind — this is what takes it over the threshold.
    agent._learn_from_purchase(session, product, {"id": "order_TEST124"})
    check("a repeat purchase makes the signal confident enough to apply",
          preferences.build_boost_map(TEST_CUSTOMER) != {},
          preferences.describe(TEST_CUSTOMER))


def test_ranking_nudge():
    header("B4. The nudge: a later relevant search moves in the right direction")
    boost_map = preferences.build_boost_map(TEST_CUSTOMER)
    if not boost_map:
        check("preference fixture available", False, "B3 did not leave a confident signal")
        return
    print(f"    (learned: {preferences.describe(TEST_CUSTOMER)})")

    baseline = catalog.search_catalog(RANKING_QUERY, top_k=6)
    boosted = catalog.search_catalog(RANKING_QUERY, top_k=6, preference_boost=boost_map)

    def rank_of(results, product_id):
        for i, p in enumerate(results):
            if p["id"] == product_id:
                return i
        return None

    def score_of(results, product_id):
        for p in results:
            if p["id"] == product_id:
                return p["match_score"]
        return None

    before_rank = rank_of(baseline, PREFERRED_IN_SEARCH)
    after_rank = rank_of(boosted, PREFERRED_IN_SEARCH)
    print(f"    {PREFERRED_IN_SEARCH}: rank {before_rank} -> {after_rank}  "
          f"(score {score_of(baseline, PREFERRED_IN_SEARCH)} -> {score_of(boosted, PREFERRED_IN_SEARCH)})")

    check("the preferred-category product scores higher",
          score_of(boosted, PREFERRED_IN_SEARCH) > score_of(baseline, PREFERRED_IN_SEARCH))
    check("and it actually moves up the list",
          before_rank is not None and after_rank is not None and after_rank < before_rank,
          f"rank {before_rank} -> {after_rank}")

    # The guardrail, and the more important half of this test: the most relevant
    # result is NOT in the preferred category, and it must still come first.
    check("the most relevant result still ranks first",
          baseline[0]["id"] == boosted[0]["id"],
          f"{boosted[0]['name']} ({boosted[0]['category']}) — relevance was not overridden")
    check("no product gained more than the documented ceiling",
          all(p.get("preference_boost", 0) <= catalog.PREFERENCE_MAX_BOOST + 1e-9 for p in boosted),
          f"ceiling {catalog.PREFERENCE_MAX_BOOST}")
    check("boosted products explain why they were boosted",
          all(p.get("preference_matched") for p in boosted if p.get("preference_boost")))


def test_no_signals_is_unchanged():
    header("B5. A customer with no history gets the original ranking, exactly")
    fresh = f"{TEST_CUSTOMER}_fresh"
    preferences.clear_signals(fresh)

    check("no signals means no boost map", preferences.build_boost_map(fresh) == {})

    baseline = catalog.search_catalog(RANKING_QUERY, top_k=6)
    unboosted = catalog.search_catalog(RANKING_QUERY, top_k=6,
                                       preference_boost=preferences.build_boost_map(fresh))
    check("scores are bit-for-bit identical to pre-feature behaviour",
          [p["match_score"] for p in baseline] == [p["match_score"] for p in unboosted])
    check("no preference fields leak into the results",
          all("preference_boost" not in p for p in unboosted))

    # And the same through the agent's search handler, which is what the model sees.
    session = fixture_session("nosignals")
    session.customer_id = fresh
    response = agent._handle_search_catalog(session, {"query": RANKING_QUERY})
    check("agent search reports no personalization for a new customer",
          "personalization" not in response)


def test_search_handler_surfaces_personalization():
    header("B6. A boosted search is explainable in the tool result and the log")
    session = fixture_session("personalized")
    response = agent._handle_search_catalog(session, {"query": RANKING_QUERY})

    personalization = response.get("personalization")
    check("the tool result declares that a boost was applied",
          bool(personalization) and personalization.get("applied") is True)
    if personalization:
        check("it names the signals used",
              "category_affinity" in personalization.get("signals", {}), str(personalization.get("signals")))
        check("it carries a human-readable explanation",
              "supplements" in (personalization.get("explanation") or ""),
              (personalization.get("explanation") or "")[:70])
        check("the model is told not to bring it up unprompted",
              "unless asked" in (personalization.get("note") or ""))


# ---------------------------------------------------------------------------

def cleanup():
    """Leave the demo database as we found it: test preferences deleted, test
    watches cancelled so they never show up in a live check run."""
    preferences.clear_signals(TEST_CUSTOMER)
    for watch_id in _created_watch_ids:
        watch = watches.get_watch(watch_id)
        if watch and watch["status"] == watches.STATUS_ACTIVE:
            watches.cancel_watch(watch_id)
    remaining = watches.get_active_watches(customer_id=TEST_CUSTOMER)
    check("no test watches left active", not remaining, f"{len(remaining)} left")
    check("no test preferences left behind", preferences.get_signals(TEST_CUSTOMER) == [])


def main():
    print("=" * 70)
    print("  OPTIONAL STRETCH TESTS — price watches + preference memory")
    print("=" * 70)
    print(f"  run id        : {RUN_ID}")
    print(f"  test customer : {TEST_CUSTOMER}")
    print("  network       : none required")

    stored_watch = test_watch_creation()
    test_trigger_rule()
    test_check_run_triggers(stored_watch)
    test_dry_run()
    test_watch_refusals()

    test_inference_rules()
    test_confidence_and_floor()
    test_exclusive_price_tier()
    test_signal_recorded_after_purchase()
    test_ranking_nudge()
    test_no_signals_is_unchanged()
    test_search_handler_surfaces_personalization()

    header("CLEANUP")
    cleanup()

    header("KNOWN LIMITATIONS — deliberate, not untested gaps")
    print("  - No scheduler: watches are checked by running app/check_price_watches.py")
    print("    by hand. The evaluation rule is tested; the cron entry is not built.")
    print("  - No notification transport: a triggered watch logs and prints. There is")
    print("    no email or SMS, and the audit row says so (notification_sent: false).")
    print("  - No authentication: agent.py's DEMO_CUSTOMER means every real session is")
    print("    the same person, so preference memory accumulates across all demo runs.")
    print("    These tests use their own customer id precisely to avoid that; a live")
    print("    demo will not, which is what makes the feature visible on stage.")

    header("RESULT")
    if failures:
        print(f"\n  {len(failures)} assertion(s) FAILED:")
        for name in failures:
            print(f"    - {name}")
        return 1
    print("\n  All assertions passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
