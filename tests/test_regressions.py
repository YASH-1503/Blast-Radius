"""Regression tests: one per bug that actually bit, plus the engine behaviours they came from.

These are not coverage tests. Every single one corresponds to something that was silently wrong in
this codebase at some point today, and the comment on each says what. A test whose failure message
does not explain the original mistake is worth very little at 2am.

They need a running database and a loaded graph (`make db && make ingest`). The GitHub-layer tests
additionally need the fetch cache and skip cleanly without it, so a fresh clone with no network can
still run most of the suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from blast_radius import (
    config,
    algorithms,
    data as synthetic,
    embeddings,
    github_ingest,
    ingest,
    memory,
    queries,
    schema,
    tools,
)
from blast_radius.db import GraphClient

TENANT = "acme"


@pytest.fixture(scope="module")
def gc() -> GraphClient:
    client = GraphClient(TENANT)
    if not client.ping():
        pytest.skip("no database; run `make db`")
    if not client.run("MATCH (s:Service) RETURN count(*) AS n").first()["n"]:
        pytest.skip("graph is empty; run `make ingest`")
    return client


# --------------------------------------------------------------------------------------
# Engine behaviours. If an image upgrade changes one of these, the queries built on it break.
# --------------------------------------------------------------------------------------

def test_node_binding_does_not_survive_an_aggregating_with(gc: GraphClient) -> None:
    """A node bound before an AGGREGATING `WITH` cannot be matched against afterwards.

    This cost the whole of Q1: `WITH s, min(length(path)) AS hops` followed by
    `MATCH (d:Deploy)-[:DEPLOYS]->(s)` returned ZERO rows with no error, while the same query using
    `WITH DISTINCT s` returned 14. Queries carry `s.name` across such a boundary and re-match on the
    indexed key instead.

    Asserted rather than merely commented so that if FalkorDB ever fixes this, the test fails and
    the workaround can be removed deliberately instead of lingering forever.
    """
    base = """
        MATCH (a:Service {name: 'api-gateway'})
        MATCH path = (a)-[:DEPENDS_ON*1..3]->(s:Service)
    """
    with_aggregate = gc.run(
        base + "WITH s, min(length(path)) AS hops MATCH (d:Deploy)-[:DEPLOYS]->(s) RETURN count(*) AS n"
    ).first()["n"]
    without_aggregate = gc.run(
        base + "WITH DISTINCT s MATCH (d:Deploy)-[:DEPLOYS]->(s) RETURN count(*) AS n"
    ).first()["n"]

    assert without_aggregate > 0, "the non-aggregating form should find deploys"

    # This behaviour is VERSION-DEPENDENT, which is the real lesson and was only discovered by
    # running the suite against two engines. On FalkorDB 6.0.1 the aggregating form returns 0 rows
    # with no error; on 4.22.x the binding survives and it returns rows. Pinning either number
    # makes the suite fail on the other engine -- and since Cloud's free tier runs 4.22.x while
    # `falkordb/falkordb:latest` served 6.0.1, both are engines this project actually runs on.
    #
    # What must hold on BOTH is that the shipped query is correct, so that is what is asserted.
    # The aggregating count is only reported.
    assert with_aggregate in (0, without_aggregate), (
        f"unexpected third behaviour from an aggregating WITH: {with_aggregate} rows against "
        f"{without_aggregate} without aggregation. Known: 0 on 6.0.1 (binding silently dies), "
        f"equal on 4.22.x (binding survives). Re-check queries.candidate_causes by hand."
    )
    rows = queries.candidate_causes("api-gateway", client=gc).rows
    assert rows, (
        "candidate_causes returned nothing. Its re-match-by-name form is written to survive the "
        "engine dropping a node binding across an aggregating WITH; if it is empty, that "
        "workaround has stopped working on this engine version."
    )


def test_shortest_path_is_rejected_in_a_match_clause(gc: GraphClient) -> None:
    """`MATCH p = shortestPath(...)` is a hard error in FalkorDB; it belongs in WITH or RETURN."""
    with pytest.raises(Exception, match="(?i)shortestpath"):
        gc.run(
            """
            MATCH (a:Service {name: 'api-gateway'}), (b:Service {name: 'pricing-legacy'})
            MATCH p = shortestPath((a)-[:DEPENDS_ON*1..4]->(b))
            RETURN length(p) AS n
            """
        )


def test_pagerank_is_positional_and_betweenness_is_a_config_map(gc: GraphClient) -> None:
    """Two calling conventions in one procedure namespace, neither documented as differing.

    `algo.pageRank('Service', 'DEPENDS_ON')` works everywhere. The positional form on
    `algo.betweenness` is REJECTED on 6.0.1 with an arity error that names the count but not the
    expected shape -- and ACCEPTED on 4.22.x. The config-map form works on both, which is why
    `algorithms.py` uses it uniformly: it is the only shape that is portable across the two engine
    versions this project runs on (Cloud free tier is 4.22.x, `latest` served 6.0.1).
    """
    assert gc.run(
        "CALL algo.pageRank('Service','DEPENDS_ON') YIELD node, score RETURN count(*) AS n"
    ).first()["n"] > 0

    # Portable shape -- the one the code actually ships. Must work on every engine.
    assert gc.run(
        "CALL algo.betweenness($p) YIELD node, score RETURN count(*) AS n",
        {"p": algorithms.SERVICE_PROJECTION},
    ).first()["n"] > 0

    # The positional form is the version-dependent part. Probing it found the most dangerous
    # difference of all: 6.0.1 REJECTS it with an arity error, and 4.22.x ACCEPTS it and returns
    # ZERO ROWS. The loud engine is the safe one. On Cloud's 4.22.x, a centrality call written the
    # positional way would have produced an empty ranking, no error, and an agent confidently
    # ranking nothing -- the same silent-empty family as six other bugs in this project.
    #
    # Engine behaviour cannot be asserted away, so it is recorded here and the SHIPPED code is
    # guarded structurally instead: algorithms.py must never call betweenness positionally.
    try:
        positional: int | None = gc.run(
            "CALL algo.betweenness('Service','DEPENDS_ON') YIELD node, score RETURN count(*) AS n"
        ).first()["n"]
    except Exception:
        positional = None  # rejected outright: 6.0.1
    assert positional in (None, 0) or positional > 0  # documents both engines; never fails

    source = Path(algorithms.__file__).read_text(encoding="utf-8")
    offenders = re.findall(r"CALL\s+algo\.betweenness\(\s*['\"]", source)
    assert not offenders, (
        "algorithms.py calls algo.betweenness with POSITIONAL arguments. That silently returns "
        "zero rows on FalkorDB 4.22.x -- which is what FalkorDB Cloud's free tier runs -- while "
        "raising a clear arity error on 6.0.1. Use the config-map form ($projection)."
    )


def test_write_statistics_are_reported_by_the_engine(gc: GraphClient) -> None:
    """The ingest report quotes these rather than its own input length.

    It previously printed `len(rows)`, so a statement whose MATCH found nothing still reported
    success -- the table said `decision authors 1` while the graph held zero.
    """
    result = gc.run("CREATE (:_TestProbe {a: 1})-[:_TESTREL]->(:_TestProbe {a: 2})")
    assert result.writes.nodes_created == 2
    assert result.writes.relationships_created == 1
    deleted = gc.run("MATCH (n:_TestProbe) DETACH DELETE n")
    assert deleted.writes.nodes_deleted == 2


# --------------------------------------------------------------------------------------
# Modelling bugs
# --------------------------------------------------------------------------------------

def test_no_deploy_node_serves_two_services(gc: GraphClient) -> None:
    """`Deploy` keyed on the bare sha collapsed two services into one node.

    One real commit can touch two mapped areas; `MERGE (d:Deploy {sha: ...})` then returned the
    SAME node for both and the second `SET` overwrote the first, so Q1 reported `media` as deployed
    with `notifications`' version. The key is `<service>@<sha12>`.
    """
    rows = gc.run(
        """
        MATCH (d:Deploy)-[:DEPLOYS]->(s:Service)
        WITH d.id AS id, count(DISTINCT s.name) AS services
        WHERE services > 1
        RETURN count(*) AS collisions
        """
    ).first()
    assert rows["collisions"] == 0


def test_nobody_is_a_member_of_two_teams(gc: GraphClient) -> None:
    """A person in every team can never be in the WRONG team.

    Membership dedup was keyed `(handle, team)`, so a prolific maintainer joined all five teams --
    which made `ownership_mismatch` structurally incapable of firing, quietly destroying the
    project's central finding. One team per person, chosen by highest commit volume.
    """
    offenders = gc.run(
        """
        MATCH (p:Person)-[:MEMBER_OF]->(t:Team)
        WITH p.handle AS handle, count(DISTINCT t) AS teams
        WHERE teams > 1
        RETURN collect(handle) AS offenders
        """
    ).first()["offenders"]
    assert offenders == [], f"these people are in more than one team: {offenders}"


def test_no_invented_person_is_in_a_real_team(gc: GraphClient) -> None:
    """`startswith('dev')` also matched a handcrafted human, who survived into a real roster."""
    offenders = gc.run(
        """
        MATCH (p:Person)-[:MEMBER_OF]->(:Team)
        WHERE p.source <> 'github' OR p.is_bot = true
        RETURN collect(DISTINCT p.handle) AS offenders
        """
    ).first()["offenders"]
    assert offenders == []


def test_every_decision_has_an_author(gc: GraphClient) -> None:
    """An `UNWIND ... MATCH ... MERGE` that matches nothing still reported its input length."""
    orphans = gc.run(
        """
        MATCH (d:Decision)
        OPTIONAL MATCH (p:Person)-[:AUTHORED]->(d)
        WITH d.id AS decision, count(p) AS authors
        WHERE authors = 0
        RETURN collect(decision) AS orphans
        """
    ).first()["orphans"]
    assert orphans == []


def test_episode_incident_is_actually_written(gc: GraphClient) -> None:
    """The ingest assembled `incident` in its row and never SET it.

    Nothing failed and no count was wrong -- a recalled episode simply could not be traced back to
    the incident it came from, a capability that silently did not exist.
    """
    rows = gc.run(
        "MATCH (e:Episode {kind: 'incident_retrospective'}) RETURN e.id AS id, e.incident AS incident"
    ).rows
    assert rows, "no seeded episodes"
    assert all(r["incident"] for r in rows), "an episode is missing its incident back-reference"


# --------------------------------------------------------------------------------------
# Property-name drift -- the bug class that appeared four times
# --------------------------------------------------------------------------------------

def test_tier_vocabulary_is_asserted_not_assumed() -> None:
    """The severity rule tested `tier = 'tier-0'`, a value no service has ever carried.

    The predicate matched nothing, so `tier0_impacted` was empty -- and the verdict was still
    `sev1`, because the revenue branch of the same CASE fired. Right answer, dead rule.
    """
    synthetic.assert_tier_vocabulary()
    assert synthetic.CRITICAL_TIER in synthetic.TIERS
    assert synthetic.tier_of("pricing-legacy") == synthetic.CRITICAL_TIER


def test_severity_reports_which_branch_decided_it(gc: GraphClient) -> None:
    """Returned so a dead branch cannot hide behind a correct verdict."""
    row = queries.severity("pricing-legacy", client=gc).first()
    assert row["severity"] == "sev1"
    assert row["sev1_because"] in {"tier", "revenue", "tier+revenue"}
    assert row["critical_tier_impacted"], "the tier branch found nothing -- is the vocabulary right?"


def test_queries_only_read_properties_that_exist(gc: GraphClient) -> None:
    """Guard for the whole class. Cypher returns null for a missing property and never errors.

    Four of these shipped: `Runbook.mttr_minutes` (really `expected_mttr_minutes`),
    `Team.escalation` (really `channel`), `Commit.subject` (really `msg`), and `Episode.incident`.
    """
    present = schema.properties_in_use(gc)
    expected = {
        "Runbook": {"expected_mttr_minutes", "steps", "title", "owner_team"},
        "Team": {"channel", "name"},
        "Commit": {"msg", "sha", "at"},
        "Deploy": {"commit_subject", "version", "at", "in_alert_window"},
        "Person": {"handle", "is_bot"},
    }
    for label, props in expected.items():
        missing = sorted(props - present[label])
        assert not missing, f"{label} is missing {missing}; present: {sorted(present[label])}"


# --------------------------------------------------------------------------------------
# The six questions
# --------------------------------------------------------------------------------------

@pytest.mark.parametrize("key", sorted(queries.QUESTIONS))
def test_every_question_answers_with_rows_and_cypher(gc: GraphClient, key: str) -> None:
    result = queries.ask(key, client=gc)
    assert result.rows, f"{key} returned no rows"
    assert result.cypher.strip()
    assert result.elapsed_ms >= 0


def test_q1_offers_three_candidates_and_the_agent_must_discriminate(gc: GraphClient) -> None:
    """An agent handed exactly one candidate has not reasoned about anything."""
    rows = queries.candidate_causes("api-gateway", client=gc).rows
    services = sorted(r["service"] for r in rows)
    assert services == ["media", "notifications", "pricing-legacy"], services


def test_q1_candidates_are_distinguishable(gc: GraphClient) -> None:
    """A repo-wide commit was the newest commit for several services at once.

    All three suspects came out with the same author and the same change text, which leaves the
    ranking looking arbitrary. A commit touching exactly one mapped area is preferred.
    """
    rows = queries.candidate_causes("api-gateway", client=gc).rows
    assert len({r["sha"] for r in rows}) == len(rows), "two candidates share a commit"
    assert len({r["change"] for r in rows}) == len(rows), "two candidates share a change description"


def test_every_question_declares_highlightable_focus(gc: GraphClient) -> None:
    """Nine of ten steps highlighted nothing because most queries return scalars, not nodes.

    Fixed in the query layer via `focus_keys` rather than in the frontend, so any consumer of the
    API gets the highlight list, not just this UI.
    """
    unfocused = []
    for key in sorted(queries.QUESTIONS):
        result = queries.ask(key, client=gc)
        if not result.focus and not result.nodes:
            unfocused.append(key)
    assert not unfocused, f"these questions highlight nothing: {unfocused}"


# --------------------------------------------------------------------------------------
# Tools
# --------------------------------------------------------------------------------------

def test_an_unknown_service_is_refused_not_answered_emptily(gc: GraphClient) -> None:
    """A bad name otherwise runs a valid query that matches nothing.

    An empty result is indistinguishable from "there is nothing to find", which is the exact
    failure mode this project exists to avoid -- so it is rejected before the query runs.
    """
    with pytest.raises(tools.ToolError, match="no such service"):
        tools.call("blast_radius", {"service": "not-a-real-service"}, client=gc)


def test_tools_reject_missing_and_unknown_arguments(gc: GraphClient) -> None:
    with pytest.raises(tools.ToolError, match="requires"):
        tools.call("blast_radius", {}, client=gc)
    with pytest.raises(tools.ToolError, match="unknown argument"):
        tools.call("severity", {"service": "api-gateway", "nope": 1}, client=gc)
    with pytest.raises(tools.ToolError, match="unknown tool"):
        tools.call("no_such_tool", {}, client=gc)


def test_every_registered_tool_runs(gc: GraphClient) -> None:
    sample = {
        "service": "pricing-legacy", "source": "api-gateway", "target": "pricing-legacy",
        "upstream": "checkout", "downstream": "pricing-legacy",
        "text": "checkout 5xx after a pricing deploy",
    }
    for name, spec in tools.REGISTRY.items():
        args = {k: sample[k] for k in spec.required if k in sample}
        assert len(args) == len(spec.required), f"{name}: test has no sample for {spec.required}"
        call = tools.call(name, args, client=gc)
        assert call.result.cypher.strip(), f"{name} returned no cypher"


# --------------------------------------------------------------------------------------
# Memory
# --------------------------------------------------------------------------------------

def test_vector_recall_returns_a_similarity_not_a_raw_distance(gc: GraphClient) -> None:
    """`db.idx.vector.queryNodes` yields DISTANCE -- lower is closer.

    Sorting that as though it were a similarity produces a confidently wrong "most similar
    incident" that looks entirely plausible in a demo.
    """
    rows = memory.recall("pricing deploy caused checkout errors", k=3, client=gc).rows
    assert rows
    distances = [r["distance"] for r in rows]
    assert distances == sorted(distances), "hits must be ordered nearest-first"
    for row in rows:
        assert row["similarity"] == pytest.approx(1.0 - row["distance"], abs=1e-3)


def test_writing_an_episode_for_an_unknown_service_fails_loudly(gc: GraphClient) -> None:
    """A silent no-op here would leave the agent believing it had remembered something."""
    with pytest.raises(RuntimeError, match="not written"):
        memory.write_episode(
            memory.Episode(text="t", summary="s", services=["no-such-service"]), client=gc
        )


def test_runbook_is_reached_through_the_incident_it_resolved(gc: GraphClient) -> None:
    """Procedural memory via episodic memory: proposed because it worked, not because it is tagged."""
    row = memory.runbook_for("pricing-legacy", client=gc).first()
    assert row["runbook"] == "RB-07"
    assert row["learned_from"].startswith("INC-")
    assert row["steps"]


def test_embedding_backend_is_reported_honestly() -> None:
    """With fastembed absent these are hashed trigrams, not semantics. Saying which is the point."""
    described = embeddings.describe()
    assert described["backend"] in {"fastembed", "hashed-trigram"}
    assert described["semantic"] is (described["backend"] == "fastembed")


# --------------------------------------------------------------------------------------
# Multi-tenancy and provenance
# --------------------------------------------------------------------------------------

def test_tenants_are_separate_graphs_with_no_filter_anywhere() -> None:
    """Isolation is the graph name, so it cannot be forgotten in a WHERE clause."""
    assert GraphClient("acme").graph_name != GraphClient("globex").graph_name
    for module in (queries, memory, algorithms):
        source = open(module.__file__).read()
        assert "tenant =" not in source.replace("tenant = ", "tenant_kw = "), (
            f"{module.__name__} appears to filter by tenant inside a query"
        )


def test_provenance_is_a_query_not_a_claim(gc: GraphClient) -> None:
    rows = queries.provenance(client=gc).rows
    by_source: dict[str, int] = {}
    for row in rows:
        by_source[row["source"]] = by_source.get(row["source"], 0) + int(row["nodes"])
    assert by_source.get("github", 0) > 100, "the real layer should dominate the node count"
    assert by_source.get("synthetic", 0) > 0, "the synthetic layer should be present and labelled"


def test_no_email_addresses_are_stored(gc: GraphClient) -> None:
    """Commit author email is public but is deliberately not kept -- only logins."""
    present = schema.properties_in_use(gc)
    for label, props in present.items():
        offenders = sorted(p for p in props if "email" in p.lower())
        assert not offenders, f"{label} stores {offenders}"


# --------------------------------------------------------------------------------------
# The GitHub layer -- cache-only, skipped on a fresh clone with no network
# --------------------------------------------------------------------------------------

@pytest.fixture(scope="module")
def layer() -> github_ingest.GitHubLayer:
    if not github_ingest.cache_status()["offline_ready"]:
        pytest.skip("no GitHub cache; run `make ingest` once with network")
    return github_ingest.fetch(offline=True)


def test_github_timestamps_are_read_as_utc() -> None:
    """`time.mktime` reads the parsed struct in the HOST zone, moving every commit by the offset.

    On an IST machine every GitHub commit shifted 5.5 hours, so recency ordering -- which the whole
    "what deployed last" question rests on -- was quietly wrong and commits could land in the
    future. `calendar.timegm` is correct.
    """
    import calendar
    import time

    value = "2026-01-02T03:04:05Z"
    expected = calendar.timegm(time.strptime("2026-01-02T03:04:05+0000", "%Y-%m-%dT%H:%M:%S%z"))
    assert github_ingest._parse_iso8601(value) == expected

    # The bug, stated as the difference it made: mktime is offset by the host's UTC offset.
    wrong = time.mktime(time.strptime("2026-01-02T03:04:05", "%Y-%m-%dT%H:%M:%S"))
    assert github_ingest._parse_iso8601(value) - wrong == -time.timezone or time.timezone == 0


def test_timezone_inference_lands_on_the_right_hemisphere(layer) -> None:
    """The derived metric that exposed the UTC bug, and independent evidence the pipeline is sound.

    FalkorDB is an Israel-based project. Inferring UTC+1..+4 from commit hours alone agrees with
    that; before the timestamp fix it inferred UTC+6..+8, which is impossible.

    Asserted on the MEDIAN across top committers, never on one person -- a nocturnal maintainer is
    placed wrongly by design and one contributor legitimately infers to UTC-3.
    """
    median = layer.median_utc_offset()
    assert median is not None
    assert 0 <= median <= 5, f"median inferred offset UTC{median:+d} is implausible for this repo"


def test_bots_are_flagged_kept_and_never_paged(layer, gc: GraphClient) -> None:
    """The top committer to one service was `dependabot[bot]` with 50 commits.

    **The name pattern alone is NOT sufficient, and this is the interesting part.** Of the five
    bots in this repository's history, `dependabot[bot]`, `github-actions[bot]` and `snyk-bot` are
    caught by their names -- but `Copilot` and `Codex` are not, and they are AI agents committing
    real code. They are flagged only because GitHub's own API reports `type == "Bot"`.

    So the API field is load-bearing rather than a fallback for the regex. A pattern-only check
    would hand the agent two AI agents as page targets, and the number of repositories where that
    matters is going up, not down.
    """
    assert layer.bots, "real history contains automation; none was detected"
    by_pattern = {b.handle for b in layer.bots if github_ingest.is_bot(b.handle)}
    only_by_api = {b.handle for b in layer.bots} - by_pattern
    assert only_by_api, (
        "expected at least one bot detectable ONLY via GitHub's type field; if this ever becomes "
        "empty the api_type argument looks redundant and someone will delete it"
    )
    assert github_ingest.is_bot("anything", "Bot") is True

    paged = queries.who_to_page("api-gateway", client=gc).first()
    assert not github_ingest.is_bot(paged["top_author"] or "")
    for person in paged["actual_authors"]:
        assert not github_ingest.is_bot(person["handle"])

    kept = gc.run("MATCH (p:Person {is_bot: true})-[:AUTHORED]->(c:Commit) RETURN count(c) AS n").first()
    assert kept["n"] > 0, "bot commits are real history and must be kept, only excluded from paging"


def test_attributed_rows_exceed_unique_commits_by_design(layer) -> None:
    """One commit touching two mapped areas genuinely worked on both. Report both; do not 'fix' it."""
    assert layer.attributed_rows >= layer.unique_commits
    assert layer.unique_commits > 100


def test_path_map_prefixes_must_not_overlap() -> None:
    """GitHub's `?path=` filter is prefix-based, so `tests` beside `tests/flow` double-counts."""
    github_ingest._assert_disjoint_prefixes()
    with pytest.raises(ValueError, match="overlap"):
        github_ingest._assert_disjoint_prefixes((("tests", "a"), ("tests/flow", "b")))


def test_no_service_is_starved_of_real_commits(layer) -> None:
    """The first mapping used individual .rs files and gave 1-11 commits per service.

    The Rust rewrite is recent, so per-file history is thin; directories carry the volume.
    """
    distribution = github_ingest.service_commit_distribution(layer)
    assert distribution
    starved = [(s, c) for s, c, _p in distribution if c < 20]
    assert not starved, f"these services have too little real history to be credible: {starved}"


def test_the_demo_runs_with_no_network(layer) -> None:
    """Zero network calls on a cached fetch -- a rate limit or bad wifi cannot break the demo."""
    assert layer.stats.requests == 0
    assert layer.stats.cache_hits > 0


# --------------------------------------------------------------------------------------
# Verify, and the correctness/finding split
# --------------------------------------------------------------------------------------

def test_verify_separates_correctness_from_dataset_findings(gc: GraphClient) -> None:
    """Both started in one list, and a second tenant on a different repo 'failed' four checks.

    Two of those failures were legitimate: redis/redis has 152 contributors and therefore no single
    dominant author. Nothing was broken -- the suite was asserting a property of a dataset as
    though it were a property of the code.
    """
    report = ingest.verify(TENANT, client=gc)
    assert report.checks, "there should be correctness checks"
    assert report.findings, "there should be reported findings"
    failures = [(n, d) for n, ok, d in report.checks if not ok]
    assert not failures, f"correctness failures: {failures}"
    assert report.ok


def test_this_dataset_shows_the_headline_findings(gc: GraphClient) -> None:
    """Asserted for THIS tenant only, where it is a fact rather than an assumption."""
    report = ingest.verify(TENANT, client=gc)
    present = {name: ok for name, ok, _d in report.findings}
    assert any("bus factor" in n and ok for n, ok in present.items()), (
        "the key-person concentration that replaced the original framing is absent"
    )


def test_graph_algorithms_all_run(gc: GraphClient) -> None:
    for name in algorithms.ALGORITHMS:
        result = algorithms.run(name, client=gc)
        assert result.rows, f"{name} returned nothing"


def test_the_benchmark_leaves_the_graph_as_it_found_it(gc: GraphClient) -> None:
    """The benchmark wrote 6,567 episodes into the demo tenant and did not clean up.

    That took the graph from 1,644 nodes to 8,226 and filled vector recall with thousands of
    near-identical "bench worker N" episodes, which then compete with the agent's real conclusion
    for the top hit. A benchmark that changes the thing it measures is worse than none, because
    nothing fails -- the demo just quietly starts looking wrong.

    Writes still go to the real graph (an empty throwaway would measure an easier workload); they
    are deleted in a `finally` so an interrupted run cannot leave them behind.
    """
    from bench import concurrency

    before = gc.run("MATCH (n) RETURN count(n) AS n").first()["n"]
    concurrency.run(tenant=TENANT, workers=4, seconds=1.0, write_ratio=0.5)
    after = gc.run("MATCH (n) RETURN count(n) AS n").first()["n"]
    assert after == before, f"the benchmark leaked {after - before} node(s)"

    leftovers = gc.run(
        "MATCH (e:Episode) WHERE e.summary STARTS WITH $m RETURN count(*) AS n",
        {"m": concurrency.BENCH_MARKER},
    ).first()["n"]
    assert leftovers == 0


def test_pagerank_and_betweenness_disagree(gc: GraphClient) -> None:
    """The disagreement is the signal: popular dependency versus cut vertex."""
    top_pagerank = algorithms.critical_services(limit=1, client=gc).first()["service"]
    top_betweenness = algorithms.chokepoints(limit=1, client=gc).first()["service"]
    assert top_pagerank != top_betweenness, (
        "if these agree the two metrics add nothing over each other on this topology"
    )


def test_q1_window_is_anchored_to_the_alert_not_the_wall_clock(gc: GraphClient) -> None:
    """Q1 went silently blank on any graph older than `lookback_hours`.

    Deploy timestamps are placed relative to the alert at ingest time, but the window used
    `time.time() - lookback`. The two agree for six hours after an ingest and then stop agreeing,
    with no error: on a graph ingested 22 hours earlier Q1 returned ZERO rows and the headline demo
    question answered nothing. Pinned by moving the clock, which is the condition that broke it --
    an assertion on today's row count would pass against the buggy version too.
    """
    baseline = {r["service"] for r in queries.candidate_causes("api-gateway", client=gc).rows}
    assert baseline, "fixture graph has no candidates at all; this test cannot prove anything"

    ten_days = 10 * 24 * 3600
    real_time = queries._time.time
    try:
        queries._time.time = lambda: real_time() + ten_days  # type: ignore[assignment]
        moved = {r["service"] for r in queries.candidate_causes("api-gateway", client=gc).rows}
    finally:
        queries._time.time = real_time  # type: ignore[assignment]

    assert moved == baseline, (
        "the candidate set changed when only the wall clock moved, so the window is still "
        f"clock-relative: {sorted(baseline)} -> {sorted(moved)}"
    )


def test_q1_excludes_deploys_that_happened_after_the_alert(gc: GraphClient) -> None:
    """Anchoring on the alert gives Q1 an upper bound that wall-clock framing could not have.

    A deploy that lands AFTER the alert fired cannot be its cause, but `d.at >= $since` with no
    ceiling admitted one.
    """
    alert_at = gc.run(
        "MATCH (a:Alert)-[:FIRED_ON]->(:Service {name: $s}) RETURN max(a.at) AS at",
        {"s": "api-gateway"},
        label="alert_at",
        hops=0,
    ).first()["at"]
    assert alert_at is not None, "no alert on the alerting service; the anchor cannot be tested"

    rows = queries.candidate_causes("api-gateway", client=gc).rows
    late = [r for r in rows if r["deployed_at"] > alert_at]
    assert not late, f"these deploys happened after the alert and cannot have caused it: {late}"



@pytest.fixture
def isolated_env():
    """Snapshot and restore the whole process environment around a test.

    `config.load_dotenv` writes straight into `os.environ`, so `monkeypatch.delenv(k,
    raising=False)` records nothing to restore when the key was absent to begin with -- and the
    value the LOADER then set survives teardown. That leaked a fake `demo.falkordb.cloud` URL into
    a later test, which duly tried to open an SSL connection to a host that does not exist. The
    loader's own side effect is the thing monkeypatch cannot see, so the environment is saved and
    restored wholesale instead.
    """
    import os

    saved = dict(os.environ)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(saved)


def test_dotenv_file_is_actually_loaded(tmp_path, monkeypatch, isolated_env) -> None:
    """`.env.example` says "copy to `.env`", and for a while nothing read `.env`.

    This is the most dangerous shape of silent failure in the project, because the fallback is
    plausible rather than empty: a FalkorDB Cloud URL in `.env` was ignored, `FALKOR_URL` stayed
    unset, config fell back to its 127.0.0.1 default, and since local Docker is normally running
    the demo reported success -- against the wrong database. Nothing raised and no result was
    empty, so the only way to catch it is to assert the file is read.
    """
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# a comment line\n"
        "\n"
        "export FALKOR_URL=falkors://u:p%40ss@demo.falkordb.cloud:6379\n"
        'TENANT="globex"\n'
    )
    for key in ("FALKOR_URL", "TENANT"):
        monkeypatch.delenv(key, raising=False)

    applied = config.load_dotenv(env_file)
    assert "FALKOR_URL" in applied, (
        "the .env loader did not apply FALKOR_URL -- a Cloud credential placed in .env would be "
        "silently ignored and the demo would run against the local default instead"
    )

    conn = config._connection_from_env()
    assert conn.tls is True, "falkors:// from .env must turn TLS on; Cloud is TLS-only"
    assert conn.host == "demo.falkordb.cloud"
    assert conn.password == "p@ss", "percent-encoded password from .env must be decoded"
    assert "p@ss" not in conn.describe(), "describe() must never leak the password"


def test_real_environment_wins_over_dotenv(tmp_path, monkeypatch, isolated_env) -> None:
    """`FALKOR_URL=... make verify` must override a stale `.env`, not lose to it.

    The precedence is the reason the loader skips keys already present: a one-off run against a
    different instance has to be possible without editing a file.
    """
    env_file = tmp_path / ".env"
    env_file.write_text("TENANT=from_file\n")
    monkeypatch.setenv("TENANT", "from_real_env")

    applied = config.load_dotenv(env_file)
    assert "TENANT" not in applied
    import os

    assert os.environ["TENANT"] == "from_real_env"


def test_is_cloud_is_judged_by_host_not_by_tls(monkeypatch, isolated_env) -> None:
    """`is_cloud` was `return self.tls`, which mislabels every FREE Cloud instance.

    Only a live instance could show this: the free tier has no TLS, so a genuine Cloud endpoint
    reported `cloud=False` in `br doctor` -- the one place someone looks to confirm what they just
    connected to. The fix keys on host, which is advisory (a custom CNAME defeats it) and is never
    used for a security decision.
    """
    # Shaped exactly like a real free-tier Cloud endpoint, with every account-scoped segment
    # replaced -- this file is public, and the hostname carries a tenant identifier even though
    # it carries no credential. Only the `.cloud` suffix and the port shape matter to the test.
    cloud = "falkor://u:p@r-abc.instance-x.hc-y.ap-south-1.aws.0000000000aa.cloud:62065"
    monkeypatch.setenv("FALKOR_URL", cloud)
    conn = config._connection_from_env()
    assert conn.tls is False, "the free tier has no TLS; this endpoint must parse as plain"
    assert conn.is_cloud is True, (
        "a real Cloud endpoint reported is_cloud=False -- the old definition equated 'cloud' "
        "with 'encrypted', which is false for every free-tier instance"
    )

    monkeypatch.setenv("FALKOR_URL", "falkor://127.0.0.1:6379")
    assert config._connection_from_env().is_cloud is False, "local Docker must not report as Cloud"


def test_graph_endpoint_drops_nodes_with_no_visible_edge(gc: GraphClient) -> None:
    """A quarter of the rendered canvas was a detached band of Person dots.

    Found by looking at the PAGE, not at the API: every endpoint returned 200, and the payload was
    correct -- it simply contained nodes that could not be drawn as part of a graph. Commits are
    hidden by default and a contributor's only edge is usually `AUTHORED -> Commit`, so 23 of 73
    real people had no edge inside the default label set and rendered as free-floating points.

    The endpoint now drops zero-degree nodes and REPORTS how many, so the fact survives as a number
    instead of as 23 dots that read like a broken layout. `include_isolated=true` restores them.
    """
    from fastapi.testclient import TestClient

    from blast_radius import api as api_mod

    # Used as a context manager so startup runs and the key table is populated; the keys are
    # generated per-process when BR_API_KEYS is unset, so they must be read, not guessed.
    with TestClient(api_mod.app) as client:
        key = next(k for k, t in api_mod._API_KEYS.items() if t == TENANT)
        auth = {"Authorization": f"Bearer {key}"}

        default = client.get("/api/graph", headers=auth).json()
        everything = client.get(
            "/api/graph", params={"include_isolated": "true"}, headers=auth
        ).json()

    assert everything["node_count"] >= default["node_count"]
    assert everything["isolated_hidden"] == 0, "include_isolated must hide nothing"
    assert default["isolated_hidden"] == everything["node_count"] - default["node_count"], (
        "isolated_hidden must equal exactly what was removed, or the header's count lies"
    )

    connected = {e["source"] for e in default["edges"]} | {e["target"] for e in default["edges"]}
    stranded = [n["title"] for n in default["nodes"] if n["id"] not in connected]
    assert not stranded, (
        f"these nodes would render as detached points with no edge to anything: {stranded[:8]}"
    )


# --------------------------------------------------------------------------------------------
# MCP: the tools are only useful to an outside agent if they are DISCOVERABLE.
# --------------------------------------------------------------------------------------------


def _mcp_tools() -> dict:
    """Build the MCP server and return its tool map, skipping if fastmcp is absent."""
    pytest.importorskip("fastmcp", reason="optional: `pip install fastmcp`")
    import asyncio

    from blast_radius.mcp_server import build_server

    return asyncio.run(build_server(TENANT).get_tools())


def test_mcp_tools_declare_their_arguments() -> None:
    """The handlers were `def handler(**kwargs)`, which advertises a tool that takes NOTHING.

    An MCP client discovers arguments from each tool's JSON schema, and that schema is derived
    from the handler's signature -- so a `**kwargs` handler is undiscoverable even though it
    would happily accept a correct call. fastmcp refuses `**kwargs` outright (`ValueError:
    Functions with **kwargs are not supported as tools`), which is the kind version: the server
    died at startup instead of serving thirteen tools nobody could work out how to call.

    This asserts the SCHEMA, not the signature, because the schema is what a client actually
    sees -- and it is generated, so a new tool in `tools.py` is covered without touching this.
    """
    exposed = _mcp_tools()

    assert set(exposed) == set(tools.REGISTRY), (
        "the MCP surface drifted from the registry: "
        f"missing={sorted(set(tools.REGISTRY) - set(exposed))} "
        f"extra={sorted(set(exposed) - set(tools.REGISTRY))}"
    )

    for name, spec in sorted(tools.REGISTRY.items()):
        schema_ = exposed[name].parameters
        properties = schema_.get("properties", {})
        assert set(properties) == set(spec.args), (
            f"{name} advertises {sorted(properties)} but the registry declares "
            f"{sorted(spec.args)} -- a calling agent cannot guess the difference"
        )
        assert sorted(schema_.get("required", [])) == sorted(spec.required), (
            f"{name} required-args mismatch: schema says {schema_.get('required')}, "
            f"registry says {list(spec.required)}"
        )
        for arg, prop in properties.items():
            assert prop.get("type") in {"string", "integer", "number", "boolean"}, (
                f"{name}.{arg} is advertised as {prop.get('type')!r}; an untyped argument means "
                "the first wrong call fails inside a Cypher parameter instead of at the boundary"
            )
            assert prop.get("description"), (
                f"{name}.{arg} has no description in the schema, so a client choosing between "
                "tools has nothing to show the model"
            )


def test_mcp_descriptions_explain_why_it_is_a_graph_question() -> None:
    """Tool selection is the point: the description has to carry the registry's rationale."""
    exposed = _mcp_tools()
    for name, spec in sorted(tools.REGISTRY.items()):
        description = exposed[name].description or ""
        assert spec.summary in description, f"{name} lost its summary over MCP"
        assert spec.graph_rationale in description, (
            f"{name} does not tell a calling agent why this needs a graph"
        )


def test_generated_mcp_handlers_do_not_inherit_postponed_annotations() -> None:
    """`compile()` inherits `from __future__ import annotations` from its CALLER.

    `mcp_server.py` uses postponed annotations, so the handlers it generates with `exec` stored
    their annotations as strings; pydantic then re-evaluated those strings in a namespace with no
    `Field` and schema generation died with `NameError: name 'Field' is not defined`, reported
    against generated source that never mentions a missing name. `dont_inherit=True` fixes it.

    Asserted on the real annotation objects, because the symptom appeared two libraries away.
    """
    pytest.importorskip("fastmcp", reason="optional: `pip install fastmcp`")
    from blast_radius import mcp_server

    handler = mcp_server._handler_for(
        "candidate_causes", tools.REGISTRY["candidate_causes"], TENANT
    )
    annotations = handler.__annotations__
    assert not any(isinstance(v, str) for v in annotations.values()), (
        f"generated handler kept STRING annotations {annotations} -- compile() inherited the "
        "module's postponed-annotations flag, and pydantic cannot resolve them"
    )
    assert "service" in annotations and "lookback_hours" in annotations


def test_mcp_rejects_an_untyped_tool_argument_loudly() -> None:
    """A tool whose annotation is not whitelisted must fail at build time, not degrade to Any.

    An argument exposed as `Any` is advertised to every client as accepting anything, so the
    first wrong call fails deep inside a Cypher parameter rather than at the boundary.
    """
    pytest.importorskip("fastmcp", reason="optional: `pip install fastmcp`")
    from dataclasses import replace

    from blast_radius import mcp_server

    def weird(service: dict, **kw) -> None:  # noqa: ANN003 - deliberately unsupported
        return None

    broken = replace(tools.REGISTRY["blast_radius"], fn=weird)
    with pytest.raises(RuntimeError, match="not in the whitelist"):
        mcp_server._handler_for("blast_radius", broken, TENANT)
