"""Load the graph: synthetic topology plus the real human layer, in one pass.

This is where the two halves meet, and where the modelling has to be exactly right -- a mistake
here does not raise, it produces a graph that answers confidently and wrongly.

Three things in this file exist because of a bug that actually happened:

**`Deploy.id` is `<service>@<sha12>`, never the bare sha.** One real commit can touch two mapped
directories, so keying a deploy on the sha alone made `MERGE (d:Deploy {sha: $sha})` return the
*same node* for two services; the second `SET` overwrote the first, and the system reported `media`
as having been deployed with `notifications`' version. No error, no warning. Synthetic data never
produced a shared sha, so this was invisible until the data was real.

**Deploy timing is synthetic even though the commit is real.** The incident has to land inside the
alert window, and real commit dates are under no obligation to co-operate. So a Deploy is a real
author and a real sha with a placed timestamp, and `source` records exactly that -- `mixed` rather
than `github` or `synthetic`, because calling it either would be a lie in one direction or the other.

**Team membership is derived from real contributors, and bots are excluded.** Nothing in `data.py`
names a person. An earlier design seeded filler humans to pad the teams and they survived the switch
to real data, so the paging roster read `['GomezGab', 'dev00', 'dev03', 'dev09']`. A graph of real
humans must not contain an invented one, so none is created.

Every statement is a single batched `UNWIND`. The row counts printed are rows *sent*, not matches
*made* -- which is itself a lesson: an `UNWIND ... MATCH ... MERGE` that matches nothing still
reports its input length, so a silently-missing endpoint looked like a success. `verify()` exists
because the ingest report cannot be trusted to detect that on its own.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import data as synthetic
from . import embeddings, github_ingest, schema
from .config import SETTINGS
from .db import GraphClient
from .github_ingest import GITHUB_SOURCE, GitHubLayer

MIXED = "mixed"          # real sha + real author, placed timestamp
SYNTHETIC = synthetic.SYNTHETIC

# The three services whose deploys land inside the alert window. The agent has to discriminate
# between them, so they are deliberately not equally plausible: see data.DEPENDS_ON.
SUSPECT_WINDOW_MINUTES: dict[str, int] = {
    "pricing-legacy": 50,
    "media": 95,
    "notifications": 140,
}


@dataclass
class IngestReport:
    steps: list[tuple[str, int, int, float]] = field(default_factory=list)
    github: dict[str, Any] = field(default_factory=dict)
    schema_report: schema.SchemaReport | None = None
    elapsed_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)

    def add(self, what: str, rows: int, created: int, ms: float) -> None:
        self.steps.append((what, rows, created, ms))

    @property
    def rows(self) -> int:
        return sum(rows for _w, rows, _c, _ms in self.steps)

    @property
    def created(self) -> int:
        return sum(created for _w, _r, created, _ms in self.steps)

    @property
    def silent_steps(self) -> list[str]:
        """Steps that were handed rows and changed nothing -- the silent-failure signature."""
        return [what for what, rows, created, _ms in self.steps if rows and created == 0]

    def explain(self) -> str:
        width = max((len(w) for w, _r, _c, _m in self.steps), default=10)
        lines = [f"{'step'.ljust(width)}  {'sent':>6}  {'created':>8}  {'ms':>7}"]
        lines += [
            f"{what.ljust(width)}  {rows:>6}  {created:>8}  {ms:>7.2f}"
            + ("   <- SENT BUT CREATED NOTHING" if rows and created == 0 else "")
            for what, rows, created, ms in self.steps
        ]
        lines.append(
            f"{'TOTAL'.ljust(width)}  {self.rows:>6}  {self.created:>8}  {self.elapsed_ms:>7.2f}"
        )
        lines.append("")
        lines.append(
            "'sent' is rows handed to the engine; 'created' is what the engine reports it actually "
            "wrote.\nThey differ legitimately: MERGE is idempotent (a re-run creates 0), and one "
            "commit\ntouching two areas creates two edges from one row. 'created' 0 on a FIRST "
            "load is a bug."
        )
        if self.github:
            lines.append("")
            lines.append(str(self.github.get("summary", "")))
        if self.warnings:
            lines.append("")
            lines += [f"warning: {w}" for w in self.warnings]
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Deploys: real commits, placed in time
# --------------------------------------------------------------------------------------

def _build_deploys(layer: GitHubLayer, now: int) -> list[dict[str, Any]]:
    """One deploy per service that has real commits, keyed `<service>@<sha12>`.

    Two filters, both learned from real data rather than anticipated:

    **The author must not be a bot.** A bot cannot be the accountable deployer, and naming one hands
    the agent a page target it must never use.

    **The commit should be DISTINCTIVE to the service.** The first version took each service's most
    recent non-bot commit, which looks obviously correct and is wrong on real history: a repo-wide
    commit -- a branch migration, a formatting pass, a CI change -- touches many mapped directories
    at once and is therefore the most recent commit for *several* services simultaneously. All three
    suspects in the demo came out deployed by the same person with the same change description
    ("Migrate the default development branch to the..."), which leaves the agent three candidates it
    cannot tell apart, and makes the ranking look arbitrary rather than reasoned.

    So a commit touching exactly one mapped area is preferred, with the broad commit kept only as a
    fallback. Nothing is discarded -- the wide commit is still in the graph, still attributed to
    every area it touched -- but it is not allowed to impersonate a targeted release.
    """
    by_service: dict[str, list[Any]] = {}
    for commit in layer.commits.values():
        for service in commit.services:
            by_service.setdefault(service, []).append(commit)

    deploys: list[dict[str, Any]] = []
    for offset, (service, commits) in enumerate(sorted(by_service.items())):
        commits.sort(key=lambda c: -c.at)
        humans = [c for c in commits if not layer.people[c.author_login].is_bot]
        focused = [c for c in humans if len(c.services) == 1]
        chosen = next(iter(focused), None) or next(iter(humans), None) or commits[0]

        minutes_ago = SUSPECT_WINDOW_MINUTES.get(service)
        if minutes_ago is None:
            # Well outside the 6h lookback, spread deterministically so ordering is stable.
            minutes_ago = 8 * 60 + (offset * 97) % (190 * 60)

        major, minor = 2, 14
        deploys.append(
            {
                "id": f"{service}@{chosen.short_sha}",
                "service": service,
                "sha": chosen.sha,
                "short_sha": chosen.short_sha,
                "version": f"{service}@{major}.{minor}.{offset % 9}",
                "at": now - minutes_ago * 60,
                "status": "success",
                "actor": chosen.author_login,
                "commit_subject": chosen.subject,
                "touched_only_this": len(chosen.services) == 1,
                "in_alert_window": service in SUSPECT_WINDOW_MINUTES,
                "source": MIXED,
            }
        )
    return deploys


def _team_membership(layer: GitHubLayer, limit_per_service: int = 3) -> list[dict[str, Any]]:
    """Attach each real contributor to exactly ONE team: the one owning the service they commit to most.

    This is what creates the central finding -- membership follows ownership of *services*, while
    `WORKED_ON` follows commits, so a person can be the top expert on a service their own team does
    not own and the graph shows it structurally.

    **One team per person is load-bearing, not tidiness.** The first version keyed its dedup on
    `(handle, team)`, which let a prolific maintainer join every team that owned any service they
    touched: on real data `swilly22` came out a member of all five teams. That quietly destroyed the
    finding, because a person who is in every team can never be in the *wrong* one -- the
    `ownership_mismatch` flag was structurally unable to fire. Synthetic data never showed this,
    since invented contributors were conveniently scoped to one area each.

    The strongest candidate per service is still what drives selection, so the mismatch survives:
    a person's single team is decided by where their volume is highest, which is frequently not the
    team owning the service they are the top author on.
    """
    best: dict[str, tuple[int, str]] = {}  # handle -> (commits, team)
    for team, _channel, owned in synthetic.TEAMS:
        for service in owned:
            for handle, count in layer.top_humans_for(service, limit=limit_per_service):
                current = best.get(handle)
                # Tie broken on team name so the assignment is deterministic across runs.
                if current is None or (count, team) > (current[0], current[1]):
                    best[handle] = (count, team)
    return [
        {"handle": handle, "team": team}
        for handle, (_count, team) in sorted(best.items())
    ]


def _decision_authors(layer: GitHubLayer) -> list[dict[str, Any]]:
    """Attach each Decision to the top real contributor on the service it concerns.

    The earlier design gave decisions a handcrafted author who ceased to exist once humans became
    real. The `MATCH` found nothing, the ingest table still printed `1`, and the "why does this
    dependency exist" answer silently lost its author.
    """
    rows: list[dict[str, Any]] = []
    for decision in synthetic.DECISIONS:
        top = layer.top_humans_for(decision["about_service"], limit=1)
        if not top:
            continue
        rows.append({"decision": decision["id"], "handle": top[0][0]})
    return rows


def _episodes_from_incidents(now: int) -> list[dict[str, Any]]:
    """Seed semantic memory from past incidents, one vector-indexed Episode each.

    These are the only episodes present at ingest. Everything the agent learns later is written by
    `memory.write_episode`, which is what makes the cold-vs-warm proof meaningful: if ingest seeded
    the agent's own conclusions, "memory persists" would be unfalsifiable.
    """
    rows: list[dict[str, Any]] = []
    for inc in synthetic.INCIDENTS:
        text = (
            f"{inc['title']}. Root cause: {inc['root_cause']} "
            f"Resolution: {inc['resolution']} Runbook {inc['runbook']}."
        )
        rows.append(
            {
                "id": f"EP-{inc['id']}",
                "kind": "incident_retrospective",
                "text": text,
                "summary": inc["title"],
                "incident": inc["id"],
                "services": list(inc["affected"]),
                "at": now - inc["days_ago"] * 86400,
                "embedding": embeddings.embed(text),
                "source": SYNTHETIC,
            }
        )
    return rows


# --------------------------------------------------------------------------------------
# Ingest
# --------------------------------------------------------------------------------------

def ingest(
    tenant: str | None = None,
    *,
    use_github: bool = True,
    offline: bool = False,
    repo: str | None = None,
    reset: bool = True,
    client: GraphClient | None = None,
) -> IngestReport:
    """Build one tenant's graph from scratch. Idempotent; `reset=False` layers onto an existing one."""
    started = time.perf_counter()
    gc = client or GraphClient(tenant)
    report = IngestReport()
    now = int(time.time())

    if reset:
        gc.drop()
    report.schema_report = schema.ensure(gc)

    world = synthetic.build()

    def step(
        what: str,
        cypher: str,
        rows: list[dict[str, Any]],
        key: str = "rows",
        **extra: Any,
    ) -> None:
        if not rows:
            report.add(what, 0, 0, 0.0)
            return
        params: dict[str, Any] = {key: rows, **extra}
        result = gc.run(cypher, params, label=f"ingest:{what}", collect_provenance=False)
        created = result.writes.created
        report.add(what, len(rows), created, result.elapsed_ms)
        if created == 0 and reset:
            # On a fresh load every step must write something. Zero means the MATCH found no
            # endpoint -- the exact silent failure that let the spike report a nonexistent edge.
            report.warnings.append(
                f"{what}: {len(rows)} rows sent, engine created nothing "
                "(a MATCH in this statement found no endpoint)"
            )

    # -- synthetic topology ------------------------------------------------------------

    step(
        "services",
        """
        UNWIND $rows AS row
        MERGE (s:Service {name: row.name})
        SET s.tier = row.tier, s.description = row.description,
            s.language = row.language, s.source = row.source
        """,
        world.services,
    )

    step(
        "databases",
        """
        UNWIND $rows AS row
        MERGE (d:Database {name: row.name})
        SET d.engine = row.engine, d.source = row.source
        """,
        world.databases,
    )

    step(
        "endpoints",
        """
        UNWIND $rows AS row
        MERGE (e:Endpoint {path: row.path})
        SET e.description = row.description, e.source = row.source
        """,
        world.endpoints,
    )

    step(
        "customers",
        """
        UNWIND $rows AS row
        MERGE (c:Customer {name: row.name})
        SET c.mrr = row.mrr, c.plan = row.plan, c.source = row.source
        """,
        world.customers,
    )

    step(
        "teams",
        """
        UNWIND $rows AS row
        MERGE (t:Team {name: row.name})
        SET t.channel = row.channel, t.source = row.source
        """,
        world.teams,
    )

    step(
        "dependencies",
        """
        UNWIND $rows AS row
        MERGE (dep:Dependency {id: row.id})
        SET dep.upstream = row.upstream, dep.downstream = row.downstream,
            dep.criticality = row.criticality, dep.source = row.source
        """,
        world.dependencies,
    )

    step(
        "DEPENDS_ON",
        """
        UNWIND $rows AS row
        MATCH (a:Service {name: row.upstream}), (b:Service {name: row.downstream})
        MERGE (a)-[:DEPENDS_ON]->(b)
        """,
        world.depends_on,
    )

    step(
        "READS_FROM",
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: row.service}), (d:Database {name: row.database})
        MERGE (s)-[:READS_FROM]->(d)
        """,
        world.reads_from,
    )

    step(
        "WRITES_TO",
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: row.service}), (d:Database {name: row.database})
        MERGE (s)-[:WRITES_TO]->(d)
        """,
        world.writes_to,
    )

    step(
        "EXPOSES",
        """
        UNWIND $rows AS row
        MATCH (s:Service {name: row.service}), (e:Endpoint {path: row.path})
        MERGE (s)-[:EXPOSES]->(e)
        """,
        world.exposes,
    )

    step(
        "USES",
        """
        UNWIND $rows AS row
        MATCH (c:Customer {name: row.customer}), (e:Endpoint {path: row.path})
        MERGE (c)-[:USES]->(e)
        """,
        world.uses,
    )

    step(
        "OWNS",
        """
        UNWIND $rows AS row
        MATCH (t:Team {name: row.team}), (s:Service {name: row.service})
        MERGE (t)-[:OWNS]->(s)
        """,
        world.owns,
    )

    # -- institutional memory ----------------------------------------------------------

    step(
        "runbooks",
        """
        UNWIND $rows AS row
        MERGE (r:Runbook {id: row.id})
        SET r.title = row.title, r.owner_team = row.owner_team,
            r.expected_mttr_minutes = row.expected_mttr_minutes,
            r.steps = row.steps, r.source = row.source
        """,
        world.runbooks,
    )

    step(
        "incidents",
        """
        UNWIND $rows AS row
        MERGE (i:Incident {id: row.id})
        SET i.title = row.title, i.severity = row.severity, i.root_cause = row.root_cause,
            i.resolution = row.resolution, i.mttr_minutes = row.mttr_minutes,
            i.at = $now - row.days_ago * 86400, i.source = row.source
        """,
        world.incidents,
        now=now,
    )

    step(
        "AFFECTED",
        """
        UNWIND $rows AS row
        UNWIND row.affected AS service_name
        MATCH (i:Incident {id: row.id}), (s:Service {name: service_name})
        MERGE (i)-[:AFFECTED]->(s)
        """,
        world.incidents,
    )

    step(
        "RESOLVED_BY",
        """
        UNWIND $rows AS row
        MATCH (i:Incident {id: row.id}), (r:Runbook {id: row.runbook})
        MERGE (i)-[:RESOLVED_BY]->(r)
        """,
        world.incidents,
    )

    step(
        "decisions",
        """
        UNWIND $rows AS row
        MERGE (d:Decision {id: row.id})
        SET d.title = row.title, d.decision = row.decision, d.rationale = row.rationale,
            d.trade_off = row.trade_off, d.revisit_after_days = row.revisit_after_days,
            d.at = $now - row.days_ago * 86400, d.source = row.source
        """,
        world.decisions,
        now=now,
    )

    step(
        "JUSTIFIES",
        """
        UNWIND $rows AS row
        MATCH (d:Decision {id: row.id}), (dep:Dependency {id: row.dependency_id})
        MERGE (d)-[:JUSTIFIES]->(dep)
        """,
        world.decisions,
    )

    step(
        "alert",
        """
        UNWIND $rows AS row
        MERGE (a:Alert {id: row.id})
        SET a.metric = row.metric, a.value = row.value, a.threshold = row.threshold,
            a.window_minutes = row.window_minutes, a.message = row.message,
            a.lookback_hours = row.lookback_hours, a.at = $now, a.source = row.source
        WITH a, row
        MATCH (s:Service {name: row.service})
        MERGE (a)-[:FIRED_ON]->(s)
        """,
        [world.alert],
        now=now,
    )

    # -- the real human layer ----------------------------------------------------------

    if use_github:
        layer = github_ingest.fetch(repo, offline=offline)
        report.github = {
            "repo": layer.repo,
            "summary": layer.explain(),
            "unique_commits": layer.unique_commits,
            "attributed_rows": layer.attributed_rows,
            "humans": len(layer.humans),
            "bots": len(layer.bots),
            "pulls": len(layer.pulls),
            "median_utc_offset": layer.median_utc_offset(),
            "cache": github_ingest.cache_status(),
        }
        if not layer.commits:
            report.warnings.append(
                "GitHub layer is empty -- the human half of the graph is missing. "
                "Run `make github` once with network to populate the cache."
            )
        _ingest_github(gc, layer, now, step, report)
    else:
        report.github = {"skipped": True}
        report.warnings.append(
            "Ran with --no-github: the graph has no real people, commits or pull requests, so "
            "'who actually knows this code' cannot be answered."
        )

    # -- semantic memory seed ----------------------------------------------------------

    step(
        "episodes",
        """
        UNWIND $rows AS row
        MERGE (e:Episode {id: row.id})
        SET e.kind = row.kind, e.text = row.text, e.summary = row.summary,
            e.at = row.at, e.incident = row.incident,
            e.embedding = vecf32(row.embedding), e.source = row.source
        WITH e, row
        UNWIND row.services AS service_name
        MATCH (s:Service {name: service_name})
        MERGE (e)-[:ABOUT]->(s)
        """,
        _episodes_from_incidents(now),
    )

    report.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return report


def _ingest_github(gc: GraphClient, layer: GitHubLayer, now: int, step: Any, report: IngestReport) -> None:
    """The real layer. Separated only to keep `ingest` readable."""

    people = [
        {
            "handle": person.handle,
            "is_bot": person.is_bot,
            "commit_count": person.commit_count,
            "utc_offset": person.utc_offset,
            "avatar_url": person.avatar_url,
            "profile_url": person.profile_url,
            "source": GITHUB_SOURCE,
        }
        for person in layer.people.values()
    ]
    step(
        "people (real)",
        """
        UNWIND $rows AS row
        MERGE (p:Person {handle: row.handle})
        SET p.is_bot = row.is_bot, p.commit_count = row.commit_count,
            p.utc_offset = row.utc_offset, p.avatar_url = row.avatar_url,
            p.profile_url = row.profile_url, p.source = row.source
        """,
        people,
    )

    commits = [
        {
            "sha": commit.sha,
            "short_sha": commit.short_sha,
            "msg": commit.subject,
            "at": commit.at,
            "author": commit.author_login,
            "services": commit.services,
            "source": GITHUB_SOURCE,
        }
        for commit in layer.commits.values()
    ]
    step(
        "commits (real)",
        """
        UNWIND $rows AS row
        MERGE (c:Commit {sha: row.sha})
        SET c.short_sha = row.short_sha, c.msg = row.msg, c.at = row.at, c.source = row.source
        """,
        commits,
    )

    step(
        "AUTHORED",
        """
        UNWIND $rows AS row
        MATCH (p:Person {handle: row.author}), (c:Commit {sha: row.sha})
        MERGE (p)-[:AUTHORED]->(c)
        """,
        commits,
    )

    step(
        "TOUCHED",
        """
        UNWIND $rows AS row
        UNWIND row.services AS service_name
        MATCH (c:Commit {sha: row.sha}), (s:Service {name: service_name})
        MERGE (c)-[:TOUCHED]->(s)
        """,
        commits,
    )

    # WORKED_ON is materialised rather than aggregated at read time. At demo scale the two are
    # indistinguishable (0.61 ms vs 0.59 ms), so speed is not the argument: "who knows this code"
    # is a CLAIM the system makes, and a claim deserves an edge that can be inspected and disputed.
    worked_on = [
        {"handle": handle, "service": service, "commits": count}
        for (handle, service), count in layer.worked_on.items()
    ]
    step(
        "WORKED_ON",
        """
        UNWIND $rows AS row
        MATCH (p:Person {handle: row.handle}), (s:Service {name: row.service})
        MERGE (p)-[w:WORKED_ON]->(s)
        SET w.commits = row.commits, w.derived_from = 'AUTHORED+TOUCHED'
        """,
        worked_on,
    )

    pulls = [
        {
            "number": pr.number,
            "title": pr.title,
            "author": pr.author_login,
            "merged_at": pr.merged_at,
            "merge_sha": pr.merge_sha,
            "source": GITHUB_SOURCE,
        }
        for pr in layer.pulls
    ]
    step(
        "pull requests (real)",
        """
        UNWIND $rows AS row
        MERGE (pr:PullRequest {number: row.number})
        SET pr.title = row.title, pr.merged_at = row.merged_at, pr.source = row.source
        """,
        pulls,
    )

    step(
        "INTRODUCED",
        """
        UNWIND $rows AS row
        MATCH (pr:PullRequest {number: row.number}), (c:Commit {sha: row.merge_sha})
        MERGE (pr)-[:INTRODUCED]->(c)
        """,
        [row for row in pulls if row["merge_sha"]],
    )

    deploys = _build_deploys(layer, now)
    step(
        "deploys (mixed)",
        """
        UNWIND $rows AS row
        MERGE (d:Deploy {id: row.id})
        SET d.sha = row.sha, d.short_sha = row.short_sha, d.version = row.version,
            d.at = row.at, d.status = row.status, d.actor = row.actor,
            d.commit_subject = row.commit_subject, d.in_alert_window = row.in_alert_window,
            d.touched_only_this = row.touched_only_this,
            d.source = row.source
        """,
        deploys,
    )

    step(
        "DEPLOYS",
        """
        UNWIND $rows AS row
        MATCH (d:Deploy {id: row.id}), (s:Service {name: row.service})
        MERGE (d)-[:DEPLOYS]->(s)
        """,
        deploys,
    )

    step(
        "CONTAINS",
        """
        UNWIND $rows AS row
        MATCH (d:Deploy {id: row.id}), (c:Commit {sha: row.sha})
        MERGE (d)-[:CONTAINS]->(c)
        """,
        deploys,
    )

    membership = _team_membership(layer)
    step(
        "MEMBER_OF",
        """
        UNWIND $rows AS row
        MATCH (p:Person {handle: row.handle}), (t:Team {name: row.team})
        WHERE p.is_bot = false
        MERGE (p)-[:MEMBER_OF]->(t)
        """,
        membership,
    )
    if not membership:
        report.warnings.append("no team membership created -- every paging query will come back empty")

    authors = _decision_authors(layer)
    step(
        "decision authors",
        """
        UNWIND $rows AS row
        MATCH (d:Decision {id: row.decision}), (p:Person {handle: row.handle})
        MERGE (p)-[:AUTHORED]->(d)
        """,
        authors,
    )
    if len(authors) < len(synthetic.DECISIONS):
        report.warnings.append(
            f"only {len(authors)} of {len(synthetic.DECISIONS)} decisions got a real author; "
            "the rest concern services with no mapped commits"
        )


# --------------------------------------------------------------------------------------
# Verification -- because the ingest report counts rows SENT, not matches MADE
# --------------------------------------------------------------------------------------

@dataclass
class VerifyReport:
    """Two kinds of statement, deliberately not mixed.

    **Correctness checks** must hold for every tenant on every dataset. A failure means the ingest
    or the model is broken, and `ok` is False.

    **Findings** are claims about what one particular dataset happens to show -- the ownership
    mismatch, the bus factor. They are true of `FalkorDB/FalkorDB` and need not be true of another
    repository, so they are REPORTED rather than asserted.

    The distinction was forced by getting it wrong. Both kinds started in one list, and pointing the
    second tenant at `redis/redis` "failed" four checks -- two of them legitimately, because redis
    has 152 contributors and therefore no single dominant author. Nothing was broken; the test was
    asserting a property of a dataset as though it were a property of the code. A suite that cries
    wolf on correct behaviour gets ignored, which costs more than the check was ever worth.
    """

    checks: list[tuple[str, bool, str]] = field(default_factory=list)
    findings: list[tuple[str, bool, str]] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Correctness only. A finding that does not hold on this dataset is not a failure."""
        return all(passed for _n, passed, _d in self.checks)

    def explain(self) -> str:
        lines = [
            f"{'PASS' if passed else 'FAIL'}  {name}  {detail}"
            for name, passed, detail in self.checks
        ]
        lines.append(f"\n{sum(1 for _n, p, _d in self.checks if p)}/{len(self.checks)} correctness checks passed")
        if self.findings:
            lines.append("\nfindings in THIS dataset (reported, not asserted):")
            lines += [
                f"{'YES' if present else ' no'}   {name}  {detail}"
                for name, present, detail in self.findings
            ]
        return "\n".join(lines)


def verify(tenant: str | None = None, *, client: GraphClient | None = None) -> VerifyReport:
    """Assert the graph can actually answer the demo, rather than merely containing rows.

    Every check here corresponds to something that was once silently broken: an `UNWIND ... MATCH`
    that matched nothing still reported its input length, so the ingest table said success while
    the graph was missing an entire edge type.
    """
    gc = client or GraphClient(tenant)
    report = VerifyReport()

    def check(name: str, cypher: str, params: dict[str, Any], predicate: Any, describe: Any) -> None:
        rows = gc.run(cypher, params, label=f"verify:{name}", collect_provenance=False).rows
        report.checks.append((name, bool(predicate(rows)), describe(rows)))

    def finding(name: str, cypher: str, params: dict[str, Any], predicate: Any, describe: Any) -> None:
        """Record what this dataset shows. Never affects `ok` -- see VerifyReport."""
        rows = gc.run(cypher, params, label=f"finding:{name}", collect_provenance=False).rows
        report.findings.append((name, bool(predicate(rows)), describe(rows)))

    check(
        "edge service reaches the suspect in 2 hops",
        """
        MATCH path = (:Service {name: 'api-gateway'})-[:DEPENDS_ON*1..4]->(s:Service {name: 'pricing-legacy'})
        RETURN min(length(path)) AS hops
        """,
        {},
        lambda rows: rows and rows[0].get("hops") == 2,
        lambda rows: f"hops={rows[0].get('hops') if rows else 'none'}",
    )

    check(
        "revenue reaches the edge via endpoints",
        """
        MATCH (c:Customer)-[:USES]->(:Endpoint)<-[:EXPOSES]-(:Service {name: 'api-gateway'})
        RETURN count(DISTINCT c) AS customers, sum(DISTINCT c.mrr) AS mrr
        """,
        {},
        lambda rows: rows and rows[0].get("customers") == len(synthetic.CUSTOMERS),
        lambda rows: f"customers={rows[0].get('customers') if rows else 0}, "
                     f"mrr={rows[0].get('mrr') if rows else 0} "
                     f"(python says {synthetic.mrr_at_risk_for('api-gateway')})",
    )

    check(
        "three deploys sit inside the alert window",
        """
        MATCH (d:Deploy)-[:DEPLOYS]->(s:Service)
        WHERE d.in_alert_window = true
        RETURN collect(s.name) AS services
        """,
        {},
        lambda rows: rows and sorted(rows[0].get("services") or []) == sorted(SUSPECT_WINDOW_MINUTES),
        lambda rows: f"services={sorted(rows[0].get('services') or []) if rows else []}",
    )

    check(
        "no Deploy node serves two services",
        """
        MATCH (d:Deploy)-[:DEPLOYS]->(s:Service)
        WITH d.id AS deploy_id, count(DISTINCT s.name) AS services
        WHERE services > 1
        RETURN count(*) AS collisions
        """,
        {},
        lambda rows: not rows or rows[0].get("collisions") == 0,
        lambda rows: f"collisions={rows[0].get('collisions') if rows else 0}",
    )

    check(
        "no invented person is in a real team",
        """
        MATCH (p:Person)-[:MEMBER_OF]->(:Team)
        WHERE p.source <> 'github' OR p.is_bot = true
        RETURN collect(DISTINCT p.handle) AS offenders
        """,
        {},
        lambda rows: not rows or not (rows[0].get("offenders") or []),
        lambda rows: f"offenders={(rows[0].get('offenders') or []) if rows else []}",
    )

    check(
        "every decision has an author",
        """
        MATCH (d:Decision)
        OPTIONAL MATCH (p:Person)-[:AUTHORED]->(d)
        WITH d.id AS decision, count(p) AS authors
        WHERE authors = 0
        RETURN collect(decision) AS orphans
        """,
        {},
        lambda rows: not rows or not (rows[0].get("orphans") or []),
        lambda rows: f"orphans={(rows[0].get('orphans') or []) if rows else []}",
    )

    check(
        "episodes are vector-indexed and non-empty",
        "MATCH (e:Episode) WHERE e.embedding IS NOT NULL RETURN count(*) AS episodes",
        {},
        lambda rows: rows and (rows[0].get("episodes") or 0) >= len(synthetic.INCIDENTS),
        lambda rows: f"episodes={rows[0].get('episodes') if rows else 0}",
    )

    check(
        "nobody is a member of two teams",
        """
        MATCH (p:Person)-[:MEMBER_OF]->(t:Team)
        WITH p.handle AS handle, count(DISTINCT t) AS teams
        WHERE teams > 1
        RETURN collect(handle) AS offenders
        """,
        {},
        lambda rows: not rows or not (rows[0].get("offenders") or []),
        lambda rows: f"offenders={(rows[0].get('offenders') or []) if rows else []}",
    )

    finding(
        "ownership mismatch -- the owning team's top member is not the top author",
        """
        MATCH (t:Team)-[:OWNS]->(s:Service)
        MATCH (expert:Person {is_bot: false})-[w:WORKED_ON]->(s)
        WITH s.name AS service, t.name AS owner, expert, w
        ORDER BY w.commits DESC
        WITH service, owner, collect({h: expert.handle, c: w.commits})[0] AS top
        MATCH (p:Person {handle: top.h})-[:MEMBER_OF]->(pt:Team)
        WITH service, owner, top, pt.name AS author_team
        WHERE author_team <> owner
        RETURN count(*) AS mismatched, collect(service)[0..4] AS examples
        """,
        {},
        lambda rows: rows and (rows[0].get("mismatched") or 0) > 0,
        lambda rows: f"mismatched={rows[0].get('mismatched') if rows else 0} services, "
                     f"e.g. {(rows[0].get('examples') or []) if rows else []}",
    )

    finding(
        "expertise concentrated in one person (bus factor)",
        """
        MATCH (p:Person {is_bot: false})-[w:WORKED_ON]->(s:Service)
        WITH s.name AS service, p.handle AS handle, w.commits AS commits
        ORDER BY commits DESC
        WITH service, collect(handle)[0] AS top
        RETURN top AS person, count(*) AS services_topped
        ORDER BY services_topped DESC
        LIMIT 1
        """,
        {},
        lambda rows: rows and (rows[0].get("services_topped") or 0) >= 5,
        lambda rows: f"{rows[0].get('person') if rows else '?'} tops "
                     f"{rows[0].get('services_topped') if rows else 0} services",
    )

    check(
        "real commits are present",
        "MATCH (c:Commit) WHERE c.source = 'github' RETURN count(*) AS commits",
        {},
        lambda rows: rows and (rows[0].get("commits") or 0) > 100,
        lambda rows: f"real commits={rows[0].get('commits') if rows else 0}",
    )

    return report


def ingest_all(
    tenants: tuple[str, ...] = ("acme", "globex"),
    *,
    use_github: bool = True,
    offline: bool = False,
    repos: dict[str, str] | None = None,
) -> dict[str, IngestReport]:
    """Load every tenant. Separate graphs, so this is the multi-tenancy demonstration.

    A per-tenant repo makes the tenant switch far stronger: two tenants then hold two genuinely
    different real codebases, rather than the same one twice.
    """
    mapping = repos or {}
    return {
        tenant: ingest(
            tenant,
            use_github=use_github,
            offline=offline,
            repo=mapping.get(tenant) or SETTINGS.github_repo,
        )
        for tenant in tenants
    }
