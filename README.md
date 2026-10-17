# Blast Radius

# blast-radius

**An on-call engineer's judgement, as a graph traversal.**

At 2am an alert says `api-gateway 5xx ratio 37%`. The senior engineer who can resolve it in ten
minutes does so by holding a map in their head: what depends on what, what shipped recently, which
customers are behind which endpoint, who actually knows that code, and what fixed it last time.
That map is a graph, and the reason one person is irreplaceable is that nobody wrote it down.

This writes it down, and then walks it. Three agents hand work to each other through the graph, and
every conclusion ships with the Cypher that produced it.

```
                 ┌──────────┐  severity, blast radius
    alert  ──▶   │  TRIAGE  │  "sev1 because tier+revenue; 6 customers, $971k"
                 └────┬─────┘
                      │  handoff written to the graph as (:AgentRun)-[:PRODUCED]->(:Episode)
                 ┌────▼──────┐  3 candidates ranked, 2 ruled out WITH REASONS
                 │ DIAGNOSIS │  "pricing-legacy: 2 hops, critical-tier, newest in window"
                 └────┬──────┘
                      │  handoff
                 ┌────▼─────┐  owner vs real author, runbook from the last incident
                 │  COMMS   │  "propose RB-07; page #payments-oncall, pull in swilly22"
                 └──────────┘
```

---

## Run it

Needs Docker and Python 3.11+. Nothing else.

```bash
make demo          # install, start FalkorDB, vendor assets, ingest, verify
make api           # then open http://127.0.0.1:8099
```

If anything is wrong, `make doctor` says what, in one command. `make help` lists every target.

The first `make demo` makes **31 GitHub API requests** (unauthenticated, no token needed) and caches
them. Every run after that makes **zero network calls** — a rate limit or bad venue wifi cannot
break the demo.

### In a terminal, without a browser

```bash
make ask            # all six questions, each with its Cypher
make run            # the three-agent loop
make memory-proof   # memory survives, proven across three OS processes
make test           # 53 regression tests
make mcp-verify     # prove the MCP transport: spawn the server, speak the protocol
make bench          # concurrency numbers
make provenance     # what is real and what is synthetic
```

---

## Tracks entered

**Track 01 — Agents That Act on Connected Data** and **Track 02 — Agent Memory and Coordination.**

Track 01: the agent decides and acts from multi-hop traversal, calls FalkorDB's own graph
algorithms as tools, exposes the same thirteen tools over **MCP** so any client can select among
them, and every step carries the Cypher that produced it. Track 02: episodic,
semantic and procedural memory, persisted in the graph and proven to survive across separate OS
processes, with handoffs between three agents recorded as nodes and one isolated graph per tenant.

Not entering Track 03 (Company Brain). Parts of it are present — `Decision` nodes trace an
architectural choice to the real contributor who made it, `OWNS` and `WORKED_ON` separate who owns
a service from who actually writes it — but there is no ingestion from tickets or chat and no
incremental refresh, so it would be a weaker entry than the two above rather than a third strong
one.

## Graph data model

Sixteen node labels, twenty relationship types, 22 indices and one vector index. Counts below are
the live `acme` graph, so the shape and the volume are the same thing.

| relationship | from → to | count | what it carries |
|---|---|---|---|
| `DEPENDS_ON` | Service → Service | 30 | the dependency topology the blast radius walks |
| `EXPOSES` | Service → Endpoint | 8 | how a request enters |
| `USES` | Customer → Endpoint | 23 | revenue reached by traversal, not by a lookup field |
| `READS_FROM` / `WRITES_TO` | Service → Database | 15 / 10 | data-layer coupling |
| `OWNS` | Team → Service | 24 | the org chart's claim |
| `MEMBER_OF` | Person → Team | 7 | who is on that team |
| `WORKED_ON` | Person → Service | 211 | the commit history's claim — compared against `OWNS` |
| `AUTHORED` | Person → Commit | 1,256 | real GitHub history |
| `TOUCHED` | Commit → Service | 2,466 | which area each commit changed |
| `INTRODUCED` | PullRequest → Commit | 78 | the PR a change arrived in |
| `CONTAINS` / `DEPLOYS` | Deploy → Commit / Service | 16 / 16 | what shipped, and where |
| `FIRED_ON` | Alert → Service | 1 | anchors the time window (see `candidate_causes`) |
| `AFFECTED` | Incident → Service | 9 | incident history, reached structurally |
| `RESOLVED_BY` | Incident → Runbook | 4 | procedural memory via the incident, not by name |
| `AUTHORED` | Person → Decision | 3 | a decision traced to a real person |
| `JUSTIFIES` | Decision → Dependency | 3 | *why* an edge exists, attached to the edge |
| `ABOUT` | Episode → Service | 25 | episodic memory, attached to what it concerned |
| `PRODUCED` | AgentRun → Episode | 8 | which run learned it |

Three labels exist only to make the agent's own work queryable: `Episode` (what a run concluded),
`AgentRun` (the run itself, with its handoffs) and `Dependency` (an edge reified so a `Decision`
can point at it). Every node also carries a `source` property — `github`, `synthetic`, `agent` or
`mixed` — which is what makes provenance a query rather than a paragraph.

## Use of AI coding assistants

Disclosed per rule 11. This project was built in a pair-programming loop with an AI assistant
(Kiro): the assistant wrote the bulk of the module code and the tests, and every design decision,
schema choice and bug diagnosis in this README was reached by running the thing against a live
FalkorDB and reading what came back. The sixteen bugs listed below were found that way rather than
by inspection, which is also why each one has a regression test pinning it instead of a comment.
No code here is unverified generation: the suite runs against a real database, both locally and
against FalkorDB Cloud.

## Why this is a graph problem, not a graph-shaped UI

The hackathon's rule is that removing the graph database should break the product. Here is the
query behind the first question the agent asks:

```cypher
MATCH (alerting:Service {name: $service})
MATCH path = (alerting)-[:DEPENDS_ON*1..3]->(suspect:Service)
WITH suspect.name AS suspect_name, min(length(path)) AS hops
MATCH (s:Service {name: suspect_name})<-[:DEPLOYS]-(d:Deploy)
WHERE d.at >= $since
RETURN s.name, hops, d.version, d.commit_subject
ORDER BY hops ASC, d.at DESC
```

"Anything the alerting service depends on, to any depth, that shipped in the last six hours." The
depth is a property of the data, not of the query. In SQL this is a recursive CTE whose join
explodes before it answers; in a vector store the question cannot be posed at all, because no
embedding of a service encodes *what depends on it*.

The same holds for the three graph algorithms the agent uses. `algo.pageRank` and
`algo.betweenness` disagree on this topology — `analytics` ranks highest by PageRank, `catalog` by
betweenness — and the disagreement is the signal: one is a popular dependency, the other is a cut
vertex whose failure partitions the system. There is no embedding for that.

**Every answer carries its own evidence.** The API returns the Cypher, the parameters, the hop
budget, the wall time and the nodes walked alongside each result, and the UI renders it. Click any
step in the browser and you are reading the query, not a description of it.

---

## Data provenance

Mixed, and the split is forced rather than chosen. Rule 6 permits public, synthetic, or your own
data — so anything genuinely public is taken from the real world, and anything that would require
private data stays synthetic.

| layer | source | why |
|---|---|---|
| `Person`, `Commit`, `PullRequest`, `AUTHORED`, `TOUCHED`, `WORKED_ON`, `MEMBER_OF` | **real** — public GitHub API | public, and verifiable by anyone reading this |
| `Service` / `DEPENDS_ON` topology | synthetic | no company publishes its dependency graph |
| `Customer`, MRR | synthetic | revenue per customer is restricted data |
| `Incident`, `Runbook`, `Decision`, `Alert` | synthetic | nobody publishes their outage history |
| `Deploy` timing | synthetic timing on a **real** commit and **real** author | the incident has to land in the alert window; real commit dates do not oblige |

"All real data" is not an option a rule-compliant submission has — three of those rows would
require private data. The useful claim is the split, stated precisely. So it is **a query, not a
paragraph**:

```cypher
MATCH (n) WHERE n.source IS NOT NULL
RETURN labels(n)[0] AS label, n.source AS source, count(*) AS nodes
ORDER BY label, source
```

2.6 ms. `make provenance`, or the chips at the top of the UI. Every node carries its own `source`,
so this output cannot go stale.

**The real layer, measured:** 1,256 unique commits, 2,466 service-attributed rows, 68 human
contributors, 5 bots, 200 pull requests, from `FalkorDB/FalkorDB`. **No email addresses are
stored** — only logins. A test asserts that.

Point it at any public repository with `GITHUB_REPO=owner/name`: the directory-to-service map is
discovered from the repo's own tree in one API call.

---

## What the real data changed

The project was designed around a finding: *the team that owns a service is not the team that
writes it.* That mismatch does exist — on **10 of 16 services**. But on real history every one of
those ten points at the same person. `swilly22` is the top author on **14 of 16** mapped areas.

So the honest headline is not misaligned ownership, it is **bus factor 1**, which is a far more
serious operational fact. No hand-written fixture would have produced it: when you invent
contributors you spread them evenly, because that is what looks reasonable while typing.

Both readings are kept. `who_to_page` reports the per-service mismatch; `key_person_risk` reports
the concentration; the agent cites whichever applies.

The demo moment, on `api-gateway`:

> Owning team **Core Platform** — escalate to `#core-platform`.
> But the top author is **swilly22**, with **77 commits**, and he is on **Payments Platform**.

Those numbers are checkable against GitHub by anyone who doubts them. That is the whole difference
between a finding and a number someone typed into a fixture.

---

## Memory that is actually proven

`make memory-proof` runs **three separate operating-system processes**:

```
phase A   fresh process   wipe agent memory, ask   -> 0 hits    [cold]
phase B   fresh process   run the agent            -> writes 1 episode
phase C   fresh process   ask the same question    -> 1 hit     [warm]
```

A and C share no memory, no file handles, no interpreter. The only thing they have in common is the
database, so if C finds what B wrote, the memory is in FalkorDB and nowhere else. A single process
reading its own cache would prove only that the cache works.

The wipe is scoped to `kind='agent_conclusion'`, so the seeded incident history stays in place and
phase A is a real zero rather than a trivially empty graph.

Three kinds of memory, which are different things rather than three names for one:

- **semantic** — vector search over episode text, on FalkorDB's own vector index
- **episodic** — what this system saw and concluded, timestamped, attached to the services involved
- **procedural** — the runbook, reached *through* the incident it resolved, not tagged by name

Structural and semantic recall both run, because they fail differently: vector recall finds a
lookalike on an unrelated service, structural recall misses a genuinely similar incident elsewhere.
The agent runs both and reports which produced what.

---

## MCP: the same thirteen tools, for any agent

The graph is not context for one agent in this repository — it is a context source any MCP client
can attach to. Claude Desktop, an IDE assistant or another orchestrator gets the identical thirteen
tools, the identical validation, and the identical evidence envelope the agent loop uses.

```bash
make mcp            # run the server (stdio transport, no network port)
make mcp-verify     # prove it: spawn the server, speak the protocol, check every tool
```

`make mcp-verify` is a real transport test, not an import check. It launches
`python -m blast_radius.mcp_server` as a separate OS process, performs the MCP `initialize`
handshake, lists the tools, asserts the exposed set equals `tools.REGISTRY` exactly, calls tools
with and without arguments, reads the `graph://provenance` resource, and checks that a **bad**
argument comes back as correctable JSON rather than a transport error that would end a calling
agent's turn:

```
initialize      ok   server='blast-radius' proto=2025-06-18
tools/list      ok   13 tools
registry parity ok   13 registered, 13 exposed
tools/call      ok   blast_radius(service=pricing-legacy)  cypher=yes
error path      ok   correctable: no such service in this tenant: no-such-service. Known...
resources/read  ok   graph://provenance -> 1614 bytes
PASS  MCP transport verified over stdio
```

To attach a client, point it at this repo's interpreter (`fastmcp` is an optional extra:
`pip install -e '.[mcp]'`):

```json
{
  "mcpServers": {
    "blast-radius": {
      "command": "/absolute/path/to/blast-radius/.venv/bin/python",
      "args": ["-m", "blast_radius.mcp_server"],
      "env": { "TENANT": "acme" }
    }
  }
}
```

`TENANT` selects which isolated graph the server serves, so two client entries pointing at the same
code reach two different graphs. Set `FALKOR_URL` in the same `env` block to serve from Cloud
instead of local Docker — the suite above passes both ways.

**The server is generated from the registry, and running it for the first time proved why that
matters.** Three problems surfaced, none of which a code read would have caught:

1. **The handlers were `def handler(**kwargs)`, which advertises a tool that accepts nothing.** An
   MCP client discovers arguments from each tool's JSON schema, and that schema is derived from the
   handler's signature — so thirteen correctly-implemented tools were undiscoverable. fastmcp
   refuses `**kwargs` outright, which is the kind version of the failure: the server died at
   startup rather than serving tools nobody could work out how to call. Each handler's signature is
   now generated from its registry function, so the arguments, their types and their defaults are
   all advertised, and a new tool in `tools.py` is still exposed automatically with nothing to keep
   in sync.
2. **`compile()` inherits `from __future__ import annotations` from its caller.** Because this
   module uses postponed annotations, the generated handlers stored theirs as strings; pydantic then
   re-evaluated those strings in a namespace with no `Field` and schema generation died with
   `NameError: name 'Field' is not defined`, reported against generated source that never mentions
   a missing name. `dont_inherit=True` fixes it, and a test asserts the annotations are real
   objects because the symptom appeared two libraries away from the cause.
3. **The registry's per-argument help was prose-only.** Each argument's description now lands on
   the schema property itself, which is what a client shows a model while *choosing* between tools
   rather than after picking one. An annotation outside a small whitelist raises at build time
   instead of degrading to `Any`, because a tool advertised as accepting anything fails inside a
   Cypher parameter rather than at the boundary.

## The Cypher this product runs on

Required by the submission rules. Each snippet is copied out of the running code, not retyped from
memory — a few `RETURN` columns are trimmed for length, and nothing else is altered. Every query is
also printed with its answer at runtime: `make ask` shows all six with the Cypher that produced
each, and the UI and the MCP server return the same text, so none of this can drift silently from
what actually ran.

**Multi-hop blast radius.** The variable-length traversal is the whole product: who breaks if this
service dies, out to four hops, joined to the paying customers behind those dependents.

```cypher
MATCH (failing:Service {name: $service})
OPTIONAL MATCH path = (dependent:Service)-[:DEPENDS_ON*1..4]->(failing)
WITH failing, collect(DISTINCT dependent) AS dependents
WITH failing, dependents, dependents + [failing] AS impacted
UNWIND impacted AS s
OPTIONAL MATCH (s)-[:EXPOSES]->(e:Endpoint)<-[:USES]-(cust:Customer)
RETURN failing.name                 AS service,
       [x IN dependents | x.name]   AS dependent_services,
       count(DISTINCT cust)         AS customer_count,
       sum(DISTINCT cust.mrr)       AS mrr_at_risk
```

**What changed just before the alert.** Note the anchor: the window is taken from the `Alert` node
in the graph, not from the wall clock. That is both a correctness fix (see *Engine behaviours* #6)
and what the question actually means — and it gives an upper bound, so a deploy that landed *after*
the alert can no longer be offered as its cause.

```cypher
MATCH (alerting:Service {name: $service})
OPTIONAL MATCH (a:Alert)-[:FIRED_ON]->(alerting)
WITH max(a.at) AS alert_at
WITH coalesce(alert_at, $wall_now) AS anchor
MATCH (alerting:Service {name: $service})
MATCH path = (alerting)-[:DEPENDS_ON*1..3]->(suspect:Service)
WITH anchor, suspect.name AS suspect_name, min(length(path)) AS hops
MATCH (s:Service {name: suspect_name})<-[:DEPLOYS]-(d:Deploy)
WHERE d.at >= anchor - $lookback_secs AND d.at <= anchor
OPTIONAL MATCH (d)-[:CONTAINS]->(c:Commit)<-[:AUTHORED]-(author:Person {is_bot: false})
RETURN s.name AS service, s.tier AS tier, hops AS hops_from_alert,
       d.version AS version, d.commit_subject AS change,
       collect(DISTINCT author.handle)[0] AS deployed_by
ORDER BY hops_from_alert ASC, deployed_at DESC
```

The `suspect.name` / re-`MATCH` shape is not clumsiness — a node binding does not survive an
aggregating `WITH` on this engine, and the version that looked cleaner returned zero rows without
erroring. *Engine behaviours* #1.

**Who actually knows this system.** This is the query that produced the project's headline finding:
one person is top author of **14 of the 16** services that map to repository directories. Note what
the query does *not* claim — `sole_author_services` comes back **0** and the thinnest bench is
**7 contributors**, so nobody is a lone maintainer. The risk is concentration of *deep* knowledge,
not absence of other hands, and the query reports both numbers so the distinction survives.

```cypher
MATCH (p:Person {is_bot: false})-[w:WORKED_ON]->(s:Service)
WITH s.name AS service, p.handle AS handle, w.commits AS commits
ORDER BY commits DESC
WITH service, collect(handle) AS ranked, collect(commits) AS counts
WITH service, ranked[0] AS top, counts[0] AS top_commits,
     coalesce(counts[1], 0) AS runner_up_commits, size(ranked) AS contributors
WITH top, count(*) AS services_topped, collect(service) AS services,
     min(contributors) AS thinnest_bench,
     sum(CASE WHEN runner_up_commits = 0 THEN 1 ELSE 0 END) AS sole_author_services
RETURN top AS person, services_topped, services, thinnest_bench, sole_author_services
ORDER BY services_topped DESC
```

**Memory recall — vector search inside the same graph**, so a remembered episode arrives already
joined to the services it was about. There is no second datastore for embeddings.

```cypher
CALL db.idx.vector.queryNodes('Episode', 'embedding', $k, vecf32($vec))
YIELD node, score
WITH node, score
WHERE $kinds IS NULL OR node.kind IN $kinds
OPTIONAL MATCH (node)-[:ABOUT]->(s:Service)
RETURN node.id AS episode, node.kind AS kind, node.summary AS summary,
       node.incident AS incident, collect(DISTINCT s.name) AS services,
       round((1.0 - score) * 1000) / 1000 AS similarity
ORDER BY score ASC
```

**The six graph algorithms**, called server-side as agent tools rather than reimplemented in
Python. Two calling conventions live in one namespace, which is a real trap — see *Engine
behaviours* #3:

```cypher
CALL algo.pageRank('Service', 'DEPENDS_ON')          YIELD node, score   -- positional
CALL algo.betweenness($projection)                   YIELD node, score   -- config map
CALL algo.HarmonicCentrality($projection)            YIELD node, score
CALL algo.WCC($projection)                           YIELD node, componentId
CALL algo.labelPropagation($projection)              YIELD node, communityId
CALL algo.BFS(a, $depth, 'DEPENDS_ON')               YIELD nodes
```

`$projection` is `{nodeLabels: ['Service'], relationshipTypes: ['DEPENDS_ON']}`. And the shortest
path between two services, which must sit in `WITH` or `RETURN` — inside `MATCH` it is rejected:

```cypher
MATCH (a:Service {name: $source}), (b:Service {name: $target})
WITH shortestPath((a)-[:DEPENDS_ON*1..4]->(b)) AS p
WHERE p IS NOT NULL
RETURN [n IN nodes(p) | n.name] AS hops, length(p) AS length, p AS path
```

## Multi-tenancy

One graph per tenant — `blastradius_acme`, `blastradius_globex`. **There is no `WHERE tenant = ...`
clause anywhere in the codebase**, and a test asserts that, because isolation you cannot forget is
better than isolation you must remember.

On top of that, the HTTP API derives the tenant **from the API key**, never from the request. A
caller cannot reach another tenant by editing a URL. Auth is on by default and the server generates
and prints a key rather than serving open; turning it off takes an explicit
`BR_AUTH_REQUIRED=false`.

---

## Performance

```
30 threads, 6s, 15% writes, loopback, ~1,659-node graph

operation              count     mean      p50      p95      p99
candidate_causes        6168     4.37     4.02     7.13    10.48
blast_radius            6086     3.81     3.48     6.69     9.13
vector_recall           6079     3.82     3.51     6.69     9.12
who_to_page             6114     3.84     3.50     6.90     9.13
episode_write           5217    15.00    14.86    23.74    28.19

5,695 ops/sec · 0 errors · overall p99 21.70 ms
```

The full three-agent loop is **17–45 ms** for ten tool calls.

**What this does not prove**, because a benchmark without its caveats is marketing: one machine over
loopback with no network in between; a demo-sized graph, and traversal cost grows with the graph; a
threaded Python client that may itself be the ceiling rather than the database. Treat it as an upper
bound.

The benchmark deletes its own writes in a `finally` block — an early version left 6,567 episodes
behind and took the graph from 1,644 nodes to 8,226, which also poisoned vector recall. A test pins
that the node count is unchanged after a run.

---

## Layout

```
src/blast_radius/
  config.py          connection (incl. Cloud TLS), API keys, settings
  db.py              client, pooling, retry, and the provenance envelope
  schema.py          16 labels, 19 edge types, 22 indices, 1 vector index
  data.py            the synthetic world: topology, money, incidents, runbooks
  github_ingest.py   the real human layer + path-map discovery + bot detection
  embeddings.py      fastembed when present, deterministic hashed fallback
  ingest.py          joins both halves, then verify() asserts the result
  queries.py         the six questions, each declaring what to highlight
  algorithms.py      pageRank, betweenness, harmonic, WCC, labelProp, BFS
  memory.py          semantic / episodic / procedural + agent handoffs
  tools.py           one typed registry, shared by agent + API + MCP
  agents.py          triage -> diagnosis -> comms, with ruled-out reasons
  api.py             authenticated FastAPI + SSE
  cli.py             br ingest / ask / run / algo / memory / doctor
  mcp_server.py      the same tools over MCP, enumerated from the registry
  proof.py           the three-process memory proof
web/index.html       single-page UI, vendored Cytoscape, no CDN
tests/               53 regression tests, one per bug that actually bit
                     + mcp_probe.py, which speaks MCP to a real server subprocess
bench/concurrency.py the numbers above
```

`tools.py` is the only way to reach a query — the agent, the API and the MCP server all dispatch
through it, so the UI shows the call the agent actually made and an MCP client gets the identical
surface with no second list to keep in sync.

---

## Known limitations

Stated here rather than discovered by a reviewer.

- **FalkorDB Cloud is verified against a live instance.** The whole suite — 53 tests, 9/9
  correctness checks, all six questions, the full agent loop and the MCP server over stdio — was
  run against a Free-tier instance in `ap-south-1`. Two corrections came out of it, both the
  opposite of what this project
  assumed for two days. **Cloud is not TLS-only:** TLS starts at the Startup tier, so a Free
  instance is reached with plain `falkor://` plus a password and a `falkors://` URL fails against
  it. And `is_cloud` was defined as "has TLS", so a genuine Cloud endpoint reported `cloud=False`
  in `br doctor` — fixed to key on host, which is advisory and never used for a security decision.
  Cloud latency is honest and worth stating: questions run 61–257 ms and the agent loop 2.7 s,
  against 0.4–4.4 ms and 66 ms on local Docker, because each of the ten tool calls crosses the
  network. Free instances also stop after a day idle and are deleted after seven with no
  persistence, so local Docker remains the reproducible path and Cloud is proof of portability.
- **Embeddings fall back to hashed trigrams** unless `fastembed` is installed. They retrieve
  adequately on this corpus but are not semantic, and `/api/health`, `br doctor` and the UI header
  all say which backend is live rather than letting you assume.
- **Timezones are inferred from commit hours**, not read from profiles — a profile `location` is
  exactly the personal data Rule 6 excludes. It is a heuristic: a nocturnal maintainer is placed
  wrongly, so the honest unit is the median across top committers, and that is what the test
  asserts. It infers UTC+2 for an Israel-based project, which is independent evidence the pipeline
  is sound.
- **Both tenants load a different real repository.** `acme` is built from `FalkorDB/FalkorDB`
  (1,256 commits, 68 humans, median UTC+2), `globex` from `redis/redis` (1,199 commits, 225 humans,
  median UTC+3) — the path map is discovered from each repo's own tree rather than hard-coded, which
  is what makes a second repo possible at all. The two graphs disagree about who matters: `swilly22`
  tops 14 services in one, `antirez` tops 5 in the other. Isolation is structural either way.
- **`INTRODUCED` attaches 78 of 200 pull requests.** The other 122 have merge commits outside the
  mapped directories. Correct, but worth knowing before someone asks why the numbers differ.
- **LangGraph is not used.** The agent state machine is explicit, because an LLM or a framework in
  the deciding path makes it impossible to tell whether the graph or the model produced the answer.
  Everything here is reproducible and every step cites its Cypher. An LLM belongs at the edges —
  turning this trace into prose — not in the deciding.
- **The UI was verified by rendering it, and that found three bugs no endpoint check could.** All
  fifteen endpoints and assets had returned 200 for days while the page still had a quarter of its
  canvas occupied by a detached band of Person dots, the bottom row of services clipped off the
  edge, and a stale highlight left over from a previous run contradicting the panel beside it. The
  first is in `/api/graph` (zero-degree nodes are dropped and counted), the other two in
  `web/index.html` (`layoutstop`-anchored `cy.fit`, and clearing the highlight on view change).
  Rendering happened under **WebKit**, because Chromium ran out of memory on this host — which
  incidentally means the UI is verified on a second engine rather than only on Chromium.

---

## Engine behaviours worth knowing

Found the hard way; each is pinned by a test so an image upgrade fails loudly instead of silently.

1. **A node binding does not survive an aggregating `WITH`.** `WITH s, min(length(path)) AS hops`
   then `MATCH (d:Deploy)-[:DEPLOYS]->(s)` returns **zero rows with no error**; the same query with
   `WITH DISTINCT s` returns 14. Carry the key and re-match.
2. **`shortestPath` is rejected inside `MATCH`** — it must be in `WITH` or `RETURN`. A divergence
   from Neo4j's Cypher.
3. **Two calling conventions in one namespace.** `algo.pageRank('Service','DEPENDS_ON')` is
   positional; `algo.betweenness({nodeLabels: [...]})` takes a config map. The error names the
   arity, not the expected shape.
4. **A missing property returns `null`, never an error.** Four shipped before being caught —
   `Runbook.mttr_minutes` (really `expected_mttr_minutes`), `Team.escalation` (really `channel`),
   `Commit.subject` (really `msg`), and `Episode.incident`, which the ingest assembled and never
   wrote. `schema.properties_in_use()` plus a test now guards the whole class.
5. **`UNWIND ... MATCH ... MERGE` that matches nothing still succeeds.** So the ingest report quotes
   the engine's own `nodes_created` / `relationships_created` instead of its input length, and flags
   any step that was handed rows and changed nothing.

And one that is ours, not the engine's, because it is the same failure mode and cost the most:

6. **A time window anchored on the wall clock silently empties the headline question.** Deploy
   timestamps are placed relative to the alert at ingest time, but `candidate_causes` filtered on
   `time.time() - lookback_hours`. The two agree for six hours after an ingest and then stop
   agreeing, with no error — on a graph ingested 22 hours earlier Q1 returned **zero rows** and the
   demo's first question answered nothing. The window is now anchored on
   `(:Alert)-[:FIRED_ON]->(service)`, which is also what the question means: *what changed just
   before the alert* is relative to the alert, not to whenever someone runs the query. Anchoring
   additionally buys an upper bound, so a deploy that landed **after** the alert can no longer be
   offered as its cause. Pinned by moving the clock ten days forward in a test, because an
   assertion on today's row count passes against the broken version too.

---

## License

MIT. The synthetic data is ours; the GitHub-derived layer is public data used under GitHub's API
terms, reduced to logins with no email addresses retained.
