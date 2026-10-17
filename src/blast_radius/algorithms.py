"""Graph algorithms, run inside the database.

These are the part of the submission that a vector store or a relational schema cannot imitate at
all. Centrality, component structure and reachability are properties of the *topology*; there is no
embedding of a service that encodes "removing this node disconnects the graph".

Every function here runs server-side. Nothing is pulled into Python and post-processed, which
matters for the honesty of the claim as much as for speed: the database is doing the computation,
not a list comprehension wearing its name.

**Two calling conventions live in one namespace, and the inconsistency is undocumented.** Measured
against this build (FalkorDB on the `latest` image, 24 procedures):

    CALL algo.pageRank('Service', 'DEPENDS_ON')                                   -- positional
    CALL algo.betweenness({nodeLabels: ['Service'], relationshipTypes: [...]})    -- config map

Passing a config map to `pageRank`, or two positional arguments to `betweenness`, `WCC` or
`HarmonicCentrality`, fails with `Received 2 arguments to procedure ..., expected at most 1`. That
error names the arity but not the expected shape, so it reads like a usage mistake rather than a
convention difference. Both forms are pinned by tests here so an image upgrade that changes either
one fails loudly instead of silently returning nothing.
"""

from __future__ import annotations

from typing import Any

from .db import GraphClient, QueryResult

# The projection every topology algorithm runs over: services and the dependency spine.
SERVICE_PROJECTION = {"nodeLabels": ["Service"], "relationshipTypes": ["DEPENDS_ON"]}


def _client(tenant: str | None, client: GraphClient | None) -> GraphClient:
    return client or GraphClient(tenant)


def critical_services(
    *,
    limit: int = 8,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """PageRank over the dependency graph: which services the most other work flows through.

    This answers a question the dependency *count* cannot. A service with three dependents that are
    themselves depended on by everything matters more than a service with ten leaf dependents, and
    PageRank is exactly that distinction. Used by the agent to break ties between candidates that
    are otherwise equally plausible.

    Note the direction. `DEPENDS_ON` points from consumer to provider, so a high score means "much
    depends on this", which is the ranking wanted for blast radius.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        CALL algo.pageRank('Service', 'DEPENDS_ON') YIELD node, score
        WITH node, score
        OPTIONAL MATCH (dependent:Service)-[:DEPENDS_ON]->(node)
        RETURN node.name            AS service,
               node.tier            AS tier,
               round(score * 10000) / 10000 AS pagerank,
               count(DISTINCT dependent)    AS direct_dependents
        ORDER BY pagerank DESC
        LIMIT $limit
        """,
        {"limit": limit},
        label="critical_services",
        hops=0,
        focus_keys=("service",),
    )


def chokepoints(
    *,
    limit: int = 8,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Betweenness centrality: services that sit on the most paths between other services.

    Distinct from PageRank and the difference is the interesting part. PageRank finds what is
    heavily depended upon; betweenness finds what traffic has to pass *through*. On this topology
    they disagree -- `analytics` ranks top by PageRank while `catalog` ranks top by betweenness --
    and the disagreement is the useful signal: the first is a popular dependency, the second is a
    cut vertex whose failure partitions the system.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        CALL algo.betweenness($projection) YIELD node, score
        RETURN node.name AS service, node.tier AS tier, score AS betweenness
        ORDER BY betweenness DESC, service ASC
        LIMIT $limit
        """,
        {"projection": SERVICE_PROJECTION, "limit": limit},
        label="chokepoints",
        hops=0,
        focus_keys=("service",),
    )


def reach_from(
    service: str,
    *,
    depth: int = 3,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """BFS from one service: everything reachable within `depth` dependency hops.

    `algo.BFS` takes the source as a bound node and the depth as a positional argument -- a third
    convention again, distinct from both of the two above. It returns the node set in one call,
    where the equivalent `MATCH (a)-[:DEPENDS_ON*1..3]->(b)` enumerates every path to every node
    and then has to be de-duplicated.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        MATCH (a:Service {name: $service})
        CALL algo.BFS(a, $depth, 'DEPENDS_ON') YIELD nodes
        RETURN size(nodes) AS reached,
               [n IN nodes | n.name] AS services
        """,
        {"service": service, "depth": depth},
        label="reach_from",
        hops=depth,
        focus_keys=("services",),
    )


def components(
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Weakly connected components over the dependency graph.

    Reported because a single component is itself a finding: all 24 services are in one component,
    so there is no isolated island and any failure is at least potentially systemic. If a future
    topology change split the graph, this number changing is the signal -- and it is cheaper to
    watch than to re-derive from paths.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        CALL algo.WCC($projection) YIELD node, componentId
        RETURN componentId AS component,
               count(*) AS services,
               collect(node.name)[0..6] AS sample
        ORDER BY services DESC
        """,
        {"projection": SERVICE_PROJECTION},
        label="components",
        hops=0,
    )


def closeness(
    *,
    limit: int = 8,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Harmonic centrality: how close a service is to everything else.

    Used for the inverse question to blast radius -- not "what does this break" but "what is this
    service exposed to". `api-gateway` tops this list, which is the expected shape for an edge
    service and acts as a sanity check that the projection is the right way round.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        CALL algo.HarmonicCentrality($projection) YIELD node, score
        RETURN node.name AS service, node.tier AS tier,
               round(score * 1000) / 1000 AS harmonic_centrality
        ORDER BY harmonic_centrality DESC, service ASC
        LIMIT $limit
        """,
        {"projection": SERVICE_PROJECTION, "limit": limit},
        label="closeness",
        hops=0,
        focus_keys=("service",),
    )


def communities(
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Label propagation: which services cluster together by dependency structure alone.

    Interesting because the clusters are computed with no knowledge of team ownership. Where a
    computed community crosses an ownership boundary, an incident in it will cross a team boundary
    too -- which is a prediction the graph makes about the org, from the code alone.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        CALL algo.labelPropagation($projection) YIELD node, communityId
        OPTIONAL MATCH (t:Team)-[:OWNS]->(node)
        RETURN communityId AS community,
               count(DISTINCT node) AS services,
               collect(DISTINCT node.name)[0..6] AS sample,
               count(DISTINCT t.name) AS teams_spanned
        ORDER BY services DESC
        """,
        {"projection": SERVICE_PROJECTION},
        label="communities",
        hops=0,
    )


ALGORITHMS: dict[str, dict[str, Any]] = {
    "pagerank": {
        "what": "what the most work flows through",
        "run": lambda **kw: critical_services(**kw),
    },
    "betweenness": {
        "what": "cut vertices -- what traffic must pass through",
        "run": lambda **kw: chokepoints(**kw),
    },
    "harmonic": {
        "what": "what each service is most exposed to",
        "run": lambda **kw: closeness(**kw),
    },
    "wcc": {
        "what": "connected components -- is anything isolated",
        "run": lambda **kw: components(**kw),
    },
    "communities": {
        "what": "clusters by dependency structure, ignoring the org chart",
        "run": lambda **kw: communities(**kw),
    },
    "bfs": {
        "what": "reachability within N hops",
        "run": lambda **kw: reach_from(kw.pop("service", "api-gateway"), **kw),
    },
}


def run(name: str, **kwargs: Any) -> QueryResult:
    if name not in ALGORITHMS:
        raise KeyError(f"unknown algorithm {name!r}; have {', '.join(sorted(ALGORITHMS))}")
    return ALGORITHMS[name]["run"](**kwargs)


__all__ = [
    "ALGORITHMS",
    "SERVICE_PROJECTION",
    "chokepoints",
    "closeness",
    "communities",
    "components",
    "critical_services",
    "reach_from",
    "run",
]
