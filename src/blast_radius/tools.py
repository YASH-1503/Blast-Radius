"""The tool registry: one typed surface that the agent, the HTTP API and the MCP server all share.

Every capability the agent has is registered here, and nothing calls a query module directly. That
single indirection buys three things that matter for the submission:

* **The agent cannot do anything the API cannot show.** A step in the UI is the same call the agent
  made, not a reconstruction of it.
* **An MCP client gets the identical surface for free.** `mcp_server.py` enumerates this registry;
  there is no second list to keep in sync.
* **Every call is logged with its own evidence.** A tool returns the `QueryResult` envelope, so the
  Cypher, the hop budget, the timing and the traversed nodes travel with the answer.

The arguments are validated here rather than inside the Cypher. A tool handed an unknown service
name would otherwise run a perfectly valid query that matches nothing and return an empty answer,
which reads identically to "there is nothing to find" -- the failure mode this whole project exists
to avoid.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable

from . import algorithms, memory, queries
from .db import GraphClient, QueryResult


@dataclass
class ToolSpec:
    """One capability: what it does, what it takes, and why it is a graph question."""

    name: str
    summary: str
    graph_rationale: str
    args: dict[str, str]
    required: tuple[str, ...]
    fn: Callable[..., QueryResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "summary": self.summary,
            "graph_rationale": self.graph_rationale,
            "args": self.args,
            "required": list(self.required),
        }


@dataclass
class ToolCall:
    """A tool invocation and its result, as the UI and the audit trail see it."""

    tool: str
    args: dict[str, Any]
    result: QueryResult
    at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        payload = self.result.to_dict()
        payload.update({"tool": self.tool, "args": self.args, "at": self.at})
        return payload


# --------------------------------------------------------------------------------------
# Registry
# --------------------------------------------------------------------------------------

REGISTRY: dict[str, ToolSpec] = {}


def tool(
    name: str,
    *,
    summary: str,
    graph_rationale: str,
    args: dict[str, str],
    required: tuple[str, ...] = (),
) -> Callable[[Callable[..., QueryResult]], Callable[..., QueryResult]]:
    def register(fn: Callable[..., QueryResult]) -> Callable[..., QueryResult]:
        REGISTRY[name] = ToolSpec(
            name=name,
            summary=summary,
            graph_rationale=graph_rationale,
            args=args,
            required=required,
            fn=fn,
        )
        return fn

    return register


@tool(
    "candidate_causes",
    summary="Deploys inside the alert lookback on anything the failing service depends on.",
    graph_rationale="Variable-depth traversal; in SQL a recursive CTE whose depth depends on data.",
    args={"service": "the alerting service", "lookback_hours": "how far back to look"},
    required=("service",),
)
def _candidate_causes(service: str, lookback_hours: int = 6, **kw: Any) -> QueryResult:
    return queries.candidate_causes(service, lookback_hours=lookback_hours, **kw)


@tool(
    "blast_radius",
    summary="Services, customers and revenue downstream of a failing service.",
    graph_rationale="Revenue is reached service -> endpoint -> customer, the path a request takes.",
    args={"service": "the suspect service"},
    required=("service",),
)
def _blast_radius(service: str, **kw: Any) -> QueryResult:
    return queries.blast_radius(service, **kw)


@tool(
    "severity",
    summary="Severity derived from tier and revenue at risk, with the deciding branch named.",
    graph_rationale="Both operands are traversals, so the verdict can be disputed by disputing the path.",
    args={"service": "the suspect service"},
    required=("service",),
)
def _severity(service: str, **kw: Any) -> QueryResult:
    return queries.severity(service, **kw)


@tool(
    "who_to_page",
    summary="The owning team, and separately whoever has actually written this service's code.",
    graph_rationale="Org chart and commit history are both edges, so their disagreement is visible.",
    args={"service": "the service needing a human"},
    required=("service",),
)
def _who_to_page(service: str, **kw: Any) -> QueryResult:
    return queries.who_to_page(service, **kw)


@tool(
    "key_person_risk",
    summary="How concentrated real expertise is -- who tops how many services.",
    graph_rationale="Aggregates the derived WORKED_ON edge across the whole service graph.",
    args={},
)
def _key_person_risk(**kw: Any) -> QueryResult:
    return queries.key_person_risk(**kw)


@tool(
    "past_incidents",
    summary="Prior incidents on this service or its dependencies, with the runbook that fixed them.",
    graph_rationale="Episodic memory reached structurally rather than by text similarity.",
    args={"service": "the service to look up"},
    required=("service",),
)
def _past_incidents(service: str, **kw: Any) -> QueryResult:
    return queries.past_incidents(service, **kw)


@tool(
    "recall",
    summary="Vector search over remembered episodes.",
    graph_rationale="Runs on the database's own vector index, not a Python similarity loop.",
    args={"text": "what the current situation looks like", "k": "how many hits"},
    required=("text",),
)
def _recall(text: str, k: int = 3, **kw: Any) -> QueryResult:
    return memory.recall(text, k=k, **kw)


@tool(
    "runbook_for",
    summary="The runbook that resolved the most recent prior incident on this service.",
    graph_rationale="Procedural memory reached THROUGH the incident it resolved, not tagged by name.",
    args={"service": "the service to fix"},
    required=("service",),
)
def _runbook_for(service: str, **kw: Any) -> QueryResult:
    return memory.runbook_for(service, **kw)


@tool(
    "why_dependency",
    summary="The recorded decision behind one dependency edge, and who wrote it.",
    graph_rationale="The reason is attached to the edge, so 'can we remove this?' is answerable.",
    args={"upstream": "consumer service", "downstream": "provider service"},
    required=("upstream", "downstream"),
)
def _why_dependency(upstream: str, downstream: str, **kw: Any) -> QueryResult:
    return queries.why_dependency(upstream, downstream, **kw)


@tool(
    "path_between",
    summary="The shortest dependency path between two services.",
    graph_rationale="Returned as a path, so the UI draws the real route.",
    args={"source": "start service", "target": "end service"},
    required=("source", "target"),
)
def _path_between(source: str, target: str, **kw: Any) -> QueryResult:
    return queries.path_between(source, target, **kw)


@tool(
    "critical_services",
    summary="PageRank over the dependency graph.",
    graph_rationale="Computed server-side by algo.pageRank; no embedding encodes topology.",
    args={"limit": "how many to return"},
)
def _critical_services(limit: int = 8, **kw: Any) -> QueryResult:
    return algorithms.critical_services(limit=limit, **kw)


@tool(
    "chokepoints",
    summary="Betweenness centrality -- services traffic must pass through.",
    graph_rationale="Cut vertices are a property of topology alone.",
    args={"limit": "how many to return"},
)
def _chokepoints(limit: int = 8, **kw: Any) -> QueryResult:
    return algorithms.chokepoints(limit=limit, **kw)


@tool(
    "provenance",
    summary="Which layers of the graph are real and which are synthetic, counted live.",
    graph_rationale="Answers the data-provenance rule with a query instead of a paragraph.",
    args={},
)
def _provenance(**kw: Any) -> QueryResult:
    return queries.provenance(**kw)


# --------------------------------------------------------------------------------------
# Dispatch
# --------------------------------------------------------------------------------------

class ToolError(ValueError):
    """Raised for an unknown tool, a missing argument, or an argument naming nothing in the graph."""


def _known_services(gc: GraphClient) -> set[str]:
    return set(
        gc.run(
            "MATCH (s:Service) RETURN s.name AS name",
            label="tools:known_services",
            collect_provenance=False,
        ).column("name")
    )


# Argument names whose value must be an existing service. Validated BEFORE the query runs: a bad
# name otherwise produces a valid query that matches nothing, and an empty result is
# indistinguishable from "nothing to find".
_SERVICE_ARGS = ("service", "source", "target", "upstream", "downstream")


def call(
    name: str,
    args: dict[str, Any] | None = None,
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> ToolCall:
    """Validate and run one tool, returning the call and its evidence together."""
    spec = REGISTRY.get(name)
    if spec is None:
        raise ToolError(f"unknown tool {name!r}; have {', '.join(sorted(REGISTRY))}")

    args = dict(args or {})
    missing = [a for a in spec.required if a not in args]
    if missing:
        raise ToolError(f"{name} requires {', '.join(missing)}")

    unexpected = [a for a in args if a not in spec.args]
    if unexpected:
        raise ToolError(f"{name} got unknown argument(s) {', '.join(sorted(unexpected))}")

    gc = client or GraphClient(tenant)
    service_values = [args[a] for a in _SERVICE_ARGS if isinstance(args.get(a), str)]
    if service_values:
        known = _known_services(gc)
        unknown = sorted(set(service_values) - known)
        if unknown:
            raise ToolError(
                f"{name}: no such service in this tenant: {', '.join(unknown)}. "
                f"Known: {', '.join(sorted(known)[:8])}..."
            )

    result = spec.fn(client=gc, **args)
    return ToolCall(tool=name, args=args, result=result)


def catalogue() -> list[dict[str, Any]]:
    """The tool list, for the UI, the MCP server and the README."""
    return [REGISTRY[name].to_dict() for name in sorted(REGISTRY)]


__all__ = ["REGISTRY", "ToolCall", "ToolError", "ToolSpec", "call", "catalogue", "tool"]
