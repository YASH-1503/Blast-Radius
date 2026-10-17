"""FalkorDB access and the provenance envelope.

The envelope is the most important design decision in this codebase. Nothing here runs a query and
returns bare rows: `GraphClient.run` returns a `QueryResult` carrying the rows, the exact Cypher, the
parameters, the nodes and edges that were walked, the node names the step is *about*, and the
timing. "Show the path behind the answer" is therefore a property of the architecture, not a feature
the UI has to remember to add. The provenance panel is a renderer for this object and the agent's
audit trail is a list of them.

Two things here were learned the hard way and are load-bearing:

* **`focus` exists because node objects alone are not enough.** A query that ends
  `RETURN s.name AS suspect` returns a string, so there is no node in the result to harvest and the
  UI lights up nothing -- which was true of nine steps in ten. Returning whole nodes everywhere
  would bloat the payload, so each query instead declares which columns hold node *names*
  (`focus_keys`) and the UI resolves them against the graph it has already loaded.
* **"Max pending queries exceeded" is backpressure, not failure.** It is a server-side admission
  queue overflowing under concurrency, and the correct response is a bounded retry.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

from falkordb import FalkorDB

from .config import SETTINGS, Connection


# --------------------------------------------------------------------------------------
# Serialisation: falkordb Node/Edge/Path objects -> plain JSON-able dicts
# --------------------------------------------------------------------------------------

_TITLE_KEYS = ("name", "handle", "id", "sha", "title", "number")


def _node_to_dict(node: Any) -> dict[str, Any]:
    labels = getattr(node, "labels", None) or []
    if isinstance(labels, str):
        labels = [labels]
    props = dict(getattr(node, "properties", {}) or {})
    props.pop("embedding", None)  # a vector is noise in a UI payload

    title = next((str(props[k]) for k in _TITLE_KEYS if props.get(k) is not None), None)
    return {
        "kind": "node",
        "id": getattr(node, "id", None),
        "labels": list(labels),
        "label": (list(labels) or ["Node"])[0],
        "title": title or str(getattr(node, "id", "")),
        "properties": props,
    }


def _edge_to_dict(edge: Any) -> dict[str, Any]:
    return {
        "kind": "edge",
        "id": getattr(edge, "id", None),
        "type": getattr(edge, "relation", None),
        "source": getattr(edge, "src_node", None),
        "target": getattr(edge, "dest_node", None),
        "properties": dict(getattr(edge, "properties", {}) or {}),
    }


def _is_node(v: Any) -> bool:
    return hasattr(v, "labels") and hasattr(v, "properties")


def _is_edge(v: Any) -> bool:
    return hasattr(v, "relation") and hasattr(v, "src_node")


def _is_path(v: Any) -> bool:
    return hasattr(v, "nodes") and hasattr(v, "edges") and callable(getattr(v, "nodes", None))


def serialise(value: Any) -> Any:
    """Recursively convert one result cell into JSON-able data."""
    if _is_path(value):
        return {
            "kind": "path",
            "nodes": [_node_to_dict(n) for n in value.nodes()],
            "edges": [_edge_to_dict(e) for e in value.edges()],
        }
    if _is_node(value):
        return _node_to_dict(value)
    if _is_edge(value):
        return _edge_to_dict(value)
    if isinstance(value, list):
        return [serialise(v) for v in value]
    if isinstance(value, dict):
        return {k: serialise(v) for k, v in value.items()}
    return value


def _harvest(cell: Any, nodes: dict[Any, dict], edges: dict[Any, dict]) -> None:
    """Pull every node/edge out of a cell so the UI can draw what was walked."""
    if _is_path(cell):
        for n in cell.nodes():
            d = _node_to_dict(n)
            nodes[d["id"]] = d
        for e in cell.edges():
            d = _edge_to_dict(e)
            edges[d["id"]] = d
        return
    if _is_node(cell):
        d = _node_to_dict(cell)
        nodes[d["id"]] = d
        return
    if _is_edge(cell):
        d = _edge_to_dict(cell)
        edges[d["id"]] = d
        return
    if isinstance(cell, list):
        for v in cell:
            _harvest(v, nodes, edges)
    elif isinstance(cell, dict):
        for v in cell.values():
            _harvest(v, nodes, edges)


# --------------------------------------------------------------------------------------
# The envelope
# --------------------------------------------------------------------------------------

@dataclass
class WriteStats:
    """What a write query ACTUALLY changed, read back from the engine.

    This exists because of a specific class of silent failure. An ingest statement shaped
    `UNWIND $rows AS row MATCH (...) MERGE (...)` that matches NOTHING still completes without
    error, and a report that prints `len(rows)` then claims success for an edge type that was
    never created. The spike shipped exactly that bug: the table said `decision authors 1` while
    the graph held zero, because the author no longer existed after the real-data swap.

    So the report quotes the engine, not the input. `created` differing from the rows sent is
    normal on a re-run (MERGE is idempotent); `created == 0` on a FIRST load is the alarm.
    """

    nodes_created: int = 0
    relationships_created: int = 0
    properties_set: int = 0
    nodes_deleted: int = 0
    relationships_deleted: int = 0

    @property
    def created(self) -> int:
        return self.nodes_created + self.relationships_created

    @property
    def deleted(self) -> int:
        return self.nodes_deleted + self.relationships_deleted

    @property
    def touched_nothing(self) -> bool:
        return self.created == 0 and self.properties_set == 0 and self.deleted == 0


@dataclass
class QueryResult:
    """Rows plus the evidence for how they were produced."""

    cypher: str
    params: dict[str, Any]
    header: list[str]
    rows: list[dict[str, Any]]
    elapsed_ms: float
    graph: str
    writes: WriteStats = field(default_factory=WriteStats)
    label: str = ""                                              # e.g. "candidate_causes"
    hops: int | None = None                                      # declared max traversal depth
    nodes: list[dict[str, Any]] = field(default_factory=list)    # node objects in the result
    edges: list[dict[str, Any]] = field(default_factory=list)    # edge objects in the result
    focus: list[str] = field(default_factory=list)               # node NAMES this step is about

    def first(self) -> dict[str, Any] | None:
        return self.rows[0] if self.rows else None

    def column(self, name: str) -> list[Any]:
        return [r.get(name) for r in self.rows if r.get(name) is not None]

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "cypher": self.cypher.strip(),
            "params": self.params,
            "hops": self.hops,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "graph": self.graph,
            "row_count": len(self.rows),
            "rows": self.rows,
            "provenance": {"nodes": self.nodes, "edges": self.edges, "focus": self.focus},
        }

    def explain(self) -> str:
        """The text a judge should be able to read beside any answer."""
        lines = [
            f"step      : {self.label or '(unnamed)'}",
            f"graph     : {self.graph}",
            f"hops      : {self.hops if self.hops is not None else 'n/a'}",
            f"rows      : {len(self.rows)}",
            f"time      : {self.elapsed_ms:.2f} ms",
            f"traversed : {len(self.nodes)} nodes / {len(self.edges)} edges",
            f"focus     : {', '.join(self.focus) if self.focus else '-'}",
            "cypher    :",
        ]
        lines += [f"    {ln}" for ln in self.cypher.strip().splitlines()]
        if self.params:
            lines.append(f"params    : {self.params}")
        return "\n".join(lines)


# --------------------------------------------------------------------------------------
# Connection pooling
# --------------------------------------------------------------------------------------
# One FalkorDB object per (connection, process). The underlying redis client is itself pooled and
# thread-safe, so building a new one per GraphClient -- as the pre-event spike did -- wastes a TCP
# handshake, and on Cloud a full TLS handshake, on every single tool call.

_POOL: dict[tuple, FalkorDB] = {}
_POOL_LOCK = threading.Lock()


def _db_for(conn: Connection) -> FalkorDB:
    key = (conn.host, conn.port, conn.username, conn.tls)
    with _POOL_LOCK:
        existing = _POOL.get(key)
        if existing is not None:
            return existing

        kwargs: dict[str, Any] = {
            "host": conn.host,
            "port": conn.port,
            "username": conn.username,
            "password": conn.password,
        }
        if conn.tls:
            # FalkorDB Cloud terminates TLS and will simply hang up on a plaintext client, which
            # surfaces as a confusing connection reset rather than a handshake error.
            kwargs["ssl"] = True
        db = FalkorDB(**kwargs)
        _POOL[key] = db
        return db


def reset_pool() -> None:
    """Drop pooled connections. Used by tests and after a credential change."""
    with _POOL_LOCK:
        _POOL.clear()


class GraphClient:
    """Thin wrapper over one tenant's graph."""

    # Server-side admission queue overflow (MAX_QUEUED_QUERIES, default 50). Backpressure, not error.
    _BACKPRESSURE = ("max pending queries", "queries exceeded")

    def __init__(self, tenant: str | None = None, connection: Connection | None = None) -> None:
        self.tenant = (tenant or SETTINGS.tenant).strip().lower()
        self.graph_name = SETTINGS.graph_name(self.tenant)
        self.connection = connection or SETTINGS.connection
        self._db = _db_for(self.connection)
        self._graph = self._db.select_graph(self.graph_name)

    # -- lifecycle ---------------------------------------------------------------------

    def ping(self) -> bool:
        self._db.connection.ping()
        return True

    def drop(self) -> None:
        try:
            self._graph.delete()
        except Exception:
            pass  # the graph did not exist, which is the desired end state anyway

    def list_graphs(self) -> list[str]:
        return list(self._db.list_graphs())

    # -- the only execution path -------------------------------------------------------

    def _execute(self, cypher: str, params: dict[str, Any], attempts: int = 5) -> Any:
        delay = 0.01
        last: Exception | None = None
        for _ in range(attempts):
            try:
                return self._graph.query(cypher, params=params)
            except Exception as exc:  # noqa: BLE001
                message = str(exc).lower()
                if not any(token in message for token in self._BACKPRESSURE):
                    raise
                last = exc
                time.sleep(delay)
                delay *= 2
        raise RuntimeError(
            f"FalkorDB still applying backpressure after {attempts} attempts: {last}"
        ) from last

    def run(
        self,
        cypher: str,
        params: dict[str, Any] | None = None,
        *,
        label: str = "",
        hops: int | None = None,
        focus_keys: tuple[str, ...] = (),
        collect_provenance: bool = True,
    ) -> QueryResult:
        """Execute one query and return it wrapped in its own evidence.

        `focus_keys` names the result columns that hold node names, so a query returning scalars
        still tells the UI which nodes to highlight.
        """
        params = params or {}
        started = time.perf_counter()
        res = self._execute(cypher, params)
        elapsed = (time.perf_counter() - started) * 1000.0

        header = [h[1] if isinstance(h, (list, tuple)) else str(h) for h in (res.header or [])]
        rows: list[dict[str, Any]] = []
        nodes: dict[Any, dict[str, Any]] = {}
        edges: dict[Any, dict[str, Any]] = {}
        focus: list[str] = []
        seen_focus: set[str] = set()

        for raw_row in res.result_set or []:
            row: dict[str, Any] = {}
            for idx, cell in enumerate(raw_row):
                key = header[idx] if idx < len(header) else f"col{idx}"
                row[key] = serialise(cell)
                if collect_provenance:
                    _harvest(cell, nodes, edges)
            rows.append(row)

            for key in focus_keys:
                value = row.get(key)
                for name in value if isinstance(value, list) else [value]:
                    if isinstance(name, str) and name and name not in seen_focus:
                        seen_focus.add(name)
                        focus.append(name)

        # A harvested node is also a focus target -- the UI then has one list to highlight.
        for node in nodes.values():
            title = node.get("title")
            if isinstance(title, str) and title and title not in seen_focus:
                seen_focus.add(title)
                focus.append(title)

        return QueryResult(
            cypher=cypher,
            params=params,
            header=header,
            rows=rows,
            elapsed_ms=elapsed,
            graph=self.graph_name,
            writes=WriteStats(
                nodes_created=int(getattr(res, "nodes_created", 0) or 0),
                relationships_created=int(getattr(res, "relationships_created", 0) or 0),
                properties_set=int(getattr(res, "properties_set", 0) or 0),
                nodes_deleted=int(getattr(res, "nodes_deleted", 0) or 0),
                relationships_deleted=int(getattr(res, "relationships_deleted", 0) or 0),
            ),
            label=label,
            hops=hops,
            nodes=list(nodes.values()),
            edges=list(edges.values()),
            focus=focus,
        )

    def run_many(self, statements: list[tuple[str, dict[str, Any] | None]]) -> list[QueryResult]:
        return [self.run(c, p) for c, p in statements]


def client(tenant: str | None = None) -> GraphClient:
    return GraphClient(tenant)


def health() -> dict[str, Any]:
    """Connection diagnostics that are safe to expose: no password, ever."""
    conn = SETTINGS.connection
    info: dict[str, Any] = {"target": conn.describe(), "tls": conn.tls, "cloud": conn.is_cloud}
    started = time.perf_counter()
    try:
        c = GraphClient()
        c.ping()
        info["ok"] = True
        info["ping_ms"] = round((time.perf_counter() - started) * 1000.0, 2)
        info["graphs"] = c.list_graphs()
    except Exception as exc:  # noqa: BLE001
        info["ok"] = False
        info["error"] = str(exc)
    return info
