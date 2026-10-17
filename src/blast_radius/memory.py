"""Agent memory, held in the graph: semantic recall, episodic writes, and agent-to-agent handoff.

Track 02 asks for memory that survives a session. The claim is cheap to make and almost always
demonstrated dishonestly -- a dictionary that outlives one function call proves nothing, and a
single process that writes then reads its own cache proves only that the cache works.

So the proof here is deliberately awkward: `proof.py` asks a question in one OS process, runs the
agent in a second, and asks again in a third. The three share no memory, no file handles and no
Python objects. The only thing they share is the database. A cold ask returns zero hits and a warm
ask returns the episode the middle process wrote, which is the only arrangement where "memory
persisted" is falsifiable.

Three kinds of memory live here, and they are different things rather than three names for one:

* **semantic** -- `recall()`, a vector search over `Episode.text`. Finds incidents that *read* like
  the current one.
* **episodic** -- `write_episode()` / `timeline()`. What this system actually saw and concluded,
  with a timestamp, attached to the services it was about.
* **procedural** -- `runbook_for()`. What to DO, reached from the incident that it resolved.

`queries.past_incidents` is the structural counterpart to `recall`: it finds incidents *connected*
to the same part of the system rather than incidents whose text is similar. Both are kept because
they fail differently -- vector recall finds a lookalike on an unrelated service, structural recall
misses a genuinely similar incident elsewhere. The agent runs both and says which produced what.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from . import embeddings
from .config import SETTINGS
from .db import GraphClient, QueryResult

EPISODE_KINDS = (
    "incident_retrospective",  # seeded from history at ingest
    "agent_conclusion",        # what the agent decided, written at run time
    "handoff",                 # one agent's state passed to the next
)


def _client(tenant: str | None, client: GraphClient | None) -> GraphClient:
    return client or GraphClient(tenant)


# --------------------------------------------------------------------------------------
# Semantic recall
# --------------------------------------------------------------------------------------

def recall(
    text: str,
    *,
    k: int = 3,
    kinds: tuple[str, ...] | None = None,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Vector search over episodes, run by the database's own index.

    `db.idx.vector.queryNodes` returns a DISTANCE, not a similarity -- lower is closer. This is the
    opposite of the intuition the word "score" invites, and sorting it the wrong way produces a
    confidently wrong "most similar incident" that looks plausible in a demo. It is converted to a
    similarity here so everything downstream reads in one direction.

    The embedding backend is reported alongside the hits on purpose. With `fastembed` installed
    these are real sentence embeddings; without it they are deterministic hashed trigrams, which
    retrieve adequately on this corpus but are not semantic. Saying which one produced a hit is the
    difference between a measurement and a claim.
    """
    gc = _client(tenant, client)
    vector = embeddings.embed(text)
    result = gc.run(
        """
        CALL db.idx.vector.queryNodes('Episode', 'embedding', $k, vecf32($vec))
        YIELD node, score
        WITH node, score
        WHERE $kinds IS NULL OR node.kind IN $kinds
        OPTIONAL MATCH (node)-[:ABOUT]->(s:Service)
        RETURN node.id      AS episode,
               node.kind    AS kind,
               node.summary AS summary,
               node.text    AS text,
               node.at      AS at,
               node.incident AS incident,
               collect(DISTINCT s.name) AS services,
               round((1.0 - score) * 1000) / 1000 AS similarity,
               round(score * 1000) / 1000        AS distance
        ORDER BY distance ASC
        """,
        {"k": k, "vec": vector, "kinds": list(kinds) if kinds else None},
        label="memory.recall",
        hops=1,
        focus_keys=("services",),
    )
    result.params = {**result.params, "vec": f"<{len(vector)}-dim {embeddings.backend()}>"}
    return result


def hit_count(
    text: str,
    *,
    kind: str = "agent_conclusion",
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> int:
    """How many episodes of one kind the recall finds. This is the number the cold/warm proof prints."""
    rows = recall(text, k=10, kinds=(kind,), tenant=tenant, client=client).rows
    return len(rows)


# --------------------------------------------------------------------------------------
# Episodic writes
# --------------------------------------------------------------------------------------

@dataclass
class Episode:
    """One thing the system concluded, ready to be stored and found again."""

    text: str
    summary: str
    kind: str = "agent_conclusion"
    services: list[str] = field(default_factory=list)
    incident: str | None = None
    run_id: str | None = None
    id: str = ""
    at: int = 0

    def __post_init__(self) -> None:
        if self.kind not in EPISODE_KINDS:
            raise ValueError(f"unknown episode kind {self.kind!r}; expected one of {EPISODE_KINDS}")
        self.at = self.at or int(time.time())
        self.id = self.id or f"EP-{self.kind[:4].upper()}-{uuid.uuid4().hex[:10]}"


def write_episode(
    episode: Episode,
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Store an episode, embed it, and attach it to the services it is about.

    The `ABOUT` edges are what make a written episode reachable structurally as well as by vector,
    so a later run finds it either by similar wording or by working on the same service. Writing
    only the vector would make the memory findable by one route and invisible to the other.

    **The guard counts the edges, not "did anything change".** The obvious check --
    `writes.touched_nothing` -- does not work here and silently passed: `MERGE (e:Episode)` plus
    `SET` creates the node and sets its properties, so the statement always reports a change even
    when every `MATCH (s:Service {name: ...})` found nothing and no `ABOUT` edge was created. The
    episode then exists, is vector-searchable, and is attached to nothing -- which is precisely the
    half-written state the guard was added to catch. So the returned edge count is compared against
    the number of services asked for.
    """
    gc = _client(tenant, client)
    wanted = list(dict.fromkeys(episode.services or []))
    result = gc.run(
        """
        MERGE (e:Episode {id: $id})
        SET e.kind = $kind, e.text = $text, e.summary = $summary, e.at = $at,
            e.incident = $incident, e.run_id = $run_id,
            e.embedding = vecf32($vec), e.source = 'agent'
        WITH e
        UNWIND $services AS service_name
        MATCH (s:Service {name: service_name})
        MERGE (e)-[:ABOUT]->(s)
        RETURN count(s) AS attached
        """,
        {
            "id": episode.id,
            "kind": episode.kind,
            "text": episode.text,
            "summary": episode.summary,
            "at": episode.at,
            "incident": episode.incident,
            "run_id": episode.run_id,
            "vec": embeddings.embed(episode.text),
            "services": wanted,
        },
        label="memory.write_episode",
        hops=1,
        collect_provenance=False,
    )
    attached = int((result.first() or {}).get("attached") or 0)
    if wanted and attached < len(wanted):
        raise RuntimeError(
            f"episode {episode.id} was not written to all its services: "
            f"attached {attached} of {len(wanted)} ({', '.join(wanted)}). "
            "A service name does not exist in this tenant."
        )
    return result


def timeline(
    *,
    limit: int = 10,
    kind: str | None = None,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Everything remembered, newest first -- the audit trail of what the system has concluded."""
    gc = _client(tenant, client)
    return gc.run(
        """
        MATCH (e:Episode)
        WHERE $kind IS NULL OR e.kind = $kind
        OPTIONAL MATCH (e)-[:ABOUT]->(s:Service)
        RETURN e.id AS episode, e.kind AS kind, e.summary AS summary, e.at AS at,
               e.run_id AS run_id, collect(DISTINCT s.name) AS services
        ORDER BY at DESC
        LIMIT $limit
        """,
        {"kind": kind, "limit": limit},
        label="memory.timeline",
        hops=1,
        focus_keys=("services",),
    )


def forget(
    *,
    kind: str = "agent_conclusion",
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> int:
    """Delete agent-written episodes. Used ONLY to set up the cold half of the persistence proof.

    Scoped to one kind by design: it must never remove the seeded incident retrospectives, because
    the proof's cold phase has to show zero *agent* memories while the historical memory the agent
    reads is still there. A blanket wipe would make the cold phase trivially empty.
    """
    gc = _client(tenant, client)
    result = gc.run(
        "MATCH (e:Episode {kind: $kind}) DETACH DELETE e",
        {"kind": kind},
        label="memory.forget",
        collect_provenance=False,
    )
    return int(result.writes.nodes_deleted)


# --------------------------------------------------------------------------------------
# Procedural memory
# --------------------------------------------------------------------------------------

def runbook_for(
    service: str,
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """The runbook that resolved the most recent prior incident on this service.

    Procedural memory reached through episodic memory: the runbook is not looked up by service, it
    is reached from the incident it actually resolved. That ordering is the claim -- the system
    proposes a fix because it worked before, not because someone tagged a document with a name.
    """
    gc = _client(tenant, client)
    return gc.run(
        """
        MATCH (i:Incident)-[:AFFECTED]->(s:Service {name: $service})
        MATCH (i)-[:RESOLVED_BY]->(r:Runbook)
        RETURN r.id AS runbook, r.title AS title, r.steps AS steps,
               r.expected_mttr_minutes AS expected_mttr_minutes,
               r.owner_team AS owner_team,
               i.id AS learned_from, i.at AS incident_at,
               i.mttr_minutes AS actual_mttr_last_time
        ORDER BY incident_at DESC
        LIMIT 1
        """,
        {"service": service},
        label="memory.runbook_for",
        hops=2,
        focus_keys=("runbook",),
    )


# --------------------------------------------------------------------------------------
# Agent-to-agent handoff
# --------------------------------------------------------------------------------------

def open_run(
    alert_id: str,
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> str:
    """Start an AgentRun node. The handoff state between agents lives in the graph, not in Python.

    This is the multi-agent requirement answered structurally. If the handoff were a dict passed
    between function calls, "three agents collaborating" would be an implementation detail with no
    evidence. As `(:AgentRun)-[:PRODUCED]->(:Episode)` it is queryable after the fact: which agent
    concluded what, in what order, on which run.
    """
    gc = _client(tenant, client)
    run_id = f"RUN-{uuid.uuid4().hex[:12]}"
    gc.run(
        """
        MERGE (r:AgentRun {id: $id})
        SET r.alert = $alert, r.started_at = $at, r.status = 'running',
            r.tenant = $tenant, r.source = 'agent'
        """,
        {"id": run_id, "alert": alert_id, "at": int(time.time()), "tenant": gc.graph_name},
        label="memory.open_run",
        collect_provenance=False,
    )
    return run_id


def record_handoff(
    run_id: str,
    *,
    from_agent: str,
    to_agent: str,
    state: dict[str, Any],
    summary: str,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> str:
    """Persist one agent's output as the next agent's input, as a graph node.

    `state` is stored flattened to scalars. FalkorDB will not store a nested map as a property, and
    the failure is a type error at write time rather than a silent truncation -- so the flattening
    is explicit here instead of being discovered during the demo.
    """
    gc = _client(tenant, client)
    episode = Episode(
        kind="handoff",
        text=f"{from_agent} -> {to_agent}: {summary}",
        summary=summary,
        services=[v for k, v in state.items() if k.endswith("service") and isinstance(v, str)],
        run_id=run_id,
    )
    write_episode(episode, tenant=tenant, client=gc)
    gc.run(
        """
        MATCH (r:AgentRun {id: $run_id}), (e:Episode {id: $ep})
        MERGE (r)-[:PRODUCED]->(e)
        SET e.from_agent = $from_agent, e.to_agent = $to_agent, e.state = $state
        """,
        {
            "run_id": run_id,
            "ep": episode.id,
            "from_agent": from_agent,
            "to_agent": to_agent,
            "state": _flatten(state),
        },
        label="memory.record_handoff",
        collect_provenance=False,
    )
    return episode.id


def _flatten(state: dict[str, Any]) -> str:
    """Scalars only, as a readable one-liner. A nested map is rejected by the engine, not truncated."""
    parts = []
    for key in sorted(state):
        value = state[key]
        if isinstance(value, (list, tuple)):
            value = ",".join(str(v) for v in value)
        parts.append(f"{key}={value}")
    return "; ".join(parts)


def close_run(
    run_id: str,
    *,
    outcome: str,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    gc = _client(tenant, client)
    return gc.run(
        """
        MATCH (r:AgentRun {id: $id})
        SET r.status = 'done', r.outcome = $outcome, r.ended_at = $at
        RETURN r.id AS run, r.outcome AS outcome
        """,
        {"id": run_id, "outcome": outcome, "at": int(time.time())},
        label="memory.close_run",
        collect_provenance=False,
    )


def run_trace(
    run_id: str,
    *,
    tenant: str | None = None,
    client: GraphClient | None = None,
) -> QueryResult:
    """Replay one agent run from the graph: every handoff, in order, with its state."""
    gc = _client(tenant, client)
    return gc.run(
        """
        MATCH (r:AgentRun {id: $id})-[:PRODUCED]->(e:Episode)
        RETURN e.from_agent AS from_agent, e.to_agent AS to_agent,
               e.summary AS summary, e.state AS state, e.at AS at
        ORDER BY at ASC
        """,
        {"id": run_id},
        label="memory.run_trace",
        hops=1,
    )


def describe() -> dict[str, Any]:
    return {
        "embedding_backend": embeddings.backend(),
        "semantic": embeddings.is_semantic(),
        "dim": SETTINGS.embedding_dim,
        "kinds": list(EPISODE_KINDS),
    }


__all__ = [
    "EPISODE_KINDS",
    "Episode",
    "close_run",
    "describe",
    "forget",
    "hit_count",
    "open_run",
    "recall",
    "record_handoff",
    "run_trace",
    "runbook_for",
    "timeline",
    "write_episode",
]
