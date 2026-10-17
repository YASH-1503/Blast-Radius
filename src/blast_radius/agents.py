"""Three agents that hand work to each other through the graph.

Triage decides how bad it is, Diagnosis decides what caused it, Comms decides who to tell and what
to propose. The division is not decoration: each one needs a different part of the graph, and the
handoff between them is a node, so afterwards you can ask the database what the agents said to each
other and in what order.

**The ranking is the product, and it must state what it rejected.** Any system can emit an answer.
What makes this one auditable is that `_rank` records, for every candidate it did not choose, the
reason it lost -- further from the alert, lower tier, lower centrality. A conclusion with no
rejected alternatives is indistinguishable from a guess that happened to be right, and a reviewer
has no way to disagree with it.

**There is no language model in this loop.** The decisions are made by graph queries and an explicit
ranking function, which is deliberate: an LLM wrapped around a graph makes it impossible to tell
whether the graph or the model produced the answer. Everything here is reproducible and every step
cites the Cypher that justified it. An LLM belongs at the edges -- turning this trace into prose --
not in the deciding.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from . import memory, tools
from .data import ALERT, CRITICAL_TIER
from .db import GraphClient

# Ranking weights. Named and exported so the UI can show WHY one candidate beat another, and so a
# reviewer can disagree with the weighting rather than with an opaque verdict.
WEIGHTS = {
    "hops": -3.0,          # closer to the alert is more likely to be the cause
    "critical_tier": 2.5,  # a critical-tier service failing explains more
    "pagerank": 12.0,      # much depends on it
    "recency": 1.5,        # deployed more recently within the window
    "focused_deploy": 1.0, # the deploy was specific to this service, not a repo-wide commit
}


@dataclass
class Step:
    """One thing an agent did, with the evidence it did it with."""

    agent: str
    action: str
    note: str
    call: tools.ToolCall | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "note": self.note,
            "evidence": self.call.to_dict() if self.call else None,
        }


@dataclass
class RunResult:
    run_id: str
    tenant: str
    alert: dict[str, Any]
    steps: list[Step] = field(default_factory=list)
    verdict: dict[str, Any] = field(default_factory=dict)
    ruled_out: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "tenant": self.tenant,
            "alert": self.alert,
            "steps": [s.to_dict() for s in self.steps],
            "verdict": self.verdict,
            "ruled_out": self.ruled_out,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "tool_calls": sum(1 for s in self.steps if s.call),
        }

    def explain(self) -> str:
        lines = [
            f"run      : {self.run_id}  ({self.tenant})",
            f"alert    : {self.alert.get('message')}",
            "",
        ]
        current = None
        for step in self.steps:
            if step.agent != current:
                current = step.agent
                lines.append(f"-- {current.upper()} " + "-" * (58 - len(current)))
            marker = "*" if step.call else " "
            lines.append(f" {marker} {step.action:22s} {step.note}")
        lines += ["", "CHOSE    : " + str(self.verdict.get("cause")), ""]
        for row in self.ruled_out:
            lines.append(f" ruled out {row['service']:16s} {row['because']}")
        lines += [
            "",
            f"severity : {self.verdict.get('severity')} ({self.verdict.get('sev1_because')})",
            f"impact   : {self.verdict.get('customers')} customers, "
            f"{self.verdict.get('mrr_at_risk')} MRR at risk",
            f"action   : {self.verdict.get('action')} -- {self.verdict.get('runbook')} "
            f"(confidence {self.verdict.get('confidence')})",
            f"page     : {self.verdict.get('page')} ({self.verdict.get('page_reason')})",
            f"time     : {self.elapsed_ms:.1f} ms, "
            f"{sum(1 for s in self.steps if s.call)} tool calls",
        ]
        return "\n".join(lines)


def _rank(
    candidates: list[dict[str, Any]],
    pagerank: dict[str, float],
) -> list[tuple[dict[str, Any], float, list[str]]]:
    """Score each candidate and record, in words, what each factor contributed.

    The reasons are built here rather than reconstructed later so that the explanation cannot drift
    from the arithmetic that actually decided the outcome.
    """
    if not candidates:
        return []
    newest = max(c.get("deployed_at") or 0 for c in candidates)
    scored: list[tuple[dict[str, Any], float, list[str]]] = []
    for c in candidates:
        reasons: list[str] = []
        hops = c.get("hops_from_alert") or 0
        score = WEIGHTS["hops"] * hops
        reasons.append(f"{hops} hop(s) from the alert")

        if c.get("tier") == CRITICAL_TIER:
            score += WEIGHTS["critical_tier"]
            reasons.append(f"{CRITICAL_TIER}-tier")
        else:
            reasons.append(f"{c.get('tier')}-tier")

        pr = pagerank.get(c["service"], 0.0)
        score += WEIGHTS["pagerank"] * pr
        reasons.append(f"pagerank {pr:.4f}")

        age_minutes = max(0, (newest - (c.get("deployed_at") or 0)) // 60)
        recency = WEIGHTS["recency"] * (1.0 if age_minutes == 0 else 0.0)
        score += recency
        reasons.append(
            "most recent deploy in the window" if age_minutes == 0
            else f"{age_minutes} min older than the newest"
        )

        if c.get("commits_in_deploy"):
            score += WEIGHTS["focused_deploy"]

        scored.append((c, score, reasons))
    return sorted(scored, key=lambda row: -row[1])


# --------------------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------------------

def run_incident(
    *,
    tenant: str | None = None,
    alert: dict[str, Any] | None = None,
    client: GraphClient | None = None,
) -> RunResult:
    """Triage -> Diagnosis -> Comms, with every handoff written to the graph."""
    started = time.perf_counter()
    gc = client or GraphClient(tenant)
    alert = dict(alert or ALERT)
    service = alert["service"]

    run_id = memory.open_run(alert["id"], client=gc)
    out = RunResult(run_id=run_id, tenant=gc.graph_name, alert=alert)

    def use(agent: str, tool_name: str, args: dict[str, Any], note: str) -> tools.ToolCall:
        call = tools.call(tool_name, args, client=gc)
        out.steps.append(Step(agent=agent, action=tool_name, note=note, call=call))
        return call

    def say(agent: str, action: str, note: str) -> None:
        out.steps.append(Step(agent=agent, action=action, note=note))

    # -- TRIAGE: how bad is this, and who is exposed? ----------------------------------
    sev = use("triage", "severity", {"service": service},
              "derive severity from tier and revenue, not from a field").result.first() or {}
    blast = use("triage", "blast_radius", {"service": service},
                "follow service -> endpoint -> customer").result.first() or {}
    say("triage", "conclude",
        f"{sev.get('severity')} because {sev.get('sev1_because')}; "
        f"{blast.get('customer_count')} customers, {blast.get('mrr_at_risk')} MRR")

    memory.record_handoff(
        run_id,
        from_agent="triage",
        to_agent="diagnosis",
        state={
            "alert_service": service,
            "severity": sev.get("severity"),
            "customers": blast.get("customer_count"),
            "mrr_at_risk": blast.get("mrr_at_risk"),
        },
        summary=f"{sev.get('severity')}, {blast.get('customer_count')} customers exposed",
        client=gc,
    )

    # -- DIAGNOSIS: what changed, and which change explains it? ------------------------
    candidates = use("diagnosis", "candidate_causes",
                     {"service": service, "lookback_hours": int(alert["lookback_hours"])},
                     "deploys in the window on anything upstream").result.rows
    pr_rows = use("diagnosis", "critical_services", {"limit": 24},
                  "pagerank, to break ties between equally close candidates").result.rows
    pagerank = {r["service"]: float(r.get("pagerank") or 0.0) for r in pr_rows}

    ranked = _rank(list(candidates), pagerank)
    if not ranked:
        say("diagnosis", "conclude", "no deploy inside the lookback window; not a deploy-caused incident")
        memory.close_run(run_id, outcome="no_candidate", client=gc)
        out.elapsed_ms = (time.perf_counter() - started) * 1000.0
        return out

    best, best_score, best_reasons = ranked[0]
    cause = best["service"]
    say("diagnosis", "choose",
        f"{cause} (score {best_score:.2f}): " + "; ".join(best_reasons))

    for other, score, reasons in ranked[1:]:
        out.ruled_out.append({
            "service": other["service"],
            "score": round(score, 2),
            "because": f"scored {score:.2f} vs {best_score:.2f} -- " + "; ".join(reasons),
        })
        say("diagnosis", "rule out", f"{other['service']}: {'; '.join(reasons)}")

    path = use("diagnosis", "path_between", {"source": service, "target": cause},
               "show the exact dependency route from the alert to the suspect").result.first() or {}

    prior = use("diagnosis", "past_incidents", {"service": cause},
                "structural memory: incidents on this part of the system").result.rows
    similar = use("diagnosis", "recall",
                  {"text": f"{alert['message']} suspect {cause}", "k": 3},
                  "semantic memory: episodes that read like this one").result.rows

    memory.record_handoff(
        run_id,
        from_agent="diagnosis",
        to_agent="comms",
        state={
            "cause_service": cause,
            "deployed_by": best.get("deployed_by"),
            "sha": (best.get("sha") or "")[:12],
            "ruled_out": [r["service"] for r in out.ruled_out],
            "prior_incidents": [p["incident"] for p in prior],
        },
        summary=f"cause is {cause}, ruled out {len(out.ruled_out)} alternative(s)",
        client=gc,
    )

    # -- COMMS: who to tell, and what to propose ---------------------------------------
    page = use("comms", "who_to_page", {"service": cause},
               "owning team vs who actually writes this code").result.first() or {}
    runbook = use("comms", "runbook_for", {"service": cause},
                  "the runbook that resolved the last incident here").result.first() or {}
    risk = use("comms", "key_person_risk", {},
               "is the expertise concentrated in one person?").result.rows

    # Page the owning team's escalation channel; name the real expert as the person to pull in.
    expert = page.get("top_author")
    mismatch = bool(page.get("ownership_mismatch"))
    page_reason = (
        f"{page.get('owning_team')} owns it, but {expert} wrote "
        f"{page.get('top_author_commits')} of its commits from {page.get('top_author_team')}"
        if mismatch else
        f"{page.get('owning_team')} both owns it and holds its top author ({expert})"
    )

    confidence = "high" if (prior or similar) and runbook.get("runbook") else "medium"
    out.verdict = {
        "cause": cause,
        "deployed_by": best.get("deployed_by"),
        "sha": (best.get("sha") or "")[:12],
        "change": best.get("change"),
        "version": best.get("version"),
        "hops_from_alert": best.get("hops_from_alert"),
        "severity": sev.get("severity"),
        "sev1_because": sev.get("sev1_because"),
        "customers": blast.get("customer_count"),
        "mrr_at_risk": blast.get("mrr_at_risk"),
        "path": path.get("hops"),
        "action": "propose_rollback" if runbook.get("runbook") else "escalate_to_human",
        "runbook": runbook.get("runbook"),
        "runbook_steps": runbook.get("steps"),
        "learned_from": runbook.get("learned_from"),
        "confidence": confidence,
        "page": page.get("escalation_channel"),
        "pull_in": expert,
        "page_reason": page_reason,
        "ownership_mismatch": mismatch,
        "key_person": (risk[0] if risk else {}),
        "prior_incidents": [p["incident"] for p in prior],
        "similar_episodes": [s["episode"] for s in similar],
    }

    say("comms", "propose",
        f"{out.verdict['action']} via {out.verdict['runbook']} "
        f"(confidence {confidence}); page {out.verdict['page']}, pull in {expert}")

    # The agent's own conclusion becomes memory. This is what the cold/warm proof detects, and it
    # is written LAST so a failure anywhere above leaves no false memory behind.
    memory.write_episode(
        memory.Episode(
            kind="agent_conclusion",
            text=(
                f"{alert['message']} Cause: {cause} deployed by {best.get('deployed_by')} "
                f"({(best.get('sha') or '')[:12]}): {best.get('change')}. "
                f"Ruled out {', '.join(r['service'] for r in out.ruled_out) or 'nothing'}. "
                f"Proposed {out.verdict['action']} via {runbook.get('runbook')}."
            ),
            summary=f"{cause} caused {alert['id']}; proposed {runbook.get('runbook')}",
            services=[service, cause],
            incident=alert["id"],
            run_id=run_id,
        ),
        client=gc,
    )
    memory.close_run(run_id, outcome=f"{out.verdict['action']} {runbook.get('runbook')}", client=gc)

    out.elapsed_ms = (time.perf_counter() - started) * 1000.0
    return out


__all__ = ["WEIGHTS", "RunResult", "Step", "run_incident"]
