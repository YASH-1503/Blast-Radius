"""The synthetic layer: topology, money, and institutional memory.

Everything in this file is invented, and says so -- every node it produces carries
`source: 'synthetic'`. That is not a shortcut, it is the only rule-compliant option for these
specific layers:

| layer                        | why it cannot be real                                        |
|------------------------------|--------------------------------------------------------------|
| Service / DEPENDS_ON         | no company publishes its internal dependency graph           |
| Customer, MRR                | revenue per named customer is restricted commercial data      |
| Incident, Runbook, Decision  | nobody publishes their outage history or postmortems          |

The *human* layer -- people, commits, pull requests, and who actually knows which code -- is real
and comes from the public GitHub API (see `github_ingest.py`). So "is your data real?" has a precise
answer rather than a yes or a no, and `schema.provenance()` returns it as a query.

**This file deliberately contains no `Person` nodes at all.** An earlier iteration seeded filler
humans here to pad out teams, and they survived the switch to real contributors: the paging roster
came out as `['GomezGab', 'dev00', 'dev03', 'dev09']` -- three invented names standing beside one
real maintainer, in a graph whose humans are supposed to be real. Teams are now populated purely
from real contributors at ingest time, so an invented person cannot be paged because none exists.

The generator is seeded, so the same seed produces the same graph, the same answer, and therefore a
demo that can be rehearsed and recorded. Random data would be unrecordable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from .config import SETTINGS

SYNTHETIC = "synthetic"

# --------------------------------------------------------------------------------------
# Services
# --------------------------------------------------------------------------------------
# `tier` is what makes severity a judgement instead of a count: two services equally far from the
# victim are not equally guilty. Tiers are critical > important > standard.

SERVICES: tuple[tuple[str, str, str], ...] = (
    # (name, tier, what it does -- shown in the UI tooltip)
    ("api-gateway",     "critical",  "Public edge. Every customer request enters here."),
    ("checkout",        "critical",  "Order placement and payment orchestration."),
    ("pricing-legacy",  "critical",  "Price and discount resolution. Pre-dates the rewrite."),
    ("identity",        "critical",  "Authentication, sessions, and token issuance."),
    ("payments",        "critical",  "Card capture and settlement."),
    ("ledger",          "critical",  "Double-entry record of every money movement."),
    ("cart",            "important", "Basket state between browse and checkout."),
    ("catalog",         "important", "Product records and availability."),
    ("inventory",       "important", "Stock counts per warehouse."),
    ("search",          "important", "Query parsing and result ranking."),
    ("fraud",           "important", "Pre-authorisation risk scoring."),
    ("invoicing",       "important", "Invoice generation and delivery."),
    ("tax",             "important", "Jurisdiction tax rate resolution."),
    ("shipping",        "important", "Carrier rates and label creation."),
    ("email",           "standard",  "Transactional email dispatch."),
    ("media",           "standard",  "Product image resize and CDN upload."),
    ("notifications",   "standard",  "Fan-out to email, SMS and push."),
    ("sms",             "standard",  "SMS gateway adapter."),
    ("push",            "standard",  "Mobile push adapter."),
    ("reviews",         "standard",  "Customer review storage and moderation."),
    ("recommendations", "standard",  "Related-product suggestions."),
    ("analytics",       "standard",  "Event collection and rollups."),
    ("feature-flags",   "standard",  "Runtime configuration toggles."),
    ("audit",           "standard",  "Append-only record of privileged actions."),
)

# --------------------------------------------------------------------------------------
# The dependency spine
# --------------------------------------------------------------------------------------
# This topology is the demo. The incident fires on `api-gateway`, and three services deployed in the
# window -- but they are not equally plausible causes:
#
#     api-gateway -> checkout -> pricing-legacy          pricing-legacy: 2 hops, tier CRITICAL
#     api-gateway -> checkout -> cart -> media           media:          3 hops, tier standard
#     api-gateway -> identity -> email -> notifications  notifications:  3 hops, tier standard
#
# So the agent has to rank, and -- this is the part that matters -- has to SAY WHAT IT REJECTED AND
# WHY. An agent that only emits an answer is indistinguishable from a lookup table.

DEPENDS_ON: tuple[tuple[str, str], ...] = (
    ("api-gateway", "checkout"),
    ("api-gateway", "identity"),
    ("api-gateway", "catalog"),
    ("api-gateway", "search"),
    ("checkout", "pricing-legacy"),
    ("checkout", "cart"),
    ("checkout", "payments"),
    ("checkout", "inventory"),
    ("checkout", "tax"),
    ("checkout", "fraud"),
    ("cart", "media"),
    ("cart", "catalog"),
    ("catalog", "media"),
    ("catalog", "search"),
    ("payments", "ledger"),
    ("payments", "fraud"),
    ("ledger", "audit"),
    ("invoicing", "ledger"),
    ("invoicing", "tax"),
    ("identity", "email"),
    ("identity", "audit"),
    ("email", "notifications"),
    ("notifications", "sms"),
    ("notifications", "push"),
    ("search", "recommendations"),
    ("recommendations", "analytics"),
    ("reviews", "media"),
    ("shipping", "inventory"),
    ("pricing-legacy", "feature-flags"),
    ("pricing-legacy", "catalog"),
)

# --------------------------------------------------------------------------------------
# Databases
# --------------------------------------------------------------------------------------
# Kept as a separate label rather than another Service, because a shared database is a different
# shape of risk: two services that share one are coupled without any DEPENDS_ON edge between them.

DATABASES: tuple[tuple[str, str], ...] = (
    ("orders-db",    "postgres"),
    ("pricing-db",   "postgres"),
    ("identity-db",  "postgres"),
    ("ledger-db",    "postgres"),
    ("catalog-db",   "mysql"),
    ("cart-cache",   "redis"),
    ("search-index", "elasticsearch"),
    ("media-store",  "s3"),
    ("events-lake",  "s3"),
)

READS_FROM: tuple[tuple[str, str], ...] = (
    ("checkout", "orders-db"),
    ("checkout", "pricing-db"),
    ("pricing-legacy", "pricing-db"),
    ("pricing-legacy", "catalog-db"),
    ("identity", "identity-db"),
    ("cart", "cart-cache"),
    ("catalog", "catalog-db"),
    ("search", "search-index"),
    ("media", "media-store"),
    ("payments", "ledger-db"),
    ("ledger", "ledger-db"),
    ("invoicing", "ledger-db"),
    ("analytics", "events-lake"),
    ("recommendations", "events-lake"),
    ("inventory", "catalog-db"),
)

WRITES_TO: tuple[tuple[str, str], ...] = (
    ("checkout", "orders-db"),
    ("pricing-legacy", "pricing-db"),
    ("identity", "identity-db"),
    ("cart", "cart-cache"),
    ("catalog", "catalog-db"),
    ("media", "media-store"),
    ("ledger", "ledger-db"),
    ("payments", "ledger-db"),
    ("analytics", "events-lake"),
    ("inventory", "catalog-db"),
)

# --------------------------------------------------------------------------------------
# Endpoints -- the only path from money to machinery
# --------------------------------------------------------------------------------------
# A Customer never points at a Service. Revenue-at-risk has to follow the route a real request
# takes: customer -> endpoint -> service -> ...dependencies. That makes the MRR figure a traversal
# the agent can show, instead of a column it can read.
#
# An earlier version gave the edge gateway no endpoints, and `api-gateway` therefore reported zero
# customers -- the most critical service in the graph looked like it had no business impact. Every
# customer-facing service needs at least one.

ENDPOINTS: tuple[tuple[str, str, str], ...] = (
    # (path, service that exposes it, human description)
    ("/v1/checkout",      "api-gateway", "Place an order"),
    ("/v1/cart",          "api-gateway", "Read and modify the basket"),
    ("/v1/catalog",       "api-gateway", "Browse products"),
    ("/v1/search",        "api-gateway", "Product search"),
    ("/v1/login",         "api-gateway", "Authenticate"),
    ("/v1/invoices",      "invoicing",   "Download invoices"),
    ("/v1/shipping/rate", "shipping",    "Quote a shipment"),
    ("/v1/reviews",       "reviews",     "Read and post reviews"),
)

# --------------------------------------------------------------------------------------
# Customers and revenue
# --------------------------------------------------------------------------------------
# Six customers reach `api-gateway`, totalling 971,000 MRR. That number is the one the agent must
# arrive at by traversal, and it is why the alert is a sev1 rather than a ticket.

CUSTOMERS: tuple[tuple[str, int, str, tuple[str, ...]], ...] = (
    # (name, mrr, plan, endpoints used)
    ("Northwind Retail",  420_000, "enterprise",
     ("/v1/checkout", "/v1/cart", "/v1/catalog", "/v1/search", "/v1/login", "/v1/invoices")),
    ("Globex Commerce",   210_000, "enterprise",
     ("/v1/checkout", "/v1/catalog", "/v1/login", "/v1/shipping/rate")),
    ("Initech Systems",   150_000, "enterprise",
     ("/v1/checkout", "/v1/cart", "/v1/login")),
    ("Umbrella Foods",     95_000, "growth",
     ("/v1/checkout", "/v1/catalog", "/v1/login", "/v1/reviews")),
    ("Soylent Corp",       61_000, "growth",
     ("/v1/search", "/v1/catalog", "/v1/login")),
    ("Hooli Shop",         35_000, "starter",
     ("/v1/cart", "/v1/catalog", "/v1/reviews")),
)

# --------------------------------------------------------------------------------------
# Teams -- who is ALLOWED to be paged
# --------------------------------------------------------------------------------------
# The gap between this and `WORKED_ON` is the sharpest finding the graph produces: the roster you
# are permitted to page is not the set of people who can actually fix the thing. Ownership is
# declared by an org chart; expertise is revealed by commit history. They drift, and the graph shows
# the drift as a structural fact rather than an opinion.
#
# Members are NOT listed here -- they are attached from real GitHub contributors at ingest.

TEAMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    # (team, escalation channel, services owned)
    ("Payments Platform", "#payments-oncall",
     ("checkout", "pricing-legacy", "payments", "ledger", "invoicing", "tax")),
    ("Core Platform", "#core-oncall",
     ("api-gateway", "identity", "feature-flags", "audit")),
    ("Discovery", "#discovery-oncall",
     ("search", "catalog", "recommendations", "reviews")),
    ("Fulfilment", "#fulfilment-oncall",
     ("inventory", "shipping", "cart")),
    ("Growth", "#growth-oncall",
     ("media", "notifications", "email", "sms", "push", "analytics")),
    ("Risk", "#risk-oncall",
     ("fraud",)),
)

# --------------------------------------------------------------------------------------
# Episodic memory: past incidents
# --------------------------------------------------------------------------------------
# INC-1041 is the precedent the agent must find. Without it the agent can only say "roll back
# something"; with it, it says "roll back, via RB-07, which fixed this exact shape last month, MTTR
# 14 minutes". That difference is the entire argument for giving an agent a memory.

INCIDENTS: tuple[dict[str, Any], ...] = (
    {
        "id": "INC-1041",
        "title": "Checkout 5xx after pricing-legacy discount-rule deploy",
        "days_ago": 34,
        "severity": "sev1",
        "affected": ("api-gateway", "checkout", "pricing-legacy"),
        "root_cause": "A discount rule change in pricing-legacy returned null for bundle SKUs; "
                      "checkout treated null as a fatal and surfaced 502 at the edge.",
        "resolution": "Rolled back pricing-legacy to the previous tag; re-released behind a flag.",
        "runbook": "RB-07",
        "mttr_minutes": 14,
    },
    {
        "id": "INC-0987",
        "title": "Cart latency spike from cart-cache eviction storm",
        "days_ago": 71,
        "severity": "sev2",
        "affected": ("cart", "checkout"),
        "root_cause": "cart-cache maxmemory policy evicted hot keys during a catalog reindex.",
        "resolution": "Raised maxmemory and moved reindex off peak.",
        "runbook": "RB-03",
        "mttr_minutes": 52,
    },
    {
        "id": "INC-0954",
        "title": "Notification fan-out backlog after sms provider timeout",
        "days_ago": 96,
        "severity": "sev3",
        "affected": ("notifications", "sms"),
        "root_cause": "Upstream SMS provider latency exceeded the fan-out worker timeout.",
        "resolution": "Shortened the provider timeout and drained the backlog.",
        "runbook": "RB-11",
        "mttr_minutes": 38,
    },
    {
        "id": "INC-0902",
        "title": "Stale prices served after pricing-db failover",
        "days_ago": 128,
        "severity": "sev2",
        "affected": ("pricing-legacy", "checkout"),
        "root_cause": "Read replica promoted mid-write; pricing-legacy cached the pre-failover view.",
        "resolution": "Flushed the price cache and pinned reads to the primary during failover.",
        "runbook": "RB-07",
        "mttr_minutes": 26,
    },
)

# --------------------------------------------------------------------------------------
# Procedural memory: runbooks
# --------------------------------------------------------------------------------------

RUNBOOKS: tuple[dict[str, Any], ...] = (
    {
        "id": "RB-07",
        "title": "Roll back a pricing-legacy release",
        "owner_team": "Payments Platform",
        "expected_mttr_minutes": 14,
        "steps": (
            "Confirm the suspect deploy tag from the deploy log.",
            "Freeze the pricing-legacy pipeline to stop a re-deploy racing the rollback.",
            "Roll back to the previous tag; the service is stateless, so no data step is needed.",
            "Flush the price cache -- a stale cache will reproduce the symptom after rollback.",
            "Watch checkout 5xx for 10 minutes before declaring recovery.",
        ),
    },
    {
        "id": "RB-03",
        "title": "Relieve cart-cache memory pressure",
        "owner_team": "Fulfilment",
        "expected_mttr_minutes": 45,
        "steps": (
            "Check eviction rate and maxmemory policy on cart-cache.",
            "Raise maxmemory if headroom exists; otherwise shed non-essential keys.",
            "Verify no catalog reindex is running concurrently.",
        ),
    },
    {
        "id": "RB-11",
        "title": "Drain a notification fan-out backlog",
        "owner_team": "Growth",
        "expected_mttr_minutes": 35,
        "steps": (
            "Identify the slow downstream adapter from worker timings.",
            "Shorten that adapter's timeout so workers stop blocking.",
            "Scale fan-out workers and drain, watching duplicate-send guards.",
        ),
    },
    {
        "id": "RB-02",
        "title": "Edge 5xx triage at api-gateway",
        "owner_team": "Core Platform",
        "expected_mttr_minutes": 20,
        "steps": (
            "Split 5xx by upstream to find which dependency is failing.",
            "Check deploys in the last 2 hours across the dependency closure.",
            "Escalate to the owning team of the implicated service.",
        ),
    },
)

# --------------------------------------------------------------------------------------
# Decisions -- why a dependency exists at all
# --------------------------------------------------------------------------------------
# Before blaming a dependency, it is worth knowing it was a deliberate choice with a recorded
# trade-off. This is the layer that turns "the graph knows what connects to what" into "the graph
# knows why", and no amount of log scraping produces it.
#
# The author is NOT set here. It is attached at ingest to the top real contributor on the service
# the decision concerns -- because an invented author in a graph of real humans is exactly the leak
# described in the module docstring.

DECISIONS: tuple[dict[str, Any], ...] = (
    {
        "id": "ADR-012",
        "title": "Keep checkout on pricing-legacy until the pricing rewrite lands",
        "days_ago": 210,
        "about_service": "pricing-legacy",
        "dependency": ("checkout", "pricing-legacy"),
        "decision": "checkout continues to call pricing-legacy synchronously.",
        "rationale": "The rewrite does not yet implement bundle or regional discount rules. "
                     "Dual-writing prices was judged a larger correctness risk than the "
                     "known fragility of the legacy path.",
        "trade_off": "Accepts a critical-tier dependency on a service nobody wants to own, in "
                     "exchange for not splitting price truth across two systems.",
        "revisit_after_days": 90,
    },
    {
        "id": "ADR-008",
        "title": "Notifications fan out asynchronously, never in the request path",
        "days_ago": 320,
        "about_service": "notifications",
        "dependency": ("email", "notifications"),
        "decision": "identity and email enqueue; notifications fans out off the request path.",
        "rationale": "A provider outage must degrade delivery latency, not availability.",
        "trade_off": "Delivery is eventually consistent, so 'did the customer get it' needs a "
                     "separate query.",
        "revisit_after_days": 365,
    },
    {
        "id": "ADR-015",
        "title": "cart-cache is a cache, not a store of record",
        "days_ago": 150,
        "about_service": "cart",
        "dependency": ("cart", "media"),
        "decision": "Basket state may be lost; checkout re-resolves from catalog on submit.",
        "rationale": "Durability for an abandoned basket is not worth a write-path dependency.",
        "trade_off": "A cache flush loses in-flight baskets, which support must be able to explain.",
        "revisit_after_days": 180,
    },
)

# --------------------------------------------------------------------------------------
# The alert that starts the demo
# --------------------------------------------------------------------------------------

ALERT: dict[str, Any] = {
    "id": "ALRT-5571",
    "service": "api-gateway",
    "metric": "http_5xx_ratio",
    "value": 0.37,
    "threshold": 0.02,
    "window_minutes": 10,
    "message": "api-gateway 5xx ratio 37% over 10m (threshold 2%)",
    # How far back `candidate_causes` looks for deploys. Wide enough to catch the real change,
    # narrow enough that the agent still has to discriminate between three of them.
    "lookback_hours": 6,
}


# --------------------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------------------

@dataclass
class SyntheticWorld:
    """Rows ready for ingest. Deterministic for a given seed."""

    services: list[dict[str, Any]] = field(default_factory=list)
    databases: list[dict[str, Any]] = field(default_factory=list)
    endpoints: list[dict[str, Any]] = field(default_factory=list)
    customers: list[dict[str, Any]] = field(default_factory=list)
    teams: list[dict[str, Any]] = field(default_factory=list)
    depends_on: list[dict[str, Any]] = field(default_factory=list)
    reads_from: list[dict[str, Any]] = field(default_factory=list)
    writes_to: list[dict[str, Any]] = field(default_factory=list)
    exposes: list[dict[str, Any]] = field(default_factory=list)
    uses: list[dict[str, Any]] = field(default_factory=list)
    owns: list[dict[str, Any]] = field(default_factory=list)
    incidents: list[dict[str, Any]] = field(default_factory=list)
    runbooks: list[dict[str, Any]] = field(default_factory=list)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    dependencies: list[dict[str, Any]] = field(default_factory=list)
    alert: dict[str, Any] = field(default_factory=dict)

    def totals(self) -> dict[str, int]:
        return {
            "services": len(self.services),
            "databases": len(self.databases),
            "endpoints": len(self.endpoints),
            "customers": len(self.customers),
            "teams": len(self.teams),
            "depends_on": len(self.depends_on),
            "incidents": len(self.incidents),
            "runbooks": len(self.runbooks),
            "decisions": len(self.decisions),
        }


def service_names() -> tuple[str, ...]:
    return tuple(name for name, _tier, _desc in SERVICES)


# The tier vocabulary, named once. Severity filters on the highest tier, and the spike shipped a
# severity query testing `tier = 'tier-0'` -- a value no service has ever carried. It matched
# nothing, `tier0_impacted` came back empty, and the verdict was still "sev1" because the revenue
# branch of the same expression fired. The answer was right for the wrong reason, which is worse
# than being wrong: a test asserting the verdict passes while half the rule is dead code.
TIERS: tuple[str, ...] = ("critical", "important", "standard")
CRITICAL_TIER = TIERS[0]


def assert_tier_vocabulary() -> None:
    """Fail at import if any service carries a tier outside the declared vocabulary."""
    unknown = sorted({tier for _n, tier, _d in SERVICES} - set(TIERS))
    if unknown:
        raise ValueError(f"services carry undeclared tiers {unknown}; expected one of {TIERS}")


def tier_of(service: str) -> str:
    for name, tier, _desc in SERVICES:
        if name == service:
            return tier
    return "standard"


def dependency_id(upstream: str, downstream: str) -> str:
    """Stable id for the reified dependency a Decision can point at."""
    return f"{upstream}->{downstream}"


# Run at import: a service carrying a tier outside TIERS would silently disable the severity rule.
assert_tier_vocabulary()


def build(seed: int | None = None) -> SyntheticWorld:
    """Materialise the synthetic world. Same seed -> byte-identical output."""
    rng = random.Random(seed if seed is not None else SETTINGS.seed)
    world = SyntheticWorld()

    world.services = [
        {
            "name": name,
            "tier": tier,
            "description": description,
            "language": rng.choice(("python", "go", "typescript", "rust", "java")),
            "source": SYNTHETIC,
        }
        for name, tier, description in SERVICES
    ]

    world.databases = [
        {"name": name, "engine": engine, "source": SYNTHETIC} for name, engine in DATABASES
    ]

    world.endpoints = [
        {"path": path, "service": service, "description": description, "source": SYNTHETIC}
        for path, service, description in ENDPOINTS
    ]
    world.exposes = [{"service": service, "path": path} for path, service, _d in ENDPOINTS]

    for name, mrr, plan, used in CUSTOMERS:
        world.customers.append(
            {"name": name, "mrr": mrr, "plan": plan, "source": SYNTHETIC}
        )
        world.uses += [{"customer": name, "path": path} for path in used]

    for team, channel, owned in TEAMS:
        world.teams.append({"name": team, "channel": channel, "source": SYNTHETIC})
        world.owns += [{"team": team, "service": service} for service in owned]

    # Reify each dependency so a Decision has something to point at, and record its criticality
    # from the downstream tier -- a dependency on a critical service is itself critical.
    for upstream, downstream in DEPENDS_ON:
        world.depends_on.append({"upstream": upstream, "downstream": downstream})
        world.dependencies.append(
            {
                "id": dependency_id(upstream, downstream),
                "upstream": upstream,
                "downstream": downstream,
                "criticality": tier_of(downstream),
                "source": SYNTHETIC,
            }
        )

    world.reads_from = [{"service": s, "database": d} for s, d in READS_FROM]
    world.writes_to = [{"service": s, "database": d} for s, d in WRITES_TO]

    world.incidents = [
        {
            "id": inc["id"],
            "title": inc["title"],
            "days_ago": inc["days_ago"],
            "severity": inc["severity"],
            "root_cause": inc["root_cause"],
            "resolution": inc["resolution"],
            "mttr_minutes": inc["mttr_minutes"],
            "runbook": inc["runbook"],
            "affected": list(inc["affected"]),
            "source": SYNTHETIC,
        }
        for inc in INCIDENTS
    ]

    world.runbooks = [
        {
            "id": rb["id"],
            "title": rb["title"],
            "owner_team": rb["owner_team"],
            "expected_mttr_minutes": rb["expected_mttr_minutes"],
            "steps": list(rb["steps"]),
            "source": SYNTHETIC,
        }
        for rb in RUNBOOKS
    ]

    world.decisions = [
        {
            "id": d["id"],
            "title": d["title"],
            "days_ago": d["days_ago"],
            "about_service": d["about_service"],
            "dependency_id": dependency_id(*d["dependency"]),
            "decision": d["decision"],
            "rationale": d["rationale"],
            "trade_off": d["trade_off"],
            "revisit_after_days": d["revisit_after_days"],
            "source": SYNTHETIC,
        }
        for d in DECISIONS
    ]

    world.alert = {**ALERT, "source": SYNTHETIC}
    return world


def mrr_at_risk_for(service: str = "api-gateway") -> int:
    """The expected answer, computed in Python purely so a test can assert the graph agrees.

    Deliberately NOT used by the agent -- if the agent read this it would be a lookup, and the
    traversal would be decoration. It exists so that `test_blast_radius_matches_python` can catch a
    modelling error in the Cypher rather than trusting it.
    """
    exposed = {path for path, svc, _d in ENDPOINTS if svc == service}
    return sum(mrr for _n, mrr, _p, used in CUSTOMERS if exposed & set(used))
