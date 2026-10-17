"""Blast Radius -- an incident-response agent whose context IS a graph.

An alert names a symptom. The answer -- what broke, whose change caused it, how much revenue is
exposed, who can actually fix it, and what fixed it last time -- is not written down anywhere. It
only exists in the *relationships* between services, deploys, commits, people, customers and past
incidents. So the context layer is a graph, and every answer is a traversal the system can show.

Package layout, in the order it is worth reading:

    config        the two decisions that shape everything: tenant-as-graph-name, and one
                  connection path for local Docker and FalkorDB Cloud alike
    db            the provenance envelope -- no query returns bare rows
    schema        the graph model, its indices, and the vector index
    data          the synthetic layer, and why these parts cannot be real
    github_ingest the real layer: public GitHub people, commits and pull requests
    embeddings    real model when installed, deterministic fallback when not
    ingest        joins both halves, then `verify()` proves the join actually matched
"""

from __future__ import annotations

__version__ = "1.0.0"
__all__ = [
    "config",
    "data",
    "db",
    "embeddings",
    "github_ingest",
    "ingest",
    "schema",
]
