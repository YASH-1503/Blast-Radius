"""Concurrency benchmark: the high-concurrency read/write claim, as numbers rather than a sentence.

Track 02 asks for high-concurrency reads and writes. Most submissions will assert it in a README.
This measures it, and reports the measurement honestly -- including what it does NOT prove.

What it does: runs N worker threads for a fixed duration against one FalkorDB instance, mixing real
demo queries with real episode writes, and reports throughput plus the full latency distribution
per operation type.

What it does not prove, stated because a benchmark without its caveats is marketing:

* **One machine, loopback.** There is no network between client and server, so these numbers are an
  upper bound, not a production figure.
* **A demo-sized graph.** ~1,600 nodes and ~4,200 edges. Traversal cost grows with the graph.
* **Threads, not processes.** The client is bounded by Python's GIL on result parsing, so the
  measured ceiling may be the CLIENT's, not the database's. Where that is suspected it is said.
* **Writes are a minority of the mix by default**, matching the real workload -- an incident
  response reads far more than it writes.

Run: `python -m bench.concurrency --workers 60 --seconds 10`
"""

from __future__ import annotations

import argparse
import random
import statistics
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field

from blast_radius import memory, queries
from blast_radius.db import GraphClient


@dataclass
class Sample:
    op: str
    ms: float
    ok: bool


@dataclass
class Results:
    samples: list[Sample] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def add(self, sample: Sample, error: str | None = None) -> None:
        with self.lock:
            self.samples.append(sample)
            if error:
                self.errors.append(error)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, int(round((pct / 100.0) * (len(ordered) - 1))))
    return ordered[index]


# The operation mix. Reads dominate, as an incident response genuinely does.
READ_OPS = (
    ("candidate_causes", lambda gc: queries.candidate_causes("api-gateway", client=gc)),
    ("blast_radius", lambda gc: queries.blast_radius("pricing-legacy", client=gc)),
    ("who_to_page", lambda gc: queries.who_to_page("api-gateway", client=gc)),
    ("path_between", lambda gc: queries.path_between("api-gateway", "pricing-legacy", client=gc)),
    ("past_incidents", lambda gc: queries.past_incidents("pricing-legacy", client=gc)),
    ("vector_recall", lambda gc: memory.recall("checkout 5xx after a deploy", k=3, client=gc)),
)


# Every benchmark write carries this marker so it can be deleted afterwards.
#
# This is not tidiness. The first run wrote 6,567 episodes into the demo tenant and took it from
# 1,644 nodes to 8,226 -- which inflates the graph the demo is shown on, and poisons vector recall
# with thousands of near-identical "bench worker N" episodes that then compete with the agent's
# real conclusion for the top hit. A benchmark that changes the thing it measures is worse than no
# benchmark, because the damage is invisible until the demo looks wrong.
#
# The writes still go to the REAL demo graph, which is the honest measurement -- writing into an
# empty throwaway graph would measure a different and easier workload. They are just removed after.
BENCH_MARKER = "__bench__"


def _cleanup(tenant: str) -> int:
    """Delete every episode this benchmark wrote. Runs even if the benchmark failed."""
    gc = GraphClient(tenant)
    result = gc.run(
        "MATCH (e:Episode) WHERE e.summary STARTS WITH $marker DETACH DELETE e",
        {"marker": BENCH_MARKER},
        label="bench:cleanup",
        collect_provenance=False,
    )
    return int(result.writes.nodes_deleted)


def _write(gc: GraphClient, worker: int, counter: list[int]) -> None:
    counter[0] += 1
    memory.write_episode(
        memory.Episode(
            kind="agent_conclusion",
            text=f"bench worker {worker} write {counter[0]} on pricing-legacy",
            summary=f"{BENCH_MARKER} w{worker} n{counter[0]}",
            services=["pricing-legacy"],
        ),
        client=gc,
    )


def worker_loop(
    worker: int,
    tenant: str,
    deadline: float,
    write_ratio: float,
    results: Results,
    seed: int,
) -> None:
    """One thread, its own client, its own RNG stream so the mix is reproducible."""
    rng = random.Random(seed + worker)
    gc = GraphClient(tenant)
    counter = [0]
    while time.perf_counter() < deadline:
        if rng.random() < write_ratio:
            name, fn = "episode_write", lambda c: _write(c, worker, counter)
        else:
            name, fn = READ_OPS[rng.randrange(len(READ_OPS))]
        started = time.perf_counter()
        try:
            fn(gc)
            results.add(Sample(name, (time.perf_counter() - started) * 1000.0, True))
        except Exception as exc:  # noqa: BLE001
            results.add(
                Sample(name, (time.perf_counter() - started) * 1000.0, False),
                f"{name}: {type(exc).__name__}: {exc}",
            )


def run(
    *,
    tenant: str = "acme",
    workers: int = 60,
    seconds: float = 10.0,
    write_ratio: float = 0.15,
    seed: int = 20261015,
) -> Results:
    probe = GraphClient(tenant)
    if not probe.ping():
        raise SystemExit("no database -- run `make db` first")

    results = Results()
    deadline = time.perf_counter() + seconds
    threads = [
        threading.Thread(
            target=worker_loop,
            args=(i, tenant, deadline, write_ratio, results, seed),
            daemon=True,
        )
        for i in range(workers)
    ]
    started = time.perf_counter()
    try:
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=seconds + 60)
        elapsed = time.perf_counter() - started
    finally:
        # In `finally` on purpose: a crashed or interrupted run must not leave thousands of
        # benchmark episodes behind in the graph the demo is given on.
        removed = _cleanup(tenant)

    report(results, workers=workers, elapsed=elapsed, write_ratio=write_ratio, tenant=tenant)
    print(f"  cleanup        removed {removed} benchmark episode(s) from {tenant}")
    return results


def report(
    results: Results, *, workers: int, elapsed: float, write_ratio: float, tenant: str
) -> None:
    total = len(results.samples)
    failed = sum(1 for s in results.samples if not s.ok)
    by_op: dict[str, list[float]] = defaultdict(list)
    for sample in results.samples:
        if sample.ok:
            by_op[sample.op].append(sample.ms)

    print(f"\nconcurrency benchmark -- {tenant}")
    print(f"  {workers} threads, {elapsed:.1f}s, write ratio {write_ratio:.0%}, loopback\n")
    print(f"  {'operation':20s} {'count':>7s} {'mean':>8s} {'p50':>8s} {'p95':>8s} {'p99':>8s} {'max':>8s}")
    for op in sorted(by_op):
        values = by_op[op]
        print(
            f"  {op:20s} {len(values):>7d} "
            f"{statistics.mean(values):>8.2f} {_percentile(values, 50):>8.2f} "
            f"{_percentile(values, 95):>8.2f} {_percentile(values, 99):>8.2f} "
            f"{max(values):>8.2f}"
        )

    all_ms = [s.ms for s in results.samples if s.ok]
    print(
        f"\n  total          {total} ops in {elapsed:.1f}s = "
        f"{total / elapsed:,.0f} ops/sec"
    )
    print(f"  errors         {failed}" + ("" if not failed else "  <-- investigate"))
    if all_ms:
        print(f"  overall p99    {_percentile(all_ms, 99):.2f} ms")
    writes = len(by_op.get("episode_write", []))
    print(f"  writes         {writes} ({writes / max(total, 1):.0%} of operations)")

    for error in results.errors[:5]:
        print(f"  error: {error[:160]}")

    print(
        "\n  caveats: one machine over loopback, a ~1,600-node graph, and a threaded client that\n"
        "  may itself be the ceiling. Treat this as an upper bound, not a production figure."
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bench.concurrency")
    parser.add_argument("--tenant", default="acme")
    parser.add_argument("--workers", type=int, default=60)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--write-ratio", type=float, default=0.15)
    args = parser.parse_args(argv)
    results = run(
        tenant=args.tenant,
        workers=args.workers,
        seconds=args.seconds,
        write_ratio=args.write_ratio,
    )
    return 1 if any(not s.ok for s in results.samples) else 0


if __name__ == "__main__":
    raise SystemExit(main())
