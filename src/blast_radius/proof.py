"""The memory-persistence proof, run as THREE separate operating-system processes.

Track 02 asks for memory that survives a session. Almost every way of demonstrating that is
circular: a dictionary that outlives a function call proves the dictionary works, and one process
writing then reading its own cache proves the cache works. Neither says anything about persistence.

So this is arranged to be falsifiable:

    phase A   a fresh python process   wipes agent memory, asks a question   -> expect 0 hits
    phase B   a fresh python process   runs the agent once                   -> writes 1 episode
    phase C   a fresh python process   asks the SAME question                -> expect >= 1 hit

Phases A and C share no memory, no file handles, no Python objects and no interpreter. The only
thing they have in common is the database. If C finds what B wrote, the memory is in FalkorDB and
nowhere else -- and if it did not, this script fails loudly rather than printing a reassuring
summary.

`forget` is scoped to `kind='agent_conclusion'` on purpose. The seeded incident retrospectives stay
in place, so phase A proves zero AGENT memories while the historical memory the agent reads is
still present. A blanket wipe would make phase A trivially empty and the proof meaningless.

Run it with `make memory-proof`, or:

    python -m blast_radius.proof --tenant acme
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

# The question both the cold and the warm phase ask. Identical by construction -- defined once here
# so the two phases cannot drift apart and accidentally compare different queries.
QUESTION = "checkout 5xx spike shortly after a pricing-legacy deploy"


# --------------------------------------------------------------------------------------
# The phases, each invoked as its own process via `-m blast_radius.proof --phase X`
# --------------------------------------------------------------------------------------

def phase_a(tenant: str) -> dict[str, object]:
    from . import memory

    removed = memory.forget(kind="agent_conclusion", tenant=tenant)
    hits = memory.hit_count(QUESTION, kind="agent_conclusion", tenant=tenant)
    return {"phase": "A", "pid": _pid(), "forgot": removed, "hits": hits}


def phase_b(tenant: str) -> dict[str, object]:
    from . import agents

    result = agents.run_incident(tenant=tenant)
    return {
        "phase": "B",
        "pid": _pid(),
        "run_id": result.run_id,
        "cause": result.verdict.get("cause"),
        "action": result.verdict.get("action"),
        "tool_calls": sum(1 for s in result.steps if s.call),
    }


def phase_c(tenant: str) -> dict[str, object]:
    from . import memory

    result = memory.recall(QUESTION, k=10, kinds=("agent_conclusion",), tenant=tenant)
    return {
        "phase": "C",
        "pid": _pid(),
        "hits": len(result.rows),
        "top": (result.rows[0].get("summary") if result.rows else None),
        "top_similarity": (result.rows[0].get("similarity") if result.rows else None),
        "run_id": (result.rows[0].get("run_id") if result.rows else None),
    }


PHASES = {"A": phase_a, "B": phase_b, "C": phase_c}


def _pid() -> int:
    import os

    return os.getpid()


# --------------------------------------------------------------------------------------
# The driver, which spawns the three processes
# --------------------------------------------------------------------------------------

def _spawn(phase: str, tenant: str) -> dict[str, object]:
    """Run one phase in a genuinely separate interpreter and return its JSON result.

    `sys.executable -m` rather than `multiprocessing` or a thread: a forked child inherits the
    parent's memory, which is exactly the thing this proof must not do. A fresh interpreter shares
    nothing.
    """
    proc = subprocess.run(
        [sys.executable, "-m", "blast_radius.proof", "--phase", phase, "--tenant", tenant],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"phase {phase} failed (exit {proc.returncode}):\n{proc.stderr[-1500:]}"
        )
    for line in reversed(proc.stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    raise RuntimeError(f"phase {phase} produced no JSON result:\n{proc.stdout[-1500:]}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="blast_radius.proof")
    parser.add_argument("--phase", choices=sorted(PHASES), help="run ONE phase (used internally)")
    parser.add_argument("--tenant", default="acme")
    args = parser.parse_args(argv)

    # Child mode: run the single phase and print its result as JSON on the last line.
    if args.phase:
        print(json.dumps(PHASES[args.phase](args.tenant)))
        return 0

    # Driver mode.
    print(f"memory persistence proof -- tenant {args.tenant}, driver pid {_pid()}")
    print("three separate OS processes; the ONLY thing they share is the database\n")

    a = _spawn("A", args.tenant)
    print(f"  phase A  pid {a['pid']:>7}  wiped {a['forgot']} agent episode(s)  "
          f"-> {a['hits']} hit(s)   [cold]")

    b = _spawn("B", args.tenant)
    print(f"  phase B  pid {b['pid']:>7}  ran the agent: {b['cause']} / {b['action']}  "
          f"({b['tool_calls']} tool calls)")

    c = _spawn("C", args.tenant)
    print(f"  phase C  pid {c['pid']:>7}  -> {c['hits']} hit(s)   [warm]")
    if c.get("top"):
        print(f"           recalled: {c['top']}  (similarity {c['top_similarity']})")

    distinct_pids = len({a["pid"], b["pid"], c["pid"]})
    cold_zero = a["hits"] == 0
    warm_found = int(c["hits"] or 0) >= 1

    print()
    print(f"  distinct pids   {distinct_pids} of 3        {'OK' if distinct_pids == 3 else 'FAILED'}")
    print(f"  cold phase      {a['hits']} hits         {'OK' if cold_zero else 'FAILED'}")
    print(f"  warm phase      {c['hits']} hits         {'OK' if warm_found else 'FAILED'}")

    if distinct_pids == 3 and cold_zero and warm_found:
        print("\n  PROVEN: a process that shares nothing with the writer found what it wrote.")
        print("  The memory is in FalkorDB. There is nowhere else it could have come from.")
        return 0

    print("\n  NOT PROVEN -- see the failing line above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
