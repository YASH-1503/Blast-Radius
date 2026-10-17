"""`br` -- the command line, which is also how the demo is driven in a terminal.

Everything the UI can do is reachable here, because a reviewer with no browser open should still be
able to run the whole thing. `br doctor` exists specifically so a judge whose setup is broken learns
*what* is broken in one command rather than reading a traceback.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from . import agents, algorithms, config, embeddings, github_ingest, ingest, memory, queries, schema
from .config import SETTINGS
from .db import GraphClient


def _print_rows(rows: list[dict[str, Any]], limit: int = 12) -> None:
    if not rows:
        print("  (no rows)")
        return
    for row in rows[:limit]:
        for key, value in row.items():
            if isinstance(value, list) and len(value) > 6:
                value = f"[{len(value)} items] {value[:4]} ..."
            print(f"  {key:24s} {value}")
        print("  " + "-" * 66)


def cmd_ingest(args: argparse.Namespace) -> int:
    tenants = tuple(t.strip() for t in args.tenant.split(",") if t.strip())
    repos = {}
    if args.repo:
        repos = {t: args.repo for t in tenants}
    reports = ingest.ingest_all(
        tenants,
        use_github=not args.no_github,
        offline=args.offline,
        repos=repos or None,
    )
    for tenant, report in reports.items():
        print(f"\n=== {tenant} ===")
        print(report.explain())
        if report.silent_steps:
            print(f"\n  STEPS THAT WROTE NOTHING: {', '.join(report.silent_steps)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    failed = 0
    for tenant in (t.strip() for t in args.tenant.split(",") if t.strip()):
        report = ingest.verify(tenant)
        print(f"\n=== {tenant} ===")
        print(report.explain())
        failed += sum(1 for _n, ok, _d in report.checks if not ok)
    return 1 if failed else 0


def cmd_ask(args: argparse.Namespace) -> int:
    keys = [args.question] if args.question != "all" else list(queries.QUESTIONS)
    for key in keys:
        spec = queries.QUESTIONS[key]
        kwargs: dict[str, Any] = {"tenant": args.tenant}
        if args.service:
            kwargs["service"] = args.service
        result = queries.ask(key, **kwargs)
        print(f"\n=== {key}: {spec['ask']}")
        print(f"    why a graph: {spec['why']}")
        print(f"    {result.elapsed_ms:.2f} ms, {len(result.rows)} row(s), "
              f"{len(result.nodes)} nodes walked")
        _print_rows(result.rows)
        if args.cypher:
            print("    cypher:")
            for line in result.cypher.strip().splitlines():
                print(f"      {line}")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    result = agents.run_incident(tenant=args.tenant)
    print(result.explain())
    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    return 0


def cmd_algo(args: argparse.Namespace) -> int:
    names = [args.name] if args.name != "all" else list(algorithms.ALGORITHMS)
    for name in names:
        result = algorithms.run(name, tenant=args.tenant)
        print(f"\n=== {name}: {algorithms.ALGORITHMS[name]['what']}  ({result.elapsed_ms:.2f} ms)")
        _print_rows(result.rows, limit=8)
    return 0


def cmd_memory(args: argparse.Namespace) -> int:
    if args.query:
        result = memory.recall(args.query, k=args.k, tenant=args.tenant)
        print(f"recall '{args.query}'  ({result.elapsed_ms:.2f} ms, "
              f"backend={embeddings.backend()})")
        _print_rows(result.rows, limit=args.k)
    else:
        result = memory.timeline(limit=args.k, tenant=args.tenant)
        print(f"timeline ({result.elapsed_ms:.2f} ms)")
        _print_rows(result.rows, limit=args.k)
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Everything a broken setup needs reported, in one place, without raising."""
    ok = True
    print("connection")
    try:
        conn = SETTINGS.connection
        # describe() is used rather than f-stringing the fields because it is guaranteed never to
        # include the password -- `br doctor` output is exactly the thing a user pastes into a chat.
        print(f"  url            {conn.describe()}")
        print(f"  tls            {conn.tls}  (cloud={conn.is_cloud})")
        print(f"  password       {'set' if conn.password else 'not set'}")
        # The `.env` line matters more than it looks: the symptom of a file that is present but
        # NOT loaded is a demo that silently runs against 127.0.0.1 while the user believes it is
        # on Cloud. Printing which keys the loader applied makes that visible in one command.
        dotenv = config.dotenv_status()
        state = "loaded" if dotenv["present"] else "absent (using defaults / real env)"
        print(f"  .env           {state}  {dotenv['path']}")
        if dotenv["present"]:
            applied = dotenv["keys_applied"]
            print(f"  .env applied   {', '.join(applied) if applied else '(none -- real env wins)'}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAILED to resolve connection: {exc}")
        print("  check FALKOR_URL -- expected falkor://, falkors://, redis:// or rediss://")
        return 1

    print("\ndatabase")
    try:
        gc = GraphClient(args.tenant)
        print(f"  ping           {gc.ping()}")
        print(f"  graph          {gc.graph_name}")
        print(f"  graphs present {gc.list_graphs()}")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAILED: {exc}")
        print("  start the database:  docker compose up -d   (or `make db`)")
        return 1

    print("\nschema")
    try:
        described = schema.describe(tenant=args.tenant)
        node_total = sum(int(r["nodes"]) for r in described.get("nodes", []))
        edge_total = sum(int(r["edges"]) for r in described.get("edges", []))
        print(f"  labels         {len(described.get('labels', []))} declared, "
              f"{len(described.get('nodes', []))} populated")
        print(f"  nodes          {node_total}")
        print(f"  edges          {edge_total}")
        if node_total == 0:
            ok = False
            print("  EMPTY          run `make ingest` -- the schema exists but holds no data")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  FAILED: {exc}")

    print("\nembeddings")
    described = embeddings.describe()
    print(f"  backend        {described.get('backend')}")
    print(f"  semantic       {described.get('semantic')}")
    if not described.get("semantic"):
        print("  note           hashed fallback in use; `pip install fastembed` for real embeddings")

    print("\ngithub layer")
    env = github_ingest.env_summary()
    print(f"  repo           {env['repo']}")
    print(f"  token          {env['token']}")
    print(f"  cache          {env['cache']['files']} file(s), "
          f"{env['cache']['bytes'] // 1024} KB, offline_ready={env['cache']['offline_ready']}")
    print(f"  stores email   {env['stores_email']}")
    if not env["cache"]["offline_ready"]:
        print("  note           no cache yet; first `br ingest` needs network (13 requests)")

    print("\napi auth")
    keys = SETTINGS.api_keys()
    print(f"  auth required  {SETTINGS.auth_required}")
    print(f"  keys from env  {len(keys)}")
    if not keys and SETTINGS.auth_required:
        print("  note           none set; the server will generate and print development keys")
    if not SETTINGS.auth_required:
        ok = False
        print("  WARNING        BR_AUTH_REQUIRED=false -- the API is OPEN to any caller")

    print("\n" + ("all good" if ok else "problems found above"))
    return 0 if ok else 1


def cmd_provenance(args: argparse.Namespace) -> int:
    result = queries.provenance(tenant=args.tenant)
    print(f"provenance ({result.elapsed_ms:.2f} ms) -- Rule 6 as a query, not a paragraph\n")
    print(f"  {'label':14s} {'source':12s} {'nodes':>7s}")
    totals: dict[str, int] = {}
    for row in result.rows:
        print(f"  {row['label']:14s} {row['source']:12s} {row['nodes']:>7}")
        totals[row["source"]] = totals.get(row["source"], 0) + int(row["nodes"])
    print()
    for source, count in sorted(totals.items()):
        print(f"  TOTAL {source:12s} {count}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="br", description="Blast Radius")
    parser.add_argument("--tenant", default=SETTINGS.tenant)
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help_text: str) -> argparse.ArgumentParser:
        """Register a subcommand that ALSO accepts --tenant.

        argparse binds a global option only BEFORE the subcommand, so `br verify --tenant acme`
        fails with "unrecognized arguments" while `br --tenant acme verify` works. The second form
        is not the one anyone types, and the error names the flag rather than the ordering -- so it
        reads like the flag does not exist. Accepting it in both positions costs one line each.
        """
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--tenant", default=None, help="override the tenant for this command")
        return p

    p = add("ingest", "build the graph")
    p.add_argument("--repo", help="override the GitHub repo for the real human layer")
    p.add_argument("--no-github", action="store_true", help="synthetic only")
    p.add_argument("--offline", action="store_true", help="cache only, never touch the network")
    p.set_defaults(func=cmd_ingest)

    p = add("verify", "assert the graph can answer the demo")
    p.set_defaults(func=cmd_verify)

    p = add("ask", "run a demo question")
    p.add_argument("question", nargs="?", default="all", choices=[*queries.QUESTIONS, "all"])
    p.add_argument("--service")
    p.add_argument("--cypher", action="store_true", help="print the Cypher for each answer")
    p.set_defaults(func=cmd_ask)

    p = add("run", "run the three-agent incident loop")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_run)

    p = add("algo", "run a graph algorithm")
    p.add_argument("name", nargs="?", default="all", choices=[*algorithms.ALGORITHMS, "all"])
    p.set_defaults(func=cmd_algo)

    p = add("memory", "recall or list episodes")
    p.add_argument("query", nargs="?")
    p.add_argument("-k", type=int, default=5)
    p.set_defaults(func=cmd_memory)

    p = add("provenance", "what is real and what is synthetic")
    p.set_defaults(func=cmd_provenance)

    p = add("doctor", "diagnose the setup")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    # The subcommand's --tenant (if given) wins; otherwise the global one, parsed first.
    global_args, _ = build_parser().parse_known_args(argv)
    args = build_parser().parse_args(argv)
    if getattr(args, "tenant", None) is None:
        args.tenant = global_args.tenant or SETTINGS.tenant
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        print("try `br doctor` to check the setup", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
