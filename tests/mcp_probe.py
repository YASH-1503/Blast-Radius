"""Speak real MCP, over a real stdio transport, to the server in `blast_radius.mcp_server`.

This is a PROBE rather than a pytest test on purpose: it spawns the server as a separate OS
process and talks the wire protocol to it, which is the only way to prove the transport works
rather than proving that `build_server()` returns an object. `tests/test_regressions.py` keeps the
cheap structural guard (the MCP surface cannot drift from `tools.REGISTRY`); this proves the
protocol end of it, and is what `make mcp-verify` runs.

Run:  .venv/bin/python tests/mcp_probe.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

REPO = Path(__file__).resolve().parent.parent
TENANT = os.environ.get("TENANT", "acme")

# Two tools chosen because they exercise DIFFERENT shapes: one takes a required argument and walks
# the dependency graph, the other takes none and reads provenance counts.
PROBE_CALLS: list[tuple[str, dict[str, object]]] = [
    ("blast_radius", {"service": "pricing-legacy"}),
    ("provenance", {}),
]


def _text_of(result: object) -> str:
    """Pull the text payload out of a CallToolResult across SDK shapes."""
    content = getattr(result, "content", None) or []
    parts = [getattr(block, "text", "") for block in content]
    return "\n".join(p for p in parts if p)


async def main() -> int:
    params = StdioServerParameters(
        command=str(REPO / ".venv" / "bin" / "python"),
        args=["-m", "blast_radius.mcp_server"],
        env={**os.environ, "TENANT": TENANT, "PYTHONUNBUFFERED": "1"},
        cwd=str(REPO),
    )

    failures: list[str] = []

    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            server_name = getattr(init.serverInfo, "name", "?")
            print(f"initialize      ok   server={server_name!r} proto={init.protocolVersion}")

            listed = await session.list_tools()
            names = sorted(t.name for t in listed.tools)
            print(f"tools/list      ok   {len(names)} tools")
            for t in sorted(listed.tools, key=lambda t: t.name):
                required = (t.inputSchema or {}).get("required") or []
                print(f"                     - {t.name}({', '.join(required)})")
                if not (t.description or "").strip():
                    failures.append(f"{t.name} has no description for a calling agent")

            # The registry is the single source of truth; the transport must expose all of it.
            sys.path.insert(0, str(REPO / "src"))
            from blast_radius import tools as tools_mod

            missing = sorted(set(tools_mod.REGISTRY) - set(names))
            extra = sorted(set(names) - set(tools_mod.REGISTRY))
            if missing:
                failures.append(f"registry tools absent over MCP: {missing}")
            if extra:
                failures.append(f"MCP exposes tools not in the registry: {extra}")
            print(f"registry parity ok   {len(tools_mod.REGISTRY)} registered, {len(names)} exposed")

            for name, args in PROBE_CALLS:
                result = await session.call_tool(name, args)
                raw = _text_of(result)
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    failures.append(f"{name} returned non-JSON over MCP: {raw[:120]!r}")
                    continue
                if "error" in payload:
                    failures.append(f"{name} errored: {payload['error']}")
                    continue
                # The whole point of this project: the answer AND the Cypher behind it.
                cypher = payload.get("cypher") or payload.get("evidence", {}).get("cypher")
                if not cypher:
                    failures.append(f"{name} returned no Cypher evidence over MCP")
                keys = ", ".join(sorted(payload)[:6])
                print(
                    f"tools/call      ok   {name}({', '.join(f'{k}={v}' for k, v in args.items())})"
                    f"  keys=[{keys}]  cypher={'yes' if cypher else 'NO'}"
                )

            # The error path is a DESIGN claim in mcp_server.py -- a bad argument comes back as
            # correctable text rather than a transport error that just ends the calling agent's
            # turn. Nothing verified it until now.
            bad = await session.call_tool("blast_radius", {"service": "no-such-service"})
            bad_raw = _text_of(bad)
            try:
                bad_payload = json.loads(bad_raw)
            except json.JSONDecodeError:
                failures.append(f"bad argument returned non-JSON: {bad_raw[:120]!r}")
            else:
                if "error" not in bad_payload:
                    failures.append("a bad service name did not produce an error payload")
                elif "Known:" not in bad_payload["error"]:
                    failures.append(
                        "error payload does not tell the agent what the valid services are: "
                        f"{bad_payload['error']!r}"
                    )
                else:
                    print(f"error path      ok   correctable: {bad_payload['error'][:68]}...")

            res = await session.list_resources()
            uris = [str(r.uri) for r in res.resources]
            print(f"resources/list  ok   {uris}")
            if "graph://provenance" not in uris:
                failures.append(f"graph://provenance not advertised, got {uris}")
            else:
                read_back = await session.read_resource("graph://provenance")  # type: ignore[arg-type]
                body = "\n".join(
                    getattr(c, "text", "") for c in getattr(read_back, "contents", [])
                )
                print(f"resources/read  ok   graph://provenance -> {len(body)} bytes")
                if not body.strip():
                    failures.append("graph://provenance read back empty")

    print()
    if failures:
        print(f"FAIL  {len(failures)} problem(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("PASS  MCP transport verified over stdio")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
