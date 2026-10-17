"""MCP server: the same tools the agent uses, exposed to any MCP-speaking client.

This is the "context for AI agents" part taken literally. The graph is not just backing one agent
in this repository -- it is a context source any agent can attach to. Claude Desktop, an IDE
assistant, or another orchestrator gets the identical thirteen tools, with the identical validation
and the identical evidence envelope.

**It enumerates `tools.REGISTRY` rather than redeclaring anything.** A second hand-written list
would be a second thing to keep in sync, and the first time they diverged the MCP surface would
quietly differ from what the agent can actually do. Adding a tool in `tools.py` adds it here.

`fastmcp` is an optional dependency. Importing this module without it gives a clear message rather
than a traceback, because the HTTP API and the CLI do not need it and a reviewer should not have to
install it to run the demo.
"""

from __future__ import annotations

import inspect
import json
from typing import Annotated, Any

from pydantic import Field

from . import tools
from .config import SETTINGS

try:
    from fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional dependency
    FastMCP = None  # type: ignore[assignment]


BANNER = """\
blast-radius MCP server

Exposes a service dependency graph with real GitHub commit history as agent context.
Every tool returns the Cypher that produced its answer, so a calling agent can show its work.
"""


def _result_payload(call: tools.ToolCall) -> str:
    """What an MCP client receives: the answer AND its provenance.

    The Cypher is included deliberately. A calling agent that can quote the query behind a claim
    can be audited by whoever reads its output; one that only has the answer cannot.
    """
    payload = call.result.to_dict()
    return json.dumps(
        {
            "tool": call.tool,
            "args": call.args,
            "rows": payload["rows"],
            "row_count": payload["row_count"],
            "evidence": {
                "cypher": payload["cypher"],
                "params": {k: v for k, v in payload["params"].items() if k != "vec"},
                "hops_allowed": payload["hops"],
                "elapsed_ms": payload["elapsed_ms"],
                "graph": payload["graph"],
                "nodes_walked": len(payload["provenance"]["nodes"]),
                "highlight": payload["provenance"]["focus"],
            },
        },
        indent=1,
        default=str,
    )



# Annotations in `tools.py` are strings (that module uses `from __future__ import annotations`),
# so they are mapped back to real types here. The map is a deliberate whitelist: an unrecognised
# annotation RAISES instead of degrading to `Any`, because a tool whose argument types are `Any`
# is advertised to every MCP client as taking anything, and the first wrong call fails inside a
# Cypher parameter rather than at the boundary.
_ANNOTATIONS: dict[str, type] = {"str": str, "int": int, "float": float, "bool": bool}


def _handler_for(tool_name: str, spec: tools.ToolSpec, tenant: str) -> Any:
    """Build an MCP handler whose SIGNATURE mirrors the registry function's.

    This is the one place the MCP surface could not simply be derived. An MCP client discovers a
    tool's arguments from its JSON schema, and fastmcp builds that schema by introspecting the
    handler's signature -- so a `def handler(**kwargs)` handler describes a tool that accepts
    NOTHING, and no calling agent can work out what to pass it. fastmcp refuses `**kwargs`
    outright, which is the kind version of that failure.

    The signature is GENERATED from `spec.fn` rather than hand-written per tool, so the guarantee
    in this module's docstring still holds: adding a tool to `tools.py` adds it here, correctly
    typed, with nothing to keep in sync.
    """
    signature = inspect.signature(spec.fn)
    declarations: list[str] = []
    passed: list[str] = []
    # The generated function's globals are a COPY of this module's, not a bare dict. pydantic
    # re-resolves a function's annotations against the module it claims to live in, so a handler
    # built in an isolated namespace fails with `NameError: name 'Field' is not defined` at schema
    # time -- long after the code that looks wrong has already run.
    namespace: dict[str, Any] = {**globals(), "_dispatch": None}

    for param in signature.parameters.values():
        # `**kw` on the registry functions exists to swallow the injected `client`; it is not an
        # argument a caller supplies, so it must not reach the schema.
        if param.kind is inspect.Parameter.VAR_KEYWORD:
            continue
        annotation = param.annotation
        type_name = annotation if isinstance(annotation, str) else getattr(
            annotation, "__name__", str(annotation)
        )
        resolved_type = _ANNOTATIONS.get(type_name)
        if resolved_type is None:
            raise RuntimeError(
                f"{tool_name}: cannot expose argument {param.name!r} over MCP -- annotation "
                f"{type_name!r} is not in the whitelist {sorted(_ANNOTATIONS)}. Add it there "
                f"deliberately rather than letting the tool advertise an untyped argument."
            )
        namespace[type_name] = resolved_type
        # The registry already documents every argument; putting that text on the SCHEMA property
        # rather than only in the prose description is what lets a client show per-argument help
        # while choosing a tool, instead of making the agent parse the docstring.
        description = spec.args.get(param.name)
        if description:
            declaration = (
                f"{param.name}: Annotated[{type_name}, Field(description={description!r})]"
            )
        else:
            declaration = f"{param.name}: {type_name}"
        if param.default is not inspect.Parameter.empty:
            declaration += f" = {param.default!r}"
        declarations.append(declaration)
        passed.append(f"{param.name!r}: {param.name}")

    def dispatch(supplied: dict[str, Any]) -> str:
        try:
            call = tools.call(tool_name, supplied, tenant=tenant)
        except tools.ToolError as exc:
            # Returned as text, not raised: a calling agent can correct a bad service name if it
            # is told what was wrong, where a transport error just ends its turn.
            return json.dumps({"error": str(exc), "tool": tool_name, "args": supplied})
        return _result_payload(call)

    namespace["_dispatch"] = dispatch
    source = (
        f"def {tool_name}({', '.join(declarations)}) -> str:\n"
        f"    return _dispatch({{{', '.join(passed)}}})\n"
    )
    # `dont_inherit=True` matters and is not tidiness. This module uses `from __future__ import
    # annotations`, and `compile()` INHERITS the caller's future flags by default -- so without it
    # the generated handler's annotations are stored as STRINGS, pydantic re-evaluates them in a
    # namespace that has no `Field`, and schema generation dies with `NameError: name 'Field' is
    # not defined` pointing at a line of source that never mentions a missing name.
    exec(  # noqa: S102 - source is built from this repo's own registry, never from user input
        compile(source, f"<mcp tool {tool_name}>", "exec", dont_inherit=True),
        namespace,
    )
    handler: Any = namespace[tool_name]

    handler.__doc__ = (
        f"{spec.summary}\n\n"
        f"Why this needs a graph: {spec.graph_rationale}\n\n"
        f"Arguments: {', '.join(f'{k} -- {v}' for k, v in spec.args.items()) or 'none'}\n"
        f"Required: {', '.join(spec.required) or 'none'}"
    )
    return handler


def build_server(tenant: str | None = None) -> Any:
    """Construct the MCP server from the shared registry."""
    if FastMCP is None:
        raise RuntimeError(
            "fastmcp is not installed. `pip install fastmcp` to expose these tools over MCP; "
            "the HTTP API (`make api`) and the CLI (`br`) work without it."
        )

    server = FastMCP("blast-radius", instructions=BANNER)
    resolved = tenant or SETTINGS.tenant

    for name, spec in sorted(tools.REGISTRY.items()):
        server.tool()(_handler_for(name, spec, resolved))

    @server.resource("graph://provenance")
    def provenance() -> str:
        """Which layers of this graph are real and which are synthetic, counted live."""
        return _result_payload(tools.call("provenance", {}, tenant=resolved))

    return server


def main() -> int:
    if FastMCP is None:
        print(
            "fastmcp is not installed.\n"
            "  pip install fastmcp\n"
            "The HTTP API and CLI do not need it:\n"
            "  make api     # http://127.0.0.1:8099\n"
            "  br ask all   # the six questions in a terminal"
        )
        return 1
    build_server().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
