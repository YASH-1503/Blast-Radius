"""Configuration, and the two decisions that shape the whole system.

**1. The tenant is the graph NAME, not a property.**
FalkorDB holds many isolated graphs in one instance, so `blastradius_acme` and `blastradius_globex`
are physically separate. Tenant isolation therefore costs nothing and needs no `WHERE tenant = $t`
clause anywhere in the codebase. There is no query to forget to filter, which is the only kind of
multi-tenancy that cannot leak by omission.

**2. Local Docker and FalkorDB Cloud differ only in a connection string.**
Cloud is password-authenticated, and TLS is a per-tier feature rather than a given: it starts at
the Startup tier, and the Free tier has none -- so a free Cloud instance is a PLAIN scheme with a
password, while a paid one is `falkors://`. A config that cannot express both is a config that
works right up until the demo is deployed, so `FALKOR_URL` is parsed here, `falkors://` turns TLS
on, and `db.py` has one connection path for every combination. This was the single largest
unproven risk carried out of pre-event exploration, and the tier split is why all four schemes are
accepted instead of just the TLS pair.
"""

from __future__ import annotations

import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

# Repo root: this file is <root>/src/blast_radius/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DOTENV_PATH = PROJECT_ROOT / ".env"


def load_dotenv(path: Path | None = None) -> dict[str, str]:
    """Read `.env` into the environment, WITHOUT overriding anything already set.

    This exists because `.env.example` opens with "Copy to `.env` and edit", and for a while
    nothing in the codebase read that file. The failure mode was the worst kind available here:
    a judge pastes a FalkorDB Cloud URL into `.env`, the loader that was supposed to pick it up
    does not exist, `FALKOR_URL` stays empty, the config falls back to its 127.0.0.1 default --
    and because local Docker is usually running, everything *appears* to work. The Cloud
    credential is silently ignored and the demo reports success against the wrong database.
    No error, no empty result, just a quiet substitution. Same family as the six other silent
    no-ops this project has hit; see README "what broke".

    A real environment variable WINS over the file, because `FALKOR_URL=... make verify` must
    be able to override a stale `.env` for a one-off run. Returns what it actually applied, so
    `br doctor` can show it.
    """
    target = DOTENV_PATH if path is None else path
    applied: dict[str, str] = {}
    try:
        raw = target.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return applied

    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        if not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        # Already-set real environment wins; an empty-string env var counts as unset so that
        # `FALKOR_URL= make ...` does not shadow the file with nothing.
        if os.environ.get(key):
            continue
        os.environ[key] = value
        applied[key] = value
    return applied


# Load before any default_factory below is evaluated.
_DOTENV_APPLIED = load_dotenv()


def dotenv_status() -> dict[str, object]:
    """What the loader found, for diagnostics. Names only -- never values, which are secrets."""
    return {
        "path": str(DOTENV_PATH),
        "present": DOTENV_PATH.exists(),
        "keys_applied": sorted(_DOTENV_APPLIED),
    }


def _env(key: str, default: str) -> str:
    value = os.environ.get(key)
    return default if value is None or value == "" else value


def _env_bool(key: str, default: bool = False) -> bool:
    raw = os.environ.get(key)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Connection:
    """Where the database is, and how to speak to it."""

    host: str
    port: int
    username: str | None
    password: str | None
    tls: bool

    @property
    def is_cloud(self) -> bool:
        """Whether this endpoint is FalkorDB Cloud, judged by HOST rather than by TLS.

        This was `return self.tls` and it was wrong in a way only a live instance could reveal:
        a real Cloud free-tier endpoint reported `cloud=False` in `br doctor`, because the free
        tier has no TLS. Equating "cloud" with "encrypted" silently mislabels every free
        instance, and `br doctor` is exactly where someone looks to confirm what they connected
        to. Host matching is not bulletproof either -- a custom CNAME would defeat it -- so this
        is advisory and reported as such, never used to make a security decision.
        """
        host = (self.host or "").lower()
        return host.endswith(".cloud") or "falkordb.io" in host or "falkordb.cloud" in host

    def describe(self) -> str:
        """Safe to log and safe to put on screen: never includes the password."""
        scheme = "falkors" if self.tls else "falkor"
        who = f"{self.username}@" if self.username else ""
        return f"{scheme}://{who}{self.host}:{self.port}"


def _connection_from_env() -> Connection:
    """`FALKOR_URL` wins; otherwise the individual vars, which default to local Docker.

    Accepted URL forms -- all four appear in FalkorDB Cloud's own console at different times:

        falkor://host:6379                  plain
        falkors://host:6379                 TLS
        redis://user:pass@host:6379         plain, credentials inline
        rediss://user:pass@host:6379        TLS, credentials inline
    """
    url = os.environ.get("FALKOR_URL", "").strip()
    if url:
        parsed = urlparse(url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in {"falkor", "falkors", "redis", "rediss"}:
            raise ValueError(
                f"FALKOR_URL has unsupported scheme {scheme!r}; "
                "expected falkor://, falkors://, redis:// or rediss://"
            )
        # A password may legitimately contain '@' or ':' -- urlparse already split on the LAST '@'
        # of the netloc, but the values arrive percent-encoded, so decode them.
        return Connection(
            host=parsed.hostname or "127.0.0.1",
            port=parsed.port or 6379,
            username=unquote(parsed.username) if parsed.username else None,
            password=unquote(parsed.password) if parsed.password else None,
            tls=scheme in {"falkors", "rediss"},
        )

    return Connection(
        host=_env("FALKOR_HOST", "127.0.0.1"),
        port=int(_env("FALKOR_PORT", "6379")),
        username=os.environ.get("FALKOR_USER") or None,
        password=os.environ.get("FALKOR_PASSWORD") or None,
        tls=_env_bool("FALKOR_TLS", False),
    )


@dataclass(frozen=True)
class Settings:
    # NOTE: the connection is resolved LAZILY, on first access, and deliberately not in a
    # `default_factory`. A malformed FALKOR_URL must fail loudly -- connecting to the wrong
    # database silently is far worse -- but if it failed during `import blast_radius.config`
    # the whole package becomes unimportable and the error surfaces as an ImportError
    # traceback from inside uvicorn's reloader instead of as a readable message. Failing on
    # first *use* keeps the diagnostics (`health()`, `br doctor`) importable and able to
    # report the problem properly.

    # Default tenant for CLI use. The HTTP API resolves it per-request from the API key.
    tenant: str = field(default_factory=lambda: _env("TENANT", "acme"))

    # Deterministic synthetic data: same seed -> same graph -> a demo that can be rehearsed.
    seed: int = field(default_factory=lambda: int(_env("SEED", "20261015")))

    embedding_dim: int = field(default_factory=lambda: int(_env("EMBEDDING_DIM", "384")))

    # Which GitHub repository supplies the real human layer.
    github_repo: str = field(default_factory=lambda: _env("GITHUB_REPO", "FalkorDB/FalkorDB"))
    github_token: str | None = field(default_factory=lambda: os.environ.get("GITHUB_TOKEN") or None)

    # ---- API authentication -----------------------------------------------------------
    # The pre-event spike served the tenant from a query parameter with no auth at all, so any
    # caller could read any tenant's graph. That is fine on loopback and indefensible anywhere
    # else, so the tenant is now derived from a key the caller presents and never from the URL.
    api_keys_raw: str = field(default_factory=lambda: _env("BR_API_KEYS", ""))
    auth_required: bool = field(default_factory=lambda: _env_bool("BR_AUTH_REQUIRED", True))

    @property
    def connection(self) -> Connection:
        return _connection_from_env()

    def graph_name(self, tenant: str | None = None) -> str:
        """One graph per tenant: `blastradius_acme`, `blastradius_globex`, ..."""
        name = (tenant or self.tenant).strip().lower()
        if not name or not all(ch.isalnum() or ch == "_" for ch in name):
            raise ValueError(f"invalid tenant name {tenant!r}: use [a-z0-9_] only")
        return f"blastradius_{name}"

    def api_keys(self) -> dict[str, str]:
        """`key:tenant,key:tenant` -> {key: tenant}.

        Generated on first use when unset, so a developer who starts the API with no environment
        still gets an authenticated server rather than an open one. The key is printed once at
        startup; it is never written to disk.
        """
        mapping: dict[str, str] = {}
        for pair in self.api_keys_raw.split(","):
            pair = pair.strip()
            if not pair or ":" not in pair:
                continue
            key, tenant = pair.split(":", 1)
            key, tenant = key.strip(), tenant.strip().lower()
            if key and tenant:
                mapping[key] = tenant
        return mapping


SETTINGS = Settings()


def generate_dev_key(tenant: str) -> str:
    """A URL-safe key for local development. Printed at startup, never persisted."""
    return f"brk_{tenant}_{secrets.token_urlsafe(18)}"
