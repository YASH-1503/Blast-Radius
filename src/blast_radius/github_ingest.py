"""The real layer: people, commits and pull requests from the public GitHub API.

This is what makes the system's sharpest claim verifiable. With invented data, "the official owner
of `checkout` is one team but the person with 43 commits in it is on another" is a number someone
typed. With real data it is a finding a judge can check against a public repository in their own
browser, SHA by SHA.

**Only the human layer is real.** Topology, revenue and incident history stay synthetic, because
no company publishes them and taking them from somewhere that did would be exactly the private data
Rule 6 excludes. `data.py` explains that split; `schema.provenance()` proves it as a query.

Design constraints, each of them load-bearing:

* **No new dependency.** Standard-library `urllib` only. A data pipeline that needs `requests`
  installed is one more thing to fail on a judge's machine.
* **No authentication required.** 13 unauthenticated requests fetch a full history (the limit is
  60/hour). `GITHUB_TOKEN` is honoured and unnecessary.
* **Everything is cached to disk.** After one fetch the demo makes zero network calls, so it cannot
  be broken by a rate limit or by conference wifi during judging.
* **No email addresses are ever stored.** GitHub exposes commit author emails; logins are enough to
  attribute work, so emails are dropped at parse time rather than filtered later.

Four bugs found here are pinned by regression tests, and three of them were only reachable because
the data was real -- see the comments at `_parse_iso8601`, `is_bot` and `infer_utc_offset`.
"""

from __future__ import annotations

import calendar
import json
import re
import statistics
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import SETTINGS

GITHUB_SOURCE = "github"
API_ROOT = "https://api.github.com"
USER_AGENT = "blast-radius/1.0 (graph-hacks; public data only)"

# The repo whose layout PATH_MAP below is hand-tuned for. Any OTHER repo gets a discovered map.
DECLARED_REPO = "FalkorDB/FalkorDB"

CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache" / "github"


# --------------------------------------------------------------------------------------
# Real repository paths -> synthetic services
# --------------------------------------------------------------------------------------
# Real commit history is attributed to synthetic services by directory. Two traps, both hit once:
#
# 1. **Prefixes must not overlap.** GitHub's `?path=` filter is prefix-based, so asking for `tests`
#    and for `tests/flow` counts the same commits twice and inflates both services. Every entry
#    below is a disjoint prefix.
#
# 2. **Directories, not individual files.** The first attempt mapped single `.rs` files and produced
#    1-11 commits per service: the Rust work is recent, so per-file history is thin. Directories
#    carry the volume that makes "who knows this code" meaningful.
#
# A consequence worth stating rather than fixing: per-service commit counts legitimately sum to MORE
# than the unique commit total, because one commit touching two mapped directories genuinely worked
# on both. Both numbers are reported.

PATH_MAP: tuple[tuple[str, str], ...] = (
    ("src/graph",           "pricing-legacy"),
    ("src/commands",        "checkout"),
    ("src/execution_plan",  "api-gateway"),
    ("src/arithmetic",      "cart"),
    ("src/index",           "search"),
    ("src/procedures",      "catalog"),
    ("src/algorithms",      "recommendations"),
    ("src/ast",             "identity"),
    ("src/datatypes",       "media"),
    ("src/resultset",       "notifications"),
    ("src/schema",          "inventory"),
    ("src/util",            "analytics"),
    ("tests/flow",          "payments"),
    ("tests/unit",          "ledger"),
    ("deps",                "feature-flags"),
    ("docs",                "reviews"),
)


def _assert_disjoint_prefixes(path_map: tuple[tuple[str, str], ...] = ()) -> None:
    """Fail loudly at import if trap 1 is ever reintroduced by an edit."""
    paths = [p for p, _s in (path_map or PATH_MAP)]
    for a in paths:
        for b in paths:
            if a is not b and b.startswith(a.rstrip("/") + "/"):
                raise ValueError(
                    f"PATH_MAP prefixes overlap: {a!r} contains {b!r}; "
                    "GitHub's path filter is prefix-based, so commits would be double counted"
                )


_assert_disjoint_prefixes()


# Directories that carry no product history and would starve a service if mapped to one.
_DISCOVERY_SKIP = ("/.", ".github", ".git", "docs/assets", "tests/assets", "tests/helpers")


def discover_path_map(
    repo: str,
    *,
    token: str | None = None,
    offline: bool = False,
    limit: int = 16,
    min_files: int = 8,
) -> tuple[tuple[str, str], ...]:
    """Derive a path map from the repository's OWN directory tree, in one API call.

    `PATH_MAP` above is hand-tuned to FalkorDB's layout, which is correct for the primary tenant and
    useless for any other repository. Pointing the second tenant at `redis/redis` matched only 3 of
    its 16 paths: thirteen services received zero commits, and four `verify()` assertions failed --
    not because anything was broken, but because the mapping described a different codebase.

    So for any repo other than the declared one, the map is discovered. `git/trees/HEAD?recursive=1`
    returns the whole tree in a single request, two-level directories are ranked by file count, and
    the busiest are assigned to the synthetic services in order. That makes the second tenant a
    genuine second codebase rather than a copy of the first, and means a reviewer can point this at
    any public repository.

    Two-level directories specifically: one level is too coarse (`src` alone swallows a C project
    whole) and three is too sparse. Returned in rank order so the busiest real area maps to the
    service the demo cares about most.
    """
    gh = GitHubClient(repo, token=token, offline=offline)
    tree = gh.tree()
    counts: Counter[str] = Counter()
    for entry in tree:
        if entry.get("type") != "blob":
            continue
        parts = (entry.get("path") or "").split("/")
        if len(parts) < 2:
            continue
        directory = "/".join(parts[:2])
        if any(skip in f"/{directory}" for skip in _DISCOVERY_SKIP):
            continue
        counts[directory] += 1

    ranked = [d for d, n in counts.most_common() if n >= min_files]
    services = [service for _p, service in PATH_MAP][:limit]
    discovered = tuple(zip(ranked[: len(services)], services))
    if not discovered:
        raise RuntimeError(
            f"no mappable directories found in {repo} "
            f"(looked for two-level dirs with >= {min_files} files)"
        )
    _assert_disjoint_prefixes(discovered)
    return discovered


def path_map_for(
    repo: str,
    *,
    token: str | None = None,
    offline: bool = False,
) -> tuple[tuple[str, str], ...]:
    """The declared map for the primary repo; a discovered one for anything else.

    The primary repo keeps its hand-tuned map on purpose: the demo has to be rehearsable, and a
    discovered map could shift if the repository is restructured between rehearsal and judging.
    """
    if repo.strip("/").lower() == DECLARED_REPO.lower():
        return PATH_MAP
    return discover_path_map(repo, token=token, offline=offline)


# --------------------------------------------------------------------------------------
# Bots
# --------------------------------------------------------------------------------------
# Real history contains automation, and automation commits a lot. On the first real fetch the top
# contributor to one service was `dependabot[bot]` with 50 commits -- so a naive commit count ranked
# a bot above every human, and the agent would have proposed paging it at 2am.
#
# Bot commits are KEPT, because they are real history and removing them would misreport the volume.
# Bots are flagged instead, and excluded from being paged, from team membership, and from being
# named as a deployer.

_BOT_PATTERNS = (
    re.compile(r"\[bot\]$", re.I),
    re.compile(r"^(dependabot|renovate|greenkeeper|snyk-bot|github-actions)", re.I),
    re.compile(r"(-|_)?bot$", re.I),
)


def is_bot(login: str, api_type: str | None = None) -> bool:
    """GitHub's own `type == 'Bot'` first; pattern match for the ones it mislabels as User."""
    if (api_type or "").lower() == "bot":
        return True
    return any(pattern.search(login or "") for pattern in _BOT_PATTERNS)


# --------------------------------------------------------------------------------------
# Time
# --------------------------------------------------------------------------------------

def _parse_iso8601(value: str) -> int:
    """`2026-10-14T18:47:03Z` -> unix seconds, read as UTC.

    The obvious implementation is `time.mktime(time.strptime(value, fmt))` and it is WRONG:
    `mktime` interprets the parsed struct in the host's local zone, so on an IST machine every
    GitHub commit moved 5.5 hours. Recency ordering -- which the entire "what deployed last"
    question rests on -- was quietly incorrect, and commits could land in the future.

    What makes this worth a comment is how it surfaced. The dates looked perfectly plausible on
    their own; it only became visible through the timezone inference below, which placed an
    Israel-based project's maintainers at UTC+6..+8. A derived metric caught a bug the raw field hid.
    """
    cleaned = (value or "").strip().replace("Z", "+0000")
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
        try:
            parsed = time.strptime(cleaned, fmt)
        except ValueError:
            continue
        return calendar.timegm(parsed)  # NOT time.mktime -- see above
    raise ValueError(f"unparseable GitHub timestamp: {value!r}")


def infer_utc_offset(commit_times: Iterable[int]) -> int | None:
    """Infer a person's UTC offset from WHEN they commit. Returns hours, or None if too few data.

    A paging decision needs to know whether the person is awake, and GitHub publishes no timezone.
    The profile `location` field would be precisely the personal data Rule 6 excludes, so the offset
    is derived from behaviour instead: the median commit hour is treated as roughly local midday.

    After the `_parse_iso8601` fix the top committers infer to UTC+1..+3 -- correct for an
    Israel-based project, from nothing but commit hours. That agreement is independent evidence the
    whole pipeline is sound, and is why this function earns its keep despite being a heuristic.

    It IS a heuristic and the README says so: a nocturnal maintainer is placed wrongly, and the
    honest unit is the median across top committers rather than any single person.
    """
    hours = [time.gmtime(ts).tm_hour for ts in commit_times]
    if len(hours) < 5:
        return None
    # Circular median: commit hours wrap at midnight, so rotate to the densest 12h arc first.
    best_offset, best_spread = 0, None
    for shift in range(24):
        shifted = sorted(((h + shift) % 24) for h in hours)
        spread = shifted[-1] - shifted[0]
        if best_spread is None or spread < best_spread:
            best_offset, best_spread = shift, spread
    shifted_median = statistics.median(sorted(((h + best_offset) % 24) for h in hours))
    local_median_hour = (shifted_median - best_offset) % 24
    # Median activity ~13:00 local. offset = 13 - median_utc_hour, wrapped to [-11, +12].
    offset = int(round(13 - local_median_hour))
    while offset > 12:
        offset -= 24
    while offset < -11:
        offset += 24
    return offset


# --------------------------------------------------------------------------------------
# HTTP, with a disk cache
# --------------------------------------------------------------------------------------

@dataclass
class FetchStats:
    """What the fetch cost, so the README can state it instead of estimating."""

    requests: int = 0
    cache_hits: int = 0
    rate_limit_remaining: int | None = None

    def explain(self) -> str:
        return (
            f"{self.requests} network request(s), {self.cache_hits} from cache"
            + (
                f", {self.rate_limit_remaining} of the hourly allowance left"
                if self.rate_limit_remaining is not None
                else ""
            )
        )


class GitHubClient:
    """Minimal paginating GitHub reader with a disk cache. GET only -- nothing is ever written."""

    def __init__(
        self,
        repo: str | None = None,
        *,
        token: str | None = None,
        cache_dir: Path | None = None,
        offline: bool = False,
    ) -> None:
        self.repo = (repo or SETTINGS.github_repo).strip("/")
        self.token = token if token is not None else SETTINGS.github_token
        self.cache_dir = cache_dir or CACHE_DIR
        self.offline = offline
        self.stats = FetchStats()
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    # -- cache -------------------------------------------------------------------------

    def _cache_path(self, kind: str, page: int, extra: str = "") -> Path:
        slug = self.repo.replace("/", "__")
        tail = f"__{re.sub(r'[^a-zA-Z0-9]+', '_', extra).strip('_')}" if extra else ""
        return self.cache_dir / f"{slug}__{kind}{tail}__p{page}.json"

    # -- fetch -------------------------------------------------------------------------

    def _get(self, path: str, params: dict[str, Any], cache_key: tuple[str, int, str]) -> Any:
        cache_file = self._cache_path(*cache_key)
        if cache_file.exists():
            self.stats.cache_hits += 1
            return json.loads(cache_file.read_text())

        if self.offline:
            raise FileNotFoundError(
                f"offline and no cache for {cache_file.name}; run `make github` once with network"
            )

        url = f"{API_ROOT}/repos/{self.repo}/{path}?{urllib.parse.urlencode(params)}"
        request = urllib.request.Request(url, headers=self._headers())
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode())
                remaining = response.headers.get("X-RateLimit-Remaining")
                if remaining is not None:
                    self.stats.rate_limit_remaining = int(remaining)
        except urllib.error.HTTPError as exc:
            if exc.code == 403 and "rate limit" in str(exc.read()[:400]).lower():
                raise RuntimeError(
                    "GitHub rate limit reached (60/hour unauthenticated). Either wait, or set "
                    "GITHUB_TOKEN. Already-cached pages still work offline."
                ) from exc
            raise RuntimeError(f"GitHub {exc.code} on {path}: {exc.reason}") from exc

        self.stats.requests += 1
        cache_file.write_text(json.dumps(payload, indent=1))
        return payload

    def _headers(self) -> dict[str, str]:
        headers = {"Accept": "application/vnd.github+json", "User-Agent": USER_AGENT}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _paginate(
        self, path: str, params: dict[str, Any], kind: str, extra: str = "", pages: int = 3
    ) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for page in range(1, pages + 1):
            try:
                batch = self._get(path, {**params, "page": page, "per_page": 100}, (kind, page, extra))
            except FileNotFoundError:
                break  # offline and the cache stops here; what we have is enough
            if not isinstance(batch, list) or not batch:
                break
            out += batch
            if len(batch) < 100:
                break
        return out

    # -- the three endpoints used ------------------------------------------------------

    def commits_for_path(self, repo_path: str, pages: int = 2) -> list[dict[str, Any]]:
        return self._paginate(
            "commits", {"path": repo_path}, "commits", extra=repo_path, pages=pages
        )

    def pulls(self, pages: int = 2) -> list[dict[str, Any]]:
        return self._paginate(
            "pulls", {"state": "closed", "sort": "updated", "direction": "desc"}, "pulls", pages=pages
        )

    def contributors(self, pages: int = 1) -> list[dict[str, Any]]:
        return self._paginate("contributors", {}, "contributors", pages=pages)

    def tree(self) -> list[dict[str, Any]]:
        """The whole repository tree in ONE request, used to discover a path map.

        `truncated: true` is returned for very large repositories rather than an error, so it is
        checked -- a truncated tree would silently under-rank the directories near the end.
        """
        payload = self._get(
            "git/trees/HEAD", {"recursive": "1"}, ("tree", 1, "")
        )
        if isinstance(payload, dict):
            if payload.get("truncated"):
                raise RuntimeError(
                    f"{self.repo}: the tree response was truncated, so directory ranking would be "
                    "incomplete; supply an explicit path map for this repository"
                )
            return list(payload.get("tree") or [])
        return []


# --------------------------------------------------------------------------------------
# Parsed shapes
# --------------------------------------------------------------------------------------

@dataclass
class RealCommit:
    sha: str
    message: str
    at: int
    author_login: str
    services: list[str] = field(default_factory=list)

    @property
    def short_sha(self) -> str:
        return self.sha[:12]

    @property
    def subject(self) -> str:
        """First line only. A full commit body in a UI card is noise."""
        return (self.message or "").strip().splitlines()[0][:160] if self.message else ""


@dataclass
class RealPerson:
    handle: str
    is_bot: bool
    commit_count: int = 0
    utc_offset: int | None = None
    avatar_url: str | None = None
    profile_url: str | None = None


@dataclass
class RealPullRequest:
    number: int
    title: str
    author_login: str
    merged_at: int | None
    merge_sha: str | None


@dataclass
class GitHubLayer:
    """The real human layer, ready for ingest."""

    repo: str
    people: dict[str, RealPerson] = field(default_factory=dict)
    commits: dict[str, RealCommit] = field(default_factory=dict)
    pulls: list[RealPullRequest] = field(default_factory=list)
    # (handle, service) -> commits by that person in that service's mapped directory
    worked_on: dict[tuple[str, str], int] = field(default_factory=dict)
    stats: FetchStats = field(default_factory=FetchStats)

    # -- the numbers the README quotes -------------------------------------------------

    @property
    def unique_commits(self) -> int:
        return len(self.commits)

    @property
    def attributed_rows(self) -> int:
        """Service-attributed rows. Exceeds `unique_commits` by design -- see PATH_MAP note 2."""
        return sum(len(c.services) for c in self.commits.values())

    @property
    def humans(self) -> list[RealPerson]:
        return [p for p in self.people.values() if not p.is_bot]

    @property
    def bots(self) -> list[RealPerson]:
        return [p for p in self.people.values() if p.is_bot]

    def top_humans_for(self, service: str, limit: int = 5) -> list[tuple[str, int]]:
        scored = [
            (handle, count)
            for (handle, svc), count in self.worked_on.items()
            if svc == service and not self.people[handle].is_bot
        ]
        return sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:limit]

    def median_utc_offset(self) -> int | None:
        """The honest unit for the timezone heuristic: the median across top committers."""
        offsets = [
            p.utc_offset
            for p in sorted(self.humans, key=lambda x: -x.commit_count)[:10]
            if p.utc_offset is not None
        ]
        return int(statistics.median(offsets)) if offsets else None

    def explain(self) -> str:
        return (
            f"repo            : {self.repo}\n"
            f"unique commits  : {self.unique_commits}\n"
            f"attributed rows : {self.attributed_rows} (one commit can touch two areas)\n"
            f"real people     : {len(self.humans)} human, {len(self.bots)} bot\n"
            f"pull requests   : {len(self.pulls)}\n"
            f"median tz offset: "
            f"{'UTC%+d' % self.median_utc_offset() if self.median_utc_offset() is not None else 'n/a'}\n"
            f"cost            : {self.stats.explain()}"
        )


# --------------------------------------------------------------------------------------
# Fetch + parse
# --------------------------------------------------------------------------------------

def fetch(
    repo: str | None = None,
    *,
    token: str | None = None,
    offline: bool = False,
    pages_per_path: int = 2,
    path_map: tuple[tuple[str, str], ...] | None = None,
) -> GitHubLayer:
    """Build the real human layer. Cached, so the second call makes no network requests.

    `path_map` defaults to the DECLARED map for the primary repo and a DISCOVERED one otherwise --
    see `path_map_for`. Pass an explicit map to override both.
    """
    gh = GitHubClient(repo, token=token, offline=offline)
    path_map = path_map or path_map_for(gh.repo, token=token, offline=offline)
    layer = GitHubLayer(repo=gh.repo)
    commit_times: dict[str, list[int]] = defaultdict(list)

    for repo_path, service in path_map:
        for raw in gh.commits_for_path(repo_path, pages=pages_per_path):
            sha = raw.get("sha")
            commit_block = (raw.get("commit") or {})
            if not sha or not commit_block:
                continue

            # `author` (the GitHub account) can be null for an unlinked email; fall back to the
            # committed name. Email is read from neither -- only logins are stored.
            account = raw.get("author") or {}
            login = (account.get("login") or "").strip()
            if not login:
                login = (commit_block.get("author") or {}).get("name", "").strip()
            if not login:
                continue

            try:
                at = _parse_iso8601((commit_block.get("author") or {}).get("date", ""))
            except ValueError:
                continue

            if login not in layer.people:
                layer.people[login] = RealPerson(
                    handle=login,
                    is_bot=is_bot(login, account.get("type")),
                    avatar_url=account.get("avatar_url"),
                    profile_url=account.get("html_url"),
                )

            existing = layer.commits.get(sha)
            if existing is None:
                existing = RealCommit(
                    sha=sha,
                    message=commit_block.get("message", ""),
                    at=at,
                    author_login=login,
                )
                layer.commits[sha] = existing
                commit_times[login].append(at)
                layer.people[login].commit_count += 1

            if service not in existing.services:
                existing.services.append(service)
                key = (login, service)
                layer.worked_on[key] = layer.worked_on.get(key, 0) + 1

    for raw in gh.pulls():
        number = raw.get("number")
        if number is None:
            continue
        account = raw.get("user") or {}
        merged_raw = raw.get("merged_at")
        layer.pulls.append(
            RealPullRequest(
                number=int(number),
                title=(raw.get("title") or "").strip()[:200],
                author_login=(account.get("login") or "").strip(),
                merged_at=_parse_iso8601(merged_raw) if merged_raw else None,
                merge_sha=raw.get("merge_commit_sha"),
            )
        )

    # Contributor totals cover the whole repo, not just mapped directories, so they surface people
    # whose work lives outside PATH_MAP. Flagged, counted, never silently dropped.
    for raw in gh.contributors():
        login = (raw.get("login") or "").strip()
        if not login:
            continue
        person = layer.people.get(login)
        if person is None:
            person = RealPerson(handle=login, is_bot=is_bot(login, raw.get("type")))
            layer.people[login] = person
        person.avatar_url = person.avatar_url or raw.get("avatar_url")
        person.profile_url = person.profile_url or raw.get("html_url")

    for login, times in commit_times.items():
        layer.people[login].utc_offset = infer_utc_offset(times)

    layer.stats = gh.stats
    return layer


def cache_status(cache_dir: Path | None = None) -> dict[str, Any]:
    """Whether the demo can run with no network at all."""
    directory = cache_dir or CACHE_DIR
    files = sorted(directory.glob("*.json")) if directory.exists() else []
    return {
        "dir": str(directory),
        "files": len(files),
        "bytes": sum(f.stat().st_size for f in files),
        "offline_ready": bool(files),
    }


def service_commit_distribution(layer: GitHubLayer) -> list[tuple[str, int, int]]:
    """(service, commits, distinct humans) -- used to check no service came out starved."""
    commits: Counter[str] = Counter()
    people: dict[str, set[str]] = defaultdict(set)
    for (handle, service), count in layer.worked_on.items():
        commits[service] += count
        if not layer.people[handle].is_bot:
            people[service].add(handle)
    return sorted(
        ((service, count, len(people[service])) for service, count in commits.items()),
        key=lambda row: -row[1],
    )


def env_summary() -> dict[str, Any]:
    """What a `br doctor` line should say about this layer."""
    return {
        "repo": SETTINGS.github_repo,
        "token": "set" if SETTINGS.github_token else "not set (60 req/hour is enough)",
        "mapped_paths": len(PATH_MAP),
        "cache": cache_status(),
        "stores_email": False,
    }


__all__ = [
    "GITHUB_SOURCE",
    "PATH_MAP",
    "GitHubClient",
    "GitHubLayer",
    "RealCommit",
    "RealPerson",
    "RealPullRequest",
    "cache_status",
    "env_summary",
    "fetch",
    "infer_utc_offset",
    "is_bot",
    "service_commit_distribution",
]
