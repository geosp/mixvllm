"""Read each node's container state and log stage, and draw the status tables."""

from __future__ import annotations

import json
import re
import shlex
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone

from rich.console import Console
from rich.markup import escape
from rich.table import Table

from . import remote
from .config import Cluster, Node

# Only lines matching this leave the node (`grep -E`): the log grows with every request.
LOG_FILTER = (
    r"Loading model from scratch|Initializing model took|checkpoint shards|Loading weights took"
    r"|Model loading took"
    r"|Capturing|kernel_warmup|Warming up|autotune|Free memory on device|init engine"
    r"|Application startup complete|NET/IB : Using|NET/Socket : Using"
    r"|Traceback|[^A-Za-z]ERROR[^A-Za-z]|Error:"
)

# Stages in startup order; a node only moves forward.
STARTING, LOADING, WARMUP, READY = "starting", "loading weights", "warmup", "ready"
_ORDER = {STARTING: 0, LOADING: 1, WARMUP: 2, READY: 3}

_TS = re.compile(r"^(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(?:\.(\d+))?Z\s?")
_SHARDS = re.compile(r"checkpoint shards:.*?\|\s*(\d+)/(\d+)")
_WEIGHTS_TOOK = re.compile(r"Loading weights took ([\d.]+) seconds")
_MODEL_TOOK = re.compile(r"Model loading took ([\d.]+) GiB memory and ([\d.]+) seconds")
_NET_USING = re.compile(r"NET/(IB|Socket) : Using (.*)")


def parse_timestamp(text: str) -> datetime | None:
    """Docker's RFC 3339 timestamps (nanoseconds, Z) -> aware datetime."""
    m = _TS.match(text)
    if not m:
        return None
    frac = (m.group(2) or "0")[:6].ljust(6, "0")
    return datetime.fromisoformat(f"{m.group(1)}.{frac}+00:00")


@dataclass
class LogProgress:
    stage: str = STARTING
    sub: str = ""  # graph capture / warmup detail
    load_pass: int = 0
    shard: tuple[int, int] | None = None
    load_seconds: float | None = None
    load_done_at: datetime | None = None
    engine_ready_at: datetime | None = None
    ready_at: datetime | None = None  # "Application startup complete" (head only)
    transport: str | None = None
    transport_ok: bool | None = None
    errors: list[str] = field(default_factory=list)

    def advance(self, stage: str) -> None:
        if _ORDER[stage] > _ORDER[self.stage]:
            self.stage = stage
            self.sub = ""

    def label(self, headless: bool = False) -> str:
        if self.stage == LOADING:
            parts = [LOADING, f"pass {max(self.load_pass, 1)}"]
            if self.shard:
                parts.append(f"{self.shard[0]}/{self.shard[1]}")
            return " · ".join(parts)
        if self.stage == WARMUP:
            if self.engine_ready_at and headless:
                return READY
            if self.engine_ready_at:
                return "starting API server"
            return self.sub or WARMUP
        return self.stage


def parse_log(lines) -> LogProgress:
    """Startup progress from `docker logs --timestamps` lines (timestamps are optional)."""
    p = LogProgress()
    for raw in lines:
        ts = parse_timestamp(raw)
        line = _TS.sub("", raw, count=1).rstrip("\n")
        # tqdm redraws with \r: the last segment is the current state
        last = line.rsplit("\r", 1)[-1] if "\r" in line else line

        if m := _NET_USING.search(line):
            if p.transport is None or m.group(1) == "Socket":
                p.transport = f"NET/{m.group(1)} : Using {m.group(2).strip()}"
                p.transport_ok = m.group(1) == "IB"
            continue
        if "Loading model from scratch" in line:
            p.advance(LOADING)
        elif "Initializing model took" in line and p.stage == LOADING:
            # printed by every rank at the start of each weight-loading pass
            p.load_pass += 1
            p.shard = None
        elif m := _SHARDS.search(last):
            p.advance(LOADING)
            p.load_pass = max(p.load_pass, 1)
            p.shard = (int(m.group(1)), int(m.group(2)))
        elif _WEIGHTS_TOOK.search(line):
            p.advance(LOADING)
            p.load_pass = max(p.load_pass, 1)
            p.shard = None
        elif m := _MODEL_TOOK.search(line):
            p.advance(WARMUP)
            p.load_seconds = float(m.group(2))
            p.load_done_at = ts
        elif p.stage == WARMUP and "Capturing" in line:
            p.sub = "graph capture"
        elif p.stage == WARMUP and ("kernel_warmup" in line or "Warming up" in line or "autotune" in line):
            p.sub = WARMUP
        elif "Free memory on device" in line or "init engine" in line:
            p.advance(WARMUP)
            p.engine_ready_at = p.engine_ready_at or ts
        elif "Application startup complete" in line:
            p.advance(READY)
            p.ready_at = ts
        elif "Traceback" in line or "ERROR" in line or "Error:" in line:
            p.errors = (p.errors + [last.strip()])[-5:]
    return p


@dataclass
class NodeStatus:
    node: Node
    reachable: bool = True
    error: str = ""
    commit: str = ""
    dirty: bool = False
    state: str = "missing"  # docker state: running, restarting, exited, ... or missing
    exit_code: int | None = None
    image: str = ""
    restarts: int = 0
    started_at: datetime | None = None
    finished_at: datetime | None = None
    log: LogProgress = field(default_factory=LogProgress)

    @property
    def running(self) -> bool:
        return self.state == "running"

    def stage(self) -> str:
        if not self.reachable:
            return "unreachable"
        if self.state == "missing":
            return "no container"
        if not self.running:
            return self.state
        return self.log.label(headless=not self.node.is_head)

    def elapsed(self, now: datetime | None = None) -> float | None:
        if not self.started_at:
            return None
        if self.log.stage == READY and self.log.ready_at:
            end = self.log.ready_at
        elif not self.node.is_head and self.log.engine_ready_at:
            end = self.log.engine_ready_at
        elif not self.running and self.finished_at:
            end = self.finished_at
        else:
            end = now or datetime.now(timezone.utc)
        return max(0.0, (end - self.started_at).total_seconds())


def _status_script(cluster: Cluster, node: Node) -> str:
    c = shlex.quote(node.container)
    return f"""
cd {cluster.repo_path} 2>/dev/null && {{
  git rev-parse --short HEAD
  git status --porcelain --untracked-files=no | head -c 1 | wc -c
}}
echo @@INSPECT
docker inspect {c} --format '{{{{json .State}}}}|{{{{.Config.Image}}}}|{{{{.RestartCount}}}}' 2>/dev/null
echo @@LOGS
started=$(docker inspect -f '{{{{.State.StartedAt}}}}' {c} 2>/dev/null) && \
  docker logs --timestamps --since "$started" {c} 2>&1 | grep -a -E {shlex.quote(LOG_FILTER)}
true
"""


def _parse_docker_time(value: str | None) -> datetime | None:
    if not value or value.startswith("0001-"):
        return None
    return parse_timestamp(value)


def node_status(cluster: Cluster, node: Node) -> NodeStatus:
    st = NodeStatus(node)
    res = remote.run(node, _status_script(cluster, node), timeout=60)
    if not res.ok or "@@INSPECT" not in res.stdout:
        st.reachable = False
        st.error = (res.stderr.strip().splitlines() or ["ssh failed"])[-1]
        return st
    head, rest = res.stdout.split("@@INSPECT\n", 1)
    inspect, logs = rest.split("@@LOGS\n", 1) if "@@LOGS\n" in rest else (rest, "")
    git = head.split()
    if git:
        st.commit = git[0]
        st.dirty = len(git) > 1 and git[1] != "0"
    inspect = inspect.strip()
    if inspect:
        state_json, st.image, restarts = inspect.rsplit("|", 2)
        state = json.loads(state_json)
        st.state = state.get("Status", "unknown")
        st.exit_code = state.get("ExitCode")
        st.restarts = int(restarts)
        st.started_at = _parse_docker_time(state.get("StartedAt"))
        st.finished_at = _parse_docker_time(state.get("FinishedAt"))
        st.log = parse_log(logs.splitlines())
    return st


def cluster_status(cluster: Cluster) -> list[NodeStatus]:
    return remote.parallel(cluster.nodes, lambda n: node_status(cluster, n))


@dataclass
class ApiStatus:
    healthy: bool
    models: list[str]
    error: str = ""


def api_status(cluster: Cluster, timeout: float = 3) -> ApiStatus:
    try:
        with urllib.request.urlopen(f"{cluster.api_url}/health", timeout=timeout) as r:
            healthy = r.status == 200
    except Exception as e:  # connection refused while starting, timeouts, HTTP errors
        return ApiStatus(False, [], str(getattr(e, "reason", e)))
    models = []
    try:
        with urllib.request.urlopen(f"{cluster.api_url}/v1/models", timeout=timeout) as r:
            models = [m["id"] for m in json.load(r).get("data", [])]
    except Exception:
        pass
    return ApiStatus(healthy, models)


def fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


_STATE_STYLE = {"running": "green", "restarting": "yellow", "exited": "red", "dead": "red"}


def status_table(statuses: list[NodeStatus], title: str | None = None) -> Table:
    images = {st.image for st in statuses if st.image}
    # one image everywhere (the normal case) goes under the table; a mismatch keeps the column
    caption = f"image {images.pop()}" if len(images) == 1 and all(st.image for st in statuses) else None
    columns = ("node", "rank", "state", "restarts", "image", "commit", "stage", "elapsed")
    rows = []
    for st in statuses:
        state = st.state if st.reachable else "?"
        stage = st.stage()
        rows.append(
            (
                st.node.name,
                str(st.node.rank),
                (state, _STATE_STYLE.get(state, "yellow")),
                str(st.restarts) if st.reachable else "-",
                st.image or "-",
                (st.commit + ("*" if st.dirty else "")) or "-",
                (stage, "green" if stage == READY else ("red" if stage in ("unreachable", "exited") else "")),
                fmt_duration(st.elapsed()),
            )
        )

    def text(cell) -> str:
        return cell[0] if isinstance(cell, tuple) else cell

    keep = [i for i, col in enumerate(columns) if not (col == "image" and caption)]
    table = Table(title=title, title_justify="left", caption=caption, caption_justify="left")
    for i in keep:
        # On a narrow terminal only the image name gives way ("…"); the rest keep their full width.
        full = max([len(columns[i])] + [len(text(row[i])) for row in rows])
        table.add_column(columns[i], no_wrap=True, overflow="ellipsis", min_width=None if columns[i] == "image" else full)
    for row in rows:
        cells = [row[i] for i in keep]
        table.add_row(*(f"[{c[1]}]{escape(c[0])}[/]" if isinstance(c, tuple) and c[1] else escape(text(c)) for c in cells))
    return table


def plain_lines(statuses: list[NodeStatus]) -> list[str]:
    """The same as status_table, one line per node, for output that isn't a terminal."""
    out = []
    for st in statuses:
        commit = (st.commit + ("*" if st.dirty else "")) or "-"
        out.append(
            f"{st.node.name} rank={st.node.rank} state={st.state if st.reachable else '?'} "
            f"restarts={st.restarts} image={st.image or '-'} commit={commit} "
            f"stage={st.stage()!r} elapsed={fmt_duration(st.elapsed())}"
        )
    return out


def print_status(console: Console, statuses: list[NodeStatus], api: ApiStatus, cluster: Cluster) -> None:
    if console.is_terminal:
        console.print(status_table(statuses))
    else:
        for line in plain_lines(statuses):
            console.print(line, markup=False, highlight=False)
    for st in statuses:
        if not st.reachable:
            console.print(f"[red]{st.node.name}: {escape(st.error)}[/]")
        elif st.log.transport_ok is False:
            console.print(f"[yellow]{st.node.name}: NCCL fell back to TCP ({escape(st.log.transport)})[/]")
        if st.reachable and st.state not in ("running", "missing") and st.log.errors:
            console.print(f"[red]{st.node.name} last error:[/] {escape(st.log.errors[-1])}", highlight=False)
    console.print(f"API {cluster.api_url}: {api_line(statuses, api)}")


def api_line(statuses: list[NodeStatus], api: ApiStatus) -> str:
    """What /health means given the nodes: refused is normal until the head finishes starting."""
    if api.healthy:
        return f"[green]healthy[/] · models: {escape(', '.join(api.models) or '?')}"
    head = statuses[0]
    if all(st.reachable and st.state == "missing" for st in statuses):
        return "[dim]down (no vllm containers running)[/]"
    if head.running and head.log.stage != READY:
        return "[dim]not up yet (opens when the head is ready)[/]"
    return f"[yellow]not answering[/] ({escape(api.error)})"
