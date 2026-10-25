"""spark-cluster: start, stop, check and report on the cluster defined in docker/cluster.yml."""

from __future__ import annotations

import argparse
import csv
import difflib
import json
import os
import shlex
import socket
import sys
import time
import urllib.request
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.panel import Panel
from rich.prompt import Confirm
from rich.table import Table

from . import checks as checks_mod
from . import progress, remote
from .config import (
    CLUSTER_FILE,
    REPO_ROOT,
    Cluster,
    ConfigError,
    load_cluster,
    load_registry,
    model_config,
    render_env,
    set_model_name,
)

console = Console(highlight=False, soft_wrap=True)

LOCK_PATH = "/tmp/spark-cluster.lock"
STARTUPS_CSV = REPO_ROOT / "docs" / "startups.csv"
POLL_SECONDS = 5
TEST_PROMPT = "Reply with one word: hello."


class Failed(Exception):
    """A command failed; the message has already been explained to the user or is in the exception."""


# ---------------------------------------------------------------- output


def error(msg: str) -> None:
    if console.is_terminal:
        console.print(Panel(escape(msg), title="error", title_align="left", border_style="red"))
    else:
        console.print(f"error: {msg}", markup=False)


def warn(msg: str) -> None:
    if console.is_terminal:
        console.print(Panel(escape(msg), title="warning", title_align="left", border_style="yellow"))
    else:
        console.print(f"warning: {msg}", markup=False)


def info(msg: str) -> None:
    console.print(msg)


_LEVEL = {checks_mod.OK: "[green]ok[/]", checks_mod.WARN: "[yellow]warn[/]", checks_mod.FAIL: "[red]FAIL[/]"}


def print_checks(results: list[checks_mod.Check]) -> bool:
    """Print check results; True if nothing failed."""
    if console.is_terminal:
        table = Table(show_header=False, box=None, padding=(0, 1))
        for c in results:
            table.add_row(_LEVEL[c.level], escape(c.what), escape(c.detail))
        console.print(table)
    else:
        for c in results:
            console.print(f"{c.level:4} {c.what}: {c.detail}", markup=False)
    failed = [c for c in results if c.level == checks_mod.FAIL]
    if failed:
        error(f"{len(failed)} check(s) failed")
    return not failed


# ---------------------------------------------------------------- lock


def _lock_owner() -> str:
    return f"{socket.gethostname()} {os.getpid()} {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ}"


@contextmanager
def cluster_lock(cluster: Cluster, force: bool = False):
    """One changing command at a time, via a lock file on the head."""
    owner = _lock_owner()
    lock = shlex.quote(LOCK_PATH)
    script = (
        (f"rm -f {lock}; " if force else "")
        + f"if ( set -C; printf '%s\\n' {shlex.quote(owner)} > {lock} ) 2>/dev/null; "
        f"then echo @@ACQUIRED; else cat {lock}; fi"
    )
    res = remote.run(cluster.head, script).check("taking the lock")
    if "@@ACQUIRED" not in res.stdout:
        holder = res.stdout.strip() or "unknown"
        host, pid = (holder.split() + ["", ""])[:2]
        hint = "Wait for it to finish."
        if host == socket.gethostname() and pid.isdigit() and not _pid_alive(int(pid)):
            hint = "Its process is gone: rerun with --force-unlock."
        elif host != socket.gethostname():
            hint = "If that run is gone, rerun with --force-unlock."
        raise Failed(f"another spark-cluster run holds {LOCK_PATH} on {cluster.head.name}: {holder}. {hint}")
    try:
        yield
    finally:
        remote.run(cluster.head, f'[ "$(cat {lock} 2>/dev/null)" = {shlex.quote(owner)} ] && rm -f {lock}')


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


# ---------------------------------------------------------------- sync


def sync(cluster: Cluster, dry_run: bool = False) -> None:
    """Write each node's .env if it changed (or show the diff)."""
    node_dir = cluster.node_dir()

    def one(node):
        wanted = render_env(cluster, node)
        res = remote.run(node, f"test -d {node_dir} || echo @@NODIR; cat {node_dir}/.env 2>/dev/null; true")
        res.check("reading .env")
        if res.stdout.startswith("@@NODIR"):
            return node, None, f"{node_dir} doesn't exist: the node is on an old commit (spark-cluster pull)"
        current = res.stdout
        if current == wanted:
            return node, "unchanged", None
        diff = "".join(
            difflib.unified_diff(
                current.splitlines(keepends=True),
                wanted.splitlines(keepends=True),
                f"{node.name}:{node_dir}/.env",
                "rendered from cluster.yml",
            )
        )
        if dry_run:
            return node, diff, None
        remote.run(
            node, f"cat > {node_dir}/.env.tmp && mv {node_dir}/.env.tmp {node_dir}/.env", input=wanted
        ).check("writing .env")
        return node, "updated", None

    problems = []
    for node, outcome, problem in remote.parallel(cluster.nodes, one):
        if problem:
            problems.append(f"{node.name}: {problem}")
        elif dry_run and outcome != "unchanged":
            console.print(f"[bold]{node.name}[/]: would change")
            console.print(outcome, markup=False)
        else:
            info(f"{node.name}: .env {outcome}")
    if problems:
        raise Failed("\n".join(problems))


# ---------------------------------------------------------------- up / down


def compose(cluster: Cluster, node, args: str) -> remote.Result:
    return remote.run(node, f"cd {cluster.node_dir()} && docker compose {args}", timeout=300)


def do_down(cluster: Cluster) -> None:
    for node in reversed(cluster.nodes):
        info(f"{node.name}: stopping")
        compose(cluster, node, "down").check("docker compose down")


def do_up(cluster: Cluster, timeout_min: float = 25, drop_caches: bool = True) -> bool | None:
    """check → sync → pull image → free memory → start in rank order → watch. True healthy,
    False failed, None when the user stopped watching."""
    info(f"[bold]Checking[/] ({cluster.model_name} on {len(cluster.nodes)} nodes)")
    if not print_checks(checks_mod.run_checks(cluster)):
        return False

    info("[bold]Syncing[/] .env")
    sync(cluster)

    info(f"[bold]Pulling[/] {cluster.image}")
    for res in remote.parallel(cluster.nodes, lambda n: remote.run(n, f"docker pull -q {shlex.quote(cluster.image)}", timeout=3600)):
        res.check("docker pull")

    if drop_caches:
        free_memory(cluster)

    for node in cluster.nodes:
        info(f"[bold]Starting[/] {node.name} (rank {node.rank})")
        compose(cluster, node, "up -d").check("docker compose up -d")

    return watch(cluster, timeout_min)


# vLLM checks free memory against gpu-memory-utilization when the engine starts. On unified
# memory the page cache counts as used, so drop it and compact first. /proc/sys/vm isn't
# namespaced: a privileged container (docker group, no sudo) writes the host's settings.
DROP_CACHES = "sync; echo 3 > /proc/sys/vm/drop_caches && echo 1 > /proc/sys/vm/compact_memory"


def free_memory(cluster: Cluster) -> None:
    info("[bold]Freeing memory[/] (drop page cache, compact)")
    mem_free = "awk '/^MemFree:/ {print $2}' /proc/meminfo"
    command = (
        f"{mem_free}; docker run --rm --privileged --entrypoint /bin/sh {shlex.quote(cluster.image)} "
        f"-c {shlex.quote(DROP_CACHES)} && {mem_free}"
    )
    for res in remote.parallel(cluster.nodes, lambda n: remote.run(n, command, timeout=300)):
        res.check("dropping caches")
        before, after = (int(v) / 2**20 for v in res.stdout.split())
        info(f"{res.node.name}: free memory {before:.0f} → {after:.0f} GiB")


def watch(cluster: Cluster, timeout_min: float) -> bool | None:
    deadline = time.monotonic() + timeout_min * 60
    baseline: dict[str, int] = {}
    last_key: list[tuple] = []
    msg = None

    def poll() -> tuple[list[progress.NodeStatus], progress.ApiStatus]:
        statuses = progress.cluster_status(cluster)
        for st in statuses:
            baseline.setdefault(st.node.name, st.restarts)
        return statuses, progress.api_status(cluster)

    def problem(statuses) -> str | None:
        for st in statuses:
            if not st.reachable:
                continue  # transient; the next poll retries
            if st.state == "missing":
                return f"{st.node.name}: the container is gone"
            if st.state != "running" or st.restarts > baseline[st.node.name]:
                why = f"exited with code {st.exit_code}" if st.state == "exited" else st.state
                if st.restarts > baseline[st.node.name]:
                    why = f"restarted ({st.restarts}×)"
                last = st.log.errors[-1] if st.log.errors else "no error line in the log"
                return f"{st.node.name}: the container {why}.\nLast error: {last}\nSee: spark-cluster logs {st.node.name}"
        return None

    title = f"Starting {cluster.model_name}"
    try:
        if console.is_terminal:
            with Live(console=console, refresh_per_second=1, transient=False) as live:
                while True:
                    statuses, api = poll()
                    live.update(progress.status_table(statuses, title=title))
                    if api.healthy or (msg := problem(statuses)) or time.monotonic() > deadline:
                        break
                    time.sleep(POLL_SECONDS)
        else:
            while True:
                statuses, api = poll()
                # print when a state or stage changes, not every time the elapsed time ticks
                key = [(st.state, st.restarts, st.stage()) for st in statuses]
                if key != last_key:
                    for line in progress.plain_lines(statuses):
                        console.print(line, markup=False)
                    last_key = key
                if api.healthy or (msg := problem(statuses)) or time.monotonic() > deadline:
                    break
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        warn("Stopped watching. The cluster keeps starting: follow it with `spark-cluster status`.")
        return None

    if api.healthy:
        readiness_summary(cluster)
        return True
    if msg:
        error(msg)
    else:
        error(f"not healthy after {timeout_min:.0f} minutes; the containers are still running (spark-cluster status)")
    return False


# ---------------------------------------------------------------- readiness summary


def test_request(cluster: Cluster, timeout: float = 120) -> tuple[float | None, str]:
    """One streamed chat request; (time to first token, error)."""
    body = json.dumps(
        {
            "model": "spark",
            "messages": [{"role": "user", "content": TEST_PROMPT}],
            "max_tokens": 64,
            "reasoning_effort": "low",
            "stream": True,
        }
    ).encode()
    req = urllib.request.Request(
        f"{cluster.api_url}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"}
    )
    start = time.monotonic()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode().strip()
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                for choice in json.loads(line[5:]).get("choices", []):
                    delta = choice.get("delta", {})
                    if delta.get("content") or delta.get("reasoning_content") or delta.get("reasoning"):
                        return time.monotonic() - start, ""
        return None, "no tokens in the reply"
    except Exception as e:
        return None, str(e)


def readiness_summary(cluster: Cluster) -> None:
    statuses = progress.cluster_status(cluster)
    head = statuses[0]
    loads = {st.node.name: st.log.load_seconds for st in statuses}
    done = [st.log.load_done_at for st in statuses if st.log.load_done_at]
    ready_at = head.log.ready_at
    warmup = (ready_at - max(done)).total_seconds() if ready_at and done else None
    total = (ready_at - head.started_at).total_seconds() if ready_at and head.started_at else None
    transport = head.log.transport or "not in the log"
    ttft, req_error = test_request(cluster)

    table = Table(title=f"{cluster.model_name} is ready", title_justify="left", show_header=False)
    table.add_column(style="bold")
    table.add_column()
    for name, secs in loads.items():
        table.add_row(f"model load · {name}", progress.fmt_duration(secs))
    table.add_row("warmup", progress.fmt_duration(warmup))
    table.add_row("total (head start → ready)", progress.fmt_duration(total))
    table.add_row("NCCL", escape(transport))
    table.add_row("test request", f"first token in {ttft:.2f} s" if ttft is not None else f"[red]failed[/]: {escape(req_error)}")
    console.print(table)

    for st in statuses:
        if st.log.transport_ok is False:
            warn(f"{st.node.name}: NCCL fell back to TCP ({st.log.transport})")

    row = {
        "date": datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        "model": cluster.model_name,
        "commit": head.commit,
        "image": cluster.image,
        "node_load_s": " ".join(f"{n}={s:.0f}" for n, s in loads.items() if s is not None),
        "warmup_s": f"{warmup:.0f}" if warmup is not None else "",
        "total_s": f"{total:.0f}" if total is not None else "",
        "nccl_transport": transport,
        "ttft_s": f"{ttft:.2f}" if ttft is not None else "",
    }
    append_startup(STARTUPS_CSV, row)
    info(f"Recorded in {STARTUPS_CSV.relative_to(REPO_ROOT)}")


def append_startup(path: Path, row: dict) -> None:
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if new:
            writer.writeheader()
        writer.writerow(row)


# ---------------------------------------------------------------- commands


def cmd_status(cluster: Cluster, args) -> int:
    statuses = progress.cluster_status(cluster)
    progress.print_status(console, statuses, progress.api_status(cluster), cluster)
    return 0 if all(st.reachable for st in statuses) else 1


def cmd_check(cluster: Cluster, args) -> int:
    return 0 if print_checks(checks_mod.run_checks(cluster)) else 1


def cmd_sync(cluster: Cluster, args) -> int:
    if args.dry_run:
        sync(cluster, dry_run=True)
        return 0
    with cluster_lock(cluster, args.force_unlock):
        sync(cluster)
    return 0


def cmd_up(cluster: Cluster, args) -> int:
    with cluster_lock(cluster, args.force_unlock):
        result = do_up(cluster, args.timeout, not args.no_drop_caches)
    return {True: 0, False: 1, None: 130}[result]


def cmd_down(cluster: Cluster, args) -> int:
    with cluster_lock(cluster, args.force_unlock):
        do_down(cluster)
    return 0


def cmd_restart(cluster: Cluster, args) -> int:
    with cluster_lock(cluster, args.force_unlock):
        do_down(cluster)
        result = do_up(cluster, args.timeout, not args.no_drop_caches)
    return {True: 0, False: 1, None: 130}[result]


def cmd_logs(cluster: Cluster, args) -> int:
    node = cluster.node(args.node) if args.node else cluster.head
    return remote.run_interactive(node, f"docker logs -f --tail={args.tail} {shlex.quote(node.container)}")


def cmd_pull(cluster: Cluster, args) -> int:
    def changes(node):
        return remote.run(node, f"cd {cluster.repo_path} && git status --porcelain --untracked-files=no").check("git status")

    dirty = [r for r in remote.parallel(cluster.nodes, changes) if r.stdout.strip()]
    if dirty:
        raise Failed(
            "local changes, not pulling:\n"
            + "\n".join(f"{r.node.name}: {', '.join(line[3:] for line in r.stdout.splitlines())}" for r in dirty)
        )
    failed = False
    pull = "git pull --ff-only 2>&1 && echo @@HEAD && git log --oneline -1"
    for res in remote.parallel(cluster.nodes, lambda n: remote.run(n, f"cd {cluster.repo_path} && {pull}")):
        if res.ok:
            output, head = res.stdout.split("@@HEAD\n", 1)
            what = "already up to date" if "Already up to date" in output else "updated"
            info(f"{res.node.name}: {what}, at {escape(head.strip())}")
        else:
            failed = True
            error(f"{res.node.name}: git pull failed\n{res.stdout.strip()}")
    return 1 if failed else 0


def _registry_entry(name: str) -> dict:
    cfg = model_config(load_registry(), name)
    if cfg is None:
        known = ", ".join(load_registry().get("models", {}))
        raise Failed(f"{name!r} isn't in docker/config/model_registry.yml (known: {known})")
    return cfg


def cmd_download(cluster: Cluster, args) -> int:
    repo_id = _registry_entry(args.model)["model"]
    head = cluster.head
    hf_home = cluster.env["HF_HOME"]
    py = (
        "from huggingface_hub import snapshot_download\n"
        f"print('DONE', snapshot_download({repo_id!r}, max_workers=8))"
    )
    # The fixed container name also keeps two downloads from running at once.
    command = (
        f"cd {cluster.node_dir()} && docker run --rm --name spark-cluster-download "
        "--entrypoint /usr/bin/env --env-file secrets.env -e HF_HOME=/root/.cache/huggingface "
        f"-v {shlex.quote(hf_home)}:/root/.cache/huggingface {shlex.quote(cluster.image)} "
        f"python3 -c {shlex.quote(py)}"
    )
    info(f"Downloading {repo_id} into {hf_home} on {head.name}")
    return remote.run_interactive(head, command)


def cmd_model(cluster: Cluster, args) -> int:
    new = args.model
    cfg = _registry_entry(new)
    split = checks_mod.gpu_split(cfg)
    if split != len(cluster.nodes):
        raise Failed(f"{new} needs {split} GPUs (tensor × pipeline parallel), the cluster has {len(cluster.nodes)} nodes")
    files = checks_mod.model_files_on_head(cluster, cfg["model"])
    if not files.get("complete"):
        raise Failed(f"{cfg['model']} isn't fully downloaded: run `spark-cluster download {new}` first")
    old = cluster.model_name
    if new == old:
        info(f"{new} is already the configured model (use `spark-cluster restart` to restart it)")
        return 0

    with cluster_lock(cluster, args.force_unlock):
        write_model_name(cluster.path, new)
        info(f"cluster.yml: MODEL_NAME {old} → {new}")
        cluster = load_cluster(cluster.path)
        do_down(cluster)
        result = do_up(cluster, args.timeout, not args.no_drop_caches)
        if result is False:
            if console.is_terminal and Confirm.ask(f"{new} didn't come up. Switch back to {old} and restart?", default=True):
                write_model_name(cluster.path, old)
                info(f"cluster.yml: MODEL_NAME {new} → {old}")
                cluster = load_cluster(cluster.path)
                do_down(cluster)
                do_up(cluster, args.timeout, not args.no_drop_caches)
            else:
                warn(f"cluster.yml still says {new}; the cluster is not serving.")
    return {True: 0, False: 1, None: 130}[result]


def write_model_name(path: Path, name: str) -> None:
    path.write_text(set_model_name(path.read_text(), name))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="spark-cluster", description=__doc__)
    p.add_argument("-c", "--config", type=Path, default=CLUSTER_FILE, help="cluster file (default: docker/cluster.yml)")
    sub = p.add_subparsers(dest="command", required=True, metavar="command")

    def add(name, fn, help, changes=False, starts=False):
        sp = sub.add_parser(name, help=help, description=help)
        sp.set_defaults(fn=fn)
        if changes:
            sp.add_argument("--force-unlock", action="store_true", help="clear a stale lock left by a dead run")
        if starts:
            sp.add_argument("--timeout", type=float, default=25, metavar="MIN", help="minutes to wait for /health (25)")
            sp.add_argument(
                "--no-drop-caches", action="store_true", help="skip dropping the page cache and compacting memory"
            )
        return sp

    add("status", cmd_status, "containers, commits and startup stage of every node")
    add("check", cmd_check, "verify commits, registry, model files, memory and host setup")
    add("sync", cmd_sync, "write each node's .env from cluster.yml", changes=True).add_argument(
        "--dry-run", action="store_true", help="show the diff, write nothing"
    )
    add("download", cmd_download, "download a registry model into the shared folder (on the head)").add_argument("model")
    add("up", cmd_up, "check, sync, pull the image, free memory, start in rank order and follow progress", changes=True, starts=True)
    add("down", cmd_down, "stop every node, last rank first", changes=True)
    add("restart", cmd_restart, "down, then up", changes=True, starts=True)
    logs = add("logs", cmd_logs, "follow a node's container log (default: the head)")
    logs.add_argument("node", nargs="?")
    logs.add_argument("--tail", type=int, default=50)
    add("pull", cmd_pull, "git pull --ff-only on every node (refuses on local changes)")
    add("model", cmd_model, "switch MODEL_NAME in cluster.yml and restart", changes=True, starts=True).add_argument("model")
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        cluster = load_cluster(args.config)
        code = args.fn(cluster, args)
    except (ConfigError, Failed, remote.RemoteError) as e:
        error(str(e))
        code = 1
    except KeyboardInterrupt:
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
