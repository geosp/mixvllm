"""spark-cluster eval: run the private evaluation suite against the loaded model, follow it, report.

The suite (prompts, hidden tests, answer keys) lives on the head at cluster.yml's evals_path, outside
this public repo. `eval run` copies evalkit/ there and starts evalkit/runner.py detached on the
head; everything else reads that run's files over SSH. See docs/model-evaluations.md.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import re
import shlex
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from rich.console import Console
from rich.live import Live
from rich.markup import escape
from rich.table import Table

from . import checks as checks_mod
from . import remote
from .config import REPO_ROOT, Cluster, load_registry, model_config
from .progress import api_status, fmt_duration

console = Console(highlight=False, soft_wrap=True)

EVALKIT = Path(__file__).with_name("evalkit")
HARNESS_FILES = ("runner.py", "sandbox.py", "shim.py", "verify.py", "grader_template.py")
PUBLIC_SUITES = REPO_ROOT / "evals" / "suites"  # task specs only; answer keys never live in the repo
_UNPACK = (
    "import base64, json, os, sys\n"
    "dest, keep = sys.argv[1], sys.argv[2] == 'keep'\n"
    "skipped = []\n"
    "for name, data in json.load(sys.stdin).items():\n"
    "    path = os.path.normpath(os.path.join(dest, name))\n"
    "    if keep and os.path.exists(path):\n"
    "        skipped.append(name); continue\n"
    "    os.makedirs(os.path.dirname(path), exist_ok=True)\n"
    "    open(path, 'wb').write(base64.b64decode(data))\n"
    "print(json.dumps(skipped))\n"
)
DEFAULT_SUITES = ["speed", "tools", "challenge1", "challenge2", "challenge3", "vision"]
POLL_SECONDS = 3
LOCK_PATH = "/tmp/spark-cluster.lock"  # same file as cli.LOCK_PATH


class EvalError(Exception):
    pass


def root(cluster: Cluster) -> str:
    if not cluster.evals_path:
        raise EvalError("cluster.yml has no evals_path: set it to the private suite folder on the head")
    return cluster.evals_path


def sh(cluster: Cluster, script: str, input: str | None = None, timeout: float = 120) -> str:
    res = remote.run(cluster.head, script, input=input, timeout=timeout)
    if not res.ok:
        raise EvalError(f"on {cluster.head.name}: {(res.stderr or res.stdout).strip()[-400:]}")
    return res.stdout


# ---------------------------------------------------------------- run


def upload(cluster: Cluster, dest: str, files: dict[str, bytes], keep_existing: bool = False) -> list[str]:
    """Write files under dest on the head (base64 over SSH); returns names skipped because they exist."""
    payload = json.dumps({k: base64.b64encode(v).decode() for k, v in files.items()})
    out = sh(cluster, f"python3 -c {shlex.quote(_UNPACK)} {shlex.quote(dest)} {'keep' if keep_existing else 'overwrite'}",
             input=payload, timeout=300)
    return json.loads(out.strip().splitlines()[-1])


def harness_files() -> dict[str, bytes]:
    return {name: (EVALKIT / name).read_bytes() for name in HARNESS_FILES}


def suite_status(cluster: Cluster, harness: str, suites: list[str]) -> dict:
    q = shlex.quote
    out = sh(cluster, f"python3 -I {q(harness + '/verify.py')} {q(root(cluster))} {q(cluster.image)} --status "
                      + " ".join(q(s) for s in suites))
    return json.loads(out.strip().splitlines()[-1])


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", text).strip("-") or "x"


def new_run_id(model: str, tag: str, when: datetime | None = None) -> str:
    return f"{(when or datetime.now()):%Y%m%d-%H%M%S}_{slug(model)}_{slug(tag)}"


def running_runs(cluster: Cluster) -> list[str]:
    """Run ids whose runner process is still alive."""
    if not cluster.evals_path:
        return []
    r = shlex.quote(cluster.evals_path)
    out = remote.run(
        cluster.head,
        f'for p in {r}/runs/*/pid; do [ -f "$p" ] || continue; d=$(dirname "$p"); '
        f'[ -f "$d/run.json" ] && continue; kill -0 "$(cat "$p")" 2>/dev/null && basename "$d"; done',
    )
    return out.stdout.split() if out.ok else []


def cmd_run(cluster: Cluster, args) -> int:
    evals = root(cluster)
    api = api_status(cluster)
    if not api.healthy:
        raise EvalError(f"the cluster isn't serving ({api.error or 'not healthy'}): start it first")
    if sh(cluster, f"test -e {LOCK_PATH} && cat {LOCK_PATH} || true").strip():
        raise EvalError("a spark-cluster command is changing the cluster (lock held): wait for it")
    busy = running_runs(cluster)
    if busy:
        raise EvalError(f"an evaluation is already running: {busy[0]} (spark-cluster eval stop {busy[0]})")

    suites = args.suites or DEFAULT_SUITES
    available = sh(cluster, f"ls {shlex.quote(evals)}/suites 2>/dev/null || true").split()
    if not available:
        raise EvalError(
            f"no evaluation suite at {evals}/suites on {cluster.head.name}. The suite (prompts, hidden tests, "
            "answer keys) is private and not in this repo: copy it there from a backup, or set evals_path "
            "in cluster.yml to where it lives. Nothing else in spark-cluster needs it."
        )
    missing = [s for s in suites if s not in available]
    if missing:
        raise EvalError(f"no such suite(s): {', '.join(missing)} (available: {', '.join(available)})")

    model = cluster.model_name
    served = [m for m in api.models if m != "spark"]
    entry = model_config(load_registry(), model)
    run_id = new_run_id(model, args.tag)
    config = {
        "run_id": run_id,
        "tag": args.tag,
        "evals_root": evals,
        "api_url": cluster.api_url,
        "image": cluster.image,
        "suites": suites,
        "repeat": args.repeat,
        "seed": args.seed,
        "effort": args.effort,
        "ctk": json.loads(args.ctk) if args.ctk else None,
        "budget": args.budget,
        "prompt_suffix": args.prompt_suffix,
        "fake_replies": args.fake_replies,
        "meta": {
            "model": model,
            "served": served,
            "registry_entry": entry,
            "commit": checks_mod.local_git("rev-parse", "--short", "HEAD") or "",
            "dirty": bool(checks_mod.local_git("status", "--porcelain")),
        },
    }
    files = harness_files()
    files["../config.json"] = json.dumps(config, indent=1).encode()
    run_dir = f"{evals}/runs/{run_id}"
    q = shlex.quote
    upload(cluster, run_dir + "/harness", files)
    # Only suites whose answer key passed `eval verify` (for their current contents) may grade a model.
    st = suite_status(cluster, run_dir + "/harness", suites)
    bad = {s: v["reason"] for s, v in st.items() if not v["verified"]}
    if bad:
        sh(cluster, f"rm -rf {q(run_dir)}")
        raise EvalError("not ready to grade: " + "; ".join(f"{s}: {r}" for s, r in bad.items()))
    pid = sh(
        cluster,
        # `cd x || exit` as its own statement: `cd x && cmd &` would background the whole list as a
        # subshell that holds ssh's output open until the run ends.
        f"cd {q(evals)} || exit 1; setsid nohup python3 -u -I {q(f'runs/{run_id}/harness/runner.py')} "
        f"{q(f'runs/{run_id}')} > {q(f'runs/{run_id}/runner.log')} 2>&1 < /dev/null & echo $!",
    ).strip()
    console.print(f"[bold]Started[/] {run_id} on {cluster.head.name} (pid {pid}): {', '.join(suites)}")
    console.print(f"  model {model} · {config['meta']['commit']} · {cluster.image}")
    console.print("  suites: " + ", ".join(f"{s} [{st[s]['hash']}]" for s in suites))
    if args.no_watch:
        console.print(f"  follow with: uv run spark-cluster eval watch {run_id}")
        return 0
    return watch(cluster, run_id)


# ---------------------------------------------------------------- watch


class Tracker:
    """Folds a run's events into one row per suite and repeat."""

    def __init__(self):
        self.rows: dict[tuple[str, int], dict] = {}
        self.order: list[tuple[str, int]] = []
        self.finished = None  # "done" | "stopped"
        self.started: datetime | None = None

    def feed(self, ev: dict) -> None:
        t = datetime.fromisoformat(ev["t"])
        kind = ev["event"]
        if kind == "run_start":
            self.started = t
            return
        if kind in ("run_done", "stopped"):
            self.finished = "done" if kind == "run_done" else "stopped"
            return
        key = (ev.get("suite", ""), ev.get("rep", 0))
        if kind == "suite_start":
            self.order.append(key)
            self.rows[key] = {"start": t, "end": None, "state": "starting", "detail": ""}
            return
        row = self.rows.get(key)
        if row is None:
            return
        row["last"] = t
        if kind == "suite_done":
            row["end"] = t
            if row["state"] not in ("error",):
                row["state"] = "done"
        elif kind == "sent":
            if "round" in ev:
                row["round"] = f"round {ev['round']}" + (f"/{ev['of']}" if "of" in ev else "")
            row["state"] = row.get("round", "sent")
        elif kind == "first_token":
            row["detail"] = f"first token {ev['seconds']}s"
        elif kind == "progress":
            row["state"] = f"{ev['phase']} · {row['round']}" if "round" in row else ev["phase"]
            row["detail"] = f"reasoning {ev['reasoning_chars']:,} · answer {ev['answer_chars']:,} chars"
        elif kind == "done" and "tokens" in ev:
            row["state"] = "answered"
            row["detail"] = f"{ev['tokens']:,} tokens, {ev['finish']}, {ev['code_blocks']} code block(s)"
        elif kind == "code":
            row["state"] = "grading"
            row["detail"] = f"{ev['lines']} lines, risky calls {ev['risky_calls']}"
        elif kind == "grade":
            row["result"] = f"{ev['passed']}/{ev['total']}" if ev.get("total") else ev.get("note", "0")
            row["detail"] = ev.get("call", row["detail"]) if ev.get("call") else row["detail"]
        elif kind == "stress_progress":
            row["state"] = "stress"
            row["detail"] = f"{ev['done']:,}/{ev['total']:,} cases, missed {ev['missed']}"
        elif kind == "stress":
            row["stress"] = (f"missed {ev['missed']} · impossible {ev['impossible']} · "
                             f"invalid {ev['invalid']} · errors {ev['errors']}")
        elif kind == "round":
            # Kept for the rest of the run, so a round that ran out of tokens stays visible
            s = f"round {ev['round']}: {ev['tokens']:,} tokens"
            if ev.get("finish") == "length":
                s += ", out of tokens"
            if not ev.get("implementation"):
                s += ", no code"
            elif not ev.get("tests"):
                s += ", no tests"
            row.setdefault("rounds", []).append(s)
        elif kind == "own_tests":
            rounds = row.setdefault("rounds", [])
            if rounds and rounds[-1].startswith(f"round {ev['round']}: "):
                rounds[-1] += f", own tests {ev['summary']}"
            else:
                rounds.append(f"round {ev['round']}: own tests {ev['summary']}")
        elif kind == "bench":
            row.setdefault("bench", {})[ev["what"]] = ev.get("tok_s", ev.get("seconds"))
            b = row["bench"]
            row["result"] = " / ".join(
                filter(None, [f"prose {b['prose']}" if "prose" in b else "",
                              f"code {b['code']} tok/s" if "code" in b else "",
                              f"short {b['short']}s" if "short" in b else ""]))
        elif kind == "done" and "prose_tok_s" in ev:
            notes = []
            if "draft_acceptance" in ev:
                notes.append(f"draft acceptance {ev['draft_acceptance']:.0%}")
            if ev.get("shared_load"):
                notes.append(f"SHARED: up to {ev['peak_other_requests']} other requests during measurement")
            row["detail"] = " · ".join(notes)
        elif kind == "answer":
            row["detail"] = f"{ev['question']}: {ev['status']}"
        elif kind == "done" and "answered" in ev:
            row["result"] = f"{ev['answered']}/{ev['total']} answered (ungraded)"
        elif kind == "error":
            row["state"] = "error"
            row["detail"] = ev["error"]

    def table(self, run_id: str, now: datetime | None = None) -> Table:
        t = Table(title=escape(run_id), title_justify="left")
        for col in ("suite", "state", "time", "result", "stress", "detail"):
            t.add_column(col, overflow="fold")
        for key in self.order:
            r = self.rows[key]
            end = r["end"] or now or r.get("last") or r["start"]
            name = key[0] if key[1] <= 1 and sum(k[0] == key[0] for k in self.order) == 1 else f"{key[0]} #{key[1]}"
            state = {"done": "[green]done[/]", "error": "[red]error[/]"}.get(r["state"], f"[yellow]{r['state']}[/]")
            detail = "\n".join(filter(None, [r["detail"], *r.get("rounds", [])]))
            t.add_row(escape(name), state, fmt_duration((end - r["start"]).total_seconds()),
                      escape(r.get("result", "")), escape(r.get("stress", "")), escape(detail))
        return t


def fetch_events(cluster: Cluster, run_id: str, offset: int) -> tuple[list[dict], int, bool]:
    """New events since byte offset, the new offset, and whether the runner is alive."""
    d = shlex.quote(f"{root(cluster)}/runs/{run_id}")
    out = sh(cluster, f"test -d {d} || {{ echo NO_SUCH_RUN; exit 0; }}; "
                      f"tail -c +{offset + 1} {d}/events.jsonl 2>/dev/null; printf '\\n\\x1e'; "
                      f"kill -0 \"$(cat {d}/pid 2>/dev/null)\" 2>/dev/null && echo alive || echo dead")
    if out.startswith("NO_SUCH_RUN"):
        raise EvalError(f"no run {run_id} (see: spark-cluster eval status)")
    body, _, alive = out.rpartition("\x1e")
    body = body[:-1]  # the newline printed before the separator
    complete = body[: body.rfind("\n") + 1] if "\n" in body else ""
    events = [json.loads(line) for line in complete.splitlines() if line.strip()]
    return events, offset + len(complete.encode()), alive.strip() == "alive"


def latest_run(cluster: Cluster) -> str:
    out = sh(cluster, f"ls -1 {shlex.quote(root(cluster))}/runs 2>/dev/null | tail -1").strip()
    if not out:
        raise EvalError("no evaluation runs yet")
    return out


def watch(cluster: Cluster, run_id: str | None) -> int:
    run_id = run_id or latest_run(cluster)
    tracker, offset, dead_polls = Tracker(), 0, 0
    try:
        if console.is_terminal:
            with Live(tracker.table(run_id), console=console, refresh_per_second=2) as live:
                while True:
                    events, offset, alive = fetch_events(cluster, run_id, offset)
                    for ev in events:
                        tracker.feed(ev)
                    live.update(tracker.table(run_id, datetime.now().astimezone()))
                    if tracker.finished:
                        break
                    dead_polls = 0 if alive else dead_polls + 1
                    if dead_polls >= 2:
                        break
                    time.sleep(POLL_SECONDS)
        else:
            while True:
                events, offset, alive = fetch_events(cluster, run_id, offset)
                for ev in events:
                    tracker.feed(ev)
                    console.print(json.dumps(ev), markup=False)
                if tracker.finished:
                    break
                dead_polls = 0 if alive else dead_polls + 1
                if dead_polls >= 2:
                    break
                time.sleep(POLL_SECONDS)
    except KeyboardInterrupt:
        console.print(f"Stopped watching; {run_id} keeps running. Reattach: uv run spark-cluster eval watch {run_id}")
        return 130
    if tracker.finished == "done":
        console.print(f"[green]Finished[/] {run_id}")
        report(cluster, runs=[run_id])
        return 0
    if tracker.finished == "stopped":
        console.print(f"[yellow]Stopped[/] {run_id}")
        return 1
    log = sh(cluster, f"tail -20 {shlex.quote(f'{root(cluster)}/runs/{run_id}/runner.log')} 2>/dev/null || true")
    raise EvalError(f"the runner for {run_id} exited without finishing. runner.log:\n{log}")


def cmd_watch(cluster: Cluster, args) -> int:
    return watch(cluster, args.run)


# ---------------------------------------------------------------- status / stop


def cmd_status(cluster: Cluster, args) -> int:
    r = shlex.quote(root(cluster))
    out = sh(cluster, f"""cd {r}/runs 2>/dev/null || exit 0
for d in $(ls -1 | tail -{args.last}); do
  if [ -f "$d/run.json" ]; then s=done
  elif kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null; then s=running
  elif grep -q '"stopped"' "$d/events.jsonl" 2>/dev/null; then s=stopped
  else s=dead; fi
  last=$(tail -1 "$d/events.jsonl" 2>/dev/null | cut -c1-32 | sed 's/.*"t": "//')
  echo "$d $s $last"
done""")
    rows = [line.split(maxsplit=2) for line in out.splitlines() if line.strip()]
    if not rows:
        console.print("No evaluation runs yet.")
        return 0
    t = Table(title="Evaluation runs", title_justify="left")
    for col in ("run", "state", "last event (UTC)"):
        t.add_column(col)
    style = {"done": "green", "running": "yellow", "stopped": "yellow", "dead": "red"}
    for run_id, state, *rest in rows:
        t.add_row(escape(run_id), f"[{style.get(state, 'white')}]{state}[/]", escape(rest[0] if rest else ""))
    console.print(t)
    return 0


def cmd_stop(cluster: Cluster, args) -> int:
    run_id = args.run or (running_runs(cluster) or [None])[0]
    if not run_id:
        raise EvalError("no evaluation is running")
    d = shlex.quote(f"{root(cluster)}/runs/{run_id}")
    out = sh(cluster, f"""p=$(cat {d}/pid 2>/dev/null) || {{ echo "no pid file"; exit 0; }}
if kill -0 "$p" 2>/dev/null; then kill -TERM "$p"; sleep 3; kill -0 "$p" 2>/dev/null && kill -KILL "$p"; echo stopped; else echo "not running"; fi
ids=$(docker ps -q --filter label=spark-eval={shlex.quote(run_id)}); [ -n "$ids" ] && docker kill $ids >/dev/null; true""")
    console.print(f"{run_id}: {out.strip()}")
    return 0


# ---------------------------------------------------------------- verify / init


def cmd_verify(cluster: Cluster, args) -> int:
    evals = root(cluster)
    q = shlex.quote
    upload(cluster, f"{evals}/.harness", harness_files())
    names = args.suites or sh(cluster, f"ls {q(evals)}/suites 2>/dev/null || true").split()
    if not names:
        raise EvalError(f"no suites under {evals}/suites (start with: spark-cluster eval init)")
    code = remote.run_interactive(
        cluster.head,
        f"cd {q(evals)} && python3 -u -I .harness/verify.py {q(evals)} {q(cluster.image)} " + " ".join(q(n) for n in names),
    )
    return 0 if code == 0 else 1


def public_suite_files() -> dict[str, bytes]:
    files = {}
    for p in sorted(PUBLIC_SUITES.rglob("*")):
        if p.is_file() and "__pycache__" not in p.parts:
            files[p.relative_to(PUBLIC_SUITES).as_posix()] = p.read_bytes()
    return files


def cmd_init(cluster: Cluster, args) -> int:
    """Create the suite folder from the public task specs; never overwrites anything already there."""
    evals = root(cluster)
    files = public_suite_files()
    template = (EVALKIT / "grader_template.py").read_bytes()
    for name in {k.split("/")[0] for k in files}:
        spec = json.loads(files[f"{name}/suite.json"])
        if spec["kind"] == "code":
            files[f"{name}/grader.py"] = template
    readme = (
        "# Model evaluation suite (private)\n\nCreated by `spark-cluster eval init` from the public specs in the "
        "mixvllm repo (evals/). The generated parts (hidden_tests.py, reference.py, mutants/, brute.py, "
        "stress_cases.py, expected answers) are answer keys: keep them here, never in the repo. Generate them "
        "with /setup-evals, check them with `spark-cluster eval verify`. Not backed up anywhere else.\n"
    ).encode()
    skipped = upload(cluster, f"{evals}/suites", files, keep_existing=True)
    skipped += upload(cluster, evals, {"README.md": readme}, keep_existing=True)
    sh(cluster, f"mkdir -p {shlex.quote(evals)}/runs")
    upload(cluster, f"{evals}/.harness", harness_files())
    written = len(files) + 1 - len(skipped)
    console.print(f"{evals} on {cluster.head.name}: {written} file(s) written, {len(skipped)} already there (kept)")
    names = sorted({k.split("/")[0] for k in files})
    st = suite_status(cluster, f"{evals}/.harness", names)
    t = Table(title="Suites", title_justify="left")
    for col in ("suite", "ready", "fingerprint", "detail"):
        t.add_column(col)
    for n in names:
        v = st[n]
        t.add_row(escape(n), "[green]yes[/]" if v["verified"] else "[yellow]no[/]", v["hash"], escape(v["reason"]))
    console.print(t)
    if any(not v["verified"] for v in st.values()):
        console.print("Next: run /setup-evals in Claude Code to generate and verify the missing answer keys.")
    return 0


# ---------------------------------------------------------------- results / report


def read_results(cluster: Cluster) -> list[dict]:
    text = sh(cluster, f"cat {shlex.quote(root(cluster))}/results.csv 2>/dev/null || true")
    return list(csv.DictReader(io.StringIO(text)))


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def summarize(rows: list[dict]) -> list[dict]:
    """One line per (model, tag, settings, run): each suite's results joined across repeats."""
    groups: dict[tuple, dict] = {}
    for r in rows:
        key = (r["run_id"],)
        g = groups.setdefault(key, {"run_id": r["run_id"], "model": r["model"], "tag": r["tag"],
                                    "settings": r["settings"], "when": r["finished"][:16], "suites": defaultdict(list)})
        g["suites"][r["suite"]].append(r)
    return list(groups.values())


def out_of_tokens(m: dict) -> str:
    """Fix-loop rounds cut off at max_tokens, which used up a fix round without a fix."""
    n = m.get("length_rounds")
    return f" · {n} round{'s' * (n > 1)} out of tokens" if n else ""


def cell(suite: str, rows: list[dict]) -> str:
    graded = [r for r in rows if r.get("metrics") and '"graded"' in r["metrics"]]
    if graded:
        rows = graded[-1:]
    parts = []
    for r in rows:
        m = json.loads(r["metrics"]) if r.get("metrics") else {}
        if m.get("error"):
            parts.append("error")
        elif suite == "speed":
            acc = f", accept {m['draft_acceptance']:.0%}" if "draft_acceptance" in m else ""
            shared = f" (shared: up to {m['peak_other_requests']} other requests)" if m.get("shared_load") else ""
            parts.append(f"{m.get('prose_tok_s')}/{m.get('code_tok_s')} tok/s, {m.get('short_s')}s{acc}{shared}")
        elif m.get("ungraded"):
            ans = sum(a["status"] == "answered" for a in m.get("answers", []))
            parts.append(f"{ans}/{r['total']} answered, ungraded")
        elif r.get("passed") != "" and r.get("total"):
            s = f"{r['passed']}/{r['total']}"
            if r.get("missed") not in ("", None):
                s += f" · missed {r['missed']}"
                extra = [f"{k} {r[k]}" for k in ("impossible", "invalid", "errors") if r.get(k) not in ("", "0", None)]
                if extra:
                    s += " · " + " · ".join(extra)
            if _num(r.get("seconds")) and suite not in ("tools",):
                s += f" · {fmt_duration(_num(r['seconds']))}"
            parts.append(s + out_of_tokens(m))
        elif m.get("note"):
            parts.append(m["note"] + out_of_tokens(m))
        else:
            parts.append("-")
    return "\n".join(parts)


def hash_tags(groups: list[dict], suites: list[str]) -> dict[str, dict[str, str]]:
    """Per suite: fingerprint -> letter, only for suites whose shown results come from several versions."""
    tags = {}
    for s in suites:
        hashes = []
        for g in groups:
            for r in g["suites"].get(s, []):
                h = r.get("suite_hash") or ""
                if h and h not in hashes:
                    hashes.append(h)
        if len(hashes) > 1:
            tags[s] = {h: "abcdefghijklmnopqrstuvwxyz"[i % 26] for i, h in enumerate(hashes)}
    return tags


def tagged_cell(suite: str, rows: list[dict], tags: dict[str, dict[str, str]]) -> str:
    text = cell(suite, rows)
    if suite in tags and rows:
        h = next((r.get("suite_hash") for r in reversed(rows) if r.get("suite_hash")), "")
        if h:
            text = f"[{tags[suite][h]}] " + text
    return text


def report(cluster: Cluster, runs: list[str] | None = None, models: list[str] | None = None, last: int = 12,
           suite_hashes: list[str] | None = None) -> None:
    rows = read_results(cluster)
    if runs:
        rows = [r for r in rows if r["run_id"] in runs]
    if models:
        rows = [r for r in rows if r["model"] in models]
    if suite_hashes:
        rows = [r for r in rows if r.get("suite_hash") in suite_hashes]
    if not rows:
        console.print("No results yet.")
        return
    groups = summarize(rows)[-last:]
    suites = [s for s in DEFAULT_SUITES if any(s in g["suites"] for g in groups)]
    suites += sorted({s for g in groups for s in g["suites"]} - set(suites))
    tags = hash_tags(groups, suites)
    t = Table(title="Evaluation results", title_justify="left", show_lines=True)
    t.add_column("model / tag / settings", overflow="fold")
    for s in suites:
        t.add_column(s, overflow="fold")
    default = cluster.model_name
    for g in groups:
        label = f"[bold]{escape(g['model'])}[/]" if g["model"] == default else escape(g["model"])
        label += f"\n{escape(g['tag'])} · {escape(g['settings'])}\n[dim]{escape(g['when'])}[/]"
        t.add_row(label, *[escape(tagged_cell(s, g["suites"].get(s, []), tags)) for s in suites])
    console.print(t)
    for s, letters in tags.items():
        console.print(f"[yellow]{s}: results from different suite versions are not comparable:[/] "
                      + ", ".join(f"[{v}] {h}" for h, v in letters.items()))
    console.print("[dim]Bold = the model cluster.yml currently serves. Code suites: hidden tests passed · "
                  "missed fits in the stress test (lower is better).[/]")


def cmd_report(cluster: Cluster, args) -> int:
    report(cluster, runs=args.run, models=args.model, last=args.last, suite_hashes=args.suite_hash)
    return 0


def cmd_grade(cluster: Cluster, args) -> int:
    """Record a person's or agent's grade for an ungraded suite (vision/audio answers)."""
    m = re.fullmatch(r"(\d+)/(\d+)", args.score)
    if not m:
        raise EvalError("score must look like 3/4")
    rows = [r for r in read_results(cluster) if r["run_id"] == args.run and r["suite"] == args.suite]
    if not rows:
        raise EvalError(f"no {args.suite} result in run {args.run}")
    base = dict(rows[-1])
    base.update(passed=m.group(1), total=m.group(2), finished=datetime.now().astimezone().isoformat(timespec="seconds"),
                metrics=json.dumps({"graded": True, "by": args.by, "note": args.note}, separators=(",", ":")))
    buf = io.StringIO()
    csv.DictWriter(buf, fieldnames=list(rows[-1].keys())).writerow(base)
    append = ("import fcntl, sys; f = open(sys.argv[1], 'a'); fcntl.flock(f, fcntl.LOCK_EX); "
              "f.write(sys.stdin.read()); f.close()")
    sh(cluster, f"python3 -c {shlex.quote(append)} {shlex.quote(root(cluster) + '/results.csv')}", input=buf.getvalue())
    console.print(f"Recorded {args.suite} = {args.score} for {args.run}")
    return 0


# ---------------------------------------------------------------- parser


def add_parsers(sub) -> None:
    ev = sub.add_parser("eval", help="evaluate the loaded model with the private test suite",
                        description="Evaluate the loaded model with the private test suite (cluster.yml evals_path).")
    es = ev.add_subparsers(dest="eval_command", required=True, metavar="action")

    run = es.add_parser("run", help="start an evaluation run on the head (detached) and follow it")
    run.add_argument("suites", nargs="*", help=f"suites to run (default: {' '.join(DEFAULT_SUITES)})")
    run.add_argument("--tag", default="default", help="names the run, e.g. low_effort (default: default)")
    run.add_argument("--repeat", type=int, help="repeats per suite (default: each suite's own, 2 for challenge2)")
    run.add_argument("--seed", type=int, default=1, help="seed of the first repeat; repeats use seed, seed+1, ...")
    run.add_argument("--effort", help="reasoning_effort to send (default: the server's)")
    run.add_argument("--ctk", help='chat_template_kwargs as JSON, e.g. \'{"enable_thinking": false}\'')
    run.add_argument("--budget", type=int, help="thinking_token_budget (suites that allow it)")
    run.add_argument("--prompt-suffix", help="text appended to code prompts")
    run.add_argument("--fake-replies", help=argparse.SUPPRESS)  # dry run: canned replies file in the fix_loop suite
    run.add_argument("--no-watch", action="store_true", help="start it and return")
    run.set_defaults(fn=cmd_run)

    w = es.add_parser("watch", help="follow a run's progress (default: the latest)")
    w.add_argument("run", nargs="?")
    w.set_defaults(fn=cmd_watch)

    st = es.add_parser("status", help="recent runs and their state")
    st.add_argument("--last", type=int, default=10)
    st.set_defaults(fn=cmd_status)

    sp = es.add_parser("stop", help="stop a run (default: the running one) and its sandbox containers")
    sp.add_argument("run", nargs="?")
    sp.set_defaults(fn=cmd_stop)

    rp = es.add_parser("report", help="results table from results.csv")
    rp.add_argument("--run", action="append", help="only this run (repeatable)")
    rp.add_argument("--model", action="append", help="only this registry model (repeatable)")
    rp.add_argument("--last", type=int, default=12, help="newest N runs (12)")
    rp.add_argument("--suite-hash", action="append", help="only results from this suite fingerprint (repeatable)")
    rp.set_defaults(fn=cmd_report)

    gr = es.add_parser("grade", help="record a grade for an ungraded suite (vision answers)")
    gr.add_argument("run")
    gr.add_argument("suite")
    gr.add_argument("score", help="e.g. 3/4")
    gr.add_argument("--by", default="person", help="who graded: person or agent")
    gr.add_argument("--note", default="", help="what was right or wrong")
    gr.set_defaults(fn=cmd_grade)

    vf = es.add_parser("verify", help="check suites' answer keys (reference, mutants, cross-check) before use")
    vf.add_argument("suites", nargs="*", help="suites to verify (default: all)")
    vf.set_defaults(fn=cmd_verify)

    it = es.add_parser("init", help="create the private suite folder from the public task specs (never overwrites)")
    it.set_defaults(fn=cmd_init)

