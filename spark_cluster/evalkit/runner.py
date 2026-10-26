"""Evaluation runner: runs on the head, detached, one evaluation run per process. Standard library only.

Started by `spark-cluster eval run`, which copies this folder into <evals>/runs/<id>/harness/ and
writes <evals>/runs/<id>/config.json. Each suite lives in <evals>/suites/<name>/ with a suite.json
that says what kind it is:

  speed     generation speed (prose, code, short answer) and speculative-decoding acceptance
  code      one prompt, extract the last ```python block, grade it with the suite's grader.py
  fix_loop  the model writes code and its own tests; failures go back to it for a few rounds
  media     image / video / audio questions; answers are recorded for grading by a person or agent
  tools     one tool-calling request, checked for the expected structured call

Progress goes to runs/<id>/events.jsonl (one JSON object per line), which `spark-cluster eval
watch` follows. Results go to runs/<id>/run.json and one row per suite and repeat in
<evals>/results.csv.
"""

from __future__ import annotations

import base64
import csv
import fcntl
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import sandbox  # noqa: E402

RUN_DIR = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else None
RESULT_FIELDS = [
    "run_id", "finished", "model", "tag", "suite", "rep", "seed", "settings", "passed", "total",
    "missed", "impossible", "invalid", "errors", "seconds", "tokens", "tok_s", "finish", "metrics",
    "commit", "image", "suite_hash",
]
FINGERPRINT_SKIP = {"verification.json"}
RISKY = re.compile(r"eval\(|exec\(|compile\(|\bast\b|\bos\b|subprocess|socket|open\(")
PROGRESS_EVERY = 15


def fingerprint(*dirs: Path) -> str:
    """Short hash of everything that decides a suite's scores: every file's path and contents
    (prompt, hidden tests, grader, stress cases...), except caches and the verification record.
    Results with different fingerprints come from different suites and aren't comparable."""
    h = hashlib.sha256()
    for d in dirs:
        d = Path(d)
        for p in sorted(d.rglob("*")):
            rel = p.relative_to(d)
            if (not p.is_file() or p.name in FINGERPRINT_SKIP or "__pycache__" in rel.parts
                    or p.suffix == ".pyc"):
                continue
            h.update(f"{d.name}/{rel.as_posix()}\0".encode())
            h.update(p.read_bytes())
            h.update(b"\0")
    return h.hexdigest()[:12]


def write_results(path: Path, rows: list[dict]) -> None:
    """Append rows to results.csv under a lock; first rewrite older files to the current columns."""
    with open(path, "a+", newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.seek(0)
        old = list(csv.DictReader(f))
        f.seek(0)
        header = next(csv.reader(f), None)
        if header is not None and header != RESULT_FIELDS:
            f.seek(0)
            f.truncate()
            w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore", restval="")
            w.writeheader()
            w.writerows(old)
        f.seek(0, os.SEEK_END)
        w = csv.DictWriter(f, fieldnames=RESULT_FIELDS, extrasaction="ignore", restval="")
        if f.tell() == 0:
            w.writeheader()
        w.writerows(rows)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Run:
    def __init__(self, run_dir: Path):
        self.dir = run_dir
        self.cfg = json.loads((run_dir / "config.json").read_text())
        self.root = Path(self.cfg["evals_root"])
        self.api = self.cfg["api_url"].rstrip("/")
        self.image = self.cfg["image"]
        self.label = f"spark-eval={self.cfg['run_id']}"
        self.events = open(run_dir / "events.jsonl", "a")
        self.rows: list[dict] = []
        self.hashes: dict[str, str] = {}

    # -- plumbing

    def emit(self, event: str, suite: str = "", rep: int = 0, **fields) -> None:
        rec = {"t": now(), "event": event, "suite": suite, "rep": rep, **fields}
        self.events.write(json.dumps(rec) + "\n")
        self.events.flush()

    def request_extras(self) -> dict:
        extra = {}
        if self.cfg.get("effort"):
            extra["reasoning_effort"] = self.cfg["effort"]
        if self.cfg.get("ctk"):
            extra["chat_template_kwargs"] = self.cfg["ctk"]
        return extra

    def settings_text(self) -> str:
        parts = [f"effort={self.cfg.get('effort') or 'default'}"]
        if self.cfg.get("ctk"):
            parts.append(f"ctk={json.dumps(self.cfg['ctk'], separators=(',', ':'))}")
        if self.cfg.get("budget"):
            parts.append(f"budget={self.cfg['budget']}")
        return " ".join(parts)

    def post(self, body: dict, timeout: float = 900) -> dict:
        req = urllib.request.Request(f"{self.api}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.load(r)

    def load(self) -> int | None:
        """Requests the server is running or queueing right now (all users), from /metrics."""
        try:
            with urllib.request.urlopen(f"{self.api}/metrics", timeout=5) as r:
                text = r.read().decode()
        except Exception:
            return None
        found = re.findall(r"^vllm:num_requests_(?:running|waiting)\{[^}]*\} ([0-9.e+]+)$", text, re.M)
        return int(sum(float(v) for v in found)) if found else None

    def stream(self, messages: list, max_tokens: int, seed: int, suite: str, rep: int, live_path: Path,
               extra: dict | None = None, timeout: float = 3600) -> dict:
        """One streamed chat turn with progress events; returns content, reasoning, usage, timing."""
        body = {"model": "spark", "max_tokens": max_tokens, "stream": True, "seed": seed,
                "stream_options": {"include_usage": True}, "messages": messages,
                **self.request_extras(), **(extra or {})}
        req = urllib.request.Request(f"{self.api}/v1/chat/completions", json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
        t0 = last = time.time()
        first = None
        reasoning, content, usage, model, finish = [], [], None, None, None
        peak_others = 0
        with open(live_path, "w") as live, urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                model = ev.get("model", model)
                usage = ev.get("usage") or usage
                for ch in ev.get("choices", []):
                    d = ch.get("delta", {})
                    r = d.get("reasoning") or d.get("reasoning_content")
                    if r:
                        reasoning.append(r)
                        live.write(r)
                        live.flush()
                    if d.get("content"):
                        content.append(d["content"])
                    finish = ch.get("finish_reason") or finish
                if first is None and (reasoning or content):
                    first = time.time() - t0
                    self.emit("first_token", suite, rep, seconds=round(first, 2))
                if time.time() - last >= PROGRESS_EVERY:
                    last = time.time()
                    n = self.load()
                    if n is not None:
                        peak_others = max(peak_others, n - 1)  # minus this request
                    self.emit("progress", suite, rep, seconds=round(last - t0),
                              phase="writing" if content else "thinking",
                              reasoning_chars=sum(map(len, reasoning)), answer_chars=sum(map(len, content)),
                              other_requests=None if n is None else max(n - 1, 0))
        secs = time.time() - t0
        tokens = (usage or {}).get("completion_tokens", 0)
        return {"model": model, "finish": finish, "usage": usage, "seconds": secs, "tokens": tokens,
                "first_token": first, "reasoning": "".join(reasoning), "content": "".join(content),
                "peak_other_requests": peak_others}

    def row(self, suite: str, rep: int, seed, **fields) -> dict:
        meta = self.cfg["meta"]
        r = {"run_id": self.cfg["run_id"], "finished": now(), "model": meta.get("model", ""),
             "tag": self.cfg.get("tag", ""), "suite": suite, "rep": rep, "seed": seed if seed is not None else "",
             "settings": self.settings_text(), "commit": meta.get("commit", ""), "image": self.image,
             "suite_hash": self.hashes.get(suite, "")}
        metrics = fields.pop("metrics", None)
        r.update(fields)
        r["metrics"] = json.dumps(metrics, separators=(",", ":")) if metrics else ""
        self.rows.append(r)
        return r

    def load_grader(self, suite: str):
        d = self.root / "suites" / suite
        sys.path.insert(0, str(d))
        try:
            spec = importlib.util.spec_from_file_location(f"grader_{suite}", d / "grader.py")
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        finally:
            sys.path.remove(str(d))
            for m in ("hidden_tests", "stress_cases"):  # each grader imports its own
                sys.modules.pop(m, None)
        return mod

    # -- suite kinds

    def run_speed(self, name: str, spec: dict) -> None:
        def metrics() -> dict:
            try:
                with urllib.request.urlopen(f"{self.api}/metrics", timeout=10) as r:
                    text = r.read().decode()
            except Exception:
                return {}
            out = {}
            for key in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total"):
                m = re.search(rf"^vllm:{key}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
                if m:
                    out[key] = float(m.group(1))
            return out

        others = []  # other users' requests, sampled while each measurement runs

        def timed(content: str, max_tokens: int, **extra):
            body = {"model": "spark", "max_tokens": max_tokens, "seed": 1,
                    "messages": [{"role": "user", "content": content}], **extra}
            done = threading.Event()

            def sample():
                while not done.wait(0.5):
                    n = self.load()
                    if n is not None:
                        others.append(max(n - 1, 0))

            sampler = threading.Thread(target=sample, daemon=True)
            sampler.start()
            t = time.time()
            try:
                r = self.post(body)
            finally:
                done.set()
                sampler.join()
            return r, time.time() - t

        # Other users' requests share the GPUs and lower everyone's tok/s; the numbers are labelled
        # as shared when that happened, rather than waiting for an idle server.
        self.emit("sent", name, 1)
        before = metrics()
        forced = spec.get("forced_tokens", 512)
        r, dt = timed(spec["prose"], forced, ignore_eos=True)
        prose = r["usage"]["completion_tokens"] / dt
        self.emit("bench", name, 1, what="prose", tok_s=round(prose, 1))
        r, dt = timed(spec["code"], forced, ignore_eos=True)
        code = r["usage"]["completion_tokens"] / dt
        self.emit("bench", name, 1, what="code", tok_s=round(code, 1))
        after = metrics()
        r, dt = timed(spec["short"], 4000, **self.request_extras())
        m = r["choices"][0]["message"]
        answer = (m.get("content") or "").strip().replace("\n", " ")
        self.emit("bench", name, 1, what="short", seconds=round(dt, 1), answer=answer[:140])
        out = {"prose_tok_s": round(prose, 1), "code_tok_s": round(code, 1), "short_s": round(dt, 2),
               "short_tokens": r["usage"]["completion_tokens"], "short_answer": answer[:300],
               "peak_other_requests": max(others, default=0)}
        if out["peak_other_requests"]:
            out["shared_load"] = True  # not comparable with idle-server numbers
        drafted = after.get("spec_decode_num_draft_tokens_total", 0) - before.get("spec_decode_num_draft_tokens_total", 0)
        if drafted > 0:
            accepted = after["spec_decode_num_accepted_tokens_total"] - before["spec_decode_num_accepted_tokens_total"]
            out["draft_acceptance"] = round(accepted / drafted, 3)
        self.row(name, 1, 1, tok_s=out["code_tok_s"], seconds=out["short_s"], metrics=out)
        self.emit("done", name, 1, **{k: v for k, v in out.items() if k != "short_answer"})

    def run_code(self, name: str, spec: dict, rep: int, seed: int) -> None:
        d = self.root / "suites" / name
        prompt = (d / spec["prompt"]).read_text()
        extra = {}
        if self.cfg.get("budget") and spec.get("allow_budget"):
            extra["thinking_token_budget"] = int(self.cfg["budget"])
        if self.cfg.get("prompt_suffix"):
            prompt += "\n\n" + self.cfg["prompt_suffix"]
        stem = self.dir / f"{name}_r{rep}"
        self.emit("sent", name, rep, seed=seed, max_tokens=spec["max_tokens"])
        resp = self.stream([{"role": "user", "content": prompt}], spec["max_tokens"], seed, name, rep,
                           Path(f"{stem}_reasoning.txt"), extra)
        Path(f"{stem}_response.json").write_text(json.dumps({k: resp[k] for k in (
            "model", "finish", "usage", "seconds", "first_token", "reasoning", "content")}, indent=1))
        blocks = re.findall(r"```python\n(.*?)```", resp["content"], re.S)
        tok_s = round(resp["tokens"] / resp["seconds"], 1) if resp["seconds"] else None
        self.emit("done", name, rep, seconds=round(resp["seconds"]), tokens=resp["tokens"],
                  finish=resp["finish"], code_blocks=len(blocks))
        base = dict(seconds=round(resp["seconds"], 1), tokens=resp["tokens"], tok_s=tok_s, finish=resp["finish"])
        if not blocks:
            self.emit("grade", name, rep, passed=0, total=None, note="no python code block")
            self.row(name, rep, seed, passed=0, metrics={"note": "no python code block"}, **base)
            return
        solution = Path(f"{stem}_solution.py")
        solution.write_text(blocks[-1])
        risky = len(RISKY.findall(blocks[-1]))
        self.emit("code", name, rep, lines=blocks[-1].count("\n") + 1, risky_calls=risky)
        grader = self.load_grader(name)
        result = grader.grade(solution, self.image,
                              lambda event, **f: self.emit(event, name, rep, **f), self.label)
        Path(f"{stem}_grade.json").write_text(json.dumps(result, indent=1))
        stress = result.get("stress") or {}
        self.row(name, rep, seed, passed=result["passed"], total=result["total"],
                 missed=stress.get("missed", ""), impossible=stress.get("impossible", ""),
                 invalid=stress.get("invalid", ""), errors=stress.get("errors", ""),
                 metrics={"risky_calls": risky, "failures": result["failures"][:5],
                          "peak_other_requests": resp["peak_other_requests"]}, **base)

    def run_fix_loop(self, name: str, spec: dict, rep: int, seed: int) -> None:
        d = self.root / "suites" / name
        grader = self.load_grader(spec["grader_suite"])
        own_runner = d / spec["own_tests_runner"]
        rounds, cap = spec.get("rounds", 3), spec.get("time_cap", 1800)
        messages = [{"role": "user", "content": (d / spec["prompt"]).read_text()}]
        start = time.time()
        impl_path, tokens, own_summary, n_own = None, 0, "", 0
        rounds_used, length_rounds = 0, 0
        quiet = lambda event, **f: None  # noqa: E731
        # Dry run: canned replies instead of the model (checks the loop and graders, no GPU needed)
        fake = json.loads((d / self.cfg["fake_replies"]).read_text()) if self.cfg.get("fake_replies") else None
        for rnd in range(1, rounds + 2):
            if time.time() - start > cap:
                self.emit("stop", name, rep, reason=f"time cap {cap}s")
                break
            rounds_used = rnd
            self.emit("sent", name, rep, round=rnd, of=rounds + 1, seed=seed)
            if fake is not None:
                resp = {"content": fake[min(rnd - 1, len(fake) - 1)], "seconds": 0.0, "tokens": 0, "finish": "stop"}
            else:
                resp = self.stream(messages, spec["max_tokens"], seed, name, rep,
                                   self.dir / f"{name}_r{rep}_round{rnd}_reasoning.txt")
            tokens += resp["tokens"]
            reply = resp["content"]
            blocks = re.findall(r"```python\n(.*?)```", reply, re.S)
            impl = next((b for b in blocks if "def test_" not in b), None)
            tests = next((b for b in reversed(blocks) if "def test_" in b), None)
            messages.append({"role": "assistant", "content": reply})
            # A round cut off at max_tokens (usually still thinking) uses up a fix round without a fix
            length_rounds += resp["finish"] == "length"
            self.emit("round", name, rep, round=rnd, seconds=round(resp["seconds"]), tokens=resp["tokens"],
                      finish=resp["finish"], implementation=bool(impl), tests=bool(tests))
            if impl:
                impl_path = self.dir / f"{name}_r{rep}_round{rnd}_solution.py"
                impl_path.write_text(impl)
                h = grader.grade(impl_path, self.image, quiet, self.label, with_stress=False)
                self.emit("hidden_report_only", name, rep, round=rnd, passed=h["passed"], total=h["total"])
            if not (impl and tests):
                feedback = ("Your reply must contain two ```python blocks: the implementation, then `def test_*` "
                            "unit tests. Please reply with both.")
            else:
                tests_path = self.dir / f"{name}_r{rep}_round{rnd}_tests.py"
                tests_path.write_text(tests)
                own = sandbox.run_script(
                    self.image,
                    {"/s/run_own_tests.py": own_runner, "/s/solution.py": impl_path, "/s/tests.py": tests_path},
                    ["/s/run_own_tests.py", "/s/solution.py", "/s/tests.py"], label=self.label)
                own_summary = next((l for l in own.splitlines() if l.startswith("OWN_SUMMARY")), "OWN_SUMMARY ?")
                self.emit("own_tests", name, rep, round=rnd, summary=own_summary.split(" ", 1)[1])
                m = re.match(r"OWN_SUMMARY (\d+)/(\d+)", own_summary)
                if m:
                    n_own = int(m.group(2))
                    if m.group(1) == m.group(2) and n_own:
                        break
                fails = "\n".join(l for l in own.splitlines() if not l.startswith("PASS "))
                feedback = ("Your tests failed when run against your implementation:\n\n```\n" + fails[:3000] +
                            "\n```\n\nFix the implementation, or a test if the test itself is wrong. "
                            "Reply with both ```python blocks again (implementation, then tests).")
            if rnd == rounds + 1:
                self.emit("stop", name, rep, reason="fix rounds exhausted")
                break
            messages.append({"role": "user", "content": feedback})
        (self.dir / f"{name}_r{rep}_conversation.json").write_text(json.dumps(messages, indent=1))
        total_s = round(time.time() - start, 1)
        if not impl_path:
            self.row(name, rep, seed, passed=0, seconds=total_s, tokens=tokens,
                     metrics={"note": "no implementation produced", "rounds": rounds_used,
                              "length_rounds": length_rounds})
            self.emit("grade", name, rep, passed=0, total=None, note="no implementation")
            return
        result = grader.grade(impl_path, self.image, lambda event, **f: self.emit(event, name, rep, **f), self.label)
        stress = result.get("stress") or {}
        self.row(name, rep, seed, passed=result["passed"], total=result["total"], missed=stress.get("missed", ""),
                 impossible=stress.get("impossible", ""), invalid=stress.get("invalid", ""),
                 errors=stress.get("errors", ""), seconds=total_s, tokens=tokens,
                 metrics={"rounds": rounds_used, "length_rounds": length_rounds, "own_tests": n_own,
                          "own_summary": own_summary.split(" ", 1)[-1]})

    def run_media(self, name: str, spec: dict) -> None:
        d = self.root / "suites" / name
        answers = []
        for q in spec["questions"]:
            data = base64.b64encode((d / q["file"]).read_bytes()).decode()
            mime = q["mime"]
            if q["type"] == "image":
                part = {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}
            elif q["type"] == "video":
                part = {"type": "video_url", "video_url": {"url": f"data:{mime};base64,{data}"}}
            else:
                part = {"type": "input_audio", "input_audio": {"data": data, "format": q.get("format", "wav")}}
            body = {"model": "spark", "max_tokens": 4000, "seed": 1,
                    "messages": [{"role": "user", "content": [part, {"type": "text", "text": q["question"]}]}]}
            self.emit("sent", name, 1, question=q["label"])
            t = time.time()
            try:
                r = self.post(body, timeout=600)
                answer = (r["choices"][0]["message"].get("content") or "").strip()
                status = "answered"
            except urllib.error.HTTPError as e:
                try:
                    msg = json.load(e).get("error", {}).get("message", "")
                except Exception:
                    msg = ""
                answer, status = f"HTTP {e.code} {msg[:200]}", "rejected"
            answers.append({"label": q["label"], "status": status, "seconds": round(time.time() - t, 1),
                            "answer": answer, "expected": q.get("expected", "")})
            self.emit("answer", name, 1, question=q["label"], status=status, answer=answer[:160])
        (self.dir / f"{name}_answers.json").write_text(json.dumps(answers, indent=1))
        self.row(name, 1, 1, total=len(answers),
                 metrics={"ungraded": True, "answers": [{k: a[k] for k in ("label", "status", "answer")} for a in answers]})
        self.emit("done", name, 1, answered=sum(a["status"] == "answered" for a in answers), total=len(answers))

    def run_tools(self, name: str, spec: dict) -> None:
        body = {"model": "spark", "max_tokens": 2000, "seed": 1, **self.request_extras(), **spec["request"]}
        self.emit("sent", name, 1)
        t = time.time()
        r = self.post(body)
        m = r["choices"][0]["message"]
        calls = m.get("tool_calls") or []
        want = spec["expect"]
        ok, detail = False, "no structured tool_calls"
        if calls:
            fn = calls[0].get("function", {})
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = None
            ok = fn.get("name") == want["name"] and args == want["arguments"]
            detail = f"{fn.get('name')}({fn.get('arguments')})"
        elif m.get("content") and want["name"] in m["content"]:
            detail = "call came back as plain text: check the registry's tool-call-parser"
        self.row(name, 1, 1, passed=int(ok), total=1, seconds=round(time.time() - t, 1), metrics={"call": detail})
        self.emit("grade", name, 1, passed=int(ok), total=1, call=detail)

    # -- driver

    def results_csv(self) -> None:
        write_results(self.root / "results.csv", self.rows)

    def suite_dirs(self, name: str) -> list[Path]:
        """The folders that decide a suite's scores (challenge3 is graded by challenge2's grader)."""
        spec = json.loads((self.root / "suites" / name / "suite.json").read_text())
        dirs = [self.root / "suites" / name]
        if spec.get("grader_suite"):
            dirs.append(self.root / "suites" / spec["grader_suite"])
        return dirs

    def main(self) -> None:
        self.hashes = {name: fingerprint(*self.suite_dirs(name)) for name in self.cfg["suites"]}
        self.emit("run_start", pid=os.getpid(), suites=self.cfg["suites"], meta=self.cfg["meta"],
                  suite_hashes=self.hashes)
        for name in self.cfg["suites"]:
            spec = json.loads((self.root / "suites" / name / "suite.json").read_text())
            kind = spec["kind"]
            reps = self.cfg.get("repeat") or spec.get("repeat", 1)
            if kind in ("speed", "media", "tools"):
                reps = 1
            for rep in range(1, reps + 1):
                seed = self.cfg.get("seed", 1) + rep - 1
                self.emit("suite_start", name, rep, kind=kind, suite_hash=self.hashes[name])
                try:
                    if kind == "speed":
                        self.run_speed(name, spec)
                    elif kind == "code":
                        self.run_code(name, spec, rep, seed)
                    elif kind == "fix_loop":
                        self.run_fix_loop(name, spec, rep, seed)
                    elif kind == "media":
                        self.run_media(name, spec)
                    elif kind == "tools":
                        self.run_tools(name, spec)
                    else:
                        raise ValueError(f"unknown suite kind {kind!r}")
                except Exception as e:
                    self.emit("error", name, rep, error=f"{type(e).__name__}: {e}"[:500])
                    self.row(name, rep, seed if kind in ("code", "fix_loop") else None,
                             metrics={"error": f"{type(e).__name__}: {e}"[:500]})
                self.emit("suite_done", name, rep)
        if not self.cfg.get("fake_replies"):  # dry runs test the harness, not a model
            self.results_csv()
        (self.dir / "run.json").write_text(json.dumps({"config": self.cfg, "results": self.rows}, indent=1))
        self.emit("run_done", rows=len(self.rows))


def _terminate(signum, frame):
    # Stop this run's sandbox containers too, then exit.
    run_id = json.loads((RUN_DIR / "config.json").read_text())["run_id"]
    ids = subprocess.run(["docker", "ps", "-q", "--filter", f"label=spark-eval={run_id}"],
                         capture_output=True, text=True).stdout.split()
    if ids:
        subprocess.run(["docker", "kill", *ids], capture_output=True)
    with open(RUN_DIR / "events.jsonl", "a") as f:
        f.write(json.dumps({"t": now(), "event": "stopped", "suite": "", "rep": 0}) + "\n")
    os._exit(143)


if __name__ == "__main__":
    signal.signal(signal.SIGTERM, _terminate)
    (RUN_DIR / "pid").write_text(str(os.getpid()))
    Run(RUN_DIR).main()
