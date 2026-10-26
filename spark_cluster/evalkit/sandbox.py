"""Run untrusted model code in a locked-down container, apart from the graders and answer keys.

Two ways in:
- Solution(...).call(fn, args): a long-lived container running shim.py, for graders that send
  test cases one at a time and check the results themselves.
- run_script(...): a one-shot container for scripts that run model code end to end (the model's
  own unit tests in challenge 3). Only the files passed in are mounted.
"""

from __future__ import annotations

import json
import os
import selectors
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

SHIM = Path(__file__).with_name("shim.py")
LOCKDOWN = [
    "--network", "none", "--read-only", "--cap-drop", "ALL", "--user", "65534:65534",
    "--memory", "2g", "--pids-limit", "64", "--security-opt", "no-new-privileges",
]
GRACE = 5.0  # seconds past a call's own timeout before the container is killed


class Other:
    """A return value that isn't plain data (a set, a custom object...), kept for error messages."""

    def __init__(self, type_name: str, text: str):
        self.type_name, self.text = type_name, text

    def __repr__(self) -> str:
        return self.text


def decode(v):
    if isinstance(v, dict):
        if "__fraction__" in v:
            return Fraction(*v["__fraction__"])
        if "__other__" in v:
            return Other(v["__other__"], v.get("repr", ""))
    if isinstance(v, list):
        return [decode(x) for x in v]
    return v


@dataclass
class Outcome:
    kind: str  # "ok", "raised", "timeout", "crash"
    value: object = None
    exc: str = ""
    mro: tuple = ()
    msg: str = ""

    def raised(self, name: str) -> bool:
        return self.kind == "raised" and name in self.mro


class LoadError(Exception):
    pass


class Solution:
    """The model's solution in its own container; graders call its functions with test arguments."""

    def __init__(self, image: str, solution: Path, label: str = "spark-eval"):
        self.image, self.solution, self.label = image, Path(solution).resolve(), label
        self.proc = None
        self.name = None
        self.stderr_file = tempfile.TemporaryFile()
        self.restarts = 0

    # -- lifecycle

    def start(self) -> None:
        self.name = f"spark-eval-{uuid.uuid4().hex[:12]}"
        cmd = ["docker", "run", "-i", "--rm", "--name", self.name, "--label", self.label, *LOCKDOWN,
               "-v", f"{self.solution}:/s/solution.py:ro", "-v", f"{SHIM}:/s/shim.py:ro",
               "--entrypoint", "python3", self.image, "-I", "/s/shim.py", "/s/solution.py"]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.stderr_file, bufsize=0)
        self._buf = b""
        first = self._read(60)
        if first is None:
            self.stop()
            raise LoadError("solution container did not start within 60 s")
        if "load_error" in first:
            self.stop()
            raise LoadError(first["load_error"])

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            subprocess.run(["docker", "kill", self.name], capture_output=True)
            try:
                self.proc.wait(10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *exc):
        self.stop()

    # -- calls

    def _read(self, timeout: float):
        """Next protocol message, None on timeout, {"crash": True} if the container exited."""
        fd = self.proc.stdout.fileno()
        sel = selectors.DefaultSelector()
        sel.register(fd, selectors.EVENT_READ)
        deadline = time.monotonic() + timeout
        try:
            while True:
                while b"\n" in self._buf:
                    line, self._buf = self._buf.split(b"\n", 1)
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
                left = deadline - time.monotonic()
                if left <= 0 or not sel.select(left):
                    return None
                chunk = os.read(fd, 65536)
                if not chunk:
                    return {"crash": True}
                self._buf += chunk
        finally:
            sel.close()

    def call(self, fn: str, args: list, timeout: int = 10) -> Outcome:
        if self.proc is None:
            self.start()
        try:
            self.proc.stdin.write((json.dumps({"fn": fn, "args": args, "timeout": timeout}) + "\n").encode())
        except (BrokenPipeError, OSError):
            reply = {"crash": True}
        else:
            reply = self._read(timeout + GRACE)
        if reply is None or "crash" in reply:
            # Hung past its own alarm, or the process died: start fresh for the next case.
            self.stop()
            self.restarts += 1
            return Outcome("timeout") if reply is None else Outcome("crash", msg=self.stderr_tail())
        if "ok" in reply:
            return Outcome("ok", decode(reply["ok"]))
        if reply.get("timeout"):
            return Outcome("timeout")
        return Outcome("raised", exc=reply["raised"], mro=tuple(reply.get("mro", ())), msg=reply.get("msg", ""))

    def stderr_tail(self, limit: int = 600) -> str:
        self.stderr_file.seek(0, os.SEEK_END)
        size = self.stderr_file.tell()
        self.stderr_file.seek(max(0, size - limit))
        return self.stderr_file.read().decode(errors="replace")


def run_script(image: str, files: dict[str, Path], argv: list[str], timeout: float = 600,
               label: str = "spark-eval") -> str:
    """Run `python3 -I <argv>` in a one-shot locked-down container with only `files` mounted
    (container path -> host path). Returns stdout+stderr without blank lines."""
    name = f"spark-eval-{uuid.uuid4().hex[:12]}"
    mounts = []
    for inside, host in files.items():
        mounts += ["-v", f"{Path(host).resolve()}:{inside}:ro"]
    cmd = ["docker", "run", "--rm", "--name", name, "--label", label, *LOCKDOWN, *mounts,
           "--entrypoint", "python3", image, "-I", *argv]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = r.stdout + r.stderr
    except subprocess.TimeoutExpired:
        subprocess.run(["docker", "kill", name], capture_output=True)
        out = "TIMEOUT: sandbox run exceeded its time limit"
    return "\n".join(line for line in out.splitlines() if line.strip())
