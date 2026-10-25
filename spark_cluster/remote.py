"""SSH to the nodes, with retries on connection and name-lookup errors, in parallel when asked."""

from __future__ import annotations

import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, TypeVar

from .config import Node

SSH_OPTIONS = ["-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
ATTEMPTS = 4
BACKOFF = 2.0  # seconds, doubled after each failed attempt

# ssh exits 255 for its own errors; these are worth another try (DNS lookups fail now and then).
RETRYABLE = (
    "Could not resolve hostname",
    "Name or service not known",
    "Temporary failure in name resolution",
    "Connection timed out",
    "Connection refused",
    "Connection reset",
    "Connection closed by",
    "kex_exchange_identification",
    "No route to host",
    "Network is unreachable",
)

T = TypeVar("T")


class RemoteError(Exception):
    pass


@dataclass
class Result:
    node: Node
    returncode: int
    stdout: str
    stderr: str

    @property
    def ok(self) -> bool:
        return self.returncode == 0

    def check(self, what: str) -> Result:
        if not self.ok:
            detail = (self.stderr or self.stdout).strip().splitlines()
            raise RemoteError(f"{self.node.name}: {what} failed: {detail[-1] if detail else self.returncode}")
        return self


def is_retryable(returncode: int, stderr: str) -> bool:
    return returncode == 255 and any(s in stderr for s in RETRYABLE)


def run(node: Node, command: str, *, input: str | None = None, timeout: float = 120) -> Result:
    """Run a shell command on a node and capture its output."""
    delay = BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        try:
            proc = subprocess.run(
                ["ssh", "-T", *SSH_OPTIONS, node.ssh, command],
                input=input,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return Result(node, 124, "", f"timed out after {timeout:.0f} s")
        if attempt < ATTEMPTS and is_retryable(proc.returncode, proc.stderr):
            time.sleep(delay)
            delay *= 2
            continue
        return Result(node, proc.returncode, proc.stdout, proc.stderr)
    raise AssertionError("unreachable")


def run_interactive(node: Node, command: str) -> int:
    """Run with a terminal attached (logs -f, downloads with progress bars)."""
    delay = BACKOFF
    for attempt in range(1, ATTEMPTS + 1):
        started = time.monotonic()
        tty = "-t" if sys.stdin.isatty() else "-T"
        code = subprocess.run(["ssh", tty, *SSH_OPTIONS, node.ssh, command]).returncode
        # stderr isn't captured here, so retry only quick 255s (ssh failing before the command ran)
        if code == 255 and time.monotonic() - started < 15 and attempt < ATTEMPTS:
            time.sleep(delay)
            delay *= 2
            continue
        return code
    raise AssertionError("unreachable")


def parallel(nodes: list[Node], fn: Callable[[Node], T]) -> list[T]:
    """fn(node) for every node at once; results in node order."""
    with ThreadPoolExecutor(max_workers=max(1, len(nodes))) as pool:
        return list(pool.map(fn, nodes))
