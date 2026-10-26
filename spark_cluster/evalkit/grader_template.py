"""Generic grader for a generated suite: `spark-cluster eval init` copies it to suites/<name>/grader.py.

The task-specific parts live next to it (written by /setup-evals, checked by `spark-cluster eval verify`):

hidden_tests.py
    FUNCTION = "evaluate"                       # the function the model must write
    CASES = [(description, args, expected), ...] # args is a list of positional arguments
    TIMEOUT = 10                                 # optional, seconds per call
    def judge(args, outcome, expected) -> str | None   # optional; None means pass
stress_cases.py (optional)
    def generate(seed) -> list[args]
    def expected(reference_module, args)         # JSON-able, e.g. whether a fit exists
    def judge_stress(args, outcome, expected) -> "ok" | "missed" | "impossible" | "invalid" | "error"

The solution runs in its own sandbox (sandbox.Solution): it only ever sees one call's arguments, and
values cross as JSON, so tuples arrive as lists and judges must accept both.
"""

import importlib.util
import json
from pathlib import Path

import hidden_tests as H
from sandbox import LoadError, Solution

HERE = Path(__file__).resolve().parent
TIMEOUT = getattr(H, "TIMEOUT", 10)
try:
    import stress_cases as S
except ImportError:
    S = None


def describe(out):
    if out.kind == "timeout":
        return f"timed out ({TIMEOUT} s)"
    if out.kind == "crash":
        return f"crashed the interpreter: {out.msg[-150:]}"
    if out.kind == "raised":
        return f"raised {out.exc}: {out.msg}"
    return None


def default_judge(args, out, expected):
    if out.kind != "ok":
        return describe(out)
    return None if out.value == expected else f"returned {out.value!r:.80}, expected {expected!r:.80}"


judge = getattr(H, "judge", default_judge)


def hidden(sol):
    passed, failures = 0, []
    for desc, args, expected in H.CASES:
        out = sol.call(H.FUNCTION, list(args), timeout=TIMEOUT)
        try:
            outcome = judge(args, out, expected)
        except Exception as e:  # malformed return values the judge didn't anticipate
            outcome = f"invalid result: {type(e).__name__}: {e}"
        if outcome is None:
            passed += 1
        else:
            failures.append(f"{desc}: {outcome}")
    return passed, failures


def stress(sol, emit):
    cases = json.loads((HERE / "stress_expected.json").read_text())
    tally = {"missed": 0, "impossible": 0, "invalid": 0, "errors": 0}
    for i, (args, expected) in enumerate(cases, 1):
        out = sol.call(H.FUNCTION, list(args), timeout=TIMEOUT)
        if out.kind != "ok":
            verdict = "error"
        else:
            try:
                verdict = S.judge_stress(args, out, expected)
            except Exception:
                verdict = "invalid"
        if verdict != "ok":
            tally["errors" if verdict == "error" else verdict] += 1
        if i % 500 == 0:
            emit("stress_progress", done=i, total=len(cases), missed=tally["missed"])
    return {"cases": len(cases), **tally}


def grade(solution_path, image, emit, label="spark-eval", with_stress=True):
    try:
        sol = Solution(image, solution_path, label)
        sol.start()
    except LoadError as e:
        return {"passed": 0, "total": len(H.CASES), "failures": [f"could not load the solution: {e}"]}
    try:
        passed, failures = hidden(sol)
        emit("grade", passed=passed, total=len(H.CASES))
        result = {"passed": passed, "total": len(H.CASES), "failures": failures}
        if with_stress and S is not None and (HERE / "stress_expected.json").exists():
            result["stress"] = stress(sol, emit)
            emit("stress", **result["stress"])
    finally:
        sol.stop()
    return result


def build_expected(reference_path, out_path, seed=1):
    """stress_expected.json: every generated case with the reference's expected value."""
    spec = importlib.util.spec_from_file_location("reference", reference_path)
    ref = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ref)
    cases = [[list(args), S.expected(ref, list(args))] for args in S.generate(seed)]
    Path(out_path).write_text(json.dumps(cases))
    return len(cases)
