"""Check that a suite's answer key can be trusted before it grades any model. Standard library only.

Run on the head by `spark-cluster eval verify` (and `--status` by `eval run`):
    python3 -I verify.py <evals root> <image> <suite>... [--status]

A code suite passes only if:
  1. its reference solution scores 100% (hidden tests, and a clean stress test if it has one)
  2. grading the reference twice gives identical results
  3. every mutant in mutants/ (deliberately broken references) scores below 100%;
     generated suites need at least MIN_MUTANTS
  4. the reference agrees with brute.py (an independent, simpler solver) on the stress cases it
     can handle, at least MIN_CROSS of them (suites with stress_cases.py and brute.py)
  5. generated suites have at least MIN_CASES hidden cases, and every category listed in SPEC.md
     ("- [category-id] ...") is named in hidden_tests.py

Suites marked "origin": "curated" in suite.json (hand-validated against real model solutions) skip
the mutant minimum, the cross-check and the shape check unless they provide those files.
The result goes to suites/<name>/verification.json, tied to the suite's fingerprint.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from runner import fingerprint  # noqa: E402

MIN_MUTANTS = 5
MIN_CROSS = 500
MIN_CASES = 15
GRADED_KINDS = ("code",)


def say(msg: str) -> None:
    print(msg, flush=True)


def load(path: Path, name: str, extra_path: Path | None = None):
    if extra_path:
        sys.path.insert(0, str(extra_path))
    try:
        for m in ("hidden_tests", "stress_cases", name):
            sys.modules.pop(m, None)
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        if extra_path:
            sys.path.remove(str(extra_path))


def perfect(result: dict) -> bool:
    s = result.get("stress") or {}
    return result["passed"] == result["total"] and not any(s.get(k) for k in ("missed", "impossible", "invalid", "errors"))


def summary(result: dict) -> str:
    s = result.get("stress")
    out = f"{result['passed']}/{result['total']}"
    if s:
        out += f", stress: missed {s['missed']}, impossible {s['impossible']}, invalid {s['invalid']}, errors {s['errors']}"
    return out


def suite_spec(root: Path, name: str) -> dict:
    return json.loads((root / "suites" / name / "suite.json").read_text())


def suite_dirs(root: Path, name: str) -> list[Path]:
    spec = suite_spec(root, name)
    dirs = [root / "suites" / name]
    if spec.get("grader_suite"):
        dirs.append(root / "suites" / spec["grader_suite"])
    return dirs


def status(root: Path, name: str) -> dict:
    """Whether a suite may be used: answer-key suites need a passing verification for their current contents."""
    d = root / "suites" / name
    if not (d / "suite.json").exists():
        return {"hash": "", "verified": False, "reason": "no suite.json"}
    spec = suite_spec(root, name)
    h = fingerprint(*suite_dirs(root, name))
    if spec["kind"] == "fix_loop":
        inner = status(root, spec["grader_suite"])
        return {"hash": h, "verified": inner["verified"], "reason": f"graded by {spec['grader_suite']}: {inner['reason']}"}
    if spec["kind"] == "media":
        missing = [q["label"] for q in spec.get("questions", []) if not q.get("expected")]
        if missing:
            return {"hash": h, "verified": True,
                    "reason": f"runs, but no expected answers to grade against ({len(missing)} missing): run /setup-evals"}
        return {"hash": h, "verified": True, "reason": "answers graded by a person or agent against expected.md"}
    if spec["kind"] not in GRADED_KINDS:
        return {"hash": h, "verified": True, "reason": "no answer key needed"}
    missing = [f for f in ("grader.py", "reference.py", "hidden_tests.py") if not (d / f).exists()]
    if missing:
        return {"hash": h, "verified": False, "reason": f"answer key not generated yet (no {', '.join(missing)}): run /setup-evals"}
    vf = d / "verification.json"
    if not vf.exists():
        return {"hash": h, "verified": False, "reason": "never verified: run spark-cluster eval verify " + name}
    v = json.loads(vf.read_text())
    if v.get("fingerprint") != fingerprint(d):
        return {"hash": h, "verified": False, "reason": "changed since it was verified: verify it again"}
    if not v.get("passed"):
        return {"hash": h, "verified": False, "reason": "last verification failed"}
    return {"hash": h, "verified": True, "reason": f"verified {v.get('date', '')[:10]}"}


def verify(root: Path, image: str, name: str) -> bool:
    d = root / "suites" / name
    spec = suite_spec(root, name)
    if spec["kind"] not in GRADED_KINDS:
        say(f"{name}: {spec['kind']} suite, nothing to verify")
        return True
    curated = spec.get("origin") == "curated"
    checks: list[dict] = []

    def check(what: str, ok: bool, detail: str) -> None:
        checks.append({"check": what, "passed": ok, "detail": detail})
        say(f"  {'ok  ' if ok else 'FAIL'} {what}: {detail}")

    say(f"== {name} ({'curated' if curated else 'generated'})")
    for f in ("grader.py", "reference.py", "hidden_tests.py"):
        if not (d / f).exists():
            check("files", False, f"missing {f}")
            return finish(d, name, checks)

    grader = load(d / "grader.py", f"grader_{name}", d)
    stress_mod = (d / "stress_cases.py").exists()
    if stress_mod and hasattr(grader, "build_expected"):
        n = grader.build_expected(d / "reference.py", d / "stress_expected.json")
        say(f"  rebuilt stress_expected.json from the reference: {n} cases")
    quiet = lambda event, **f: None  # noqa: E731

    # 1. reference is perfect
    t = time.time()
    ref = grader.grade(d / "reference.py", image, quiet)
    check("reference scores 100%", perfect(ref), f"{summary(ref)} in {time.time() - t:.0f}s"
          + (f"; first failures: {ref['failures'][:3]}" if ref["failures"] else ""))

    # 2. deterministic
    again = grader.grade(d / "reference.py", image, quiet)
    check("repeat grading identical", again == ref, "same result twice" if again == ref else f"second run: {summary(again)}")

    # 3. mutants
    mutants = sorted((d / "mutants").glob("*.py")) if (d / "mutants").is_dir() else []
    if len(mutants) < MIN_MUTANTS and not (curated and not mutants):
        check("mutants", False, f"{len(mutants)} in mutants/, need at least {MIN_MUTANTS}")
    hidden = load(d / "hidden_tests.py", "hidden_tests", d)
    descs = [c[0] for c in getattr(hidden, "CASES", [])] if hasattr(hidden, "FUNCTION") else []
    caught_by: dict[str, int] = {str(x): 0 for x in descs}
    for m in mutants:
        r = grader.grade(m, image, quiet)
        caught = not perfect(r)
        for f in r["failures"]:
            for desc in caught_by:
                if f.startswith(f"{desc}:"):
                    caught_by[desc] += 1
        check(f"mutant {m.name} is caught", caught, summary(r))
    if mutants and descs:
        idle = [x for x, n in caught_by.items() if n == 0]
        say(f"  note: {len(idle)} of {len(descs)} hidden cases catch no mutant" + (f" (e.g. {idle[:3]})" if idle else ""))

    # 4. cross-check against an independent solver
    if stress_mod and (d / "brute.py").exists():
        sc = load(d / "stress_cases.py", "stress_cases", d)
        ref_mod = load(d / "reference.py", "reference_x", d)
        brute = load(d / "brute.py", "brute_x", d)
        applicable = getattr(brute, "applicable", lambda args: True)
        agree = disagree = 0
        examples = []
        for args in sc.generate(1):
            args = list(args)
            if not applicable(args):
                continue
            a, b = sc.expected(ref_mod, args), brute.expected(args)
            if a == b:
                agree += 1
            else:
                disagree += 1
                if len(examples) < 3:
                    examples.append({"args": args, "reference": a, "brute": b})
        check("reference agrees with brute.py", disagree == 0 and agree >= MIN_CROSS,
              f"{agree} agree, {disagree} disagree (need ≥{MIN_CROSS} compared)" + (f"; e.g. {examples}" if examples else ""))
    elif not curated and stress_mod:
        check("reference agrees with brute.py", False, "stress suites need brute.py, an independent solver")

    # 5. shape
    if not curated:
        n = len(descs)
        check("enough hidden cases", n >= MIN_CASES, f"{n} cases (need ≥{MIN_CASES})")
        spec_md = (d / "SPEC.md").read_text() if (d / "SPEC.md").exists() else ""
        cats = re.findall(r"^\s*-\s*\[([a-z0-9-]+)\]", spec_md, re.M)
        src = (d / "hidden_tests.py").read_text()
        missing = [c for c in cats if c not in src]
        check("SPEC.md categories covered", bool(cats) and not missing,
              f"{len(cats) - len(missing)}/{len(cats)} named in hidden_tests.py" + (f"; missing {missing}" if missing else ""))
    return finish(d, name, checks)


def finish(d: Path, name: str, checks: list[dict]) -> bool:
    passed = bool(checks) and all(c["passed"] for c in checks)
    record = {"suite": name, "passed": passed, "fingerprint": fingerprint(d),
              "date": datetime.now(timezone.utc).isoformat(timespec="seconds"), "checks": checks}
    (d / "verification.json").write_text(json.dumps(record, indent=1))
    say(f"{'PASSED' if passed else 'FAILED'} {name} (fingerprint {record['fingerprint']})")
    return passed


def main() -> None:
    root, image = Path(sys.argv[1]), sys.argv[2]
    names = [a for a in sys.argv[3:] if not a.startswith("--")]
    if "--status" in sys.argv:
        print(json.dumps({n: status(root, n) for n in names}))
        return
    ok = all([verify(root, image, n) for n in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
