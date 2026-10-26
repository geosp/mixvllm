import json
from datetime import datetime
from fractions import Fraction
from pathlib import Path

from spark_cluster import eval as ev
from spark_cluster.config import load_cluster
from spark_cluster.evalkit import runner, sandbox, shim


def test_value_round_trip():
    value = (Fraction(1, 3), [("laptop", 0, 0, 0, 36, 25, 2)], None, True, 2.5, {1, 2})
    back = sandbox.decode(json.loads(json.dumps(shim.encode(value))))
    assert back[0] == Fraction(1, 3) and isinstance(back[0], Fraction)
    assert back[1] == [["laptop", 0, 0, 0, 36, 25, 2]]  # tuples come back as lists; graders accept both
    assert back[2:5] == [None, True, 2.5]
    assert isinstance(back[5], sandbox.Other) and back[5].type_name == "set"


def test_ints_stay_ints():
    # challenge 1 must fail an int answer ("returned int 7, expected Fraction 7"), as the old grader did
    back = sandbox.decode(json.loads(json.dumps(shim.encode(7))))
    assert type(back) is int


def test_outcome_raised_follows_mro():
    out = sandbox.Outcome("raised", exc="MyError", mro=("MyError", "ValueError", "Exception"))
    assert out.raised("ValueError") and not out.raised("TypeError")


def test_run_id():
    when = datetime(2026, 10, 25, 1, 8, 30)
    assert ev.new_run_id("qwen3.8-flash-next", "low effort/2", when) == "20261025-010830_qwen3.8-flash-next_low-effort-2"


def test_risky_calls_counted_like_before():
    code = "import os\nx = eval('1')\nopen('f')\nsocket = 1\n"
    assert len(runner.RISKY.findall(code)) == 4


def events(*records):
    base = {"t": "2026-10-25T05:00:00+00:00", "suite": "", "rep": 0}
    return [{**base, **r} for r in records]


def test_tracker_folds_a_code_suite():
    tr = ev.Tracker()
    for e in events(
        {"event": "run_start"},
        {"event": "suite_start", "suite": "challenge2", "rep": 1, "t": "2026-10-25T05:00:00+00:00"},
        {"event": "progress", "suite": "challenge2", "rep": 1, "phase": "thinking", "reasoning_chars": 1200,
         "answer_chars": 0},
        {"event": "done", "suite": "challenge2", "rep": 1, "tokens": 9000, "finish": "stop", "code_blocks": 1},
        {"event": "grade", "suite": "challenge2", "rep": 1, "passed": 18, "total": 18},
        {"event": "stress", "suite": "challenge2", "rep": 1, "missed": 0, "impossible": 0, "invalid": 0, "errors": 0},
        {"event": "suite_done", "suite": "challenge2", "rep": 1, "t": "2026-10-25T05:03:05+00:00"},
        {"event": "run_done"},
    ):
        tr.feed(e)
    row = tr.rows[("challenge2", 1)]
    assert tr.finished == "done"
    assert row["state"] == "done" and row["result"] == "18/18"
    assert row["stress"].startswith("missed 0")
    table = tr.table("run")
    assert table.row_count == 1


def test_tracker_keeps_fix_loop_rounds():
    # challenge 3 on 2026-10-25: round 1 thought until max_tokens, round 2's tests failed, round 3 fixed them
    tr = ev.Tracker()
    c3 = {"suite": "challenge3", "rep": 1}
    for e in events(
        {"event": "suite_start", **c3},
        {"event": "sent", **c3, "round": 1, "of": 4},
        {"event": "round", **c3, "round": 1, "tokens": 32000, "finish": "length", "implementation": False,
         "tests": False},
        {"event": "sent", **c3, "round": 2, "of": 4},
        {"event": "progress", **c3, "phase": "thinking", "reasoning_chars": 14420, "answer_chars": 0},
    ):
        tr.feed(e)
    row = tr.rows[("challenge3", 1)]
    assert row["state"] == "thinking · round 2/4"
    assert row["rounds"] == ["round 1: 32,000 tokens, out of tokens, no code"]
    for e in events(
        {"event": "round", **c3, "round": 2, "tokens": 14411, "finish": "stop", "implementation": True,
         "tests": True},
        {"event": "own_tests", **c3, "round": 2, "summary": "10/12"},
        # events from before `finish` and `of` were recorded still fold
        {"event": "sent", **c3, "round": 3},
        {"event": "round", **c3, "round": 3, "tokens": 4356, "implementation": True, "tests": True},
        {"event": "own_tests", **c3, "round": 3, "summary": "12/12"},
    ):
        tr.feed(e)
    assert row["rounds"][1:] == ["round 2: 14,411 tokens, own tests 10/12", "round 3: 4,356 tokens, own tests 12/12"]
    assert tr.table("run").row_count == 1


def test_tracker_marks_errors():
    tr = ev.Tracker()
    for e in events(
        {"event": "suite_start", "suite": "speed", "rep": 1},
        {"event": "error", "suite": "speed", "rep": 1, "error": "URLError: refused"},
        {"event": "suite_done", "suite": "speed", "rep": 1},
        {"event": "stopped"},
    ):
        tr.feed(e)
    assert tr.rows[("speed", 1)]["state"] == "error"
    assert tr.finished == "stopped"


def test_fetch_events_keeps_partial_lines(monkeypatch):
    lines = [json.dumps({"t": "2026-10-25T05:00:00+00:00", "event": "sent", "suite": "s", "rep": 1})]
    partial = '{"t": "2026-10-25T05:00:0'
    body = lines[0] + "\n" + partial
    monkeypatch.setattr(ev, "sh", lambda cluster, script, **kw: body + "\n\x1ealive\n")
    monkeypatch.setattr(ev, "root", lambda cluster: "/evals")
    got, offset, alive = ev.fetch_events(None, "run", 0)
    assert [g["event"] for g in got] == ["sent"]
    assert offset == len(lines[0]) + 1  # the partial line is read again next time
    assert alive


def result_row(**kw):
    row = {k: "" for k in runner.RESULT_FIELDS}
    row.update(run_id="r1", model="m", tag="t", settings="effort=default", finished="2026-10-25T05:00:00")
    row.update(kw)
    return row


def test_cells():
    speed = result_row(suite="speed", metrics=json.dumps({"prose_tok_s": 43.1, "code_tok_s": 50.8, "short_s": 2.7,
                                                           "draft_acceptance": 0.61}))
    assert ev.cell("speed", [speed]) == "43.1/50.8 tok/s, 2.7s, accept 61%"
    c2 = [result_row(suite="challenge2", rep="1", passed="18", total="18", missed="0", impossible="0",
                     invalid="0", errors="0", seconds="185"),
          result_row(suite="challenge2", rep="2", passed="18", total="18", missed="5", impossible="0",
                     invalid="2", errors="0", seconds="240")]
    assert ev.cell("challenge2", c2) == "18/18 · missed 0 · 3m05s\n18/18 · missed 5 · invalid 2 · 4m00s"
    vision = result_row(suite="vision", total="4", metrics=json.dumps(
        {"ungraded": True, "answers": [{"status": "answered"}] * 3 + [{"status": "rejected"}]}))
    assert ev.cell("vision", [vision]) == "3/4 answered, ungraded"
    graded = result_row(suite="vision", passed="3", total="4", metrics=json.dumps({"graded": True, "by": "agent"}))
    assert ev.cell("vision", [vision, graded]) == "3/4"
    c3 = result_row(suite="challenge3", passed="18", total="18", missed="0", seconds="1012",
                    metrics=json.dumps({"rounds": 3, "length_rounds": 1}))
    assert ev.cell("challenge3", [c3]) == "18/18 · missed 0 · 16m52s · 1 round out of tokens"
    none = result_row(suite="challenge3", passed="0", metrics=json.dumps(
        {"note": "no implementation produced", "length_rounds": 4}))
    assert ev.cell("challenge3", [none]) == "no implementation produced · 4 rounds out of tokens"


def test_summarize_groups_by_run():
    rows = [result_row(run_id="a", suite="speed"), result_row(run_id="a", suite="tools"),
            result_row(run_id="b", suite="speed")]
    groups = ev.summarize(rows)
    assert [g["run_id"] for g in groups] == ["a", "b"]
    assert set(groups[0]["suites"]) == {"speed", "tools"}


def test_evals_path_is_optional(tmp_path):
    text = (
        "repo_path: ~/x\napi_url: http://h:8000\nenv:\n  MODEL_NAME: m\n  VLLM_IMAGE: i\n"
        "  VLLM_ENTRYPOINT: /usr/bin/env\n  HF_HOME: /h\nnodes:\n  - {name: a, ssh: a, mesh_ip: 1.1.1.1}\n"
    )
    p = tmp_path / "cluster.yml"
    p.write_text(text)
    assert load_cluster(p).evals_path is None
    p.write_text(text.replace("env:", "evals_path: /mnt/shared/evals/\nenv:"))
    assert load_cluster(p).evals_path == "/mnt/shared/evals"


def test_speed_cell_flags_shared_load():
    m = {"prose_tok_s": 34.0, "code_tok_s": 39.5, "short_s": 2.7, "shared_load": True, "peak_other_requests": 2}
    assert ev.cell("speed", [result_row(suite="speed", metrics=json.dumps(m))]) == (
        "34.0/39.5 tok/s, 2.7s (shared: up to 2 other requests)")


def test_load_sums_running_and_waiting(monkeypatch):
    text = (
        "# HELP vllm:num_requests_running x\n"
        'vllm:num_requests_running{engine="0",model_name="m"} 3.0\n'
        'vllm:num_requests_waiting{engine="0",model_name="m"} 2.0\n'
        'vllm:num_requests_swapped{engine="0",model_name="m"} 9.0\n'
    )

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return text.encode()

    monkeypatch.setattr(runner.urllib.request, "urlopen", lambda *a, **k: Resp())
    r = runner.Run.__new__(runner.Run)
    r.api = "http://x"
    assert r.load() == 5


# ---------------------------------------------------------------- fingerprints, results.csv, report tags


def test_fingerprint_stable_and_sensitive(tmp_path):
    d = tmp_path / "challenge1"
    (d / "__pycache__").mkdir(parents=True)
    (d / "prompt.md").write_text("task")
    (d / "hidden_tests.py").write_text("CASES = []")
    first = runner.fingerprint(d)
    (d / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (d / "verification.json").write_text("{}")
    assert runner.fingerprint(d) == first  # caches and the verification record don't count
    (d / "hidden_tests.py").write_text("CASES = [1]")
    assert runner.fingerprint(d) != first
    assert len(first) == 12


def test_fingerprint_covers_extra_dirs(tmp_path):
    a, b = tmp_path / "challenge3", tmp_path / "challenge2"
    a.mkdir()
    b.mkdir()
    (a / "prompt.md").write_text("x")
    (b / "grader.py").write_text("y")
    alone, both = runner.fingerprint(a), runner.fingerprint(a, b)
    (b / "grader.py").write_text("z")
    assert runner.fingerprint(a) == alone and runner.fingerprint(a, b) != both


def test_results_csv_migrates_old_header(tmp_path):
    path = tmp_path / "results.csv"
    old_fields = [f for f in runner.RESULT_FIELDS if f != "suite_hash"]
    path.write_text(",".join(old_fields) + "\n" + ",".join("r1" if f == "run_id" else "" for f in old_fields) + "\n")
    runner.write_results(path, [{"run_id": "r2", "suite": "speed", "suite_hash": "abc"}])
    import csv

    rows = list(csv.DictReader(path.open()))
    assert list(rows[0].keys()) == runner.RESULT_FIELDS
    assert [r["run_id"] for r in rows] == ["r1", "r2"] and rows[1]["suite_hash"] == "abc"


def test_results_csv_new_file(tmp_path):
    path = tmp_path / "results.csv"
    runner.write_results(path, [{"run_id": "r1"}])
    runner.write_results(path, [{"run_id": "r2"}])
    lines = path.read_text().splitlines()
    assert lines[0] == ",".join(runner.RESULT_FIELDS) and len(lines) == 3


def test_report_tags_only_mixed_suites():
    groups = ev.summarize([
        result_row(run_id="a", suite="challenge1", suite_hash="aaa", passed="45", total="45"),
        result_row(run_id="a", suite="speed", suite_hash="sss", metrics=json.dumps({"prose_tok_s": 1, "code_tok_s": 2, "short_s": 3})),
        result_row(run_id="b", suite="challenge1", suite_hash="bbb", passed="53", total="53"),
        result_row(run_id="b", suite="speed", suite_hash="sss", metrics=json.dumps({"prose_tok_s": 1, "code_tok_s": 2, "short_s": 3})),
    ])
    tags = ev.hash_tags(groups, ["challenge1", "speed"])
    assert set(tags) == {"challenge1"}  # speed came from the same suite version: no tag
    assert ev.tagged_cell("challenge1", groups[1]["suites"]["challenge1"], tags).startswith("[b] 53/53")
    assert not ev.tagged_cell("speed", groups[0]["suites"]["speed"], tags).startswith("[")


def test_public_specs_hold_no_answer_keys():
    files = ev.public_suite_files()
    names = {Path(k).name for k in files}
    assert {"prompt.md", "suite.json"} <= names
    for forbidden in ("hidden_tests.py", "reference.py", "brute.py", "stress_cases.py", "stress_expected.json",
                      "expected.md", "grader.py", "verification.json"):
        assert forbidden not in names, forbidden
    assert not any("mutants/" in k or Path(k).name.startswith("fake_") for k in files)
    vision = json.loads(files["vision/suite.json"])
    assert all("expected" not in q for q in vision["questions"])
