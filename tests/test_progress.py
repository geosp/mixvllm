from datetime import datetime, timezone
from pathlib import Path

from spark_cluster.progress import LOADING, READY, WARMUP, parse_log, parse_timestamp

DATA = Path(__file__).parent / "data"
HEAD = (DATA / "head.log").read_text().splitlines()
WORKER = (DATA / "worker.log").read_text().splitlines()


def upto(lines, marker, nth=1):
    """Lines up to and including the nth line containing marker."""
    seen = 0
    for i, line in enumerate(lines):
        if marker in line:
            seen += 1
            if seen == nth:
                return lines[: i + 1]
    raise AssertionError(f"{marker!r} not found {nth}×")


def test_timestamp():
    ts = parse_timestamp("2026-10-24T07:19:41.106610709Z INFO:     Application startup complete.")
    assert ts == datetime(2026, 10, 24, 7, 19, 41, 106610, tzinfo=timezone.utc)
    assert parse_timestamp("no timestamp here") is None


def test_head_complete():
    p = parse_log(HEAD)
    assert p.stage == READY and p.label() == "ready"
    assert p.load_seconds == 432.086896
    assert p.load_pass == 2
    assert p.load_done_at == datetime(2026, 10, 25, 3, 8, 7, 900744, tzinfo=timezone.utc)
    assert p.ready_at == datetime(2026, 10, 25, 3, 11, 53, 375502, tzinfo=timezone.utc)
    assert p.transport.startswith("NET/IB : Using [0]roceP2p1s0f0:1/RoCE")
    assert p.transport_ok is True
    assert p.errors == []


def test_shard_progress_first_pass():
    p = parse_log(upto(HEAD, "6/11"))
    assert p.stage == LOADING
    assert p.label() == "loading weights · pass 1 · 6/11"


def test_shard_progress_second_pass():
    p = parse_log(upto(HEAD, "3/11", nth=2))
    assert p.label() == "loading weights · pass 2 · 3/11"


def test_loading_without_shard_bars():
    # only rank 0 prints the tqdm bars
    assert parse_log(upto(WORKER, "Loading model from scratch")).label() == "loading weights · pass 1"
    assert parse_log(upto(WORKER, "Initializing model took", nth=2)).label() == "loading weights · pass 2"


def test_no_phantom_pass_between_passes():
    # a pass ending doesn't start the next one: only "Initializing model took" does
    assert parse_log(upto(WORKER, "Loading weights took")).label() == "loading weights · pass 1"
    assert parse_log(upto(WORKER, "Loading weights took", nth=2)).label() == "loading weights · pass 2"
    assert parse_log(upto(HEAD, "Loading weights took", nth=2)).label() == "loading weights · pass 2"


def test_graph_capture_and_warmup():
    p = parse_log(upto(HEAD, "Capturing CUDA graphs"))
    assert p.stage == WARMUP and p.label() == "graph capture"
    assert parse_log(upto(HEAD, "JIT kernel warmup starting")).label() == "warmup"


def test_head_engine_ready_before_api():
    p = parse_log(upto(HEAD, "init engine"))
    assert p.label() == "starting API server"


def test_worker_ready_when_engine_ready():
    p = parse_log(WORKER)
    assert p.load_seconds == 410.723719
    assert p.load_pass == 2
    assert p.label(headless=True) == "ready"
    assert p.engine_ready_at == datetime(2026, 10, 25, 3, 11, 8, 252861, tzinfo=timezone.utc)


def test_tcp_fallback():
    lines = [
        "2026-10-24T07:09:15.5Z spark-01:258:258 [0] NCCL INFO NET/Socket : Using [0]enP2p1s0f0np0:192.168.100.1<0>",
    ]
    p = parse_log(lines)
    assert p.transport_ok is False
    assert "NET/Socket" in p.transport


def test_errors_collected():
    lines = HEAD[:5] + [
        "2026-10-24T07:10:00.0Z (EngineCore pid=186) ERROR 10-24 07:10:00 [core.py:1] EngineCore failed to start.",
        "2026-10-24T07:10:00.1Z RuntimeError: CUDA error: out of memory",
    ]
    p = parse_log(lines)
    assert p.errors[-1] == "RuntimeError: CUDA error: out of memory"
    assert len(p.errors) == 2
