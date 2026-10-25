from spark_cluster.config import Node
from spark_cluster.progress import READY, ApiStatus, LogProgress, NodeStatus, api_line

HEAD = Node("spark-01", "spark-01", "192.168.100.1", 0)
WORKER = Node("spark-02", "spark-02", "192.168.100.2", 1)
REFUSED = ApiStatus(False, [], "[Errno 111] Connection refused")


def status(node, state="running", stage="loading weights"):
    return NodeStatus(node, state=state, log=LogProgress(stage=stage))


def test_healthy():
    line = api_line([status(HEAD, stage=READY)], ApiStatus(True, ["qwen3.8-flash-next", "spark"]))
    assert "healthy" in line and "qwen3.8-flash-next, spark" in line


def test_refused_while_starting_is_not_a_warning():
    line = api_line([status(HEAD), status(WORKER)], REFUSED)
    assert "not up yet" in line and "Errno" not in line


def test_down():
    line = api_line([status(HEAD, state="missing"), status(WORKER, state="missing")], REFUSED)
    assert "no vllm containers" in line


def test_ready_but_not_answering_warns():
    assert "Connection refused" in api_line([status(HEAD, stage=READY)], REFUSED)


def test_head_exited_warns():
    assert "Connection refused" in api_line([status(HEAD, state="exited"), status(WORKER)], REFUSED)
