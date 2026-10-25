from spark_cluster.checks import FAIL, OK, WARN, check_host, check_memory, check_model, check_model_files
from spark_cluster.config import Node

REGISTRY = {
    "models": {
        "tp2": {"model": "org/tp2", "tensor-parallel-size": 2, "gpu-memory-utilization": 0.7},
        "tp2pp2": {"model": "org/big", "tensor-parallel-size": 2, "pipeline-parallel-size": 2},
        "single": {"model": "org/small"},
    }
}
GiB = 2**30


def levels(checks):
    return [c.level for c in checks]


def test_model_ok():
    checks, cfg = check_model(REGISTRY, "tp2", 2)
    assert levels(checks) == [OK, OK]
    assert cfg["model"] == "org/tp2"


def test_model_unknown():
    checks, cfg = check_model(REGISTRY, "nope", 2)
    assert cfg is None
    assert checks[0].level == FAIL and "tp2" in checks[0].detail


def test_gpu_split_must_match_nodes():
    assert levels(check_model(REGISTRY, "tp2pp2", 2)[0]) == [OK, FAIL]
    assert levels(check_model(REGISTRY, "tp2pp2", 4)[0]) == [OK, OK]
    assert levels(check_model(REGISTRY, "single", 2)[0]) == [OK, FAIL]


def test_memory_estimate():
    cfg = REGISTRY["models"]["tp2"]
    assert check_memory(cfg, 124 * GiB).level == OK  # 62 GiB per node < 0.7 × 121
    assert check_memory(cfg, 180 * GiB).level == WARN  # 90 GiB per node
    assert check_memory(REGISTRY["models"]["single"], GiB).level == WARN  # no utilization set


def test_model_files():
    assert levels(check_model_files("org/x", {"present": False})) == [FAIL]
    missing = {"present": True, "complete": False, "missing": ["model-00002.safetensors"]}
    assert "model-00002" in check_model_files("org/x", missing)[0].detail
    complete = {"present": True, "complete": True, "missing": [], "weight_bytes": GiB, "incomplete_blobs": 3}
    assert levels(check_model_files("org/x", complete)) == [OK, WARN]


NODE = Node("spark-02", "spark-02", "192.168.100.2", 1)
NFS = {
    "mount_point": "/mnt/shared",
    "device": "0:68",
    "fstype": "nfs4",
    "source": "192.168.100.1:/mnt/shared",
    "options": "rw,relatime,rw,vers=4.1,proto=rdma,port=20049",
    "read_ahead_kb": 16384,
}
FACTS = {
    "commit": "abc1234def",
    "changes": "",
    "node_dir": True,
    "secrets_env": True,
    "infiniband": True,
    "hf_home": True,
    "mount": NFS,
}


def test_host_all_good():
    assert set(levels(check_host(NODE, FACTS, "abc1234def"))) == {OK}


def test_host_local_disk_skips_nfs_checks():
    facts = {**FACTS, "mount": {**NFS, "fstype": "ext4", "options": "rw", "read_ahead_kb": 128}}
    assert set(levels(check_host(NODE, facts, "abc1234def"))) == {OK}


def test_host_problems():
    facts = {
        **FACTS,
        "commit": "0000000aaa",
        "changes": " M docker/config/model_registry.yml",
        "secrets_env": False,
        "infiniband": False,
        "mount": {**NFS, "options": "rw,proto=tcp", "read_ahead_kb": 128},
    }
    failed = {c.what for c in check_host(NODE, facts, "abc1234def") if c.level == FAIL}
    assert failed == {
        "spark-02 commit",
        "spark-02 local changes",
        "spark-02 secrets.env",
        "spark-02 RDMA",
        "spark-02 NFS",
        "spark-02 read-ahead",
    }


def test_host_old_checkout():
    facts = {**FACTS, "node_dir": False, "secrets_env": False}
    checks = check_host(NODE, facts, "abc1234def")
    assert [c.what for c in checks if c.level == FAIL] == ["spark-02 docker/node"]
