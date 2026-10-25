import pytest

from spark_cluster.config import (
    CLUSTER_FILE,
    ENV_HEADER,
    ConfigError,
    load_cluster,
    node_env,
    render_env,
    set_model_name,
)

SAMPLE = """\
repo_path: ~/projects/mixvllm
api_url: http://10.0.0.96:8000/
master_port: 29501

env:
  # the model
  MODEL_NAME: qwen3.8-flash-next   # default
  VLLM_IMAGE: vllm/vllm-openai:v0.32.0
  VLLM_ENTRYPOINT: /usr/bin/env
  HF_HOME: /mnt/shared/huggingface
  NCCL_IB_TC: 106
  NCCL_NET_PLUGIN: shared-value

nodes:
  - name: spark-01
    ssh: spark-01
    mesh_ip: 192.168.100.1
    env: {NCCL_NET_PLUGIN: none}
  - name: spark-02
    ssh: spark-02
    mesh_ip: 192.168.100.2
  - name: spark-03
    ssh: spark-03
    mesh_ip: 192.168.100.3
"""


@pytest.fixture
def cluster(tmp_path):
    path = tmp_path / "cluster.yml"
    path.write_text(SAMPLE)
    return load_cluster(path)


def test_ranks_and_head(cluster):
    assert [n.rank for n in cluster.nodes] == [0, 1, 2]
    assert cluster.head.name == "spark-01"
    assert cluster.api_url == "http://10.0.0.96:8000"
    assert [n.container for n in cluster.nodes] == ["vllm-server", "vllm-worker", "vllm-worker"]


def test_derived_values(cluster):
    env = node_env(cluster, cluster.nodes[2])
    assert env["NODE_NAME"] == "spark-03"
    assert env["CONTAINER_NAME"] == "vllm-worker"
    assert env["VLLM_HOST_IP"] == "192.168.100.3"
    assert env["CLUSTER_NNODES"] == "3"
    assert env["CLUSTER_NODE_RANK"] == "2"
    assert env["CLUSTER_MASTER_ADDR"] == "192.168.100.1"
    assert env["CLUSTER_MASTER_PORT"] == "29501"
    assert env["NCCL_IB_TC"] == "106"  # ints become strings


def test_node_overrides_win(cluster):
    assert node_env(cluster, cluster.nodes[0])["NCCL_NET_PLUGIN"] == "none"
    assert node_env(cluster, cluster.nodes[1])["NCCL_NET_PLUGIN"] == "shared-value"


def test_render_env(cluster):
    text = render_env(cluster, cluster.head)
    assert text.startswith(ENV_HEADER)
    body = text[len(ENV_HEADER) :].splitlines()
    assert body[0] == "NODE_NAME=spark-01"
    assert "MODEL_NAME=qwen3.8-flash-next" in body
    assert "NCCL_NET_PLUGIN=none" in body
    assert all("=" in line for line in body)
    assert text.endswith("\n")


@pytest.mark.parametrize("key", ["CLUSTER_NODE_RANK", "VLLM_HOST_IP", "MASTER_ADDR", "WORLD_SIZE"])
def test_reserved_keys_rejected(tmp_path, key):
    path = tmp_path / "cluster.yml"
    path.write_text(SAMPLE.replace("  NCCL_IB_TC: 106\n", f"  NCCL_IB_TC: 106\n  {key}: x\n"))
    with pytest.raises(ConfigError, match=key):
        load_cluster(path)


def test_missing_required_env(tmp_path):
    path = tmp_path / "cluster.yml"
    path.write_text(SAMPLE.replace("  VLLM_IMAGE: vllm/vllm-openai:v0.32.0\n", ""))
    with pytest.raises(ConfigError, match="VLLM_IMAGE"):
        load_cluster(path)


def test_set_model_name_keeps_comments():
    new = set_model_name(SAMPLE, "qwen3.8-27b")
    assert "  MODEL_NAME: qwen3.8-27b   # default\n" in new
    assert "  # the model\n" in new
    assert new.replace("qwen3.8-27b", "qwen3.8-flash-next") == SAMPLE


def test_repo_cluster_file_loads():
    cluster = load_cluster(CLUSTER_FILE)
    assert cluster.head.mesh_ip == node_env(cluster, cluster.nodes[-1])["CLUSTER_MASTER_ADDR"]
