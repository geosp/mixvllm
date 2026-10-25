"""Everything `spark-cluster check` verifies before the cluster starts."""

from __future__ import annotations

import json
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from . import remote
from .config import REPO_ROOT, Cluster, Node, load_registry, model_config, node_env

GPU_MEMORY_GIB = 121  # GB10 unified memory per node
MIN_READ_AHEAD_KB = 16384  # NFS read-ahead for near local-disk load speed
PROBE = Path(__file__).with_name("node_probe.py")

OK, WARN, FAIL = "ok", "warn", "fail"


@dataclass
class Check:
    level: str  # ok / warn / fail
    what: str
    detail: str = ""


def gpu_split(cfg: dict) -> int:
    return int(cfg.get("tensor-parallel-size", 1)) * int(cfg.get("pipeline-parallel-size", 1))


def check_model(registry: dict, model_name: str, n_nodes: int) -> tuple[list[Check], dict | None]:
    """MODEL_NAME is in the registry and its GPU split matches the node count."""
    cfg = model_config(registry, model_name)
    if cfg is None:
        known = ", ".join(registry.get("models", {}))
        return [Check(FAIL, f"model {model_name}", f"not in the registry (known: {known})")], None
    checks = [Check(OK, f"model {model_name}", f"in the registry ({cfg['model']})")]
    split = gpu_split(cfg)
    if split != n_nodes:
        checks.append(
            Check(
                FAIL,
                "GPU split",
                f"tensor-parallel-size × pipeline-parallel-size = {split}, but the cluster has "
                f"{n_nodes} node(s) with one GPU each",
            )
        )
    else:
        checks.append(Check(OK, "GPU split", f"{split} GPUs = {n_nodes} nodes"))
    return checks, cfg


def check_memory(cfg: dict, weight_bytes: int) -> Check:
    """Rough: the checkpoint's share per node must fit in gpu-memory-utilization × node memory."""
    util = cfg.get("gpu-memory-utilization")
    if util is None:
        return Check(WARN, "memory estimate", "gpu-memory-utilization isn't set (vLLM's default is too high on unified memory)")
    per_node = weight_bytes / 2**30 / gpu_split(cfg)
    budget = float(util) * GPU_MEMORY_GIB
    detail = f"weights {per_node:.1f} GiB per node, budget {budget:.1f} GiB ({util} × {GPU_MEMORY_GIB} GiB)"
    if per_node >= budget:
        return Check(WARN, "memory estimate", detail + ": the weights alone don't fit")
    return Check(OK, "memory estimate", detail)


def check_model_files(model_id: str, info: dict) -> list[Check]:
    if not info.get("present"):
        return [Check(FAIL, "model files", f"{model_id} isn't downloaded (spark-cluster download <model>)")]
    if not info.get("complete"):
        missing = ", ".join(info["missing"][:5]) or "no weight files"
        return [Check(FAIL, "model files", f"{model_id} is incomplete: missing {missing}")]
    checks = [Check(OK, "model files", f"{model_id} complete, {info['weight_bytes'] / 2**30:.1f} GiB of weights")]
    if info.get("incomplete_blobs"):
        checks.append(
            Check(WARN, "model files", f"{info['incomplete_blobs']} leftover *.incomplete blob(s) from an interrupted download")
        )
    return checks


def check_host(node: Node, facts: dict, local_commit: str | None) -> list[Check]:
    """The repo checkout, secrets and host setup of one node, from node_probe.py's facts."""
    name = node.name
    out = []
    commit = facts.get("commit")
    if commit is None:
        out.append(Check(FAIL, f"{name} repo", "not a git checkout at repo_path"))
    elif local_commit and commit != local_commit:
        out.append(Check(FAIL, f"{name} commit", f"{commit[:7]}, local HEAD is {local_commit[:7]} (spark-cluster pull)"))
    else:
        out.append(Check(OK, f"{name} commit", commit[:7]))
    if facts.get("changes"):
        files = ", ".join(line[3:] for line in facts["changes"].splitlines()[:5])
        out.append(Check(FAIL, f"{name} local changes", f"in docker/config or docker/node: {files}"))
    if not facts.get("node_dir"):
        out.append(Check(FAIL, f"{name} docker/node", "missing: the node is on an old commit"))
    elif not facts.get("secrets_env"):
        out.append(Check(FAIL, f"{name} secrets.env", "missing (copy docker/node/secrets.env.example, chmod 600)"))
    else:
        out.append(Check(OK, f"{name} secrets.env", "present"))

    if facts.get("infiniband"):
        out.append(Check(OK, f"{name} RDMA", "/dev/infiniband present"))
    else:
        out.append(Check(FAIL, f"{name} RDMA", "/dev/infiniband missing or has no uverbs devices"))

    mount = facts.get("mount")
    if not facts.get("hf_home") or not mount:
        out.append(Check(FAIL, f"{name} model folder", "HF_HOME doesn't exist on this node"))
    elif mount["fstype"].startswith("nfs"):
        opts = mount["options"].split(",")
        if "proto=rdma" not in opts:
            out.append(Check(FAIL, f"{name} NFS", f"{mount['mount_point']} isn't mounted with proto=rdma"))
        else:
            out.append(Check(OK, f"{name} NFS", f"{mount['mount_point']} over RDMA"))
        ra = mount.get("read_ahead_kb")
        if ra is None or ra < MIN_READ_AHEAD_KB:
            out.append(
                Check(
                    FAIL,
                    f"{name} read-ahead",
                    f"{ra} KB on {mount['mount_point']}, need ≥ {MIN_READ_AHEAD_KB} "
                    "(nfs-shared-readahead.service, see docker/README.md)",
                )
            )
        else:
            out.append(Check(OK, f"{name} read-ahead", f"{ra} KB"))
    else:
        out.append(Check(OK, f"{name} model folder", f"local {mount['fstype']}"))
    return out


def probe(cluster: Cluster, node: Node, model_id: str | None = None) -> dict:
    env = node_env(cluster, node)
    args = [cluster.repo_path, env["HF_HOME"]] + ([model_id] if model_id else [])
    # repo_path may start with ~, which the probe expands itself
    res = remote.run(node, "python3 - " + " ".join(shlex.quote(a) for a in args), input=PROBE.read_text())
    res.check("host probe")
    return json.loads(res.stdout)


def local_git(*args: str) -> str | None:
    proc = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def run_checks(cluster: Cluster, model_name: str | None = None) -> list[Check]:
    model_name = model_name or cluster.model_name
    registry = load_registry()
    checks, cfg = check_model(registry, model_name, len(cluster.nodes))

    local_commit = local_git("rev-parse", "HEAD")
    if local_git("status", "--porcelain", "--", "docker/config", "docker/node/docker-compose.yml"):
        checks.append(Check(WARN, "local changes", "docker/config or the compose file has uncommitted changes the nodes won't see"))

    model_id = cfg["model"] if cfg else None

    def node_checks(node: Node) -> list[Check]:
        try:
            facts = probe(cluster, node, model_id if node.is_head else None)
        except remote.RemoteError as e:
            return [Check(FAIL, f"{node.name} reachable", str(e))]
        out = check_host(node, facts, local_commit)
        if node.is_head and model_id:
            out += check_model_files(model_id, facts["model"])
            if facts["model"].get("complete"):
                out.append(check_memory(cfg, facts["model"]["weight_bytes"]))
        return out

    for result in remote.parallel(cluster.nodes, node_checks):
        checks += result
    return checks


def model_files_on_head(cluster: Cluster, model_id: str) -> dict:
    return probe(cluster, cluster.head, model_id)["model"]
