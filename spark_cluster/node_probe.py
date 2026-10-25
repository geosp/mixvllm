"""Collect host facts on a node for `spark-cluster check`; prints one JSON object.

Sent over SSH and run with the node's own python3 (`python3 - <repo_path> <hf_home> [<model repo>]`),
so it uses the standard library only.
"""

import json
import os
import subprocess
import sys


def git(repo, *args):
    proc = subprocess.run(["git", "-C", repo, *args], capture_output=True, text=True)
    return proc.stdout.strip() if proc.returncode == 0 else None


def mount_of(path):
    """The /proc/self/mountinfo entry holding path: mount point, fstype, options, read-ahead."""
    path = os.path.realpath(path)
    best = None
    with open("/proc/self/mountinfo") as f:
        for line in f:
            pre, post = line.split(" - ", 1)
            fields = pre.split()
            mount_point = fields[4].replace("\\040", " ")
            inside = path == mount_point or path.startswith(mount_point.rstrip("/") + "/")
            if inside and (best is None or len(mount_point) >= len(best["mount_point"])):
                fstype, source, super_opts = post.split()[:3]
                best = {
                    "mount_point": mount_point,
                    "device": fields[2],  # major:minor, also the bdi name
                    "fstype": fstype,
                    "source": source,
                    "options": fields[5] + "," + super_opts,
                }
    if best:
        try:
            with open(f"/sys/class/bdi/{best['device']}/read_ahead_kb") as f:
                best["read_ahead_kb"] = int(f.read())
        except OSError:
            best["read_ahead_kb"] = None
    return best


def model_files(hf_home, repo_id):
    """Is the model complete in the Hugging Face cache, and how big are its weights?"""
    base = os.path.join(hf_home, "hub", "models--" + repo_id.replace("/", "--"))
    info = {"present": False, "complete": False, "missing": [], "weight_bytes": 0, "incomplete_blobs": 0}
    try:
        with open(os.path.join(base, "refs", "main")) as f:
            snapshot = os.path.join(base, "snapshots", f.read().strip())
    except OSError:
        return info
    if not os.path.isdir(snapshot):
        return info
    info["present"] = True
    blobs = os.path.join(base, "blobs")
    if os.path.isdir(blobs):
        info["incomplete_blobs"] = sum(1 for b in os.listdir(blobs) if b.endswith(".incomplete"))

    index = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(index):
        with open(index) as f:
            weights = sorted(set(json.load(f)["weight_map"].values()))
    else:
        weights = sorted(n for n in os.listdir(snapshot) if n.endswith(".safetensors"))
    # gpt-oss style checkpoints keep extra copies (original/, metal/); only top-level files count.
    required = ["config.json"] + weights
    for name in required:
        p = os.path.join(snapshot, name)
        if not os.path.exists(p):  # follows the symlink into blobs/
            info["missing"].append(name)
        elif name in weights:
            info["weight_bytes"] += os.path.getsize(p)
    info["complete"] = bool(weights) and not info["missing"]
    return info


def main():
    repo, hf_home = os.path.expanduser(sys.argv[1]), sys.argv[2]
    model = sys.argv[3] if len(sys.argv) > 3 else None
    node_dir = os.path.join(repo, "docker", "node")
    facts = {
        "hostname": os.uname().nodename,
        "commit": git(repo, "rev-parse", "HEAD"),
        "changes": git(repo, "status", "--porcelain", "--", "docker/config", "docker/node"),
        "node_dir": os.path.isdir(node_dir),
        "secrets_env": os.path.isfile(os.path.join(node_dir, "secrets.env")),
        "infiniband": os.path.isdir("/dev/infiniband")
        and any(n.startswith("uverbs") for n in os.listdir("/dev/infiniband")),
        "hf_home": os.path.isdir(hf_home),
        "mount": mount_of(hf_home) if os.path.exists(hf_home) else None,
    }
    if model:
        facts["model"] = model_files(hf_home, model)
    print(json.dumps(facts))


if __name__ == "__main__":
    main()
