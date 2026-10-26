"""Runs inside the solution's sandbox container: loads the model's code and answers calls over a pipe.

Usage (inside the container): python3 -I shim.py /s/solution.py

Protocol, one JSON object per line:
  first line out:  {"ready": true} or {"load_error": "<traceback>"}
  in:              {"fn": "choose_box", "args": [...], "timeout": 10}
  out:             {"ok": <value>} | {"raised": "ValueError", "mro": [...], "msg": "..."} | {"timeout": true}

Only the solution file and this shim are mounted, so the model's code has no answer key to read:
the grader sends each test case's arguments and checks the result on its own side. The protocol
uses private copies of stdin/stdout; the solution's own prints and reads go to stderr and /dev/null.
"""

import importlib.util
import json
import os
import signal
import sys
import traceback
from fractions import Fraction


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout()


def encode(v):
    """JSON-safe form of a return value; the grader turns it back into Python values."""
    if isinstance(v, Fraction):
        return {"__fraction__": [v.numerator, v.denominator]}
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, (list, tuple)):
        return [encode(x) for x in v]
    return {"__other__": type(v).__name__, "repr": repr(v)[:300]}


def main() -> None:
    proto_in = os.fdopen(os.dup(0), "r")
    proto_out = os.fdopen(os.dup(1), "w")
    devnull = os.open(os.devnull, os.O_RDONLY)
    os.dup2(devnull, 0)
    os.dup2(2, 1)  # anything the solution prints, even via os.write(1, ...), lands on stderr
    sys.stdin, sys.stdout = open(os.devnull), sys.stderr

    def send(obj) -> None:
        proto_out.write(json.dumps(obj) + "\n")
        proto_out.flush()

    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(30)
        spec = importlib.util.spec_from_file_location("solution", sys.argv[1])
        mod = importlib.util.module_from_spec(spec)
        sys.modules["solution"] = mod
        spec.loader.exec_module(mod)
    except BaseException:
        send({"load_error": traceback.format_exc(limit=5)[-2000:]})
        return
    finally:
        signal.alarm(0)
    send({"ready": True})

    for line in proto_in:
        req = json.loads(line)
        signal.alarm(int(req.get("timeout", 10)))
        try:
            fn = getattr(mod, req["fn"])
            result = {"ok": encode(fn(*req["args"]))}
        except Timeout:
            result = {"timeout": True}
        except Exception as e:
            result = {"raised": type(e).__name__, "mro": [c.__name__ for c in type(e).__mro__], "msg": str(e)[:300]}
        finally:
            signal.alarm(0)
        try:
            send(result)
        except (TypeError, ValueError):  # e.g. a float NaN key; report rather than die
            send({"ok": {"__other__": "unencodable", "repr": repr(result)[:300]}})


if __name__ == "__main__":
    main()
