"""Run a model's own unit tests against its own implementation. Run: python3 run_own_tests.py solution.py tests.py

The implementation is registered as module `solution`, so tests can `from solution import choose_box`.
Each test_* function gets 10 seconds. Prints PASS/FAIL per test and an OWN_SUMMARY line.
"""

import importlib.util
import signal
import sys
import traceback


class Timeout(Exception):
    pass


def _alarm(signum, frame):
    raise Timeout("test exceeded 10 seconds")


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    signal.signal(signal.SIGALRM, _alarm)
    try:
        signal.alarm(30)
        load(sys.argv[1], "solution")
        tests = load(sys.argv[2], "model_tests")
    except BaseException:
        print("FAIL <import>\n" + traceback.format_exc(limit=5))
        print("OWN_SUMMARY 0/0 (could not import implementation or tests)")
        return
    finally:
        signal.alarm(0)
    names = sorted(n for n in dir(tests) if n.startswith("test_") and callable(getattr(tests, n)))
    passed = 0
    for name in names:
        signal.alarm(10)
        try:
            getattr(tests, name)()
            passed += 1
            print(f"PASS {name}")
        except BaseException:
            print(f"FAIL {name}\n{traceback.format_exc(limit=5)}")
        finally:
            signal.alarm(0)
    print(f"OWN_SUMMARY {passed}/{len(names)}")


if __name__ == "__main__":
    main()
