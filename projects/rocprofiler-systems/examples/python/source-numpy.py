#!@PYTHON_EXECUTABLE@

import os
import sys
import time
import rocprofsys
from rocprofsys.user import region as omni_user_region

_prefix = ""


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


try:
    import numpy as np

    def inefficient(n):
        print(f"[{_prefix}] ... running inefficient({n}) (1)")
        a = 0
        for i in range(n):
            a += i
            for j in range(n):
                a += j
        _len = a * n * n
        _ret = np.random.rand(_len).sum()
        print(f"[{_prefix}] ... sum of {_len} random elements: {_ret}")
        return _ret

except ImportError as e:
    print(f"ImportError: {e}")
    import random

    def _sum(arr):
        print(f"----  in _sum")
        return sum(arr)

    def inefficient(n):
        print(f"[{_prefix}] ... running inefficient({n})")
        a = 0
        for i in range(n):
            a += i
            for j in range(n):
                a += j
        _len = a * n * n
        _arr = [random.random() for _ in range(_len)]
        _ret = _sum(_arr)
        print(f"[{_prefix}] ... sum of {_len} random elements: {_ret}")
        return _ret


@rocprofsys.profile()
def run(n):
    _ret = 0
    _ret += fib(n)
    _ret += inefficient(n)
    return _ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-iterations", help="Number", type=int, default=3)
    parser.add_argument("-v", "--value", help="Starting value", type=int, default=20)
    parser.add_argument(
        "-s",
        "--stop-profile",
        help="Stop tracing after given iterations",
        type=int,
        default=0,
    )
    args = parser.parse_args()

    _prefix = os.path.basename(__file__)
    print(f"[{_prefix}] Executing {args.num_iterations} iterations...\n")
    for i in range(args.num_iterations):
        with omni_user_region(f"main_loop"):
            if args.stop_profile > 0 and i == args.stop_profile:
                rocprofsys.user.stop_trace()
            ans = run(args.value)
            print(f"[{_prefix}] [{i}] result of run({args.value}) = {ans}\n")
