#!@PYTHON_EXECUTABLE@

import os
import sys
import omnitrace

_prefix = ""


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


try:
    import numpy as np

    def inefficient(n):
        print(f"[{_prefix}] ... running inefficient({n})")
        a = 0
        for i in range(n):
            a += i
            for j in range(n):
                a += j
        _len = a * n * n * n
        _ret = np.random.rand(_len).sum()
        print(f"[{_prefix}] ... sum of {_len} random elements: {_ret}")
        return _ret

except ImportError:
    import random

    def _sum(arr):
        return sum(arr)

    def inefficient(n):
        print(f"[{_prefix}] ... running inefficient({n})")
        a = 0
        for i in range(n):
            a += i
            for j in range(n):
                a += j
        _len = a * n * n * n
        _arr = [random.random() for _ in range(_len)]
        _ret = _sum(_arr)
        print(f"[{_prefix}] ... sum of {_len} random elements: {_ret}")
        return _ret


@omnitrace.profile()
def run(n):
    _ret = 0
    _ret += fib(n)
    _ret += inefficient(n)
    return _ret


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--num-iterations", help="Number", type=int, default=3
    )
    parser.add_argument(
        "-v", "--value", help="Starting value", type=int, default=20
    )
    args = parser.parse_args()

    _prefix = os.path.basename(__file__)
    print(f"[{_prefix}] Executing {args.num_iterations} iterations...\n")
    for i in range(args.num_iterations):
        ans = run(args.value)
        print(f"[{_prefix}] [{i}] result of run({args.value}) = {ans}\n")
