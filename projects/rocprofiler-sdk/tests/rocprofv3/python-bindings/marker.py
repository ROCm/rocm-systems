#!@Python3_EXECUTABLE@

import os
import roctx
import random

from roctx.context_decorators import RoctxRange

_prefix = ""


@RoctxRange("fib")
def fib(n, nmin):
    with RoctxRange(f"fib(n={n})" if n >= nmin else None):
        return n if n < 2 else (fib(n - 1, nmin) + fib(n - 2, nmin))


@RoctxRange("sum")
def _sum(arr):
    with RoctxRange(f"sum(nelem={len(arr)})"):
        return sum(arr)


def inefficient(n):
    roctx.rangePush(f"inefficient({n})")
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    _len = a * n * n
    _arr = [random.random() for _ in range(_len)]
    _ret = _sum(_arr)
    roctx.rangePop()
    return _ret


def run(n):
    idx = roctx.rangeStart(f"run({n})")
    _ret_a = fib(n, max([n / 2, n - 10]))
    _ret_b = inefficient(n)
    roctx.rangeStop(idx)
    return (_ret_a, _ret_b)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-iterations", help="Number", type=int, default=3)
    parser.add_argument("-v", "--value", help="Starting value", type=int, default=20)
    args = parser.parse_args()

    _prefix = os.path.basename(__file__)
    roctx.mark(f"iterations: {args.num_iterations}")
    for i in range(args.num_iterations):
        with RoctxRange("main loop"):
            ans_a, ans_b = run(args.value)
            print(f"[{_prefix}] [{i}] result of run({args.value}) = {ans_a}, {ans_b}\n")
