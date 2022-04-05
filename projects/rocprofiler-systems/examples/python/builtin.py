#!@PYTHON_EXECUTABLE@

import sys
import numpy as np


def fib(n):
    return n if n < 2 else (fib(n - 1) + fib(n - 2))


def inefficient(n):
    print(f"inefficient: {n}")
    a = 0
    for i in range(n):
        a += i
        for j in range(n):
            a += j
    arr = np.random.rand(a * n * n * n)
    sum = arr.sum()
    print(f"sum: {sum}")
    return sum


@profile
def run(nfib):
    ret = 0
    ret += fib(nfib)
    ret += inefficient(nfib)
    return ret


if __name__ == "__main__":
    nfib = int(sys.argv[1]) if len(sys.argv) > 1 else 20
    for i in range(5):
        ans = run(nfib)
        print(f"[{i}] fibonacci({nfib}) = {ans}")
