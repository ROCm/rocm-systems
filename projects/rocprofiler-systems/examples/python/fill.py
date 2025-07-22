#!@PYTHON_EXECUTABLE@

import os
import sys
import time
import rocprofsys
from rocprofsys.user import region as omni_user_region
from rocprofsys.profiler import config as omni_config

_prefix = ""


def loop(n):
    pass


@rocprofsys.profile()
def run(i, n, v):
    for l in range(n * n):
        loop(v + l)
    return v + (n * n)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-iterations", help="Number", type=int, default=100)
    parser.add_argument("-v", "--value", help="Starting value", type=int, default=10)
    args = parser.parse_args()

    omni_config.include_args = True
    _prefix = os.path.basename(__file__)
    print(f"[{_prefix}] Executing {args.num_iterations} iterations...\n")
    ans = 0
    for i in range(args.num_iterations):
        beg = ans
        ans = run(i, args.value, beg)
        print(f"[{_prefix}] [{i}] result of run({args.value}, {beg}) = {ans}")
