# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier:  MIT

"""
Sample PyTorch workload for rocprof-compute --trace-ml-api torch.

    rocprof-compute profile --experimental --trace-ml-api torch --no-roof \
        -n simple_net -- python3 ./simple_net.py

    --user-range         Wrap the first Linear in a user ROCTX range.
                         Requires ROCTX Python.
    --same-line-linear   Two Linear.forward calls from the same source line.
    --no-backward        Skip loss.backward(). Cannot combine with --backward-thread.
    --backward-thread    Run loss.backward() on a worker thread. Cannot combine
                         with --no-backward.
"""

import argparse
import sys
import threading

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import roctx
except ImportError:
    roctx = None


class SimpleNet(nn.Module):
    def __init__(self, same_line_linear=False, user_range=False):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.same_line_linear = same_line_linear
        self.user_range = user_range

    def forward(self, x):
        if self.same_line_linear:
            # two Linear.forward calls on one source line
            # fmt: off
            x = self.fc1(x); x = self.fc2(x)  # noqa: E702
            # fmt: on
            return x
        if self.user_range:
            roctx.rangePush("Fc1_UserDefinedMarker")
            try:
                x = self.fc1(x)
            finally:
                roctx.rangePop()
        else:
            x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="SimpleNet --trace-ml-api torch sample"
    )
    parser.add_argument(
        "--user-range",
        action="store_true",
        help="Wrap the first Linear in a user ROCTX range",
    )
    parser.add_argument(
        "--same-line-linear",
        action="store_true",
        help="Two Linear.forward calls from the same source line",
    )
    exclusive = parser.add_mutually_exclusive_group()
    exclusive.add_argument(
        "--no-backward",
        action="store_true",
        help="Skip loss.backward()",
    )
    exclusive.add_argument(
        "--backward-thread",
        action="store_true",
        help="Run loss.backward() on a worker thread",
    )
    args = parser.parse_args(argv)
    if args.user_range and roctx is None:
        parser.error("--user-range requires ROCTX Python")
    return args


def run_backward(loss, caught):
    """Thread target: run loss.backward() and keep any exception in caught."""
    try:
        loss.backward()
    except BaseException:
        caught.append(sys.exc_info()[1:])


def backward_on_worker_thread(loss):
    """Run loss.backward() on a worker thread and re-raise its exception here."""
    caught = []
    worker = threading.Thread(target=run_backward, args=(loss, caught))
    worker.start()
    worker.join()
    if caught:
        err, tb = caught[0]
        raise err.with_traceback(tb)


def main(argv=None):
    args = parse_args(argv)
    if not torch.cuda.is_available():
        print("GPU is required for this sample. Exiting.")
        sys.exit(1)

    model = SimpleNet(
        same_line_linear=args.same_line_linear,
        user_range=args.user_range,
    ).cuda()
    x = torch.randn(5, 10).cuda()

    for _ in range(1):
        output = model(x)
        loss = output.sum()
        if args.backward_thread:
            backward_on_worker_thread(loss)
        elif not args.no_backward:
            loss.backward()

    print("Training completed")


if __name__ == "__main__":
    main()
