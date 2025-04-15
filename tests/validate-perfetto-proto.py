#!/usr/bin/env python3

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import sys
import os
import argparse
from perfetto.trace_processor import TraceProcessor, TraceProcessorConfig


def load_trace(inp, max_tries=5, retry_wait=1, bin_path=None):
    """Occasionally connecting to the trace processor fails with HTTP errors
    so this function tries to reduce spurious test failures"""

    tries = 0
    tp = None

    # Check if bin_path is set and if it exists
    print("trace_processor path: ", bin_path)
    if bin_path and not os.path.isfile(bin_path):
        print(f"Path {bin_path} does not exist. Using the default path.")
        bin_path = None

    while tp is None:
        try:
            if bin_path:
                config = TraceProcessorConfig(bin_path=bin_path)
                tp = TraceProcessor(trace=inp, config=config)
            else:
                tp = TraceProcessor(trace=inp)
            break
        except Exception as ex:
            sys.stderr.write(f"{ex}\n")
            sys.stderr.flush()

            if tries >= max_tries:
                raise
            else:
                import time

                time.sleep(retry_wait)
        finally:
            tries += 1
    return tp


def validate_perfetto(data, labels, counts, depths, useSubstringForLabels=False):
    """
    Validates the given perfetto data against expected labels, counts, and depths.

    Args:
        data (list of dict): A list of dictionaries where each dictionary contains
            'label' (str), 'count' (int), and 'depth' (int) keys.
        labels (list of str): A list of expected labels.
        counts (list of int): A list of expected counts corresponding to the labels.
        depths (list of int): A list of expected depths corresponding to the labels.
        useSubstringForLabels (bool): If True, checks if the label in data contains
            the expected label as a substring. If False, checks for exact matches.
    Raises:
        RuntimeError: If any of the labels, counts, or depths in the data do not match
            the expected values.
    """

    if not data and labels:
        raise RuntimeError("Data is empty but labels are not")

    expected = []
    for litr, citr, ditr in zip(labels, counts, depths):
        entry = []
        _label = litr
        if ditr > 0:
            _label = "{}".format(litr)
        entry = [_label, citr, ditr]
        expected.append(entry)

    for ditr, eitr in zip(data, expected):
        _label = ditr["label"]
        _count = ditr["count"]
        _depth = ditr["depth"]

        if useSubstringForLabels:
            if eitr[0] not in _label:
                raise RuntimeError(
                    f"Mismatched prefix: {_label} does not contain {eitr[0]}"
                )
        else:
            if _label != eitr[0]:
                raise RuntimeError(f"Mismatched prefix: {_label} vs. {eitr[0]}")

        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth: {_depth} vs. {eitr[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l",
        "--labels",
        nargs="+",
        type=str,
        help="Expected labels. Not to be used with '-s'",
        default=[],
    )
    parser.add_argument(
        "-c", "--counts", nargs="+", type=int, help="Expected counts", default=[]
    )
    parser.add_argument(
        "-d", "--depths", nargs="+", type=int, help="Expected depths", default=[]
    )
    parser.add_argument(
        "-s",
        "--label-substrings",
        nargs="+",
        type=str,
        help="Expected labels substrings. Not to be used with '-l'",
        default=[],
    )
    parser.add_argument(
        "-m", "--categories", nargs="+", help="Perfetto categories", default=[]
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print the processed perfetto data"
    )
    parser.add_argument("-i", "--input", type=str, help="Input file", required=True)
    parser.add_argument(
        "-t", "--trace_processor_shell", type=str, help="Path of trace_processor_shell"
    )
    parser.add_argument(
        "--key-names",
        type=str,
        help="Require debug args contain a specific key",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--key-counts",
        type=int,
        help="Required number of debug args",
        default=[],
        nargs="*",
    )
    parser.add_argument(
        "--counter-names",
        type=str,
        help="Require counter name in the traces",
        default=[],
        nargs="*",
    )

    args = parser.parse_args()

    # check for mutually exclusive arguments
    if args.labels and args.label_substrings:
        raise RuntimeError(
            "Cannot specify both expected labels and expected label substrings"
        )

    labels = args.labels if args.labels else args.label_substrings

    if len(labels) != len(args.counts) or len(labels) != len(args.depths):
        raise RuntimeError(
            "The same number of labels, counts, and depths must be specified"
        )

    tp = load_trace(args.input, bin_path=args.trace_processor_shell)

    if tp is None:
        raise ValueError(f"trace {args.input} could not be loaded")

    pdata = {}
    # get data from perfetto
    qr_it = tp.query("SELECT name, depth, category FROM slice")
    # loop over data rows from perfetto
    for row in qr_it:
        if args.categories and row.category not in args.categories:
            continue
        if row.name not in pdata:
            pdata[row.name] = {}
        if row.depth not in pdata[row.name]:
            pdata[row.name][row.depth] = 0
        # accumulate the call-count per name and per depth
        pdata[row.name][row.depth] += 1

    perfetto_data = []
    for name, itr in pdata.items():
        for depth, count in itr.items():
            _e = {}
            _e["label"] = name
            _e["count"] = count
            _e["depth"] = depth
            perfetto_data.append(_e)

    # demo display of data
    if args.print:
        print(f"Printing Perfetto Data {args.categories}")
        for itr in perfetto_data:
            n = 0 if itr["depth"] < 2 else itr["depth"] - 1
            lbl = "{}{}{}".format(
                "  " * n, "|_" if itr["depth"] > 0 else "", itr["label"]
            )
            print("| {:40} | {:6} | {:6} |".format(lbl, itr["count"], itr["depth"]))

    ret = 0
    try:
        validate_perfetto(
            perfetto_data,
            labels,
            args.counts,
            args.depths,
            useSubstringForLabels=args.label_substrings is not None,
        )

    except RuntimeError as e:
        print(f"Fail: {e}")
        ret = 1

    for key_name, key_count in zip(args.key_names, args.key_counts):
        slice_args = tp.query(
            f"select * from slice join args using (arg_set_id) where key='debug.{key_name}'"
        )
        count = 0
        if args.print:
            print(f"{key_name} (expected: {key_count}):")
        for row in slice_args:
            count += 1
            if args.print:
                for key, val in row.__dict__.items():
                    print(f"  - {key:20} :: {val}")
        print(f"Number of entries with {key_name} = {count} (expected: {key_count})")
        if key_count != count:
            ret = 1

    for counter_name in args.counter_names:
        sum_counter_values = tp.query(
            f"""SELECT SUM(counter.value) AS total_value FROM counter_track JOIN counter ON
              counter.track_id = counter_track.id WHERE counter_track.name LIKE
              '%{counter_name}%'"""
        )
        total_value = 0

        for row in sum_counter_values:
            total_value = row.total_value if row.total_value is not None else -1

        if args.print:
            print(f"Total value of {counter_name} is {total_value}")

        if total_value <= 0:
            print(f"Fail: Counter {counter_name} is not found in the traces")
            ret = 1

    if ret == 0:
        print(f"{args.input} validated")
    else:
        print(f"Failure validating {args.input}")

    sys.exit(ret)
