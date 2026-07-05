#!/usr/bin/env python3

# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

import sys
import os
import time
import argparse
from collections import defaultdict

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

    if len(labels) != len(counts) or len(labels) != len(depths):
        raise RuntimeError(
            "labels, counts, and depths must have the same length "
            f"(got {len(labels)}, {len(counts)}, {len(depths)})"
        )

    expected = [[litr, citr, ditr] for litr, citr, ditr in zip(labels, counts, depths)]

    for ditr, eitr in zip(data, expected):
        _label = ditr["label"]
        _count = ditr["count"]
        _depth = ditr["depth"]

        if useSubstringForLabels:
            if eitr[0] not in _label:
                raise RuntimeError(
                    f"Mismatched label (substring): {_label!r} does not contain {eitr[0]!r}"
                )
        else:
            if _label != eitr[0]:
                raise RuntimeError(
                    f"Mismatched label (exact): {_label!r} vs expected {eitr[0]!r}"
                )

        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth: {_depth} vs. {eitr[2]}")


def validate_perfetto_by_label(
    data,
    labels,
    counts,
    useSubstringForLabels=False,
):
    """
    Validate slice rows by matching each expected label to trace names (aggregate mode).

    For each expected label, find matching slice rows in ``data`` by label (exact or
    substring). Matching slice counts are summed **across all depths** so stack depth is
    not part of validation. Trace rows whose names do not match any expected label are
    ignored.

    If ``counts`` is empty: require **at least one** matching slice per label (presence).

    If ``counts`` is non-empty: it must parallel ``labels``, and the summed occurrence
    count must equal each expected integer (exact match).

    Slice ``count`` values come from Perfetto aggregation: number of slice records for
    that kernel name at that depth; summing yields total kernel dispatches for that name.
    """
    presence_only = len(counts) == 0
    if not presence_only and len(counts) != len(labels):
        raise RuntimeError(
            "counts must have one entry per label, or be omitted for presence-only mode"
        )

    totals_by_slice_name = defaultdict(int)
    for srow in data:
        totals_by_slice_name[srow["label"]] += srow["count"]

    for i, litr in enumerate(labels):
        if useSubstringForLabels:
            total = sum(cnt for name, cnt in totals_by_slice_name.items() if litr in name)
        else:
            total = totals_by_slice_name.get(litr, 0)

        if presence_only:
            if total < 1:
                raise RuntimeError(f"No slice found for expected label '{litr}'")
            continue

        citr = counts[i]
        if total != citr:
            raise RuntimeError(
                f"Mismatched count for expected label '{litr}': "
                f"got {total}, expected {citr}"
            )


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
        "-d",
        "--depths",
        nargs="+",
        type=int,
        help="Expected depths (positional mode). Omit for aggregate-by-name mode.",
        default=[],
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
    parser.add_argument(
        "--check-counter-pairing",
        action="store_true",
        help="Verify each counter track has paired start/end entries (even count, last value is 0)",
    )

    args = parser.parse_args()

    # check for mutually exclusive arguments
    if args.labels and args.label_substrings:
        raise RuntimeError(
            "Cannot specify both expected labels and expected label substrings"
        )

    labels = args.labels if args.labels else args.label_substrings
    aggregate_by_name = not args.depths

    if labels:
        if aggregate_by_name:
            count_mode = "presence-only" if not args.counts else "exact counts per label"
            print(
                "Perfetto slice validation mode: aggregate-by-name "
                f"(sum counts across depths, ignore unmatched slices, {count_mode})"
            )
            if args.counts and len(args.counts) != len(labels):
                raise RuntimeError(
                    "With -d omitted, provide no -c (presence-only) or one count per label"
                )
        else:
            print(
                "Perfetto slice validation mode: positional "
                "(match label, count, and depth per trace row, in order)"
            )
            if len(labels) != len(args.counts) or len(labels) != len(args.depths):
                raise RuntimeError(
                    "The same number of labels, counts, and depths must be specified "
                    "when -d is provided"
                )

    if args.key_names or args.key_counts:
        if len(args.key_names) != len(args.key_counts):
            raise RuntimeError(
                "--key-names and --key-counts must have the same number of entries"
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
        use_substrings = bool(args.label_substrings)
        if labels:
            if aggregate_by_name:
                validate_perfetto_by_label(
                    perfetto_data,
                    labels,
                    args.counts,
                    useSubstringForLabels=use_substrings,
                )
            else:
                validate_perfetto(
                    perfetto_data,
                    labels,
                    args.counts,
                    args.depths,
                    useSubstringForLabels=use_substrings,
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

    if args.counter_names and args.print:
        all_counter_tracks = tp.query(
            "SELECT DISTINCT name FROM counter_track ORDER BY name"
        )
        track_names = [row.name for row in all_counter_tracks]
        print(f"Available counter tracks ({len(track_names)}):")
        for name in track_names:
            print(f"  - {name}")

    for counter_name in args.counter_names:
        if args.print:
            matching_tracks = tp.query(
                f"""SELECT counter_track.name, COUNT(counter.id) AS num_entries,
                  SUM(counter.value) AS sum_value, MIN(counter.value) AS min_value,
                  MAX(counter.value) AS max_value
                  FROM counter_track JOIN counter ON counter.track_id = counter_track.id
                  WHERE counter_track.name LIKE '%{counter_name}%'
                  GROUP BY counter_track.name ORDER BY counter_track.name"""
            )
            track_rows = []
            for row in matching_tracks:
                track_rows.append(row)
                print(
                    f"  Track: {row.name} | entries={row.num_entries} "
                    f"sum={row.sum_value} min={row.min_value} max={row.max_value}"
                )
            if not track_rows:
                print(f"  No counter tracks matching '%{counter_name}%' found in trace")

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

    if args.check_counter_pairing and args.counter_names:
        for counter_name in args.counter_names:
            tracks = tp.query(f"""SELECT counter_track.id, counter_track.name,
                  COUNT(counter.id) AS num_entries
                  FROM counter_track JOIN counter ON counter.track_id = counter_track.id
                  WHERE counter_track.name LIKE '%{counter_name}%'
                  GROUP BY counter_track.id""")
            for row in tracks:
                if row.num_entries % 2 != 0:
                    print(
                        f"Fail: Counter track '{row.name}' has {row.num_entries} entries "
                        f"(expected even number for paired start/end)"
                    )
                    ret = 1
                else:
                    last_value = tp.query(f"""SELECT counter.value FROM counter
                          WHERE counter.track_id = {row.id}
                          ORDER BY counter.ts DESC LIMIT 1""")
                    for val_row in last_value:
                        if val_row.value != 0:
                            print(
                                f"Fail: Counter track '{row.name}' last value is "
                                f"{val_row.value} (expected 0 for end marker)"
                            )
                            ret = 1

    if ret == 0:
        print(f"{args.input} validated")
    else:
        print(f"Failure validating {args.input}")

    sys.exit(ret)
