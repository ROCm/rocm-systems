#!/usr/bin/env python3

import sys
import argparse
from perfetto.trace_processor import TraceProcessor


def validate_perfetto(data, labels, counts, depths):
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

        if _label != eitr[0]:
            raise RuntimeError(f"Mismatched prefix: {_label} vs. {eitr[0]}")
        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth: {_depth} vs. {eitr[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-l", "--labels", nargs="+", type=str, help="Expected labels", default=[]
    )
    parser.add_argument(
        "-c", "--counts", nargs="+", type=int, help="Expected counts", default=[]
    )
    parser.add_argument(
        "-d", "--depths", nargs="+", type=int, help="Expected depths", default=[]
    )
    parser.add_argument(
        "-p", "--print", action="store_true", help="Print the processed perfetto data"
    )
    parser.add_argument("-i", "--input", type=str, help="Input file", required=True)

    args = parser.parse_args()

    if len(args.labels) != len(args.counts) or len(args.labels) != len(args.depths):
        raise RuntimeError(
            "The same number of labels, counts, and depths must be specified"
        )

    tp = TraceProcessor(trace=(args.input))
    pdata = {}
    # get data from perfetto
    qr_it = tp.query("SELECT name, depth FROM slice")
    # loop over data rows from perfetto
    for row in qr_it:
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
            args.labels,
            args.counts,
            args.depths,
        )

    except RuntimeError as e:
        print(f"{e}")
        ret = 1
    if ret == 0:
        print(f"{args.input} validated")
    sys.exit(ret)
