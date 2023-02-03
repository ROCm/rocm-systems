#!/usr/bin/env python3

import sys
import json
import argparse


def validate_json(data, labels, counts, depths):
    expected = []
    for litr, citr, ditr in zip(labels, counts, depths):
        entry = []
        _label = litr
        if ditr > 0:
            _label = "{}|_{}".format("  " * (ditr - 1), litr)
        entry = [_label, citr, ditr]
        expected.append(entry)

    for ditr, eitr in zip(data, expected):
        _prefix = ditr["prefix"]
        _depth = ditr["depth"]
        _count = ditr["entry"]["laps"]
        _idx = _prefix.find(">>>")
        if _idx is not None:
            _prefix = _prefix[(_idx + 4) :]
        if _prefix != eitr[0]:
            raise RuntimeError(f"Mismatched prefix: {_prefix} vs. {eitr[0]}")
        if _count != eitr[1]:
            raise RuntimeError(f"Mismatched count for {_prefix}: {_count} vs. {eitr[1]}")
        if _depth != eitr[2]:
            raise RuntimeError(f"Mismatched depth for {_prefix}: {_depth} vs. {eitr[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--metric", type=str, help="JSON metric", required=True)
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

    ret = 0
    with open(args.input) as f:
        data = json.load(f)

        # demo display of data
        if args.print:
            for itr in data["timemory"][args.metric]["ranks"][0]["graph"]:
                _prefix = itr["prefix"]
                _depth = itr["depth"]
                _count = itr["entry"]["laps"]
                _idx = _prefix.find(">>>")
                if _idx is not None:
                    _prefix = _prefix[(_idx + 4) :]

                print("| {:40} | {:6} | {:6} |".format(_prefix, _count, _depth))

        try:
            validate_json(
                data["timemory"][args.metric]["ranks"][0]["graph"],
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
