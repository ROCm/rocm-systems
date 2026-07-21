#!/usr/bin/env python3
"""Flatten the device-api test matrix JSON into tab-separated rows for bash.

Rows: "bench_args\\t...", "debug_env\\t-x ...", "suite\\tname\\tenv\\targs\\tbins".
"""
import json
import sys


def _xflags(items):
    return " ".join("-x " + e for e in (items or []))


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    base = data.get("base_env", []) or []
    print("bench_args\t" + (data.get("bench_args", "") or ""))
    print("debug_env\t" + _xflags(data.get("debug_env", [])))
    for s in data.get("suites", []) or []:
        env = base + (s.get("env", []) or [])
        print("\t".join([
            "suite",
            s.get("name", ""),
            _xflags(env),
            s.get("args", "") or "",
            " ".join(s.get("bins", []) or []),
        ]))


if __name__ == "__main__":
    main()
