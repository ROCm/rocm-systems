#!/usr/bin/env python3
"""Analyze bootstrap_net_probe CSV output.

Proves (or refutes) the "variance is transport/network" hypothesis by:
  1. Reporting per-config ring-allgather latency distribution (p50/p99/max, CV).
  2. Correlating slow iterations with kernel TCP counters (RTT, retransmits).
  3. Counting delayed-ACK signature stalls (steps clustered near ~40ms).
  4. Comparing NODELAY/QUICKACK configs: if the tail collapses when Nagle is
     disabled / quick-ack is on, the variance is transport-induced, not CPU.

Usage: analyze_bootstrap_net_probe.py <run_root_dir>
"""
import csv
import math
import os
import statistics as st
import sys


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def cv(xs):
    if not xs or st.mean(xs) == 0:
        return 0.0
    return st.pstdev(xs) / st.mean(xs) * 100.0


def corr(a, b):
    n = min(len(a), len(b))
    if n < 2:
        return None
    a, b = a[:n], b[:n]
    ma, mb = sum(a) / n, sum(b) / n
    va = sum((x - ma) ** 2 for x in a)
    vb = sum((y - mb) ** 2 for y in b)
    if va <= 0 or vb <= 0:
        return None
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / math.sqrt(va * vb)


def load_config(cfg_dir):
    rows = []
    for fn in os.listdir(cfg_dir):
        if not fn.endswith(".csv"):
            continue
        with open(os.path.join(cfg_dir, fn)) as f:
            for r in csv.DictReader(f):
                rows.append(r)
    return rows


def analyze_config(tag, rows):
    iters = [r for r in rows if r["phase"] == "iter"]
    setup = [r for r in rows if r["phase"] == "setup"]
    if not iters:
        print(f"\n## {tag}: no iter rows")
        return None
    ag = [float(r["allgather_us"]) for r in iters]
    step = [float(r["max_step_us"]) for r in iters]
    rtt = [float(r["rtt_us"]) for r in iters]
    rttvar = [float(r["rttvar_us"]) for r in iters]
    retr = [float(r["retrans_delta"]) for r in iters]
    cwnd = [float(r["snd_cwnd"]) for r in iters]
    castate = [int(r["ca_state"]) for r in iters]
    ready = [float(r["ready_us"]) for r in setup]

    # delayed-ACK signature: single step >= 35ms (40ms timer minus slack).
    dack = sum(1 for s in step if s >= 35000.0)
    retr_iters = sum(1 for x in retr if x > 0)
    nonopen_ca = sum(1 for c in castate if c != 0)  # 0 = TCP_CA_Open

    out = {
        "tag": tag,
        "n": len(iters),
        "ag_p50": pct(ag, .5), "ag_p99": pct(ag, .99), "ag_max": max(ag),
        "ag_cv": cv(ag),
        "step_p50": pct(step, .5), "step_p99": pct(step, .99), "step_max": max(step),
        "rtt_p50": pct(rtt, .5), "rtt_p99": pct(rtt, .99),
        "rttvar_p99": pct(rttvar, .99),
        "retr_total": sum(retr), "retr_iters": retr_iters,
        "dack_stalls": dack, "nonopen_ca": nonopen_ca,
        "cwnd_min": min(cwnd) if cwnd else 0,
        "ready_p99_us": pct(ready, .99) if ready else None,
        "corr_ag_rtt": corr(ag, rtt),
        "corr_ag_retr": corr(ag, retr),
        "corr_ag_rttvar": corr(ag, rttvar),
        "corr_step_rtt": corr(step, rtt),
    }
    return out


def fmt(x, nd=0):
    if x is None:
        return "n/a"
    return f"{x:.{nd}f}"


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    root = sys.argv[1]
    cfgs = sorted(
        d for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and not d.startswith(".")
    )
    results = []
    for cfg in cfgs:
        rows = load_config(os.path.join(root, cfg))
        res = analyze_config(cfg, rows)
        if res:
            results.append(res)

    print("# bootstrap_net_probe analysis:", root)
    print("\n## Ring-allgather latency by config (us)")
    print("| config | iters | ag p50 | ag p99 | ag max | ag CV | step p50 | step p99 | step max | ready p99 |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['tag']} | {r['n']} | {fmt(r['ag_p50'])} | {fmt(r['ag_p99'])} | "
              f"{fmt(r['ag_max'])} | {fmt(r['ag_cv'],1)}% | {fmt(r['step_p50'])} | "
              f"{fmt(r['step_p99'])} | {fmt(r['step_max'])} | {fmt(r['ready_p99_us'])} |")

    print("\n## TCP-level evidence by config")
    print("| config | rtt p50 us | rtt p99 us | rttvar p99 us | retrans total | iters w/ retrans | dACK stalls (>=35ms) | non-Open ca_state | min cwnd |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['tag']} | {fmt(r['rtt_p50'])} | {fmt(r['rtt_p99'])} | {fmt(r['rttvar_p99'])} | "
              f"{fmt(r['retr_total'])} | {r['retr_iters']} | {r['dack_stalls']} | "
              f"{r['nonopen_ca']} | {fmt(r['cwnd_min'])} |")

    print("\n## Variance attribution (correlation of slow iterations with TCP counters)")
    print("| config | corr(ag, rtt) | corr(ag, rttvar) | corr(ag, retrans) | corr(step, rtt) |")
    print("|---|---:|---:|---:|---:|")
    for r in results:
        print(f"| {r['tag']} | {fmt(r['corr_ag_rtt'],2)} | {fmt(r['corr_ag_rttvar'],2)} | "
              f"{fmt(r['corr_ag_retr'],2)} | {fmt(r['corr_step_rtt'],2)} |")


if __name__ == "__main__":
    main()
