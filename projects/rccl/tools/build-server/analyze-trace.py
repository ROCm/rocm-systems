#!/usr/bin/env python3
"""Analyze build-trace.json from rccl-build-server.

Reconstructs the task DAG from trace event naming conventions, then computes:
  - Critical path (longest weighted path, lower bound regardless of thread count)
  - Simulated optimal makespan for P threads (greedy list scheduling)
  - Actual wall time
  - Slack percentages

Usage:
  python3 analyze-trace.py build-trace.json [num_threads]
"""

import json
import heapq
import sys
from collections import defaultdict


def load_trace(path):
    with open(path) as f:
        return json.load(f)


def build_dag(events):
    """Build a DAG from trace events using naming conventions.

    Returns (nodes, edges) where:
      nodes: dict of node_id -> duration_us
      edges: dict of node_id -> list of successor node_ids
    """
    by_cat = defaultdict(dict)
    for e in events:
        key = (e["cat"], e.get("name", ""))
        by_cat[e["cat"]][e.get("name", "")] = e["dur"]

    nodes = {}
    edges = defaultdict(list)
    rev_edges = defaultdict(list)

    callee_names = sorted(by_cat.get("callee_fe", {}).keys())
    kernel_names = sorted(by_cat.get("kernel_fe", {}).keys())
    host_names = sorted(by_cat.get("host_compile", {}).keys())

    for name in callee_names:
        fe = f"callee_fe:{name}"
        be = f"callee_be:{name}"
        asm = f"callee_asm:{name}"
        nodes[fe] = by_cat["callee_fe"].get(name, 0)
        nodes[be] = by_cat["callee_be"].get(name, 0)
        nodes[asm] = by_cat["callee_asm"].get(name, 0)
        edges[fe].append(be)
        edges[be].append(asm)

    for name in kernel_names:
        fe = f"kernel_fe:{name}"
        be = f"kernel_be:{name}"
        patch = f"kernel_patch:{name}"
        asm = f"kernel_asm:{name}"
        nodes[fe] = by_cat["kernel_fe"].get(name, 0)
        nodes[be] = by_cat["kernel_be"].get(name, 0)
        nodes[patch] = by_cat["kernel_patch"].get(name, 0)
        nodes[asm] = by_cat["kernel_asm"].get(name, 0)
        edges[fe].append(be)
        edges[be].append(patch)
        # all callee_be -> kernel_patch
        for cn in callee_names:
            edges[f"callee_be:{cn}"].append(patch)
        edges[patch].append(asm)

    # lld_r
    if "lld_r" in by_cat:
        lld_name = list(by_cat["lld_r"].keys())[0]
        lld_id = "lld_r"
        nodes[lld_id] = by_cat["lld_r"][lld_name]
        for cn in callee_names:
            edges[f"callee_asm:{cn}"].append(lld_id)
        for kn in kernel_names:
            edges[f"kernel_asm:{kn}"].append(lld_id)

    # SPLIT chain
    for cat_id, cat_key in [("split_cobj", "split_cobj"),
                             ("split_hipfb", "split_hipfb"),
                             ("split_host", "split_host")]:
        if cat_key in by_cat:
            name = list(by_cat[cat_key].keys())[0]
            nodes[cat_id] = by_cat[cat_key][name]

    if "split_cobj" in nodes:
        edges["lld_r"].append("split_cobj")
    if "split_hipfb" in nodes:
        edges["split_cobj"].append("split_hipfb")
    if "split_host" in nodes:
        edges["split_hipfb"].append("split_host")

    # Host compiles
    for name in host_names:
        hid = f"host_compile:{name}"
        nodes[hid] = by_cat["host_compile"][name]

    # Onerank
    if "host_subprocess" in by_cat:
        name = list(by_cat["host_subprocess"].keys())[0]
        nodes["host_subprocess"] = by_cat["host_subprocess"][name]

    # Final link
    if "final_link" in by_cat:
        name = list(by_cat["final_link"].keys())[0]
        nodes["final_link"] = by_cat["final_link"][name]
        if "split_host" in nodes:
            edges["split_host"].append("final_link")
        for hn in host_names:
            edges[f"host_compile:{hn}"].append("final_link")
        if "host_subprocess" in nodes:
            edges["host_subprocess"].append("final_link")

    # Build reverse edges
    for u, succs in edges.items():
        for v in succs:
            rev_edges[v].append(u)

    return nodes, dict(edges), dict(rev_edges)


def topo_sort(nodes, edges):
    """Kahn's algorithm."""
    in_deg = defaultdict(int)
    for n in nodes:
        in_deg[n] += 0
    for u, succs in edges.items():
        for v in succs:
            in_deg[v] += 1

    queue = [n for n in nodes if in_deg[n] == 0]
    order = []
    while queue:
        n = queue.pop(0)
        order.append(n)
        for s in edges.get(n, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                queue.append(s)
    return order


def critical_path(nodes, edges, rev_edges):
    """Compute the critical path (longest weighted path from source to sink).

    Returns (cp_length_us, cp_nodes).
    """
    order = topo_sort(nodes, edges)

    # dist[n] = longest path ending at n (inclusive)
    dist = {}
    pred = {}
    for n in order:
        best = 0
        best_pred = None
        for p in rev_edges.get(n, []):
            if dist[p] > best:
                best = dist[p]
                best_pred = p
        dist[n] = best + nodes[n]
        pred[n] = best_pred

    sink = max(dist, key=dist.get)
    cp_len = dist[sink]

    path = []
    cur = sink
    while cur is not None:
        path.append(cur)
        cur = pred[cur]
    path.reverse()

    return cp_len, path


def simulate_schedule(nodes, edges, rev_edges, num_workers):
    """Greedy list scheduling with priority = longest-path-to-sink.

    Returns simulated makespan in microseconds.
    """
    order = topo_sort(nodes, edges)

    # Compute longest path to sink for priority
    rev_order = list(reversed(order))
    lp_to_sink = {}
    for n in rev_order:
        best_succ = 0
        for s in edges.get(n, []):
            if lp_to_sink[s] > best_succ:
                best_succ = lp_to_sink[s]
        lp_to_sink[n] = nodes[n] + best_succ

    # In-degree for readiness
    in_deg = defaultdict(int)
    for n in nodes:
        in_deg[n] += 0
    for u, succs in edges.items():
        for v in succs:
            in_deg[v] += 1

    # Ready queue: max-heap by priority (negate for min-heap)
    ready = []
    for n in nodes:
        if in_deg[n] == 0:
            heapq.heappush(ready, (-lp_to_sink[n], n))

    # Worker availability times (min-heap)
    workers = [0.0] * num_workers

    finish_time = {}

    while ready:
        # Pick highest-priority ready task
        _, task = heapq.heappop(ready)

        # Earliest start = max(worker_avail, all predecessors finished)
        earliest = 0.0
        for p in rev_edges.get(task, []):
            if finish_time[p] > earliest:
                earliest = finish_time[p]

        # Assign to earliest-available worker that is free at or after earliest
        worker_avail = heapq.heappop(workers)
        start = max(worker_avail, earliest)
        end = start + nodes[task]
        finish_time[task] = end
        heapq.heappush(workers, end)

        # Check successors
        for s in edges.get(task, []):
            in_deg[s] -= 1
            if in_deg[s] == 0:
                heapq.heappush(ready, (-lp_to_sink[s], s))

    return max(finish_time.values()) if finish_time else 0.0


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <trace.json> [num_threads]")
        sys.exit(1)

    trace_path = sys.argv[1]
    num_threads = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    events = load_trace(trace_path)
    nodes, edges, rev_edges = build_dag(events)

    total_work = sum(nodes.values())
    actual_wall = max(e["ts"] + e["dur"] for e in events) - min(e["ts"] for e in events)

    cp_len, cp_path = critical_path(nodes, edges, rev_edges)
    sim_makespan = simulate_schedule(nodes, edges, rev_edges, num_threads)

    theoretical_lower = max(cp_len, total_work / num_threads)

    print(f"Trace: {trace_path}")
    print(f"Nodes: {len(nodes)}")
    print(f"Threads: {num_threads}")
    print()
    print(f"{'Metric':<30s} {'Time (s)':>10s} {'vs Actual':>10s}")
    print("-" * 52)
    print(f"{'Total work':<30s} {total_work/1e6:>10.1f}")
    print(f"{'Actual wall time':<30s} {actual_wall/1e6:>10.1f} {'':>10s}")
    print(f"{'Critical path':<30s} {cp_len/1e6:>10.1f} {cp_len/actual_wall*100:>9.1f}%")
    print(f"{'Work / threads':<30s} {total_work/num_threads/1e6:>10.1f} "
          f"{total_work/num_threads/actual_wall*100:>9.1f}%")
    print(f"{'Theoretical lower bound':<30s} {theoretical_lower/1e6:>10.1f} "
          f"{theoretical_lower/actual_wall*100:>9.1f}%")
    sim_label = f"Simulated optimal (P={num_threads})"
    print(f"{sim_label:<30s} {sim_makespan/1e6:>10.1f} "
          f"{sim_makespan/actual_wall*100:>9.1f}%")
    print(f"{'Scheduling slack':<30s} "
          f"{(actual_wall - sim_makespan)/1e6:>10.1f} "
          f"{(actual_wall - sim_makespan)/actual_wall*100:>9.1f}%")

    print()
    print(f"Critical path ({len(cp_path)} nodes, {cp_len/1e6:.1f}s):")
    for n in cp_path:
        cat, _, name = n.partition(":")
        dur_s = nodes[n] / 1e6
        print(f"  {cat:<18s} {name:<40s} {dur_s:>8.2f}s")


if __name__ == "__main__":
    main()
