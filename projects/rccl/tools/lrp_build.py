#!/usr/bin/env python3
"""Longest-Remaining-Path (LRP) build scheduler.

Parses a Ninja build file, computes critical-path priorities using a structural
cost model, and executes all tasks with optimal scheduling via asyncio.

Usage:
    python3 tools/lrp_build.py [build_dir] [-j N] [--target TARGET] [--dry-run]

Defaults: build_dir=., -j=nproc, target=all
"""

import argparse
import asyncio
import os
import re
import sys
import time
from collections import defaultdict
from heapq import heappush, heappop

# ---------------------------------------------------------------------------
# 1. Ninja file parser
# ---------------------------------------------------------------------------

def _read_ninja_lines(path):
    """Read a .ninja file, handling $\\n line continuations and include."""
    result = []
    with open(path) as f:
        buf = ""
        for raw in f:
            raw = raw.rstrip("\n")
            if raw.endswith("$"):
                buf += raw[:-1]
                continue
            line = buf + raw
            buf = ""
            result.append(line)
        if buf:
            result.append(buf)
    return result

class NinjaRule:
    __slots__ = ("name", "command", "description", "depfile", "rspfile",
                 "rspfile_content", "restat")
    def __init__(self, name):
        self.name = name
        self.command = ""
        self.description = ""
        self.depfile = ""
        self.rspfile = ""
        self.rspfile_content = ""
        self.restat = False

class NinjaBuildEdge:
    __slots__ = ("outputs", "implicit_outputs", "rule", "explicit_inputs",
                 "implicit_inputs", "order_only", "variables")
    def __init__(self):
        self.outputs = []
        self.implicit_outputs = []
        self.rule = ""
        self.explicit_inputs = []
        self.implicit_inputs = []
        self.order_only = []
        self.variables = {}

    @property
    def inputs(self):
        return self.explicit_inputs + self.implicit_inputs

    @property
    def all_deps(self):
        return self.explicit_inputs + self.implicit_inputs + self.order_only

def parse_ninja(build_dir):
    """Parse build.ninja (and included files) under *build_dir*."""
    rules = {"phony": NinjaRule("phony")}
    edges = []
    global_vars = {}

    def expand_var(s, local_vars=None):
        def repl(m):
            name = m.group(1) or m.group(2)
            if local_vars and name in local_vars:
                return local_vars[name]
            if name in global_vars:
                return global_vars[name]
            return ""
        return re.sub(r"\$\{(\w+)\}|\$(\w+)", repl, s).replace("$$", "$")

    files_to_parse = [os.path.join(build_dir, "build.ninja")]
    while files_to_parse:
        ninja_path = files_to_parse.pop()
        lines = _read_ninja_lines(ninja_path)
        i = 0
        while i < len(lines):
            line = lines[i]
            stripped = line.lstrip()

            if not stripped or stripped.startswith("#"):
                i += 1
                continue

            if line.startswith("include "):
                inc_path = expand_var(line[8:].strip())
                if not os.path.isabs(inc_path):
                    inc_path = os.path.join(build_dir, inc_path)
                files_to_parse.append(inc_path)
                i += 1
                continue

            if line.startswith("rule "):
                rule = NinjaRule(line[5:].strip())
                i += 1
                while i < len(lines) and lines[i].startswith("  "):
                    kv = lines[i].strip()
                    eq = kv.find("=")
                    if eq != -1:
                        k = kv[:eq].strip()
                        v = kv[eq+1:].strip()
                        if k == "command": rule.command = v
                        elif k == "description": rule.description = v
                        elif k == "depfile": rule.depfile = v
                        elif k == "rspfile": rule.rspfile = v
                        elif k == "rspfile_content": rule.rspfile_content = v
                        elif k == "restat": rule.restat = True
                    i += 1
                rules[rule.name] = rule
                continue

            if line.startswith("build "):
                edge = NinjaBuildEdge()
                rest = line[6:]
                colon_idx = rest.find(": ")
                if colon_idx == -1:
                    i += 1
                    continue
                output_part = rest[:colon_idx]
                after_colon = rest[colon_idx+2:]

                if " | " in output_part:
                    out_str, impl_str = output_part.split(" | ", 1)
                    edge.outputs = out_str.split()
                    edge.implicit_outputs = impl_str.split()
                else:
                    edge.outputs = output_part.split()

                tokens = after_colon.split()
                if tokens:
                    edge.rule = tokens[0]
                    deps_part = tokens[1:]
                else:
                    edge.rule = "phony"
                    deps_part = []

                section = "explicit"
                for tok in deps_part:
                    tok_expanded = expand_var(tok)
                    if tok_expanded == "|":
                        section = "implicit"
                        continue
                    if tok_expanded == "||":
                        section = "order_only"
                        continue
                    if section == "explicit":
                        edge.explicit_inputs.append(tok_expanded)
                    elif section == "implicit":
                        edge.implicit_inputs.append(tok_expanded)
                    else:
                        edge.order_only.append(tok_expanded)

                i += 1
                while i < len(lines) and lines[i].startswith("  "):
                    kv = lines[i].strip()
                    eq = kv.find("=")
                    if eq != -1:
                        k = kv[:eq].strip()
                        v = kv[eq+1:].strip()
                        edge.variables[k] = v
                    i += 1
                edges.append(edge)
                continue

            if "=" in line and not line[0].isspace():
                eq = line.find("=")
                k = line[:eq].strip()
                v = line[eq+1:].strip()
                if k not in ("ninja_required_version", "default"):
                    global_vars[k] = v
                i += 1
                continue

            if line.startswith("default "):
                i += 1
                continue

            i += 1

    return rules, edges, global_vars

def resolve_command(edge, rule, global_vars):
    """Expand a rule's command template with edge + global variables."""
    if rule.name == "phony":
        return None

    local_vars = dict(global_vars)
    local_vars.update(edge.variables)
    local_vars["in"] = " ".join(edge.explicit_inputs)
    local_vars["out"] = " ".join(edge.outputs)
    if edge.explicit_inputs:
        local_vars["in_newline"] = "\n".join(edge.explicit_inputs)
    local_vars.setdefault("TARGET_FILE", " ".join(edge.outputs))

    cmd = rule.command
    changed = True
    for _ in range(5):
        if "$" not in cmd:
            break
        new_cmd = re.sub(
            r"\$\{(\w+)\}|\$(\w+)",
            lambda m: local_vars.get(m.group(1) or m.group(2), ""),
            cmd
        ).replace("$$", "$")
        if new_cmd == cmd:
            break
        cmd = new_cmd

    return cmd

def resolve_description(edge, rule, global_vars):
    """Expand a rule's description template."""
    if rule.name == "phony":
        return f"phony {' '.join(edge.outputs)}"
    local_vars = dict(global_vars)
    local_vars.update(edge.variables)
    local_vars["out"] = " ".join(edge.outputs)
    local_vars["in"] = " ".join(edge.explicit_inputs)
    desc = rule.description
    if not desc:
        desc = edge.variables.get("DESC", edge.outputs[0] if edge.outputs else "")
    return re.sub(
        r"\$\{(\w+)\}|\$(\w+)",
        lambda m: local_vars.get(m.group(1) or m.group(2), ""),
        desc
    ).replace("$$", "$")

# ---------------------------------------------------------------------------
# 2. Structural cost model
# ---------------------------------------------------------------------------

COST_TABLE = [
    # (regex matched against DESC, cost_ms)
    (re.compile(r"SPLIT\[bc\].*premulsum.*bf16.*pipe"), 15000),
    (re.compile(r"SPLIT\[bc\].*(?:premulsum|minmax).*(?:bf16|f16)"), 12000),
    (re.compile(r"SPLIT\[bc\].*(?:premulsum|minmax)"), 9000),
    (re.compile(r"SPLIT\[bc\].*(?:bf16|f16).*pipe"), 9000),
    (re.compile(r"SPLIT\[bc\].*(?:bf16|f16)"), 6000),
    (re.compile(r"SPLIT\[bc\]"), 3300),
    (re.compile(r"SPLIT\[asm\+patch\]"), 3000),
    (re.compile(r"SPLIT\[dev\].*premulsum.*bf16.*pipe"), 12000),
    (re.compile(r"SPLIT\[dev\].*(?:premulsum|minmax).*(?:bf16|f16)"), 10000),
    (re.compile(r"SPLIT\[dev\].*(?:premulsum|minmax)"), 7000),
    (re.compile(r"SPLIT\[dev\].*(?:bf16|f16).*pipe"), 7000),
    (re.compile(r"SPLIT\[dev\].*(?:bf16|f16)"), 5000),
    (re.compile(r"SPLIT\[dev\].*from patched asm"), 1000),
    (re.compile(r"SPLIT\[dev\]"), 2600),
    (re.compile(r"SPLIT\[link\]"), 100),
    (re.compile(r"SPLIT\[hipfb\]"), 150),
    (re.compile(r"SPLIT\[host\]"), 900),
    (re.compile(r"Building CXX object"), 5000),
    (re.compile(r"Linking CXX shared library"), 100),
    (re.compile(r"Linking CXX executable"), 100),
    (re.compile(r"Creating library symlink"), 10),
    (re.compile(r"Hipifying"), 150),
    (re.compile(r"Updating git version"), 50),
]
DEFAULT_COST = 5000

def estimate_cost(desc):
    for pattern, cost in COST_TABLE:
        if pattern.search(desc):
            return cost
    return DEFAULT_COST

# ---------------------------------------------------------------------------
# 3. DAG construction + LRP backward pass
# ---------------------------------------------------------------------------

def build_dag(edges, rules, global_vars, target):
    """Build the dependency DAG.  Returns (nodes, successors, predecessors,
    pred_count, node_cmd, node_desc, node_cost)."""

    output_to_edge = {}
    for edge in edges:
        for out in edge.outputs:
            output_to_edge[out] = edge
        for out in edge.implicit_outputs:
            output_to_edge[out] = edge

    node_cmd = {}
    node_desc = {}
    node_cost = {}
    successors = defaultdict(set)
    predecessors = defaultdict(set)

    visited = set()
    stack = [target]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        edge = output_to_edge.get(node)
        if edge is None:
            node_cmd[node] = None
            node_desc[node] = None
            node_cost[node] = 0
            continue

        primary = edge.outputs[0]
        if primary in node_cmd:
            if node != primary:
                visited.add(node)
            continue

        rule = rules.get(edge.rule)
        cmd = resolve_command(edge, rule, global_vars) if rule else None
        desc = resolve_description(edge, rule, global_vars) if rule else ""
        cost = estimate_cost(desc) if cmd else 0

        node_cmd[primary] = cmd
        node_desc[primary] = desc
        node_cost[primary] = cost

        for alias in edge.outputs[1:] + edge.implicit_outputs:
            node_cmd[alias] = None
            node_desc[alias] = None
            node_cost[alias] = 0
            successors[alias] = set()
            predecessors[primary].discard(alias)

        for dep in edge.all_deps:
            successors[dep].add(primary)
            predecessors[primary].add(dep)
            if dep not in visited:
                stack.append(dep)

    # Include all reachable nodes (including phony) in the schedulable set.
    # Phony nodes are zero-cost instant tasks that propagate dependencies.
    reachable = set()
    stack2 = [target]
    while stack2:
        n = stack2.pop()
        if n in reachable:
            continue
        reachable.add(n)
        edge = output_to_edge.get(n)
        if edge:
            primary = edge.outputs[0]
            reachable.add(primary)
            for dep in edge.all_deps:
                if dep not in reachable:
                    stack2.append(dep)

    schedulable = set()
    for n in reachable:
        if n in node_cmd:
            schedulable.add(n)

    # Build clean predecessor/successor sets within the schedulable set
    clean_succs = defaultdict(set)
    clean_preds = defaultdict(set)
    for n in schedulable:
        for p in predecessors.get(n, set()):
            if p in schedulable:
                clean_preds[n].add(p)
                clean_succs[p].add(n)
            else:
                p_edge = output_to_edge.get(p)
                if p_edge:
                    pp = p_edge.outputs[0]
                    if pp in schedulable and pp != n:
                        clean_preds[n].add(pp)
                        clean_succs[pp].add(n)

    real_pred_count = {}
    for n in schedulable:
        real_pred_count[n] = len(clean_preds[n])

    return (schedulable, clean_succs, clean_preds, real_pred_count,
            node_cmd, node_desc, node_cost)

def compute_lrp(nodes, successors, node_cost):
    """Compute longest-remaining-path for each node (backward pass)."""
    lrp = {}
    in_degree = defaultdict(int)
    for n in nodes:
        for s in successors.get(n, set()):
            if s in nodes:
                in_degree[n]  # touch
                in_degree[s] += 1

    reverse_topo = []
    queue = [n for n in nodes if not successors.get(n, set()) & nodes]
    while queue:
        n = queue.pop()
        reverse_topo.append(n)
        for p in nodes:
            if n in successors.get(p, set()):
                pass

    visited = set()
    topo = []
    def dfs(n):
        if n in visited:
            return
        visited.add(n)
        for s in successors.get(n, set()):
            if s in nodes:
                dfs(s)
        topo.append(n)

    for n in nodes:
        dfs(n)

    for n in topo:
        succ_lrp = 0
        for s in successors.get(n, set()):
            if s in nodes and s in lrp:
                succ_lrp = max(succ_lrp, lrp[s])
        lrp[n] = node_cost.get(n, 0) + succ_lrp

    return lrp

# ---------------------------------------------------------------------------
# 4. Async LRP scheduler
# ---------------------------------------------------------------------------

async def run_build(nodes, successors, predecessors, pred_count, node_cmd,
                    node_desc, node_cost, lrp, max_jobs, build_dir, dry_run):
    executable = {n for n in nodes if node_cmd.get(n) is not None}
    total = len(executable)
    done_count = 0
    failed = []
    start_time = time.monotonic()

    ready = []
    remaining = dict(pred_count)
    seq = 0

    def mark_done(node):
        """Propagate completion of *node* to its successors."""
        nonlocal seq
        for s in successors.get(node, set()):
            if s in remaining:
                remaining[s] -= 1
                if remaining[s] == 0:
                    heappush(ready, (-lrp.get(s, 0), seq, s))
                    seq += 1

    def drain_phony():
        """Complete any phony tasks sitting at the top of the ready queue."""
        while ready and ready[0][2] not in executable:
            _, _, node = heappop(ready)
            mark_done(node)

    for n in nodes:
        if remaining[n] == 0:
            heappush(ready, (-lrp.get(n, 0), seq, n))
            seq += 1

    drain_phony()

    sem = asyncio.Semaphore(max_jobs)
    pending = set()

    async def execute(node):
        nonlocal done_count
        async with sem:
            cmd = node_cmd[node]
            desc = node_desc.get(node, node)
            done_count += 1
            elapsed = time.monotonic() - start_time
            print(f"\r[{done_count}/{total}] ({elapsed:.1f}s) {desc}",
                  flush=True)

            if dry_run:
                return node, True, ""

            proc = await asyncio.create_subprocess_shell(
                cmd,
                cwd=build_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await proc.communicate()
            ok = proc.returncode == 0
            if not ok:
                output = ""
                if stdout:
                    output += stdout.decode(errors="replace")
                if stderr:
                    output += stderr.decode(errors="replace")
                return node, False, output
            return node, True, ""

    stop = False

    async def schedule():
        nonlocal stop
        while (ready or pending) and not stop:
            drain_phony()

            while ready and not stop:
                if ready[0][2] not in executable:
                    drain_phony()
                    continue
                _, _, node = heappop(ready)
                task = asyncio.ensure_future(execute(node))
                pending.add(task)

                if len(pending) >= max_jobs:
                    break

            if not pending:
                break

            done_tasks, _ = await asyncio.wait(
                pending, return_when=asyncio.FIRST_COMPLETED)

            for task in done_tasks:
                pending.discard(task)
                node, ok, output = task.result()
                if not ok:
                    failed.append((node, output))
                    stop = True
                    break
                mark_done(node)

    await schedule()

    if pending:
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)

    wall = time.monotonic() - start_time
    print()

    if failed:
        for node, output in failed:
            desc = node_desc.get(node, node)
            print(f"\nFAILED: {desc}", file=sys.stderr)
            if output:
                print(output, file=sys.stderr)
        print(f"\nBuild FAILED after {wall:.1f}s ({done_count}/{total} tasks completed)",
              file=sys.stderr)
        return False

    print(f"Build OK: {total} tasks in {wall:.1f}s")
    return True

# ---------------------------------------------------------------------------
# 5. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="LRP build scheduler — optimal critical-path scheduling")
    parser.add_argument("build_dir", nargs="?", default=".",
                        help="Ninja build directory (default: .)")
    parser.add_argument("-j", "--jobs", type=int, default=os.cpu_count(),
                        help="Max parallel jobs (default: nproc)")
    parser.add_argument("--target", default="all",
                        help="Build target (default: all)")
    parser.add_argument("--dry-run", "-n", action="store_true",
                        help="Parse and schedule but don't execute")
    args = parser.parse_args()

    build_dir = os.path.abspath(args.build_dir)
    ninja_file = os.path.join(build_dir, "build.ninja")
    if not os.path.exists(ninja_file):
        print(f"Error: {ninja_file} not found", file=sys.stderr)
        sys.exit(1)

    t0 = time.monotonic()
    print(f"Parsing {ninja_file} ...", flush=True)
    rules, edges, global_vars = parse_ninja(build_dir)
    t1 = time.monotonic()
    print(f"  {len(edges)} build edges, {len(rules)} rules in {t1-t0:.2f}s")

    print("Building DAG ...", flush=True)
    (nodes, succs, preds, pred_count,
     node_cmd, node_desc, node_cost) = build_dag(
        edges, rules, global_vars, args.target)
    t2 = time.monotonic()

    executable = [n for n in nodes if node_cmd.get(n) is not None]
    phony_count = len(nodes) - len(executable)
    print(f"  {len(executable)} executable tasks, {phony_count} phony ({len(nodes)} total), in {t2-t1:.2f}s")

    print("Computing LRP priorities ...", flush=True)
    lrp = compute_lrp(nodes, succs, node_cost)
    t3 = time.monotonic()

    top_lrp = sorted(
        [n for n in nodes if node_cmd.get(n) is not None],
        key=lambda n: lrp.get(n, 0), reverse=True)
    print(f"  Longest path: {lrp.get(top_lrp[0], 0)/1000:.1f}s (estimated)")
    print(f"  Top 5:")
    for n in top_lrp[:5]:
        desc = node_desc.get(n, n)
        if desc and len(desc) > 80:
            desc = desc[:77] + "..."
        print(f"    {lrp.get(n,0)/1000:.1f}s  {desc}")
    print(f"  LRP computed in {t3-t2:.2f}s")

    print(f"\nStarting build with {args.jobs} jobs{' (dry run)' if args.dry_run else ''} ...\n",
          flush=True)

    ok = asyncio.run(run_build(
        nodes, succs, preds, pred_count, node_cmd, node_desc, node_cost,
        lrp, args.jobs, build_dir, args.dry_run))

    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
