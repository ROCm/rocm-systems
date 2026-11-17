# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Select new heads of PRs and local branches and run rj_isa_diff_pr on each."""

import fnmatch
import json
import os
import shutil
import signal
import subprocess
import time

import ci_lib as L
import comment


def touches(paths, prefixes):
    return any(p.startswith(x) for p in paths for x in prefixes)


def list_prs(cfg):
    prs = []
    for author in cfg["authors"]:
        prs += json.loads(
            L.gh(
                [
                    "pr",
                    "list",
                    "-R",
                    cfg["repo"],
                    "--author",
                    author,
                    "--state",
                    "open",
                    "--limit",
                    "200",
                    "--json",
                    "number,title,headRefOid,updatedAt,files",
                ]
            )
        )
    return prs


def pr_subjects(prs, cfg, tier):
    """Codegen PRs, most recently updated first, whose head has no finished @p tier run."""
    subs = []
    for p in sorted(prs, key=lambda p: p["updatedAt"], reverse=True):
        if not touches([f["path"] for f in p.get("files") or []], cfg["paths"]):
            continue
        if L.finished(L.run_dir("pr", p["number"], p["headRefOid"], tier)):
            continue
        subs.append(
            {
                "kind": "pr",
                "id": p["number"],
                "title": p["title"],
                "head": p["headRefOid"],
            }
        )
    return subs


def fetch_main():
    L.git(L.WORK / "llvm-project", "fetch", "--quiet", L.UPSTREAM, "+main:refs/ci/main")


def resolve_pr(s):
    """Fetch the PR head; its base is the merge base with main, as on GitHub."""
    repo = L.WORK / "llvm-project"
    L.git(
        repo,
        "fetch",
        "--quiet",
        L.UPSTREAM,
        f"+pull/{s['id']}/head:refs/ci/pr/{s['id']}",
    )
    s["head"] = L.git(repo, "rev-parse", f"refs/ci/pr/{s['id']}").strip()
    s["base"] = L.git(repo, "merge-base", s["head"], "refs/ci/main").strip()


def branch_subjects(cfg, tier):
    """Branches of the host clone matching cfg["local_branches"] with a new codegen head.
    A head that changes no codegen path is recorded as NOT_CODEGEN and not examined again.
    """
    if not cfg["local_branches"] or not (L.HOST_LLVM / ".git").exists():
        return []
    repo = L.WORK / "llvm-project"
    refs = L.git(
        L.HOST_LLVM,
        "for-each-ref",
        "--format=%(refname:short) %(objectname)",
        "refs/heads/",
    )
    subs = []
    for line in refs.splitlines():
        name, sha = line.split()
        if not any(fnmatch.fnmatchcase(name, pat) for pat in cfg["local_branches"]):
            continue
        d = L.run_dir("branch", name, sha, tier)
        if L.finished(d):
            continue
        L.git(
            repo,
            "fetch",
            "--quiet",
            str(L.HOST_LLVM),
            f"+refs/heads/{name}:refs/local/{name}",
        )
        base = L.git(repo, "merge-base", sha, "refs/ci/main").strip()
        s = {"kind": "branch", "id": name, "title": name, "head": sha, "base": base}
        if not touches(
            L.git(repo, "diff", "--name-only", base, sha).split(), cfg["paths"]
        ):
            L.write_run(d, dict(s, tier=tier, status="NOT_CODEGEN", finished=L.now()))
            continue
        subs.append(s)
    return subs


def build_dir(s):
    # A PR keeps its own build (a force-push rebuilds little); local branches share one.
    name = f"pr-{s['id']}" if s["kind"] == "pr" else "branches"
    return L.WORK / "builds" / name


def driver_argv(s, cfg, tier, out):
    argv = [
        str(L.TOOL_BIN / "rj_isa_diff_pr"),
        "--repo",
        str(L.WORK / "llvm-project"),
        "--base",
        s["base"],
        "--head",
        s["head"],
        "--build-llc",
        str(build_dir(s)),
        "--out",
        str(out),
        "--jobs",
        str(cfg["jobs"]),
        "--job-timeout",
        str(cfg["job_timeout"][tier]),
    ]
    corpus = L.WORK / "corpus"
    if tier == "deep":
        argv.append("--all-tests")
        if corpus.is_dir() and any(corpus.iterdir()):
            argv += [
                "--corpus",
                str(corpus),
                "--corpus-mcpu",
                cfg["corpus_mcpu"],
                f"--corpus-args={cfg['corpus_args']}",
            ]
    return argv


def run_driver(argv, log_path, timeout_s):
    """The driver's exit code, or None if it timed out (its whole process group is killed)."""
    with open(log_path, "w") as log:
        p = subprocess.Popen(
            argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return p.wait(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
            return None


def run_subject(s, cfg, tier):
    """Run one (subject, head, tier) once; return its run record (None if done before)."""
    d = L.run_dir(s["kind"], s["id"], s["head"], tier)
    if L.finished(d):
        return None
    prev = L.latest_run(s["kind"], s["id"], tier)
    if d.exists():
        shutil.rmtree(d)  # an attempt that did not finish
    rec = {
        "kind": s["kind"],
        "id": s["id"],
        "title": s["title"],
        "head": s["head"],
        "base": s["base"],
        "tier": tier,
        "tool": os.environ.get("ISADIFF_TOOL_SHA", "unknown"),
        "started": L.now(),
    }
    L.write_run(d, rec)
    rc = run_driver(
        driver_argv(s, cfg, tier, d / "out"),
        d / "driver.log",
        cfg["run_timeout_hours"][tier] * 3600,
    )
    summary = d / "out" / "summary.json"
    rec["exit"] = rc
    if summary.exists():
        rec["status"] = json.loads(summary.read_text())["status"]
    else:
        rec["status"] = "TIMEOUT" if rc is None else "ERROR"
    rec["finished"] = L.now()
    L.write_run(d, rec)
    if prev is None or prev.get("status") != rec["status"]:
        with open(L.WORK / "results" / "events.log", "a") as f:
            f.write(
                f"{rec['finished']} {s['kind']}-{L.slug(s['id'])} {tier} "
                f"{prev.get('status') if prev else 'new'} -> {rec['status']} "
                f"(head {s['head'][:12]})\n"
            )
    comment.write_draft(d)
    rec["dir"] = str(d)
    if comment.may_auto_post(cfg, rec):
        comment.post_run(cfg, rec)
    return rec


def write_index():
    res = L.WORK / "results"
    rows = []
    for sd in sorted(p for p in res.iterdir() if p.is_dir()):
        for tier in ("quick", "deep"):
            runs = [L.read_run(p.parent) for p in sd.glob(f"*-{tier}/run.json")]
            runs = [
                r
                for r in runs
                if r.get("finished") and r.get("status") != "NOT_CODEGEN"
            ]
            if runs:
                rows.append(max(runs, key=lambda r: r["finished"]))
    rows.sort(key=lambda r: r["finished"], reverse=True)
    (res / "index.json").write_text(json.dumps(rows, indent=1) + "\n")
    lines = [
        "| subject | tier | head | status | finished | report |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        name = f"#{r['id']} {r['title'][:60]}" if r["kind"] == "pr" else r["id"]
        rel = os.path.relpath(r["dir"], res)
        lines.append(
            f"| {name.replace('|', '/')} | {r['tier']} | `{r['head'][:12]}` "
            f"| {r['status']} | {r['finished']} | {rel}/out/report.md |"
        )
    (res / "index.md").write_text("\n".join(lines) + "\n")


def prune(open_prs):
    """Delete the builds of PRs that are no longer open."""
    keep = {f"pr-{p['number']}" for p in open_prs}
    for b in (L.WORK / "builds").glob("pr-*"):
        if b.name not in keep:
            shutil.rmtree(b)
    L.git(L.WORK / "llvm-project", "worktree", "prune")


def scan(cfg, tier, pr=None):
    start = time.time()
    fetch_main()
    if pr is not None:
        p = json.loads(
            L.gh(
                [
                    "pr",
                    "view",
                    str(pr),
                    "-R",
                    cfg["repo"],
                    "--json",
                    "number,title,headRefOid",
                ]
            )
        )
        subs = [
            {
                "kind": "pr",
                "id": p["number"],
                "title": p["title"],
                "head": p["headRefOid"],
            }
        ]
    else:
        prs = list_prs(cfg)
        if tier == "deep":
            prune(prs)
        subs = pr_subjects(prs, cfg, tier) + branch_subjects(cfg, tier)
    for i, s in enumerate(subs):
        if time.time() - start > cfg["budget_hours"][tier] * 3600:
            print(
                f"{L.now()} budget reached; {len(subs) - i} left for the next scan",
                flush=True,
            )
            break
        if s["kind"] == "pr":
            resolve_pr(s)
        print(f"{L.now()} {tier} {s['kind']} {s['id']} {s['head'][:12]}", flush=True)
        rec = run_subject(s, cfg, tier)
        if rec:
            print(f"{L.now()}   {rec['status']} {rec['dir']}", flush=True)
    write_index()
