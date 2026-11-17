# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Unit tests of isadiff-ci (no Docker, network or GitHub needed)."""

import json
import stat
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ci_lib as L  # noqa: E402

CFG = L.load_config("/nonexistent/config.json")


@pytest.fixture
def work(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "WORK", tmp_path)
    (tmp_path / "results").mkdir()
    return tmp_path


def put_run(kind, ident, head, tier, **fields):
    d = L.run_dir(kind, ident, head, tier)
    rec = {
        "kind": kind,
        "id": ident,
        "title": "T",
        "head": head,
        "base": "b" * 40,
        "tier": tier,
        "tool": "c" * 40,
    }
    rec.update(fields)
    L.write_run(d, rec)
    return d


def test_config_defaults_and_override(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps({"jobs": 4}))
    cfg = L.load_config(p)
    assert cfg["jobs"] == 4 and cfg["repo"] == "llvm/llvm-project"
    assert CFG["post"]["mode"] == "manual"


def test_run_dir_layout(work):
    d = L.run_dir("branch", "users/lialan/x", "0123456789abcdef", "deep")
    assert d == work / "results" / "branch-users_lialan_x" / "0123456789ab-deep"


def test_latest_run_ignores_unfinished(work):
    put_run(
        "pr", 7, "a" * 40, "quick", status="PASS", finished="2026-11-17T01:00:00-08:00"
    )
    put_run("pr", 7, "b" * 40, "quick", started="2026-11-17T02:00:00-08:00")
    assert L.latest_run("pr", 7)["status"] == "PASS"
    assert L.finished(L.run_dir("pr", 7, "a" * 40, "quick"))
    assert not L.finished(L.run_dir("pr", 7, "b" * 40, "quick"))
    assert L.latest_run("pr", 7, "deep") is None


import scan  # noqa: E402


def pr(n, head, updated, paths):
    return {
        "number": n,
        "title": f"PR {n}",
        "headRefOid": head,
        "updatedAt": updated,
        "files": [{"path": p} for p in paths],
    }


def test_touches_matches_prefixes():
    assert scan.touches(["llvm/lib/Target/AMDGPU/SIISelLowering.cpp"], CFG["paths"])
    assert not scan.touches(
        ["llvm/lib/Transforms/InstCombine/InstCombineCasts.cpp"], CFG["paths"]
    )


def test_pr_subjects_new_codegen_heads_newest_first(work):
    prs = [
        pr(1, "a" * 40, "2026-11-01T00:00:00Z", ["llvm/lib/CodeGen/X.cpp"]),
        pr(2, "b" * 40, "2026-11-02T00:00:00Z", ["llvm/test/CodeGen/AMDGPU/y.ll"]),
        pr(3, "c" * 40, "2026-11-03T00:00:00Z", ["mlir/lib/Z.cpp"]),
    ]
    put_run(
        "pr", 1, "a" * 40, "quick", status="PASS", finished="2026-11-17T00:00:00-08:00"
    )
    assert [s["id"] for s in scan.pr_subjects(prs, CFG, "quick")] == [2]
    assert [s["id"] for s in scan.pr_subjects(prs, CFG, "deep")] == [2, 1]


def sh(cwd, *args):
    return subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.name=t", "-c", "user.email=t@t", *args],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_branch_subjects_selects_codegen_nightly_branches(work, tmp_path, monkeypatch):
    host = tmp_path / "host"
    host.mkdir()
    sh(host, "init", "-q", "-b", "main")
    sh(host, "commit", "-q", "--allow-empty", "-m", "base")
    heads = {}
    for name, path in (
        ("users/lialan/selectiondag_nightly_a", "llvm/lib/CodeGen/A.cpp"),
        ("users/lialan/selectiondag_nightly_b", "llvm/docs/B.rst"),
        ("users/lialan/other", "llvm/lib/CodeGen/C.cpp"),
    ):
        sh(host, "checkout", "-q", "-b", name, "main")
        (host / path).parent.mkdir(parents=True, exist_ok=True)
        (host / path).write_text("x\n")
        sh(host, "add", path)
        sh(host, "commit", "-q", "-m", name)
        heads[name] = sh(host, "rev-parse", "HEAD")
    sh(host, "checkout", "-q", "main")
    monkeypatch.setattr(L, "HOST_LLVM", host)
    subprocess.run(
        ["git", "clone", "-q", "--no-checkout", str(host), str(work / "llvm-project")],
        check=True,
    )
    sh(work / "llvm-project", "update-ref", "refs/ci/main", "refs/heads/main")

    subs = scan.branch_subjects(CFG, "quick")

    assert [s["id"] for s in subs] == ["users/lialan/selectiondag_nightly_a"]
    assert subs[0]["head"] == heads["users/lialan/selectiondag_nightly_a"]
    docs = L.read_run(
        L.run_dir(
            "branch",
            "users/lialan/selectiondag_nightly_b",
            heads["users/lialan/selectiondag_nightly_b"],
            "quick",
        )
    )
    assert docs["status"] == "NOT_CODEGEN"  # recorded, so it is not examined again


import comment  # noqa: E402

FAKE_DRIVER = '''#!/usr/bin/env python3
import json, os, sys, time
from pathlib import Path
def verdict(row):
    return row["v"]
def poison_sensitive(row):
    return row.get("poison", False)
if __name__ == "__main__":
    if os.environ.get("FAKE_SLEEP"):
        time.sleep(60)
    out = Path(sys.argv[sys.argv.index("--out") + 1])
    out.mkdir(parents=True)
    rows = [
        {"v": "mismatch", "file": "llvm/test/CodeGen/AMDGPU/t.ll", "function": "f",
         "target": "gfx950", "regions": [{"status": "LOCAL_MISMATCH",
                                          "first_mismatch": "v0 lane 1: old 0x1 new 0x2"}]},
        {"v": "pass", "file": "llvm/test/CodeGen/AMDGPU/t.ll", "function": "g",
         "target": "gfx950", "regions": []},
    ]
    (out / "summary.json").write_text(json.dumps(
        {"status": "MISMATCH", "functions": rows, "skipped": [], "argv": sys.argv}))
    sys.exit(1)
'''


@pytest.fixture
def fake_tool(work, monkeypatch):
    b = work / "bin"
    b.mkdir()
    p = b / "rj_isa_diff_pr"
    p.write_text(FAKE_DRIVER)
    p.chmod(p.stat().st_mode | stat.S_IXUSR)
    monkeypatch.setattr(L, "TOOL_BIN", b)
    monkeypatch.setattr(L, "_driver", None)
    return b


REC = {
    "kind": "pr",
    "id": 1,
    "head": "a" * 40,
    "base": "b" * 40,
    "tier": "quick",
    "tool": "c" * 40,
    "status": "MISMATCH",
}


def test_render_lists_functions_that_did_not_pass(fake_tool):
    rows = [
        {
            "v": "mismatch",
            "file": "llvm/test/CodeGen/AMDGPU/t.ll",
            "function": "f",
            "target": "gfx950",
            "poison": True,
            "regions": [
                {
                    "status": "LOCAL_MISMATCH",
                    "resolved_by": "r0.ctx",
                    "first_mismatch": "old",
                },
                {"status": "LOCAL_MISMATCH", "first_mismatch": "v1 | lane 3"},
            ],
        },
        {
            "v": "pass",
            "file": "t.ll",
            "function": "g",
            "target": "gfx950",
            "regions": [],
        },
    ]
    text = comment.render(
        REC, {"functions": rows, "skipped": [{"file": "x.ll", "reason": "r"}]}
    )
    assert text.startswith(L.MARKER)
    assert (
        "`aaaaaaaaaaaa`" in text and "`bbbbbbbbbbbb`" in text and "**MISMATCH**" in text
    )
    assert "1 mismatch, 1 pass" in text
    assert (
        "| mismatch (poison-sensitive) | `t.ll` | gfx950 | `f` | v1 \\| lane 3 |"
        in text
    )
    assert "`g`" not in text and "Skipped: 1 file" in text


def test_render_without_summary(fake_tool):
    text = comment.render(dict(REC, status="TIMEOUT"), None)
    assert "**TIMEOUT**: the run exceeded its time limit." in text


def test_render_truncates_long_tables(fake_tool):
    rows = [
        {
            "v": "inconclusive",
            "file": "t.ll",
            "function": f"f{i}",
            "target": "gfx950",
            "regions": [],
            "notes": ["x" * 150],
        }
        for i in range(2000)
    ]
    text = comment.render(
        dict(REC, status="INCOMPLETE"), {"functions": rows, "skipped": []}
    )
    assert len(text) <= comment.LIMIT and "more in the full report" in text


def fake_gh_factory(calls, existing):
    def fake_gh(args, input=None):
        calls.append((args, input))
        if args[:2] == ["api", "user"]:
            return "lialan\n"
        if "--paginate" in args:
            return existing
        return "https://github.com/llvm/llvm-project/pull/1#issuecomment-123\n"

    return fake_gh


def test_upsert_edits_existing_comment(work, monkeypatch):
    calls = []
    monkeypatch.setattr(L, "gh", fake_gh_factory(calls, "123\n"))
    url = comment.upsert("llvm/llvm-project", 1, work / "c.md")
    assert calls[-1][0][:4] == [
        "api",
        "-X",
        "PATCH",
        "repos/llvm/llvm-project/issues/comments/123",
    ]
    assert f"body=@{work / 'c.md'}" in calls[-1][0] and url.endswith("issuecomment-123")


def test_upsert_posts_when_there_is_none(work, monkeypatch):
    calls = []
    monkeypatch.setattr(L, "gh", fake_gh_factory(calls, ""))
    comment.upsert("llvm/llvm-project", 1, work / "c.md")
    assert calls[-1][0][:4] == [
        "api",
        "-X",
        "POST",
        "repos/llvm/llvm-project/issues/1/comments",
    ]
    find = next(a for a, _ in calls if "--paginate" in a)
    assert 'select(.user.login == "lialan"' in find[-1] and L.MARKER in find[-1]


def test_auto_post_only_listed_prs_and_safe_statuses():
    cfg = dict(CFG, post={"mode": "auto", "auto_prs": [7]})
    rec = {"kind": "pr", "id": 7, "status": "PASS"}
    assert comment.may_auto_post(cfg, rec)
    assert not comment.may_auto_post(cfg, dict(rec, status="MISMATCH"))
    assert not comment.may_auto_post(cfg, dict(rec, id=8))
    assert not comment.may_auto_post(CFG, rec)  # manual by default


def test_post_dry_run_does_not_call_gh(work, fake_tool, monkeypatch, capsys):
    monkeypatch.setattr(L, "gh", lambda *a, **k: pytest.fail("gh called"))
    put_run(
        "pr",
        3,
        "a" * 40,
        "quick",
        status="NO_APPLICABLE_TESTS",
        finished="2026-11-17T00:00:00-08:00",
    )
    assert comment.post(CFG, 3, yes=False) is None
    out = capsys.readouterr().out
    assert L.MARKER in out and "dry run" in out


def test_post_yes_records_url(work, fake_tool, monkeypatch):
    calls = []
    monkeypatch.setattr(L, "gh", fake_gh_factory(calls, ""))
    d = put_run(
        "pr", 3, "a" * 40, "quick", status="PASS", finished="2026-11-17T00:00:00-08:00"
    )
    assert comment.post(CFG, 3, yes=True).endswith("issuecomment-123")
    assert json.loads((d / "posted.json").read_text())["url"].endswith(
        "issuecomment-123"
    )


def test_reply_sends_body_on_stdin(monkeypatch):
    calls = []
    monkeypatch.setattr(L, "gh", fake_gh_factory(calls, ""))
    comment.reply(CFG, 5, 99, "Done in the latest push.", yes=True)
    args, body = calls[-1]
    assert args[3] == "repos/llvm/llvm-project/pulls/5/comments/99/replies"
    assert "body=@-" in args and body == "Done in the latest push."


def test_driver_argv_tiers(work):
    (work / "corpus").mkdir()
    (work / "corpus" / "k.llir").write_text("")
    s = {"kind": "pr", "id": 9, "head": "a" * 40, "base": "b" * 40}
    deep = scan.driver_argv(s, CFG, "deep", work / "out")
    assert (
        deep[deep.index("--base") + 1] == "b" * 40
        and deep[deep.index("--head") + 1] == "a" * 40
    )
    assert deep[deep.index("--build-llc") + 1] == str(work / "builds" / "pr-9")
    assert "--all-tests" in deep and deep[deep.index("--corpus") + 1] == str(
        work / "corpus"
    )
    assert "--corpus-args=-O3 -relocation-model=pic" in deep
    assert deep[deep.index("--job-timeout") + 1] == "1800"
    quick = scan.driver_argv(s, CFG, "quick", work / "out")
    assert "--all-tests" not in quick and "--corpus" not in quick
    b = scan.driver_argv(
        dict(s, kind="branch", id="users/x/y"), CFG, "quick", work / "out"
    )
    assert b[b.index("--build-llc") + 1] == str(work / "builds" / "branches")


def test_run_subject_records_status_event_and_draft(work, fake_tool):
    put_run(
        "pr", 5, "c" * 40, "quick", status="PASS", finished="2026-11-16T00:00:00-08:00"
    )
    s = {"kind": "pr", "id": 5, "title": "t", "head": "d" * 40, "base": "e" * 40}
    rec = scan.run_subject(s, CFG, "quick")
    d = L.run_dir("pr", 5, "d" * 40, "quick")
    assert rec["status"] == "MISMATCH" and rec["exit"] == 1 and L.finished(d)
    assert "PASS -> MISMATCH" in (work / "results" / "events.log").read_text()
    text = (d / "comment.md").read_text()
    assert "v0 lane 1: old 0x1 new 0x2" in text and "`g`" not in text
    assert scan.run_subject(s, CFG, "quick") is None  # each head runs once per tier


def test_run_subject_timeout_kills_the_driver(work, fake_tool, monkeypatch):
    monkeypatch.setenv("FAKE_SLEEP", "1")
    cfg = dict(CFG, run_timeout_hours={"quick": 1 / 3600, "deep": 1})
    s = {
        "kind": "branch",
        "id": "users/x/b",
        "title": "b",
        "head": "f" * 40,
        "base": "e" * 40,
    }
    rec = scan.run_subject(s, cfg, "quick")
    assert rec["status"] == "TIMEOUT" and rec["exit"] is None


def test_index_lists_latest_run_per_subject_and_tier(work):
    put_run(
        "pr", 4, "a" * 40, "quick", status="PASS", finished="2026-11-16T00:00:00-08:00"
    )
    put_run(
        "pr",
        4,
        "b" * 40,
        "quick",
        status="MISMATCH",
        finished="2026-11-17T00:00:00-08:00",
    )
    put_run(
        "pr",
        4,
        "b" * 40,
        "deep",
        status="INCOMPLETE",
        finished="2026-11-17T05:00:00-08:00",
    )
    scan.write_index()
    md = (work / "results" / "index.md").read_text()
    assert "MISMATCH" in md and "INCOMPLETE" in md and "| PASS |" not in md
    assert len(json.loads((work / "results" / "index.json").read_text())) == 2


def test_prune_removes_builds_of_closed_prs(work, monkeypatch):
    for n in ("pr-1", "pr-2", "branches"):
        (work / "builds" / n).mkdir(parents=True)
    monkeypatch.setattr(L, "git", lambda *a: "")
    scan.prune([{"number": 2}])
    assert sorted(p.name for p in (work / "builds").iterdir()) == ["branches", "pr-2"]


import cli  # noqa: E402


def test_cli_dispatch(work, monkeypatch):
    seen = []
    monkeypatch.setattr(
        scan, "scan", lambda cfg, tier, pr: seen.append(("scan", tier, pr))
    )
    monkeypatch.setattr(
        comment, "post", lambda cfg, pr, yes: seen.append(("post", pr, yes))
    )
    monkeypatch.setattr(
        comment,
        "reply",
        lambda cfg, pr, cid, body, yes: seen.append(("reply", pr, cid, body, yes)),
    )
    monkeypatch.setattr(sys, "stdin", __import__("io").StringIO("body text"))
    cli.main(["scan", "--tier", "deep"])
    cli.main(["scan", "--pr", "239124"])
    cli.main(["post", "243123"])
    cli.main(["reply", "243123", "77", "--yes"])
    assert seen == [
        ("scan", "deep", None),
        ("scan", "quick", 239124),
        ("post", 243123, False),
        ("reply", 243123, 77, "body text", True),
    ]
