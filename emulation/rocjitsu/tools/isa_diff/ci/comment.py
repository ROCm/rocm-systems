# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""The PR comment of a run: draft it, post it (one comment per PR, edited in place) and
reply to review threads. Nothing is published without --yes or an auto_prs entry."""

import json
from pathlib import Path

import ci_lib as L

AUTO_STATUSES = {"PASS", "INCOMPLETE"}
LIMIT = 60000  # GitHub rejects comments over 65536 characters
STATUS_TEXT = {
    "PASS": "every changed function was compared and passed",
    "MISMATCH": "a changed function computes a different result",
    "INCOMPLETE": "no mismatch was found, but not everything was covered",
    "ERROR": "the tool failed",
    "NO_APPLICABLE_TESTS": "no changed function this tool can run",
    "TIMEOUT": "the run exceeded its time limit",
}


def cell(text, n=160):
    return str(text).replace("\n", " ")[:n].replace("|", "\\|")


def detail(row):
    """The first unresolved difference, else the first unresolved item or note."""
    for g in row.get("regions") or []:
        if g.get("status") == "LOCAL_MISMATCH" and not g.get("resolved_by"):
            if g.get("first_mismatch"):
                return g["first_mismatch"]
    for x in list(row.get("unresolved") or []) + list(row.get("notes") or []):
        return x
    return ""


def render(rec, summary):
    drv = L.driver()
    rows = (summary or {}).get("functions", [])
    skipped = (summary or {}).get("skipped", [])
    counts = {}
    for r in rows:
        counts[drv.verdict(r)] = counts.get(drv.verdict(r), 0) + 1
    head = [
        L.MARKER,
        f"**rj_isa_diff** (RocJITsu ISA differential testing, run locally) on "
        f"`{rec['head'][:12]}` against merge base `{rec['base'][:12]}` "
        f"({rec['tier']} scan, tool `{rec['tool'][:12]}`)",
        "",
        f"**{rec['status']}**: {STATUS_TEXT.get(rec['status'], rec['status'])}.",
    ]
    if rows:
        head.append(
            f"Changed functions compared: {len(rows)} ("
            + ", ".join(f"{n} {v}" for v, n in sorted(counts.items()))
            + ")."
        )
    if skipped:
        head.append(
            f"Skipped: {len(skipped)} file{'s' if len(skipped) != 1 else ''} "
            "(see the full report)."
        )
    tail = [
        "",
        "A pass means that no counterexample was found among emulated boundary and "
        "random inputs; it is not a proof of equivalence.",
    ]
    body = []
    bad = [r for r in rows if drv.verdict(r) != "pass"]
    if bad:
        body += [
            "",
            f"<details><summary>Functions that did not pass ({len(bad)})</summary>",
            "",
            "| verdict | test | target | function | detail |",
            "|---|---|---|---|---|",
        ]
        size = sum(len(x) + 1 for x in head + tail + body) + 20
        for i, r in enumerate(bad):
            v = drv.verdict(r) + (
                " (poison-sensitive)" if drv.poison_sensitive(r) else ""
            )
            line = (
                f"| {v} | `{cell(Path(r['file']).name, 80)}` | {r.get('target', '')} "
                f"| `{cell(r['function'], 100)}` | {cell(detail(r))} |"
            )
            if size + len(line) > LIMIT - 200:
                body.append(f"| … | {len(bad) - i} more in the full report | | | |")
                break
            body.append(line)
            size += len(line) + 1
        body += ["", "</details>"]
    return "\n".join(head + body + tail) + "\n"


def write_draft(d):
    rec = L.read_run(d)
    p = Path(d) / "out" / "summary.json"
    text = render(rec, json.loads(p.read_text()) if p.exists() else None)
    (Path(d) / "comment.md").write_text(text)
    return text


def may_auto_post(cfg, rec):
    p = cfg["post"]
    return (
        rec["kind"] == "pr"
        and p.get("mode") == "auto"
        and rec["id"] in p.get("auto_prs", [])
        and rec["status"] in AUTO_STATUSES
    )


def find_comment(repo, pr, login):
    """The id of our earlier comment on the PR (it carries MARKER), or None."""
    out = L.gh(
        [
            "api",
            "--paginate",
            f"repos/{repo}/issues/{pr}/comments",
            "--jq",
            f'.[] | select(.user.login == "{login}" and (.body | contains("{L.MARKER}")))'
            " | .id",
        ]
    )
    ids = out.split()
    return ids[0] if ids else None


def upsert(repo, pr, body_file):
    """Post the comment, or edit the earlier one; return its URL."""
    login = L.gh(["api", "user", "--jq", ".login"]).strip()
    cid = find_comment(repo, pr, login)
    if cid:
        args = ["api", "-X", "PATCH", f"repos/{repo}/issues/comments/{cid}"]
    else:
        args = ["api", "-X", "POST", f"repos/{repo}/issues/{pr}/comments"]
    return L.gh(args + ["-F", f"body=@{body_file}", "--jq", ".html_url"]).strip()


def post_run(cfg, rec):
    d = Path(rec["dir"])
    url = upsert(cfg["repo"], rec["id"], d / "comment.md")
    (d / "posted.json").write_text(json.dumps({"url": url, "posted": L.now()}) + "\n")
    return url


def post(cfg, pr, yes):
    """Print the latest run's comment; with @p yes, post or edit it on the PR."""
    rec = L.latest_run("pr", pr)
    if rec is None:
        raise SystemExit(f"no finished run for PR {pr}")
    print(write_draft(rec["dir"]))
    if not yes:
        print(f"(dry run: add --yes to post this to {cfg['repo']}#{pr})")
        return None
    url = post_run(cfg, rec)
    print(url)
    return url


def reply(cfg, pr, comment_id, body, yes):
    """Reply in the thread of review comment @p comment_id."""
    print(body)
    if not yes:
        print(
            f"(dry run: add --yes to reply to review comment {comment_id} on "
            f"{cfg['repo']}#{pr})"
        )
        return None
    url = L.gh(
        [
            "api",
            "-X",
            "POST",
            f"repos/{cfg['repo']}/pulls/{pr}/comments/{comment_id}/replies",
            "-F",
            "body=@-",
            "--jq",
            ".html_url",
        ],
        input=body,
    ).strip()
    print(url)
    return url
