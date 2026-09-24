# Prepared PR descriptions — paste-and-delete

This branch exists only to hand two files to a human. It is **not** meant to be
reviewed or merged, and nothing here is built or tested. Delete the branch once
the descriptions have been pasted.

## Why the files are here rather than applied directly

Cursor's `ManagePullRequest` tool can only rewrite PR descriptions that it
authored itself. The descriptions on #11968 and #11970 were hand-written, so it
refuses with "the current description is not agent-managed", and the `gh` CLI
available to the agent is read-only. A human has to do the paste.

## What changed relative to what is live on the PRs today

Both descriptions pin their Reviewer Guide permalinks to commits the branches
have long since moved past, so a reviewer who follows the guide reads code the
PR no longer contains:

| PR | Live description pins to | What that link shows | What the head actually has |
|---|---|---|---|
| #11968 | `758072b`, `queue.cpp#L318-L322` | `!spm::is_any_active()` | `spm::is_active_on_agent(...)` at L446-L450 |
| #11970 | `70bef64`, `queue.cpp#L318-L328` | `!counters::is_any_active()` | `counters::is_active_on_agent(...)` at L448-L452 |

The replacements:

1. Repin every permalink to the current head — `47a0139` for #11968,
   `79a456a` for #11970 — with line ranges re-measured against the tree at those
   commits. Every link was checked for both HTTP 200 and a correct page title.
2. Add an **Agent-scoped interception gate** paragraph to the Technical Details
   of both. That change (`4bbc5a8` on #11968, `87c4313` on #11970) is the newest
   substantive work on either branch and was described nowhere.
3. #11970 only: a **Callback-thread refcount** section covering `79a456a`, and a
   ninth Reviewer Guide item pointing at it.
4. #11968 only: two stale symbol names corrected. The numbered list and the
   mermaid diagram referred to `spm::write_hook` and `signal_completion_hook`,
   neither of which exists — the head commit is literally
   "rename the SPM queue hooks to the agreed phase names".

The Motivation sections, the #8887 / #8891 stack banners, the Questions to
Answer, Issue Tracking and Submission Checklist blocks are carried over
unchanged.
