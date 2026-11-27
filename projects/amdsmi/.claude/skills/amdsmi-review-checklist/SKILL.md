---
name: amdsmi-review-checklist
description: Code review checklist for amdsmi (ROCm/rocm-systems projects/amdsmi) derived from a 6-month retro of real GitHub PR review comments and Jira ROCM defects. Load when reviewing an amdsmi diff/PR, or auditing a fix for an amdsmi defect, to check the change against recurring high-frequency mistake categories instead of re-deriving them from scratch.
---

# amdsmi Review Checklist

Derived 2026-11 from: a sample of 46 merged PRs (22 with substantive human
review) touching `projects/amdsmi` over the prior 6 months, plus a
title-weighted sample of ~150+ Jira `ROCM` "Defect" tickets (components AMD
SMI - BM/Guest/CPU/Host) over the same window. Themes marked **[confirmed by
both PR review + Jira defects]** are the strongest signal — they caused real
shipped bugs *and* reviewers already know to look for them. Re-run this retro
periodically; it's a sample, not a census.

## How to use
When reviewing an amdsmi diff (PR or local), walk the diff against each
section below. Most relevant to diffs that touch sysfs reads, status-code
returns, public headers, CLI output, or ASIC/backend gating — skip sections
that obviously don't apply (e.g. a docs-only or CI-only change).

## 1. ASIC/platform/backend gating correctness **[confirmed by both]**
New or variant hardware (new ASIC, APU, backend) needs its
`AMDSMI_STATUS_NOT_SUPPORTED`/early-return logic checked for **ordering**
against any new guard code being added, not just presence. A new assertion or
check added *before* an existing early-return will fire on hardware that
should have already been filtered out (ROCM-33815). Also check whether
existing grouping/iteration logic (e.g. hive_id-based grouping for
multi-GPU resets) silently assumes semantics that don't hold for a new
fabric/backend type (ROCM-34328 — UALink reused XGMI's hive-grouping key and
silently skipped 3 of 4 GPUs).

## 2. WSL/DXG gating — no CI safety net
There is **no WSL CI yet**. Any diff touching gating/early-return logic that
could affect the WSL/DXG backend must be scrutinized by hand — a regression
here will not be caught automatically (ROCM-33815, ROCM-34371; PR #12365 had
three separate platform false-positives: hypervisor, EPYC/container missing
`/dev/hsmp`, built-in (non-module) driver missing
`/sys/module/<m>/initstate`). Ask explicitly: "if this runs under WSL, does
the gate still trigger in the right place?"

## 3. Status/error-code selection and propagation **[confirmed by both]**
- Is the chosen `AMDSMI_STATUS_*` actually right for the failure, not just
  "some error code"? Seen picked wrong: NOT_FOUND vs NOT_SUPPORTED
  (ROCM-33270), "elevation required" vs NOT_SUPPORTED (ROCM-30638),
  UNEXPECTED_DATA for a merely-missing/empty sysfs file (ROCM-30061,
  ROCM-29739).
- Is `ret` actually set on **every** branch, including early-exit/drop paths?
  A path that falls through without setting it silently reports SUCCESS
  (PR #10638).
- When refactoring, don't drop a pre-existing specific status code in favor
  of a more generic one (PR #7617 — `AMDSMI_STATUS_DIRECTORY_NOT_FOUND`).

## 4. CLI exit code must reflect real outcome, not logging verbosity
`rocm_smi.py`'s exit-code logic has been gated on `--loglevel` rather than
actual outcome at least twice (ROCM-24108, then recurred for a *different*
clock type at ROCM-25289 — the first fix was path-specific, not systemic).
For any CLI change that can reject/fail an operation, trace every failure
branch to the actual process exit code independent of verbosity settings.
Don't assume a previous fix to this class of bug was comprehensive.

## 5. Public API/ABI stability **[confirmed by both]**
- Adding a field to a public struct (`amdsmi_fabric_info_t`,
  `amdsmi_driver_info_t`) without a version/soname bump breaks callers built
  against the old header — happened twice within three weeks, both reverted
  (ROCM-33570, ROCM-33173). Any change to a struct in the public header
  needs an explicit offset/size audit (PR #10446 did this well: full
  `offsetof`/`static_assert`-equivalent review, plus confirming Python/Rust
  bindings stayed in sync).
- New public API logic placed in the wrong file (e.g. directly in
  `amd_smi.cc` instead of `amd_smi_utils.cc`, PR #6043) — check against the
  file-organization conventions in the `amdsmi-context` skill.
- Don't overload an unrelated existing exception/status type for a new
  condition instead of adding the correct dedicated one (PR #12365/#12707).

## 6. Memory/bounds safety **[confirmed by both]**
- Unbounded `sscanf("%s", buf)` / `sprintf` / `strcpy` into a fixed-size
  stack buffer — this exact idiom has recurred copy-pasted across at least
  5 files (`fdinfo.cc`, `rocm_smi.cc`, `rocm_smi_binary_parser.cc`,
  `amd_smi.cc`, `amd_smi_utils.cc`; ROCM-29536 + independent SEC-* scan
  tickets). Treat this as a repo-wide smell, not a one-off.
- A function returning `AMDSMI_STATUS_SUCCESS` without writing its out-param
  on some code path hands the caller uninitialized memory (ROCM-30854).
- Invalid free / use-after-free from `std::ifstream`/filebuf mishandling
  (ROCM-25398); OOB index, unchecked `gmtime_r`, pointer-arithmetic UB, a
  reused all-zero sentinel value colliding with real data (PR #10638 — a
  single new API drew ~20 distinct findings like this from one thorough
  reviewer). Prefer RAII over manual malloc/free in new C++ code.

## 7. Telemetry correctness: swapped fields / unit mismatches
Check field-by-field when copying into an output struct, not just that
"a value gets written somewhere" — `subvendor_id`/`subsystem_id` assigned
crosswise, and `power_cap`/`power_limit` copied in watts into a struct whose
documented contract is microwatts (missing ×1e6), both in the same ticket
(ROCM-34323).

## 8. Concurrency/threading
Mutex locked by one thread and unlocked by a different thread after the
locking thread has already exited (ROCM-32128, flagged by TSAN, root cause
never fully confirmed). Hard to catch by inspection alone — flag any new
cross-thread mutex ownership pattern for extra scrutiny or ask for a
TSAN/ASAN run.

## 9. Test coverage gating before merge
- HW-dependent fixes (e.g. fan speed, OverDrive) should have actual
  hardware validation results in the PR, not just "compiles" (PR #9036 —
  two reviewers required this before approving).
- A genuinely new C++-level API needs real tests exercising it, not just a
  syntax-level check (PR #10638 was merged-pending exactly this).
- Insist on a real green CI run before trusting a correctness claim in the
  PR description (PR #10446).

## 10. Documentation currency
- Don't conflate behavior across ASIC generations in shared docs (XGMI
  topology docs mixed GPU↔GPU vs GPU↔CPU hop semantics across generations,
  PR #7372) — describe the generational difference in terms of the
  public-facing capability/topology distinction, not by codename (see #11).
- Distinguish "capability exists" from "AMD-validated" in partition/support
  docs (PR #13217).
- Remove scratch/working docs (e.g. stray `DESIGN-*.md`) before merge
  (PR #12365).
- Document surprising sentinel values (e.g. `current` frequency can be `-1`
  in sleep mode, PR #6043).

## 11. No internal product codenames in public-facing text
`rocm-systems` is a **public** repo. PR titles, commit messages, and
header/doc prose should not reference internal product/ASIC codenames (e.g.
"MI4XX") even when the underlying hardware-generation distinction is real
and worth documenting — phrase it in terms of the public capability/behavior
difference instead.
Also internal confluence page link should not be in PR or commit description.

## 12. Code hygiene (lower severity, but very recurring)
- `uint64_t` vs bare `int`/`size_t` comparisons risking `-Wsign-compare`
  (PR #12494 had 4 separate occurrences in one diff).
- Magic numbers without an explanatory constant/comment (PR #10638,
  PR #7914).
- Prefer reusing existing logic over duplicating it (PR #6043, #12494,
  #5724).
- A review comment that reads as AI-generated (e.g. prefixed "claude:",
  unusually formal offsetof/static_assert-style analysis) still needs a
  human sanity pass before being trusted as the final word (seen in PR
  #10446).

## Context worth keeping in mind
Roughly half of sampled amdsmi PRs (24/46) merged with **no substantive
human review** at all — mostly genuine docs/dependency/CI-only changes, but
it means this checklist mostly only helps if someone actually reads the
diff. Apply it with extra weight on anything touching sysfs reads,
ASIC/backend gating, public headers, or CLI exit paths — that's where misses
in this retro actually became shipped defects.
