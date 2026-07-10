# Implementation Workflow: `rocprofsys::common::path` filesystem module

Automated, phase-gated plan for implementing
[`design-filesystem-module.md`](design-filesystem-module.md). Companion to the
[usage survey](analysis-filesystem-usage.md) and the [API reference](reference-filesystem-apis.md).

> **HARD RULE — NEVER PUSH.** This workflow never runs `git push` (nor opens PRs) under
> any circumstance, including on a fully green run. Commits stay **local only**. Pushing
> is a manual step the user performs themselves, after their own review. No phase, gate,
> or auto-advance step may push.

**Execution model: auto-advance on green gate, self-heal on red.** Each phase runs
its verification gate automatically; if the gate is green the next phase starts without
waiting for sign-off. **If a gate breaks (red), the agent attempts to fix it itself**
(see §2.2) before halting. Execution **halts and reports** only on (a) a red gate that
survives the bounded fix attempts, or (b) entry into the two designated manual-review
phases (**P-dl**, **P-final**), which are too high-risk to auto-advance past.
**Committing is allowed** (one commit per phase, local only); **pushing is never
allowed** (see hard rule above).

**Decision logging.** Every non-trivial decision, fix attempt, and gate result is
appended to [`impl-log.md`](impl-log.md) as the run progresses (see §8). The log is the
audit trail for what the automation did and why.

---

## 1. Why this is safely automatable

The design is **~95% behavior-preserving refactor**. That means the existing test
suite is the *equivalence oracle*: any red test that is not one of the small set of
**intended** behavior changes is a real regression. The intended changes are the
complete, closed list:

1. `Exists_BrokenSymlink` flips `EXPECT_TRUE` → `EXPECT_FALSE` (single `exists()` is
   now `fs::exists` semantics — true-for-dirs, false-for-broken-link).
2. Three latent dir-bugs get fixed by the unified `exists()` / idempotent `make_dirs`:
   `argparse.cpp:136` (`_libdir`), `impl.cpp:398` (`_config_folder`),
   `instrument.cpp:296` (`omni_root`).
3. `is_text_file` → `is_elf` with **inverted call-site sense** at `instrument.hpp:169`
   and `:237` (`if(is_text_file) reject` → `if(!is_elf) reject`).

Everything else must be byte-for-byte behavior-preserving, so "tests stay green" is a
valid automated pass/fail signal.

---

## 2. The verification gate (runs between every phase)

Grounded in this repo's actual build/test layout (single aggregate gtest binary
`rocprof-sys-unit-tests`, built into `build/debug/bin/`, `BUILD_TESTING=ON`).

```bash
# --- CHECK 1: compiles ---
# API/test-only phases (P0): fast, targeted
cmake --build build/debug --target rocprof-sys-unit-tests -j
# call-site phases (P1..): full build catches every consumer
cmake --build build/debug -j

# --- CHECK 2: tests green ---
# Fast inner loop while iterating a phase:
./build/debug/bin/rocprof-sys-unit-tests --gtest_filter='PathTest.*:discovery_test.*'
# Full suite as the actual gate (catches cross-subsystem breakage):
cmake --build build/debug -j && ctest --test-dir build/debug --output-on-failure
#   (or run ./build/debug/bin/rocprof-sys-unit-tests with no filter)

# --- CHECK 3: monotonic ref counter ---
# Must be strictly LESS than the previous phase's value; NEVER higher.
grep -rn 'tim::filepath\|namespace filepath\s*=' source/ | wc -l
```

**Gate = green** iff: CHECK 1 exits 0, CHECK 2 shows 0 failures (modulo the intended
changes in §1), and CHECK 3 is `<=` the prior phase's count (`==` allowed for P0/P1
which are additive/move-only; `<` required for every call-site phase).

**Red gate → try to fix, then halt if unresolved** (see §2.1). Do **not** auto-advance
past a red gate.

### 2.1 Red gate — self-heal protocol

When a gate goes red, the agent attempts to fix it before halting, within these bounds:

1. **Classify the failure first** and log it to `impl-log.md`:
   - **Intended change** (§1 closed list) → update the test/expectation to match the
     design (e.g. flip `Exists_BrokenSymlink`), log it as such, re-run the gate. This is
     not a regression.
   - **Genuine regression** → proceed to fix attempts.
2. **Bounded fix attempts: up to 3 per phase.** Each attempt: form a hypothesis, apply
   the smallest change that addresses it, re-run the gate, log the attempt and outcome.
3. **Scope guard.** Fixes must stay within the current phase's semantics. The agent may
   **not**, to make a gate pass: change the design's intended behavior, weaken/delete an
   unrelated test, `#if 0`/comment out code, or silence a warning-as-error without a real
   fix. If the only way to green is one of these → **halt** instead.
4. **Escalate on exhaustion.** If 3 attempts do not restore green, or the fix would
   breach the scope guard → **halt**, log the final state (failing check, diff since
   phase start, hypotheses tried and why each failed), and await sign-off.
5. Build-infra breakage (missing include, CMake target wiring, link error) is in-scope
   to fix and does not count as a design deviation.

### 2.2 Extra gates for specific phases

- **P0**: the new pinned regression tests (design §7) must exist and pass —
  realpath verbatim-fallback, `exists` true-for-dirs + false-for-broken-link,
  `parent_path` full edge table (§5.1), `open` auto-mkdir + `./base` fallback,
  `is_elf` magic-byte semantics.
- **P-dl**: real `LD_PRELOAD` smoke run (below), not just unit tests.
- **P-final**: CHECK 3 must reach **exactly 0** under `source/`.

---

## 3. Phase DAG

Dependency-ordered; each phase sized to keep a reviewable diff (<~400 lines).

```
P0 (foundation) ──▶ P1 (split header) ──┬─▶ P2 core        ─┐
                                        ├─▶ P3 binary      ─┤
                                        ├─▶ P4 runtime     ─┤
                                        ├─▶ P5 avail       ─┼─▶ P-final (delete)
                                        ├─▶ P6 causal-bin  ─┤
                                        ├─▶ P7 python      ─┤
                                        ├─▶ P8 instrument  ─┤
                                        └─▶ P-dl (isolated)─┘
```

**Critical path:** `P0 → P1 → {P2..P8, P-dl} → P-final`. P0 is the keystone — nothing
else is safe until the new API + its pinned tests are green. P2..P8 and P-dl are
mutually independent (different files); with auto-advance they run **serially** in ID
order behind the shared build dir (the build/test gate cannot truly parallelize on one
build tree). P-final is blocked by *all* of them.

| Phase | Scope (files) | Build | Auto-advance? |
|-------|---------------|-------|---------------|
| **P0 Foundation** | New `source/lib/common/path.hpp` API (§5 of design) + new unit tests in `test_path.cpp`. **Additive** — old bodies stay, no call sites touched yet. | unit-tests target | yes on green |
| **P1 Split header** | Move `get_rocprofsys_root`/`get_internal_*` → new `install_layout.hpp`; `get_link_map`/`get_origin` → new `link_map.hpp`. Pure move + include fixups. | full | yes on green |
| **P2 core** | `config.cpp`, `argparse.cpp`, `common.hpp`, `rocpd/.../database.cpp`, `trace_cache/discovery.cpp`, `perfetto.cpp` | full | yes on green |
| **P3 binary** | `analysis.cpp`, `link_map.cpp`, `dwarf_entry.cpp`, `symbol.cpp` | full | yes on green |
| **P4 runtime** | `coverage.cpp`, `causal/data.cpp`, `causal/experiment.cpp`, `kokkosp.cpp`, `library.cpp` | full | yes on green |
| **P5 avail** | `rocprof-sys-avail/common.cpp` (`file_exists`), `generate_config.cpp` | full | yes on green |
| **P6 causal-bin** | `rocprof-sys-causal/impl.cpp` (incl. latent-bug fix) | full | yes on green |
| **P7 python** | `python/libpyrocprofsys.cpp` | full | yes on green |
| **P8 instrument** | `details.cpp`, `internal_libs.cpp`, `module_function.cpp`, `rocprof-sys-instrument.{cpp,hpp}`, `info.hpp`. **Largest — split into sub-PRs** (helpers first: `is_file`/`is_directory`/`exists`/`canonicalize`, then resolvers). Also `bin/common/tool_runner.cpp`, `preset_registry.cpp`. | full | yes on green |
| **P-dl Preload** | `rocprof-sys-dl/dl.cpp` — **isolated, MANUAL review** | full + smoke | **NO — halt & report** |
| **P-final Delete** | Remove `tim::filepath` includes/aliases (incl. `common.hpp:88`) + old `common::path` syscall bodies; delete duplicate `is_text_file`. **MANUAL review** | full + grep==0 | **NO — halt & report** |

Subsystem list is taken verbatim from the analysis doc's "Category → file index".

---

## 4. P-dl smoke verification (design §12.1)

The `LD_PRELOAD`ed `dl` library is the highest-risk consumer (injected into arbitrary
targets). Auto-advance is disabled here. The gate additionally requires:

1. `dl.cpp` migrated in its **own** commit (independently revertable/bisectable).
2. **Real end-to-end instrument run** of an actual target binary, exercising the live
   `dl` path sites: the `indirect` ctor (`find_path`/`dirname`/`basename`) and
   `readlink("/proc/self/exe")`. Unit tests are not sufficient.
3. Across **both** build configs the design reasoned about: default (dynamic
   `libstdc++.so.6`) **and** `ROCPROFSYS_BUILD_STATIC_LIBSTDCXX=ON`.
4. **Forced-fallback run** — point at a missing lib / non-existent path so the
   `catch(...)`→fallback and the "no `filesystem_error` crosses the `LD_PRELOAD`
   boundary" guarantee are actually exercised in the injected context.
5. Validate **before** P-final deletes the old `tim::filepath` bodies, so old-vs-new is
   comparable and revertable.

---

## 5. Per-phase loop (what the agent does each phase)

```
1. Append a "Phase Pn — START" entry to impl-log.md (goal, target files, baseline ref count).
2. Read the phase's target files + the design's migration mapping (§8) for those spellings.
3. Apply the migration (old spelling → new API) for that subsystem only.
4. Run the gate (§2). If red → self-heal protocol (§2.1), logging each fix attempt.
   Iterate until CHECK 1 & 2 are green or the self-heal bound is hit.
5. Verify CHECK 3 (ref counter strictly decreased for call-site phases).
6. Commit the phase, **local only** (one commit per phase; P8/P-dl may be
   multi-commit). **Never `git push`** — see the hard rule at the top of this doc.
7. Append a "Phase Pn — DONE" entry to impl-log.md (result, new ref count, commit SHA,
   any intended-change decisions, any deviations).
8. If phase ∈ {P-dl, P-final}: HALT, report, await sign-off.
   Else if gate green: advance to next phase in ID order.
   Else (red, self-heal exhausted): HALT, report per §2.1.
```

**Migration mapping** to apply is design doc §8 (old spelling → new). The before/after
exemplars are design §9.

---

## 5a. Decision log (`impl-log.md`)

The agent maintains [`impl-log.md`](impl-log.md) in this directory as an append-only
run journal. **Append, never rewrite** prior entries — it is the audit trail.

**What gets logged** (as it happens, not in a final batch):
- Phase START / DONE markers (goal, files, baseline + resulting `tim::filepath` ref count,
  commit SHA).
- Every **decision**: an intended-change classification (§1), a design-interpretation
  choice, a deviation from the plan and its justification, a deferred item.
- Every **self-heal fix attempt** (§2.1): the red symptom, the hypothesis, the change
  made, and the outcome (fixed / still red).
- Every **halt**: why, the final failing state, and what sign-off is needed to resume.

**Suggested entry format:**
```markdown
## Phase P2 — core  (2026-07-08)
- START: baseline ref count = 137; targets: config.cpp, argparse.cpp, ...
- DECISION: argparse.cpp:136 `_libdir` — intended dir-bug fix (§1.2); exists() now
  true-for-dirs so guard works. Not a regression.
- FIX ATTEMPT 1: link error `undefined ref path::make_dirs` → missing inline in header;
  added `inline`. Re-ran gate → green.
- DONE: gate green; ref count 137 → 121; commit abc1234.
```

Keep entries terse and factual. This file is planning scratch (gitignored region /
not shipped); it exists so a human can reconstruct what the automation did and why.

---

## 6. Rollback

Each phase is one commit (or a small commit series). A red gate that can't be resolved
→ `git revert` that phase's commit(s); the DAG above shows what it blocks (only
P-final depends on all; P2..P8/P-dl are independent, so reverting one does not block
the others).

---

## 7. Open coordination note

Design §5.1 (issue #7) assumes the **post-`tim-rem-string-manipulation`** state: no
`common::join` / `TIMEMORY_JOIN`, path-joining standardized on `fmt::format("{}/{}")`.
That sibling branch overlaps this one in `path.hpp` / instrument / config. **Sequence
them together** or rebase this work on top of it before P2 begins, else P2/P8 will
reintroduce join churn this design deliberately avoids.
```
