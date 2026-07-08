# Implementation Workflow: `rocprofsys::common::path` filesystem module

Automated, phase-gated plan for implementing
[`design-filesystem-module.md`](design-filesystem-module.md). Companion to the
[usage survey](analysis-filesystem-usage.md) and the [API reference](reference-filesystem-apis.md).

> **HARD RULE — NEVER PUSH.** This workflow never runs `git push` (nor opens PRs) under
> any circumstance, including on a fully green run. Commits stay **local only**. Pushing
> is a manual step the user performs themselves, after their own review. No phase, gate,
> or auto-advance step may push.

**Execution model: auto-advance on green gate.** Each phase runs its verification
gate automatically; if the gate is green the next phase starts without waiting for
sign-off. Execution **halts and reports** only on (a) a red gate, or (b) entry into
the two designated manual-review phases (**P-dl**, **P-final**), which are too
high-risk to auto-advance past. **Committing is allowed** (one commit per phase, local
only); **pushing is never allowed** (see hard rule above).

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

**Red gate → halt.** Report the failing check, the diff since the phase started, and
whether the failure is a candidate "intended change" (§1) or a genuine regression.
Do **not** auto-advance.

### 2.1 Extra gates for specific phases

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
1. Read the phase's target files + the design's migration mapping (§8) for those spellings.
2. Apply the migration (old spelling → new API) for that subsystem only.
3. Run the gate (§2). Iterate until CHECK 1 & 2 are green.
4. Verify CHECK 3 (ref counter strictly decreased for call-site phases).
5. Commit the phase, **local only** (one commit per phase; P8/P-dl may be
   multi-commit). **Never `git push`** — see the hard rule at the top of this doc.
6. If phase ∈ {P-dl, P-final}: HALT, report, await sign-off.
   Else if gate green: advance to next phase in ID order.
   Else (red): HALT, report the regression vs. the §1 intended-change list.
```

**Migration mapping** to apply is design doc §8 (old spelling → new). The before/after
exemplars are design §9.

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
