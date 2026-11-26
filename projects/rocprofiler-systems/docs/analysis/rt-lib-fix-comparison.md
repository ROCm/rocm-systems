# Technical Comparison: Fix Options — `libdyninstAPI_RT.so` Lookup Failure

## 1. Root cause

`rocprof-sys-instrument` runtime instrumentation mode injects Dyninst's runtime library into
the target process via `LD_PRELOAD`. Stock Dyninst
(`external/dyninst/dyninstAPI/src/unix.C`, `PCProcess::getDyninstRTLibName()`) resolves this
library by a **hardcoded, literal, unversioned basename**:

```cpp
std::string rt_base = "libdyninstAPI_RT";
rt_base += ".so";
if(!BinaryEdit::getResolvedLibraryPath(rt_base, rt_paths) || rt_paths.empty())
{
    startup_printf("Could not find libdyninstAPI_RT.so in search path\n");
    return false;
}
```

rocprofiler-systems builds the real shared object under its own branded name,
`librocprof-sys-rt.so.<version>` (`source/lib/rocprof-sys-rt/CMakeLists.txt`, `OUTPUT_NAME
${BINARY_NAME_PREFIX}-rt`, `VERSION`/`SOVERSION` set), and creates `libdyninstAPI_RT.so` as a
**relative symlink** to it at install time:

```cmake
add_custom_target(... ALL ${CMAKE_COMMAND} -E create_symlink
    ../$<TARGET_FILE_NAME:rocprofiler-systems-rt-library>
    ${CMAKE_SHARED_LIBRARY_PREFIX}dyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX} ...)
```

TheRock's wheel-packaging step (`build_tools/build_python_packages.py`,
`populate_runtime_files()`) does not preserve differently-named symlinks when building the
wheel — this is **documented, intentional policy**
(`docs/packaging/python_packaging.md`: *"[runtime packages] do not contain symlinks... only
SONAME libraries are included"*). Confirmed empirically in this investigation
(`/tmp/cmake-symlink-test`) that CMake's own `install(FILES)` *does* preserve a symlink's
relative-path text verbatim — the loss happens specifically in TheRock's wheel-build step, not
in CMake's install step. Result: the alias never reaches the pip wheel, and Dyninst's
hardcoded search fails — `FatalError 68, create process failed bootstrap`.

## 2. Fix A — consumer-side lookup (`rocm-systems#16489` + `dyninst#34`)

**Mechanism, two coordinated changes:**

1. `rocprof-sys-instrument.cpp::find_dyn_api_rt()` builds an ordered candidate list —
   `DYNINSTAPI_RT_LIB` env override (validated: must point to a readable `.so`/`.so.*`), then
   `libdyninstAPI_RT.so`, then (for non-rewrite mode) `librocprof-sys-rt.so`, then the static
   `.a` archive — and resolves the first hit via `resolve_library_path()`, exporting it as
   `DYNINSTAPI_RT_LIB` for the child process.
2. Dyninst's own `unix.C::getDyninstRTLibName()` is patched (`dyninst#34`) with a new
   `usableDyninstRTLibFromEnv()` gate that validates and resolves `DYNINSTAPI_RT_LIB` to an
   absolute path *before* falling back to the hardcoded search — previously this env var was
   read by the binary-rewrite path only, never by process-creation/runtime-instrumentation.

**Why A requires both halves:** rocprofiler-systems setting `DYNINSTAPI_RT_LIB` is a no-op for
runtime instrumentation unless Dyninst's process-creation path actually reads it — which it
doesn't, in any released or currently-merged Dyninst. Hence the hard dependency
`#16489 → dyninst#34`.

**Dependency graph:** `dyninst#34` (ROCm/dyninst, `dyninst_13` branch) → `rocm-systems#16489`
(bumps `external/dyninst` submodule pin to a commit containing #34's fix) → eventually a
`TheRock` submodule bump (pattern seen in prior, now-superseded PRs `#10221`/`#10215`) to reach
the actual wheel. Three sequential review gates across two-to-three repos.

**Verification status:** not authored in this investigation; reviewed by reading the diffs and
PR metadata only. No build/test performed on this path.

## 3. Fix B — real-file duplication (`users/anujshuk-amd/rt-lib-real-copy`)

**Mechanism:** single-operator change in `source/lib/rocprof-sys-rt/CMakeLists.txt`:

```diff
- ${CMAKE_COMMAND} -E create_symlink
- ../$<TARGET_FILE_NAME:rocprofiler-systems-rt-library>
+ ${CMAKE_COMMAND} -E copy
+ $<TARGET_FILE:rocprofiler-systems-rt-library>
  ${CMAKE_SHARED_LIBRARY_PREFIX}dyninstAPI_RT${CMAKE_SHARED_LIBRARY_SUFFIX}
```

`libdyninstAPI_RT.so` is now a second, independent, ordinary ELF regular file at build/install
time, before TheRock's packaging ever runs. There is no symlink in this pair for
`populate_runtime_files()` to encounter.

**Build-verified in this investigation**, whole-project build, `build/debug`:

```
lib/rocprofiler-systems/libdyninstAPI_RT.so : ELF 64-bit LSB shared object, x86-64,
  BuildID[sha1]=10aad8b5975fdc183059b7bd99d62004addee265, with debug_info, not stripped
lib/librocprof-sys-rt.so.1.11.0            : ELF 64-bit LSB shared object, x86-64,
  BuildID[sha1]=10aad8b5975fdc183059b7bd99d62004addee265, with debug_info, not stripped
```

Identical BuildID (`cmp` confirmed byte-for-byte identical content), confirming the copy
executed correctly against the real linked artifact, not a stale/mismatched source. The
original SONAME chain (`librocprof-sys-rt.so → .so.1 → .so.1.11.0`) is untouched — this is
purely additive.

**What Fix B does *not* change:** ELF SONAME tags, RPATH (`$ORIGIN` on both targets,
unaffected), the project's branded-name-is-primary convention used in `dl.cpp:1248/1336` and
`internal_libs.cpp:393`, and the existing install directory layout
(`DESTINATION ${CMAKE_INSTALL_LIBDIR}` for the real lib; the alias installs into
`lib/${PROJECT_NAME}` exactly as before — only its content-generation mechanism changed).

## 4. Fix C — rename the real artifact to Dyninst's expected name (`users/anujshuk-amd/rt-lib-rename-dyninst`)

**Mechanism:** instead of keeping `librocprof-sys-rt.so.<version>` as the real file and aliasing
`libdyninstAPI_RT.so` to it (today's layout, or Fix B's duplicate-of-it layout), build the
*one* real file directly under the literal name Dyninst's hardcoded search expects, with no
version suffix at all — matching `rt_base = "libdyninstAPI_RT"; rt_base += ".so";` exactly:

```cmake
set_target_properties(rocprofiler-systems-rt-library
    PROPERTIES
        OUTPUT_NAME dyninstAPI_RT          # was: ${BINARY_NAME_PREFIX}-rt
        # VERSION/SOVERSION removed: Dyninst never resolves a SONAME chain for this lib
        LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME}
        BUILD_RPATH "$ORIGIN"
        INSTALL_RPATH "$ORIGIN"
)
install(TARGETS rocprofiler-systems-rt-library
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/${PROJECT_NAME})   # was: bare ${CMAKE_INSTALL_LIBDIR}
```

`librocprof-sys-rt.so` becomes the symlink (reversed from today), created as a bare filename in
the same directory as the real file.

**A real defect found and corrected during this investigation:** the first version of this fix
installed the real file to bare `${CMAKE_INSTALL_LIBDIR}` (`lib/`). TheRock's wheel-packaging
artifact patterns only capture `lib/librocprof-sys*` and `lib/rocprofiler-systems/**` — neither
matches `lib/libdyninstAPI_RT*`, so the real file would have been **silently excluded from the
wheel entirely**, a worse outcome than the bug being fixed. Corrected by installing into
`lib/${PROJECT_NAME}` (already covered by the second glob pattern) and adding
`LIBRARY_OUTPUT_DIRECTORY` so the build tree mirrors the same layout — required because the
pytest/ctest suite runs directly against `ROCPROFSYS_BUILD_DIR` (the build tree), not an
installed prefix, so the build-tree symlink must resolve correctly too, independent of install
behavior.

**Verification status:** configure-tested only (reaches past this file with no new errors,
confirmed by reconfiguring with the change present vs. absent). **Not build-tested** — unlike
Fix B, the corrected version has not been compiled and inspected on disk.

**What changes that Fix A/B do not touch:** the real file's canonical name inverts (Dyninst's
name becomes primary, the project's branded name becomes the symlink); `VERSION`/`SOVERSION`
(and therefore the ELF SONAME tag) are dropped for this one library; the real file's location
moves from `lib/` to `lib/${PROJECT_NAME}/`.

## 5. Fix D — hard link instead of symlink or copy (not yet implemented)

**Mechanism:** same target as Fix B, but `${CMAKE_COMMAND} -E create_hardlink` in place of
`-E copy`. A hard link is a second directory entry for the *same inode* — `lstat()`/`st_mode`
shows an ordinary regular file, identical to Fix B from any tool's point of view, clearing the
same "no symlinks in wheels" policy concern.

**Technical distinction from Fix B:** for every **non-pip install** (RPM/DEB/tarball/source),
a hard link costs **zero additional disk** — one inode, two directory entries, versus Fix B's
permanent full duplicate in every install. For the **wheel/zip case specifically**, the
distinction disappears: zip has no hardlink concept, so whatever tool walks the install tree to
build the wheel will materialize two independent copies of the bytes regardless of whether the
source tree used a hardlink or a plain copy. Net effect: Fix D strictly dominates Fix B for
every install path except the one this investigation is about, where the two are equivalent.

**Risks not present in Fix B:** requires both names to resolve to the same filesystem (true
here, same install prefix — not a practical concern); if `rocprofiler_systems_strip_target`
strips one name's inode in place *after* the hardlink is created, it would also mutate the
other name, since they share storage — must confirm stripping happens before link creation
(today's ordering already strips before the symlink/copy step, so this should carry over
correctly, but has not been re-verified for the hardlink variant specifically); not
build-tested in this investigation.

## 6. Fix E — lazy self-healing in the Python entry-point wrapper (not yet implemented)

**Mechanism:** a different implementation layer entirely — not CMake, not C++, not Dyninst.
The pip-installed `rocprofsys` package already ships a Python wrapper
(`rocprofsys/__init__.py` / `__main__.py`) as the console-script entry point for
`rocprof-sys-instrument`. Before exec'ing the real binary, the wrapper would resolve its own
installed package location (`importlib.resources` or `__file__` — reliable, independent of
Dyninst's internal file search), check whether `libdyninstAPI_RT.so` exists next to
`librocprof-sys-rt.so`, and create it (copy or hard link) on first run if missing, then exec.

**Distinguishing properties:**
- Touches no CMake, no C++, no Dyninst, no TheRock packaging script — pure Python, in code
  rocprofiler-systems already owns and ships.
- Cost is paid lazily, once, only by installs that actually hit the bug — not by every build or
  package regardless of whether the bug would ever manifest for that install.
- Bypasses TheRock's packaging policy question entirely, since nothing about packaging changes;
  the fix executes after install, at first invocation.

**Risks:**
- Requires the venv/site-packages directory to be writable at first run. Fails — with a worse,
  less diagnosable error than today's — in a read-only or locked-down install (containers
  running as non-root with a read-only filesystem, some HPC module systems).
- A profiling tool mutating its own install tree as a side effect of being invoked is a
  surprising behavior some reviewers may object to on principle, independent of correctness.
- Only fixes the one binary that is actually routed through this Python wrapper; any other
  entry point or direct binary invocation that bypasses the wrapper remains exposed to the
  original bug.
- Not implemented or tested in this investigation — concept-level only.

## 7. Direct technical comparison

| Dimension | Fix A | Fix B | Fix C | Fix D | Fix E |
|---|---|---|---|---|---|
| **Failure domain addressed** | Name-resolution logic in the consumer | File-existence/packaging-transform domain | Canonical-naming domain (meet Dyninst's expectation directly) | Same as B, cheaper for non-pip installs | Runtime self-repair, outside build/packaging entirely |
| **Covers `DYNINSTAPI_RT_LIB` set to an arbitrary custom path** | Yes | No | No | No | No |
| **Covers a hypothetical future rename/alias of another internal library** | Yes, cheaply | No — new target per case | No — new rename+relocate per case | No — new target per case | Possibly, if the wrapper's logic is written generically |
| **Dependency surface** | 2 repos, sequential merge order | 1 repo, self-contained | 1 repo, self-contained | 1 repo, self-contained | 1 repo, self-contained |
| **Disk/package cost** | None | Permanent duplicate, every install | None | None for non-pip installs; same as B for the wheel/zip case | None for installs that never hit the bug; one duplicate for those that do, paid once |
| **Changes an existing target's identity (name/version/output dir)** | No | No | Yes | No | No |
| **Verified in this investigation** | Yes — built, and functionally re-tested with the alias removed to simulate the pip-wheel case; no `FatalError 68` | Yes — built, byte-identical content confirmed | Partially — configure-tested, not built | No — not implemented | No — not implemented |
| **Known defect already found and corrected** | N/A | None found | Yes — original version silently dropped from the wheel glob entirely, corrected | N/A | N/A |

## 8. Test results — Fix A

### 8.1 Existing CTest suite (regression check)

Branch `users/anujshuk-amd/dyninst-rt-lookup` checked out; `external/dyninst` submodule
confirmed pinned to `66c7d320d` (contains dyninst#34's fix). Whole-project build run, then
`ROCPROFSYS_INSTALL_TESTING=ON` reconfigured to generate the full pytest-driven CTest suite —
**755 tests** generated (none of which previously exercised this specific lookup path).

Ran the 12 lightweight CLI tests that directly exercise `rocprof-sys-instrument`:

```
ctest -R "rocprofiler-systems-instrument-" --output-on-failure -j4
```

| Test | Result |
|---|---|
| rocprofiler-systems-pytest-config | Passed (2.75s) |
| rocprofiler-systems-instrument-help | Passed (0.56s) |
| rocprofiler-systems-instrument-simulate-ls | Passed (19.29s) |
| rocprofiler-systems-instrument-simulate-lib | Passed (4.29s) |
| rocprofiler-systems-instrument-simulate-lib-basename | Passed (4.04s) |
| rocprofiler-systems-instrument-write-log | Passed (12.90s) |
| rocprofiler-systems-instrument-exe-only | Passed (10.56s) |
| rocprofiler-systems-instrument-max-library-functions | Passed (10.23s) |
| rocprofiler-systems-instrument-invalid-input-non-elf | Passed (0.50s) |
| rocprofiler-systems-instrument-invalid-input-missing-target | Passed (0.59s) |
| rocprofiler-systems-instrument-invalid-input-reinstrument | Passed (5.70s) |
| rocprofiler-systems-test-cleanup | Passed (0.37s) |

**12/12 passed. No regression introduced by Fix A in existing instrument-CLI coverage.**

### 8.2 New permutation tests (written to cover this issue specifically)

No existing test in the 755-test suite simulates a missing `libdyninstAPI_RT.so` alias — the
exact condition a pip wheel produces. Wrote a standalone pytest file,
`test_rt_lib_lookup.py` (not wired into CMake/CTest; run directly against the built
`rocprof-sys-instrument`), covering the permutation space of file presence and
`DYNINSTAPI_RT_LIB` state:

| Test | Permutation | Expected outcome | Result (Fix A) |
|---|---|---|---|
| `test_alias_present_no_env_var` | Alias present, no override (today's non-pip baseline) | Succeeds, no `FatalError` | **Passed** |
| `test_alias_missing_no_env_var` | Alias absent, no override (**the exact reported pip-wheel bug**) | Falls back to `librocprof-sys-rt.so`, succeeds, no `FatalError :: 68` | **Passed** |
| `test_alias_missing_valid_env_override` | Alias absent, `DYNINSTAPI_RT_LIB` set to the real file's absolute path | Override honored, no rejection warning, succeeds | **Passed** |
| `test_alias_missing_invalid_env_override` | Alias absent, `DYNINSTAPI_RT_LIB` set to a nonexistent path | Invalid override rejected with a warning, still falls back and succeeds | **Passed** |
| `test_both_missing_fails_cleanly` | Both the alias and the real file absent | Fails cleanly (non-zero exit, clear message) — must not hang, segfault, or succeed silently | **Passed** |

**5/5 passed against Fix A.** Verified directly against real process output in each case
(e.g. for the pip-wheel-equivalent case: `Resolved 'librocprof-sys-rt.so' to
'.../lib/librocprof-sys-rt.so'`, `DYNINST_API_RT: .../lib/librocprof-sys-rt.so`, exit code 0,
no `FatalError`). One correction made during test development: the diagnostic lines quoted
above are emitted on **stdout**, not stderr — the first test run falsely failed 3/5 cases by
checking the wrong stream; fixed by checking both streams, after which the underlying raw
process output confirmed all 5 behaviors were already correct even before the fix.

## 9. Residual uncertainty, all fixes

- **Verification performed, by fix:**
  - **Fix A** — independently functionally verified in this investigation. Checked out
    `users/anujshuk-amd/dyninst-rt-lookup` (confirmed `external/dyninst` submodule already
    pinned to `66c7d320d`, containing dyninst#34's fix), built `rocprof-sys-instrument`, then
    removed `libdyninstAPI_RT.so` from the build tree to simulate exactly what a pip wheel
    ships (only `librocprof-sys-rt.so` present). Running `rocprof-sys-instrument --verbose=1
    -- /bin/true` with no `DYNINSTAPI_RT_LIB` set showed the fallback resolve
    `librocprof-sys-rt.so`, export it as `DYNINST_API_RT`, and proceed through process
    creation, library parsing, instrumentation-function discovery, and snippet insertion with
    no `FatalError :: 68` — exit code 0. (The `rocm-systems#16489` PR description additionally
    states the author's own testing covered "functionality with and without aliases, with
    invalid variable settings, and in archive situations" — broader than what was reproduced
    here.)
  - **Fix B** — build-tree verified in this investigation. Whole-project build completed;
    `lib/rocprofiler-systems/libdyninstAPI_RT.so` confirmed to be a real regular file (not a
    symlink, via `-L` test), byte-identical to `lib/librocprof-sys-rt.so.1.11.0` (same
    BuildID, `cmp` match). This is the only option independently confirmed against an actual
    compiled output in this investigation.
  - **Fix C** — configure-verified only. The CMake file was confirmed to parse and the
    project configure step reaches past it with no new errors (checked both with the change
    present and with it reverted, isolating it from an unrelated pre-existing configure
    failure elsewhere). Not compiled; the resulting files were never produced or inspected.
  - **Fix D** — not implemented, not verified. Concept and expected CMake diff described only.
  - **Fix E** — not implemented, not verified. Concept described only; no code written.
- **No fix** — including Fix B, the most-verified option — has been exercised through an actual
  `cmake --install` into a staging prefix followed by a simulated TheRock wheel-build pass.
  A true end-to-end repro (build → install → package → `pip install` → run
  `rocprof-sys-instrument`) has not been performed for any of the five.
- None of the fixes were checked against every ROCm repo for a hypothetical out-of-tree
  consumer that might depend on current naming/versioning specifics of
  `librocprof-sys-rt.so`; absence of evidence here is not proof of absence.

## 10. Summary

- **Fix A** is correct and general — it fixes the underlying lookup logic and would cover
  future cases of this same class of bug — and is now functionally verified in this
  investigation (built, tested with the alias removed, no `FatalError 68`). Its remaining
  obstacle is not correctness but delivery: it depends on two upstream reviews in two separate
  repositories, in a fixed dependency order, neither complete at the time of writing.
- **Fix B** is correct for the reported bug specifically, is fully self-contained to this
  repository, is already pushed (`users/anujshuk-amd/rt-lib-real-copy`), and is the only option
  with build-verified evidence behind it in this investigation. Its cost is a small, permanent
  duplicate file and a non-technical risk that a reviewer may object to duplication on
  principle, independent of any functional flaw.
- **Fix C** removes Fix B's duplication cost entirely but is a more invasive diff — it inverts
  which name is canonical for this library and drops its SONAME versioning. It already required
  one correction during this investigation (the wheel-glob miss) and remains build-unverified.
- **Fix D** strictly dominates Fix B for every install path except the wheel/zip case, where it
  is equivalent; it is unimplemented and untested.
- **Fix E** is architecturally distinct — a runtime fix rather than a build/packaging fix — and
  trades a one-time, bug-triggered disk cost for a dependency on write access to the install
  tree at first run; it is concept-level only, unimplemented and untested.
