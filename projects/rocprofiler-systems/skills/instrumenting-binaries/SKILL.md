---
name: instrumenting-binaries
description: Guides instrumentation with `rocprof-sys-instrument` - choosing runtime instrumentation vs binary rewrite (`-o`), including or excluding functions and modules with regex options, previewing what will be instrumented with a dry run (`--simulate`, `--dump-info`, `--print-*`), tuning granularity (`--min-instructions`, `--min-address-range`, `--instrument-loops`), rewriting libraries, and diagnosing why a function is missing from, or too numerous in, a trace. Use when the user asks about rocprof-sys-instrument, binary rewrite, instrumenting a shared library, function or module include/exclude regexes, dry runs, or the difference between a function and a module. Do NOT use to pick a `--preset` (use `choosing-a-profiling-preset`), to list settings, counters or ROCm domains (use `discovering-profiling-options`), or to attach to a running process (use `rocprof-sys-attach`).
---

# Instrumenting Binaries

`rocprof-sys-instrument` decides which functions of a program get instrumented and
either does it while the program runs or writes a new instrumented binary. For the
full, current flag list run `rocprof-sys-instrument --help` (use
`--help | grep -A3 -- --<flag>` for one flag). This skill explains the behavior
that `--help` does not state. Everything below was checked on a real build (v1.9.0;
the size thresholds and `--print-dir` also on v1.10.0) unless it is marked
"upstream guidance".

## Prerequisites

- `rocprof-sys-instrument` and `rocprof-sys-run` on `PATH`; if not, run
  `source <install-prefix>/share/rocprofiler-systems/setup-env.sh`.
- Build the target with `-g` when you can: modules are then source files and the
  tool can report file and line information. Without debug info a whole binary is
  one module named after the binary.

## Runtime instrumentation vs binary rewrite

| | Runtime instrumentation | Binary rewrite |
| --- | --- | --- |
| Selected by | the default (no `-o`) | `-o [<output>]` |
| Command | `rocprof-sys-instrument [opts] -- ./app args` | `rocprof-sys-instrument [opts] -o app.inst -- ./app` |
| What gets instrumented | the executable **and** the shared libraries it loads | only the target's own code; shared libraries are not instrumented |
| Cost (upstream guidance) | more memory, slower to set up, more run-time overhead | much faster to set up, lower overhead |
| Runs the app | yes, immediately, with `args` | no; run `rocprof-sys-run -- ./app.inst args` later |
| Arguments after the target | passed to the app | ignored (give them when you run the result) |

Measured on a small test program with one shared library: runtime mode saw 753
available functions in 7 modules and instrumented the library's function; binary
rewrite saw 14 functions in 4 modules and did not.

- Pick **runtime** when the performance of the libraries the app loads matters.
- Pick **binary rewrite** for the lowest overhead, for repeated runs, and for MPI
  applications (upstream guidance).
- `--exe-only` (runtime only) leaves every shared library out;
  `--max-library-functions N` skips shared libraries with more than `N` functions
  (default 20000; `0` disables the check).
- `-o` with no file name writes `<name>.inst` in the current directory (target in the
  current directory) or `instrumented/<name>` for a library.
- To profile a process that is already running use `rocprof-sys-attach`, not this tool.

## Function vs module

- A **function** is one symbol, for example `big_a` or `Domain::BuildMesh`.
- A **module** is the container a function is defined in: its **source file** when
  the binary has debug info (`/path/mod_a.cpp`), otherwise the **binary or library**
  itself (`/path/app`). Shared libraries appear as their own modules in runtime mode.

```bash
rocprof-sys-instrument --simulate -o app.inst --print-available modules   -- ./app
rocprof-sys-instrument --simulate -o app.inst --print-available functions -- ./app
rocprof-sys-instrument --simulate -o app.inst --print-available pair      -- ./app
```

`pair` prints one `[module] --> [function][#instructions]` line per function.
Module options (`-MI`, `-ME`, `-MR`) act on every function in the matching modules;
function options (`-I`, `-E`, `-R`) act on individual functions.

## Including, excluding and restricting

| Option | Meaning |
| --- | --- |
| `-I` / `-MI` (include) | Force these in, even if the size heuristics would skip them. Does **not** narrow anything else. |
| `-R` / `-MR` (restrict) | Instrument **only** what matches, ignoring the size heuristics. |
| `-E` / `-ME` (exclude) | Never instrument these. Always applied, and wins over include and restrict. |

Matching rules (all verified):

- Patterns are regular expressions matched anywhere in the name (substring, not
  anchored). `tiny` matches `tiny_a` and `tiny_b`; write `^tiny_a$` for an exact match.
- Several patterns after one option are OR'd: `-I tiny_a medium_b` is the same as
  `-I 'tiny_a|medium_b'`.
- Module patterns match the module name (the source file or binary path). Function
  patterns match the function name, and also the signature when `--label args` is set
  (so `-I '\(long int\)$'` only works together with `--label args`).
- Exclude beats include at any level: `-ME 'mod_' -I tiny_a` instruments nothing.
- `-MR mod_a` instruments **every** function of `mod_a`, including tiny ones and ones
  with loops, regardless of the size thresholds.

Gotcha: if you give both `--module-restrict` and `--function-restrict`, the function
restriction is silently ignored. `-MR mod_a -R '^medium_b$'` instrumented all of
`mod_a`, not the union the upstream docs describe. Use one of the two, or combine
`-MR` with `-E`.

Examples:

```bash
# Only one source file
rocprof-sys-instrument -o app.inst -MR 'solver\.cpp$' -- ./app
# Everything the heuristics pick, minus one noisy library
rocprof-sys-instrument -o app.inst -ME 'libm\.so' -- ./app
# Add a small function the heuristics skip
rocprof-sys-instrument -o app.inst -I '^small_helper$' -- ./app
```

## Dry run

A dry run shows what would be instrumented without instrumenting anything.

```bash
rocprof-sys-instrument --simulate [options] -- ./app
rocprof-sys-instrument --simulate -o app.inst --print-instrumented pair -- ./app
rocprof-sys-instrument --dump-info --simulate -o app.inst -- ./app
```

What `--simulate` does (verified):

- Exits after the analysis. It does **not** run the application and does **not**
  write the rewritten binary, even with `-o`.
- Prints a per-module summary (`N instrumented funcs in <module>`).
- With `--print-available`, `--print-instrumented`, `--print-excluded`,
  `--print-coverage` or `--print-overlapping` it also prints those lists. Each takes
  `functions`, `modules` or `pair`; the `+` forms (`functions+`, `pair+`) also show
  the function signature when `--label args` is used.

What `--dump-info` does:

- Writes `available`, `instrumented`, `excluded` and `overlapping` reports as
  `.txt` and `.json` to `rocprofsys-<name>-output/instrumentation/`, where `<name>` is
  the target (runtime) or the `-o` output name (rewrite, e.g. `rocprofsys-app.inst-output`).
- `--print-format json` (or `txt`) limits the formats. Set `ROCPROFSYS_OUTPUT_PATH`
  to move the directory. `--print-dir` had no effect in either build tested.
- The `.txt` files list address, address range, instruction count, linkage,
  visibility, module and function. The `.json` files add a per-function `heuristics`
  block (see "Why is a function missing?").
- Without `--simulate`, `--dump-info` still produces the rewritten binary as usual.

Workflow: dry run, read `instrumented` and `excluded`, adjust the regexes or
thresholds, repeat, then drop `--simulate`.

## Granularity

A function is instrumented only if it passes **both** size checks (verified):

- at least `--min-instructions` instructions (default 1024), and
- an address range of at least `--min-address-range` bytes (default 4 x the
  instruction minimum, so 4096).

Interactions that surprise people:

- Giving only `-i N` sets the address-range minimum to 0 (only instruction count
  matters), and giving only `-r N` sets the instruction minimum to 0. Giving both
  applies both. `-i 0` or `-r 0` instruments every function.
- The environment variable `ROCPROFSYS_DEFAULT_MIN_INSTRUCTIONS` changes the default
  instruction minimum, and the default range with it.
- Functions that contain loops use `--min-instructions-loop` and
  `--min-address-range-loop`, which follow the non-loop values unless set. These
  options decide whether a function with a loop is instrumented at all.
- `-l` / `--instrument-loops` additionally instruments the loops inside instrumented
  functions. In a profile each loop shows as a child entry named with its source
  range, for example `loop_a [{449,54}-{449,83}]`, and the function's own self time
  drops accordingly.

Example from a small test program (instruction counts: 7, 16, 79, 429, 3657):

| Options | Instrumented functions |
| --- | --- |
| default | only the 3657-instruction function |
| `-i 100` | the 429 and 3657 functions |
| `-i 20` | 79, 429 and 3657 |
| `-i 0` | all five |
| `-i 50 -r 1000` | 429 and 3657 (the 79-instruction function is only 278 bytes) |

More detail means more overhead and a bigger trace. Lower the thresholds or add `-l`
to see inside hot code; raise them, restrict to modules, or use `--exe-only` to cut
overhead. Other gates (`--dynamic-callsites`, `--traps`, `--loop-traps`,
`--allow-overlapping`, `--linkage`, `--visibility`) are described in `--help`.

## Why is a function missing, or why is there too much?

Missing function:

1. List what was skipped: `--simulate --print-excluded pair`, or `--dump-info --simulate`.
2. Open `excluded.json`, find the function and read its `heuristics` block. The true
   entries explain the exclusion:
   - `is_address_range_constrained`, `is_num_instructions_constrained` (and the
     `is_loop_*` variants): too small. Lower `-i`/`-r`, or add `-I`.
   - `is_user_excluded`: one of your `-E`/`-ME` patterns matches it.
   - `is_user_restricted`: a `-R`/`-MR` pattern did not match it.
   - `is_overlapping_constrained`: needs `--allow-overlapping`.
   - `is_entry_trap_constrained`, `is_exit_trap_constrained`: needs `--traps`.
   - `is_linkage_constrained`, `is_visibility_constrained`: see `--linkage`, `--visibility`.
   - `is_internal_constrained`: it belongs to the profiler's own libraries.
3. If the function is in a shared library, remember binary rewrite never sees it.

Too much data or too slow: raise `-i`, restrict to the modules you care about
(`-MR`), exclude noisy ones (`-ME`), drop `-l`, or add `--exe-only`.

## Libraries, RPATH and troubleshooting

Rewriting a shared library so a normal executable loads the instrumented copy:

```bash
rocprof-sys-instrument -o ./app.inst -- ./app
mkdir inst && rocprof-sys-instrument -o inst/libfoo.so -- ./libfoo.so
LD_LIBRARY_PATH=$PWD/inst:$LD_LIBRARY_PATH rocprof-sys-run -- ./app.inst
ldd ./app.inst | grep libfoo      # must resolve into inst/
```

- Keep the same file name for the rewritten library (`libfoo.so`, or the soname
  that `ldd` shows) and write it to a different directory.
- **RPATH vs RUNPATH:** a binary that has `DT_RPATH` ignores `LD_LIBRARY_PATH`, so the
  original library is still loaded (verified; no `foo_*` entries appeared in the
  profile). Check with `objdump -p <exe> | egrep 'RPATH|RUNPATH'`. `RUNPATH` honors
  `LD_LIBRARY_PATH` (verified). Relink with `-Wl,--enable-new-dtags`, or change the
  entry with `patchelf --remove-rpath` / `--set-rpath` (upstream guidance, not tested
  here, `patchelf` was not installed).
- **Library as a runtime target:** `rocprof-sys-instrument -- ./libfoo.so` prints
  "is not executable ... Switching to binary rewrite mode and assuming
  '--simulate --all-functions'", so you get a dry-run approximation and no run.
  `-f/--force` is only meant for executables whose name looks like a library;
  forcing it on a real `.so` aborted inside Dyninst in testing.
- **Main function not found:** pass it with `-m <symbol>` (or `--main-function`).
- **"Failed to transform trace ... in function '<f>'"** (upstream guidance, not
  reproduced): add `-E '<f>'`. Functions that share code with it can still pull it
  in; exclude those too, or exclude the whole module with `-ME`.

## Running the result and embedding defaults

- Run a rewritten binary with `rocprof-sys-run -- ./app.inst args`; the instrumented
  functions then appear in the profile or trace like any other.
- `--env VAR=VALUE ...` embeds defaults in the new binary so a later session does not
  need to export them. A variable set at run time still overrides the embedded value
  (verified with `ROCPROFSYS_PROFILE`).
- `--label args file line return` adds that information to function names (for
  example `big_a(long int)` with `args`).
- `-M sampling` instruments only `main` to start and stop the sampler (verified), and
  is deprecated upstream; for sampling use `rocprof-sys-sample` or a preset.

## Common Mistakes

| Mistake | Fix |
| --- | --- |
| Expecting binary rewrite to instrument shared libraries | It only covers the target's own code. Use runtime mode, or rewrite the library too. |
| Using `-I` to narrow the selection | `-I` only adds. Narrow with `-R`/`-MR`, or exclude with `-E`/`-ME`. |
| Writing `-I tiny` and expecting an exact match | Patterns are unanchored substrings. Use `^tiny_a$`. |
| Combining `-MR` and `-R` | The function restriction is ignored. Use one, or `-MR` plus `-E`. |
| Lowering only `-i` and expecting range to still apply | One option given sets the other to 0. Pass both to apply both. |
| Passing app arguments after the target in a rewrite | They are ignored; pass them to `rocprof-sys-run -- ./app.inst args`. |
| Putting a rewritten library next to a binary with `DT_RPATH` | `LD_LIBRARY_PATH` is ignored. Check with `objdump -p` and relink or patch the rpath. |
| Expecting `--simulate -o` to produce the binary | A dry run writes only reports; remove `--simulate`. |
| Relying on `--print-dir` | It had no effect in tested builds; use `ROCPROFSYS_OUTPUT_PATH`. |
