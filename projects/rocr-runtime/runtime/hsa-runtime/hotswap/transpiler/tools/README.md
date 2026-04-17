# Transpiler tools

Standalone utilities that wrap the running ROCR runtime. They are **not**
part of the transpiler's `ctest` suite — they exercise a live
Salmon-enabled `libhsa-runtime64.so` from the outside and exist to answer
questions that can only be answered at runtime.

Everything here is built separately from the main CMake project.

## At a glance

| tool                                     | question it answers                                                                 |
|------------------------------------------|-------------------------------------------------------------------------------------|
| [`compare_transpilers`](#compare_transpilers)        | Does the code object *load* under legacy vs Salmon?                                 |
| [`compare_correctness/`](compare_correctness/README.md) | Does the translated kernel produce the *correct numerical output* under each engine? |

"Does it load" is a weak signal. A code object can load cleanly and
still produce wrong numerical results, because the loader only validates
structural invariants (ELF format, ISA tags, kernel descriptor fields)
and not kernel semantics. `compare_correctness` closes that gap by
dispatching each translated kernel end-to-end and comparing the produced
output buffers against a CPU reference.

## `compare_transpilers` — load-level smoke test

> **Scope.** This is a **load-level** smoke test.  A `PASS` here means
> ROCR accepted the code object (`hsa_executable_load_agent_code_object`
> + `hsa_executable_freeze` both succeeded); the kernel is never
> dispatched, no inputs are supplied, no outputs are compared.  A
> transpiler that silently miscompiles will still `PASS` here.  Use
> [`compare_correctness/`](compare_correctness/README.md) for numerical
> correctness comparison.

This tool is the cheap first filter: scan a whole directory of `.co`
files (e.g. `rocblas/library`, AITER dumps, ...) to see which ones each
engine can even load.  Cases where the two engines disagree on load
success are the natural candidates to hand-author into
`compare_correctness/` for a real correctness comparison.

Head-to-head between the two engines that can run inside ROCR's
hotswap hook:

| Mode                 | Activated by                     | Implementation                           |
|----------------------|----------------------------------|------------------------------------------|
| **Legacy transpiler**| `HSA_HOTSWAP_IR_RAISER` unset    | Byte-level rewriter in `hotswap/transpiler.cpp` |
| **Salmon**           | `HSA_HOTSWAP_IR_RAISER=1`        | LLVM IR raiser in `hotswap/transpiler/`   |

The loader reads `HSA_HOTSWAP_IR_RAISER` exactly once (a `static
const char*` initializer), so a single process is pinned to one mode for
its entire lifetime. There is also no in-process fallback — Salmon fails
hard rather than delegating to the legacy path. The only way to get
apples-to-apples numbers is to run each code object under each mode in
its own subprocess and diff the outcomes. That is what this tool does.

### Build

Requires a **Salmon-enabled ROCR build** — that is, a ROCR tree configured
with `-DROCR_ENABLE_IR_RAISER=ON` and built to completion. The resulting
`libhsa-runtime64.so` contains both the legacy transpiler and Salmon.

```bash
make ROCR_BUILD=$HOME/rocr-runtime/build        # release
make ROCR_BUILD=$HOME/rocr-runtime/build debug  # -O0 -g
```

The `Makefile` links with `-rpath=$(ROCR_BUILD)/rocr/lib`, so the produced
binary will load the Salmon-enabled ROCR directly without needing
`LD_LIBRARY_PATH` at run time.

Plain `g++` invocation is equivalent:

```bash
g++ -std=c++17 -O2 \
  -I"$ROCR_BUILD/rocr/include" \
  compare_transpilers.cpp \
  -o compare_transpilers \
  -L"$ROCR_BUILD/rocr/lib" -Wl,-rpath,"$ROCR_BUILD/rocr/lib" \
  -lhsa-runtime64
```

### Run

```bash
# Single kernel
HSA_HOTSWAP_RULES=/dev/null \
  ./compare_transpilers path/to/kernel.co

# Whole directory
HSA_HOTSWAP_RULES=/dev/null \
  ./compare_transpilers /opt/rocm-7.2.1/lib/rocblas/library --recursive

# Explicit target ISA (forwarded as HSA_HOTSWAP_ISA_OVERRIDE to both modes)
HSA_HOTSWAP_RULES=/dev/null \
  ./compare_transpilers ../kernels/aiter_gfx950 --isa=gfx942

# Machine-readable output
HSA_HOTSWAP_RULES=/dev/null \
  ./compare_transpilers ./corpus --json > compare.json
```

`HSA_HOTSWAP_RULES` is required: without it, ROCR's hotswap hook does not
activate at all and both modes will look identical. The tool warns if it
is unset in the parent environment.

### Strategy

For every input file, the parent forks+execs itself twice — once as
`--child-mode=legacy`, once as `--child-mode=salmon`. Each child:

1. Sets (or unsets) `HSA_HOTSWAP_IR_RAISER` before touching ROCR.
2. Calls `hsa_init`, walks to the first GPU agent.
3. Times a single `hsa_executable_load_agent_code_object` +
   `hsa_executable_freeze` round-trip.
4. Writes one line `RESULT <PASS|FAIL|ERROR> <ms>` to stdout and exits.

The parent parses those lines, collects child stderr (for diagnostics on
failing runs), and classifies each file:

| Verdict        | Meaning                                                          |
|----------------|------------------------------------------------------------------|
| `both-pass`    | Loaded cleanly under both engines.                               |
| `both-fail`    | Neither engine could handle it.                                  |
| `LEGACY-ONLY`  | Legacy loaded, Salmon failed. Possible Salmon regression.        |
| `SALMON-ONLY`  | Salmon loaded, legacy failed. Capability Salmon adds.            |
| `error`        | Process-level problem on at least one side (hsa_init, fork, …).  |

Per-file isolation is deliberate: any state that one load leaks into
ROCR cannot influence the next file's comparison. It is slower than
batching, but this tool is for honest numbers, not throughput.

### Output

Human format (default):

```
!! CAVEAT: LOAD-LEVEL SMOKE TEST ONLY !!
   A 'PASS' here means ROCR accepted the code object
   (load + hsa_executable_freeze succeeded).  It does NOT
   mean the kernel will produce correct results when
   dispatched — a miscompilation is invisible to this tool.
   For numerical correctness, use tools/compare_correctness/.

=== compare_transpilers (load-level smoke test) ===
  target ISA   : gfx942
  inputs       : 27
  executable   : /home/…/compare_transpilers
  strategy     : per-file fork+exec isolation, both modes

  [1/27] aiter_gfx950/gemm_bf16_a.co  legacy=FAIL  salmon=PASS
  [2/27] …

FILE                                                          LEGACY      SALMON      VERDICT
-----------------------------------------------------------------------------------------------
aiter_gfx950/gemm_bf16_a.co                                   FAIL     6ms  PASS   182ms  SALMON-ONLY
aiter_gfx950/fmha_fwd.co                                      FAIL     5ms  FAIL   213ms  both-fail
    legacy stderr: …
    salmon stderr: …
…

=== Summary ===
  target ISA   : gfx942
  inputs       : 27
  both pass    : 18
  both fail    :  1
  LEGACY only  :  0  (salmon regressed relative to legacy)
  SALMON only  :  8  (salmon handles cases legacy does not)
  errors       :  0  (process-level failures, excluded from above)

  legacy pass rate: 18 / 27  (avg  5.3 ms)
  salmon pass rate: 26 / 27  (avg 190.8 ms)
```

JSON format (`--json`): a top-level object with a `"scope"` string
restating the load-level-only caveat, the target `"isa"`, and a
`"files"` array containing one object per input.  Each file object
carries `verdict` plus `legacy` and `salmon` sub-objects with `status`,
`duration_ms`, `exit_code`, `signal`, and a `stderr_tail` excerpt.
Intended for piping into diffs between runs (e.g. when tracking Salmon
load coverage over time).

### Exit code

- `0` — all files completed under both engines (even if they *disagreed*
  on pass/fail; disagreement is the signal this tool exists to surface,
  not an error).
- `1` — at least one file had a process-level `ERROR` on either side.
- `2` — argument parsing / environment / filesystem problems in the
  parent (tool did not even start comparing).

### Known limitations

- **Load-level comparison only.** The tool does not yet diff the produced
  HSACO bytes or dispatch the loaded kernel. "Both pass" means both
  engines produced *a* code object that ROCR accepted — not that the two
  code objects are semantically equivalent. Hash-of-HSACO and
  execute-and-compare-outputs are natural follow-ups; for Salmon,
  `HSA_SALMON_DUMP_DIR` already exposes the final HSACO, but the legacy
  path has no equivalent dump.
- **No parallelism.** Files are processed sequentially. A corpus of
  hundreds of kernels takes minutes, not seconds. If that becomes a
  problem, a bounded worker pool in the parent is straightforward; the
  per-file fork isolation already makes workers independent.
- **Selects the first GPU agent.** Multi-GPU systems always compare
  against agent 0.
