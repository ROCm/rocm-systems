# RocFuzz User Manual

This is the operator-facing guide for the current RocFuzz prototype. The
architecture, productization gaps, and rocjitsu DBI design rationale remain in
[../README.md](../README.md) and
[dbi-literature-survey.md](dbi-literature-survey.md).

## Build

Build the preload from the rocjitsu build tree:

```sh
cmake --build emulation/rocjitsu/build --target rocjitsu_afl_preload
```

Run a target under the preload:

```sh
LD_PRELOAD=/path/to/librocjitsu_afl_preload.so ./target
```

The preload uses one adaptive coverage policy. It prefers previous-BB branch
edges where rocjitsu can prove the site is safe, falls back to fixed branch
counters where needed, and reports skipped sites through structured patch logs.
Users should not select between coverage modes during normal fuzzing.

## Persistent Hooks

Persistent harnesses can call these C entry points when the preload is present:

```c++
extern "C" int rocjitsu_afl_persistent_begin();
extern "C" int rocjitsu_afl_persistent_end();
```

`rocjitsu_afl_persistent_begin()` resets device counters and per-workitem
`previous_bb` state for one input. `rocjitsu_afl_persistent_end()` synchronizes
the device, copies counters back, quantizes them, and merges them into AFL's
`trace_bits`.

Set `ROCJITSU_AFL_PERSISTENT=1` or `COV_PERSISTENT=1` so ordinary
`hipDeviceSynchronize()` calls inside the target do not prematurely merge the
same iteration's device coverage.

## Environment

| Variable | Effect |
| --- | --- |
| `ROCJITSU_AFL_VERBOSE=1` | Print patching, launch, and merge diagnostics. |
| `ROCJITSU_AFL_PATCH_REPORT=/path/report.jsonl` | Append structured patch, loader, shadow-module, CCOB rebuild, and device-edge-delta records. |
| `ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1` | Fail if kernels launch but no nonzero device edge slots are observed. |
| `ROCJITSU_AFL_PERSISTENT=1` or `COV_PERSISTENT=1` | Enable explicit persistent iteration boundaries. |
| `ROCJITSU_AFL_KERNEL_INCLUDE=s` | Only patch kernels whose names contain `s`. |
| `ROCJITSU_AFL_KERNEL_EXCLUDE=s` | Skip kernels whose names contain `s`. |

Diagnostic overrides are available for regression tests and instrumentation
debugging, but they are not public coverage modes:
`ROCJITSU_AFL_DEBUG_DISABLE_EDGES=1`, `ROCJITSU_AFL_DEBUG_LAUNCH_ONLY=1`,
`ROCJITSU_AFL_DEBUG_SKIP_ENTRY_PROBE=1`, `ROCJITSU_AFL_DEBUG_EDGE_LIMIT=N`,
`ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOT_POLICY=fixed|hashed`,
`ROCJITSU_AFL_DEBUG_BRANCH_EDGE_LIMIT=N`,
`ROCJITSU_AFL_DEBUG_REQUIRE_LIVENESS_REGISTERS=1`,
`ROCJITSU_AFL_DEBUG_FORCE_FRESH_SGPRS=1`,
`ROCJITSU_AFL_DEBUG_FORCE_FRESH_VGPRS=1`,
`ROCJITSU_AFL_DEBUG_ALLOW_OPAQUE_FRESH_REGISTERS=1`,
`ROCJITSU_AFL_DEBUG_FIXED_EDGE_SLOTS=1`,
`ROCJITSU_AFL_DEBUG_DISABLE_VGPR_SCRATCH_SPILLS=1`,
`ROCJITSU_AFL_DEBUG_BRANCH_EDGE_SLOTS=1`,
`ROCJITSU_AFL_DEBUG_DISABLE_RUNTIME_SHADOW_MODULES=1`, and
`ROCJITSU_AFL_DEBUG_FORCE_RUNTIME_SHADOW_MODULES=1`.

## AFL++ Examples

For the library examples:

```sh
cd emulation/rocjitsu/fuzzer/examples
./scripts/setup-therock.sh
scripts/run-afl-example.sh rocblas-sgemm -V 60
scripts/run-afl-example.sh rocfft-c2c -V 60
scripts/run-afl-example.sh rocrand-uniform -V 60
scripts/run-afl-example.sh rocsparse-spmv -V 60
scripts/run-afl-example.sh rocsolver-getrf -V 60
scripts/run-afl-example.sh miopen-activation -V 60
```

KPACK/fatbin runtime launches use shadow-module redirection by default when the
preload can cache the selected source ELF from the active registration.

## Coverage Gates

For smoke tests that must fail when no device edge is observed, add:

```sh
ROCJITSU_AFL_REQUIRE_DEVICE_EDGES=1
```

Treat this as a short wiring check. Do not leave it enabled for long campaigns
if valid mutations can skip every patched device edge.

For the maintained bounded example coverage gate without an AFL campaign, run:

```sh
cmake --build emulation/rocjitsu/fuzzer/examples/build --target \
  rocfuzz_example_smoke_policy_gates
```

Useful lower-level validation targets include:

```sh
cmake --build emulation/rocjitsu/build --target \
  rocjitsu_afl_code_object_image_unit \
  rocjitsu_afl_probe_decode_unit \
  rocjitsu_afl_instruction_relocator_unit \
  rocjitsu_afl_dbi_smoke \
  rocjitsu_afl_ccob_report_smoke \
  rocjitsu_afl_multi_ccob_rebuild_report_smoke \
  rocjitsu_afl_kpack_shadow_report_smoke \
  rocjitsu_afl_kpack_ccob_safe_edge_report_smoke
```

See [../afl-dbi/README.md](../afl-dbi/README.md) for the longer target list,
gfxip table, wave64 checks, loader-table smoke, safe CCOB/high-edge KPACK
report checks, and CCOB inspection commands.

## Interpreting Results

- Keep launch-only examples, but report them explicitly as launch-only crash
  harnesses rather than device branch-coverage results.
- Do not interpret low AFL edge counts as an AFL failure by default. First
  inspect patch logs for selected/skipped sites, launch counts, shadow-module
  events, CCOB rebuild status, and nonzero device edge slots.
- Use `afl-dbi/tools/summarize_patch_report.py` to convert
  `ROCJITSU_AFL_PATCH_REPORT` JSONL into per-loader and per-kernel coverage
  summaries before classifying low edge counts.
- Use `examples/scripts/summarize-coverage-baseline.py` or the
  `rocfuzz_example_coverage_baseline_report` target to regenerate the
  smoke-level per-example coverage baseline from maintained JSONL reports.
- Keep crash-focused fuzzing separate from numeric checking.
