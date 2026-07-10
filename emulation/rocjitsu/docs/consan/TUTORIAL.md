# ConSan Tutorial

This tutorial is the team-facing entry point for trying ConSan in rocJITsu. It
covers both top-level flavors:

- `supercollider`: redundant-access LDS checking. This is the shortest useful
  mode to try on a normal workload.
- `moi`: structured memory-order instrumentation with three versioned
  `standard-v1` engine profiles. On gfx1201 these profiles are qualified over
  the same broad compatibility tier as `supercollider`; their precision and
  diagnostic shapes intentionally differ.

ConSan runs through the HSA tools hook and patches final native RDNA4 /
`gfx1201` GPU code objects at load time. It does not require rebuilding the
application being tested. [SPILLING.md](SPILLING.md) explains how MOI obtains
temporary registers without relying on globally safe register numbers.

## Prerequisites

Use existing build directories:

- `ROCM_SYSTEMS_DIR`: checkout of `ROCm/rocm-systems` containing ConSan.
- `ROCJITSU_BUILD_DIR`: rocJITsu CMake build directory.
- `ROCM_DIST_DIR`: ROCm installation or TheRock-built ROCm distribution.
- `IREE_BUILD_DIR`: optional HIP-enabled IREE build directory with CTest
  metadata.

Build the hook:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" --target rocjitsu_dbi_hooks -j8
```

Common environment:

```sh
export HSA_TOOLS_LIB="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export LD_LIBRARY_PATH="$ROCM_DIST_DIR/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export RJ_CONSAN_LOG=1
```

The first log line to look for is a ConSan summary with:

```text
patches=N modified=true
```

`modified=true` means the HSA hook loaded patched code-object bytes.
`patches=N` counts emitted DBI patches.

## Choosing A Flavor

Use `supercollider` first when the goal is to show that ConSan can instrument a
real workload today:

```sh
export RJ_CONSAN_FLAVOR=supercollider
```

Use `moi` when the goal is to exercise structured records, exact-shadow
diagnostics, sampled watchpoints, or barrier/atomic ordering controls:

```sh
export RJ_CONSAN_FLAVOR=moi
export RJ_CONSAN_MOI_ENGINE=record_replay    # or inline_shadow, sampled
```

The MOI engine choices are:

| Engine | What it does now | Best current use |
| --- | --- | --- |
| `record_replay` | DBI probes write access/barrier/atomic records; host code replays them into diagnostics. | Highest-observability reference/debug mode. |
| `inline_shadow` | DBI probes update/check exact shadow state on the GPU for supported native multi-cell and admitted group-flat LDS forms. | Exact GPU-side checking for supported forms. |
| `sampled` | DBI probes publish compact sampled watchpoint entries; host teardown scans them, with optional immediate checking. | Lower-overhead, lower-fidelity broad mode. |

Selecting an engine activates its conservative `standard-v1` profile. Report
buffers and scratch resources are automatic; ordinary runs do not require a
buffer size or register number. Advanced composition knobs remain opt-in.

## Non-Vacuity Guards

Passing a known-correct workload is not enough evidence. The hook might have
loaded but not patched anything. Use guards to make the result meaningful.

`RJ_CONSAN_REQUIRE_PATCH=1` rejects a code object when ConSan sees supported
candidate sites for the selected mode but cannot patch any of them:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
```

For MOI auto-buffer runs, `RJ_CONSAN_MOI_REQUIRE_RECORDS=1` fails at process
teardown if no visible ConSan state was produced:

```sh
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1
```

Diagnostic-focused MOI runs can use:

```sh
export RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1
export RJ_CONSAN_MOI_FORBID_DIAGNOSTICS=1
```

Do not enable both diagnostic guards at the same time.

## Tutorial 1: SuperCollider On Any HIP/HSA Program

This is the shortest useful external snapshot.

```sh
export RJ_CONSAN_FLAVOR=supercollider
export RJ_CONSAN_DELAY_MODE=sleep
export RJ_CONSAN_DELAY=1
export RJ_CONSAN_MAX_PATCHES=1

./your-hip-or-hsa-program
```

Expected evidence:

```text
ConSan summary ... patches=1 modified=true
```

Once a focused workload is known to contain supported LDS sites, add:

```sh
export RJ_CONSAN_REQUIRE_PATCH=1
```

What this mode instruments today:

- selected native LDS `ds_load_*` and `ds_store_*` instructions;
- selected likely group/LDS `flat_load_*` and `flat_store_*` instructions;
- compact sites through local or appended trampoline caves when possible.

What it reports:

- default: `s_trap 0` on mismatch;
- optional: one-word marker buffer with `RJ_CONSAN_REPORT_BUFFER=0xADDR`.

## Tutorial 2: SuperCollider On IREE E2E Tests

Use this when you have an existing HIP-enabled IREE build directory.

Focused WMMA smoke:

```sh
export RJ_CONSAN_FLAVOR=supercollider
export RJ_CONSAN_DELAY_MODE=sleep
export RJ_CONSAN_DELAY=1
export RJ_CONSAN_MAX_PATCHES=4
export RJ_CONSAN_REQUIRE_PATCH=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/rocm_specific/check_rocm_hip_wmma_matmul_f16_wmma_matmul_f16\.mlir$' \
  --parallel 1 \
  --output-on-failure
```

Broader illustrative IREE e2e LDS-relevant inventory:

```sh
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/(encoding|linalg|math|matmul|rocm_specific|stablehlo_ops)/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 \
  --output-on-failure
```

This narrower illustrative regular expression has previously selected 152
tests. The authoritative current broad tier is the 209-test `tier2` command in
`tests/dbi/consan_test_matrix.sh`, which passed under SuperCollider and all
three MOI engines.

```text
100% tests passed, 0 tests failed out of 152
```

Representative patch evidence from the focused WMMA run:

```text
kind=local-cave-lds-load-check-trap anchor=0x3cc trampoline=0x810 original_size=8 scratch_vgpr=104
```

What this proves:

- the HSA hook loaded;
- ConSan found supported native LDS sites in final IREE RDNA4 code;
- at least one supported site was patched in each instrumentable code object;
- the known-correct IREE workload still produced correct results.

What this does not prove:

- it does not prove that the IREE kernel has a race;
- it does not produce a structured race diagnostic;
- it does not validate all LDS opcodes.

## Tutorial 3: MOI Record/Replay On IREE TileAndFuse

This mode records structured access events and replays them on the host. It is
the clearest MOI mode for seeing DBI-written data from real IREE kernels.

```sh
export RJ_CONSAN_FLAVOR=moi
export RJ_CONSAN_MOI_ENGINE=record_replay
export RJ_CONSAN_MAX_PATCHES=4
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 \
  --output-on-failure
```

Observed evidence in the current workspace:

```text
100% tests passed, 0 tests failed out of 5
```

Typical verbose evidence:

```text
ConSan MOI record_replay engine emitted an appended-cave first-light access record probe
ConSan proof patch ... kind=trampoline-moi-access-record-store ...
ConSan MOI auto report ... access_records=4 visible_records=4 ...
ConSan MOI auto record ... kind=2 ... lds_offset=616 lds_bytes=8 ...
```

What this proves:

- MOI can patch final IREE LDS instructions;
- DBI probes executed and wrote records into an HSA-tool-owned report buffer;
- the host teardown path can decode those records.

Current limitations:

- static record slots are the default;
- dynamic per-lane append requires `RJ_CONSAN_MOI_DYNAMIC_ACCESS_RECORDS=1`;
  its scalar state is selected automatically;
- IREE e2e tests are correctness tests, not intentional race tests.

## Tutorial 4: MOI Sampled Mode

Sampled mode is the lower-fidelity MOI path. It publishes compact sampled
watchpoint entries directly from DBI probes and checks them host-side at
teardown.

```sh
export RJ_CONSAN_FLAVOR=moi
export RJ_CONSAN_MOI_ENGINE=sampled
export RJ_CONSAN_MAX_PATCHES=4
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 \
  --output-on-failure
```

Optional static site throttling:

```sh
export RJ_CONSAN_MOI_SAMPLE_STRIDE=2
export RJ_CONSAN_MOI_SAMPLE_OFFSET=0
```

To keep every eligible site patched but sample one deterministic owner residue
at runtime:

```sh
export RJ_CONSAN_MOI_RUNTIME_SAMPLE_STRIDE=4
export RJ_CONSAN_MOI_RUNTIME_SAMPLE_OFFSET=0
```

The runtime stride must be a power of two in `1..1024`. Unselected waves skip
sample delay, packing, and publication; auto-buffer generations prevent stale
entries from participating in teardown replay.

For the experimental immediate checker:

```sh
export RJ_CONSAN_MOI_SAMPLED_CHECK=1
export RJ_CONSAN_MOI_REQUIRE_DIAGNOSTICS=1
```

The checker compares each sampled site with the preceding site slot and logs
`sampled_immediate_conflicts` as soon as the GPU increments the shared counter.
It is deliberately lower fidelity than host replay.

Interpretation:

- visible sampled entries prove DBI probes executed;
- sampled conflicts are lower-fidelity diagnostics;
- a clean sampled run is inconclusive and must not be presented as proof of no
  races.

## Tutorial 5: MOI Inline Shadow

Inline shadow is the exact GPU-side MOI engine. It performs direct exact-shadow
updates and emits compact diagnostics for its supported instruction forms;
unsupported forms are counted and skipped explicitly.

For focused IREE patchability smoke with hardware-ID owner initialization:

```sh
export RJ_CONSAN_FLAVOR=moi
export RJ_CONSAN_MOI_ENGINE=inline_shadow
export RJ_CONSAN_MOI_REQUIRE_RECORDS=1
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_MOI_OWNER_SOURCE=hw_id
export RJ_CONSAN_MAX_PATCHES=1

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*rdna4_tileandfusewmma.*rocm_hip$' \
  --parallel 8 \
  --output-on-failure
```

Observed evidence in the current workspace:

```text
100% tests passed, 0 tests failed out of 5
```

Interpretation:

- the inline-shadow engine can patch and execute in real IREE kernels;
- IREE correctness means non-corruption, not race detection;
- scratch, owner, epoch, and scalar registers were selected or preserved
  automatically.

For intentional race diagnostics, prefer the focused rocJITsu HIP controls
rather than known-correct IREE tests:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" -j8 \
  -R 'ConSanInlineShadowTest\.DbiReportsCrossWaveRace' \
  --output-on-failure
```

Related controls cover barrier ordering and same-address atomic handoff.

## Optional: Barrier Fault Injection

This is a destructive diagnostic, not a sanitizer mode. It proves that ConSan
can modify synchronization instructions in final native code.

```sh
export RJ_CONSAN_FLAVOR=supercollider
export RJ_CONSAN_DELAY=2
export RJ_CONSAN_REQUIRE_PATCH=1
export RJ_CONSAN_FAULT_DROP_BARRIER=1
export RJ_CONSAN_FAULT_BARRIER_INDEX=0

ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/rocm_specific/check_rocm_hip_wmma_matmul_f16_wmma_matmul_f16\.mlir$' \
  --parallel 1 \
  --output-on-failure
```

A timeout or failure here is expected when the selected barrier is semantically
important. This is not a polished race report; it is an explicit fault
injection check.

## Troubleshooting

No ConSan logs:

- Check `HSA_TOOLS_LIB`.
- Check that the process uses the same ROCm runtime whose loader honors HSA
  tools.
- Check `LD_LIBRARY_PATH` includes the ROCm `lib` directory when needed.

Logs show `patches=0 modified=false`:

- The workload may not contain supported LDS sites.
- Increase `RJ_CONSAN_LOG` for more inventory detail.
- Try the IREE WMMA or TileAndFuse commands above.

`RJ_CONSAN_REQUIRE_PATCH=1` makes broad runs fail:

- This guard is best for focused workloads.
- Broad applications often load many helper code objects, some of which have no
  supported sites or contain unsupported forms.

MOI inline-shadow reports a resource-plan skip:

- Direct-kernel scratch, owner, epoch, and scalar choices are automatic.
- A full SGPR file, dynamic private stack, or unresolved shared helper is
  rejected explicitly rather than instrumented with an unsafe register.
- `SPILLING.md` describes the supported resource and failure boundary.

Sampled mode reports no conflicts:

- That is not proof of no races.
- Sampled publishes runtime-generation-qualified entries and scans them
  host-side; optional immediate checking remains lower fidelity.
- Use exact `record_replay` or `inline_shadow` controls when a definitive race
  diagnostic is required.

## Reading Order

- `README.md`: overview and recommended starting point.
- `TUTORIAL.md`: commands to run.
- `DESIGN.md`: current architecture and capability boundaries.
- `USAGE.md`: detailed environment-variable reference.
- `PLAN.md`: dependency DAG, acceptance evidence, and deferred target breadth.
