# Local ConSan Testing

This is the local runbook for ConSan development workspaces. It uses paths
relative to the workspace root so the same commands apply to the gfx1201 and
gfx950 machines. Architecture-specific results below name the machine on which
they were obtained.

See [SPILLING.md](SPILLING.md) for the resource-path invariants exercised by
the focused and live spill tests below.

## Local Paths

The expected source/build layout is:

```text
$WORKSPACE_ROOT/
|-- TheRock/
|   `-- rocm-systems/
|-- TheRock-build/                 # out-of-tree build (one alternative)
|-- TheRock/build/                 # in-tree build (the other alternative)
|-- hip-moi/
|-- hip-moi-build/
|-- hip-moi-build-gfx1201-rocjitsu/ # gfx1201 guest build
|-- iree/
|-- iree-build/
|-- iree-build-gfx1201-rocjitsu/    # gfx1201 guest build
|-- rocjitsu-build-gfx1201-emu/     # gfx1201 emulator build
`-- rocjitsu-test-corpus/
```

From the `rocm-systems` checkout, initialize the common paths with:

```sh
export ROCM_SYSTEMS_DIR="$(git rev-parse --show-toplevel)"
export WORKSPACE_ROOT="$(cd "$ROCM_SYSTEMS_DIR/../.." && pwd)"
export ROCJITSU_SOURCE_DIR="$ROCM_SYSTEMS_DIR/emulation/rocjitsu"
export ROCJITSU_BUILD_DIR="$ROCJITSU_SOURCE_DIR/build"
if test -d "$WORKSPACE_ROOT/TheRock-build/dist/rocm"; then
  export THEROCK_BUILD_DIR="$WORKSPACE_ROOT/TheRock-build"
elif test -d "$WORKSPACE_ROOT/TheRock/build/dist/rocm"; then
  export THEROCK_BUILD_DIR="$WORKSPACE_ROOT/TheRock/build"
else
  echo "No TheRock ROCm build found under $WORKSPACE_ROOT" >&2
  return 1 2>/dev/null || exit 1
fi
export ROCM_DIST_DIR="$THEROCK_BUILD_DIR/dist/rocm"
export HOST_LLVM_DIR="$HOME/LLVM-21.1.8-Linux-X64"
export CC="$HOST_LLVM_DIR/bin/clang"
export CXX="$HOST_LLVM_DIR/bin/clang++"
export HIP_MOI_SOURCE_DIR="$WORKSPACE_ROOT/hip-moi"
export HIP_MOI_BUILD_DIR="$WORKSPACE_ROOT/hip-moi-build"
export IREE_SOURCE_DIR="$WORKSPACE_ROOT/iree"
export IREE_BUILD_DIR="$WORKSPACE_ROOT/iree-build"
export ROCJITSU_TEST_CORPUS_DIR="$WORKSPACE_ROOT/rocjitsu-test-corpus"
export RJ_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export GFX1201_ROCJITSU_BUILD_DIR="$WORKSPACE_ROOT/rocjitsu-build-gfx1201-emu"
export GFX1201_HIP_MOI_BUILD_DIR="$WORKSPACE_ROOT/hip-moi-build-gfx1201-rocjitsu"
export GFX1201_IREE_BUILD_DIR="$WORKSPACE_ROOT/iree-build-gfx1201-rocjitsu"
export GFX1201_EMULATOR="$GFX1201_ROCJITSU_BUILD_DIR/tools/rocjitsu/rocjitsu"
export GFX1201_EMULATOR_CONFIG="$ROCJITSU_SOURCE_DIR/configs/gfx1201_r9700.json"
```

Override any individual variable after this block if a build directory uses a
different location. Both the sibling `TheRock-build` layout and TheRock's
in-tree `TheRock/build` layout are detected. The source repositories `hip-moi`,
`iree`, and `rocjitsu-test-corpus` are siblings of `TheRock`; the last one is
optional for the authoritative ConSan tier matrix.

## Ground Rules

- Use the workspace's TheRock ROCm build at `$ROCM_DIST_DIR` for runtime tests.
- Use `$CC` and `$CXX` (Clang 21.1.8 above) for host C/C++ configuration. HIP
  compilation still uses the compiler supplied by `$ROCM_DIST_DIR`.
- Limit GPU test parallelism to about `8`.
- Do not run multiple IREE `ctest` sweeps against the same build directory at
  the same time; several tests share temporary paths.
- Build C++ with all useful CPU cores. The `-j8` rule is for GPU tests, not
  compilation.

Common environment:

```sh
export RJ_ROCM="$ROCM_DIST_DIR"
export ROCM_PATH=$RJ_ROCM
export HIP_PATH=$RJ_ROCM
export LD_LIBRARY_PATH=$RJ_ROCM/lib
export HSA_TOOLS_LIB=$RJ_HOOK
```

## Build And Unit Tests

Build:

```sh
cmake --build "$ROCJITSU_BUILD_DIR" \
  --target rocjitsu_tests rocjitsu_dbi_hooks
```

Focused ConSan unit/synthetic tests:

```sh
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
"$ROCJITSU_BUILD_DIR/tests/rocjitsu_tests" \
  '--gtest_filter=ConSanResourcePlan.*:ConSanMoi.*:SpillManager.*:InstructionBuilder.*'
```

Known gfx950 checkpoint result at commit `bd297cdbfc`:

- Full `rocjitsu_tests`, including registered benchmark-style tests: 1539/1539.
- The authoritative ConSan/resource/placement focused filter documented in
  `SPILLING.md`: 263/263. The commit-qualified count makes later additions easy
  to distinguish from this accepted checkpoint.

Later qualification checkpoints:

- A complete 1542/1542 run and a later 1548/1548 run were recorded as tests
  were added. At `7fab090280`, the current complete suite passes 1549/1549.
- At `e75d265ad7`, the broad focused filter printed in `SPILLING.md` passes
  268/268.
- The allocation/MOI/spill/emitter subset printed immediately above: 256/256.
  These totals describe different filters and are intentionally reported
  separately.
- The explicit local gfx1201 encoding and target-object subset
  (`--gtest_filter=*Gfx1201*:*gfx1201*`): 3/3 at its recorded checkpoint.
  Physical-device gfx1201 tests still require that workspace; the emulator
  tier below supplies separate live guest evidence on this gfx950 host.

gfx950 environment and spill baseline:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R '^ConSanSpillHipTest\.Gfx950.*ScratchRoundTrip$' \
  --output-on-failure
```

Known gfx950 result:

- The agent is an AMD Instinct MI355X, target
  `amdgcn-amd-amdhsa--gfx950:sramecc+:xnack-`, wave64, using TheRock HSA runtime
  1.21 and ROCk/amdgpu 6.14.14.
- 2/2 standalone CDNA4 spill tests pass. They execute the exact
  `scratch_store_dword` / `scratch_load_dword` plus `s_waitcnt vmcnt(0)`
  sequence emitted by the gfx950 backend under full and partial EXEC masks.

Focused gfx1201 spill hardware smoke:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R '^ConSanSpillHipTest.Gfx1201VgprScratchRoundTrip$' \
  --output-on-failure
```

Known local result:

- 1/1 passed. The kernel has a 32-byte fixed private segment and executes the
  same address-free `scratch_store_b32` / `scratch_load_b32` encodings emitted
  by the R1C backend around a deliberately clobbered live VGPR.

Focused MOI spill vertical and live regression slice:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R '^(ConSanSpillHipTest|ConSanMoiHipTest\.)' \
  --parallel 8 --output-on-failure
```

Known local result:

- 15/15 passed. This includes forced-spill record/replay and sampled tests whose
  original kernel private size is zero. The hook raises the patched kernel's
  dispatch-private size (to 12 and 20 bytes respectively); both tests keep
  eight values live across the patched LDS access, verify every value after
  restoration, and require a visible MOI record/entry.
- The forced tier and kernel selector are internal CTest controls, not public
  ConSan configuration.

Automatic inline-shadow resource checks:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R '^ConSanInlineShadowTest\.Dbi(ReportsCrossWaveRace|BarrierEpochOrdersCrossWaveAccesses|PrivateEpochBarrierOrdersCrossWaveAccesses)$' \
  --parallel 3 --output-on-failure
```

Known local result:

- 3/3 passed without scratch, owner, epoch, or SGPR numbers or
  `RJ_CONSAN_MOI_INIT_OWNER_EPOCH`. The private-epoch control forces the
  zero-private entry/access/barrier representation and raises its dispatch
  private size to 20 bytes.

Full targeted MOI resource hardware tier:

```sh
ctest --test-dir "$ROCJITSU_BUILD_DIR" \
  -R '^(ConSanSpillHipTest|ConSanInlineShadowTest|ConSanMoiHipTest)\.' \
  --parallel 8 --output-on-failure
```

Known local result:

- 29/29 passed on gfx1201. The inline-shadow race, barrier, atomic, private
  epoch, and `hw_id` variants carry no register-number environment variables.
  Dynamic access and barrier-record variants also choose their EXEC/VCC/SCC
  scalar state automatically. Forced record/replay and sampled spill controls
  remain in the tier. Auto-buffer tests use per-engine defaults, and a
  deliberate 144-byte dynamic-record case verifies visible overflow.

## SuperCollider Coverage

Broad IREE e2e compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=supercollider \
RJ_CONSAN_REQUIRE_PATCH=1 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201`.

hip-moi compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=supercollider \
RJ_CONSAN_CHECK_TRAP_MODE=lds \
ctest --test-dir "$HIP_MOI_BUILD_DIR" \
  --parallel 8 --output-on-failure
```

Known local result:

- hip-moi full suite: 189/189 passed on `gfx1201` with
  `RJ_CONSAN_CHECK_TRAP_MODE=lds`.
- A separate strict-provenance inventory of `NoPipelineProd16x8` found no
  strongly proven `Group` flat sites. Its local helpers contained 31
  `MaybeGroup` sites; the seven kernels contained 903 flat sites classified as
  679 `Unknown` and 224 `MaybePrivate`, while helper functions contained 1508
  flat sites classified as 31 `MaybeGroup`, 226 `MaybePrivate`, and 1251
  `Unknown`. The likely policy is therefore required to cover those helper
  candidates; strict mode intentionally excludes them.
- The default `RJ_CONSAN_CHECK_TRAP_MODE=all` is not currently the broad
  hip-moi compatibility recipe; it has hit ambiguous flat/local-cave cases.

## MOI Coverage

Use `RJ_CONSAN_FLAVOR=moi` plus one explicit engine:

```sh
RJ_CONSAN_MOI_ENGINE=record_replay
RJ_CONSAN_MOI_ENGINE=sampled
RJ_CONSAN_MOI_ENGINE=inline_shadow
```

### MOI Record/Replay

R1D descriptor-pressure check:

- `check_rocm_hip_scan_configured.mlir` completed 5/5 checks under a 30-second
  timeout with forced spill planning scoped to the large
  `scan_64x256xf32` kernel.
- That kernel has 640 DS operations currently classified as unsupported access
  kinds, so it reaches a precise pre-allocation blocker rather than a spill
  patch. There was no hang and no silent high-VGPR borrowing.
- Live forced-spill coverage uses a kernel compiled with zero private bytes.
  Record/replay and sampled instrumentation both bind the patched kernel symbol
  to its loaded kernel object and raise the AQL dispatch-private size before
  execution.

Broad IREE compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay \
RJ_CONSAN_MAX_PATCHES=4 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201` without a buffer-size
  variable.

Guarded TileAndFuse non-vacuity run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=record_replay \
RJ_CONSAN_MAX_PATCHES=4 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.
- A strict-provenance inventory of the representative f16 TileAndFuse object
  found 185 native LDS sites and no flat sites in its two compute kernels. The
  separately loaded ROCclr support object had 18 `Unknown` flat sites and no
  `Group`/`MaybeGroup` sites. This workload therefore does not depend on the
  heuristic flat path.

### MOI Sampled

Broad IREE compatibility:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=sampled \
RJ_CONSAN_MAX_PATCHES=4 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --output-on-failure
```

Known local result:

- IREE e2e broad sweep: 209/209 passed on `gfx1201` without a buffer-size
  variable.

Guarded TileAndFuse non-vacuity run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=sampled \
RJ_CONSAN_MAX_PATCHES=4 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*large_rdna4_tileandfusewmma.*_rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.

### MOI Inline Shadow

Targeted TileAndFuse run:

```sh
HSA_TOOLS_LIB=$RJ_HOOK \
ROCM_PATH=$RJ_ROCM HIP_PATH=$RJ_ROCM LD_LIBRARY_PATH=$RJ_ROCM/lib \
RJ_CONSAN_FLAVOR=moi \
RJ_CONSAN_MOI_ENGINE=inline_shadow \
RJ_CONSAN_MOI_OWNER_SOURCE=hw_id \
RJ_CONSAN_MAX_PATCHES=1 \
RJ_CONSAN_REQUIRE_PATCH=1 \
RJ_CONSAN_MOI_REQUIRE_RECORDS=1 \
ctest --test-dir "$IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/matmul/e2e_matmul_rocm_.*rdna4_tileandfusewmma.*rocm_hip$' \
  --parallel 8 --output-on-failure
```

Known local result:

- TileAndFuse guarded subset: 5/5 passed on `gfx1201`.
- A focused f16 case with eight probes reports `[0,8)` diagnostic ranges,
  proving that a native B64 access uses the multi-cell exact-shadow path.
- The live rocJITsu tier also includes a two-wave B128 store race whose four
  cells are instrumented and diagnosed.
- A strongly classified, zero-offset `flat_store_b32` control with no native DS
  access reports the same cross-wave `[0,4)` conflict under inline shadow and
  record/replay. The inline run uses `RJ_CONSAN_FLAT_PROVENANCE=strict`.
- Scan/softmax regression subset: 3/3 passed on `gfx1201`.
- IREE e2e broad sweep: 209/209 passed on `gfx1201` with a 60-second per-test
  timeout and no register-number or buffer-size configuration. The earlier
  scan/softmax hang has not reproduced since the common resource rollout.

## hip-moi Semantic Controls

Run the independent reference suite without a ConSan hook:

```sh
ctest --test-dir "$HIP_MOI_BUILD_DIR" \
  --parallel 8 --timeout 120 --output-on-failure
```

Known local result:

- 189/189 passed on `gfx1201`, covering reference kernels, attention/WMMA,
  barriers, atomics/fences, positive race diagnostics, and tutorial cases.

## Repeatable Tier Matrix

`tests/dbi/consan_test_matrix.sh` is the authoritative fail-fast entry point.
It requires `ROCJITSU_BUILD_DIR`, `IREE_BUILD_DIR`, `HIP_MOI_BUILD_DIR`, and
`ROCM_DIST_DIR`; pass `tier0`, `tier1`, `tier2`, or `all`. See `USAGE.md` for
the exact invocation and the meaning of each tier.

On gfx950, the standard broad SuperCollider profile intentionally finds the
two typed mismatches documented below. Consequently `tier2` and `all` stop
after that profile with a nonzero status, preserving fail-fast behavior; run
the three MOI broad commands from their sections above to collect the
remaining compatibility evidence. A final `ConSan all matrix passed.` is
expected for the gfx1201 emulator/physical workflow, whose broad
SuperCollider profile has no expected failing diagnostic row.

On a gfx950 host, qualify actual gfx1201 guest code through Rocjitsu by
overriding the three build directories and asking the matrix to wrap each
CTest process. Setting `CONSAN_GPU_ARCH` explicitly is required: the native
agent enumerator correctly reports the host's gfx950, not the emulated guest.

```sh
export ROCJITSU_BUILD_DIR="$GFX1201_ROCJITSU_BUILD_DIR"
export HIP_MOI_BUILD_DIR="$GFX1201_HIP_MOI_BUILD_DIR"
export IREE_BUILD_DIR="$GFX1201_IREE_BUILD_DIR"
export RJ_HOOK="$ROCJITSU_BUILD_DIR/lib/rocjitsu/src/rocjitsu/hooks/librocjitsu_dbi_hooks.so"
export CONSAN_GPU_ARCH=gfx1201
export CONSAN_CTEST_EMULATOR="$GFX1201_EMULATOR"
export CONSAN_CTEST_EMULATOR_CONFIG="$GFX1201_EMULATOR_CONFIG"

revision="$(git rev-parse HEAD)"
evidence="$WORKSPACE_ROOT/consan-gfx1201-emulator-$revision.log"
set -o pipefail
CTEST_PARALLEL_LEVEL=8 \
  "$ROCJITSU_SOURCE_DIR/tests/dbi/consan_test_matrix.sh" all \
  2>&1 | tee "$evidence"
```

The emulator-aware tier1 runs hip-moi twice: once with `HSA_TOOLS_LIB`
explicitly cleared as an independent guest baseline and once with the
documented SuperCollider/`lds`-trap profile. The broad IREE profiles run one
after another. To establish a broad no-hook emulator baseline independently
of ConSan, use the same CTest wrapper with an empty hook:

```sh
HSA_TOOLS_LIB= \
"$GFX1201_EMULATOR" --config "$GFX1201_EMULATOR_CONFIG" -- \
ctest --test-dir "$GFX1201_IREE_BUILD_DIR" \
  -R '^iree/tests/e2e/.*(rocm_hip|rocm-rocm)' \
  --parallel 8 --timeout 60 --output-on-failure
```

These commands exercise gfx1201 binaries on the gfx950 host; they are not a
claim that `rocm_agent_enumerator` exposes a physical gfx1201. Keep their
evidence separate from the physical-machine snapshot below.

For later physical-device corroboration after integrating shared changes, run
the complete matrix from a clean gfx1201 checkout and retain
revision-qualified evidence:

```sh
test -z "$(git status --porcelain)" || {
  echo "Refusing to qualify a dirty checkout" >&2
  return 1 2>/dev/null || exit 1
}
revision="$(git rev-parse HEAD)"
evidence="$WORKSPACE_ROOT/consan-gfx1201-$revision.log"
set -o pipefail
{
  printf 'revision=%s\n' "$revision"
  "$ROCM_DIST_DIR/bin/rocm_agent_enumerator"
  CONSAN_GPU_ARCH=gfx1201 CTEST_PARALLEL_LEVEL=8 \
    "$ROCJITSU_SOURCE_DIR/tests/dbi/consan_test_matrix.sh" all
} 2>&1 | tee "$evidence"
```

The final line must be `ConSan all matrix passed.` and the recorded agent list
must contain `gfx1201`. This physical rerun is useful corroboration, not a
prerequisite for the accepted emulator gate. Do not reuse an older physical
baseline as evidence for a revision containing newer shared policy or emitter
changes.

Recorded architecture evidence across the development workspaces:

| Target | Workspace hardware | Tier evidence | Status |
| --- | --- | --- | --- |
| gfx942 | no current workspace | no current native ConSan tier | deferred |
| gfx950 | available in this workspace | native spill/identity/compact-engine tiers pass; current guarded selection is eight safe VecDistMFMA rows plus scan/softmax; current broad evidence is recorded below | accepted locally: 1549-test full suite and all four current broad profiles complete |
| gfx1201 | physical device available in the earlier gfx1201 workspace; emulated on this gfx950 host | historical physical evidence plus an accepted-revision emulator tier0/tier1 run and the separately revision-qualified broad profiles below | physical baseline retained; emulator focused, selected, and broad gates complete |
| gfx1250 | no current workspace | decoder/encoder sources and synthetic dispatch coverage only | deferred |

The harness uses 30-second focused-GPU, 60-second IREE, and 120-second hip-moi
per-test limits. A failure or timeout terminates the current profile and all
later tiers; it cannot be overwritten by a later green summary.

Historical physical gfx1201 snapshot (2026-07-10):

| Tier | SuperCollider | MOI record/replay | MOI sampled | MOI inline shadow |
| --- | ---: | ---: | ---: | ---: |
| tier0 | included in 183 unit + 37 live | passed | passed | passed |
| tier1 selected IREE | 8/8 | 8/8 | 8/8 | 8/8 |
| tier2 broad IREE | 209/209 | 209/209 | 209/209 | 209/209 |

The tier1 semantic control run was 189/189. There were no timeouts. The tier0
guards are the non-vacuity evidence: they require patches/records/diagnostics
where appropriate and cover spill, overflow, ordering, and unsupported-site
reporting. Tier1 and tier2 establish output compatibility and absence of
resource-induced hangs; they do not imply every loaded object was patchable.

gfx950-hosted gfx1201 emulator snapshot (2026-07-14):

| Evidence | Result | Revision |
| --- | --- | --- |
| Focused ConSan unit/synthetic filter | 268/268 | `7fab090280` |
| Focused ConSan emulator live gate | 37/37; real hook patches, positive diagnostics, trap propagation, and forced-spill live-value preservation | `7fab090280` |
| hip-moi guest baseline | 46/46 CTest entries: 180 nested GTests plus six tutorials | `7fab090280` |
| hip-moi SuperCollider with `RJ_CONSAN_CHECK_TRAP_MODE=lds` | 46/46 CTest entries: 180 nested GTests plus six tutorials | `7fab090280` |
| Selected IREE, each of SuperCollider, record/replay, sampled, and inline shadow | guarded TileAndFuse 5/5 plus unguarded scan/softmax compatibility 3/3 | `7fab090280` |
| Broad IREE guest baseline, no hook | 209/209 in 174.07 seconds, or 181.78 seconds including the Rocjitsu wrapper | `bd1225dcd6` |
| Broad IREE SuperCollider | 209/209 in 175.42 seconds, or 183.30 seconds wrapped | `547be88a52` |
| Broad IREE record/replay, four-patch limit | 209/209 in 205.28 seconds, or 213.14 seconds wrapped | `12244d713f` |
| Broad IREE sampled, four-patch limit | 209/209 in 179.96 seconds, or 187.37 seconds wrapped | `3b378881da` |
| Broad IREE inline shadow, one patch and `hw_id` owner | 209/209 in 178.32 seconds, or 186.32 seconds wrapped | `0ca2953479` |

The emulator was rebuilt from parent revision `e75d265ad7`, then tier0 and
tier1 both passed at that revision with the same 268 unit, 37 live, 46+46
hip-moi, four guarded 5-test, and four compatibility 3-test totals recorded
above. Those exact logs are:

- `$WORKSPACE_ROOT/e75d265ad7_rocjitsu-build-gfx1201-emu-rebuild.log`
- `$WORKSPACE_ROOT/e75d265ad7_consan-gfx1201-emulator-tier0.log`
- `$WORKSPACE_ROOT/e75d265ad7_consan-gfx1201-emulator-tier1.log`

The final selector-only child revision was then rerun rather than inferred.
Its accepted aggregate results and logs are:

- tier0: 268 focused plus 37 live, 305/305 total, in
  `$WORKSPACE_ROOT/7fab090280_consan-gfx1201-emulator-tier0.log`;
- tier1: 46 baseline hip-moi plus 46 hooked hip-moi, 20 guarded IREE, and 12
  compatibility IREE, 124/124 total, in
  `$WORKSPACE_ROOT/7fab090280_consan-gfx1201-emulator-tier1.log`.

The broad emulator profiles ran serially against the shared IREE build tree.
They completed without a waiver, timeout, unexpected guest fault, or
post-failure continuation. The no-hook run is an independent Rocjitsu guest
baseline; it is not ConSan evidence. Likewise, the broad green profiles prove
compatibility rather than non-vacuity. The focused gate and guarded selected
profiles supply the required patch, record, spill, and positive-diagnostic
evidence.

The emulator, hip-moi guest, and IREE guest trees were rebuilt at
`e75d265ad7`. Commit `7fab090280` changes only the gfx950 branch of the tier1
CTest selector; it does not change the gfx1201 selection, ConSan
implementation, Rocjitsu emulator, or guest binaries. Even so, tier0 and
tier1 were rerun at that exact child revision as recorded above. The broad
rows remain explicitly revision-qualified rather than being presented as if
all four were rerun after a selector-only change that cannot affect them.

The gfx950 guarded tier1 selector changed at `7fab090280`. Its ten rows are
the eight safe VecDistMFMA cases (f16, f32, transposed-B f16, f8E4M3FN, i8,
and the f16, bf16, and i8 block-batched cases) plus configured scan and
softmax. These matrix workloads retain a proven patchable site for each
profile. TileAndFuse MFMA objects are no longer the guarded non-vacuity set:
some contain saturated owners that require a live-VGPR spill, and those
objects correctly receive complete-object typed containment until injected
scratch operations have an explicit MFMA pipeline-hazard schedule. This is a
test-selection correction, not an expansion or removal of the documented
MFMA spill support boundary.

Earlier completed gfx950 snapshot (2026-07-14, before the current selector
change):

| Tier | SuperCollider | MOI record/replay | MOI sampled | MOI inline shadow |
| --- | ---: | ---: | ---: | ---: |
| tier0 focused | 266/266 broad focused unit/synthetic filter; 256/256 allocation/MOI/spill/emitter subset | passed guarded live controls | passed guarded live controls | passed guarded live controls |
| tier1 selected IREE | 10/10 | 10/10 | 10/10 | 10/10 |
| tier2 broad IREE (259 selected) | 257 ordinary passes + 2 typed `s_trap 0` mismatch outcomes | 259/259 | 259/259 | 259/259 serialized |

The complete raw/record-replay/sampled/inline omitted-coordinate live matrix is
14/14. The inline selected row includes the former high-SGPR and load
destination/address-overlap regressions. SuperCollider's two tier2 residuals
are sanitizer findings with typed termination, not numerical corruption,
replacement-object load failure, timeout, or resource failure. These broad
compatibility results do not enlarge supported opcode coverage. Conversely,
tier0's patch/record/diagnostic/overflow guards and the independent semantic
controls are the evidence that instrumentation executed and that intentional
races/order controls behaved as expected; a green known-correct IREE result
alone is only compatibility evidence.

The final inline-shadow sweep was serialized because the IREE build tree has a
shared `test_tmpdir`; it passed 259/259 in 137.65 seconds. A separate
parallel-8 stress run passed 258/259, with IREE 1833 failing numerically once
and then passing 20/20 immediate serial repetitions. This stress observation
is not folded into the clean serialized compatibility result.

Current `7fab090280` gfx950 evidence:

| Tier | SuperCollider | MOI record/replay | MOI sampled | MOI inline shadow |
| --- | ---: | ---: | ---: | ---: |
| tier0 focused | 268/268 unit/synthetic plus 58/58 native live controls | same shared result | same shared result | same shared result |
| tier1 selected IREE | 10/10 | 10/10 | 10/10 | 10/10 |
| tier2 broad IREE (259 selected) | 257 ordinary passes plus typed `s_trap 0` mismatch outcomes in tests 1838 and 2861 | 259/259 | 259/259 | 259/259 serialized |

The two SuperCollider outcomes are the same bounded sanitizer findings
described for the earlier snapshot, not numerical corruption, a timeout, or a
resource hang. Tier1 additionally runs the independent hip-moi suite 44/44
without a hook and 44/44 under the documented SuperCollider profile. The
guarded IREE rows use the ten-row safe selection described above and require
patches for every profile and visible records for MOI.

Retained current native logs are:

- `$WORKSPACE_ROOT/7fab090280-gfx950-tier0.log`
- `$WORKSPACE_ROOT/7fab090280-gfx950-tier1.log`
- `$WORKSPACE_ROOT/7fab090280-gfx950-tier2-serial.log` (SuperCollider)
- `$WORKSPACE_ROOT/7fab090280-gfx950-tier2-record-replay.log`
- `$WORKSPACE_ROOT/7fab090280-gfx950-tier2-sampled.log`
- `$WORKSPACE_ROOT/7fab090280-gfx950-tier2-inline-shadow-serial.log`
- `$WORKSPACE_ROOT/7fab090280-rocjitsu-tests-full.log`
- `$WORKSPACE_ROOT/7fab090280-rocjitsu-python-pytest.log`

Final typed exclusions are part of the supported boundary, rather than hidden
test skips:

- MOI excludes decoder-error owners from candidate and CFG planning while
  retaining valid owners in the same object.
- The standard record/sampled path fails open when a kernel-lifetime scalar
  identity assignment cannot be proven for every selected owner.
- gfx950 non-MFMA live-VGPR spilling is hardware-tested. A CDNA4 MFMA owner
  that itself requires a live spill receives typed `NoLegalWindow` containment
  for the complete code-object transaction until injected scratch operations
  have an explicit MFMA pipeline-hazard schedule.

## rocjitsu-test-corpus

Recorded gfx1201 workspace status:

- `gfx1201` CTS corpus under SuperCollider: 59/59 passed.
- IREE corpus: compile-only cases passed, but runtime cases were blocked by
  the corpus using an `iree-run-module` binary without HIP driver support.
- Kernels corpus: configure was blocked by a `hipblas` dependency missing from
  the local TheRock ROCm dist. A hip-matmul-only config or adding hipBLAS would
  unblock this.

## Architecture Matrix

| Architecture | Live-GPU workspace coverage |
| --- | --- |
| `gfx1201` | The 2026-07-10 results above are the historical physical-device baseline. This gfx950 workspace additionally executes actual gfx1201 guest binaries through Rocjitsu, with separate revision-qualified emulator evidence above. |
| `gfx950` | Available. Native focused and current safe selected tiers pass, and all four broad profiles reach their documented accepted outcomes at `7fab090280`; final cross-architecture acceptance is tracked in [PLAN_GFX950.md](PLAN_GFX950.md). |
| `gfx942` | No current live-GPU workspace. |
| `gfx1250` | No current live-GPU workspace. |
