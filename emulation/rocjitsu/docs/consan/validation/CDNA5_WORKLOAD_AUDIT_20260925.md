# CDNA5 workload selection and execution audit — September 25, 2026

The initial ledger reset relied too heavily on the old manifest. This follow-up
checks the available projects, architecture-specific implementations, inputs,
runtime linkage, and actual uninstrumented execution. It establishes how to run
each row in [STATUS_CDNA5.md](STATUS_CDNA5.md). It does **not** qualify either
ConSan engine; those cells remain empty pending clean instrumentation and fresh,
reviewed fault trials.

## Checked workspace and runtime

The audit uses `/home/benoit/workspace`, RocJITsu's GCC build, and
`configs/gfx1250_mi455x.json`. The source checkouts are hip-moi `fa9abe6`,
rocjitsu-test-corpus `1d42f89`, and iree-test-suites `49f46d6`. The separate
`consan-pytorch-gfx1250-venv` supplies PyTorch
`2.15.0a0+rocm10.1.0a20260822` and Triton `3.8.0`. Its runtime probe actually
reported `gfx1250`, dispatched a numeric canary, and loaded the current ConSan
hook. No `HSA_OVERRIDE_GFX_VERSION` is used.

All audit artifacts live in:

```text
/home/benoit/workspace/consan-validation/cdna5-audit-20260925/
```

- `env.sh`: complete, sourceable environment for this host.
- `doctor-final.json`: all 33 selected workload IDs pass the prerequisite checks.
- `commands-final.json`: every expanded clean argv, input path, filter, and
  deadline, including **all** Tensile shards.
- `baseline/*/command.json`, `output.log`, `result.json`: baseline executions;
  `oracle.json` is also retained for clients that emit a structured oracle.
- `hipmoi-build.log`, `hipmoi-fresh/`: rebuild and reruns from current hip-moi
  source, separate from the historical binaries.
- `tensile-first-shard/`: first-shard probes, preserving initial failures.
  `baseline/tensile-sk-mxf8gemm-explicit/` retains the timing-floor repair rerun.
- `f16-deadline-probe/`: diagnostic execution beyond the old 30-second deadline.

The following prerequisites needed explicit repair in the environment:

| Dependency | Selected path or correction |
| --- | --- |
| ConSan hook and launcher | `rocjitsu-gcc-build`, ahead of the SDK's stale `librocjitsu.so` |
| HIP compiler/runtime | `/home/benoit/venv/lib/python3.14/site-packages/_rocm_sdk_devel` |
| OpenMP | Add that SDK's `lib/llvm/lib`; otherwise the HipKittens executable cannot load |
| Sharktank Python | `/home/benoit/venv/bin/python`, with `iree-build/compiler/bindings/python` and `iree-build/runtime/bindings/python` on `PYTHONPATH` |
| PyTorch Python | `consan-pytorch-gfx1250-venv/bin/python`, explicitly selected |
| Tensile Python | `TheRock/.venv/bin/python3`; `/home/benoit/venv` lacks the required YAML/joblib packages |
| Tensile source | `TheRock/rocm-libraries/projects/hipblaslt/tensilelite` |
| Tensile rocisa module | Add `tensilelite-client-gfx1250-build/tensilelite/rocisa` to `PYTHONPATH` |
| Tensile client | `tensilelite-client-gfx1250-build/tensilelite/client/tensilelite-client`; the manifest's default `build_tmp` client does not exist here |
| Legacy monitoring dependencies | The existing client requires `libamd_smi.so.26`, and the Tensile driver requires a `rocm-smi` executable. `compat/lib` and `compat/bin` link only these dependencies from `/home/benoit/gpu-venv`. The rest of that older SDK is **not** added to the final library search path. These are setup dependencies, not emulator performance measurements. |

## Exact maintained entry points

On this host, source the audited environment and define the launcher:

```bash
source /home/benoit/workspace/consan-validation/cdna5-audit-20260925/env.sh
validation=/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/tests/dbi/consan/consan_validation.py
launcher='["/home/benoit/workspace/rocjitsu-gcc-build/tools/rocjitsu/rocjitsu","--config","/home/benoit/workspace/TheRock/rocm-systems/emulation/rocjitsu/configs/gfx1250_mi455x.json","--"]'

python3 "$validation" --target gfx1250 doctor --workload all \
  --launcher-json "$launcher" --json

# Substitute any ID from the tables below. This expands every clean shard,
# environment setting, and fault policy without executing it.
python3 "$validation" --target gfx1250 explain \
  --workload d128-block --profile all --json
```

For subsequent instrumented clean qualification, first resolve allowlist
prerequisites as described in [VALIDATION.md](VALIDATION.md). This host has no
physical gfx1250 for native rocprofv3 discovery. The baseline probes below run
without ConSan and do not claim to resolve that prerequisite. Do not substitute
a gfx1201 native trace for the gfx1250 executed-kernel inventory.

After selecting the campaign's allowlist policy, use this command for each
non-Tensile ID, with a new artifact root per configuration:

```bash
workload=d128-block
python3 "$validation" --target gfx1250 run --workload "$workload" \
  --profile all --phase clean --include-baseline \
  --launcher-json "$launcher" --artifact-root "$PWD/cdna5-$workload-clean"
```

For a Tensile row, use the same command **without `--launcher-json`** for the
clean run: its Python driver compiles on the host and the maintained
`run_tensile_client_with_rocjitsu.sh` wrapper launches each native client in the
emulator using `CONSAN_VALIDATION_ROCJITSU_EXE` and
`CONSAN_VALIDATION_ROCJITSU_CONFIG`. Fault runs additionally need emulated
health probes; retain the launcher there as required by the validation guide.
A full clean row executes every shard listed below. The first-shard audit is
not a substitute for that gate.

## hip-moi: individual selection and port assessment

All five selected implementations already exist under `tests/instrumented/`.
Their gfx1250 WMMA operation is `__builtin_amdgcn_wmma_f32_16x16x32_f16`, with
16 FP16 operand elements per lane and wave32. This is an actual layout change
from the RDNA4 K=16/eight-element operation, not an architecture-label change.
The packing and host references in each source were inspected.

For these commands, the executable prefix is
`hip-moi-build-gfx1250-tests/tests/hip_moi_instrumented_gfx1250_`.
Run the executable under the launcher above, followed by the exact
`--gtest_filter` below. All five passed using both the existing artifacts and a
fresh build of current source.

| ID | Executable suffix | Clean filter | Why selected |
| --- | --- | --- | --- |
| `d128-block` | `d128_attention_block_test` | `HipMoiGfx1250D128AttentionBlock.*` | D128/V128, two-wave attention, shared scores and normalization state; both exact and sampled source-checker variants |
| `d128-pressure` | `d128_attention_pressure_test` | `HipMoiGfx1250D128AttentionPressure.*` | Four host-reference tests covering full-KV and wide-key double buffering, including LDS reuse under pressure |
| `wmma-attention` | `wmma_attention_block_test` | `HipMoiGfx1250WmmaAttentionBlock.*` | Small QK/PV WMMA attention with dense shared score/weight scratch and padded PV operands |
| `streamk-arrival` | `wmma_streamk_arrival_counter_test` | `HipMoiGfx1250WmmaStreamKArrivalCounter.AcqRelFetchAddOrdersWmmaPartials` | Cross-wave publication through an acquire/release arrival counter; the intentionally relaxed sibling is not a clean control |
| `tree-atomic-or` | `wmma_streamk_tree_atomic_or_test` | `HipMoiGfx1250WmmaStreamKTreeAtomicOr.AcqRelBitmaskOrdersWmmaPartials` | Four-wave partial results published through an atomic-OR release sequence; the intentionally relaxed sibling is not a clean control |

Fresh build, using the SDK from the audited environment:

```bash
cmake -S /home/benoit/workspace/hip-moi \
  -B /home/benoit/workspace/hip-moi-build-consan-cdna5-20260925 -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_HIP_ARCHITECTURES=gfx1250 \
  -DCMAKE_CXX_COMPILER="$CONSAN_VALIDATION_ROCM_ROOT/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER="$CONSAN_VALIDATION_ROCM_ROOT/llvm/bin/clang++" \
  -DCMAKE_HIP_COMPILER_ROCM_ROOT="$CONSAN_VALIDATION_ROCM_ROOT" \
  -DCMAKE_PREFIX_PATH="$CONSAN_VALIDATION_ROCM_ROOT"
```

Build the five selected targets:

```bash
cmake --build /home/benoit/workspace/hip-moi-build-consan-cdna5-20260925 -j8 --target \
  hip_moi_instrumented_gfx1250_d128_attention_block_test \
  hip_moi_instrumented_gfx1250_d128_attention_pressure_test \
  hip_moi_instrumented_gfx1250_wmma_attention_block_test \
  hip_moi_instrumented_gfx1250_wmma_streamk_arrival_counter_test \
  hip_moi_instrumented_gfx1250_wmma_streamk_tree_atomic_or_test
```

The fresh baseline commands substitute
`hip-moi-build-consan-cdna5-20260925` for `hip-moi-build-gfx1250-tests`.
The maintained validation manifest still names the latter; rebuild that path
with the same configuration before qualifying a changed hip-moi source revision.

Other hip-moi candidates were reviewed individually:

| Candidate | gfx1250 state | Selection decision |
| --- | --- | --- |
| WMMA register handoff (`013`) | Existing port, fragment-handoff and PV-MMA host-reference tests | Reserve for localizing operand-handoff failures; current WMMA row provides the larger attention oracle |
| No-score-LDS WMMA attention (`014`) | Existing port | Reserve as a reduction of the shared-score workload, rather than another initial E2E row |
| No-score-LDS D128 attention (`015`) | Existing port | Same rationale; D128 and pressure rows retain more shared-state interaction |
| Ping-pong private LDS (`016`) | Existing port | Useful ownership control; less cross-wave interaction than selected rows |
| Cooperative LDS (`017`) | Existing port with synchronized and intentionally unsynchronized tests | Good follow-up barrier discriminator; only synchronized filters would be clean controls |
| Wide cooperative LDS (`018`) | Existing port | Reserve for bank/address coverage follow-up; pressure row is the initial larger footprint |
| LDS alias handoff (`036`) | Existing port with positive and negative tests | Good follow-up for lifetime/reuse diagnosis; do not run its whole suite as a clean ConSan control |
| Jakub cooperative matmul | Only RDNA4 source exists in this checkout; the historical gfx1250 source commit named by the old fault spec is also absent | A feasible future port, not a runnable row today. It would add an eight-wave production-shaped GEMM distinct from the small attention fixtures. |

A Jakub port would keep the eight-wave scheduling and FP32 accumulator layout,
but change the WMMA K dimension from 16 to 32, FP16 fragment width and packing
stride from 8 to 16, all fragment storage/load sizes, and the intrinsic call
signature. The lane/slot packing must be checked over the enlarged K range.
The doubled operand LDS footprint and both pipeline variants need compilation
and exact host-reference runs. The existing clean variants are no-pipeline,
pipelined, and double-buffered; there is no current `ProducerSkewProd16x8`
variant to back the removed fault specification. A port must discover new fault
sites instead of restoring that historical binary identity. No hip-moi source
changes were needed for the five selected rows.

## IREE/Sharktank and PyTorch: exact existing workloads

All entries below passed their uninstrumented emulator baseline. Qwen's build
manifest also passed the runner's freshness/provenance check.

| IDs | Actual command/input selection |
| --- | --- |
| `qwen-prefill` | `iree-run-module --device=hip`, `iree-test-suites-build/torch_models/qwen3-600m/gfx1250/qwen3-600m.vmfb`, real weights from `hf/qwen3-600m/real_weights.irpa`, input `1x5xi64` from `inference_input.0.bin`, expected `1x5x151936xf32` from `inference_output.0.bin`, threshold `0.05` |
| `tp1-prefill` | `consan_sharktank_validation.py --suite-root /home/benoit/workspace/iree-test-suites --workload tp1 --mode prefill --repetitions 1 --label tp1-prefill-clean` |
| `tp1-decode-combined` | Same driver, `--workload tp1 --mode decode-combined --repetitions 1 --skip-warmup --label tp1-decode-combined-clean` |
| `tp2-family` | Same driver, `--workload tp2 --mode prefill --repetitions 1 --label tp2-family-clean` |
| `tp2-decode` | Same driver, `--workload tp2 --mode decode --repetitions 1 --skip-warmup --label tp2-decode-clean` |
| `tp2-combined` | Same driver, `--workload tp2 --mode combined --repetitions 1 --skip-warmup --label tp2-combined-clean` |
| `clip-bf16` | Same driver, `--workload clip-bf16 --mode all --repetitions 1 --label clip-bf16-clean` |
| `pytorch-tdm-descriptor-add` | `consan_pytorch_validation.py --workload tdm-descriptor-add --repetitions 1 --label pytorch-tdm-descriptor-add-clean`; existing one-CTA and two-CTA tensor-descriptor kernels |
| `pytorch-cluster-load-sync` | Same driver, `--workload cluster-load-sync --repetitions 1 --label pytorch-cluster-load-sync-clean`; gfx1250 cluster copies and barrier sentinels |
| `pytorch-torch-mode` | Same driver, `--workload torch-mode --repetitions 1 --label pytorch-torch-mode-clean` |
| `pytorch-torch-topk` | Same driver, `--workload torch-topk --repetitions 1 --label pytorch-torch-topk-clean` |
| `pytorch-torch-sort` | Same driver, `--workload torch-sort --repetitions 1 --label pytorch-torch-sort-clean` |
| `pytorch-scatter-reduce` | Same driver, `--workload scatter-reduce --repetitions 1 --label pytorch-scatter-reduce-clean`; retain as a global-access scope control, not presumed LDS coverage |
| `pytorch-torch-histc` | Same driver, `--workload torch-histc --repetitions 1 --label pytorch-torch-histc-clean` |
| `pytorch-norm-softmax` | Same driver, `--workload norm-softmax --repetitions 1 --label pytorch-norm-softmax-clean` |

Both Python drivers live beside `consan_validation.py`; use their respective
interpreters from the environment table and prepend the emulator launcher.
Sharktank uses the existing toy MLIR/IRPA assets and compiles for `HIP_TARGET=gfx1250`.
The PyTorch rows use the existing operations and target-specific Triton APIs;
no RDNA4-only workload was ported into this set.

## rocjitsu-test-corpus and Tensile

The only kernel-case manifest explicitly listing `gfx1250` under the corpus's
`corpus/kernels/cases` is HipKittens naive BF16. Its runner, CMake target and
built executable exist. The ConSan row uses the already-defined bounded
`64 64 32 1 1` arguments, with a host numerical oracle, rather than the corpus's
larger default `256 256 256 1 1`. It passed under the emulator after fixing the
OpenMP library path. The executable is:

```text
rocjitsu-test-corpus-build/kernels-gfx1250-hipkittens/cases/hipkittens/hipkittens_gemm_bf16fp32_gfx1250_naive
```

CDNA4 HipKittens FP8/MXFP8, HIP matmul, Stream-K and rocBLAS rows are not copied
into CDNA5: those existing selections do not advertise gfx1250 support. The
Tensile configurations below already exist for gfx1250 and cover explicit
scaled operands, tensor-descriptor memory operations, sparse GEMM, and Stream-K.
No corpus workload port was made.

The configuration paths below are relative to
`rocjitsu-test-corpus/corpus/tensile/configs/Tensile/Tests/common/`, except the
bounded smoke fixture, which lives beside the validation scripts. All are
passed to `consan_tensile_validation.py` by the maintained runner, with
`--gpu-target gfx1250`, full numeric validation, one repetition and the exact
size selections in `commands-final.json`.

| ID | Existing configuration | Clean shards | Audit baseline probe |
| --- | --- | ---: | --- |
| `tensile-sk-mxf8gemm-explicit` | `streamk/gfx1250/sk_mxf8gemm_explicit.yaml` | 1 | Passed after removing correctness timing floor |
| `tensile-sk-mxf4gemm-explicit` | `streamk/gfx1250/sk_mxf4gemm_explicit.yaml` | 1 | Passed |
| `tensile-spmm-tdm-f16-transposes` | `sparse/gfx1250/spmm_tdm_f16_transposes.yaml` | 1 | Passed |
| `tensile-spmm-tdm-all` | `sparse/gfx1250/spmm_tdm_all.yaml` | 4 | Passed (first shard only) |
| `tensile-sk-mxf8f4gemm-tdm` | `streamk/gfx1250/sk_mxf8f4gemm_tdm.yaml` | 3 | Passed (first shard only) |
| `tensile-sk-mxf8gemm-tdm` | `streamk/gfx1250/sk_mxf8gemm_tdm.yaml` | 6 | Passed (first shard only) |
| `tensile-sk-mxf4gemm-tdm` | `streamk/gfx1250/sk_mxf4gemm_tdm.yaml` | 6 | Passed (first shard only) |
| `tensile-sk-sgemm-runtime-smoke` | `emulation/rocjitsu/tests/dbi/consan/fixtures/gfx1250_tensile_streamk_smoke.yaml` | 1 | Passed |
| `tensile-sk-sgemm-quick` | `streamk/gfx1250/sk_sgemm_quick.yaml` | 6 | Passed (first shard only) |
| `tensile-sk-f8gemm-quick` | `streamk/gfx1250/sk_f8gemm_quick.yaml` | 9 | Passed (first shard only) |
| `tensile-sk-hgemm-quick` | `streamk/gfx1250/sk_hgemm_quick.yaml` | 6 | Passed (first shard only) |
| `tensile-spmm-f8-ml` | `sparse/gfx1250/spmm_f8_ml.yaml` | 3 | Passed (first shard only) |

The audited 33 IDs now have successful baseline probes. Sharded Tensile rows
have only their **first** shard executed in this audit; all-shard clean
qualification and detector fault sensitivity remain to be measured.

### Deferred FP16 sparse sweep

`spmm_f16_sb.yaml` is present and compiles for gfx1250, but is not an appropriate
unchanged clean denominator. The old 30-second deadline first hid the problem.
With a diagnostic 300-second inner / 360-second outer deadline, execution
finished in 66.4 seconds with 200 CSV rows and four passing clients, but the
strict oracle rejected `DID_NOT_SATISFY_ASSERTS` entries. For example, the
K=16 problem is paired with GlobalSplitU solutions requiring K>=64. This is a
solution/input applicability mismatch, not a ConSan finding. The full failed
command, output, and reasons are retained in `f16-deadline-probe/`.

The row is deferred from both the manifest and table. A future bounded selection
could keep compatible existing problems and solutions with an explicit new
denominator. It must not silently skip rejected solutions or treat them as
numerical passes. The selected `spmm_tdm_f16_transposes.yaml` already provides
working sparse FP16 coverage.

### Validation-runner repair

MXFP8 explicit passed both numerical rows but initially failed the inherited
250 ms benchmark-duration floor (197.2 ms observed). gfx1250 **clean and fault**
Tensile commands now request a zero minimum aggregate duration. The runner
still rejects invalid/nonpositive timing, numerical failures, missing clients,
and missing target code objects. Overhead/benchmark phases retain their timing
floor. The repaired MXFP8 baseline passed. Unit tests cover the phase distinction
and preserve the existing oracle/timing validation tests.

The final manifest/runner test suite passed (207 tests); the Tensile runner
suite also passed (38 tests). These are orchestration tests, not detector
qualification evidence.
