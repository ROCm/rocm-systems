# RocJITsu corpus policy

`gfx1250_b0_a0_semantic_tests.json` selects semantic programs whose instruction
forms have implemented runtime translations. The companion
`gfx1250_b0_a0_semantic_rewrites.json` pins the exact non-negative offline
rewrite count for every selected executable. A zero count qualifies a
deliberate copy-path fix through runtime comparison.

These source-coverage programs are intentionally outside this qualification
until their translations or semantic fixtures are ready:

- `barrier_id_minus1_scc_test`
- `barrier_id_minus2_scc_test`
- `flat_scratch_scalar_hi_test`
- `flat_scratch_scalar_lo_test`
- `flat_scratch_vector_hi_test`
- `flat_scratch_vector_lo_test`
- `fp8_e5m3_pack_test`
- `monitor_sleep_bounded_test`
- `monitor_sleep_unbounded_test`
- `permute_pk16_test`

The offline translator currently rejects those forms as unsupported. They are
not silently accepted or treated as passing translations.

## Offline translation SHA baseline

The `rocjitsu-test-corpus` workflow records the SHA-256 of every successful
gfx1250 B0-to-A0 input and translated output after the release-lane corpus
tests succeed. A develop push uploads the canonical manifest as the
`gfx1250-b0-a0-sha-pairs` workflow artifact for 45 days. A manual workflow
dispatch on the `develop` branch also refreshes the baseline when the official
corpus repository and pinned gfx1250 corpus ref are left unchanged. The
manifest includes the fixed translation profile, pinned corpus commit,
input-manifest hash, package-lock hash, ROCm SDK version, and source commit; it
never contains the code objects themselves.

Pull-request runs upload a seven-day candidate manifest without receiving
write access. A separate trusted workflow selects the newest completed,
unexpired artifact from a successful develop push or canonical manual run,
then compares output hashes only when the corpus, input manifest, package lock,
and SDK provenance match. Before using a candidate artifact, the trusted
workflow also requires the expected source workflow path and an exact match to
the current head of a same-repository PR. Changed outputs or incompatible
provenance create or update one non-blocking warning comment on the PR. When a
later run matches, the workflow removes its stale comment.

`record-gfx1250-dbt-sha-pairs.py` runs a bounded, translation-only collection
pass after the corpus tests and validates the resulting manifest. The external
corpus harness does not expose the translated bytes, so keeping collection
separate avoids coupling its interface to this workflow. The preceding
harness run qualifies the included input set under its timeout and memory
policy; the collector repeats the per-object timeout and does not rerun
declared exclusions. It streams each output through a temporary file and
retains only its size and SHA-256. Its `finalize` command requires every pinned
corpus input to have either a successful pair or a matching declared
exclusion, which prevents partial runs from becoming a develop baseline.

## Near-timeout reporting

With `--warn-perf`, `run-corpus-tests.sh` warns about passing tests whose
runtime approaches the pytest timeout.

## gfx1201 simulator exclusions

`fpsan_global_load_tr_gfx12_w64_test` is excluded because its wave64
`global_load_tr` path currently attempts to write a lane outside the gfx1201
simulator wavefront. This is a simulator modeling gap, not a change introduced
by the corpus SDK nightly.

## Sanitizer simulator coverage

The Clang and GCC ASan+UBSan lanes run the same target-qualified corpus as the
release lane against a sanitizer-instrumented RocJITsu build. They use longer
per-test timeouts to accommodate the instrumented simulator.

Each case runs through `env` → `setpriv` → the process supervisor → `timeout` →
`rocjitsu` → the optional HIP preload helper → the corpus executable. The
run-wrapper `timeout` owns the per-test deadline and retains command output;
pytest gets 15 seconds of cleanup headroom as a failsafe. Each target's
failed-test rerun also has a 20-minute budget in CI, within the 60-minute
workflow step.

Clang sanitizer runs load HIP at child startup through the corpus-only helper.
This keeps the shared Clang ASan runtime, simulator interposer, and HIP runtime
in loader order without adding corpus-specific policy to the `rocjitsu`
command-line interface. The GCC lane uses its executable-linked ASan runtime
and leaves HIP loading to the corpus executable.

Leak detection is disabled for the sanitizer corpus subprocesses because each
target group mixes HIP-backed and pure simulator cases, while LeakSanitizer's
stop-the-world scan stalls on HIP's multi-gigabyte mappings. The current corpus
wrapper cannot vary that setting per individual case; ASan and UBSan remain
enabled throughout.

## What each file here is for

The directory holds two unrelated harnesses that happen to share a name. One
qualifies the *functional* corpus (does the simulator compute the right answer
on real programs); the other measures *time* (does a timing model agree with
hardware). Nothing in the first group reads a timing model, and nothing in the
second group asserts a result value.

### The functional corpus

| File | Purpose |
|---|---|
| `run-corpus-tests.sh` | Runs the pytest corpus under each simulated target. The entry point for everything in this group. |
| `corpus-process-supervisor.sh` | Runs one case below a pinned session leader so no process-group member outlives it. |
| `corpus-hip-preload.sh` | Loads HIP at child startup for the Clang sanitizer lane, keeping the ASan runtime, the interposer and HIP in loader order. |
| `report-near-timeout-tests.py` | Ranks passing tests by how close they came to their per-test timeout. Wired to `--warn-perf`. |
| `gfx*_skip_tests.json` | Per-target exclusions, with the reason each one carries. |
| `gfx1250_b0_a0_semantic_*.json`, `dbt/` | The gfx1250 B0-to-A0 translation corpus: selected programs, pinned rewrite counts, expected failures, package lock. |
| `validate-gfx1250-semantic-translations.py` | Checks a translation run against those pinned expectations. |
| `record-gfx1250-dbt-sha-pairs.py` | Collects the input/output SHA-256 manifest described above, and validates it. |
| `test-validate-gfx1250-semantic-translations.py`, `test-record-gfx1250-dbt-sha-pairs.py` | Unit tests for the two scripts above. |

### The timing-model harness

| File | Purpose |
|---|---|
| `rocm-meter.py` | Runs a suite of torch and Triton kernels and writes a JSON report: per case, a `torch.cuda.Event` bracket and a torch-profiler operator breakdown. Runs on hardware and, unmodified, under the emulator. |
| `run-meter-emulated.sh` | Runs that suite under `rocjitsu`, one emulator process per kernel family, writing one shard report each. A single serial pass is too slow to iterate a model against. |
| `meter_extract.py` | Pulls per-case timings out of one report, or diffs two, for looking at a single family by hand. Explains why `kernel_us` and `device_timing` differ. |
| `test_meter_extract.py` | Unit tests for the extractor. Run by ctest. |
| `meter_score.py` | Compares an emulated run against a hardware run and scores it. Owns the decision of *what* is compared; read its module docstring before reading any number it prints. `--json` emits the scored comparison. |
| `meter_report.py` | Turns one or more of those JSON payloads into a self-contained Markdown or HTML report: coverage, accuracy per category, the log2-ratio spread and its histogram, what the model gets wrong, model-versus-model, and the wall-clock cost of running the model. Adds no data to the comparison. |

`meter_report.py --selftest` renders a synthetic payload in a temporary
directory, checks that the hand-computed statistics reach both formats, that
the HTML references no external host, and that two renders of the same input
are byte-identical.

## Validating a timing model end to end

The whole sequence, from a clean tree to a report. It is written for the
gfx950 MI355X config; swap the config and the reference run together, never
one of them.

**1. Record the reference run on hardware.** On a real MI355X, with a ROCm
torch on the path:

```sh
python3 tests/corpus/rocm-meter.py \
    --tier standard --samples 5 --warmups 5 \
    --output ~/meter/real-gfx950.json
```

This is the only step that needs the physical part, and its output is the
thing everything downstream is scored against. Keep it; re-recording it under
a different torch or a different ROCm changes the target.

**2. Build rocjitsu, including the timing models.**

```sh
cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j24
```

The models load as their own shared objects and the loader refuses one whose
ABI version does not match, so a partial rebuild is a diagnostic rather than a
mystery. Build the whole target list.

**3. Select the model in the architecture config.** The `timing` block of
`configs/gfx950_mi355x_kmd.json` names it. Copy the config per model rather
than editing one in place, so a run's config is recoverable afterwards:

```sh
sed 's/"model": "leaky"/"model": "des"/' \
    configs/gfx950_mi355x_kmd.json > /tmp/gfx950-des.json
```

For the untimed baseline of step 6, take a copy with the whole `timing` block
removed. There is no "model": "none" -- a model is loaded or it is not.

**4. Run the suite under the emulator, once per model.** Time it: the wall
clock is an input to the report.

```sh
start=$SECONDS
tests/corpus/run-meter-emulated.sh \
    --config /tmp/gfx950-des.json \
    --out ~/meter/emulated-des
echo "des wall: $((SECONDS - start))s"
```

`METER_PYTHON` (or `--python`) selects the interpreter; it needs a ROCm torch,
which is not the system python. conv2d NCHW is excluded by default because it
reaches MIOpen, which does not terminate under emulation within any useful
budget; `--with-nchw` includes it behind the per-shard timeout. Every case the
run does not produce is reported by step 6 as a coverage gap, which is the
intended treatment: a dropped case is not a passed one.

**5. Score each emulated run against the reference.**

```sh
python3 tests/corpus/meter_score.py \
    --real ~/meter/real-gfx950.json \
    --emulated ~/meter/emulated-des/*.json \
    --tolerance-pct 20 \
    --json ~/meter/scored-des.json
```

Repeat for every model. The shard reports are merged here, so overlapping
`--kernel` filters are harmless.

**6. Render the report.**

```sh
python3 tests/corpus/meter_report.py \
    --scored "des=$HOME/meter/scored-des.json" \
    --scored "leaky=$HOME/meter/scored-leaky.json" \
    --wall none=744 --wall des=3120 --wall leaky=982 \
    --title "gfx950 MI355X: des vs leaky" \
    --out ~/meter/report
```

`--scored` takes `LABEL=PATH`, so the path is written with `$HOME` rather
than `~`: a tilde after `=` is not expanded by every shell.

`--wall` takes the seconds from step 4, plus the untimed baseline run, and
turns them into the multiplier the timing model costs. Omit them and the
report says the cost was not measured rather than leaving the section out;
an unmeasured cost must not read as a free one. `--generated-at TEXT` stamps a
date; without it the output carries none and two runs of the same input are
byte-identical.

The report opens with coverage, not accuracy. A suite that quietly drops cases
flatters whatever is left, so the count of reference cases that were never
scored belongs beside the accuracy claim.

### Reading the result

The score is the **spread** of `log2(model / measured)`, in octaves. Not the
bias, which is a median and cancels a model whose errors are large in both
directions; not a correlation, which is nearly free on a corpus spanning four
orders of magnitude. See `docs/timing-model.md` for the argument, and the
module docstring of `meter_score.py` for which two quantities are compared and
why the emulated bracket is scored against the hardware *kernel* time rather
than against the hardware bracket.
