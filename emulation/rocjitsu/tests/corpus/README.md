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
