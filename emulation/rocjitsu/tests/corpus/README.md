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

## Temporary near-timeout skips

The release corpus job applies a 15-second timeout to each pytest call. A test
from any corpus suite may be temporarily added to the corresponding suite in a
target's skip-list JSON file when its measured call time is approximately
13–15 seconds. This prevents normal CI load variation from turning a passing
test into a timeout failure and adding noise to the release gate.

This policy applies to `iree`, `kernels`, `cts`, `llama`, and future corpus
suites. Only call durations are relevant because the job sets
`timeout_func_only=true`; setup and teardown durations do not justify a skip.
Record the source run, target, suite, and selector here for each temporary
near-timeout skip.

The following cases were identified in CI test:
`rocjitsu-test-corpus / test (release)`

| Target | Suite | Test selector |
| --- | --- | --- |
| gfx942 | llama | `llama.gfx942.backend_ops.FLASH_ATTN_EXT.2dce6045f861` |
| gfx942 | llama | `llama.gfx942.backend_ops.FLASH_ATTN_EXT.09f30cac3778` |
| gfx942 | llama | `llama.gfx942.backend_ops.FLASH_ATTN_EXT.b951a29db5dd` |
| gfx942 | llama | `llama.gfx942.backend_ops.MUL_MAT.3f2054def40b` |
| gfx942 | llama | `llama.gfx942.backend_ops.MUL_MAT.12d6ef2eb102` |
| gfx950 | llama | `llama.gfx950.backend_ops.FLASH_ATTN_EXT.09f30cac3778` |
| gfx950 | llama | `llama.gfx950.backend_ops.MUL_MAT.12d6ef2eb102` |
| gfx950 | llama | `llama.gfx950.backend_ops.MUL_MAT.3f2054def40b` |
| gfx950 | llama | `llama.gfx950.backend_ops.MUL_MAT_ID.8e571d7d711d` |
| gfx950 | llama | `llama.gfx950.backend_ops.FLASH_ATTN_EXT.b951a29db5dd` |
| gfx950 | llama | `llama.gfx950.backend_ops.FLASH_ATTN_EXT.2dce6045f861` |
| gfx1100 | llama | `llama.gfx1100.backend_ops.MUL_MAT_ID.bbb77a88fa25` |

Remove a temporary entry after the test has sufficient headroom below the
timeout or the timeout policy changes. Keep functional failures and permanent
unsupported cases governed by their own skip rationale.
