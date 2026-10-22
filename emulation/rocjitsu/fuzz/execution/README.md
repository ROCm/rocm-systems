# AMDGPU execution fuzzing

This target runs arbitrary and generated instruction streams through the
wavefront and compute-unit execution path. Register state and mapped data
memory come from the input, and a fixed instruction budget bounds every run.

An input has three parts:

- Byte zero selects `gfx908`, `gfx90a`, `gfx942`, `gfx950`, `gfx1010`,
  `gfx1030`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`, `gfx1250`, or
  `gfx1251`. Their stable selector values are 0 through 11. Other values do no
  work.
- Byte one selects asynchronous execution, concurrent issuers, and full EXEC.
- The remaining bytes are the instruction stream.

`--target NAME`, `--async auto|on|off`, and `--issuers auto|N` override those
header fields during focused replay without changing the saved input. Generated
WMMA inputs request full EXEC so an async override preserves their guest state.

`--random` combines arbitrary bytes with streams assembled from generated
target-specific encodings. Generated streams usually reach execution, while
arbitrary inputs continue to exercise the decoder boundary. The JSON result
includes `inputs_executed`, which counts inputs that reached at least one
execution callback.

Eligible CDNA4 MFMA and CDNA5 WMMA instructions share one asynchronous helper
by default. Input bits can select synchronous or asynchronous execution and can
run the same stream on two CUs concurrently. `--async-helpers N` changes the
shared pool size; zero disables it. Random mode uses up to eight fuzz threads,
bounded by the process CPU affinity. `--threads N` overrides that concurrency.

## Build and smoke test

From this directory's project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-execution -GNinja -DBUILD_TESTING=ON
ninja -C build-execution rj_execution_fuzz rj_execution_fuzz_test
ctest --test-dir build-execution -j4 \
  -R '^(ExecutionFuzz|ExecutionCampaign)' --output-on-failure

build-execution/fuzz/execution/rj_execution_fuzz \
  --random 4 --seed 0 --output execution-smoke
```

The active input and its numeric seed remain as `current.bin` and
`current.seed` if the executable exits early or receives a fatal signal.

## Bounded sanitizer campaign

Configure the complete simulator with ASan and UBSan, then use the campaign
runner. It checkpoints between batches and resumes from `campaign-state.json`
when invoked again with the same output directory.
Guest validation failures remain visible to the campaign and are retained as
findings.

```sh
cmake -S . -B build-execution-sanitize -GNinja \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-execution-sanitize rj_execution_fuzz

python3 fuzz/execution/run_campaign.py \
  --binary build-execution-sanitize/fuzz/execution/rj_execution_fuzz \
  --output execution-campaign --duration 3600 --workers 8 --target gfx1250
```

The runner defaults to automatic per-input target selection across all listed
targets and eight isolated
worker processes, capped by the CPUs available to the process. Pass
`--target NAME` to pin a campaign. Workers own disjoint seed ranges and
separate active input files, while the parent serializes checkpoints and crash
collection.
The runner passes `--threads 1` to each process by default so its process count
is the campaign concurrency. Use `--threads-per-worker N` for a TSan campaign
that needs concurrent execution inside each process. `--async`, `--issuers`,
and `--async-helpers` override those execution modes and become part of the
checkpoint identity.
Successful batch measurements are appended to `events.jsonl`. A retained input
is copied to `crashes/<sha256>.bin`; the adjacent JSON file records its first
seed, target, exit status, diagnostic output, and replay command. Identical
inputs are deduplicated by content; related failures with different bytes are
retained separately for later classification. After a multi-threaded failure,
the worker preserves every active input and replays the complete batch with one
thread. This gives deterministic failures exact seed attribution and prevents a
signal from skipping untested seeds. Successful batch statistics exclude
failed attempts. The worker then continues collecting inputs until the duration
expires. Run one campaign process per target to divide a machine's CPUs across
targets in parallel.

## ThreadSanitizer concurrency campaign

`--threads N` executes independent wavefront/CU inputs concurrently inside one
process. All harness threads share the asynchronous matrix helper pool, so a
TSan build observes both concurrent simulator construction/execution and CDNA5
WMMA publication, helper execution, completion, and teardown. A successful run
reports a nonzero `async_executed` count. If the process terminates early, each
`thread-N` output directory retains its current input and numeric seed.

```sh
cmake -S . -B build-execution-tsan -GNinja -DRJ_ENABLE_TSAN=ON
ninja -C build-execution-tsan rj_execution_fuzz rj_execution_fuzz_test
ctest --test-dir build-execution-tsan -j4 \
  -R '^ExecutionFuzzInputTest.ThreadedInputsShareAsyncWmmaHelpers$' --output-on-failure

TSAN_OPTIONS=halt_on_error=1 python3 fuzz/execution/run_campaign.py \
  --binary build-execution-tsan/fuzz/execution/rj_execution_fuzz \
  --output execution-tsan --duration 3600 --workers 8 \
  --threads-per-worker 8 --async-helpers 4 --target gfx1250
```

## AFL++

Configure the whole project with the AFL++ compiler wrappers so coverage reaches
the simulator execution backend, then create an initial corpus and run the raw
persistent entry point:

```sh
AFL_ROOT=/path/to/AFLplusplus
cmake -S . -B build-execution-afl -GNinja \
  -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
  -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-execution-afl rj_execution_fuzz

build-execution-afl/fuzz/execution/rj_execution_fuzz \
  --emit-seeds execution-seeds --target gfx1250

"$AFL_ROOT/afl-fuzz" -i execution-seeds -o execution-findings \
  -m none -g 1 -G 1024 -- \
  build-execution-afl/fuzz/execution/rj_execution_fuzz \
  --afl --target gfx1250
```

The generated corpus contains 128 inputs spanning 8, 32, 256, and 1024 bytes,
including every automatic async and issuer mode. Omit `--target` from both seed
generation and replay when intentionally fuzzing target selection.

Signals, sanitizer findings, and unexpected simulator failures remain visible to
AFL++. Decoder rejection and unsupported instructions routed through the
wave-failure path remain ordinary inputs.

Campaign deadlines and interrupts preserve in-flight inputs under `stopped/`.
These records are replay candidates, excluded from the crash and failure counts;
being active when the campaign stops does not establish a simulator hang.
Longer random runs can uncover host failures; the campaign runner retains those
under `crashes/` and continues from its checkpoint.
