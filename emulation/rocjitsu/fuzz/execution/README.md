# AMDGPU execution fuzzing

This target drives arbitrary and generated instruction streams through the
wavefront and compute-unit execution path for a curated set of AMDGPU targets.
Register state and mapped data memory are initialized deterministically from the
input. Every run has a fixed instruction budget so loops and waits cannot stall
the campaign.

`--random` mixes arbitrary bytes with long streams assembled from generated
target-specific encodings. The latter bias makes most inputs reach execution,
while the arbitrary subset continues to explore the decoder boundary. Without
`--target`, byte zero selects one of `gfx908`, `gfx90a`, `gfx942`, `gfx950`,
`gfx1010`, `gfx1030`, `gfx1100`, `gfx1151`, `gfx1200`, `gfx1201`, and `gfx1250`.
This covers every execution-enabled AMDGPU ISA family.
Byte one selects async and issuer modes, and the
remaining bytes form the instruction stream. `--target NAME` and the
async/issuer CLI options override their respective header fields without
changing that layout, so saved inputs can move between automatic and focused
replay. The JSON result reports both the number of inputs
and `inputs_executed`, the inputs for which at least one instruction reached the
execution callbacks. Eligible CDNA4 MFMA and CDNA5 WMMA instructions use one
shared asynchronous helper by default. Input bits select synchronous or
asynchronous execution on each capable target and occasionally execute the
same stream on two CUs concurrently. `--async auto|on|off` and
`--issuers auto|N` pin those choices for focused reproduction;
`--async-helpers N` changes the shared pool size, and zero disables it.
Random mode also defaults to up to eight fuzz threads based on the CPUs
available to the process. `--threads N` pins that outer concurrency.

Every mode executes arbitrary instruction bytes through the same wavefront and
compute-unit path. Each run initializes registers and mapped memory from the
input and stops at a fixed instruction budget.

## Build and smoke test

From this directory's project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-execution -GNinja -DBUILD_TESTING=ON
ninja -C build-execution rj_execution_fuzz rj_execution_fuzz_test
ctest --test-dir build-execution -j4 \
  -R '^(ExecutionFuzz|ExecutionCampaign)' --output-on-failure

build-execution/fuzz/execution/rj_execution_fuzz \
  --random 1000 --seed 0 --output execution-smoke
```

The active input and its numeric seed remain as `current.bin` and
`current.seed` if the executable exits early or receives a fatal signal.

## Bounded sanitizer campaign

Configure the complete simulator with ASan and UBSan, then use the campaign
runner. It checkpoints between batches and resumes from `campaign-state.json`
when invoked again with the same output directory.
Unsupported guest operands that escape the simulator as host exceptions are
retained as findings. Once the simulator translates them into wave failures,
the same inputs remain ordinary campaign cases.

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
