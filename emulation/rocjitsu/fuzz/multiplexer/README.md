# Multiplexer fuzzing

`rj_fuzz` provides one broad entry point for the simulator's focused fuzz
harnesses. Byte zero selects a registered harness; every remaining byte belongs
to that harness. Selector IDs are stable: zero selects decode, one selects
wavefront/CU execution, and two selects code-object loading. Unassigned values
are ordinary no-op inputs, leaving room to add targets without reinterpreting
saved corpora.

The decode route accepts the focused harness's exact 16-byte payload and uses
gfx1250. The execution route accepts a nonempty execution-fuzzer input of at most
1024 bytes, including its target and mode header. The code-object route accepts the
remaining payload unchanged. Strip the selector byte to replay a finding with
the corresponding focused binary:

```sh
tail -c +2 multiplexer-input.bin > focused-input.bin
rj_decode_fuzz --input focused-input.bin --json --target gfx1250
rj_execution_fuzz --input focused-input.bin
rj_code_object_fuzz --input focused-input.bin
```

The focused `rj_decode_fuzz`, `rj_execution_fuzz`, and `rj_code_object_fuzz`
binaries remain the preferred entry points for dedicated campaigns and crash
minimization.

The canonical [rocjitsu-fuzz workflow](../../../../.github/workflows/rocjitsu-fuzz.yml)
runs this target after relevant updates land on `ROCm/rocm-systems`'s `develop`
branch. It exercises sequential 32-worker ASan+UBSan and TSan campaigns and
reports the outcome on the merged pull request. Add the
`rocjitsu-presubmit-fuzz` label to a same-repository pull request to request the
same campaigns before merging. The workflow can also be started manually from
the Actions UI. A manual run on a pull request head updates the same presubmit
report when GitHub associates that commit with exactly one open pull request
targeting `develop`.

## Build and test

From this directory's project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-multiplexer -GNinja -DBUILD_TESTING=ON
ninja -C build-multiplexer rj_fuzz rj_multiplexer_fuzz_test
ctest --test-dir build-multiplexer -j4 \
  -R '^(MultiplexerFuzzTest\.|MultiplexerFuzzCliTest$)' --output-on-failure
```

## AFL++ with sanitizers

Build every simulator object with the AFL++ compiler wrappers, emit the
structured seeds, and start the multiplexer target:

```sh
AFL_ROOT=/path/to/AFLplusplus
export AFL_LLVM_THREADSAFE_INST=1
cmake -S . -B build-multiplexer-afl -GNinja \
  -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
  -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-multiplexer-afl rj_fuzz

build-multiplexer-afl/fuzz/multiplexer/rj_fuzz --emit-seeds multiplexer-seeds
"$AFL_ROOT/afl-fuzz" -i multiplexer-seeds -o multiplexer-findings -m none -- \
  build-multiplexer-afl/fuzz/multiplexer/rj_fuzz --afl
```

Keep every emitted seed and do not apply one multiplexer `-g`/`-G` size
window: decode, execution, and code-object inputs have different useful sizes.

Set `AFL_LLVM_THREADSAFE_INST=1` while compiling every instrumented object.
Execution inputs can select concurrent issuers, so AFL++'s ordinary counters
would otherwise lose updates from worker threads and report additional coverage
variance.

The multiplexer target uses the AFL++ forkserver with one input per child so
lazy initialization, execution helpers, and target switches cannot make
coverage depend on an earlier input. The focused targets use persistent loops
when maximum throughput on one subsystem matters more than cross-target
isolation.

Replay a saved input with `rj_fuzz --input FILE`. The JSON result identifies
the selected harness and reports target-specific work completed before return.
