# AFL smoke tests

This directory contains small AFL++ smoke tests that verify the AFL compiler
wrapper and `afl-fuzz` are available. The hello-world target is intentionally
simple and non-crashing, while the hipBLASLt transform target exercises a real
ROCm library entrypoint with bounded inputs.

## Build AFL++

If AFL++ is not already installed on `PATH`, build the bundled submodule:

```bash
git submodule update --init emulation/rocjitsu/third_party/AFLplusplus
make -C emulation/rocjitsu/third_party/AFLplusplus source-only
```

## Compile the test

Configure rocjitsu with tests enabled, then build one of the instrumented
targets:

```bash
cmake -S emulation/rocjitsu -B build/rocjitsu -G Ninja -DBUILD_TESTING=ON
cmake --build build/rocjitsu --target hello_world_afl
cmake --build build/rocjitsu --target hello_world_afl_persistent
cmake --build build/rocjitsu --target two_vector_add_afl
cmake --build build/rocjitsu --target two_vector_add_afl_persistent
cmake --build build/rocjitsu --target hipblaslt_transform_afl
cmake --build build/rocjitsu --target hipblaslt_transform_afl_persistent
```

If AFL++ is built outside the rocjitsu tree, pass its checkout path:

```bash
cmake -S emulation/rocjitsu -B build/rocjitsu -G Ninja \
  -DBUILD_TESTING=ON \
  -DRJ_AFL_ROOT=$HOME/code/AFLplusplus
```

The compiled binaries are written to:

```text
build/rocjitsu/fuzzer/tests/bin/hello_world_afl
build/rocjitsu/fuzzer/tests/bin/hello_world_afl_persistent
build/rocjitsu/fuzzer/tests/bin/two_vector_add_afl
build/rocjitsu/fuzzer/tests/bin/two_vector_add_afl_persistent
build/rocjitsu/fuzzer/tests/bin/hipblaslt_transform_afl
build/rocjitsu/fuzzer/tests/bin/hipblaslt_transform_afl_persistent
```

## Fuzz the test

Create a seed corpus and run AFL++ for a bounded smoke run. The regular target
reads one input from `stdin` and exits:

```bash
mkdir -p build/rocjitsu/fuzzer/tests/hello_world_inputs
printf 'hello\n' > build/rocjitsu/fuzzer/tests/hello_world_inputs/seed.txt
rm -rf build/rocjitsu/fuzzer/tests/hello_world_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i build/rocjitsu/fuzzer/tests/hello_world_inputs \
  -o build/rocjitsu/fuzzer/tests/hello_world_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/hello_world_afl
```

The persistent target is built from the same source with
`RJ_AFL_PERSISTENT_MODE` defined. It calls `__AFL_INIT()` once, then processes
inputs inside `__AFL_LOOP()` using AFL++'s shared testcase buffer:

```bash
rm -rf build/rocjitsu/fuzzer/tests/hello_world_persistent_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i build/rocjitsu/fuzzer/tests/hello_world_inputs \
  -o build/rocjitsu/fuzzer/tests/hello_world_persistent_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/hello_world_afl_persistent
```

The two-vector-add target compiles its GPU kernel at runtime with HIPRTC, then
loads the resulting code object with the HIP module API. This is intentional:
the host fuzzer must be compiled with AFL++'s compiler wrapper so AFL can
instrument the fuzz target, which means we cannot build this example as a
normal `hipcc` translation unit. HIPRTC keeps device compilation separate from
the AFL-instrumented host binary. This mirrors the direction of future
rocjitsu dynamic binary translation and instrumentation work, where generated
or loaded device code can be observed and instrumented after the host harness is
already under AFL control.

```bash
rm -rf build/rocjitsu/fuzzer/tests/two_vector_add_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i emulation/rocjitsu/fuzzer/tests/seeds/two_vector_add \
  -o build/rocjitsu/fuzzer/tests/two_vector_add_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/two_vector_add_afl
```

The persistent two-vector-add target uses the same HIPRTC harness with
`RJ_AFL_PERSISTENT_MODE` defined:

```bash
rm -rf build/rocjitsu/fuzzer/tests/two_vector_add_persistent_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i emulation/rocjitsu/fuzzer/tests/seeds/two_vector_add \
  -o build/rocjitsu/fuzzer/tests/two_vector_add_persistent_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/two_vector_add_afl_persistent
```

The hipBLASLt transform target interprets AFL input bytes as bounded matrix
dimensions, transposition flags, leading dimensions, scalar values, and input
matrix values before calling `hipblasLtMatrixTransform`. It uses the checked-in
seed corpus:

```bash
rm -rf build/rocjitsu/fuzzer/tests/hipblaslt_transform_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i emulation/rocjitsu/fuzzer/tests/seeds/hipblaslt_transform \
  -o build/rocjitsu/fuzzer/tests/hipblaslt_transform_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/hipblaslt_transform_afl
```

The persistent hipBLASLt target uses the same harness with
`RJ_AFL_PERSISTENT_MODE` defined:

```bash
rm -rf build/rocjitsu/fuzzer/tests/hipblaslt_transform_persistent_findings

AFL_I_DONT_CARE_ABOUT_MISSING_CRASHES=1 \
AFL_NO_UI=1 \
AFL_SKIP_CPUFREQ=1 \
afl-fuzz \
  -i emulation/rocjitsu/fuzzer/tests/seeds/hipblaslt_transform \
  -o build/rocjitsu/fuzzer/tests/hipblaslt_transform_persistent_findings \
  -V 1 -- \
  build/rocjitsu/fuzzer/tests/bin/hipblaslt_transform_afl_persistent
```

CTest runs the same flow with:

```bash
ctest --test-dir build/rocjitsu -R RocFuzz.HelloWorldAFL -V
ctest --test-dir build/rocjitsu -R RocFuzz.TwoVectorAddAFL -V
ctest --test-dir build/rocjitsu -R RocFuzz.HipblasLtTransformAFL -V
```

## Read AFL output

AFL++ writes findings under the output directory:

```text
build/rocjitsu/fuzzer/tests/hello_world_findings/default/
build/rocjitsu/fuzzer/tests/hello_world_persistent_findings/default/
build/rocjitsu/fuzzer/tests/two_vector_add_findings/default/
build/rocjitsu/fuzzer/tests/two_vector_add_persistent_findings/default/
build/rocjitsu/fuzzer/tests/hipblaslt_transform_findings/default/
build/rocjitsu/fuzzer/tests/hipblaslt_transform_persistent_findings/default/
```

Useful files and directories:

- `queue/` contains inputs that reached new coverage. Replay one with
  `build/rocjitsu/fuzzer/tests/bin/hello_world_afl < path/to/queue/id:...`.
- `crashes/` contains crashing inputs if the target ever crashes.
- `hangs/` contains inputs that exceeded AFL's timeout.
- `fuzzer_stats` contains summary counters such as executions, paths found, and
  crashes.

For example:

```bash
sed -n '1,40p' build/rocjitsu/fuzzer/tests/hello_world_findings/default/fuzzer_stats
ls build/rocjitsu/fuzzer/tests/hello_world_findings/default/queue
```
