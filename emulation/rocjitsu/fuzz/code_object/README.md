# Code-object fuzzing

`rj_code_object_fuzz` drives arbitrary bytes through the in-memory AMDGPU ELF
and HIP fat-binary loaders. Inputs that parse successfully also exercise section
classification, kernel-descriptor discovery, target lookup, load-segment sizing,
and image loading.

The built-in generator starts from a standalone gfx1250 code object, a host ELF
containing that object in a Clang offload bundle, and a truncated ELF header. It
then applies deterministic structural mutations or generates raw bytes. A seed
always reproduces the same input.

## Build and test

From the project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-code-object -GNinja -DBUILD_TESTING=ON
ninja -C build-code-object rj_code_object_fuzz rj_code_object_fuzz_test
ctest --test-dir build-code-object -j4 \
  -R '^(CodeObjectFuzzTest\.|CodeObjectFuzzCliTest$|CodeObjectFuzz(Model|Loader)Only\.SymbolBoundary$)' \
  --output-on-failure
```

Run a multithreaded local campaign with:

```sh
rj_code_object_fuzz --random 64 --seed 0 --threads auto \
  --output code-object-results
```

`load_hip_fatbin` does not yet bound `num_code_objs`, and
`CodeObject::load_to_memory` does not validate the program-header table.
Generated campaigns therefore reach known loader failures within the first few
hundred inputs. The separate [loader-hardening PR](https://github.com/ROCm/rocm-systems/pull/13704)
addresses those bounds and table-layout checks. Apply it before a sustained
campaign. When combining the changes, keep executable validation in the shared
`parse_image()` path so file-backed and in-memory inputs receive the same checks.
After that prerequisite lands, remove this qualification and increase the bounded
CLI smoke count; its current limit only characterizes the loader in this stack.

The final JSON reports how many inputs reached each parsing stage. While a case
is running, every worker keeps `current.seed` in its output directory. Generated
inputs are deterministic, so a sanitizer or fatal signal leaves a small set of
exactly reproducible candidates without putting full-input writes in the hot
path. A caught host exception is saved as `failure.bin` with its seed. Replay a
caught exception with:

```sh
rj_code_object_fuzz --input code-object-results/failure.bin
```

Replay a worker's active seed with:

```sh
rj_code_object_fuzz --random 1 \
  --seed "$(cat code-object-results/thread-0/current.seed)" \
  --threads 1 --output code-object-replay
```

`--emit-seeds DIRECTORY` writes the built-in corpus, and an AFL++ instrumented
build can consume it with `--afl`.

## AFL++ with sanitizers

Configure the loader and harness with the AFL++ compiler wrappers so coverage
reaches the parsing code:

```sh
AFL_ROOT=/path/to/AFLplusplus
cmake -S . -B build-code-object-afl -GNinja \
  -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
  -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-code-object-afl rj_code_object_fuzz

build-code-object-afl/fuzz/code_object/rj_code_object_fuzz \
  --emit-seeds code-object-seeds
"$AFL_ROOT/afl-fuzz" -i code-object-seeds -o code-object-findings \
  -m none -- \
  build-code-object-afl/fuzz/code_object/rj_code_object_fuzz --afl
```

The seed destination must be absent or empty. Use a fresh directory for each
emission so stale files cannot enter the AFL++ corpus.

Rejected objects are ordinary inputs. Exceptions, signals, sanitizer findings,
and harness invariant failures remain crashes for AFL++ to retain.
