<!-- SPDX-License-Identifier: MIT -->

# Independent runtime contract tests

This standalone CMake root builds one required test per selected family. It does not configure runtime producers. Compilation and execution are separate operations; CTest launches completed executables without building, linking or downloading anything.

| Family | Suite | First required test | Build inputs | Execution subject |
|---|---|---|---|---|
| `runtime-api` | `amdf-host` | AMDF negotiation, output preservation and inert instance/system-scope lifetime | Native C compiler and the public `ROCmRuntimeAPI::AMDF` header target | Explicit runtime prefix containing `lib/libamdf.so` |
| `ddi` | `ddi-host` | Aligned host allocation, independent lifetime and repeatable cleanup | Pinned Rust/Cargo and the selected rocddi source closure | rocddi already linked into the executable |
| `kfd` | `kfd` | KFD version-query ioctl and repeat-query consistency | Native C compiler, test-owned KFD/DRM headers and Linux sysroot | Prepared `/dev/kfd` access and compatible KFD interface |

These are narrow contract checks, not complete API, DDI or driver qualification. The KFD case exercises a real ioctl, not just device discovery; it does not submit GPU work or make firmware claims. There is no independently packaged HSA test or cross-layer suite in this slice. Private implementation unit tests retain their existing owners.

## Build each family

Build prerequisites are native Linux, CMake 3.25+, Git for source identity records, and a single-configuration generator such as Ninja. Native consumers need a C compiler. DDI additionally needs the already-installed toolchain in `../rust-toolchain.toml`; it uses the existing workspace lockfile and frozen/offline package-selected Cargo commands. DDI alone discovers no C compiler, public-header package, LLD or Python. API/KFD selections discover no Rust tools.

The public header package is independently installable and contains only AMDF headers:

```sh
cmake -S runtimes/api-headers -B build/runtime-api-headers -G Ninja
cmake --install build/runtime-api-headers --prefix "$PWD/build/runtime-api-headers-prefix"

cmake -S runtimes/testing -B build/api-tests -G Ninja \
  -DROCM_RUNTIMES_TEST_FAMILIES=runtime-api \
  -DROCmRuntimeAPI_DIR="$PWD/build/runtime-api-headers-prefix/lib/cmake/ROCmRuntimeAPI"
cmake --build build/api-tests

cmake -S runtimes/testing -B build/ddi-tests -G Ninja \
  -DROCM_RUNTIMES_TEST_FAMILIES=ddi
cmake --build build/ddi-tests

cmake -S runtimes/testing -B build/kfd-tests -G Ninja \
  -DROCM_RUNTIMES_TEST_FAMILIES=kfd
cmake --build build/kfd-tests
```

The KFD headers are already in `drivers/kfd/uapi`, with their source revision and refresh procedure. This first direct-ioctl consumer needs no thunk/support library or code-object fixture. A later consumer must declare its own actual support closure; this is not a promise that every KFD test is dependency-free.

DDI uses normal Cargo configuration/cache discovery and the toolchain's native linker. `ROCM_RUNTIMES_CARGO`, `ROCM_RUNTIMES_RUSTC`, optional `ROCM_RUNTIMES_CARGO_HOME` and optional `ROCM_RUNTIMES_CARGO_CONFIG` can select prepared inputs; explicit home/config overrides are absolute paths, and invalid inputs fail. Toolchain auto-installation and dependency downloads are disabled. Debug/default, Release, RelWithDebInfo and MinSizeRel select the matching existing Cargo profiles. A changed rocddi implementation rebuilds/relinks the test; changing a runtime prefix cannot substitute that implementation.

Select the union with `-DROCM_RUNTIMES_TEST_FAMILIES="runtime-api;ddi;kfd"`, supplying the public-header package as above. Use a fresh build directory for a different family selection so a completed prefix never retains artifacts from a previously selected family. Unsupported families/platforms and empty selections fail during configuration.

## Stage, relocate and run

Each build stages a completed prefix at `<build>/tests/`. After a successful build, copy this directory or install it to another location:

```sh
cmake --install build/api-tests --prefix "$PWD/build/api-test-prefix"
```

The prefix has `bin/<family>/` executables, `cmake/suites/` native registrations, `cmake/RunTests.cmake`, and `share/rocm-runtime-tests/<family>.txt` build records. The records identify the repository revision, exact source/header hashes, compiler/profile and executable hash. Source paths in those records are historical provenance, not execution dependencies. CTest retains the producer's record unchanged and records the launched executable's hash separately; it does not parse a second metadata protocol. Completed prefixes remain immutable inputs.

The runtime can be built separately with the existing producer; this is not part of the API test build:

```sh
cmake -S runtimes -B build/runtime -G Ninja -DBUILD_TESTING=OFF
cmake --build build/runtime
```

That producer has its own prerequisites described in the [runtime README](../README.md). Alternatively, supply an already-built compatible runtime with the same prefix layout.

```sh
ctest -S build/api-test-prefix/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=amdf-host \
  -DRUNTIME_PREFIX:PATH="$PWD/build/runtime" \
  -DRESULTS_DIRECTORY:PATH="$PWD/build/results/amdf-run-1"

ctest -S build/ddi-tests/tests/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=ddi-host \
  -DRESULTS_DIRECTORY:PATH="$PWD/build/results/ddi-run-1"

# Execute only inside a compatible, already-prepared KFD environment.
ctest -S build/kfd-tests/tests/cmake/RunTests.cmake -V \
  -DTEST_SUITE:STRING=kfd \
  -DRESULTS_DIRECTORY:PATH="$PWD/build/results/kfd-run-1"
```

Each results directory must be fresh/empty, absolute and outside both completed prefixes, which can be read-only. Rerun the same command with a new results directory. DDI and KFD reject `RUNTIME_PREFIX`. Case filtering is not supported by these one-test suites; `TEST_CASE` fails explicitly. Use the script entry point, not `ctest --test-dir <prefix>`.

The API loader opens the explicit image, checks the symbol belongs to that image, uses the public function type and holds the handle until resources are destroyed. Each invocation is a fresh process; `dlclose` does not reset a `NODELETE` runtime. The selected runtime's `lib/` is prepended to `LD_LIBRARY_PATH`, retaining existing search paths and intentional `LD_PRELOAD` interposition. This initial check records the primary image hash and loaded path; qualification of the complete transitive loaded-library closure is separate work.

The script preserves `run.txt`, `build.txt`, JUnit and CTest logs/XML under results. Missing prerequisites, an unknown suite, missing artifacts, failed assertions and a skipped/not-executed required test fail the invocation. Preflight failures retain a diagnostic `run.txt` when the results location is valid; no synthetic passing JUnit result is created for a test that did not run.

Hardware reservation, device permissions, prepared emulation/interposition and any outer recovery remain environment-owned. The KFD test requires interface major 1 and minor at least 1; it does not require a kernel matching the snapshot's minor 19. Record hardware/model identity and allocation/session cleanup alongside the run when claiming device qualification. Without usable `/dev/kfd`, this suite fails rather than skips. A host-only pass never completes the three-test milestone.

## Check the infrastructure

With the runtime producer's prepared prerequisites and Python 3 available:

```sh
python3 -B runtimes/testing/tests/test_standalone.py
```

This builds actual separate family artifacts and a real runtime, exercises host contracts from moved read-only prefixes with build tools absent from `PATH`, and checks input/selection failures, binary link isolation, loader rejection and evidence retention. Invalid-provider fixtures only test loader failures; they never count as positive runtime evidence. On hosts without `/dev/kfd`, it also verifies a required KFD run fails with zero skipped tests. An actual passing KFD run in a prepared environment is a separate acceptance requirement.
