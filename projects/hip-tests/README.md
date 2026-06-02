## What is this repository for?

This repository provides unit tests for [HIP](../hip) implementation.

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated. AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes. THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

©2026 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches

The hip-tests repository maintains several branches. The branches that are of importance are:

* Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest release is rocm-5.4, main branch will be the repository based on this release.
* Develop branch: This is the default branch, on which the new features are still under development and visible. While this may be of interest to many, it should be noted that this branch and the features under development might not be stable.
* Release branches. These are branches corresponding to each ROCM release, listed with release tags, such as rocm-5.4, etc.

## Release tagging

hip-tests releases typically follow a naming convention for each ROCM release to help differentiate them.

* rocm x.yy: These are the stable releases based on the ROCM release.
  This type of release is typically made once a month.

## Build HIP Catch tests

For building HIP from source, please check instructions on the [HIP page](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html).

HIP catch tests can be built via the following instructions:

1. Clone the hip-tests source code from the repository, with definition of branch. The default branch is `develop`, as an example,
```bash
$ git clone -b develop https://github.com/ROCm/rocm-systems.git
$ cd rocm-systems/projects/hip-tests
$ export HIP_TESTS_DIR="$(readlink -f .)"
```
`hip-tests` for AMD platform now rely on `amdclang` to build, which is shipped with ROCm installation.
Although individual tests will compile with `hipcc`, ideally you should use `amdclang++`.

2. Build the catch tests
```bash
$ cd "$HIP_TESTS_DIR"
$ mkdir -p build; cd build
$ cmake ../catch -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH=<HIP-Installed-Path> -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DCMAKE_HIP_COMPILER=amdclang++ -DOFFLOAD_ARCH_STR="--offload-arch=gfxXXXX"
$ make -j$(nproc) build_tests
$ ctest # run tests
```

Replace `gfxXXXX` with your GPU architecture. Run `rocminfo` and use the **Name** shown for the GPU agent (for example `gfx1200`):

```bash
$ /opt/rocm/bin/rocminfo
...
*******
Agent 2
*******
  Name:                    gfx1200
```

Pass that value to `--offload-arch` (for example `--offload-arch=gfx1200`).

`amdclang` and `amdclang++` are installed with ROCm in the same `bin` directory as `hipcc` (commonly `/opt/rocm/bin`). If CMake reports that the compiler is missing, ensure that directory is on `PATH`, or set `-DCMAKE_CXX_COMPILER`, `-DCMAKE_C_COMPILER`, and `-DCMAKE_HIP_COMPILER` to the full paths under your ROCm installation.

HIP catch tests are built under the folder `$HIP_TESTS_DIR/build`.

### Build HIP Catch2 standalone test

`standalone_main.cc`, which was previously used, has been removed. We now rely on the main function provided by Catch2 to build standalone tests. The test suite has been upgraded to Catch2 v3 (specifically v3.8.1). This transition from v2 to v3 introduced several significant changes in how Catch2 integrates with hip-tests. Most importantly, Catch2 is no longer a single-header library; it is now a compiled library that must be linked into the test executable.

#### Getting Catch2

##### Option 1: System-wide install (recommended)

For developers working on personal machines, installing Catch2 system-wide is strongly recommended. This avoids the download and build steps within hip-tests and leads to noticeably faster build times.

- `git clone https://github.com/catchorg/Catch2.git -b v3.8.1 --depth 1`
- `cd Catch2`
- `mkdir build && cd build`
- `cmake .. -DCMAKE_BUILD_TYPE=Release`
- `make -j$(nproc)`

During installation, you may need superuser privileges to perform a global install.

- `sudo make install`

Note: On Linux systems, the default installation path is `/usr/local/`.

##### Option 2: FetchContent (build from source)

Without a system Catch2, hip-tests’ CMake pulls it in with FetchContent. Set `HIP_PATH` to your ROCm install (for example `/opt/rocm`); an empty value can break configuration. Run `make Catch2WithMain`—that target depends on `Catch2`, so both libraries are built.

```bash
cd rocm-systems
export ROCM_SYSTEMS_DIR=$PWD
export HIP_PATH=/opt/rocm
mkdir -p tests
cd tests
cmake ../projects/hip-tests/catch -DHIP_PLATFORM=amd -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HIP_PATH
make -j$(nproc) Catch2WithMain
export DEPS_PATH=$PWD/_deps
```

Outputs are under `$DEPS_PATH`; set `LD_LIBRARY_PATH` if the linker cannot find Catch2 at run time.

#### Standalone build commands

The examples below assume `ROCM_SYSTEMS_DIR` is set to the root of the rocm-systems checkout:

```bash
export ROCM_SYSTEMS_DIR=<path_to_rocm_systems>
```

##### Without hip-tests main

Compile one test `.cc` with Catch2 supplying `main`. The hip-tests test runner and its shared context are not used.

**System-wide Catch2** — link `Catch2` and `Catch2Main` (Catch2’s `main`):

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  --offload-arch=native                                                              \
  -L /usr/local/lib -lCatch2 -lCatch2Main                                            \
  -x hip <path_to_test>
```

**FetchContent Catch2** — compile `catch_main.cpp` and link `Catch2` only:

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  -I $DEPS_PATH/catch2-src/src                                                       \
  -I $DEPS_PATH/catch2-build/generated-includes                                      \
  --offload-arch=native                                                              \
  -L $DEPS_PATH/catch2-build/src                                                     \
  $DEPS_PATH/catch2-src/src/catch2/internal/catch_main.cpp                           \
  -lCatch2                                                                           \
  -x hip <path_to_test>
```

Note: `<path_to_test>` is the path to a test source file, for example `projects/hip-tests/catch/unit/device/hipRuntimeGetVersion.cc`.

##### With hip-tests main

This includes the hip-tests test context (`hip_test_context.cc`) and `main.cc`, which provide the `TestContext` (platform/device info) and picojson-based config parsing.

With a system-wide Catch2 install:

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/external/picojson                    \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/main.cc              \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_context.cc  \
  -L /usr/local/lib                                                                  \
  --offload-arch=native                                                              \
  -lCatch2                                                                           \
  -x hip <path_to_test>
```

With FetchContent Catch2:

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/external/picojson                    \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  -I $DEPS_PATH/catch2-src/src                                                       \
  -I $DEPS_PATH/catch2-build/generated-includes                                      \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/main.cc              \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_context.cc  \
  --offload-arch=native                                                              \
  -L $DEPS_PATH/catch2-build/src                                                     \
  -lCatch2                                                                           \
  -x hip <path_to_test>
```

#### Build options

##### `ENABLE_YAML_TAGS`

The cmake option `ENABLE_YAML_TAGS` (default `ON`) controls whether test-case tags from the YAML configuration are compiled into the build. When enabled, `hip_tests_config.hh` is generated from the YAML files under `catch/config/configs/` and the `HIP_TEST_CASE` macro expands to `TEST_CASE` with the configured tags. When disabled, `HIP_TEST_CASE` expands to `TEST_CASE` with empty tags.

To disable it in a cmake build:

```bash
cmake ../catch -DENABLE_YAML_TAGS=OFF ...
```

This skips generation of `hip_tests_config.hh` and the YAML validation step, which can be useful for faster iteration or when the YAML configs are not needed.

For standalone `amdclang++` builds, `ENABLE_YAML_TAGS` is not defined by default since there is no cmake involved. To opt in, pass `-DENABLE_YAML_TAGS` on the compiler command line and provide the include path to the build directory containing the generated `hip_tests_config.hh`:

```bash
amdclang++                                                                           \
  -DENABLE_YAML_TAGS                                                                 \
  -I <build_dir>                                                                     \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  --offload-arch=native                                                              \
  -L /usr/local/lib -lCatch2 -lCatch2Main                                            \
  -x hip <path_to_test>
```

`<build_dir>` is the cmake build directory where `hip_tests_config.hh` was generated (e.g., the same directory used for `cmake ../catch ...`). Without a prior cmake build that generated this header, the compilation will fail.

When `ENABLE_YAML_TAGS` is not defined (the default for standalone builds), tests compile and run normally but all tag-based filtering (e.g., `ctest -L level_2`) is unavailable.

##### Test levels (optional)

Each test case in the YAML config is assigned a level (`level_0` through `level_4`) that controls test parameters such as memory sizes, block sizes, iteration counts, and warmups. Level definitions live in `catch/config/configs/definitions.yaml`. For example, `level_0` is a quick smoke test with small sizes, while `level_2` is a comprehensive run with sizes up to 2 GB.

When building through cmake, levels work automatically. The build system generates `hip_test_parameters.hh` from `definitions.yaml`, and the event listener in `hip_test_listener.cc` detects the active level from ctest tags (e.g., `ctest -L level_0`) and loads the corresponding parameters into `TestParameterStore`.

For standalone builds, levels are an opt-in feature that requires compiling additional source files from `catch/hipTestMain/`:

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/external/picojson                    \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/main.cc              \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_context.cc  \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_features.cc \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_params.cc   \
  -x hip $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/hipTestMain/hip_test_listener.cc \
  -L /usr/local/lib                                                                  \
  --offload-arch=native                                                              \
  -lCatch2                                                                           \
  -x hip <path_to_test>
```

This requires `hip_test_parameters.hh` to have been generated by a prior cmake build (it lives in the build directory). Without these additional files, standalone builds still work but all tests use hardcoded default parameters regardless of level.

### Building with address sanitizer

To build catch tests with Address Sanitizer options, use the cmake option `-DENABLE_ADDRESS_SANITIZER=ON`.
