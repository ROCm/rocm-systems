## What is this repository for? ###

This repository provides unit tests for  [HIP](../hip) implementation.

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

©2025 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches

The hip-tests repository maintains several branches. The branches that are of importance are:

* Main branch: This is the stable branch. It is up to date with the latest release branch, for example, if the latest release is rocm-5.4, main branch will be the repository based on this release.
* Develop branch: This is the default branch, on which the new features are still under development and visible. While this maybe of interest to many, it should be noted that this branch and the features under development might not be stable.
* Release branches. These are branches corresponding to each ROCM release, listed with release tags, such as rocm-5.4, etc.

## Release tagging

hip-tests releases are typically naming convention for each ROCM release to help differentiate them.

* rocm x.yy: These are the stable releases based on the ROCM release.
  This type of release is typically made once a month.

## Build HIP Catch tests

For building HIP from source, please check instructions on the [HIP page](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html).  

HIP catch tests can be built via the following instructions:

1 .Clone the hip-tests source code from the repository, with definition of branch. The default branch is `develop`, as an example,
```bash
$ git clone -b develop https://github.com/ROCm/rocm-systems.git
$ cd rocm-systems/projects/hip-tests
$ export HIP_TESTS_DIR="$(readlink -f hip-tests)"
```
`hip-tests` for AMD platform now rely on `amdclang` to build, which is shipped with ROCm installation.
Although individual tests will compile with `hipcc`, ideally you should use `amdclang++`.

2. Build the catch tests
```bash
$ cd "$HIP_TESTS_DIR"
$ mkdir -p build; cd build
$ cmake ../catch -DHIP_PLATFORM=amd -DCMAKE_PREFIX_PATH=<HIP-Installed-Path> -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_COMPILER=amdclang++ -DCMAKE_C_COMPILER=amdclang -DCMAKE_HIP_COMPILER=amdclang++ -DOFFLOAD_ARCH_STR="--offload-arch=gfxXXXX"
$ make -j32 build_tests
$ ctest # run tests
```

HIP catch tests are built under the folder `$HIP_TESTS_DIR/build`.

### Build HIP Catch2 standalone test

`standalone_main.cc` is now removed, and we use the main shipped with Catch2 to build standalone tests.
The test suite has moved to Catch2v3 (v3.8.1 to be exact).
Moving from v2 to v3 came with some fundamental changes in how Catch2 interacts with hip-tests.
Starting with, it is no longer a single header, it is now a library, which needs to be linked to the test exe.

If you are on your personal machine, it is highly recommended to install Catch2v3 (v3.8.1) locally on your system.
This helps skip the download/build part in hip-tests and results in faster builds overall.
HIP Catch2 supports building standalone tests, for example,

### Steps to install Catch2v3 locally

- `git clone https://github.com/catchorg/Catch2.git -b v3.8.1 --depth 1`
- `cd Catch2`
- `mkdir build && cd build`
- `cmake .. -DCMAKE_BUILD_TYPE=Release`
- `make -j8`

Install step, you might need to have superuser permission to be able to install globally.
The default install location for linux is `/usr/local/`.

- `make install`

With Catch2 installed globally we can build individual Catch2 test like this:

```bash
export ROCM_SYSTEMS_DIR=<path_to_rocm_systems>
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/external/picojson                    \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  --offload-arch=native                                                              \
  -L /usr/local/lib -lCatch2 -lCatch2Main                                            \
  -x hip <path_to_test>
```
The command above builds hip without the `main` from hip-tests, if the purpose is just to run the test with as simple commands as possible, go with this.

If you want to use `main` provided by hip-tests, use the following command:

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

### Steps to use FetchContent to get Catch2

Build Catch2 with hip-tests

```bash
export HIP_PATH=<path_where_hip_is_installed>
cd rocm-systems
export ROCM_SYSTEMS_DIR=$PWD
mkdir tests
cd tests
cmake ../projects/hip-tests/catch -DHIP_PLATFORM=amd -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=$HIP_PATH
make -j$(nproc) Catch2
export DEPS_PATH=$PWD/_deps
```

We now use the Catch2 we just build to link to the tests.

For standalone tests:

```bash
amdclang++                                                                           \
  -I $ROCM_SYSTEMS_DIR/projects/hip-tests/catch/include                              \
  -I $DEPS_PATH/catch2-src/src                                                       \
  -I $DEPS_PATH/catch2-build/generated-includes                                      \
  --offload-arch=native                                                              \
  -L $DEPS_PATH/catch2-build/src                                                     \
  $DEPS_PATH/catch2-src/src/catch2/internal/catch_main.cpp                           \
  -lCatch2                                                                           \
  -x hip <test_file.cc>
```

To use our main:

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
  -x hip <test_file.cc>
```

You might need to set `LD_LIBRARY_PATH` to be Catch2 location.

```bash
$ hipcc $HIP_TESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc -I ./catch/include ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 -o hipPointerGetAttributes
$ ./hipPointerGetAttributes
```

### Building with address sanitizer

To build catch tests with Address Sanitizer options, use the cmake option `-DENABLE_ADDRESS_SANITIZER=ON`.
