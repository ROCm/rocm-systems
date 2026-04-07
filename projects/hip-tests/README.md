# What is the hip-tests repository for?

This repository provides unit tests for the [HIP API](../hip) implementation.

## DISCLAIMER

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

©2026 Advanced Micro Devices, Inc. All Rights Reserved.

## Repository branches

The `rocm-systems/projects/hip-tests` repository maintains several branches. The branches that are of importance are:

* Main branch: This is the stable branch. It is up to date with the latest release branch. For example, if the latest release is rocm-7.2, main branch will be the repository based on this release.
* Develop branch: This is the default branch, on which the new features are still under development and visible. While this may be of interest to many, it should be noted that this branch and the features under development might not be stable.
* Release branches: These are branches corresponding to each ROCM release, listed with release tags, such as rocm-7.2, etc.

## Building HIP Catch tests

To build HIP from source, review instructions on the [HIP page](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html).  

`rocm-systems/projects/hip-tests` can be built using the following instructions. To start you must setup the environment needed to build the HIP tests by setting the ``ROCM_PATH`` environent variable to point to the current installation of ROCm:

```bash
  export ROCM_PATH=/opt/rocm # or the appropriate path for your installation
  echo $ROCM_PATH
```

1. Clone the `hip-tests` project as part of the `rocm-systems` repository, specifying the branch of interest. The default branch is `develop`, as an example:

```bash
  git clone -b release/rocm-rel-7.2 https://github.com/ROCm/rocm-systems.git
```

Alternatively, you can clone the hip-tests package separately using sparse-checkout as described in the [Contributing to](https://github.com/ROCm/rocm-systems/blob/develop/CONTRIBUTING.md) documentation. 

```bash
  git clone --no-checkout --filter=blob:none https://github.com/ROCm/rocm-systems.git
  cd rocm-systems
  git sparse-checkout init --cone
  git sparse-checkout set projects/hip-tests
  git checkout release/rocm-rel-7.2 # or the specific branch of interest
```

2. Set the `HIPTESTS_DIR` environment variable by running the following from outside the `hip-tests` folder: 

```bash
  export HIPTESTS_DIR="$(readlink -f hip-tests)"
  echo $HIPTESTS_DIR
```

### Build all HIP catch tests

```bash
  cd "$HIPTESTS_DIR"
  mkdir -p build; cd build
  cmake ../catch/ -DHIP_PLATFORM=amd
  make -j$(nproc) build_tests
  ctest # run tests
```

HIP catch source files are found in `$HIPTESTS_DIR/catch`. Catch tests are built under the folder `$HIPTESTS_DIR/build/catch_tests`.

Note
---

To build catch tests with [Address Sanitizer](https://rocm.docs.amd.com/projects/llvm-project/en/latest/conceptual/using-gpu-sanitizer.html) options, use the cmake option `-DENABLE_ADDRESS_SANITIZER=ON`.


### Build HIP standalone catch tests

HIP Catch2 supports compiling standalone tests using ``amdclang++`` for example:

```bash
  amdclang++ -D__HIP_PLATFORM_AMD__ -x hip ./catch/unit/memory/hipPointerGetAttributes.cc \
  -I ./catch/include ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 \
  -I $HIP_PATH/include -L$HIP_PATH/lib -lamdhip64 -o hipPointerGetAttributes
```

Or using ``hipcc``:

```bash
  hipcc $HIP_TESTS_DIR/catch/unit/memory/hipPointerGetAttributes.cc -I ./catch/include \
  ./catch/hipTestMain/standalone_main.cc -I ./catch/external/Catch2 -o hipPointerGetAttributes
```

And then run the test:

```bash
  ./hipPointerGetAttributes
```

### Build Samples

The CMakeLists.txt file in the hip-tests/samples folder can be used to build and package samples.

CMakeLists.txt can support shared and static libs of hip-rocclr runtime. The same steps can be followed for both.

To build a specific sample (e.g. `0_Intro/bit_extract`) run:
```bash
$ cd samples/0_Intro/bit_extract
$ mkdir -p build && cd build
$ cmake ..
$ make all
```

To build all samples together, run:
```bash
$ cd hip-tests
$ mkdir -p build && cd build
$ rm -rf * # (to clear up)
$ cmake ../samples
$ make build_samples
```

In order to build specific samples (Intro, Utils or Cookbook) run:
```bash
$ make build_intro
$ make build_utils
$ make build_cookbook
```

Note that if you want debug version, add `-DCMAKE_BUILD_TYPE=Debug` in cmake cmd.

3. To package samples and generate packages. From hip-tests/build:
```bash
$ cmake ../samples
$ make package_samples
```

Note
---

Sample `2_Cookbook/22_cmake_hip_lang` is not included in top-level `cmake`. To build this sample from the top-level, uncomment Line 43 inside `samples/2_Cookbook/CMakeLists.txt`.
