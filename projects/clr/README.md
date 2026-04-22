# AMD CLR - Compute Language Runtimes

AMD CLR (Compute Language Runtime) is a component of the HIP runtime that contains the source code for AMD’s compute‑language runtimes `HIP` and `OpenCL™`.

## Disclaimer

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard version changes, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.

AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.

THIS INFORMATION IS PROVIDED "AS IS." AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2025 Advanced Micro Devices, Inc. All Rights Reserved.

OpenCL™ is a registered trademark of Apple Inc.

## Project Organization

- `hipamd` - contains implementation of `HIP` language on AMD platform. It is hosted at [hipamd](./hipamd)
- `opencl` - contains implementation of [OpenCL™](https://www.khronos.org/opencl/) on AMD platform. Now it is hosted at [opencl](./opencl)
- `rocclr` - contains compute runtime used in `HIP` and `OpenCL™`. This is hosted at [rocclr](./rocclr)

## How to build/install

### Prerequisites

Please refer to Quick Start Guide in [ROCm Docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

Building CLR requires the ROCm stack to be installed on the system. In addition, the following dependencies must be present as prerequisites:

```bash
sudo apt install rocm-llvm-dev
pip3 install CppHeaderParser
```

For OpenCL, you need to install the ICD loader package provided by the distribution:

```bash
sudo apt-get install ocl-icd-opencl-dev
```

### Linux

1. Clone the repository:
   ```bash
   git clone git@github.com:ROCm/rocm-systems.git
   ```

2. Build **HIP**:
   ```bash
   export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
   export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"

   cd "$CLR_DIR"
   mkdir -p build; cd build
   cmake -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd ..
   make -j$(nproc)
   make install
   ```

3. Build **OpenCL™**:
   ```bash
   cd "$CLR_DIR"
   mkdir -p build; cd build
   cmake -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCLR_BUILD_HIP=OFF -DCLR_BUILD_OCL=ON ..
   make -j$(nproc)
   ```

   > **Note:** Build both `OCL` and `HIP` together by passing `-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON` to the configure command.

For detailed instructions, please refer to [How to build HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html).

### Windows

1. Install the prerequisites:
   - **Visual Studio C++ 2022/2026** — [Download](https://visualstudio.microsoft.com/downloads/)
   - **Python 3.12** — `winget install -e --id Python.Python.3.12`
   - **CMake** — `winget install -e --id Kitware.CMake`
   - **DVC** — `winget install -e --id Iterative.DVC`
   - **TheRock package** — Download the tarball archive and extract to `c:/opt`.
     See [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-from-tarballs).

2. Set up environment variables:
   ```cmd
   set HIP_COMMON_DIR=c:/github/rocm-systems/projects/hip
   set HIPCC_BIN_DIR=c:\opt\rocm\bin
   ```
   - `HIP_COMMON_DIR` — path to [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
   - `HIPCC_BIN_DIR` — path to the hipcc directory. If you have TheRock installed you can point it to `<Installation_of_therock>/rocm/bin`.

   For parallel builds, also set:
   ```cmd
   set CMAKE_BUILD_PARALLEL_LEVEL=<num_parallel_builds>
   ```
   This controls the number of parallel compilations for a single project using MSVC.

3. Create a build directory:
   ```cmd
   mkdir hipamd
   cd hipamd
   ```

4. Configure and build.

   **Public release build (ROCR + PAL static lib):**
   ```cmd
   cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=%HIP_COMMON_DIR% -DHIPCC_BIN_DIR=%HIPCC_BIN_DIR% -DCMAKE_INSTALL_PREFIX=..\install -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/
   cmake --build . --config Release -j 6 --target install
   ```

   **Release build (ROCR backend only):**
   ```cmd
   cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=%HIP_COMMON_DIR% -DHIPCC_BIN_DIR=%HIPCC_BIN_DIR% -DCMAKE_INSTALL_PREFIX=..\install -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/
   cmake --build . --config Release -j 6 --target install
   ```

## Tests

### HIP

`hip-tests` is a separate project hosted at [hip-tests](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip-tests).
Refer to that repository for build and run instructions.

### OpenCL™

Users can use `ocltst` to test with, which can be generated via the command line:
```bash
cmake -DCMAKE_PREFIX_PATH="$PWD/opencl/amdocl;/opt/rocm/" -DCLR_BUILD_HIP=OFF -DCLR_BUILD_OCL=ON -DBUILD_TESTS=ON ..
```

The `ocltst` binaries are built under the folder `$CLR_DIR/build/tests/ocltst`.

Additionally, [OpenCL-CTS](https://github.com/rocm/opencl-cts) can be used to validate OpenCL functionality.

## Release Notes

Release notes are published in the [CLR change log](./CHANGELOG.md).