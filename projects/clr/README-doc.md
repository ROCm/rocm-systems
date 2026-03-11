# AMD CLR - Compute Language Runtimes

AMD CLR (Compute Language Runtime) contains source code for AMD's compute language runtimes: `HIP` and `OpenCL™`.

## Project Organisation

- [`hipamd`](./hipamd) — implementation of `HIP` on AMD platform.
- [`opencl`](./opencl) — implementation of [OpenCL™](https://www.khronos.org/opencl/) on AMD platform.
- [`rocclr`](./rocclr) — shared compute runtime used by both `HIP` and `OpenCL™`.

## How to build/install

### Prerequisites

Please refer to Quick Start Guide in [ROCm Docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

Building clr requires `rocm-hip-libraries` meta package, which provides the pre-requisites for clr.
If you need to build static clr library, `rocm-llvm-dev` package should be installed which has support for compilation of the static library.

### Linux

1. Clone this repository.
2. Create a build directory:
   ```bash
   cd clr && mkdir build && cd build
   ```
3. Configure for **HIP**:
   ```bash
   cmake .. -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_COMMON_DIR -DHIPCC_BIN_DIR=$HIPCC_BIN_DIR
   ```
   - `HIP_COMMON_DIR` — path to [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
   - `HIPCC_BIN_DIR` — path to the hipcc directory (e.g. `$ROCM_PATH/bin`).

   Or configure for **OpenCL™**:
   ```bash
   cmake .. -DCLR_BUILD_OCL=ON
   ```

   > **Tip:** Pass both `-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON` to build HIP and OpenCL™ together.

4. Build and install:
   ```bash
   make
   make install
   ```

To build as a **static library**, add `-DBUILD_SHARED_LIBS=OFF` and `-DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/llvm"`.

For detailed instructions, refer to [How to build HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html).

### Windows

1. Install the prerequisites:
   - **Visual Studio C++ 2022/2026** — [Download](https://visualstudio.microsoft.com/downloads/)
   - **Python 3.12** — `winget install -e --id Python.Python.3.12`
   - **CMake** — `winget install -e --id Kitware.CMake`
   - **DVC** — `winget install -e --id Iterative.DVC`
   - **TheRock package** — Download the tarball archive and extract to `c:/opt`.
     See [TheRock Releases](https://github.com/ROCm/TheRock/blob/main/RELEASES.md#installing-from-tarballs).

2. Set up environment variables:
   ```bash
   set HIP_COMMON_DIR=c:/github/rocm-systems/projects/hip
   set HIPCC_BIN_DIR=c:\opt\rocm\bin
   ```
   - `HIP_COMMON_DIR` — path to [HIP](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip).
   - `HIPCC_BIN_DIR` — path to the hipcc directory. If you have TheRock installed you can point it to `<Installation_of_therock>/rocm/bin`.

   For parallel builds, also set:
   ```bash
   set CMAKE_BUILD_PARALLEL_LEVEL=<num_parallel_builds>
   ```
   This controls the number of parallel compilations for a single project using MSVC.

3. Create a build directory:
   ```bash
   mkdir hipamd
   cd hipamd
   ```

4. Configure and build.

   **Public release build (ROCR + PAL static lib):**
   ```bash
   cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Release -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=%HIP_COMMON_DIR% -DHIPCC_BIN_DIR=%HIPCC_BIN_DIR% -DCMAKE_INSTALL_PREFIX=..\install -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/
   cmake --build . --config Release -j 6 --target install
   ```

   **Debug build (ROCR backend only):**
   ```bash
   cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Debug -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=%HIP_COMMON_DIR% -DHIPCC_BIN_DIR=%HIPCC_BIN_DIR% -DCMAKE_INSTALL_PREFIX=..\install -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=OFF -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/
   cmake --build . --config Debug -j 6 --target install
   ```

#### Private developer setup (HIP runtime + PAL)

5. Clone the following repositories into the **same parent folder** as `rocm-systems`:
   1. **bootstrap** — <https://github.com/AMD-Radeon-Driver/bootstrap>
      Run the bootstrap script and select **option 27: Header** (Mainline headers only for 3D UMDs → `AMD-Radeon-Driver/drivers`).
   2. **PAL** — <https://github.com/AMD-Radeon-Driver/pal>
      Make sure to init/sync submodules (`git submodule update --init --recursive`).
   3. **wkmi** — <https://github.com/AMD-ROCm-Internal/wkmi/>

   Expected folder layout:
   ```
   bootstrap/
   drivers/
   pal/
   rocm-systems/
   wkmi/
   ```

6. Configure and build (debug, ROCR + PAL backends, source-built PAL):
   ```bash
   cmake ../rocm-systems/projects/clr -DCMAKE_BUILD_TYPE=Debug -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=%HIP_COMMON_DIR% -DHIPCC_BIN_DIR=%HIPCC_BIN_DIR% -DCMAKE_INSTALL_PREFIX=..\install -D__HIP_ENABLE_PCH=OFF -DROCCLR_ENABLE_HSA=ON -DROCCLR_ENABLE_PAL=ON -D__HIP_ENABLE_RTC=ON -DUSE_PROF_API=OFF -DROCR_DLL_LOAD=OFF -DAMD_COMPUTE_WIN=../../../shared/amdgpu-windows-interop/ -DLIB_SRC_BUILD=ON
   cmake --build . --config Debug -j 6 --target install
   ```

## Tests

`hip-tests` is a separate project hosted at [hip-tests](../hip-tests). Refer to that repository for instructions on running the test suite.

## Release Notes

See the [CLR change log](./CHANGELOG.md) for a record of changes in each release.

## Disclaimer

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2023 Advanced Micro Devices, Inc. All Rights Reserved.

OpenCL™ is registered Trademark of Apple
