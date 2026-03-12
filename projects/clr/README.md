# AMD CLR - Compute Language Runtimes

AMD CLR (Compute Language Runtime) is a component of the HIP runtime that contains the source code for AMD’s compute‑language runtimes `HIP` and `OpenCL™`.

## Disclaimer

he information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

©2025 Advanced Micro Devices, Inc. All Rights Reserved.

OpenCL™ is registered Trademark of Apple

## Project Organisation

- `hipamd` - contains implementation of `HIP` language on AMD platform. It is hosted at [hipamd](./hipamd)
- `opencl` - contains implementation of [OpenCL™](https://www.khronos.org/opencl/) on AMD platform. Now it is hosted at [opencl](./opencl)
- `rocclr` - contains compute runtime used in `HIP` and `OpenCL™`. This is hosted at [rocclr](./rocclr)

## How to build/install

### Prerequisites

Please refer to Quick Start Guide in [ROCm Docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

Building CLR requires the ROCm stack to be installed on the system. In addition, the following dependencies must be present as prerequisites:

```cpp
sudo apt install rocm-llvm-dev
pip3 install CppHeaderParser
```

For OpenCL, you need to install the ICD loader package provided by the distribution:

```cpp
sudo apt-get install ocl-icd-opencl-dev
```

### Build runtime from source

- Clone the repository

```cpp
git clone git@github.com:ROCm/rocm-systems.git
```

- Build HIP:

```cpp
export CLR_DIR="$(readlink -f rocm-systems/projects/clr)"
export HIP_DIR="$(readlink -f rocm-systems/projects/hip)"

cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DHIP_COMMON_DIR=$HIP_DIR -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCMAKE_INSTALL_PREFIX=$PWD/install -DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=OFF -DHIP_PLATFORM=amd ..
make -j$(nproc)
make install
```

- Build OpenCL™:

```cpp
cd "$CLR_DIR"
mkdir -p build; cd build
cmake -DCMAKE_PREFIX_PATH="/opt/rocm/" -DCLR_BUILD_HIP=OFF -DCLR_BUILD_OCL=ON ..
make -j$(nproc)
```

Note: Users can also build `OCL` and `HIP` at the same time by passing `-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON` to configure command.

For detail instructions, please refer to [how to build HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html)

## Tests

- Test HIP

`hip-tests` is a separate project hosted at [hip-tests](https://github.com/ROCm/rocm-systems/tree/develop/projects/hip-tests).

To run `hip-tests` please go to the repository and follow the steps.

- Test OpenCL™:

Users can use `ocltst` to test with, which can be generated via the command line:
```cpp
cmake -DCMAKE_PREFIX_PATH="$PWD/opencl/amdocl;/opt/rocm/" -DCLR_BUILD_HIP=OFF -DCLR_BUILD_OCL=ON cmake -DBUILD_TESTS=ON ..
```

The `ocltst` are build under the foler `$CLR_DIR/build/tests/ocltst`.

Additionally, OpenCL‑CTS can be used to validate OpenCL functionality. Refer to the [OpenCL-CTS](https://github.com/rocm/opencl-cts) repository for details.

## Release notes

HIP provides release notes in [CLR change log](./CHANGELOG.md), which has the records of changes in each release.