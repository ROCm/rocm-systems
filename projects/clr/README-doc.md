# AMD CLR - Compute Language Runtimes

AMD CLR (Compute Language Runtime) contains source codes for AMD's compute languages runtimes: `HIP` and `OpenCL™`.

## Project Organisation

- `hipamd` - contains implementation of `HIP` language on AMD platform. It is hosted at [ROCm/clr/hipamd](https://github.com/ROCm/clr/tree/develop/hipamd)
- `opencl` - contains implementation of [OpenCL™](https://www.khronos.org/opencl/) on AMD platform. Now it is hosted at [ROCm/clr/opencl](https://github.com/ROCm/clr/tree/develop/opencl)
- `rocclr` - contains compute runtime used in `HIP` and `OpenCL™`. This is hosted at [ROCm/clr/rocclr](https://github.com/ROCm/clr/tree/develop/rocclr)

## How to build/install

### Prerequisites

Please refer to Quick Start Guide in [ROCm Docs](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html).

Building clr requires `rocm-hip-libraries` meta package, which provides the pre-requisites for clr.
If you need to build static clr library, `rocm-llvm-dev` package should be installed which has support for compilation of the static library.

### Linux

- Clone this repository

- `cd clr && mkdir build && cd build`

- For HIP:

    - `cmake .. -DCLR_BUILD_HIP=ON -DHIP_COMMON_DIR=$HIP_COMMON_DIR -DHIPCC_BIN_DIR=$HIPCC_BIN_DIR`

        - `HIP_COMMON_DIR` points to [HIP](https://github.com/ROCm/HIP)

        - `HIPCC_BIN_DIR` points to hipcc directory, if you have ROCm installed you can point it to `$ROCM_PATH/bin`

- For OpenCL™:

    - `cmake .. -DCLR_BUILD_OCL=ON`

- Build and install

```
make
make install
```

Users can also build `OCL` and `HIP` at the same time by passing `-DCLR_BUILD_HIP=ON -DCLR_BUILD_OCL=ON` to configure command.

HIP/CLR can be built as a static library, users need to add more options in `cmake` command,
 `-DBUILD_SHARED_LIBS=OFF` and `-DCMAKE_PREFIX_PATH="/opt/rocm/;/opt/rocm/llvm`.

For detail instructions, please refer to [how to build HIP](https://rocm.docs.amd.com/projects/HIP/en/latest/install/build.html)

## Tests

`hip-tests` is a separate repository hosted at [hip-tests](https://github.com/ROCm/hip-tests).

To run `hip-tests` please go to the repository and follow the steps.

## Release notes

HIP provides release notes in [CLR change log](./CHANGELOG.md), which has the records of changes in each release.

## Disclaimer

The information presented in this document is for informational purposes only and may contain technical inaccuracies, omissions, and typographical errors. The information contained herein is subject to change and may be rendered inaccurate for many reasons, including but not limited to product and roadmap changes, component and motherboard versionchanges, new model and/or product releases, product differences between differing manufacturers, software changes, BIOS flashes, firmware upgrades, or the like. Any computer system has risks of security vulnerabilities that cannot be completely prevented or mitigated.AMD assumes no obligation to update or otherwise correct or revise this information. However, AMD reserves the right to revise this information and to make changes from time to time to the content hereof without obligation of AMD to notify any person of such revisions or changes.THIS INFORMATION IS PROVIDED ‘AS IS.” AMD MAKES NO REPRESENTATIONS OR WARRANTIES WITH RESPECT TO THE CONTENTS HEREOF AND ASSUMES NO RESPONSIBILITY FOR ANY INACCURACIES, ERRORS, OR OMISSIONS THAT MAY APPEAR IN THIS INFORMATION. AMD SPECIFICALLY DISCLAIMS ANY IMPLIED WARRANTIES OF NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR ANY PARTICULAR PURPOSE. IN NO EVENT WILL AMD BE LIABLE TO ANY PERSON FOR ANY RELIANCE, DIRECT, INDIRECT, SPECIAL, OR OTHER CONSEQUENTIAL DAMAGES ARISING FROM THE USE OF ANY INFORMATION CONTAINED HEREIN, EVEN IF AMD IS EXPRESSLY ADVISED OF THE POSSIBILITY OF SUCH DAMAGES. AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced Micro Devices, Inc. Other product names used in this publication are for identification purposes only and may be trademarks of their respective companies.

© 2023 Advanced Micro Devices, Inc. All Rights Reserved.

OpenCL™ is registered Trademark of Apple
