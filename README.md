# ROCR Runtime

This ROCm Runtime (ROCr) repo combines 2 previously separate repos into a single repo:
- The HSA Runtime (`hsa-runtime`) for AMD GPU application development and
- The ROCt Thunk Library (`libhsakmt`), a "thunk" interface to the ROCm kernel driver (ROCk), used by the runtime.

## Infrastructure

The HSA runtime is a thin, user-mode API that exposes the necessary interfaces to access and interact with graphics hardware driven by the AMDGPU driver set and the ROCK kernel driver. Together they enable programmers to directly harness the power of AMD discrete graphics devices by allowing host applications to launch compute kernels directly to the graphics hardware.

The capabilities expressed by the HSA Runtime API are:

* Error handling
* Runtime initialization and shutdown
* System and agent information
* Signals and synchronization
* Architected dispatch
* Memory management
* HSA runtime fits into a typical software architecture stack.

The HSA runtime provides direct access to the graphics hardware to give the programmer more control of the execution. An example of low level hardware access is the support of one or more user mode queues provides programmers with a low-latency kernel dispatch interface, allowing them to develop customized dispatch algorithms specific to their application.

The HSA Architected Queuing Language is an open standard, defined by the HSA Foundation, specifying the packet syntax used to control supported AMD/ATI Radeon (c) graphics devices. The AQL language supports several packet types, including packets that can command the hardware to automatically resolve inter-packet dependencies (barrier AND & barrier OR packet), kernel dispatch packets and agent dispatch packets.

In addition to user mode queues and AQL, the HSA runtime exposes various virtual address ranges that can be accessed by one or more of the system's graphics devices, and possibly the host. The exposed virtual address ranges either support a fine grained or a coarse grained access. Updates to memory in a fine grained region are immediately visible to all devices that can access it, but only one device can have access to a coarse grained allocation at a time. Ownership of a coarse grained region can be changed using the HSA runtime memory APIs, but this transfer of ownership must be explicitly done by the host application.

Programmers should consult the HSA Runtime Programmer's Reference Manual for a full description of the HSA Runtime APIs, AQL and the HSA memory policy.

## Known issues

* Each HSA process creates an internal DMA queue, but there is a system-wide limit of four DMA queues. When the limit is reached HSA processes will use internal kernels for copies.

## Artifacts produced by the build

- **libhsakmt (ROCt)** - User-mode API interfaces for interacting with the ROCk driver
- **Runtime (ROCr)** - Core runtime supporting HSA standards
- **rocrtst** - Runtime test suites for HSA implementation validation and performance testing
- **kfdtest** - Validation tests for ROCt

## Building the ROCR Runtime

### Target platform requirements
Please see the [ROCm System requirements (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html).

Ensure you have the following installed:

- CMake 3.7 or higher
- `libelf-dev`
- `g++`
- `libdrm-amdgpu-dev` or `libdrm-dev`
- `pkg-config`
- `rocm-core`
- `rocm-llvm-dev`

### ROCr & ROCt Build Instructions
1. **Clone this repository and cd into its root**
2. **Prepare the build directory**
    ```sh
    mkdir build && cd build
    ```
3. **Configure the build (example)**
    ```sh
    cmake -DCMAKE_INSTALL_PREFIX=<rocm install dir> ..
    ```
    e.g: 
    ```
    cmake -DCMAKE_INSTALL_PREFIX=/opt/rocm ..
    ```
4. **Compile the project**
    ```sh
    make
    ```
5. **Install the runtime**
    ```sh
    make install
    ```
6. **(Optional) Build packages**
    ```sh
    make package
    ```
#### Non-default CMake Build Options
- *Produce a release build instead of debug*
    ```sh
    -DCMAKE_BUILD_TYPE=Release
    ```

- *Control whether libhsakmt and libhsa-runtime are shared or static*
    libhsakmt is always built as a static library that gets linked into libhsa-runtime, so there is a single library generated called libhsa-runtime64{.so/.a}. If `BUILD_SHARED_LIBS` is not set, this is a shared library by default. Setting `BUILD_SHARED_LIBS` to `OFF` will make it static.
    ```sh
    -DBUILD_SHARED_LIBS=ON   # or OFF for static lib
    ```
### Building the tests
#### rocrtst
1. **Go to rocrtst root**
   ```sh
   cd <rocr-runtime>/rocrtst/suites/test_common
   ```
2. **Prepare the build directory**
    ```sh
    mkdir build && cd build
    ```
3. **Configure the build**
   Example configuration:
    ```sh
    cmake \
        -DCMAKE_PREFIX_PATH="<rocm install root>;<llvm install root>" \
        -DROCM_DIR="$ROCM_INSTALL_PATH" \
        -DOPENCL_DIR="<rocm install root>" \
        ..
    ```
4. **Compile the project**
    ```sh
    make
    make rocrtst_kernels
    ```
5. ** Run the tests
    Make sure libhsa-runtime.so is in the library path; e.g.,
    ```sh
    $ LD_LIBRARY_PATH=<rocm install root> ./rocrtst -h    # See help options
    ```
#### kfdtest
1. **Go to kfdtest root**
   ```sh
   cd <rocr-runtime>/libhsakmt/tests/kfdtest
   ```
2. **Prepare the build directory**
    ```sh
    mkdir build && cd build
    ```
3. **Configure the build**
   Example configuration:
    ```sh
    cmake \
        -DCMAKE_PREFIX_PATH="<rocm install root>" \
        -DROCM_DIR="$ROCM_INSTALL_PATH" \
        ..
    ```
4. **Compile the project**
    ```sh
    make
    ```
## Using the ROCR Runtime

After installation, you can link against the runtime by using the provided CMake package configurations. For example, to use the ROCR runtime in your project:

```cmake
find_package(hsa-runtime64 1.0 REQUIRED)
add_executable(MyApp main.cpp)
target_link_libraries(MyApp PRIVATE hsa-runtime64::hsa-runtime64)
```
## Disclaimer
The information contained herein is for informational purposes only, and is
subject to change without notice. While every precaution has been taken in the
preparation of this document, it may contain technical inaccuracies, omissions
and typographical errors, and AMD is under no obligation to update or otherwise
correct this information. Advanced Micro Devices, Inc. makes no representations
or warranties with respect to the accuracy or completeness of the contents of
this document, and assumes no liability of any kind, including the implied
warranties of noninfringement, merchantability or fitness for particular
purposes, with respect to the operation or use of AMD hardware, software or
other products described herein. No license, including implied or arising by
estoppel, to any intellectual property rights is granted by this document.
Terms and limitations applicable to the purchase or use of AMD's products are
as set forth in a signed agreement between the parties or in AMD's Standard
Terms and Conditions of Sale.

AMD, the AMD Arrow logo, and combinations thereof are trademarks of Advanced
Micro Devices, Inc. Other product names used in this publication are for
identification purposes only and may be trademarks of their respective
companies.

Copyright © 2014-2024 Advanced Micro Devices, Inc. All rights reserved.
