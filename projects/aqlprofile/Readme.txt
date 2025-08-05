HSA extension AMD AQL profile library.
Provides AQL packets helper methods for perfcounters (PMC) and SQ threadtraces (SQTT).

Library supports GFX9 APIs.
The library source tree:
 - doc  - Documentation, the API specification and the presentation
 - <hsa-runtime>/inc/hsa_ven_amd_aqlprofile.h - AMD AQL profile library public API
 - src  - AMD AQL profile library sources
   - core - AQL API sources
   - pm4 - cmd/pmc/sqtt pm4 builders
   - def - Generated GFXIP definition headers
 - test - library test suite
   - ctrl - Test control
   - util - Test utils
   - simple_convolution - Simple convolution test kernel

Build environment:

$ export CMAKE_PREFIX_PATH=<path to hsa-runtime includes>:<path to hsa-runtime library>
$ export CMAKE_BUILD_TYPE=<debug|release> # release by default
$ export CMAKE_DEBUG_TRACE=1 # 1 to enable debug tracing

To build with the current installed ROCM:

$ export CMAKE_PREFIX_PATH=/opt/rocm/lib:/opt/rocm/include/hsa

$ cd .../aqlprofile
$ mkdir build
$ cd build
$ cmake ..
$ make

To regenerate src/def headers:

Need to use 'clang' compiler:
$ export CXX=/usr/bin/clang++
$ export CC=/usr/bin/clang

'mygen' make target to regenerate the headers from full set of gfxip headers:
$ make mygen

To reset the generated headers:
$ make mygenreset

To run the test:

$ cd ../aqlprofile/build
$ export LD_LIBRARY_PATH=$PWD
$ run.sh

To enabled error messages logging to '/tmp/aql_profile_log.txt':

$ export HSA_VEN_AMD_AQLPROFILE_LOG=1

To enable verbose tracing:

$ export AQLPROFILE_TRACE=1

To recompile kernel object:

$ /opt/rocm/opencl/bin/clang -cl-std=CL2.0 -include /opt/rocm/opencl/include/opencl-c.h -nogpulib -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/opencl.amdgcn.bc -Xclang -mlink-bitcode-file -Xclang /opt/rocm/amdgcn/bitcode/ockl.amdgcn.bc -target amdgcn-amd-amdhsa -mcpu=gfx906 vector_add_kernel.cl -o vector_add_kernel.so

With newer device-libs layout, use this recompile command:
$ /opt/rocm/opencl/bin/clang -cl-std=CL2.0 -include /opt/rocm/opencl/include/opencl-c.h --hip-device-lib-path=/opt/rocm/amdgcn/bitcode -target amdgcn-amd-amdhsa -mcpu=gfx906 vector_add_kernel.cl -o vector_add_kernel.so

### ROCm 5.7
Added support for GFX10/GFX11
