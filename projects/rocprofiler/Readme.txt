ROC Profiler library.
Profiling with metrics and traces based perfcounters (PMC) and SQ threadtraces (SQTT)
and AqlProfile HSA extension.

Library supports GFX8/GFX9 APIs.
The library source tree:
 - doc  - Documentation
 - inc/rocprofiler.h - Library public API
 - src  - Library sources
   - core - Library API sources
   - util - Library utils sources
   - xml - XML parser
 - test - Library test suite
   - ctrl - Test controll
   - util - Test utils
   - simple_convolution - Simple convolution test kernel

Build environment:

$ export CMAKE_PREFIX_PATH="<path to hsa-runtime includes>;<path to hsa-runtime library>"
$ export CMAKE_BUILD_TYPE=<debug|release> # release by default
$ export CMAKE_DEBUG_TRACE=1 # 1 to enable debug tracing

To build with the current installed ROCM:

$ cd .../rocprofiler
$ mkdir build
$ cd build

$ export CMAKE_PREFIX_PATH=/home/evgeny/git/compute/out/ubuntu-16.04/16.04
$ cmake ..
$ make

$ cmake -DCMAKE_PREFIX_PATH=/opt/rocm ..
$ make

To run the test:

$ cd .../rocprofiler/build
$ export LD_LIBRARY_PATH=$PWD
$ export HSA_TOOLS_LIB=librocprofiler64.so
$ run.sh

To enabled error messages logging to '/tmp/rocprofiler_log.txt':

$ export ROCPROFILER_LOG=1

To enable verbose tracing:

$ export ROCPROFILER_TRACE=1
