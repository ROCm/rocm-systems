# ROCprofiler-SDK:  Application Profiling, Tracing, and Performance Analysis

***
Note: rocprofiler-sdk is currently `not` supported as part of the public ROCm software stack and is only distributed as a beta
release to customers.
***

## Overview

ROCProfiler-SDK is AMD’s new and improved tooling infrastructure, providing a hardware-specific low-level performance analysis interface for profiling and tracing GPU compute applications. To see what's changed [Click Here](source/docs/about.md)

## GPU Metrics

- GPU hardware counters
- HIP API tracing
- HIP kernel tracing
- HSA API tracing
- HSA operation tracing
- Marker(ROCtx) tracing

## Tool Support

rocprofv3 is the command line tool built using the rocprofiler-sdk library and shipped with the ROCm stack. To see details on
the command line options of rocprofv3, please see rocprofv3 user guide
[Click Here](source/docs/rocprofv3.md)

## Documentation

We make use of doxygen to generate API documentation automatically. The generated document can be found in the following path:

``` bash
<ROCM_PATH>/share/html/rocprofiler-sdk
```

ROCM_PATH by default is /opt/rocm
It can be set by the user in different locations if needed.

## Build and Installation

```bash
git clone https://git@github.com:ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
cmake                                         \
      -B rocprofiler-sdk-build                \
      -D ROCPROFILER_BUILD_TESTS=ON           \
      -D ROCPROFILER_BUILD_SAMPLES=ON         \
      -D ROCPROFILER_BUILD_DOCS=ON            \
      -D CMAKE_INSTALL_PREFIX=/opt/rocm       \
       rocprofiler-sdk-source

cmake --build rocprofiler-sdk-build --target all --parallel 8
```

To install ROCprofiler, run:

```bash
cmake --build rocprofiler-sdk-build --target install
```

Please see the detailed section on build and installation here: [Click Here](/source/docs/installation.md)

## Support

Please report in the Github Issues.

## Limitations

- Individual XCC mode is not supported.
