# ROCprofiler-SDK:  Application Profiling, Tracing, and Performance Analysis

> [!IMPORTANT]
We are phasing out development and support for `ROCTracer, ROCprofiler, rocprof, and rocprofv2` in favour of `ROCprofiler-SDK` and `rocprofv3` in upcoming ROCm releases. Starting with the `ROCm 6.4` release, only critical defect fixes will be addressed for older versions of the profiling tools and libraries. We encourage all users to upgrade to the latest version of the `ROCprofiler-SDK` library and the `rocprofv3` tool to ensure continued support and access to new features.

    Please note that we anticipate the end of life for ROCprofiler V1/V2 and ROCTracer within nine months after the ROCm 7.0 release, aligning with the Q1 2026.

## Overview

ROCprofiler-SDK is AMD’s new and improved tooling infrastructure, providing a hardware-specific low-level performance analysis interface for profiling and tracing GPU compute applications. To see what's changed, [Click Here](source/docs/conceptual/comparing-with-legacy-tools.rst)

> [!NOTE]
> The published documentation is available at [ROCprofiler-SDK documentation](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/index.html) in an organized, easy-to-read format, with search and a table of contents. The documentation source files reside in the `rocprofiler-sdk/source/docs` folder of this repository. As with all ROCm projects, the documentation is open source. For more information on contributing to the documentation, see [Contribute to ROCm documentation](https://rocm.docs.amd.com/en/latest/contribute/contributing.html).

## GPU Metrics

- GPU hardware counters
- Dispatch Counter Collection
- Device Counter Collection
- PC Sampling (Host Trap)
- Thread trace and ROCprof trace decoder (SQTT, ATT).

## API Trace Support

- HIP API tracing
- HSA API tracing
- Marker (ROCTx) tracing
- Memory copy tracing
- Memory allocation tracing
- Page Migration Event tracing
- Scratch Memory tracing
- RCCL API tracing
- rocDecode API tracing
- rocJPEG API tracing

## Parallelism API Support

- HIP
- HSA
- MPI
- Kokkos-Tools (KokkosP)
- OpenMP-Tools (OMPT)

## Tool Support

rocprofv3 is the command line tool built using the rocprofiler-sdk library and shipped with the ROCm stack. To see details on
the command line options of rocprofv3, please see rocprofv3 user guide
[Click Here](source/docs/how-to/using-rocprofv3.rst)

## Documentation

We make use of doxygen to generate API documentation automatically. The generated document can be found in the following path:

``` bash
<ROCM_PATH>/share/html/rocprofiler-sdk
```

ROCM_PATH by default is /opt/rocm
It can be set by the user in different locations if needed.

## Build and Installation

```bash
git clone https://github.com/ROCm/rocprofiler-sdk.git rocprofiler-sdk-source
cmake                                         \
      -B rocprofiler-sdk-build                \
      -DCMAKE_INSTALL_PREFIX=/opt/rocm        \
      -DCMAKE_PREFIX_PATH=/opt/rocm           \
       rocprofiler-sdk-source

cmake --build rocprofiler-sdk-build --target all --parallel $(nproc)
```

To install ROCprofiler, run:

```cmake --build rocprofiler-sdk-build --target install```

Please see the detailed section on build and installation here: [Click Here](source/docs/install/installation.rst)

## Support

Please report issues on GitHub OR send an email to <dl.ROCm-Profiler.support@amd.com>

## Limitations

- Individual XCC mode is not supported.

- By default, PC sampling API is disabled. To use PC sampling. Setting the `ROCPROFILER_PC_SAMPLING_BETA_ENABLED` environment variable grants access to the PC Sampling experimental beta feature. This feature is still under development and may not be completely stable.
  - **Risk Acknowledgment**: By activating this environment variable, you acknowledge and accept the following potential risks:
    - **Hardware Freeze**: This beta feature could cause your hardware to freeze unexpectedly.
    - **Need for Cold Restart**: In the event of a hardware freeze, you may need to perform a cold restart (turning the hardware off and on) to restore normal operations.
    Please use this beta feature cautiously. It may affect your system's stability and performance. Proceed at your own risk.

  - At this point, we do not recommend stress-testing the beta implementation.

  - Correlation IDs provided by the PC sampling service are verified only for HIP API calls.

  - Timestamps in PC sampling records might not be 100% accurate.

  - For low PC-sampling frequencies with intervals < 65k cycles, a lot of error samples might be delivered. We're working on optimizing this to allow lower sampling frequencies.

- gfx10, gfx11 and gfx12 requires a stable power state for counter collection. This includes Radeon 7000 GPUs.
  ```bash
  # For device <N>. Use 'rocm-smi' or 'amd-smi monitor' to see device number.
  sudo amd-smi set -g <N> -l stable_std
  # After profiling, set power state back to 'auto'
  sudo amd-smi set -g <N> -l auto
  ```

  The gfx version can be found via `amd-smi static --asic -g <N>` in the `TARGET_GRAPHICS_VERSION` field:

  ```bash
  $ amd-smi static -a -g 2
  GPU: 2
      ASIC:
          MARKET_NAME: Navi 33 [Radeon Pro W7500]
          VENDOR_ID: 0x1002
          VENDOR_NAME: Advanced Micro Devices Inc. [AMD/ATI]
          SUBVENDOR_ID: 0x1002
          DEVICE_ID: 0x7489
          SUBSYSTEM_ID: 0x0e0d
          REV_ID: 0x00
          ASIC_SERIAL: N/A
          OAM_ID: N/A
          NUM_COMPUTE_UNITS: 28
          TARGET_GRAPHICS_VERSION: gfx1102
  ```




> [!WARNING]
> The latest mainline version of AQLprofile can be found at [https://repo.radeon.com/rocm/misc/aqlprofile/](https://repo.radeon.com/rocm/misc/aqlprofile/). However, it's important to note that updates to the public AQLProfile may not occur as frequently as updates to the rocprofiler-sdk. This discrepancy could lead to a potential mismatch between the AQLprofile binary and the rocprofiler-sdk source.
