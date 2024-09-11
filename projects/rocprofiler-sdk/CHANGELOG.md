# Changelog for ROCprofiler-SDK

Full documentation for ROCprofiler-SDK is available at [Click Here](source/docs/index.md)

## ROCprofiler-SDK for AFAR I

### Additions

- HSA API Tracing
- Kernel Dispatch Tracing
- Kernel Dispatch Counter Collection
  - Instances are reported as single dimensions
  - No serialization

## ROCprofiler-SDK for AFAR II

### Additions

- HIP API Tracing
- ROCTx Tracing
- Tracing ROCProf Tool V3
- Packaging Documentation
- ROCTx start/stop
- Memory Copy Tracing

## ROCprofiler-SDK for AFAR III

### Additions

- Kernel Dispatch Counter Collection – (includes serialization and multidimensional instances)
- Kernel serialization
- Serialization on/off handling
- ROCprof Tool Plugin Interface V3 for Counters and Dimensions
- List metrics support
- Correlation-id retirement
- HIP and HSA trace distinction
  - --hip-runtime-trace          For Collecting HIP Runtime API Traces
  - --hip-compiler-trace         For Collecting HIP Compiler generated code Traces
  - --hsa-core-trace                For Collecting HSA API Traces (core API)
  - --hsa-amd-trace                For Collecting HSA API Traces (AMD-extension API)
  - --hsa-image-trace             For Collecting HSA API Traces (Image-extension API)
  - --hsa-finalizer-trace          For Collecting HSA API Traces (Finalizer-extension API)

## ROCprofiler-SDK for AFAR IV

### Additions

- Page Migration Reporting (API)
- Scratch Memory Reporting (API)
- Kernel Dispatch Callback Tracing (API)
- External Correlation ID Request Service (API)
- Buffered counter collection record headers (API)
- Remove HSA dependency from counter collection (API)
- rocprofv3 Multi-GPU support in single-process (tool)

## ROCprofiler-SDK for AFAR V

### Additions

- Agent/Device Counter Collection (API)
- Single JSON output format support (tool)
- Perfetto output format support(.pftrace) (tool)
- Input YAML support for counter collection (tool)
- Input JSON support for counter collection (tool)
- Application Replay (Counter collection)
- PC Sampling (Beta)(API)
- ROCProf V3 Multi-GPU Support:
  - Multi-process (multiple files)

### Fixes

- SQ_ACCUM_PREV and SQ_ACCUM_PREV_HIRE overwriting issue

### Changes

- rocprofv3 tool now needs `--` in front of application. For detailed uses, please [Click Here](source/docs/rocprofv3.md)

## ROCprofiler-SDK for AFAR VI

### Additions

- OTF2 Tool Support
- Kernel and Range Filtering
- Counter Collection Definitions in YAML
- Documentation updates (SQ Block, Counter Collection, Tracing, Tool Usage)
- Added rocprofv3 option --kernel-rename
- Added rocprofv3 options for perfetto settings (buffer size, etc.)
- Added CSV columns for kernel trace
  - Thread_Id
  - Dispatch_Id
- Added CSV column for counter_collection

### Fixes

- Miscellaneous bug fixes

## ROCprofiler-SDK for AFAR VII

### Additions

### Changes

- Support `--marker-trace` on application linked against old (roctracer) ROCTx (i.e. `libroctx64.so`)

### Fixes

- Creation of subdirection when rocprofv3 `--output-file` contains a folder path
- Fix misaligned stores (undefined behavior) for buffer records
