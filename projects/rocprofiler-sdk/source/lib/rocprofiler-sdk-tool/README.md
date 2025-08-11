# ROCProfiler Tool Library

This is a tool that gets registered with the
rocprofiler to obtain its services.
The tool is built as a shared library and is named as
rocprofiler-sdk-tool.
The library can be preloaded using LD_PRELOAD
to facilitate its registration as a tool
with the rocprofiler.

The user through rocprofv3 script can select the
options to obtain tracing and counter collection
services from the rocprofiler.

Currently, this tool supports kernel trace and the
hsa-api trace.
The tool uses the following environment variables
to read the user choices.

- `ROCPROF_KERNEL_TRACE=1` to obtain kernel trace
- `ROCPROF_HSA_API_TRACE=1` to obtain hsa api trace

The user can also specify the output filename and output file path
to which the traces are written to.

- `ROCPROF_OUTPUT_PATH=<directory>` to set the output directory path
- `ROCPROF_OUTPUT_FILE_NAME=<filename-without-extension>` to set the output file name

## CHANGELOG

The tool design is similar to its earlier versions.
However, not all features that the earlier versions supported are supported by
this tool.
