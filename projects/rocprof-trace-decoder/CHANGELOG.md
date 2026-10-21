# Rocprof Trace Decoder Changelog

## Unreleased version 0.2.3

### Added

- Added gfx117x WMMA instruction encodings (8- and 16-cycle variants), decoded as VALU instructions.

## Changelog for ROCm 10.2

### Added

- Python wrapper (`rocprof_trace_decoder`) for the C API.
- Python samples, including `samples/hidden_latency_hotspots.py`, which ranks instructions by non-hidden latency.
- `rocprof_trace_decoder.analysis` Python subpackage for post-processing decoded records. Its first module, `hidden_latency`, is used by the ATT Skill.
  - Reports hidden idle, stall, and issue cycles per PC and per (shader engine, SIMD), matching the hidden-latency analysis in ROCprof Compute Viewer.
  - Matrix instructions are classified from the ISA text and counted on both the MATRIX and VALU pipes.
  - The existing `rcv` module moved into the `analysis` subpackage.

### Changed

- The Python bindings now look for versioned library names (`librocprof-trace-decoder.so.X`, `.so.X.Y`, `.so.X.Y.Z`) and pick the highest installed version numerically, so `.so.0.10` is preferred over `.so.0.9`.
- The `llvm-objdump` lookup in the Python `codegen` module now also covers ROCm installed as Python wheels, in addition to `ROCM_HOME`/`ROCM_PATH`, `/opt/rocm`, and `PATH`.

### Resolved issues

- Fixed stitching failures with `flat_prefetch_*` instructions on gfx1250, where the wave reported the instruction as VMEM and the ISA reported it as FLAT.
- Improved stitching accuracy when the trace has lost data.
- `rocprof_trace_decoder_parse_data()` no longer lets C++ exceptions escape the C ABI on malformed shader data. It now returns `ROCPROFILER_THREAD_TRACE_DECODER_STATUS_ERROR_INVALID_SHADER_DATA`, like the handle-based `rocprof_trace_decoder_parse()` API.
- `REG_INIT` packets in gfx10 debug output now print the `me` field.
