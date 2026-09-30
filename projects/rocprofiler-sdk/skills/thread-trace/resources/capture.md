# Capturing a Thread Trace

## Setup

- rocprofv3 from ROCm 7.0 or later, with thread trace support for your GPU. The
  [supported devices](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#supported-devices)
  table lists the validated architectures and what each supports (counters inside the
  trace, for example, only on some); treat hardware not listed there as unverified until a
  capture works.
- The rocprof-trace-decoder library. Check
  `ls ${ROCM_PATH:-/opt/rocm}/lib/librocprof-trace-decoder.so*`; if your ROCm does not ship
  it, build it from rocm-systems:

  ```bash
  git clone --depth 1 https://github.com/ROCm/rocm-systems.git
  cmake -S rocm-systems/projects/rocprof-trace-decoder -B decoder-build
  cmake --build decoder-build -j
  export ROCPROF_ATT_LIBRARY_PATH=$PWD/decoder-build/lib        # found by rocprofv3
  export ROCPROF_TRACE_DECODER_LIB=$PWD/decoder-build/lib/librocprof-trace-decoder.so
  export PYTHONPATH=$PWD/rocm-systems/projects/rocprof-trace-decoder/python
  ```

  rocprofv3 finds the library through `ROCPROF_ATT_LIBRARY_PATH` (a directory) or
  `--att-library-path`. The other two variables are for `att_mine.py`, which also needs
  Python 3.10 or later with `pyelftools` 0.31 or later.
- Line tables, so instructions map to source lines: build with `-gline-tables-only` (or
  `-g`). Kernels built without them, such as most library kernels, are still traced;
  results then name instruction addresses, which the disassembly of the captured code
  object resolves.

## Capture

```bash
rocprofv3 --att --kernel-include-regex '<kernel name regex>' -d capture -- <application> <args>
```

| Need | Option |
| --- | --- |
| A steady-state dispatch rather than the first | `--kernel-iteration-range N-N` (the Nth matching dispatch; the application must launch the kernel at least N times) |
| Several consecutive kernels in one trace | `--att-consecutive-kernels N` |
| A different compute unit (a WGP on RDNA; default 1) | `--att-target-cu N` |
| More shader engines (default only the first) | `--att-shader-engine-mask 0x...` |
| Fewer SIMDs, for a long kernel that loses data | `--att-simd-select 0x...` |
| A larger trace buffer | `--att-buffer-size N` |
| Only some GPUs | `--att-gpu-index N,...` |

Run `rocprofv3 --help` for the full list. By default rocprofv3 traces only the first
dispatch of each kernel that matches the regex.

**Unknown kernel names.** Capture without `--kernel-include-regex`: every traced dispatch
gets its own `stats_*_dispatch_<n>.csv`, whose first row names the kernel. C++ kernel
names are mangled; match a distinctive part of the name.

## Output

| Output | What it is |
| --- | --- |
| `stats_*_dispatch_<n>.csv` | Per-instruction summary of the traced waves, summed over waves; `att_mine.py` reads it with the per-wave records (see [reading-the-trace.md](reading-the-trace.md#commands)) |
| `ui_output_*_dispatch_<n>/` | Per-wave JSON for the ROCprof Compute Viewer |
| `*/*_shader_engine_<se>_<id>.att` | The raw trace, one file per shader engine |
| `*/*_code_object_id_<n>.out` | The code objects the trace refers to |
| `*_results.db` | rocprofv3's database for the run; its `rocpd_info_agent` table holds the GPU's properties (`cu_count`, `simd_per_cu`, `wave_front_size`, `max_waves_per_simd`, `lds_size_in_kb`, and more). With `--output-format csv` they are in `*_agent_info.csv` instead. |

The code objects are also the kernel's ISA: `llvm-objdump -d` the one whose symbols
include your kernel (`llvm-objdump -t <file> | grep <name>`; low ids are often runtime
kernels).

## Troubleshooting

`att_mine.py capture summary` reports the checks below.

| Symptom | Meaning and remedy |
| --- | --- |
| `Data Lost` | The trace buffer overflowed and the capture is invalid. Raise `--att-buffer-size`, trace fewer SIMDs, or use a smaller problem that keeps the per-wave work unchanged. |
| Zero waves | No work landed on the traced compute unit; it does not mean the kernel is fast. Launch more workgroups or trace another compute unit. |
| `Stitch Incomplete` or unresolved instructions | Some instructions could not be matched to the code object; later costs in those waves are not attributed. Check that the capture's code objects belong to the binary that ran. |
| Waves with a context switch | Not steady state; capture again. |
| LDS size 0 in a kernel that uses LDS | The compiler removed the allocation, for example when LDS is touched only by inline assembly. |

## Further reading

- rocprofv3 thread trace options, outputs, and troubleshooting:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>
- Decoder source and Python API: `projects/rocprof-trace-decoder` in
  <https://github.com/ROCm/rocm-systems>
