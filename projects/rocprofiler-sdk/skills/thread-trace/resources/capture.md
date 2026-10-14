# Capturing a Thread Trace

## Setup

- rocprofv3 from ROCm 10.0 or later, with thread trace support for your GPU. The
  [supported devices](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html#supported-devices)
  table lists the validated architectures; treat hardware not listed there as unverified
  until a capture works.
- The rocprof-trace-decoder library and its Python package, which ship with ROCm 10.0 and
  later; rocprofv3 and `att_mine.py` find them in ROCm's `lib` directory without setup.
  Check with `ls ${ROCM_PATH:-/opt/rocm}/lib/librocprof-trace-decoder.so*`. If
  `import rocprof_trace_decoder` fails, put `${ROCM_PATH:-/opt/rocm}/lib/python3/site-packages`
  on `PYTHONPATH` (ROCm's environment setup does this). `att_mine.py` also needs Python 3.10
  or later, `pyelftools` 0.31 or later, and `llvm-objdump` (from ROCm or on `PATH`).
- A decoder outside ROCm's `lib` directory (one built from
  `projects/rocprof-trace-decoder` in rocm-systems, for example): always pass its directory
  to rocprofv3 with `--att-library-path <dir>`, and point `att_mine.py` to it with
  `ROCPROF_TRACE_DECODER_LIB=<dir>/librocprof-trace-decoder.so`.

### Source lines (optional)

Build the kernel as you run it, with its optimization flags (a debug build, `-O0`, runs
different code and timing). ATT works at the instruction level without debug information.
To map instructions to source lines, add `-g` or `-gline-tables-only` (which emits only
the line tables). Without either, instructions have no
source line: `lines` and `compare` group them all under `?`, so read `hotspots` and the
disassembly of the captured code object instead.

## Capture

Add these to the command in [SKILL.md](../SKILL.md#how-to-use-att):

| Need | Option |
| --- | --- |
| A steady-state dispatch rather than the first | `--kernel-iteration-range N` (the Nth dispatch of each matching kernel; the application must launch it at least N times), or `N-M` for dispatches N to M, whose waves the decoder-based reports pool (`stats` lists each dispatch's CSV rows separately) |
| Several consecutive kernels in one trace | `--att-consecutive-kernels N` |
| A different compute unit (a WGP on RDNA; default 1) | `--att-target-cu N` |
| More shader engines (default only the first) | `--att-shader-engine-mask 0x...` |
| Only some SIMDs | `--att-simd-select` (a SIMD bitmask on gfx9, where the other SIMDs' waves are still listed, without instructions; on RDNA, the one SIMD to trace) |
| A larger trace buffer, when rocprofv3 warns that the buffer is full | `--att-buffer-size N` |
| Only some GPUs | `--att-gpu-index N,...` (system GPU indices: `HIP_VISIBLE_DEVICES` does not renumber them) |

Run `rocprofv3 --help` for the full list.

**Unknown kernel names.** Capture without `--kernel-include-regex`: every traced dispatch
gets a `stats_*_dispatch_<n>.csv` whose rows name its kernel. The regex is searched in the
demangled name (for example `gemm_kernel(int, ...)`): a distinctive part is enough, with
regex characters such as parentheses escaped. Then capture the kernel you want into a new
directory. rocprofv3 does not clear the output directory and by default names each run's
files by its process ID, so a reused directory keeps the earlier capture's `.att` files
beside the new ones. `att_mine.py` decodes every file in a directory together, and stops with
`Code object id 1 is used by both ...` when two runs' code objects share an id (each run
numbers them from 1). `summary` reports how many dispatches the traced waves came from.

## Output

| Output | What it is |
| --- | --- |
| `stats_*_dispatch_<n>.csv` | Per-instruction summary of the traced waves, summed over waves; `att_mine.py stats` ranks it ([reading-the-trace.md](reading-the-trace.md#the-stats-csv)) |
| `ui_output_*_dispatch_<n>/` | The decoded trace as JSON, written for the ROCprof Compute Viewer and readable directly ([below](#the-ui_output-directory)) |
| `*_shader_engine_<se>_<n>.att` | The raw trace: one file per traced shader engine and dispatch on each GPU traced (consecutive kernels share a file) |
| `*_code_object_id_<n>.out` | Every code object loaded during the run, runtime kernels included |
| `*_results.db` | rocprofv3's database for the run, with the GPU's properties (compute units, SIMDs per CU, wave size, maximum waves per SIMD, LDS size); `summary` prints them. With `--output-format csv` they are in `*_agent_info.csv`. |

### The `ui_output` directory

| File | What it holds |
| --- | --- |
| `se*_sm*_sl*_wv*.json` | One traced wave, under its `wave` key: `instructions` (`[time, category, stall, duration, line]`, where `line` is the instruction's row in `code.json`), `timeline` (`[state, duration]`), and `waitcnt`: for executed waits, per counter, the earlier memory instructions each required to complete (`[wait line, [[line, 0], ...]]`). rocprofv3 works these out from the wave's instruction sequence, assuming each counter's memory instructions complete in order; scalar and flat memory instructions can complete out of order, so after one, waits on the counters it uses are listed only when they wait for zero, until one does. Only `s_load` and `s_store` count as scalar memory instructions, so in a kernel with others (such as `s_buffer_load`) the list can name the wrong instructions. Waits matched to no instruction are left out. Instructions without an address (code object 0, address 0) are skipped, and the list stops, with the warning `Analysis failed!`, at the first instruction whose address is not in the disassembly |
| `code.json` | The disassembly: one row per instruction with its source line and totals over the traced waves, plus a `; <kernel>` row before each kernel; `header` names the columns |
| `wstates<k>.json` | How many traced waves were in wave state `k` over time (`time`, `state`); `k` is the state's `WaveStateType` value (1 IDLE, 2 EXEC, 3 WAIT, 4 STALL). `att_mine.py` uses the instruction records instead ([stalls.md](stalls.md#wave-time)) |
| `occupancy.json` | Wave starts and ends on the traced shader engines (`occupancy_fields` names the fields), dispatches and other trace events per shader engine (`events`), and kernel names by `kernel_id` (`dispatches`) |
| `filenames.json` | The GPU generation, the requested counter names, and an index of the wave and other-SIMD files |
| `source_*`, `snapshots.json` | Copies of the source files the code objects refer to, when they exist on this machine |
| `realtime.json`, `other_simd_se*.json` | Shader clock and reference clock pairs, and (gfx11 and later) memory instructions (VMEM, FLAT, LDS) issued on the other SIMD |

The code objects are also the kernel's ISA: `llvm-objdump -d` the one whose symbols
include your kernel (`llvm-objdump -t <file> | grep <name>`).

## Troubleshooting

`att_mine.py capture summary` reports the decoder's warnings and the other checks below,
except the buffer-full warning, which rocprofv3 prints during the capture.

| Symptom | Meaning and remedy |
| --- | --- |
| `Data Lost` | The profiler dropped part of the trace because of bandwidth limits. The rest is still decoded, but the totals miss what was dropped. Trace fewer shader engines or a smaller problem. |
| `Thread trace buffer full!` (rocprofv3) | The trace buffer filled before the kernel ended. Raise `--att-buffer-size`, or trace a smaller problem that keeps the per-wave work unchanged. |
| `Wave incomplete` | The trace ended before some waves did; their lifetimes and totals are cut short. A full buffer can show only as this warning and fewer waves than expected. |
| `Stitch Incomplete` or unresolved instructions | Some trace tokens could not be matched to the disassembly, and instructions that could not be resolved are left out of the per-instruction totals. Check that the capture's code objects belong to the binary that ran. |
| Zero waves | No work landed on the traced compute unit (on RDNA, the traced SIMD; on gfx9 with `--att-simd-select`, the other SIMDs' waves are still counted, without instructions); it does not mean the kernel is fast. Launch more workgroups, or trace another compute unit or SIMD. |
| Waves with a context switch | The wave was switched out and back during the trace; treat its timing with care, or capture again. |

## Further reading

- rocprofv3 thread trace options, outputs, and troubleshooting:
  <https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/latest/how-to/using-thread-trace.html>
- Decoder source and Python API: `projects/rocprof-trace-decoder` in
  <https://github.com/ROCm/rocm-systems>
