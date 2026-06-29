# Python Modules

This document summarizes the Python modules added for the rocprof-trace-decoder
API and ATT tool. The package is intentionally small: the reusable API lives in
`rocprof_trace_decoder`, while the top-level scripts are thin command-line
entry points.

The Python API wraps the handle-based decoder functions from
`rocprof_trace_decoder.h` and the record types from `trace_decoder_types.h`.
The quick-scan and standalone trace extraction APIs are intentionally not
wrapped here.

## Module Overview

| Module | Role | API Status |
| --- | --- | --- |
| `rocprof_trace_decoder` | Public package entry point | Public |
| `rocprof_trace_decoder.records` | Python record/enumeration model for decoder output | Public |
| `rocprof_trace_decoder.bindings` | `ctypes` wrapper around `librocprof-trace-decoder` | Public for `Decoder`, internal for C mirrors |
| `rocprof_trace_decoder.code_index` | ISA lookup and instruction-stat accumulation | Public utility |
| `rocprof_trace_decoder.rcv` | ROCprof Compute Viewer JSON writer | Internal utility |
| `rocprof_trace_decoder.att` | High-level ATT decode orchestration with explicit trace metadata | Public utility |
| `generate_code.py` | Generates `code.json`, snapshots, and source copies from code objects | Tool module |
| `att_tool.py` | CLI script for conventional rocprofiler ATT file names | Tool script |

## `rocprof_trace_decoder`

`rocprof_trace_decoder` is the import root for users who want the Python API.
It re-exports the main wrapper class, the code/stat index helpers, and the
dataclasses/enums that represent decoded records.

Typical direct API usage starts here:

```python
from rocprof_trace_decoder import Decoder, CodeIndex

code = CodeIndex.from_code_json("code.json")
with Decoder() as decoder:
    records = decoder.parse_file("trace.att", isa=code)
```

The package also defines `__version__`. The version currently matches the
project version used by the Python packaging metadata.

## `rocprof_trace_decoder.records`

`records.py` is the public Python model for the decoder data. It contains
`IntEnum` classes for the decoder status, info, record, event, dispatch,
shader-data, wave-state, and instruction-category values, plus dataclasses for
the record payloads emitted by the C decoder.

The dataclasses are deliberately plain Python objects. They do not expose
`ctypes` fields or raw C pointers. This keeps downstream users insulated from
the callback lifetime rules and struct-layout details in the C API.

Important dataclasses include:

- `Pc`: `(address, code_object_id)` key used for ISA lookup and stats.
- `Instruction`: one decoded instruction execution event in a wave.
- `Wave`: wave lifetime, timeline states, and decoded instruction stream.
- `Occupancy`, `PerfEvent`, `ShaderData`, `Realtime`, `OtherSimdInstruction`,
  `Event`, and `Dispatch`: direct Python equivalents of the public decoder
  payloads.
- `TraceRecords`: the aggregate returned by `Decoder.parse*`, with one list per
  record type and a `batches` list preserving callback batch order.

This is the best module for users to import when they want type names for
annotations or record inspection.

## `rocprof_trace_decoder.bindings`

`bindings.py` owns the direct `ctypes` interface to
`librocprof-trace-decoder`. It defines the private C struct mirrors, callback
signatures, status checking, shared-library discovery, and conversion from raw
callback batches into `records.py` objects.

The main public class is `Decoder`. It manages a handle created with
`rocprof_trace_decoder_create_handle()` and destroyed with
`rocprof_trace_decoder_destroy_handle()`. `Decoder` supports:

- `parse(data, isa=...)`
- `parse_file(path, isa=...)`
- `parse_chunks(chunks, isa=...)`
- `load_code_object_data(...)`
- `load_code_object(...)`
- `unload_code_object(...)`
- `info_string(...)`
- `status_string(...)`

For ATT tool usage, the high-level path normally passes a `CodeIndex` as the
ISA provider. The lower-level code-object load methods remain available for
users who explicitly want the decoder's built-in disassembly mode.

The module also contains the `IsaProvider` protocol. Any object with an
`isa_for_pc(pc) -> tuple[str, int] | None` method can provide instructions to
the decoder callback.

## `rocprof_trace_decoder.code_index`

`code_index.py` is the shared bridge between static code metadata and dynamic
trace records. It has two related responsibilities.

First, it provides ISA text to the decoder. `CodeIndex.from_code_json()` loads
the generated `code.json` and builds a map from `Pc(code_object_id, address)` to
instruction text, source text, line number, and estimated instruction size.
Because `CodeIndex` implements `isa_for_pc`, it can be passed directly to
`Decoder.parse_file(..., isa=code_index)`.

Second, it accumulates instruction statistics from decoded waves. As each wave
is processed, `accumulate_wave()` updates per-instruction hit count, latency,
stall, and idle counters. The same implementation is used by the final ATT tool
and by `test/csv_test.py`, which keeps test validation and output generation on
one stats path.

The module can write updated `code.json` and `stats_*.csv` files. It can also
load control CSVs through `from_stats_csv()` and compare accumulated counters
with expected values through `validate_expected()`.

This module is intentionally not responsible for ROCprof Compute Viewer sidecar
JSON files. That output format belongs to `rcv.py`.

## `rocprof_trace_decoder.rcv`

`rcv.py` writes ROCprof Compute Viewer sidecar JSON. Its main class,
`RcvOutputWriter`, receives decoded records grouped by shader engine and writes
viewer-facing files such as:

- `filenames.json`
- `occupancy.json`
- `realtime.json`
- `wstates*.json`
- `se*_perfcounter.json`
- `se*_sm*_sl*_wv*.json`
- `other_simd_se*_*.json`
- `shaderdata_*_*.json`

The writer also calls `CodeIndex.accumulate_wave()` while writing wave files, so
the final stats CSV and viewer files are produced from the same decoded waves.

This module is best treated as an internal formatting layer. Its JSON schema is
chosen to match the ROCprof Compute Viewer conventions, not to be a general
Python API for trace analysis.

## `rocprof_trace_decoder.att`

`att.py` is the high-level orchestration module for turning already identified
ATT traces into final CSV/JSON outputs. It does not parse trace file names.
Callers must pass explicit `AttTrace` objects containing the trace path, shader
engine id, and run id.

The primary reusable function is `generate_att_outputs(...)`. It takes a small
set of explicit inputs and handles the standard output file naming:

- input traces: `AttTrace(path=..., shader_engine=..., run=...)`
- existing `code.json`
- output directories: `ui_output_<name><run>`
- stats files: `stats_<output-dir-name>.csv`

This boundary is intentional. The reusable API can parse an ATT buffer from any
file name. Naming conventions such as `*_shader_engine_<se>_<run>.att` belong in
scripts that know how rocprofiler wrote a specific output directory.

## `generate_code.py`

`generate_code.py` creates the static code metadata consumed by `CodeIndex` and
the ATT tool. It reads GPU code objects, extracts disassembly with
`llvm-objdump`, reads DWARF line information with `pyelftools`, and writes:

- `code.json`
- `snapshots.json` when source files are found
- `source_*` copies for snapshotted source files

The module is both an implementation detail of `rocprof_trace_decoder.att` and
a standalone helper command. It is installed as `rocprof-trace-generate-code`
for users who want to generate `code.json` explicitly.

One important normalization lives here: branch operands printed by
`llvm-objdump` as symbolic labels are converted back to the raw SOPP immediate
form expected by the decoder stitcher. This keeps `generate_code.py` output
compatible with the decoder's PC stitching behavior.

## `att_tool.py`

`att_tool.py` is the user-facing command-line script. It accepts a mixed list of
`.att`, `.out`, `.co`, `.hsaco`, `code.json`, and directory inputs; generates
`code.json` when needed; parses the conventional rocprofiler ATT file names; and
then calls `rocprof_trace_decoder.att.generate_att_outputs()`.

This is the only Python layer that assumes ATT files follow the
`*_shader_engine_<se>_<run>.att` naming convention. Users with arbitrary trace
file names can bypass this script and call `Decoder.parse_file()` or
`generate_att_outputs()` with explicit `AttTrace` metadata.

Developers can run the source-tree script directly:

```bash
python3 python/att_tool.py trace.att code_object.out
```

from the checkout without installing the package first.

Installed users should prefer the generated `rocprof-att-tool` launcher or the
Python packaging console script of the same name.

## Typical Flow

The one-command ATT path is:

1. `att_tool.py` discovers `.att` traces and code objects from CLI inputs.
2. `att_tool.py` runs `generate_code.py` if no `code.json` was provided.
3. `att_tool.py` parses conventional trace names into explicit `AttTrace`
   objects.
4. `CodeIndex` loads `code.json` and becomes the decoder ISA provider.
5. `Decoder` parses each `.att` file and returns `TraceRecords`.
6. `RcvOutputWriter` writes viewer JSON files and updates `CodeIndex` stats.
7. `CodeIndex` writes the final updated `code.json` and `stats_*.csv`.

This division keeps the wrapper reusable: users can stop at `Decoder` and
`TraceRecords` for custom analysis, use `CodeIndex` for CSV/stat workflows, or
use `generate_att_outputs()` for the complete ROCprof Compute Viewer output
pipeline.
