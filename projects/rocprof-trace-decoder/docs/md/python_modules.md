# Python Modules

This document summarizes the Python modules added for the rocprof-trace-decoder
API and source-tree ATT helpers. The installable Python package is intentionally
small: only the reusable API under `rocprof_trace_decoder` is installed.

The Python API wraps the handle-based decoder functions from
`rocprof_trace_decoder.h` and the record types from `trace_decoder_types.h`.
The quick-scan and standalone trace extraction APIs are intentionally not
wrapped here.

The API can be installed as a normal Python package (`python3 -m pip install .`)
or with CMake when `BUILD_PYTHON=ON`. The CMake install path defaults to the
ROCm-prefix-relative Python site-packages directory `lib/python3/site-packages`.
This package is pure Python over `ctypes`, so the CMake install is shared by
supported Python 3 minor versions. Packagers can override
`ROCPROF_TRACE_DECODER_PYTHON_INSTALL_DIR`. Users running directly from a ROCm
prefix need that directory on `PYTHONPATH`; ROCm environment setup such as the
rocprofiler-sdk setup script/modulefile already prepends the matching ROCm
Python directory. TheRock wheel packaging exposes this package as a top-level
Python package from the profiler wheel instead of relying on a nested
ROCm-prefix path being present on `sys.path`.

## Module Overview

| Module | Role | API Status |
| --- | --- | --- |
| `rocprof_trace_decoder` | Public package entry point | Public |
| `rocprof_trace_decoder.records` | Python record/enumeration model for decoder output | Public |
| `rocprof_trace_decoder.bindings` | `ctypes` wrapper around `librocprof-trace-decoder` | Public for `Decoder`, internal for C mirrors |
| `rocprof_trace_decoder.code_index` | ISA lookup and instruction-stat accumulation | Public utility |
| `rocprof_trace_decoder.codegen` | Builds in-memory code metadata from explicit code object inputs | Public utility |
| `rocprof_trace_decoder.rcv` | ROCprof Compute Viewer JSON writer | Internal utility |
| `rocprof_trace_decoder.att` | High-level ATT decode orchestration with explicit trace metadata | Public utility |
| `generate_code.py` | Generates `code.json`, snapshots, and source copies from code objects | Tool module |
| `att_tool.py` | CLI script for ATT output generation | Tool script |

## `rocprof_trace_decoder`

`rocprof_trace_decoder` is the import root for users who want the Python API.
It re-exports the main wrapper class, the explicit code object helpers, the
code/stat index helpers, and the dataclasses/enums that represent decoded
records.

Typical direct API usage starts here:

```python
from rocprof_trace_decoder import CodeObject, Decoder, generate_code_artifacts

artifacts = generate_code_artifacts([
    CodeObject(path="kernel.hsaco", code_object_id=0),
])
with Decoder() as decoder:
    records = decoder.parse(att_bytes, isa=artifacts.code_index)
```

Users who want the decoder's built-in disassembly mode can keep one decoder
handle alive, upload code-object bytes to that handle, parse ATT bytes, and
unload by id:

```python
from rocprof_trace_decoder import Decoder

with Decoder() as decoder:
    decoder.load_code_object(
        code_object_bytes,
        load_id=0,
        load_addr=0x8000,
        load_size=0x4000,
    )
    records = decoder.parse(att_bytes)
    decoder.unload_code_object(0)
```

Built-in disassembly requires a decoder shared library built with the COMGR or
LLVM disassembly backend. Builds without that backend raise `DecoderError` with
`DecoderStatus.ERROR_NOT_IMPLEMENTED` when code objects are loaded or parsed
without a custom ISA provider.

The package also defines `__version__`. The value is stored in `_version.py`,
and the Python packaging metadata reads the same value through
`tool.setuptools.dynamic`, so wheel builds and runtime imports use one source.

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

- `parse(data, isa=...)`, where `data` is ATT bytes.
- `parse_file(path, isa=...)`, a convenience wrapper around `parse(...)`.
- `parse_chunks(chunks, isa=...)`, for callers that intentionally decode each
  chunk as a separate trace buffer.
- `load_code_object(data, load_id=..., load_addr=..., load_size=...)`, where
  `data` is code-object bytes or a path. `load_size` is the loaded memory range
  size and is intentionally distinct from the code-object byte length, which is
  inferred from `data`.
- `load_code_object_file(path, load_id=..., load_addr=..., load_size=...)`.
- `load_code_object_data(load_id, load_addr, load_size, data)`, retained for
  compatibility with the original wrapper spelling.
- `unload_code_object(...)`
- `info_string(...)`
- `status_string(...)`

`Decoder` is the public Python handle object. Raw C handle values stay private;
users keep per-handle state by keeping the `Decoder` instance alive.

For ATT tool usage, the high-level path normally passes a `CodeIndex` as the
ISA provider. Users who explicitly want the decoder's built-in disassembly mode
can load and unload code-object bytes on the same `Decoder` before parsing ATT
bytes. That mode requires a COMGR/LLVM-enabled decoder library; otherwise the
C API reports `ERROR_NOT_IMPLEMENTED`.

The module also contains the `IsaProvider` protocol. Any object with an
`isa_for_pc(pc) -> tuple[str, int] | None` method can provide instructions to
the decoder callback.

Shared-library lookup is ordered so user overrides win before ROCm defaults:
explicit `lib_path`, `ROCPROF_TRACE_DECODER_LIB`, non-empty `LD_LIBRARY_PATH`
entries, `ROCM_HOME/lib`, `ROCM_PATH/lib`, TheRock package library roots,
`/opt/rocm/lib`, then the platform loader fallback. The lookup accepts both the
unversioned library name and the decoder SONAME used by TheRock runtime wheels.

## `rocprof_trace_decoder.code_index`

`code_index.py` is the shared bridge between static code metadata and dynamic
trace records. It has two related responsibilities.

First, it provides ISA text to the decoder. A `CodeIndex` is normally produced
by `generate_code_artifacts()` from explicit code objects, or loaded from an
existing RCV `code.json` with `CodeIndex.from_code_json(path)`. It maps
`Pc(code_object_id, address)` to instruction text, source text, line number,
and estimated instruction size. Because `CodeIndex` implements `isa_for_pc`, it
can be passed directly to `Decoder.parse_file(..., isa=code_index)`.

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

## `rocprof_trace_decoder.codegen`

`codegen.py` builds static code metadata directly from explicit code object
inputs. The public input type is `CodeObject(path=..., code_object_id=...)`.
`path` may be a string or `Path`. The module does not scan directories, infer
ids from names, write files, or require `code.json`. Those behaviors belong to
source-tree scripts or RCV writers.

The main helper, `generate_code_artifacts(...)`, returns a `CodeArtifacts`
object containing:

- `code_index`: a `CodeIndex` ready to use as the decoder ISA provider.
- `source_paths`: source files discovered from DWARF comments, suitable for
  the RCV snapshot writer when the caller wants viewer source snapshots.

The module reads DWARF line information with `pyelftools`, disassembles with
`llvm-objdump`, and extracts labels from ELF symbols. Function symbols and
debug labels become RCV label rows using the viewer-compatible `; _...` form.

No caller needs to run `generate_code.py` before using this API. Scripts that
want `code.json` can call `generate_code_artifacts(...)` and then explicitly
write the returned `CodeIndex`.

## `rocprof_trace_decoder.rcv`

`rcv.py` writes ROCprof Compute Viewer sidecar JSON. Its main class,
`RcvOutputWriter`, receives decoded records grouped by shader engine and writes
viewer-facing files such as:

- `filenames.json`
- `occupancy.json`
- `realtime.json`
- `se*_perfcounter.json`
- `se*_sm*_sl*_wv*.json`
- `other_simd_se*_*.json`
- `shaderdata_*_*.json`

The writer also calls `CodeIndex.accumulate_wave()` while writing wave files, so
the final stats CSV and viewer files are produced from the same decoded waves.

This module is best treated as an internal formatting layer. Its JSON schema is
chosen to match the ROCprof Compute Viewer conventions, not to be a general
Python API for trace analysis.

`RcvOutputWriter` accepts only the exact output formats `json` and `csv`.
Schema literals such as `navi`, `filenames.json`, and `se*_...` file names are
viewer-format details and intentionally stay in this module.

## `rocprof_trace_decoder.att`

`att.py` is the high-level orchestration module for turning already identified
ATT traces into final CSV/JSON outputs. It does not parse trace file names.
Callers must pass explicit `AttTrace` objects containing the trace path, shader
engine id, and run id.

The primary reusable function is `generate_att_outputs(...)`. It takes a small
set of explicit inputs and handles the standard output file naming:

- input traces: `AttTrace(path=..., shader_engine=..., run=...)`
- optional `CodeIndex`
- optional explicit `CodeObject(path=..., code_object_id=...)` inputs
- optional explicit source paths for RCV snapshots
- output directories: `ui_output_<name><run>`
- stats files: `stats_<output-dir-name>.csv`

The `formats` argument accepts `json`, `csv`, or both as a comma-separated
string or iterable. Unknown names are rejected instead of being matched as
substrings.

This boundary is intentional. The reusable API can parse an ATT buffer from any
file name. Naming conventions such as `*_shader_engine_<se>_<run>.att` belong in
scripts that know how rocprofiler wrote a specific output directory. Existing
`code.json` files are likewise a script concern: scripts may load them with
`CodeIndex.from_code_json(path)` and pass the resulting object into
`generate_att_outputs(...)`. Callers that have code object files do not need an
intermediate `code.json`; they can pass explicit `CodeObject` values to
`generate_att_outputs(...)`, or call `generate_code_artifacts(...)` themselves
and pass the resulting `CodeIndex`.

## `generate_code.py`

`generate_code.py` is a source-tree helper for writing the RCV static code
files. It reads GPU code objects, extracts disassembly with
`llvm-objdump`, reads DWARF line information with `pyelftools`, and writes:

- `code.json`
- `snapshots.json` when source files are found
- `source_*` copies for snapshotted source files

The script is not part of the installable API and is not required by
`rocprof_trace_decoder.att`. The reusable code object API lives in
`rocprof_trace_decoder.codegen`.

When code objects are not provided explicitly, the command scans the current
directory for `.hsaco`, `.out`, `.co`, and `.o` ELF files. The script maps those
files to explicit `CodeObject` values before calling the reusable API.

The reusable codegen implementation normalizes branch operands printed by
`llvm-objdump` as symbolic labels back to the raw SOPP immediate form expected
by the decoder stitcher. This keeps both `generate_code.py` output and
in-memory `CodeIndex` generation compatible with the decoder's PC stitching
behavior.

## `att_tool.py`

`att_tool.py` is the user-facing command-line script. It accepts a mixed list of
`.att`, `.out`, `.co`, `.hsaco`, `code.json`, and directory inputs. Code
objects alone generate only `code.json`, `snapshots.json`, and source snapshots.
ATT inputs do not require `code.json` or code object files at argument
validation time.

When ATT inputs are present, the script uses `code.json` or code objects only
when they are provided by the user or found through an explicitly provided
directory input. Otherwise it calls
`rocprof_trace_decoder.att.generate_att_outputs()` without ISA metadata.
Whether a given trace can be decoded without ISA metadata is determined by the
decoder library.

When an ATT file follows the conventional
`*_shader_engine_<se>_<run>.att` naming convention, the script uses that
metadata. Arbitrarily named `.att` files are also accepted; they default to
run `1` and receive shader-engine ids in discovery order, skipping ids already
claimed by conventional names. Users who need exact metadata for nonstandard
multi-SE captures can bypass the script and call `generate_att_outputs()` with
explicit `AttTrace` objects.

Developers can run the source-tree script directly:

```bash
python3 python/att_tool.py trace.att code_object.out
```

from the checkout without installing the package first.

## Typical Flow

The one-command ATT path is:

1. `att_tool.py` discovers `.att` traces, `code.json`, and code objects from
   explicit CLI inputs and directory inputs.
2. `att_tool.py` calls `rocprof_trace_decoder.codegen` when code objects are
   available and a new `CodeIndex` is needed. If there are no ATT traces, the
   command writes only RCV static code files.
3. `att_tool.py` turns ATT trace paths into explicit `AttTrace` objects, using
   trace filename metadata when present and deterministic defaults otherwise.
4. When code metadata is available, `CodeIndex` becomes the decoder ISA
   provider.
5. `Decoder` parses each `.att` file and returns `TraceRecords`.
6. `RcvOutputWriter` writes viewer JSON files and updates `CodeIndex` stats.
7. `CodeIndex` writes the final updated `code.json` and `stats_*.csv`.

This division keeps the wrapper reusable: users can stop at `Decoder` and
`TraceRecords` for custom analysis, use `CodeIndex` for CSV/stat workflows, or
use `generate_att_outputs()` for the complete ROCprof Compute Viewer output
pipeline.
