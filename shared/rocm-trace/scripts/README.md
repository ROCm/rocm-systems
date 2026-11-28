# rocm_trace_gen: dispatch-table probe generator

Generates a TraceLogging probe for every entry of a runtime's dispatch tables, and the
function that installs them, as the runtime contract's RC-API-1 and RC-API-2 require.
Each probe writes two events per call:

- `<api>_enter` (opcode start): the arguments;
- `<api>_exit` (opcode stop): `result`, the outputs the call wrote (recorded only when
  it succeeded), and `exception`, true when the call threw. A probe that sees an
  exception writes the exit event and rethrows, so every enter has its exit.

The transport timestamps each event and records the thread; a consumer pairs enter
and exit per thread, in order. With no session listening, an event costs one load and
a branch: TraceLoggingWrite evaluates its field expressions only when the event is
enabled, so the probes need no `ROCM_TRACE_ENABLED` check (that is for work done
before an event, which these probes do not do).

Each event name is written from exactly one place, so it always has one set of fields:
LTTng corrupts a trace in which one event name has two layouts. What a probe does not
record (an output after a failure, or anything the capture level excludes) is written
as 0 or "".

The runtimes check the generated files in. Their builds do not run the generator, so
they need neither Python nor libclang.

## Running

```
pip install -r requirements.txt   # Python 3.8 or later

python3 rocm_trace_gen.py --config <runtime>.yaml --output-dir <dir> \
    -I <include dir> ... [-D NAME[=VALUE] ...] \
    --resource-dir "$(clang -print-resource-dir)" [--check]
```

- `-I` and `-D` are what the runtime's own build uses for the headers in `includes`.
- `--resource-dir` gives libclang its builtin headers (`stddef.h`); the pip wheel has
  none. Any error while parsing fails the run, because a missing header silently turns
  types into `int`.
- `--check` writes nothing and fails, with a diff, if the files in `--output-dir`
  differ from what would be generated. CI runs it.

## Configuration

One YAML file per runtime, kept next to the generated files and reviewed by the
runtime's owners. Every table entry is traced; the file holds only what the C types
cannot say.

```yaml
version: 1
provider: rocm_hip                  # the TraceLogging provider symbol
output_name: hip_trace_probes       # hip_trace_probes.h, .cpp and .json
includes:                           # the translation unit that is parsed, in order;
  - hip/hip_runtime.h               # also included by the generated header
  - hip/amd_detail/hip_api_trace.hpp
tables:
  - name: HipDispatchTable
    size_member: size               # member holding the table's size in bytes
success:                            # result type -> value meaning success; outputs
  hipError_t: hipSuccess            # are recorded on exit only after it
defaults:
  level: information                # information | verbose
capture_level: rocm_hip_capture     # optional: the runtime's int holding the
                                    # ROCM_TRACE_CAPTURE level (see below)
apis:
  hipMalloc:
    out: [ptr]                      # pointers the call writes through; on exit,
                                    # the value written is recorded as ptr
    keywords: [ALLOC]               # added to API, which every event has
  hipDeviceGetName:
    out: [name]
    strings: [name]                 # a char-pointer output recorded as a string;
                                    # may also name result, or a char* input that
                                    # holds text (an API that is not const-correct)
  hipModuleGetFunction:
    uncapped: [kname]               # a string recorded whole: kernel names and
                                    # URIs only (RC-GEN-8)
  hipStreamAddCallback:
    omit: [userData]                # not recorded; may name result
  hsa_signal_load_relaxed:
    level: verbose
  hipApiName:
    skip: true                      # no probe; the entry is left as it is
    reason: "called by tracing tools"
```

Names in `out`, `strings`, `uncapped` and `omit` are field names (below); `result` is
the return value. `const char*` inputs are recorded as strings without being listed
(RC-API-2); a `char*` input is recorded as an address unless `strings` names it.

The generator stops with every problem it finds: an API or argument that does not
exist, `out` on something that is not a pointer, `strings` on something that is not a
`char`-pointer output, `char*` input or result (or on a `const char*` input, which is a
string already), `uncapped` on something that is not a string, `skip`
without a reason, unknown keys or keywords, a parameter whose type has no field
mapping, two fields with one name, and an event with more fields than TraceLogging
allows (96).

## Field names

Field names are the parameter names of the public declaration (RC-API-1): the
function's own declaration when the parsed headers have one with the same parameter
types, else the table's. A parameter with no name is `arg_<N>`, `<N>` its 1-based
position. A parameter named `result`, `tid`, `timestamp`, `exception` or `arg_*` gets
the prefix `arg_`. An output is recorded on exit under its parameter's name; on enter,
the same name holds the pointer.

## Field mapping

| C type | Field |
|---|---|
| integers of 1, 2, 4 or 8 bytes | `TraceLogging[U]Int8/16/32/64` |
| enums | the integer field of the enum's underlying type |
| `bool` | `TraceLoggingBoolean` |
| `float`, `double` | `TraceLoggingFloat32`, `TraceLoggingFloat64` |
| pointers, handles, function pointers, non-const `char*` inputs | `TraceLoggingPointer` |
| `const char*` inputs; `char*` inputs, `char*` outputs and results listed in `strings` | `TraceLoggingCountedString`, at most `ROCM_TRACE_STRING_CAP` (256) bytes; `TraceLoggingString` if `uncapped`. A null pointer is recorded as "" |
| structs passed by value (`dim3`, `hipExtent`, nested structs) | every component (RC-API-1), one field per member: `<name>_<member>`, `<name>_<member>_<member>` |
| arrays of integers, enums, bools or pointers | `TraceLogging<type>FixedArray` |
| arrays of `float` or `double` | one field per element, `<name>_<index>`: the LTTng flavour has no floating-point array fields |
| arrays of structs | one set of fields per element: `<name>_<index>_<member>` |
| unions | their bytes, `TraceLoggingUInt8FixedArray` (which member is active is unknown); an anonymous union member is `<name>_union` |
| anything else (`long double`, incomplete types) | an error, until the parameter is listed in `omit` |

## Capture levels

With `capture_level` set, the probes read that `int` (defined by the runtime, set from
the `ROCM_TRACE_CAPTURE` environment variable before the tables are installed) and
record, by `ROCM_TRACE_CAPTURE_*` level (`rocm/trace/values.h`):

| Level | Records |
|---|---|
| `FULL` (default) | everything |
| `SCALARS` | no strings and no output values: those fields are "" and 0 |
| `NONE` | only `result` and `exception`: every argument field is 0 or "" |

## Generated code

- `<output_name>.h` includes the configured headers and declares
  `void rocm_trace_install_<Table>(<Table>* table)` for each table.
- `<output_name>.cpp` declares the provider (the runtime defines and registers it),
  holds one probe per entry, and defines the install functions.
- `<output_name>.json`: the probe schema (below).

`rocm_trace_install_<Table>` saves each non-null entry and replaces it with its probe.
It touches only entries inside the table's size, so a table from an older runtime is
safe; null, skipped, and retired (`void*`) entries are left as they are. Calling it
again on the same table is harmless. It is not thread-safe: call it before the table
is published. There is one saved copy per table type, so install one instance of each.

The runtime calls it where it finishes filling the table and before it hands the table
to tools (for HIP, at the end of the table-filling step in `GetDispatchTableImpl`,
before `ToolsInit`), so the probe is the entry nearest the implementation and tools
that wrap the table see it as the implementation (API-HLD-16).

The generated code uses only public headers, so it can be built into the runtime or
into a separate library the runtime loads.

The generated files are wrapped in `// clang-format off` so that formatting tools
leave them byte-identical to the generator's output.

## Probe schema

`<output_name>.json` describes every event the probes write, field by field, with what
a reader of the trace needs to turn the fields back into the call's arguments and that
the trace itself does not carry: the C types, the enumerators of enum types, the
members a struct was flattened into, and what each capture level records. The runtime
installs it with the library (for HIP, `share/rocm-trace/rocm_hip.json`), and tools such
as the out-of-process collector read it to format arguments as rocprofiler-sdk does in
process (`kind = 1` is `hipMemcpyHostToDevice`).

```json
{
  "schema": "rocm-trace-probe-schema", "version": 1, "provider": "rocm_hip", ...
  "enums": {"hipMemcpyKind": {"signed": false, "size": 4,
            "enumerators": {"hipMemcpyHostToHost": 0, "hipMemcpyHostToDevice": 1, ...}}},
  "skipped": [{"api": "...", "table": "...", "reason": "..."}],
  "events": [
    {"name": "hipMemcpyAsync_enter", "api": "hipMemcpyAsync", "table": "HipDispatchTable",
     "phase": "enter", "opcode": "start", "level": "information", "keywords": ["API"],
     "fields": [
        {"name": "kind", "role": "argument", "param": "kind", "type": "hipMemcpyKind",
         "encoding": "UInt32", "enum": "hipMemcpyKind", "capture": "scalars"}, ...]}
  ]
}
```

Each field has, in the order the event writes them:

| Key | |
|---|---|
| `name` | the event field name (RC-API-1) |
| `role` | `argument` (enter), `output`, `result` or `exception` (exit) |
| `param`, `param_type` | the parameter it comes from and that parameter's type, when it differs from `type` |
| `path` | for a flattened struct or array: the members and indices from the parameter (`["grid", "x"]`) |
| `type` | the C type recorded; an unnamed type is spelled `(unnamed struct)` |
| `encoding` | the TraceLogging field type (`UInt32`, `Pointer`, `CountedString`, `UInt8FixedArray`, ...), with `count` for arrays and `capped` for strings cut at `ROCM_TRACE_STRING_CAP` |
| `enum` | the enum type, whose enumerators are in `enums` |
| `capture` | the lowest `ROCM_TRACE_CAPTURE` level that records it (`none`, `scalars`, `full`); below it the field is 0 or "" |
| `on_success` | outputs: recorded only when the call succeeded |

The schema is generated, checked in and checked by `--check` like the probes. Each field
is on one line, so a changed API shows as a small diff.

## Keeping the output current

1. A table that gains an entry fails to compile: the generated source asserts the
   table's size as it was at generation.
2. A changed signature fails to compile: the probe no longer converts to the entry's
   type.
3. A changed configuration or generator is caught by `--check` in CI, for the probes
   and the schema.

## Tests

```
cd shared/rocm-trace/scripts
python3 -m pytest tests
```

- Golden files: `tests/data/fake_table.h` has one entry per mapping and edge case;
  `tests/data/expected/` holds its output. After an intended change to the output,
  rerun with `ROCM_TRACE_GEN_UPDATE_GOLDEN=1` and review the diff.
- Every configuration error in the list above has a test.
- With a rocm-systems checkout, the in-tree HIP tables are parsed, their entry counts
  compared with CLR's `HIP_ENFORCE_ABI_VERSIONING`, and every entry given a probe.
  Needs a clang resource directory (found from `clang` or set in
  `ROCM_TRACE_GEN_RESOURCE_DIR`).
- `test_build.py` compiles the generated code, runs it over stub entries, and records
  it in an LTTng session, each when the machine has what it needs:
  - compile: `ROCM_TRACE_GEN_TRACELOGGING_INCLUDE` (the LTTng flavour's `include/`);
  - run: also `ROCM_TRACE_GEN_TRACELOGGING_SOURCE` (its `LTTng/` directory) and
    liblttng-ust 2.13 or later;
  - record: also `lttng`, `lttng-sessiond` and `babeltrace2`. In a container, give
    `/dev/shm` room for the per-CPU ring buffers (for example `--shm-size=1g`).
