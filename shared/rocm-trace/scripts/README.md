# rocm_trace_gen: dispatch-table probe generator

Generates a TraceLogging probe for every entry of a runtime's dispatch tables, and the
function that installs them. Each probe writes two events per call:

- `<api>_enter` (opcode start): the arguments;
- `<api>_exit` (opcode stop): the result, and the outputs the call wrote, recorded only
  when the call succeeded.

The transport timestamps each event and records the thread; a consumer pairs enter
and exit per thread, in order. With no session listening, an event costs one load and
a branch: TraceLoggingWrite evaluates its field expressions only when the event is
enabled.

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
output_name: hip_trace_probes       # hip_trace_probes.h and .cpp
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
apis:
  hipMalloc:
    out: [ptr]                      # pointers the call writes through; on exit,
                                    # the value written is recorded as ptr_out
    keywords: [ALLOC]               # added to API, which every event has
  hipDeviceGetName:
    out: [name]
    strings: [name]                 # char pointers recorded as strings, not
                                    # addresses; may name result
  hipIpcOpenMemHandle:
    omit: [handle]                  # not recorded; may name result
  hsa_signal_load_relaxed:
    level: verbose
  hipApiName:
    skip: true                      # no probe; the entry is left as it is
    reason: "called by tracing tools"
```

The generator stops with every problem it finds: an API or argument that does not
exist, `out` on something that is not a pointer, `strings` on something that is not a
`char` pointer, `skip` without a reason, unknown keys or keywords, and any parameter
whose type has no field mapping.

## Field mapping

| C type | Field |
|---|---|
| integers of 1, 2, 4 or 8 bytes | `TraceLogging[U]Int8/16/32/64` |
| enums | `TraceLoggingInt32` (`Int64` for 8-byte enums) |
| `bool` | `TraceLoggingBoolean` |
| `float`, `double` | `TraceLoggingFloat32`, `TraceLoggingFloat64` |
| pointers, handles, function pointers | `TraceLoggingPointer` |
| `char` pointers listed in `strings` | `TraceLoggingString` (a null pointer is recorded as "") |
| structs passed by value with at most 8 number, enum or pointer members (`dim3`, `hipExtent`) | one field per member, `<name>_<member>` |
| anything else (arrays, larger or nested structs) | an error, until the parameter is listed in `omit` |

## Generated code

- `<output_name>.h` includes the configured headers and declares
  `void rocm_trace_install_<Table>(<Table>* table)` for each table.
- `<output_name>.cpp` declares the provider (the runtime defines and registers it),
  holds one probe per entry, and defines the install functions.

`rocm_trace_install_<Table>` saves each non-null entry and replaces it with its probe.
It touches only entries inside the table's size, so a table from an older runtime is
safe; null, skipped, and retired (`void*`) entries are left as they are. Calling it
again on the same table is harmless. It is not thread-safe: call it before the table
is published. There is one saved copy per table type.

The generated files are wrapped in `// clang-format off` so that formatting tools
leave them byte-identical to the generator's output.

## Keeping the output current

1. A table that gains an entry fails to compile: the generated source asserts the
   table's size as it was at generation.
2. A changed signature fails to compile: the probe no longer converts to the entry's
   type.
3. A changed configuration or generator is caught by `--check` in CI.

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
