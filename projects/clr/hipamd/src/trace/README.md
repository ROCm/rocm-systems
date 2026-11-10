# rocm_hip API events

With `HIP_ENABLE_ROCM_TRACE=ON`, every HIP API call that goes through the dispatch tables writes
`<api>_enter` and `<api>_exit` through TraceLogging: LTTng-UST on Linux, ETW on Windows. A
profiler outside the process records them from a trace session; nothing is loaded into the
application. The events follow the ROCm runtime contract (RC-API-1, RC-API-2); the API tracing
design and plan are in sdk-future-design (`proposals/api-tracing-hld.md`,
`proposals/api-tracing-plan.md`).

| File | |
|---|---|
| `rocm_trace_hip.yaml` | What the generator cannot read from the headers: outputs, strings, kernel names |
| `hip_trace_probes.h`, `.cpp` | Generated from the dispatch tables and `rocm_trace_hip.yaml`: one probe per entry and the install functions. Do not edit |
| `hip_trace_probes.json` | Generated with them: the probe schema, every event field with its C type, enum names and capture rule. Installed as `share/rocm-trace/rocm_hip.json` for tools that read the trace. Do not edit |
| `hip_trace.hpp`, `.cpp` | The `rocm_hip` provider, `ROCM_TRACE_DISABLE`, `ROCM_TRACE_CAPTURE`, and the install call |

## Behaviour

- The probes are installed when the runtime fills its dispatch tables
  (`GetDispatchTableImpl` in `hip_api_trace.cpp`), before the tables are handed to tools, so
  a tool that wraps the tables calls the probe and the probe calls the implementation.
- With no session, an event costs one enabled check.
- `ROCM_TRACE_DISABLE=1` installs nothing.
- `ROCM_TRACE_CAPTURE=scalars` records no strings and no output values; `none` keeps only
  `result` and `exception`.

## Building

- **Linux:** needs the LTTng flavour of [microsoft/tracelogging](https://github.com/microsoft/tracelogging)
  (`LTTng/`, built and installed with CMake) and liblttng-ust 2.13 or later; CMake finds them
  with `find_package(tracelogging)`.
- **Windows:** TraceLogging is in the Windows SDK.

## Recording

```sh
lttng create hip && lttng enable-event --userspace 'rocm_hip:*' && lttng start
./app
lttng stop && lttng destroy && babeltrace2 ~/lttng-traces/hip*
```

```bat
logman start hip -p {07416523-fe27-5716-e852-d18774a03743} 0x1 4 -o hip.etl -ets
app.exe
logman stop hip -ets && tracerpt hip.etl -o hip.xml -of XML
```

## Regenerating

After changing `rocm_trace_hip.yaml`, or when `hip_api_trace.hpp` gains or changes an entry (the
build fails with "HipDispatchTable changed after the trace probes were generated"), regenerate
and commit the result. From the repository root:

```sh
pip install -r shared/rocm-trace/scripts/requirements.txt
python3 shared/rocm-trace/scripts/rocm_trace_gen.py \
    --config projects/clr/hipamd/src/trace/rocm_trace_hip.yaml \
    --output-dir projects/clr/hipamd/src/trace \
    -I <build>/hipamd/include -I projects/clr/hipamd/include -I projects/hip/include \
    -D __HIP_PLATFORM_AMD__ --resource-dir "$(clang -print-resource-dir)"
```

`<build>/hipamd/include` holds the build's `hip/hip_version.h`. `-DCLR_ROCM_TRACE_CODEGEN_CHECK=ON`
makes the build run the same command with `--check`.
