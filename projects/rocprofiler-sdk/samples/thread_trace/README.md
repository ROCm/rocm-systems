# Thread Trace and ROCprof Trace Decoder

Two flavours of thread-trace capture share a single `main.cpp` and differ
only by which agent `.so` is preloaded and how long the application runs
(`RUN_SECONDS`).

## Services

- Thread trace in device profiling mode
- ROCprof Trace Decoder decodes the received thread trace data
- Simple agent start/stop using roctx marker control
- Buffered agent start on first code-object load, stop in `tool_fini`

## Files

### [agent_simple.cpp](agent_simple.cpp) (`thread-trace-simple-client.so`)

- Configures thread trace in all GPU agents found with
  `rocprofiler_configure_device_thread_trace_service`
- Receives the trace data in `shader_data_callback` and calls
  `rocprof_trace_decoder_parse` to decode the data inline
- `parse` increments hitcount/latencies by pc address
- At application end, `tool_fini` writes the top hotspots into
  `thread_trace.log`

### [agent_buffered.cpp](agent_buffered.cpp) (`thread-trace-buffered-client.so`)

- Triple-buffered SQTT capture that runs the gfx9 quick-scan inline on each
  shader-data chunk as it arrives
- Reports streaming bandwidth and buffer-interrupt flags across the run
- Creates one ATT context per GPU agent and starts it automatically on that
  agent's first code-object load
- Saves one dispatch slice and decodes it at shutdown to write disassembled
  hotspots into `thread_trace_buffered.log`

### [main.cpp](main.cpp)

- Allocates 3 streams and selects the trace duration by `RUN_SECONDS`
- Warms up, calls `roctxProfilerResume`, then alternates `divide_kernel` and
  `looping_lds_kernel` until the fixed launch count or `RUN_SECONDS=N` time
  budget elapses, then `roctxProfilerPause`
