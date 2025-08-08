# Thread Trace and ROCprof Trace Decoder

## Services

- Thread trace in device profiling mode
- ROCprof Trace Decoder decodes the received thread trace data
- Thread trace start/stop using roctx

## Properties

### [agent.cpp](agent.cpp):

- Configures thread trace in all GPU agents found with `rocprofiler_configure_device_thread_trace_service`
- Waits until `roctxProfilerResume` is called to start thread trace
- Stops tracing at `roctxProfilerPause`
- Receives the trace data in `shader_data_callback` and calls `rocprofiler_trace_decode` to decode the data
- `rocprofiler_trace_decode` calls `parse` (a lambda)
- `parse` receives the dedecoded data and increments hitcount/latencies by pc address
- At application end, `tool_fini` calls `gen_output_stream` to write the top hotspots into `thread_trace.log`

### [main.cpp](main.cpp):

- Defines a few different kernels and runs them
- The first loop iteration warms up the kernels
- The second iteration calls `roctxProfilerResume` to start thread trace
- After the loop ends, `roctxProfilerPause` is called to stop tracing
