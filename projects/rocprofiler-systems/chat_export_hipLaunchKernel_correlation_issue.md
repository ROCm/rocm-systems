# Chat Export: hipLaunchKernel and Kernel Correlation Issue

**Date:** October 13, 2025  
**Topic:** ROCProfiler SDK kernel dispatch correlation and hipLaunchKernel tracing domains

## Discussion Summary

### Context
User was investigating performance differences across ROCm versions using Tuolumne and found that ROCm 7.0.0 beta2 (RC4) version of `rocprof-sys-run` was not capturing the event task graph correctly. Kernel events were appearing as top-level events instead of being nested under `hipLaunchKernel`.

### Key Questions and Analysis

#### 1. User Question: "Which domain is used for hipLaunchKernel?"

**Answer:** `hipLaunchKernel` uses **TWO** tracing domains:

1. **HIP Runtime API Callback Tracing** (`ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API`)
   - **Purpose**: Traces the `hipLaunchKernel*` API call itself
   - **When**: When the application calls `hipLaunchKernel()`
   - **Data captured**: Function entry/exit, parameters, timing of the API call
   - **Type**: **Callback tracing** (immediate)

2. **Kernel Dispatch Buffered Tracing** (`ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH`)
   - **Purpose**: Traces the actual GPU kernel execution
   - **When**: When the kernel runs on the GPU (after being dispatched)
   - **Data captured**: Kernel execution timing, GPU queue info, workgroup dimensions
   - **Type**: **Buffered tracing** (batched)

#### 2. The Complete Flow:

```cpp
// Application code
hipLaunchKernelGGL(myKernel, grid, block, 0, 0, args...);

Step 1: API Call Tracing

// ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API triggered
case ROCPROFILER_CALLBACK_TRACING_HIP_RUNTIME_API:
{
    tool_tracing_callback_start(category::rocm_hip_api{}, record, user_data, ts);
    // Records: function name, arguments, start/end timestamps
    break;
}

Step 2: Kernel Execution Tracing (happens later when kernel runs on GPU)

// ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH triggered  
if(header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
{
    auto* record = static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
    
    // Extract kernel information
    auto _name = tim::demangle(_kern_sym_data->kernel_name);
    auto _beg_ns = record->start_timestamp;    // GPU execution start
    auto _end_ns = record->end_timestamp;      // GPU execution end
    auto _agent_id = record->dispatch_info.agent_id;
    auto _queue_id = record->dispatch_info.queue_id;
    
    // Cache for analysis
    cache_kernel_dispatch(record, _stream_id);
}

3. Tracing Domain Types Explained
Callback Tracing Domains (ROCPROFILER_CALLBACK_TRACING_*):

Events delivered immediately via callback functions
Real-time event processing, API call interception
Lower overhead per event, but can impact application performance
Buffered Tracing Domains (ROCPROFILER_BUFFER_TRACING_*):

Events collected in buffers and delivered in batches
High-throughput event collection, minimal impact on application performance
Better for high-frequency events like kernel dispatches

4. Why Kernel Dispatch is Buffered
Kernel dispatch is buffered because:

High Frequency: GPU kernels can launch thousands of times per second
Performance Critical: Immediate callbacks would severely impact GPU performance
Batch Processing: Kernel execution data is better processed in batches
Timeline Reconstruction: Buffered data allows building accurate execution timelines
The Correlation Problem
Issue Description
User reported that in ROCm 7.0.0, kernel events were appearing as:

Top-level events (incorrect)
Sometimes under hipDeviceSynchronize (incorrect)
Sometimes under hipGetLastError (incorrect)
Instead of properly nested under hipLaunchKernel (correct)
Expected behavior (6.4.2):

|0>>> |_hipLaunchKernel                    | 1000001 |    1 | wall_clock | sec  | 10.6376
|0>>>   |_mbind                            |       1 |    2 | wall_clock | sec  | 0.000007
|0>>>   |_void emulateRajaForall<...>      | 100...

Actual behavior (7.0.0):

|0>>> |_hipLaunchKernel                    | 1000001 |    1 | wall_clock | sec  | 10.0499
|0>>>   |_mbind                            |       1 |    2 | wall_clock | sec  | 0.000004
|0>>> |_hipGetLastError                    | 1000001 |    1 | wall_clock | sec  | 1.01038e
|0>>> |_hipDeviceSynchronize               | 1000001 |    1 | wall_clock | sec  | 19.526
|0>>> |_void emulateRajaForall<...>        | 100...  // <- Should be under hipLaunchKernel

Root Cause Analysis
Key Areas to Investigate:
Stream ID Correlation Logic - rocprofiler-sdk.cpp lines 1406-1420:

uint64_t _stream_id = get_stream_id(record).handle;
if(_stream_id == 0)
{
    // kernel_dispatch is not associated with a HIP stream
    _group_by_queue = true;
}

2. Correlation ID Management - In buffered tracing callback:
auto _corr_id = record->correlation_id.internal;

3. Stream ID Stack Management - Around lines 1849-1865:
void tool_hip_stream_callback(rocprofiler_callback_tracing_record_t record, ...)
{
    if(record.operation == ROCPROFILER_HIP_STREAM_SET)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
            stream_id_push(stream_id);
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
            stream_id_pop();
    }
}

The Core Issue

The problem is that kernel dispatch events are not being properly correlated with their originating HIP API calls. The correlation_id.internal field should link kernel dispatches back to their launching hipLaunchKernel calls, but this correlation is breaking down in ROCm 7.0.0.

Resolution Strategy
Primary Fix Location: tool_tracing_buffered() function around line 1405 in rocprofiler-sdk.cpp

Key Steps:

1. Verify correlation ID assignment in set_kernel_rename_and_stream_correlation_id()
2. Check correlation lookup logic in tool_tracing_buffered()
3. Ensure proper timing of correlation ID establishment
4. Validate stream context maintenance across API call boundaries

Debugging Steps:

1. Enable verbose logging: export ROCPROFSYS_VERBOSE=3
2. Check domain configuration: export   ROCPROFSYS_ROCM_DOMAINS="hip_runtime_api,kernel_dispatch"
3. Test grouping settings: export ROCPROFSYS_ROCM_GROUP_BY_QUEUE=false

Summary
For hipLaunchKernel:

HIP Runtime API domain captures the host-side API call
Kernel Dispatch domain captures the GPU-side kernel execution
Correlation IDs should link these two events together
The issue is in the correlation mechanism failing in ROCm 7.0.0, causing kernel events to appear incorrectly in the trace hierarchy
The fix requires ensuring proper correlation between buffered kernel events and their originating callback API events through the correlation_id.internal field.

Next Steps
Focus investigation on the tool_tracing_buffered function around line 1374
Examine correlation ID assignment and lookup mechanisms
Test with different ROCProfiler SDK domain configurations
Validate stream correlation logic for proper event nesting
Files Referenced
rocprofiler-sdk.cpp - Main implementation
rocprofiler-sdk.cpp - Domain configuration
Tuolumne performance analysis results showing the correlation issue
