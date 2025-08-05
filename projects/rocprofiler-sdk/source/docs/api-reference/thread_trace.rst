.. meta::
    :description: ROCprofiler-SDK is a tooling infrastructure for profiling general-purpose GPU compute applications running on the ROCm software stack
    :keywords: ROCprofiler-SDK API reference, Thread trace, ROCprof Trace Decoder, SQTT, ATT, GPU tracing

.. _thread-trace:

ROCprof Trace Decoder and thread trace APIs
======================================================

Thread trace is a profiling method that provides fine-grained insight into GPU kernel execution by collecting detailed traces of shader instructions executed by the GPU. This feature captures GPU occupancy, instruction execution times, fast performance counters, and other detailed performance data. Thread trace utilizes GPU hardware instrumentation to record events as they happen, resulting in precise timing information about wave (threads) execution behavior.

ROCprofiler-SDK provides wrapper APIs for the ROCprof Trace Decoder, a library to decode thread trace data.

.. note::

    Thread trace can generate large amounts of data, especially when profiling complex applications or longer execution runs. This might require handling potentially high volumes of trace data, so it’s recommended to implement appropriate filtering strategies to focus on the specific parts of interest in your application.

.. note::

    ROCprof Trace Decoder is a binary-only library and can be found `here <https://github.com/ROCm/rocprof-trace-decoder/releases>`_.

Thread trace service API
------------------------------------

This section describes how to use the ROCprofiler-SDK thread trace API to configure and use the thread trace service. For fully functional examples, see `Samples <https://github.com/ROCm/rocprofiler-sdk/tree/amd-mainline/samples/thread_trace>`_.

tool_init() setup
++++++++++++++++++

Here are the steps to set up ``tool_init()`` for thread trace:

1. Configure callback tracing for code objects to get disassembly information:


.. code-block:: cpp

    rocprofiler_context_id_t ctx{0};
    ROCPROFILER_CALL(rocprofiler_create_context(&ctx), "context creation failed");

    ROCPROFILER_CALL(
        rocprofiler_configure_callback_tracing_service(ctx,
                                                       ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                       nullptr,
                                                       0,
                                                       tool_codeobj_tracing_callback,
                                                       nullptr),
        "code object tracing service configure");

2. The thread trace service is tied to a GPU agent. To extract the list of available agents, use the ``rocprofiler_query_available_agents`` function:

.. code-block:: cpp

    std::vector<rocprofiler_agent_id_t> agents{};

    ROCPROFILER_CALL(
        rocprofiler_query_available_agents(
            ROCPROFILER_AGENT_INFO_VERSION_0,
            [](rocprofiler_agent_version_t, const void** _agents, size_t _num_agents, void* _data) {
                auto* agent_v = static_cast<std::vector<rocprofiler_agent_id_t>*>(_data);
                for(size_t i = 0; i < _num_agents; ++i)
                {
                    auto* agent = static_cast<const rocprofiler_agent_v0_t*>(_agents[i]);
                    if(agent->type == ROCPROFILER_AGENT_TYPE_GPU) 
                        agent_v->emplace_back(agent->id);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            sizeof(rocprofiler_agent_v0_t),
            &agents),
        "Failed to iterate agents");

3. Optionally, specify the configuration parameters:

.. code-block:: cpp

    std::vector<rocprofiler_thread_trace_parameter_t> params{};
    
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK, 0xF});
    
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU, 0});
    
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT, 0xF});
    
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE, 1u<<30}); // 1 GB
    
The configuration parameters are described here:

- ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK: Configures the Shader Engine (SE) mask, which determines the SEs to be traced. This is a bitmask where each bit corresponds to a SE. For MI3xx, each hex digit corresponds to an XCD. It's highly recommended to trace only one SE at a time to avoid data loss.

- ROCPROFILER_THREAD_TRACE_PARAMETER_TARGET_CU: Configures the target Compute Unit (CU) or WGP. Instruction tracing can only operate on a single CU or WGP at a time. The same target is used for all SEs in ``ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK``.
  
- ROCPROFILER_THREAD_TRACE_PARAMETER_SIMD_SELECT: Configures SIMD selection. For gfx9, this is a bitmask where each bit corresponds to a SIMD lane. For example, 0xF selects all SIMD lanes in the ``target_cu``. For gfx10, gfx11, and gfx12, this selects a single SIMD ID to trace. Results are taken mod4 for compatibility with gfx9 so 0xF selects SIMD3 of the target WGP.

- ROCPROFILER_THREAD_TRACE_PARAMETER_BUFFER_SIZE: Configures the buffer size. This buffer is shared among all SEs specified in ROCPROFILER_THREAD_TRACE_PARAMETER_SHADER_ENGINE_MASK. There is a minimal side effect to specifying a larger buffer size, except for increased VRAM usage.


The thread trace can be configured in two primary modes: device-wide or per-dispatch, as described in the following sections.

Device thread trace
+++++++++++++++++++

To enable thread trace service asynchronously or independently of kernel dispatches on a device, use:

.. code-block:: cpp

    // For device thread trace, it's recommended to create a separate context just to enable and disable the service independently.
    for(auto agent_id : agents)
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_device_thread_trace_service(
                ctx,
                agent_id,
                params.data(),
                params.size(),
                shader_data_callback,
                nullptr),
            "thread trace service configure");
    }

Dispatch thread trace
+++++++++++++++++++++

To enable selective thread trace based on specific kernel dispatches, use the dispatch-based configuration. By default, only the traced kernels are serialized in dispatch tracing. An optional parameter is provided to serialize all kernels, which ensures no parallel kernel execution during tracing.

.. code-block:: cpp

    // (Optional) To serialize ALL kernels, not just the traced ones
    // This ensures no parallel kernel execution during tracing
    params.push_back({ROCPROFILER_THREAD_TRACE_PARAMETER_SERIALIZE_ALL, 1});

    // Define dispatch callback to control thread trace
    rocprofiler_thread_trace_control_flags_t
    dispatch_callback(rocprofiler_agent_id_t agent_id,
                      rocprofiler_queue_id_t queue_id,
                      rocprofiler_async_correlation_id_t correlation_id,
                      rocprofiler_kernel_id_t kernel_id,
                      rocprofiler_dispatch_id_t dispatch_id,
                      void* userdata,
                      rocprofiler_user_data_t* dispatch_userdata)
    {
        // Trace only the desired kernels
        if(target_kernel_id == kernel_id) 
            return ROCPROFILER_THREAD_TRACE_CONTROL_START_AND_STOP;
            
        return ROCPROFILER_THREAD_TRACE_CONTROL_NONE;
    }
    
    // Configure dispatch-based thread trace
    for(auto agent_id : agents)
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_dispatch_thread_trace_service(
                ctx,
                agent_id,
                params.data(),
                params.size(),
                dispatch_callback,
                shader_data_callback,
                nullptr),
            "thread trace service configure");
    }

For device-wide thread trace, starting the context automatically begins data capture. Some application warmup is recommended before starting the device thread trace. For the dispatch thread trace, this step is not necessary as tracing doesn't start automatically.

To start the context after all services are configured, use:

.. code-block:: cpp

    auto status = rocprofiler_start_context(ctx);

    // Run your application workload here.
    
To stop the context to end data collection for device-wide thread trace, use:

.. code-block:: cpp

    status = rocprofiler_stop_context(ctx);

ROCprof Trace Decoder API
--------------------------------

The thread trace functionality requires you to install the ROCprof Trace Decoder package separately. This package provides the necessary decoder library for processing thread trace data. Ensure to install this package on your system before using the thread trace feature.

Trace Decoder setup
++++++++++++++

To decode the raw thread trace data, create and initialize a Trace Decoder:

.. code-block:: cpp

    rocprofiler_thread_trace_decoder_handle_t decoder{};
    
    // Create the Trace Decoder with the path to the decoder library
    ROCPROFILER_CALL(
        rocprofiler_thread_trace_decoder_create(&decoder, "/opt/rocm/lib"),
        "thread trace decoder creation");

    // Adds code object load information, reported by the code object tracing service
    ROCPROFILER_CALL(rocprofiler_thread_trace_decoder_codeobj_load(decoder,
                                                                   code_object_id,
                                                                   load_delta,
                                                                   load_size,
                                                                   data,
                                                                   datasize),
                    "code object load");

Code object tracking
++++++++++++++++++++

To properly decode instruction addresses, track the code object information:

.. code-block:: cpp

    void
    tool_codeobj_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                  rocprofiler_user_data_t* /* user_data */,
                                  void* /* userdata */)
    {
        if(record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT ||
           record.operation != ROCPROFILER_CODE_OBJECT_LOAD)
            return;

        // Optionally, ROCPROFILER_CALLBACK_PHASE_UNLOAD can be handled by calling
        // rocprofiler_thread_trace_decoder_codeobj_unload(decoder, data->code_object_id);
        if(record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD) return;

        auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);
        // TODO: Handle file storage types
        if(data->storage_type == ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE) return;

        auto* memorybase = reinterpret_cast<const void*>(data->memory_base);
        
        // Register code object with Trace Decoder
        ROCPROFILER_CALL(
            rocprofiler_thread_trace_decoder_codeobj_load(
                decoder,
                data->code_object_id,
                data->load_delta,
                data->load_size,
                memorybase,
                data->memory_size),
            "code object loading to decoder");
    }

Processing thread trace data
----------------------------

.. note::

    In the provided samples, thread trace data is processed immediately within the shader data callbacks for simplicity. In practice, it's recommended to save the data to a file or buffer and process it after the application completes. The rate at which thread trace generates data tends to be higher (GB/s) than the rate at which it can be processed (MB/s). Deferred processing is strongly recommended to avoid performance bottlenecks.

The thread trace service asynchronously delivers raw trace data via a dedicated callback ``shader_data_callback``. This data must be processed using the Trace Decoder to generate useful information:

.. code-block:: cpp

    void
    shader_data_callback(rocprofiler_agent_id_t agent,
                         int64_t shader_engine_id,
                         void* data,
                         size_t data_size,
                         rocprofiler_user_data_t userdata)
    {
        // Process shader callback data using the Trace Decoder.
        auto status = rocprofiler_trace_decode(decoder_handle,
                                               trace_decoder_callback,
                                               data,
                                               data_size,
                                               userdata);
    }

Decoder callback
++++++++++++++++

The trace decoder provides decoded information through a callback:

.. code-block:: cpp

    // Callback for decoded thread trace data
    void
    trace_decoder_callback(rocprofiler_thread_trace_decoder_record_type_t record_type,
                           void* trace_events,
                           uint64_t trace_size,
                           void* userdata)
    {
        switch(record_type)
        {
            case ROCPROFILER_THREAD_TRACE_DECODER_RECORD_WAVE:
            {
                // Process wave information
                auto* waves = static_cast<rocprofiler_thread_trace_decoder_wave_t*>(trace_events);
                for(uint64_t i = 0; i < trace_size; ++i)
                {
                    // Process wave data (timeline, instruction execution, etc.)
                }
                break;
            }
            
            // Handle other record types as needed
        }
    }

Trace Decoder info events
++++++++++++++++++

The Trace Decoder provides important information about the quality and comprehensiveness of the trace data through ``ROCPROFILER_THREAD_TRACE_DECODER_RECORD_INFO`` events. It is important to handle these events to understand potential issues with your trace data:

- ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST

  This event indicates that part of the trace data was dropped either due to hardware bandwidth limitations or buffer overflows. Receiving this event implies that portions of your trace might be missing or unreliable, which can affect the accuracy of any analysis based on the trace data.
   
  **Possible causes:**

  - The trace buffer size was too small for the workload

  - Memory bandwidth was exceeded
   
  **Recommended actions:**

  - Increase buffer sizes if possible

  - Reduce the number of SEs or SIMD lanes being traced

  - Disable ``ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTER`` or increase ``ROCPROFILER_THREAD_TRACE_PARAMETER_PERFCOUNTERS_CTRL`` if enabled



- ROCPROFILER_THREAD_TRACE_DECODER_INFO_STITCH_INCOMPLETE

  This event indicates that the Trace Decoder was unable to find the PC (Program Counter) address for one or more  traced instructions. Affected instructions will have their "pc" field set to zero.
   
  **Possible causes:**

  - The trace was started in the middle of a kernel execution:

    - If the trace was started after the kernel execution began, the Trace Decoder might not have received the necessary context to find the PC for all instructions.

    - Subsequent dispatches function normally.

  - Missing code object registration

  - Runtime kernels present in the trace: These are not always reported in the code object tracing callbacks

  - The ``ROCPROFILER_THREAD_TRACE_DECODER_INFO_DATA_LOST`` event was triggered. If parts of the trace were missing, important information might not have been available to the decoder.

  - There is a possible bug in the Trace Decoder. If you suspect this, report it to the ROCprofiler team.

For more information about the data structures and functions available for thread trace decoding, see the following headers:

- `trace_decoder.h <https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder.h>`_

- `trace_decoder_types.h <https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/trace_decoder_types.h>`_

- `core.h <https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/core.h>`_

- `dispatch.h <https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/dispatch.h>`_

- `agent.h <https://github.com/ROCm/rocprofiler-sdk/blob/amd-mainline/source/include/rocprofiler-sdk/experimental/thread-trace/agent.h>`_
