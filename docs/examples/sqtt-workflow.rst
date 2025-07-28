.. meta::
  :description: A typical workflow for collecting detailed instruction-level traces
  :keywords: AQLProfile, ROCm, API, how-to, SQTT

*******************************
SQ Thread Trace (SQTT) workflow
*******************************

The SQ Thread Trace workflow focuses on collecting detailed instruction-level traces. 
This workflow relies on creating a profile object, generating command packets, and iterating over output buffers: 

1. **Intercept the kernel dispatch**: The SDK intercepts the kernel dispatch.
2. **Create a SQTT profile object**: A profile object is created for SQTT, specifying trace parameters and output buffers.
3. **Generate SQTT command packets**: Start, stop, and read packets for SQTT are generated and injected into the queue.
4. **Submit packets and run the kernel**: The kernel and SQTT packets are submitted for execution.
5. **Collect the trace buffer**: The trace output buffer is collected after execution.
6. **Iterate and decode trace data**: The SDK iterates over the trace buffer and decodes the SQTT data for analysis.

The SDK abstracts queue interception and packet management so tool developers can focus on results.

Key API code snippets
=====================

These API snippets use the legacy interfaces from ``hsa_ven_amd_aqlprofile.h``. These are provided for understanding purposes only.  
For new development, refer to the updated APIs in ``aql_profile_v2.h``.  

In the ``rocprofiler-sdk`` codebase, these APIs are wrapped and orchestrated in the ``aql``, ``hsa``, and ``thread_trace`` folders for queue interception, packet construction, and result iteration.

.. note::

    The ``rocprofiler-sdk`` is migrating to these newer interfaces; the old APIs may be deprecated in future releases.

Define parameters and profile
------------------------------

.. code:: cpp

    hsa_ven_amd_aqlprofile_parameter_t params[] = {
        { HSA_VEN_AMD_AQLPROFILE_PARAMETER_NAME_ATT_BUFFER_SIZE, 16 } // 16 MB buffer
    };

    hsa_ven_amd_aqlprofile_profile_t profile = {
        .agent = agent,
        .type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_TRACE,
        .events = nullptr,
        .event_count = 0,
        .parameters = params,
        .parameter_count = sizeof(params)/sizeof(params[0]),
        .output_buffer = {trace_ptr, trace_size},
        .command_buffer = {cmd_ptr, cmd_size}
    };


Generate SQTT start/stop packets
---------------------------------

.. code:: cpp

    hsa_ext_amd_aql_pm4_packet_t sqtt_start_pkt, sqtt_stop_pkt;
    hsa_ven_amd_aqlprofile_start(&profile, &sqtt_start_pkt);
    hsa_ven_amd_aqlprofile_stop(&profile, &sqtt_stop_pkt);


Submit packets and run the kernel
---------------------------------

.. code:: cpp

    queue->Submit(&sqtt_start_pkt);
    queue->Submit(&kernel_pkt);
    queue->Submit(&sqtt_stop_pkt);


Iterate and decode trace data
-----------------------------

.. code:: cpp

    hsa_ven_amd_aqlprofile_iterate_data(
        &profile,
        [](hsa_ven_amd_aqlprofile_info_type_t info_type,
        hsa_ven_amd_aqlprofile_info_data_t* info_data,
        void* user_data) -> hsa_status_t {
            if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_TRACE_DATA) {
                // info_data->trace_data.ptr, info_data->trace_data.size
                decode_trace(info_data->trace_data.ptr, info_data->trace_data.size);
            }
            return HSA_STATUS_SUCCESS;
        },
        nullptr
    );


