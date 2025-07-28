.. meta::
  :description: A typical workflow for collecting PMC data
  :keywords: AQLProfile, ROCm, API, how-to, PMC 

******************************************
Performance Monitor Control (PMC) workflow
******************************************

This page describes a typical workflow for collecting PMC data using AQLProfile (as integrated in ``rocprofiler-sdk``). 
This workflow relies on creating a profile object, generating command packets, and iterating over output buffers:

1. **Intercept kernel dispatch**: The SDK intercepts kernel dispatch packets submitted to the GPU queue.
2. **Create a profile object**: A profile/session object is created, specifying the agent (GPU), events (counters), and output buffers.
3. **Generate command packets**: Start, stop, and read command packets are generated and injected into the queue around the kernel dispatch.
4. **Submit packets and run the kernel**: The kernel and profiling packets are submitted to the GPU queue for execution.
5. **Collect the output buffer**: After execution, the output buffer is read back from the GPU.
6. **Iterate and extract the results**: The SDK iterates over the output buffer to extract and report counter results.

The SDK abstracts queue interception and packet management so tool developers can focus on results.

Key API code snippets
=====================

These API snippets use the legacy interfaces from ``hsa_ven_amd_aqlprofile.h``. These are provided for understanding purposes only.  
For new development, refer to the updated APIs in ``aql_profile_v2.h``.

.. note::

   The ``rocprofiler-sdk`` is migrating to these newer interfaces; the old APIs may be deprecated in future releases.

Define the events and profile
-----------------------------

.. code:: cpp

   // Select events (counters) to collect
   hsa_ven_amd_aqlprofile_event_t events[] = {
      { HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 }, // Example: SQ block, instance 0, counter 2
      { HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 }
   };

   // Create profile object
   hsa_ven_amd_aqlprofile_profile_t profile = {
      .agent = agent, // hsa_agent_t
      .type = HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
      .events = events,
      .event_count = sizeof(events)/sizeof(events[0]),
      .parameters = nullptr,
      .parameter_count = 0,
      .output_buffer = {output_ptr, output_size},
      .command_buffer = {cmd_ptr, cmd_size}
   };


Validate events
---------------

.. code:: cpp

   bool valid = false;
   hsa_ven_amd_aqlprofile_validate_event(agent, &events[0], &valid);
   if (!valid) {
      // Handle invalid event
   }


Generate command packets
-------------------------

.. code:: cpp

   hsa_ext_amd_aql_pm4_packet_t start_pkt, stop_pkt, read_pkt;
   hsa_ven_amd_aqlprofile_start(&profile, &start_pkt);
   hsa_ven_amd_aqlprofile_stop(&profile, &stop_pkt);
   hsa_ven_amd_aqlprofile_read(&profile, &read_pkt);


Submit packets and run the kernel
---------------------------------

.. code:: cpp

   // Pseudocode: inject packets into HSA queue
   queue->Submit(&start_pkt);
   queue->Submit(&kernel_pkt);
   queue->Submit(&stop_pkt);
   queue->Submit(&read_pkt);


Iterate and extract results
----------------------------

.. code:: cpp

   hsa_ven_amd_aqlprofile_iterate_data(
      &profile,
      [](hsa_ven_amd_aqlprofile_info_type_t info_type,
         hsa_ven_amd_aqlprofile_info_data_t* info_data,
         void* user_data) -> hsa_status_t {
         if (info_type == HSA_VEN_AMD_AQLPROFILE_INFO_PMC_DATA) {
               printf("Event: block %d, id %d, value: %llu\n",
                  info_data->pmc_data.event.block_name,
                  info_data->pmc_data.event.counter_id,
                  info_data->pmc_data.result);
         }
         return HSA_STATUS_SUCCESS;
      },
      nullptr
   );
