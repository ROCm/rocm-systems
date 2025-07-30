AQLprofile APIs
===============

Learn about the typical APIs used in AQLprofile.

The APIs in ``aqlprofile_v2.h`` are designed for use with ``rocprofiler-sdk``, and are actively maintained and recommended for all new development.

.. note::

    The APIs in ``hsa_ven_amd_aqlprofile.h`` are used by legacy tools such as ``rocprof`` and ``rocprofv2``. You should use the new ``aqlprofile_v2.h`` APIs instead.

From header ``aql_profile_v2.h``
--------------------------------

+------------------------------------+------------------------------------------------------------------------------------------+
| API Name                           | Purpose                                                                                  |
+====================================+==========================================================================================+
| ``aqlprofile_register_agent``      | Registers an agent for profiling using basic agent info.                                 |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_register_agent_info`` | Registers an agent for profiling using extended agent info and versioning.               |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_get_pmc_info``        | Retrieves information about PMC profiles (e.g., buffer sizes, counter data).             |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_validate_pmc_event``  | Checks if a given PMC event is valid for the specified agent.                            |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_pmc_create_packets``  | Creates AQL packets (start, stop, read) for PMC profiling and returns a handle.          |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_pmc_delete_packets``  | Deletes PMC profiling packets and releases associated resources.                         |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_pmc_iterate_data``    | Iterates over PMC profiling results using a callback.                                    |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_att_create_packets``  | Creates AQL packets (start, stop) for Advanced Thread Trace (SQTT) and returns a handle. |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_att_delete_packets``  | Deletes ATT profiling packets and releases associated resources.                         |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_att_iterate_data``    | Iterates over thread trace (SQTT) results using a callback.                              |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_iterate_event_ids``   | Iterates over all possible event coordinate IDs and names using a callback.              |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_iterate_event_coord`` | Iterates over all event coordinates for a given agent and event using a callback.        |
+------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_att_codeobj_marker``  | Creates a marker packet for code object events in thread trace workflows.                |
+------------------------------------+------------------------------------------------------------------------------------------+

Callback Typedefs
~~~~~~~~~~~~~~~~~

+------------------------------------------+------------------------------------------------------------------------------------------+
| Callback Typedef Name                    | Purpose                                                                                  |
+==========================================+==========================================================================================+
| ``aqlprofile_memory_alloc_callback_t``   | Callback for allocating memory buffers for profiles (PMC/ATT).                           |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_memory_dealloc_callback_t`` | Callback for deallocating memory buffers allocated for profiles.                         |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_memory_copy_t``             | Callback for copying memory (used internally by the profiler).                           |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_pmc_data_callback_t``       | Used with ``aqlprofile_pmc_iterate_data`` to process each PMC profiling result.          |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_att_data_callback_t``       | Used with ``aqlprofile_att_iterate_data`` to process each thread trace (SQTT) result.    |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_eventname_callback_t``      | Used with ``aqlprofile_iterate_event_ids`` to process event coordinate IDs and names.    |
+------------------------------------------+------------------------------------------------------------------------------------------+
| ``aqlprofile_coordinate_callback_t``     | Used with ``aqlprofile_iterate_event_coord`` to process event coordinate information.    |
+------------------------------------------+------------------------------------------------------------------------------------------+

From header ``hsa_ven_amd_aqlprofile.h`` (Legacy)
-------------------------------------------------

+--------------------------------------------------+------------------------------------------------------------------------------------------+
| API Name                                         | Purpose                                                                                  |
+==================================================+==========================================================================================+
| ``hsa_ven_amd_aqlprofile_validate_event``        | Checks if a given event (counter) is valid for the specified GPU agent.                  |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_start``                 | Populates an AQL packet with commands to start profiling (PMC or SQTT).                  |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_stop``                  | Populates an AQL packet with commands to stop profiling.                                 |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_read``                  | Populates an AQL packet with commands to read profiling results from the GPU.            |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_legacy_get_pm4``        | Converts an AQL packet to a PM4 packet blob (for legacy devices).                        |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_att_marker``            | Inserts a marker (correlation ID) into the ATT (thread trace) buffer.                    |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_get_info``              | Retrieves various profile information, such as buffer sizes or collected data.           |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_iterate_data``          | Iterates over the profiling output data (PMC results or SQTT trace) using a callback.    |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_error_string``          | Returns a human-readable error string for the last error.                                |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_iterate_event_ids``     | Iterates over all possible event IDs and names for the agent.                            |
+--------------------------------------------------+------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_iterate_event_coord``   | Iterates over all event coordinates for a given agent and event.                         |
+--------------------------------------------------+------------------------------------------------------------------------------------------+

.. _callback-typedefs-1:

Callback Typedefs
~~~~~~~~~~~~~~~~~

+---------------------------------------------------+------------------------------------------------------------------------------------------------+
| Callback Typedef Name                             | Purpose                                                                                        |
+===================================================+================================================================================================+
| ``hsa_ven_amd_aqlprofile_data_callback_t``        | Used with ``hsa_ven_amd_aqlprofile_iterate_data`` to process each profiling result (PMC/SQTT). |
+---------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_eventname_callback_t``   | Used with ``hsa_ven_amd_aqlprofile_iterate_event_ids`` to process event IDs and names.         |
+---------------------------------------------------+------------------------------------------------------------------------------------------------+
| ``hsa_ven_amd_aqlprofile_coordinate_callback_t``  | Used with ``hsa_ven_amd_aqlprofile_iterate_event_coord`` to process event coordinate info.     |
+---------------------------------------------------+------------------------------------------------------------------------------------------------+
