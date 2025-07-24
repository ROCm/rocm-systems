AQLprofile APIs
===============

Learn about the typical APIs used in AQLprofile.

The APIs in ``hsa_ven_amd_aqlprofile.h`` are used by legacy tools such
as ``rocprof`` and ``rocprofv2``. These APIs may be deprecated in the
future as development focus shifts to the new ``aqlprofile_v2.h`` APIs.

The APIs in ``aqlprofile_v2.h`` are designed for use with
``rocprofiler-sdk``, and are actively maintained and recommended for all
new development.

From header ``aql_profile_v2.h``
--------------------------------

+--------------------+-------------------------------------------------+
| API Name           | Purpose                                         |
+====================+=================================================+
| ``aqlprofil        | Registers an agent for profiling using basic    |
| e_register_agent`` | agent info.                                     |
+--------------------+-------------------------------------------------+
| ``aqlprofile_reg   | Registers an agent for profiling using extended |
| ister_agent_info`` | agent info and versioning.                      |
+--------------------+-------------------------------------------------+
| ``aqlprof          | Retrieves information about PMC profiles (e.g., |
| ile_get_pmc_info`` | buffer sizes, counter data).                    |
+--------------------+-------------------------------------------------+
| ``aqlprofile_va    | Checks if a given PMC event is valid for the    |
| lidate_pmc_event`` | specified agent.                                |
+--------------------+-------------------------------------------------+
| ``aqlprofile_pm    | Creates AQL packets (start, stop, read) for PMC |
| c_create_packets`` | profiling and returns a handle.                 |
+--------------------+-------------------------------------------------+
| ``aqlprofile_pm    | Deletes PMC profiling packets and releases      |
| c_delete_packets`` | associated resources.                           |
+--------------------+-------------------------------------------------+
| ``aqlprofile_      | Iterates over PMC profiling results using a     |
| pmc_iterate_data`` | callback.                                       |
+--------------------+-------------------------------------------------+
| ``aqlprofile_at    | Creates AQL packets (start, stop) for Advanced  |
| t_create_packets`` | Thread Trace (SQTT) and returns a handle.       |
+--------------------+-------------------------------------------------+
| ``aqlprofile_at    | Deletes ATT profiling packets and releases      |
| t_delete_packets`` | associated resources.                           |
+--------------------+-------------------------------------------------+
| ``aqlprofile_      | Iterates over thread trace (SQTT) results using |
| att_iterate_data`` | a callback.                                     |
+--------------------+-------------------------------------------------+
| ``aqlprofile_i     | Iterates over all possible event coordinate IDs |
| terate_event_ids`` | and names using a callback.                     |
+--------------------+-------------------------------------------------+
| ``aqlprofile_ite   | Iterates over all event coordinates for a given |
| rate_event_coord`` | agent and event using a callback.               |
+--------------------+-------------------------------------------------+
| ``aqlprofile_at    | Creates a marker packet for code object events  |
| t_codeobj_marker`` | in thread trace workflows.                      |
+--------------------+-------------------------------------------------+

Callback Typedefs
~~~~~~~~~~~~~~~~~

+---------------------+------------------------------------------------+
| Callback Typedef    | Purpose                                        |
| Name                |                                                |
+=====================+================================================+
| ``aqlprofile_memory | Callback for allocating memory buffers for     |
| _alloc_callback_t`` | profiles (PMC/ATT).                            |
+---------------------+------------------------------------------------+
|``aqlprofile_memory  | Callback for deallocating memory buffers       |
|_dealloc_callback_t``| allocated for profiles.                        |
+---------------------+------------------------------------------------+
| ``aqlprof           | Callback for copying memory (used internally   |
| ile_memory_copy_t`` | by the profiler).                              |
+---------------------+------------------------------------------------+
| ``aqlprofile_pm     | Used with ``aqlprofile_pmc_iterate_data`` to   |
| c_data_callback_t`` | process each PMC profiling result.             |
+---------------------+------------------------------------------------+
| ``aqlprofile_at     | Used with ``aqlprofile_att_iterate_data`` to   |
| t_data_callback_t`` | process each thread trace (SQTT) result.       |
+---------------------+------------------------------------------------+
| ``aqlprofile_eve    | Used with ``aqlprofile_iterate_event_ids`` to  |
| ntname_callback_t`` | process event coordinate IDs and names.        |
+---------------------+------------------------------------------------+
| ``aqlprofile_coor   | Used with ``aqlprofile_iterate_event_coord``   |
| dinate_callback_t`` | to process event coordinate information.       |
+---------------------+------------------------------------------------+

From header ``hsa_ven_amd_aqlprofile.h``
----------------------------------------

+----------------------+-----------------------------------------------+
| API Name             | Purpose                                       |
+======================+===============================================+
| ``hsa_ven_amd_aqlprof| Checks if a given event (counter) is valid    |
| ile_validate_event`` | for the specified GPU agent.                  |
+----------------------+-----------------------------------------------+
| ``hsa_ven_am         | Populates an AQL packet with commands to      |
| d_aqlprofile_start`` | start profiling (PMC or SQTT).                |
+----------------------+-----------------------------------------------+
| ``hsa_ven_a          | Populates an AQL packet with commands to stop |
| md_aqlprofile_stop`` | profiling.                                    |
+----------------------+-----------------------------------------------+
| ``hsa_ven_a          | Populates an AQL packet with commands to read |
| md_aqlprofile_read`` | profiling results from the GPU.               |
+----------------------+-----------------------------------------------+
| ``hsa_ven_amd_aqlprof| Converts an AQL packet to a PM4 packet blob   |
| ile_legacy_get_pm4`` | (for legacy devices).                         |
+----------------------+-----------------------------------------------+
| ``hsa_ven_amd_aql    | Inserts a marker (correlation ID) into the    |
| profile_att_marker`` | ATT (thread trace) buffer.                    |
+----------------------+-----------------------------------------------+
| ``hsa_ven_amd_a      | Retrieves various profile information, such   |
| qlprofile_get_info`` | as buffer sizes or collected data.            |
+----------------------+-----------------------------------------------+
| ``hsa_ven_amd_aqlpr  | Iterates over the profiling output data (PMC  |
| ofile_iterate_data`` | results or SQTT trace) using a callback.      |
+----------------------+-----------------------------------------------+
| ``hsa_ven_amd_aqlpr  | Returns a human-readable error string for the |
| ofile_error_string`` | last error.                                   |
+----------------------+-----------------------------------------------+
| ``hs                 | Iterates over all possible event IDs and      |
| a_ven_amd_aqlprofile | names for the agent.                          |
| _iterate_event_ids`` |                                               |
+----------------------+-----------------------------------------------+
| ``hsa_               | Iterates over all event coordinates for a     |
| ven_amd_aqlprofile_i | given agent and event.                        |
| terate_event_coord`` |                                               |
+----------------------+-----------------------------------------------+

.. _callback-typedefs-1:

Callback Typedefs
~~~~~~~~~~~~~~~~~

+-----------------------+-----------------------------------------------+
| Callback Typedef Name | Purpose                                       |
+=======================+===============================================+
| ``hsa_ven_amd_aqlprof | Used with                                     |
| ile_data_callback_t`` | ``hsa_ven_amd_aqlprofile_iterate_data`` to    |
|                       | process each profiling result (PMC/SQTT).     |
+-----------------------+-----------------------------------------------+
| ``hsa                 | Used with                                     |
| _ven_amd_aqlprofile_e | ``hsa_ven_amd_aqlprofile_iterate_event_ids``  |
| ventname_callback_t`` | to process event IDs and names.               |
+-----------------------+-----------------------------------------------+
| ``hsa                 | Used with                                     |
| _ven_amd_aqlprofile_c | ``hsa_ven_amd_aqlprofile_iterate_event_coord``|
| oordinate_callback_t``| to process event coordinate info.             |
+-----------------------+-----------------------------------------------+
