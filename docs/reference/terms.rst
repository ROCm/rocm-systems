Terms
=====

Agents
------

Agents represent computational devices (CPUs, GPUs) in the Heterogeneous
System Architecture (HSA) runtime. In AQLprofile, agents are discovered
via HSA APIs and encapsulated in the ``AgentInfo`` structure. Each agent
contains metadata such as device type, name, compute unit count, and
memory pools.

Agents are enumerated using HSA API ``hsa_iterate_agents``, and their
properties are queried via another HSA API, ``hsa_agent_get_info``.
Agents are used to target specific GPUs for profiling, and to allocate
resources such as command buffers and memory pools.

Counters and events
-------------------

Performance counters are special circuits on the hardware that count
specific GPU events (e.g., cycles, instructions, cache hits). Events
specify which counters to collect, identified by block name, block
index, and counter ID.

-  Events are described using ``hsa_ven_amd_aqlprofile_event_t``
   structures.
-  Events are grouped into profiles and collected during profiling
   sessions.

.. code:: cpp

   const hsa_ven_amd_aqlprofile_event_t events_arr1[] = {
       {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 2 /*CYCLES*/},
       {HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ, 0, 3 /*BUSY_CYCLES*/},
       // ...
   };

Counter blocks
--------------

Counter blocks correspond to hardware units on the GPU (e.g., SQ, TCC,
TCP). Each block exposes a set of counters/events.

-  Block names (e.g., ``HSA_VEN_AMD_AQLPROFILE_BLOCK_NAME_SQ``) map to
   specific hardware blocks.
-  Events specify both the block and the counter within that block.

Command packets
---------------

Command packets are AQL or PM4 packets that encode profiling commands
for the GPU. They're constructed and written into command buffers.

They're built using AQLprofile APIs or helper functions and submitted to
the GPU via HSA queues.

.. code:: cpp

   bool Queue::Submit(hsa_ext_amd_aql_pm4_packet_t* packet) {
       // Write packet to queue and signal doorbell
   }

Profile object
--------------

The profile object encapsulates all information required to perform a
profiling session. It's represented by the
``hsa_ven_amd_aqlprofile_profile_t`` struct, which includes the agent,
event type, list of events, command buffer, and additional parameters.

Profile objects are constructed by specifying the agent, event type
(PMC, SQTT), events to collect, and associated buffers. They're passed
to AQLprofile APIs to start, stop, and read profiling data.

.. code:: cpp

   hsa_ven_amd_aqlprofile_profile_t *profile =
       new hsa_ven_amd_aqlprofile_profile_t{
           agent_info->dev_id,
           HSA_VEN_AMD_AQLPROFILE_EVENT_TYPE_PMC,
           events,
           num_events,
           NULL,
           0,
           0,
           0};

Command buffers
---------------

Command buffers are memory regions that store AQL packets and PM4
commands which control GPU profiling operations. They're allocated per
agent, and must meet alignment and size requirements dictated by the
hardware.

Output buffer
-------------

Output buffers are memory regions that store outputs such as counter
values and thread trace tokens. They're allocated using HSA memory pools
associated with the agent.
