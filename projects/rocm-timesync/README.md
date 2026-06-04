# ROCm Timesync

ROCm Timesync is an open source library that support precision time
synchronization on AMD platforms. It works in conjunction with
[rocr-runtime](../rocr-runtime) to support translation of time from HSA agents
to a common global system timeline.

A goal is to support alignment of timestamps across HSA agents with precision
targets of 1 µs or less, in systems where appropriate HW mechanisms exist to
allow this (e.g., PTP/PTM)

## PTP background

TODO

## Overview

The goal of this project is to support alignment of timnestamps generated from different time domains onto a single
unified timeline with high precision (in the range of 10ns - 1us).

The basic architecture for MI4xx generation systems is to take raw timestamps produced by the GPU and translate them
onto the system's POSIX `CLOCK_REALTIME` timeline. A topic for future product generations is to consider adding a
programmable real-time clock to the GPU which can then surface REALTIME aligned timestamps directly into SW, but this is 
out of scope for MI4xx, and thus out of scope for this project at the moment.

The primary challenge with performing GPU->realtime timestamp translation is that the adjust factor is not constant.
This is owing to several reasons:
- On dGPU configurations, CPU and GPU clocks are expected to drift independently
- On PTP enabled systems, the node's realtime clock does not tick based solely on CPU, but rather is disciplined by a
  network clocksource.

For this reason, our approach is constantly sample synchronized GPU/realtime timestamps ("crosststamps") and use them to perform
translations between clock domains when needed.

A notional workflow we need to support is something like this:
1. User runs a PTP-enabled workload via rocprof
2. The spawned ROCR instance ensures that a dedicated thread or threads are established to do the following:
    - Collect crosststamps from KFD at some high frequency needed to support PTP-level precision (e.g., 100Hz)
    - Store crosststamps in a persistent storage of some form (e.g., a time-series database or table). The storage must
      support subsequent querying for timestamp translation.
    - **Note: whether these operations are done by the same or separate threads, whether those threads are part of the
      ROCr instance or a separate system daemon(s), and how persistent storage is managed are design considerations we
      will elaborate on.**
3. Kernel dispatch/completion events produced by the workload, which include raw GPU timestamps, are surfaced into rocprof
5. rocprof calls into ROCR through HSA API to convert these raw GPU timestamps to the system/realtime timeline (e.g.,
via something like
[hsa_amd_profiling_convert_tick_to_system_domain](https://github.com/ROCm/rocm-systems/blob/users/bkocolos/precision-time/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L999)
6. ROCR's implementation of this API vectors into ROCm timesync, which queries the persistent timestamp storage backend
with the provided GPU timestamp to produce the translation offset to realtime.

## Architecture

There are multiple ways to design this system. Some major questions to answer include:
- Which thread is responsible for producing crosststamps?
- Which thread is responsbile for storing crosststamps in persistent storage?
- Are these threads part of the same process or different processes?
- Are these threads part of the ROCR process or a dedicated ROCM timesync process(es)?
- What policy (or policies) are in place regarding regarding sampling rates and long-term persistence of timestamp data?
  Are these configurable or static?

We discuss two different models that land at different points in this design space

### In-process

The simplest design is to simply extend ROCr with mechanisms to measure, store, and query crosststamps. The diagram
below illustrates the process

![](doc/img/in-process.png)

This design has the following properties:
- 

### Out-of-process production, in-process storage

### Out-of-process

Production done by dedicated thread, ROCr consumes and stores in a shared instance  

![](doc/img/out-of-process.png)

In contrast to an in-process design, an out-of-process design removes the need
for every timesync client to store timestamp data. The benefit of this model is
that each process does not need to attach to an lttng stream and marshal
data to persistent storage. Timesync data for PTP is not process specific,
thus there is no need for each process to maintain its own copy.

#### Consumer API

The downside of an out-of-process design is that it requires each ROCR instance
to communicate via explicit IPC with the ROCm-timesync service to get timesync
data. We envision an API with at least the following API functions

1. `query_timesync_freq(hsa_agent_id_t agent, uint32_t *hz int *num_hz)`
    - return to the caller a set of streaming frequencies, in Hz, supported by ROCR-timesync for agent `agent` on this system

2. `enable_timesync_freq(hsa_agent_id_t agent, uint32_t freq)`
    - enable timestamp generation at `freq` Hz for agent `agent`

    - **Notes**
        - If not already running, deploys new thread which attaches to the *lttng* data stream and marshals data into the persistent datastore (e.g., *InfluxDB*)
        - If already running, update a counter indicating the presence of an additional active consumer

3. `disable_timesync_freq(hsa_agent_id_t agent, uint32_t freq)`
    - disable timestamp generation at `freq` Hz for agent `agent`

    - **Notes**
        - Decrement count of active consumers for *{agent, freq} lttng stream*.
            - If this count reaches 0, thread can detach from *lttng* data stream.
        - Decrement count of total active consumers across *all lttng streams*.
            - If this count reached 0, the backend storage can be deleted, pruned, etc
                - Exact decision is according to policy; the upshot is that such storage is not needed for ROCR time translation as there are no active ROCR instances.

4. `attach_stream(hsa_agent_id_t agent, uint32_t freq)`
    - Exports raw *{agent, freq} lttng stream* for external consumption (e.g., debugging or piping to a customer datastore)

5. `translate_time_to_system(hsa_agent_id_t agent, uint64_t agent_timestamp)`
    - Translates 'timestamp' from 'agent' domain to system (PTP) domain by querying the internal backend data store
