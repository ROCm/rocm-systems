# ROCR Timesync

ROCR Timesync is an open source library that support precision time
synchronization on AMD platforms. It works in conjunction with
[rocr-runtime](../rocr-runtime) to support translation of time from HSA agents
to a common global system timeline.

A goal is to support alignment of timestamps across HSA agents with precision
targets of 1 µs or less, in systems where appropriate HW mechanisms exist to
allow this (e.g., PTP/PTM)

### Background

TODO

## Overview

## Architecture

### In-process

![](doc/img/in-process.png)

### Out-of-process

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

Notes:
- If not already running, deploys new thread which attaches to the *lttng* data stream and marshals data into the persistent datastore (e.g., *InfluxDB*)
- If already running, update a counter indicating the presence of an additional active consumer

3. `disable_timesync_freq(hsa_agent_id_t agent, uint32_t freq)`
- disable timestamp generation at `freq` Hz for agent `agent`

Notes:
- Decrement count of active consumers for *{agent, freq} lttng stream*.
    - If this count reaches 0, thread can detach from *lttng* data stream.
- Decrement count of total active consumers across *all lttng streams*.
    - If this count reached 0, the backend storage can be deleted, pruned, etc
        - Exact decision is according to policy; the upshot is that such storage is not needed for ROCR time translation as there are no active ROCR instances.

4. attach_stream(hsa_agent_id_t agent, uint32_t freq)
- Exports raw *{agent, freq} lttng stream* for external consumption (e.g., debugging or piping to a customer datastore)

5. translate_time_to_system(hsa_agent_id_t agent, uint64_t agent_timestamp)
- Translates 'timestamp' from 'agent' domain to system (PTP) domain by querying the internal backend data store
