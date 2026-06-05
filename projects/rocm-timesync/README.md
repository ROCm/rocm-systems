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

The primary challenge with performing GPU->realtime timestamp translation is that the adjustment factor is not constant.
This is due to several reasons:
- On dGPU configurations, CPU and GPU clocks are expected to drift independently
- On PTP enabled systems, the node's realtime clock does not tick based solely on CPU, but rather is disciplined by a
  network clocksource.

For this reason, our approach is to continuously sample synchronized GPU/realtime timestamps ("crosststamps") at high
frequency, and use them to perform translations between clock domains when needed.

A notional workflow we need to support is something like this:
1. User runs an application under rocprof
2. A dedicated thread (or threads) is established to do the following:
    - Collect crosststamps from KFD at some high frequency needed to support PTP-level precision (e.g., 100Hz)
    - Store crosststamps in some form of storage (e.g., a time-series database or simple in-memory hashtable). The
      storage must support subsequent querying for timestamp translation.
    - **Note: whether these operations are done by the same or separate threads, whether those threads are part of the
      ROCR instance or a separate system daemon(s), and how storage is managed are design considerations we
      will elaborate on below.**
3. Kernel dispatch/completion events produced by the workload, which include raw GPU timestamps, are surfaced into rocprof
5. rocprof calls into ROCR through HSA API to convert these raw GPU timestamps to the system/realtime timeline (e.g.,
via something like
[hsa_amd_profiling_convert_tick_to_system_domain](https://github.com/ROCm/rocm-systems/blob/users/bkocolos/precision-time/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L999))
6. ROCR's implementation of this API vectors into ROCm timesync, which queries the persistent timestamp storage backend
with the provided GPU timestamp to produce the offset to translate to realtime.

## Architecture

Some key design questions include:
- Which thread is responsible for producing crosststamps?
- Which thread is responsbile for consuming and storing crosststamps?
- Are these threads part of the same process or different processes?
- Are these threads part of the ROCR process or a dedicated ROCM timesync process(es)?
- What policy (or policies) are in place regarding regarding sampling rates and long-term persistence of timestamp data?
  Are these configurable or static?

We discuss two different models that land at different points in this design space below.

### In-process

The simplest design is to simply extend ROCR with mechanisms to measure, store, and query crosststamps. We call this
"**in-process**" because everything is done through threads that are part of main host process. The diagram
below illustrates the architecture

![](doc/img/in-process.png)

The key steps:
- A new thread `ROCR-timesync-r_m` queries KFD for crosststamps of the form (`CLOCK_REALTIME` timestamp, GPU `m`
  timestamp). It may use the existing `AMDKFD_IOC_GET_CLOCK_COUNTERS` `ioctl()` call or some TBD similar new interface.
  It performs these queries at a frequency needed for the system/application's target precision. This can be
  communicated via a system-wide configuration or an application-specific configuration like an HSA env variable.
- This thread stores timestamps in its local memory using some searchable data-structure, possibly something as simple
  as an `std::map` mapping GPU timestamp to system timestamp
- ROCR's implementation of `hsa_amd_profiling_tick_to_system_domain()` invokes a translation function provided by `ROCm
  timesync` -- i.e., `translate()` - which queries this data structure and applies the offset.


#### Pros/Cons of in-process

Pros:
+ Simplicity: no new processes, standalone system daemons, or external SW dependencies are needed
- Data retention: data is resident in memory as long as the process is running. When a process completes, its timestamp
  data goes away

Cons:
- Space inefficient: every ROCR instance stores timestamp data leading to duplication (nothing about a crosststamp is
  process specific)
- Time inefficient: an `std::map()` is likely not going to perform insertions/queries as efficiently as a mature
  time-series database (TSDB)


### Out-of-process

On the other end of the spectrum is an "**out-of-process**" which is designed to address these inefficiencies. The
diagram below illustrates one such architecture.

![](doc/img/out-of-process.png)

The key steps:
- A standalone `ROCm-timesync` system service is deployed. It runs `rocm-timesync-d_m` which query KFD for
  (`CLOCK_REALTIME`, GPU `m`) crosststamps. We envision one thread per GPU on the system 
- These threads publish data streams througgh a tracing infrastructure such as [lttng](https://lttng.org/).
- On the ROCR side, `ROCR-timesync-r_m` consumes the data streams it needs (e.g., the GPUs its process is using),
  and stores this data in a shared TSDB.
- ROCR's implementation of `hsa_amd_profiling_tick_to_system_domain()` invokes a translation function provided by `ROCm
  timesync` -- i.e., `translate()` - which queries the TSDB and applies the offset.

#### Regarding lttng

`lttng` is a open-source tracing infrastructure in Linux. Using it in this way lets us decouple the producer
(`ROCM-timesync service`) from the consumer (`ROCR/librocr-timesync-consumer.so`). This has some nice benefits,
including:

+ Producer/consumer do not explicltiy communicate via an API. 

    - One alternative architecture involves consumers explicitly
    calling into producers via an API to configure things like sampling frequencies. The approach here is simpler: we just
    publish data streams at multiple frequencies and allow consumers to attach to the one(s) they need. For example, a
    PTP-enabled multi-node workload needing ~100ns precision for performance analysis may need sampling at a high frequency
    such as 100 Hz, while a client application can tolerate lower precision such as 1 Hz. 
    
        - We envision publishing `M * N` streams, where `M` is the number of GPUs on the node and `N` is the
          number of precisions/frequencies supported.

        - Importantly, `lttng` provides mechanisms for producers to determine at runtime if a stream has any active
          consumers. They do not need to call into KFD and produce if there are no active consumers for a given
          (`M`,`N`)


#### Pros/Cons of out-of-process

Pros:
+ Space efficiency: while each consumer does independently populate timestamp data to a shared TSDB, the fact that
consumers all consume data from a common set of lttng streams mean that the TSDB can perform deduplication (or simply
overwrite) if multiple consumers store the same data. TSDBs also have built-in mechanisms for compression, downsampling,
and other mechanisms that can drastically reduce the storage needed for timestamp data.
+ Time efficiency: A mature TSDB will have very fast insertion and query mechanisms for time-series data.
+ Producer/consumer can evolve independently

Cons:
- Simplicity: clearly this is not as simple as the in-process design; however, we strike some balance by using a
  fully API-less design and relying on a system service in `lttng` that will likely already be shipping with future
  versions of ROCm.
- Data retention: the fact that data is now stored out of process means that it is not straightforwards to reap old
  data. There must be some mechanism to tag on insertion with the corresponding consumer process(es), or absent that a
  downsampling process to gradually decrease and ultimately evict data as it ages.
