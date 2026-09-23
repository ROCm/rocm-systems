# ROCm Timesync Architectures

This page explores some of the key architecture questions for this project, such as:
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

On the other end of the spectrum is an "**out-of-process**" approach which is designed to address these inefficiencies.
The diagram below illustrates one such architecture.

![](doc/img/out-of-process.png)

The key steps:
- A standalone `ROCm-timesync` system service is deployed. It runs `rocm-timesync-d_m` which query KFD for
  (`CLOCK_REALTIME`, GPU `m`) crosststamps. We envision one thread per GPU on the system 
- These threads publish data streams through a tracing infrastructure such as [lttng](https://lttng.org/).
- On the ROCR side, `ROCR-timesync-r_m` consumes the data streams it needs (e.g., the GPUs its process is using),
  and stores this data in a shared TSDB.
- ROCR's implementation of `hsa_amd_profiling_tick_to_system_domain()` invokes a translation function provided by `ROCm
  timesync` -- i.e., `translate()` - which queries the TSDB and applies the offset.


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
- Data retention: the fact that data is now stored out of process means that it is not straightforward to reap old
  data. There must be some mechanism to tag on insertion with the corresponding consumer process(es), or absent that a
  downsampling process to gradually decrease and ultimately evict data as it ages.

