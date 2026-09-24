# ROCm Timesync

ROCm Timesync is an open source library for precision time synchronization on AMD platforms. It works in conjunction with [rocr-runtime](../rocr-runtime) to support translation of time from HSA agents to a common global system timeline.

A goal is to support alignment of timestamps across HSA agents with precision targets of 1 µs or less, in systems where appropriate HW mechanisms exist to allow this (e.g., PTP/PTM)

## PTP background

[Precision Time Protocol](https://standards.ieee.org/ieee/1588/6825/) is an IEEE standard for time synchronization in
Ethernet based networks. PTP is seeing growing adoption in datacenters as a mechanism to achieve sub-microsecond
alignment across thousands of end-systems attached through the Ethernet fabric.

Support for sub-microsecond precision in MI4xx systems requires a range of HW and SW support:
- Ethernet switches and end-host NICs must support HW-level packet timestamping
- Ethernet switches must implement IEEE-1588 GMC, TC and/or BC capabilities.
- NIC or host system software must implement IEEE OC capabilities, and through them, constantly discipline the host's
  CPU-based POSIX `CLOCK_REALTIME` clock.
- GPU clocks cannot be programmed dynamically on MI4xx; therefore, GPU timestamps must be post-processed to translate
  them onto the systems realtime clock domain, thus aligning them to the network's PTP clock.

## Overview

Because clocks drift over time, there is generally no fixed ratio between the tick rate of a GPU clock and that of the
host CPU (this is to some extent not true in A+A systems, for which some shortcuts can be taken). For this reason, in
order to accurately translate a GPU timestamp to the system timeline requires host software to maintain a running
history of how much the CPU and GPU clocks differ at various points over time. Depending on the independent a) drift rate
of the CPU clock, b) drift rate of the GPU clock, and c) target precision level, this running history may need to be
populated with samples of CPU/GPU timestamps collected at very frequent intervals -- possibly up to 100 Hz.

Our approach is thus to continuously sample synchronized GPU/realtime timestamps ("crosststamps") at high
frequency, and use them to perform translations between clock domains when needed.

A notional workflow we need to support is something like this:
1. User runs an application under rocprof
2. A dedicated thread (or threads) is established to do the following:
    - Collect crosststamps from KFD at some high frequency needed to support PTP-level precision (e.g., 100Hz)
    - Store crosststamps in some form of storage (e.g., a time-series database or simple in-memory hashtable). The
      storage must support subsequent querying for timestamp translation.
    - **Note: whether these operations are done by the same or separate threads, whether those threads are part of the
      ROCR instance or a separate system daemon(s), and how storage is managed are design considerations we
      elaborate on [here](./doc/arch.md).**
3. Kernel dispatch/completion events produced by the workload, which include raw GPU timestamps, are surfaced into rocprof
4. rocprof calls into ROCR through HSA API to convert these raw GPU timestamps to the system/realtime timeline (e.g.,
via something like
[hsa_amd_profiling_convert_tick_to_system_domain](https://github.com/ROCm/rocm-systems/blob/users/bkocolos/precision-time/projects/rocr-runtime/runtime/hsa-runtime/inc/hsa_ext_amd.h#L999))
5. ROCR's implementation of this API vectors into ROCm timesync, which queries the persistent timestamp storage backend
with the provided GPU timestamp to produce the offset to translate to realtime.

## Components

The following diagram lays out the current design of the system. We discuss each component below.

![](./doc/img/out-of-process.png)

### rocm-timesyncd

`rocm-timesync` is a new standalone ROCm system service. One instance of this service runs per node. The service:
- queries KFD for crosststamps by calling into
  [libhsakmt](https://github.com/ROCm/rocm-systems/tree/users/bkocolos/precision-time/projects/rocr-runtime/libhsakmt).
- stores crosststamps into a ringbuffer(s)

#### Configuration

`rocm-timesync` accepts a config with the following format:
```yml
channels:
  hz_precision_high:
    hz: <hz>
    order: <order>
  hz_precision_low:
    hz: <hz>
    order: <order>
```

`hz_precision_high` and `hz_precision_low` define parameters for 2 channels that the process can publish timestamp data
to. By default:
- `hz_precision_high.hz=100`
- `hz_precision_high.order=20`
- `hz_precision_low.hz=1`
- `hz_precision_low.order=12`

This means the high precision channel contains crosststamps sampled at 100 Hz, and the channel consumes 2^20 = 1MiB
of system memory. The low precision channel contains crosststamps sampled at 1 Hz and consumes 2^12 = 4KiB of system
memory.

These channels are internally implemented as ringbuffers. Old data will be overwritten by the producer once the pointer
wraps around to the front of the buffer.

### librocm-timesync

`librocm-timesync` is a shared library that connects to the channels published by `rocm-timesyncd`, streams them into
a backend (memory and/or storage), and uses that backend to implement time translation calls made to it by ROCR. This
library is linked into the process runtime when ROCR declares it as a runtime dependency (more on this below)

It is assumed that an arbitrarily long amount of time may elapse between a) when the library streams a crosststamp from the
ringbuffer, and b) the point at which that crosststamp is needed to service a translaton request. Based no this
assumption, a persistent backend time-series database is used to store crosststamps. The current version uses
[InfluxDB](https://www.influxdata.com/).

The library can be configured to only store data in memory rather than in a database, though this presents a tradeoff:
either a) tolerate potentially excessive consumption of system memory, or b) require eviction of old timestamps, and
thus tolerate a loss of precision for translations overlapping those entries. While the same tradeoff _is_ presented to
a TSDB backend, one assumes that persistent disk storage is more readily available either on the node, or elsewhere in
the cluster.

When a TSDB is present, the library will still place a memory cache in front of it, so that it can quickly handle the
common case of translating recently generated GPU timestamps directly from memory rather than requiring a database
query.

#### Configuration

`librocm-timesync` accepts a config with the following format:
```yml
db:
  influx:
    host: <InfluxDB hostname>
    port: <InxlufDB port>
    database: <InfluxDB database name>
cache:
  max_entries_per_gpu: <cache entries>
```

All entries in the config are optional and default to:
```yml
cache:
  max_entries_per_gpu: -1
```

Wherenegative cache size means unlimited (i.e., the database lives entirely in memory)

In contrast, a config like this puts a fixed size cache with up to 1024 crosststamps per GPU in front of an InfluxDB
instance reachable at https://influxdb:8086 with a database named "timesync":
```yml
db:
  influx:
    host: influxdb
    port: 8086
    database: timesync
cache:
  max_entries_per_gpu: 1024
```

### `rocr-runtime`

#### API

## Code

- [source/producer](./source/producer) implements the `rocm-timesyncd` system service, which calls into KFD to query
  crosststamps at configurable intervals
- [source/consumer](./source/consumer) implements the `librocm-timesync` shared library, which is linked by
  [rocr-runtime](https://github.com/ROCm/rocm-systems/tree/users/bkocolos/precision-time/projects/rocr-runtime). This
  shared library streams crosststamps from the `rocm-timesyncd` into backend storage, and uses that storage to implement

## Design Options

[doc/arch.md](./doc/arch.md) explores some of the key design considerations motivating this approach in more detail. 
