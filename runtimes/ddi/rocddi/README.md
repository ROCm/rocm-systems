<!-- Copyright (c) 2026 Advanced Micro Devices, Inc. -->
<!-- SPDX-License-Identifier: MIT -->

# rocddi

rocddi is the private Rust device layer used by
[libamdf](../libamdf/README.md) and [libhsa](../libhsa/README.md). It owns
hardware discovery, driver contexts, device activation, and native memory and
queue resources. The frontends own their C APIs, public handles, and runtime
policy. rocddi is an `rlib`; it exports no C API and promises no stable Rust
interface.

## Runtime model

A `Session` is a frontend's scope for discovery, activation, and driver
shutdown. Creating one constructs an instance of each compiled backend
without activating a device or acquiring a GPU context. Session clones share
those driver instances. Discovery returns passive `Endpoint` snapshots.
`Session::populate_endpoint` adds optional facts to the same endpoint record,
then `Session::activate` returns a `Device` that owns its driver state. Both
operations revalidate discovered facts. GPU
operations require the checked `Device::gpu` view. Each discovery reads
current backend topology; endpoint records can become stale. Separate sessions
have independent driver identities. Sessions using
`DriverContextLifetime::Process` reuse the native primary KFD context.

`DriverContextLifetime::Process` permits a process-owned driver context to
survive session shutdown; `DriverContextLifetime::Session` requires the context
and its bindings to be released with the session. `Session::destroy` requires
other session clones, devices, and driver-backed resources to be released
first. Failed native cleanup retains its owners for retry or safe process
teardown. See the [safety guide](docs/safety.md) for these ownership rules.

The current backend is Linux KFD/DRM on x86-64 and AArch64. It discovers GPU
endpoints; rocddi has no CPU or NPU device backend yet. Host-only memory works
without an activated GPU. Availability of GPU operations depends on the
discovered device and driver capabilities.

`topology::host::snapshot` reads CPU identity, online CPUs, NUMA domains,
memory capacity, and cache records as passive OS facts. The reads are
best-effort and are not one atomic topology generation. A host domain is not
an endpoint or activated `Device`; frontends decide how to expose it.
`host::clock` reads the live profiling timebase. `host::cache` qualifies a
recipe for one mapping policy, and the mapping owner keeps the entire line
cover live through maintenance. Cache geometry from topology does not qualify
that operation. `memory::host` owns host allocations and reservations.

## Code layout

- `session.rs` coordinates driver instances and their context lifetime;
  `topology/` holds passive endpoint and host topology, while `host/` groups
  the live profiling clock and cache recipes.
- `driver.rs` defines private backend contracts; `driver/linux_kfd/` implements
  the current GPU backend.
- `device/` owns activated devices, events, GPU queues, copies, and profiling.
- The Linux GPU event service owns KFD's first signal-page transaction across
  AMDF and HSA. Direct native queue owners retain frontend-supplied signal and
  scratch backing through proven native retirement.
- `memory/storage/` owns runtime records; `memory/host.rs` provides host-only
  backing; `memory/device/` owns device-qualified allocations, registrations,
  and virtual mappings; `memory/interop/` holds platform sharing interfaces.
- `os/` provides host services used by the core and the Linux backend.

## Build and verification

See the [workspace instructions](../../README.md) for CMake and Cargo commands.
The Rust unit and integration tests cover host contracts. The ignored GPU CTS
cases require a qualified host and do not run in the default suite. The runtime
is early-access and is not part of the repository's default build or installation.
