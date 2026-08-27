# amdsmi-device-discovery Specification

## Purpose

Defines how `libamd_smi.so` decides what exists on a machine, what each thing is
called, and which kernel interface answers a given question. This is the layer
underneath every metric: `amdsmi_init()` builds a fixed tree of sockets and
processors, hands out opaque handles into it, and every later `amdsmi_*` call is
a lookup against that tree plus a read from one of five data sources.

These concerns are one capability because they are not separable in the
implementation. The enumeration order is a consequence of the data source
(sorted KFD topology nodes); the identifiers a caller may key on are a
consequence of which source produced them; and the degradation behavior in a
container, under a restricted `/sys`, or without `libdrm_amdgpu` is a
consequence of which source a query depends on. Specifying them apart would
leave the interesting cases — the ones where the answer differs between sources
— unowned.

Scope is the C library's discovery and transport layer: `amdsmi_init()`,
`amdsmi_shut_down()`, the socket and processor handle APIs, the identity getters
(`amdsmi_get_gpu_device_bdf`, `..._uuid`, `..._kfd_info`,
`..._enumeration_info`), and the `rocm_smi` sysfs layer they are built on. It
deliberately excludes the metric semantics of individual getters, the `amd-smi`
CLI, and the Python bindings.

Where a runtime dependency has packaging consequences, this capability states
the load contract and the degradation, and defers the question of which channel
ships which file to [amdsmi-install-layout].

## Requirements

### Requirement: KFD Topology Is The Enumeration Source Of Truth

GPU enumeration SHALL be driven by walking
`/sys/class/kfd/kfd/topology/nodes/<N>/`, not by scanning `/sys/class/drm` or
`/dev/dri`. A topology node SHALL become a GPU processor only when its
`gpu_id`, `domain`, `location_id`, and `drm_render_minor` properties are all
readable and `gpu_id` is non-zero; a `gpu_id` of zero identifies a CPU node and
SHALL be skipped. A discovered device whose BDF has no corresponding readable
KFD node SHALL then be dropped from the device list.

Consequently the presence of a `/dev/dri/renderD*` character device is neither
necessary nor sufficient for a GPU to be enumerated, and a DRM device bound to a
non-AMD driver never appears at all.

#### Scenario: Enumeration survives a container with no device nodes

- **WHEN** the process can read `/sys/class/kfd` and `/sys/class/drm` but
  `/dev/dri` is empty or absent
- **THEN** `amdsmi_init()` succeeds and reports the full socket and processor
  set with correct BDF, UUID, KFD ids, and DRM card/render numbers, because all
  of those come from sysfs; only the queries that open a render node degrade

#### Scenario: A non-AMD DRM device does not shift the enumeration

- **WHEN** the host has a BMC display adapter bound to a non-AMD DRM driver that
  holds `/sys/class/drm/card0`, and two AMD GPUs on `card1` and `card2`
- **THEN** AMD SMI reports exactly two GPU processors, because the BMC adapter
  has no KFD topology node — but the DRM card indices it reports remain 1 and 2,
  since those are kernel-assigned across all DRM drivers

#### Scenario: An unreadable topology directory fails initialization

- **WHEN** `/sys/class/kfd/kfd/topology/nodes/` cannot be opened, for example in
  a container that mounts a restricted `/sys`
- **THEN** `amdsmi_init()` returns `AMDSMI_STATUS_NOT_INIT` rather than
  reporting zero devices, so a caller cannot mistake a masked sysfs for a
  machine with no GPUs

#### Scenario: A restricted /sys/class/drm is equally fatal

- **WHEN** `/sys/class/kfd` is fully readable but `/sys/class/drm` is masked
- **THEN** `amdsmi_init()` still fails with `AMDSMI_STATUS_NOT_INIT`, because
  the device objects that back every sysfs read are rooted at a
  `/sys/class/drm` path; both trees are required, not just KFD

### Requirement: GPU Processor Order Is Ascending BDF

Discovered GPU devices SHALL be ordered by a stable sort on the 64-bit
BDF/PCI id with the partition-id field (bits 31:28) masked out, so that
partitions of one physical device stay adjacent and in KFD node order relative
to each other. The resulting position is the AMD SMI GPU index used by
`amdsmi_get_processor_handles()` and by every internal `rsmi_*` call.

This order is a function of PCI topology alone. It SHALL NOT be assumed to match
the DRM card index, the DRM render minor, the KFD node id, or the HIP device
ordinal, all of which are assigned by the kernel or derived differently.

#### Scenario: The AMD SMI index disagrees with every other enumeration

- **WHEN** a host has GPUs at `0000:03:00.0` (KFD node 2, `card2`,
  `renderD129`) and `0000:63:00.0` (KFD node 1, `card1`, `renderD128`)
- **THEN** AMD SMI index 0 is the `0000:03:00.0` device while HIP ordinal 0 is
  the `0000:63:00.0` device, because `hip_id` is `node_id` minus the smallest
  GPU node id and therefore follows KFD node order, not BDF order

#### Scenario: Adding or removing a GPU renumbers the others

- **WHEN** a GPU is added at a bus address that sorts before an existing device,
  or a device is removed
- **THEN** the indices of the remaining devices shift, so an index recorded in
  one boot is not a valid key in another; BDF is the identifier that survives

### Requirement: Sockets Group Processors By Physical Device

Each GPU SHALL be filed under a socket whose identifier is the uppercase hex
string `DOMAIN:BUS:DEVICE` — four, two and two digits — deliberately omitting
the PCI function and the partition-id bits, so every partition of one physical
package lands in one socket. CPU sockets SHALL instead be identified by the
decimal socket index, and AI-NIC and switch sockets by the same
`DOMAIN:BUS:DEVICE` form as GPUs. Sockets SHALL be appended in discovery order:
GPU sockets first, then CPU sockets, then NIC and switch sockets.

`amdsmi_get_processor_handles()` and the untyped
`AMDSmiSocket::get_processor_count()` SHALL return the socket's **GPU**
processors only. Callers needing CPU, CPU-core, AI-NIC, or switch processors
SHALL use `amdsmi_get_processor_handles_by_type()`, which selects the
per-type list.

#### Scenario: A socket handle is not a per-type container

- **WHEN** a socket holds both a GPU and CPU-core processors and a caller
  invokes `amdsmi_get_processor_handles()` on it
- **THEN** only the GPU processors are returned and the CPU-core processors are
  invisible, because the untyped accessor is hard-wired to the GPU vector;
  querying by type is the only way to see the rest

#### Scenario: Socket namespaces cannot collide

- **WHEN** a host has both GPU sockets and ESMI-discovered CPU sockets
- **THEN** the CPU socket named `0` never merges with a GPU socket, because GPU
  socket identifiers always carry the `DOMAIN:BUS:DEVICE` colon form and are
  compared as strings

### Requirement: Handles Are Views Into One Session's Tree

Handles are pointers into the tree this capability builds. Their opaqueness,
the membership lookup that validates them, and the statuses an unrecognized one
produces are specified in [amdsmi-c-api-abi]; what this capability adds is that
the tree behind them is constructed once per session and destroyed with it.
Handles SHALL therefore be treated as belonging to the enumeration that
produced them: they SHALL NOT be persisted, compared across sessions, or shared
between processes, and re-enumeration after a full teardown SHALL be assumed to
produce different pointers for the same devices.

A second `amdsmi_init()` while already initialized SHALL NOT re-enumerate; it
increments the reference count and leaves the existing tree, and therefore
every outstanding handle, untouched.

#### Scenario: A library and its host application share one enumeration

- **WHEN** an application calls `amdsmi_init()` and then loads a plugin that
  also calls `amdsmi_init()`
- **THEN** the second call returns success without re-enumerating, so the two
  components see the same devices in the same order, and the plugin's
  `amdsmi_shut_down()` does not invalidate the application's handles

#### Scenario: A cached handle is a dangling pointer, not a stale token

- **WHEN** a program initializes, records its processor handles, shuts down, and
  initializes again
- **THEN** the objects behind the first set were deleted and reallocated, so the
  recorded pointers address freed or repurposed storage rather than an
  identifiably out-of-date device

#### Scenario: The processor "info" string is not an identifier for a GPU

- **WHEN** `amdsmi_get_processor_info()` is called on a GPU processor
- **THEN** the returned string is an uninitialized index value that varies from
  run to run, because only CPU and CPU-core processors are constructed with a
  processor index; a caller needing an ordinal for a GPU must use its position
  in `amdsmi_get_processor_handles()`

### Requirement: Stable And Unstable Device Identifiers

The library SHALL expose the identifiers below, and callers SHALL treat only
those marked stable as keys that survive a reboot or a driver reload.

| Identifier | API | Source | Stable key? |
| ---------- | --- | ------ | ----------- |
| BDF | `amdsmi_get_gpu_device_bdf` | KFD `domain` + `location_id` | Yes — a property of the PCI slot |
| GPU UUID | `amdsmi_get_gpu_device_uuid` | ASIC serial + PCI device id + partition index | Yes, when the ASIC exposes a unique id |
| HIP UUID | `amdsmi_get_gpu_enumeration_info().hip_uuid` | `GPU-` plus the 16-hex ASIC unique id | Yes, same precondition |
| KFD id (`kfd_id`) | `amdsmi_get_gpu_kfd_info` | KFD node `gpu_id` | Kernel-assigned at probe |
| KFD node id (`node_id`, `hsa_id`) | `amdsmi_get_gpu_kfd_info`, `..._enumeration_info` | topology node number | Kernel-assigned at probe |
| DRM card index | `amdsmi_get_gpu_enumeration_info().drm_card` | `/sys/class/drm/card<N>` | No — a global DRM counter shared with non-AMD drivers |
| DRM render minor | `amdsmi_get_gpu_enumeration_info().drm_render` | KFD `drm_render_minor` | No — probe-order dependent |
| HIP ordinal (`hip_id`) | `amdsmi_get_gpu_enumeration_info` | `node_id` minus smallest GPU `node_id` | No — shifts when any node is added or removed |
| AMD SMI index | position in `amdsmi_get_processor_handles` | BDF sort | Only within one hardware configuration |
| OAM id | `amdsmi_get_gpu_enumeration_info().oam_id` | XGMI physical id | `0xFFFFFFFF` when the ASIC has no XGMI |
| CUID | `amdsmi_get_gpu_device_cuid` | external `amdcuid` library | Absent unless built with `BUILD_CUID` |

`amdsmi_get_processor_handle_from_bdf()` SHALL be the supported way to turn a
recorded identifier back into a handle; it walks every socket and compares all
four BDF fields, returning `AMDSMI_STATUS_API_FAILED` when nothing matches.

#### Scenario: A monitoring agent keys its time series correctly

- **WHEN** an agent must attribute samples to the same physical GPU across a
  reboot that reorders DRM node assignment
- **THEN** keying on BDF or GPU UUID keeps the series intact, while keying on
  the DRM card index, render minor, or AMD SMI index silently reattributes data
  to a different device

#### Scenario: UUID generation fails rather than inventing a value

- **WHEN** the ASIC does not expose a unique id through
  `rsmi_dev_unique_id_get`
- **THEN** `amdsmi_get_gpu_device_uuid()` returns the underlying error instead
  of emitting a UUID built from defaults, so a caller cannot mistake a
  synthesized constant for a real device identity

#### Scenario: CUID is unavailable in the default build

- **WHEN** a caller invokes `amdsmi_get_gpu_device_cuid()` on a library built
  with the default options
- **THEN** it returns `AMDSMI_STATUS_NOT_SUPPORTED`, because `BUILD_CUID`
  defaults to off and the whole code path is compiled out

### Requirement: Partitions Appear As Peer Processors Under One Socket

When a GPU is partitioned, the kernel exposes one KFD topology node per
partition, and each SHALL therefore become its own AMD SMI GPU processor. All
partitions of one package SHALL share a socket, because the socket identifier
drops the function and partition-id bits. The partition index SHALL be read from
the PCI id bits 31:28, falling back to the function bits 2:0 when bits 31:28 are
zero but the function is non-zero — a fallback the driver's varying encodings
require. `amdsmi_get_gpu_kfd_info().current_partition_id` SHALL be
`0xFFFFFFFF` when no partition id can be determined.

The partition index SHALL also be folded into the generated GPU UUID, so
partitions of one physical ASIC — which share an ASIC serial and PCI device id —
receive distinct UUIDs.

#### Scenario: Partitions of one board are not mistaken for separate boards

- **WHEN** a partitioned accelerator presents several KFD nodes at the same
  domain, bus, and device
- **THEN** `amdsmi_get_socket_handles()` reports a single socket whose GPU
  processor list holds every partition, so a caller counting sockets counts
  physical boards while a caller counting processors counts partitions

#### Scenario: A non-partitionable device reports a benign partition id

- **WHEN** the device does not support partitioning
- **THEN** `current_partition_id` is 0 and the partition-related getters return
  `AMDSMI_STATUS_NOT_SUPPORTED`, rather than the partition id being reported as
  unavailable

### Requirement: Data Source Precedence Per Query Class

Each query class SHALL be answered by one designated source; there is no
fallback chain between sources except where stated below.

| Source | Answers | Behavior when unavailable |
| ------ | ------- | ------------------------- |
| KFD topology sysfs (`/sys/class/kfd/kfd/topology/nodes`) | existence of devices, BDF, KFD id, node id, partition id, caches, IO links | fatal to `amdsmi_init()` |
| DRM sysfs (`/sys/class/drm/renderD<N>/device/...`) | board info, RAS counters and bad pages, power caps, `pp_dpm_*` frequency tables, `pp_features`, UALink fabric attributes | per-query error; the attribute is reported as unavailable |
| `libdrm_amdgpu` ioctl on `/dev/dri/renderD<N>` | VBIOS info, ASIC device info, VRAM usage, marketing name, driver name and date, virtualization mode | per-query error; enumeration is unaffected |
| `/proc/<pid>/fdinfo` plus `/sys/class/kfd/kfd/proc` | per-process VRAM, GTT, CPU memory and engine usage | process is reported with only the coarse KFD-derived fields |
| ESMI / `amd_hsmp` | AMD CPU socket and core telemetry | CPU processors are not created at all |

`libdrm` SHALL be treated as strictly optional: `AMDSmiDrm::init()` failing is
ignored by `populate_amd_gpu_devices()`, and each ioctl-backed getter loads
`libdrm` on its own. When `libdrm` never loaded, the device's BDF and DRM render
path are left unpopulated, which in turn makes every DRM-sysfs query fail as
well, since those paths are built from the render node name.

#### Scenario: Per-process memory has a documented degraded form

- **WHEN** `gpuvsmi_get_pid_info()` cannot read a process's `fdinfo` directory
- **THEN** the process still appears in the list, carrying the PID and the VRAM
  figure that came from KFD, because per-process detail is best-effort while
  process *presence* comes from a different source

#### Scenario: An ioctl-backed query fails without disturbing discovery

- **WHEN** `/dev/dri/renderD<N>` cannot be opened
- **THEN** `amdsmi_get_gpu_vbios_info()` and `amdsmi_get_gpu_asic_info()` return
  `AMDSMI_STATUS_FILE_ERROR` while sockets, processors, BDF, UUID and KFD info
  remain fully correct

#### Scenario: A missing sysfs attribute is not an error for the whole call

- **WHEN** `/sys/module/amdgpu/version` does not exist, as with an in-tree
  `amdgpu`
- **THEN** `smi_amdgpu_get_driver_version()` reports the version string as
  `N/A` and returns success, so the rest of `amdsmi_get_gpu_driver_info()` still
  runs

### Requirement: libdrm_amdgpu Is Loaded By dlopen With Ordered Candidates

`libdrm_amdgpu` SHALL be resolved at runtime by `dlopen`, never linked, and the
loader SHALL try, in order: the SONAME captured from `pkg-config` at build time,
the bare `libdrm_amdgpu.so`, and `librocm_sysdeps_drm_amdgpu.so.1`. The first
candidate that opens wins. For each candidate the loader SHALL first probe with
`RTLD_NOLOAD`, adopting an already-loaded copy rather than opening a second one,
and SHALL fall back to `RTLD_LAZY`. A diagnostic SHALL be emitted only after
every candidate has failed, and the failure SHALL be reported as
`AMDSMI_STATUS_FAIL_LOAD_MODULE`.

The third candidate exists because relocatable ROCm trees ship a
SONAME-renamed vendored copy under `lib/rocm_sysdeps`; the packaging
consequences of that are specified in [amdsmi-install-layout]. This capability
requires only that the search list stay ordered from most specific to most
general, so a host-provided `libdrm_amdgpu` is preferred over a vendored one.

#### Scenario: A vendored ROCm tree resolves its renamed library

- **WHEN** the host has no `libdrm_amdgpu.so.1` on the linker path but a
  relocatable ROCm install provides `librocm_sysdeps_drm_amdgpu.so.1`
- **THEN** the third candidate loads and the ioctl-backed getters work, so a
  self-contained ROCm tree does not need a system libdrm package

#### Scenario: The process never holds two copies of libdrm

- **WHEN** the hosting application has already loaded `libdrm_amdgpu` itself
- **THEN** the `RTLD_NOLOAD` probe adopts that handle and `unload()` balances
  the reference it took, so AMD SMI does not introduce a second copy of the
  library into a process that also runs a graphics or compute stack

#### Scenario: A missing libdrm degrades rather than aborts

- **WHEN** no candidate can be opened
- **THEN** `amdsmi_init()` still succeeds and enumerates every GPU, because
  `AMDSmiDrm::init()`'s return value is deliberately discarded; the devices
  simply carry no resolved render-node path, and both the ioctl-backed and the
  DRM-sysfs-backed getters fail for them

### Requirement: Non-GPU Processor Classes Are Discovered Independently

CPU, AI-NIC, and Broadcom NIC/switch discovery SHALL each be gated on its own
`amdsmi_init()` flag and SHALL be non-fatal: a failure in any of them SHALL
leave the other classes usable. Specifically:

- `AMDSMI_INIT_AMD_CPUS` requires the library to be built with
  `ENABLE_ESMI_LIB` (on by default only for x86_64) and requires `esmi_init()`
  to succeed, which in turn requires the `amd_hsmp` kernel module with HSMP
  enabled in firmware. When `esmi_init()` fails, CPU discovery SHALL be skipped
  with a diagnostic on stderr and `amdsmi_init()` SHALL still return success.
- Each CPU socket SHALL contribute one `AMDSMI_PROCESSOR_TYPE_AMD_CPU`
  processor plus one `AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE` processor per physical
  core on that socket.
- `AMDSMI_INIT_AMD_NICS` scans `/sys/bus/pci/devices` for the AI-NIC bridge
  (vendor `0x1dd8`, device `0x0008`) and pairs it with downstream ports found
  through `/sys/class/net`, sorting both NICs and their ports by BDF. Finding no
  NIC SHALL return success with an empty list.
- A single NIC that fails to populate SHALL be skipped with a log entry rather
  than aborting the remaining NIC discovery.

#### Scenario: An AMD CPU without amd_hsmp does not break GPU monitoring

- **WHEN** the host has an AMD CPU but the `amd_hsmp` module is not loaded, and
  the caller passes `AMDSMI_INIT_AMD_APUS`
- **THEN** `amdsmi_init()` returns success, the CPU processor list is empty, and
  every GPU is enumerated normally — the failure is reported on stderr rather
  than propagated, because CPU telemetry is a bonus and not a precondition

#### Scenario: CI without NIC hardware initializes cleanly

- **WHEN** `AMDSMI_INIT_AMD_NICS` is requested on a machine with no AI-NIC
- **THEN** discovery returns `SMI_NIC_STATUS_NO_DATA`, which is translated to
  success with zero NIC processors, so a test suite need not branch on hardware
  presence

### Requirement: A Missing Driver Is Distinct From A Masked Topology

When GPU population fails, `amdsmi_init()` SHALL consult
`/sys/module/amdgpu/initstate` before reporting. If that file is absent or does
not read `live`, the result SHALL be `AMDSMI_STATUS_DRIVER_NOT_LOADED`;
otherwise the underlying `rocm_smi` status SHALL be mapped through, which for a
topology-discovery failure is `AMDSMI_STATUS_NOT_INIT`. These two outcomes tell
an operator whether to load a driver or to fix container or permission
configuration, so they SHALL remain distinct.

#### Scenario: Driver loaded but topology hidden

- **WHEN** `amdgpu` reports `live` but `/sys/class/kfd` is masked by the
  container runtime
- **THEN** `amdsmi_init()` returns `AMDSMI_STATUS_NOT_INIT`, which is the signal
  that the driver is fine and the sandbox is not

#### Scenario: The driver is mid-transition

- **WHEN** `/sys/module/amdgpu/initstate` reads `coming` or `going`
- **THEN** the state is not `live` and `AMDSMI_STATUS_DRIVER_NOT_LOADED` is
  returned, so a caller racing a modprobe gets a retryable answer rather than an
  opaque initialization error

### Requirement: Privilege Requirements Are Narrow And Explicitly Surfaced

Discovery and device identity SHALL require no elevated privilege beyond read
access to sysfs. Membership in the `render` or `video` group SHALL be required
only for the queries that open `/dev/dri/renderD<N>`, and a failure to open that
node SHALL be surfaced as `AMDSMI_STATUS_FILE_ERROR` — the status reflects the
failed open, not the reason for it.

Effective-UID-zero SHALL be required by exactly two read paths, each of which
returns `AMDSMI_STATUS_NO_PERM` before doing any work:
`amdsmi_get_gpu_accelerator_partition_profile_config()` and
`amdsmi_get_gpu_cper_entries()`. Errors originating in sysfs SHALL be mapped
from `errno`, with `EACCES` and `EROFS` becoming a permission status and
`EPERM`, `ENOENT` and `ENOTSUP` becoming `NOT_SUPPORTED`; `EROFS` is mapped to
permission specifically so a read-only `/sys` in an unprivileged container is
distinguishable from a feature the kernel does not implement.

#### Scenario: An unprivileged caller enumerates everything

- **WHEN** a user in no GPU-related group runs a program that initializes AMD
  SMI
- **THEN** sockets, processors, BDFs, UUIDs, KFD ids and DRM card and render
  numbers are all reported correctly, and only the ioctl-backed getters fail —
  so an inventory tool needs no privilege at all

#### Scenario: A root-only gate hides the real answer from a normal user

- **WHEN** `amdsmi_get_gpu_accelerator_partition_profile_config()` is called on
  a device that does not support partitioning
- **THEN** a non-root caller receives `AMDSMI_STATUS_NO_PERM` and a root caller
  receives `AMDSMI_STATUS_NOT_SUPPORTED`, because the privilege check precedes
  the capability check; a non-root caller cannot distinguish "needs root" from
  "not supported here"

#### Scenario: Another user's GPU process is skipped, not misreported

- **WHEN** the process list sweep hits a PID whose `/proc/<pid>/fd` is
  unreadable because it belongs to another user
- **THEN** that entry is treated as inconclusive and skipped, so a permission
  error is never mistaken for PID-namespace isolation when deciding how to read
  KFD process data

### Requirement: The WSL Backend Replaces Native Discovery Entirely

When the library is built with `ENABLE_WSL_BACKEND`, GPU population SHALL first
attempt the WSL path and SHALL use the native `rocm_smi` plus `libdrm` path only
when that attempt reports `AMDSMI_STATUS_NOT_SUPPORTED`. The WSL path SHALL be
selected by the presence of `/dev/dxg`, SHALL obtain its device list from
`librocdxg.so.1` (falling back to `librocdxg.so`) via `dlopen` and HSA KMT node
properties, and SHALL create one socket per GPU with an identifier of the form
`wsl:<processor-type>:<bdf-as-uint>`. On the WSL path `amdsmi_shut_down()`
SHALL close the KFD channel and unload `librocdxg` instead of calling
`rsmi_shut_down()`.

Because there is no `/dev/dri` and no KFD sysfs under WSL, every query that
would use those SHALL be answered by the backend or return
`AMDSMI_STATUS_NOT_SUPPORTED`; the `rsmi_*` wrapper short-circuits to
`NOT_SUPPORTED` whenever a device carries a backend.

#### Scenario: A WSL guest with the driver library missing is reported precisely

- **WHEN** `/dev/dxg` exists but neither `librocdxg.so.1` nor `librocdxg.so` can
  be opened
- **THEN** population returns `AMDSMI_STATUS_DRIVER_NOT_LOADED` and logs the
  reason, rather than falling through to the native path and reporting a bare
  Linux machine with no GPUs

#### Scenario: Native hosts are unaffected by the build option

- **WHEN** a library built with `ENABLE_WSL_BACKEND` runs on a native Linux host
- **THEN** the absence of `/dev/dxg` yields `AMDSMI_STATUS_NOT_SUPPORTED`
  immediately and the native KFD path runs unchanged, so one binary serves both
  environments

#### Scenario: A partially usable librocdxg is rejected wholesale

- **WHEN** `librocdxg` opens but one of the required symbols cannot be bound
- **THEN** the handle is closed, the symbol table is cleared, and the load is
  treated as a failure, so no query can call through a half-populated function
  table
