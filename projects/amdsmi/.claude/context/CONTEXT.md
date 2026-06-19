# amd-smi — Context Glossary

> Agent-owned working knowledge. **Not public documentation** — for the AMD-SMI
> agents only. Public docs live in `docs/`. This is the only agent-owned file
> checked into the repo; specs and other hand-off artifacts are ephemeral and
> live under `${TMPDIR:-/tmp}/amdsmi-agent-specs/`.
>
> Canonical domain terms only: what a word *means* in amd-smi, and which meaning is
> intended when it's ambiguous. No implementation walkthroughs, no API reference.
> Keep entries to a line or two. Add a term the moment an `amdsmi-interrogate` session resolves it.

## How To Use

- When a term is ambiguous in a design or conversation, look here first.
- If a design uses a term in a way that conflicts with an entry, stop and reconcile.
- Update inline during an `amdsmi-interrogate` session — don't batch.

## Terms

<!-- Format:
### Term
Definition. Disambiguation if the word is overloaded (e.g. "device" → which of N).
-->

### device

Informal synonym for **any** `amdsmi_processor_handle` (amdsmi.h:294), regardless of
processor type: GPU, CPU, CPU-core, APU, AMD NIC, Broadcom NIC, or Broadcom switch
(`amdsmi_processor_type_t`, amdsmi.h:354). **Not** an API type — the word leaks in from
ROCm-SMI's `rsmi_..._device_...` naming and the CLI's `--gpu` indexing. Broader than a
GPU and broader than a socket (a socket can own multiple processor handles). When a
design says "device", confirm **which processor type(s)** it means.
Disambiguation: device ≠ socket ≠ node.

### socket

Opaque grouping **parent** (`amdsmi_socket_handle`, amdsmi.h:295); owns N processor
handles, enumerated via `amdsmi_get_processor_handles(socket, ...)`. **Not** necessarily
a physical CPU package. CPU sockets are keyed by physical CPU socket id; GPU sockets are
keyed by a BDF-derived `get_gpu_socket_id` (src/amd_smi/amd_smi_system.cc:367), and
**multiple processors can share one socket** ("Multiple devices may share the same
socket" in that loop). The `amdsmi_init` flags decide which types populate sockets
(`AMDSMI_INIT_AMD_GPUS` vs `|AMDSMI_INIT_AMD_CPUS`).
Disambiguation: socket ≠ device, socket ≠ node.

### node

`amdsmi_node_handle` (amdsmi.h:302) = the **whole system / baseboard**. Obtained from
`amdsmi_get_node_handle()`, which **requires** the processor handle to be OAM ID 0
(amdsmi.h:3289-3291). In datacenter terms each system is a "node"; the handle represents
the baseboard and is populated with baseboard-level metrics (power, temps).
Disambiguation: this `node` (whole-system baseboard, OAM 0) is a **different** use of the
word than `AMDSMI_AFFINITY_SCOPE_NODE` (that one is NUMA scope). node ≠ socket ≠ processor.

### OAM ID

The `oam_id` field in `amdsmi_asic_info_t` (amdsmi.h:1135), sourced via
`amdsmi_get_gpu_asic_info`. **Platform-dependent** meaning: on some machines a **physical**
slot/socket index (header line 990 calls it "Physical XGMI ID / OAM ID"; line 1135 calls
it "Corresponds to socket number"), on some machines a **logical** firmware-assigned ID
only. `0xFFFFFFFF` = N/A / not supported. OAM 0 is the designated **baseboard aggregator**
— `amdsmi_get_node_handle` requires it (amdsmi.h:3290). "OAM" (OCP Accelerator Module) is
also a card **form factor** (`AMDSMI_CARD_FORM_FACTOR_OAM`, amdsmi.h:1000). **Not** the
same as the CLI `--gpu` index.

### BDF

`amdsmi_bdf_t` (amdsmi.h:963) is a 64-bit union: bitfields `function_number:3`,
`device_number:5`, `bus_number:8`, `domain_number:48`, plus `as_uint` (full packed 64-bit
view). Note the **48-bit** domain — far wider than the classic 16-bit PCI domain. Roles in
amd-smi: (A) the PCIe **Domain:Bus:Device.Function locator**; (B) the cross-layer
**identity / join key** — GPU `socket_id` is BD-derived (amd_smi_system.cc:367), NIC
sockets are BDF-keyed, and `amdsmi_get_processor_handle_from_bdf` is the reverse lookup;
(C) a user-facing **selector** (CLI `--bdf`).
Disambiguation: BDF identifies a **processor handle**, not a socket or node.

### GPU index

The user-facing "GPU N" (N=0,1,2…) is the **enumeration index** from `amdsmi_get_processor_handles()` discovery order — the CLI assigns it via `enumerate(device_handles)` (amdsmi_cli/amdsmi_helpers.py). CLI `--gpu` also accepts BDF and UUID; "ID" there means this index. **Not** `amdsmi_get_gpu_id()`, which is a device-**type** id and identical across same-SKU cards (amdsmi.h:3565-3568) — never an identity. The processor **handle** is a runtime opaque token, non-persistent across restarts.
Disambiguation: GPU index ≠ `gpu_id` (device type) ≠ handle ≠ OAM ID.

### compute partition

The **active** XCC-grouping mode of a GPU: `amdsmi_compute_partition_type_t` SPX/DPX/TPX/QPX/CPX (amdsmi.h:489-511), get/set via `amdsmi_get_gpu_compute_partition()` / `..._set_...`. GPU-**global** state (applies to all processes). **Not** the per-process `amdsmi_kfd_info_t.current_partition_id` (amdsmi.h:1148-1153), which is a process-scoped KFD/XCP assignment.
Disambiguation: compute partition (GPU-global mode) ≠ `current_partition_id` (per-process KFD).

### accelerator partition

The **umbrella profile** — `amdsmi_accelerator_partition_profile_t` (amdsmi.h:1193-1203), queried via `amdsmi_get_gpu_accelerator_partition_profile()`. One profile **bundles**: a `profile_type` (the compute-partition mode, SPX/DPX/CPX…), `memory_caps` (compatible NPS memory modes), and resource layout (`num_partitions`, `resources[]`). Setting it selects a whole configuration; **compute partition** and **memory partition** are the orthogonal sub-dimensions *within* it. (ROCm 7.14.0, CHANGELOG.md.)
Disambiguation: accelerator partition = the profile bundling compute + memory + resources; not a peer of compute partition but its **parent**.

### memory partition

The **NPS** (Nodes-Per-Socket) HBM-interleaving axis: `amdsmi_memory_partition_type_t` NPS1/NPS2/NPS4/NPS8 (amdsmi.h:523-534). Orthogonal to compute partition; declared as `memory_caps` within an accelerator-partition profile.
Disambiguation: memory partition (NPS, HBM interleaving) ≠ compute partition (XCC grouping).

### activity

Engine utilization percentage; "activity", "utilization", and "usage" are used **synonymously** in amd-smi. The meaningful axis is the sampling: **average** (`amdsmi_engine_usage_t.gfx/umc/mm_activity`, amdsmi.h:1374-1380, from gpu_metrics `average_*_activity`, 2313-2318) vs **instantaneous** (`gfx_busy_inst`, amdsmi.h:2113) vs **accumulated** (`gfx_busy_acc`, amdsmi.h:2117).
Disambiguation: activity = utilization = usage; pin the axis (average vs instantaneous vs accumulated), not the word.

### power cap

The **active, settable** power limit: `amdsmi_set_power_cap()` / `amdsmi_power_cap_info_t` (amdsmi.h:1042-1049). **Not** `amdsmi_power_info_t.power_limit` (amdsmi.h:1352), which is a **read-only** info field — there is no `set_gpu_power_limit()`. Enforcement points: **PPT0** (lower limit, filtered input) and **PPT1** (higher limit, raw input) (amdsmi.h:1051-1052). PPT = Package Power Tracking.
Disambiguation: power cap (settable) ≠ power_limit (read-only field); PPT0/PPT1 are the two firmware enforcement points.

### socket power

Current GPU socket power draw. **Platform-split fields** in `amdsmi_power_info_t` (amdsmi.h:1340-1341): `current_socket_power` = instantaneous, **MI300+ only**; `average_socket_power` = windowed average, **Navi / MI200 and earlier**. `socket_power` is W on Linux-baremetal, µW on host.
Disambiguation: which field is valid depends on ASIC generation (current vs average).

### junction

Hottest on-die temperature; `AMDSMI_TEMPERATURE_TYPE_JUNCTION` is a **literal enum alias** of `AMDSMI_TEMPERATURE_TYPE_HOTSPOT` (same value, amdsmi.h:545) — the two words are interchangeable. **Edge** (`..._TYPE_EDGE`) is a **separate, legacy** sensor, not the junction.
Disambiguation: junction ≡ hotspot (same sensor); both ≠ edge (legacy) ≠ VRAM/HBM temps.

### fabric

In amd-smi, "fabric" = the GPU-to-GPU **XGMI** scale-up interconnect (`tagFabric` group, amdsmi.h:5482; `AMDSMI_LINK_TYPE_XGMI`, 1242). The on-chip **Data Fabric clock** (FCLK, `AMDSMI_CLK_TYPE_DF`, amdsmi.h:445-447) is a **different** use of the word — a clock domain, not the interconnect.
Disambiguation: fabric (XGMI interconnect) ≠ Data Fabric clock (FCLK on-chip clock domain).

### violation status

Time-based throttling metrics — PVIOL/TVIOL percentages via `amdsmi_get_violation_status()` — available on **MI300+ only** (gpu_metrics v1.6+). On **Navi / MI100 / MI200** (v1.3) use the gpu_metrics `throttle_status` / `indep_throttle_status` bitflags instead, which report *whether* throttling is happening now, not *how much* over time. (docs/conceptual/gpu-violations.md.)
Disambiguation: violation status (MI300+, time %) ≠ throttle_status (older gen, instantaneous bitflags).

### bad pages

Retired / reserved VRAM pages — "bad pages" and "retired pages" are the **same set**, same struct `amdsmi_retired_page_record_t` (amdsmi.h:1945-1958), via `amdsmi_get_gpu_bad_page_info()` / `amdsmi_get_gpu_memory_reserved_pages()`. State is the `amdsmi_memory_page_status_t` enum: **PENDING** (flagged, awaiting retirement window) → **RESERVED** (retired, unavailable) or **UNRESERVABLE** (failed). amd-smi reports address/size/status only — the **error class (CE vs UE) driving retirement is not captured or exposed** here; CE/UE counts live separately in `amdsmi_get_gpu_ecc_count()` (see docs/conceptual/ras.md).
Disambiguation: bad pages ≡ retired pages (one set, status enum distinguishes pending/reserved/unreservable); error class is NOT part of it — use ECC count for CE/UE.
