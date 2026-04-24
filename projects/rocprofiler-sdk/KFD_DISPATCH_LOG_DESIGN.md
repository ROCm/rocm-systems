# Plan: Migrate Dispatch-Log Setup to KFD Profiler Ioctl + Self-Describing Format Descriptor

> **Status:** Design proposal. Not yet implemented.
> Companion to `KNOWN_ISSUES.md` and `FIRMWARE_RING_HYBRID_DESIGN.md`.

This document describes a planned KFD/UAPI evolution that:

1. **Migrates the MEC dispatch profiling ring buffer setup** out of `AMDKFD_IOC_UPDATE_QUEUE` (where it was awkwardly bolted on) and into a new sub-op of the existing `AMDKFD_IOC_PROFILER` ioctl.
2. **Adds a self-describing record format descriptor** (JSON) so the rocprofiler-sdk no longer has to hardcode `mec_dispatch_record_16` C structs to decode the ring buffer, and so future record types can be added without re-shipping userspace.

Critically: **no MEC firmware change is required.** The descriptor declares each record type's size statically, so the firmware can keep writing exactly what it writes today.

---

## 1. Executive summary

The current `dispatch_record_buffer_addr/size` fields appended to
`kfd_ioctl_update_queue_args` (`include/uapi/linux/kfd_ioctl.h:99-109`)
are wrong on three counts:

* **Wrong ioctl.** UPDATE_QUEUE is a hot, generic operation. Profiling
  buffer setup is an optional profiler-scoped operation. Coupling them
  pollutes the queue-update path and makes it harder to deprecate the
  feature later.
* **Wrong byte-size validation.** The kernel literal `* 40` at
  `kfd_process_queue_manager.c:633` is mismatched against the firmware's
  current 16-byte records. Validation passes only because user buffers
  end up oversized.
* **Wrong format coupling.** The kernel-side validation hardcodes a
  record size; userspace tools hardcode `mec_dispatch_record_16` C
  structs. Adding a new record type requires coordinated
  kernel + libhsakmt + ROCr + rocprofiler-sdk releases.

This plan does two things:

**Task 1 — Migrate to `AMDKFD_IOC_PROFILER`.** Revert
`kfd_ioctl_update_queue_args` and its handler to upstream-clean shape.
Add `KFD_IOC_PROFILER_DISPATCH_LOG = 3` as a new sub-op of the existing
union-discriminated profiler ioctl, with two op modes: `SET` (configure
the ring buffer) and `GET_DESCRIPTOR` (fetch the kernel-owned JSON
format descriptor). The handler reuses `pqm_update_queue_properties`'s
machinery via a new extracted helper `pqm_update_dispatch_record`, so
the MQD write at `kfd_mqd_manager_v9.c:379-392` is unchanged.

**Task 2 — Self-describing JSON descriptor (KFD-owned).** The KFD
driver itself ships a JSON blob describing each record type (id, name,
size, fields). Userspace queries the descriptor from the kernel via a
new `GET_DESCRIPTOR` sub-op. Any tool — rocprofiler-sdk, a crash
dumper, a custom profiler — can ask KFD what the wire format is
without coordinating with the producer side (libhsakmt/ROCr). The
kernel never parses the JSON; it just hands back a static byte array.
**No firmware change is needed** — the size for each `record_type` is
declared in the descriptor, not encoded in the wire record.

Both changes bump `KFD_IOCTL_MINOR_VERSION` 22→23 and
`KFD_IOC_PROFILER_VERSION_NUM` 1→2. The `dispatch_record_buffer_*`
UAPI fields on `kfd_ioctl_update_queue_args` are removed (private
branch; no upstream consumers).

---

## 2. Task 1 — Migrate dispatch_record_buffer to profiler ioctl

### 2.1 UAPI revert in `include/uapi/linux/kfd_ioctl.h`

**Delete lines 106-108:**

```c
	__u64 dispatch_record_buffer_addr;	/* to KFD: GPU VA (0 = disable) */
	__u32 dispatch_record_buffer_size;	/* to KFD: capacity in records */
	__u32 pad;
```

After the edit, `kfd_ioctl_update_queue_args` matches `kfd_ioctl.h.orig`
(upstream-clean):

```c
struct kfd_ioctl_update_queue_args {
	__u64 ring_base_address;	/* to KFD */
	__u32 queue_id;			/* to KFD */
	__u32 ring_size;		/* to KFD */
	__u32 queue_percentage;		/* to KFD */
	__u32 queue_priority;		/* to KFD */
};
```

**Bump version (line 53):**

```c
#define KFD_IOCTL_MINOR_VERSION 23
```

**Add changelog entry (after line 51):**

```c
 * - 1.23 - Move dispatch_record_buffer setup from UPDATE_QUEUE to PROFILER ioctl
```

### 2.2 chardev handler revert in `amd/amdkfd/kfd_chardev.c`

**Delete lines 498-501:**

```c
	properties.dispatch_record_buffer_addr =
		args->dispatch_record_buffer_addr;
	properties.dispatch_record_buffer_size =
		args->dispatch_record_buffer_size;
```

After the edit, `kfd_ioctl_update_queue` (`kfd_chardev.c:453-513`)
matches `kfd_chardev.c.orig` exactly.

### 2.3 Refactor `pqm_update_queue_properties` in `amd/amdkfd/kfd_process_queue_manager.c`

The dispatch_record_bo block at lines 609-655 and the property-copy at
lines 662-665 must move out of `pqm_update_queue_properties` so that
function returns to upstream-clean shape.

Two design options were considered:

* **Option A (rejected):** keep both inside `pqm_update_queue_properties`,
  gate dispatch_record handling on a "profiling_only" flag. Rejected
  because UPDATE_QUEUE callers will never set the flag and the code
  becomes dead in that path.
* **Option B (chosen):** extract the dispatch_record block into a new
  function `pqm_update_dispatch_record` and call it from the new
  profiler sub-op handler.

**New function** (added after `pqm_update_queue_properties` at
`kfd_process_queue_manager.c:673`):

```c
/**
 * pqm_update_dispatch_record - configure or clear the MEC firmware
 *                              dispatch profiling ring for a queue
 * @pqm:               process queue manager (caller holds p->mutex)
 * @qid:               KFD queue id
 * @addr:              GPU VA of the ring buffer (0 = disable)
 * @num_records:       capacity in records (power-of-2; non-zero when addr != 0)
 * @record_size_bytes: size of one record in bytes (non-zero when addr != 0)
 *
 * The kernel uses record_size_bytes only for BO-mapping size validation.
 * Userspace MUST keep this in sync with the descriptor; see Task 2.
 *
 * Returns 0 on success; -EINVAL on bad inputs; -EFAULT if the buffer
 * is not GPU-mapped; -ENODEV if no pdd; or whatever amdgpu_bo_reserve
 * or dqm->ops.update_queue returns.
 */
int pqm_update_dispatch_record(struct process_queue_manager *pqm,
			       unsigned int qid,
			       u64 addr,
			       u32 num_records,
			       u32 record_size_bytes);
```

**Body** (extracted from current lines 609-655, with the `* 40` literal
replaced by `record_size_bytes` and an overflow guard added):

```c
int pqm_update_dispatch_record(struct process_queue_manager *pqm,
			       unsigned int qid,
			       u64 addr,
			       u32 num_records,
			       u32 record_size_bytes)
{
	struct process_queue_node *pqn;
	struct kfd_process_device *pdd;
	struct amdgpu_vm *vm;
	struct queue *q;
	struct amdgpu_bo *new_bo = NULL;
	int err;

	pqn = get_queue_by_qid(pqm, qid);
	if (!pqn || !pqn->q) {
		pr_debug("No queue %d for dispatch-record update\n", qid);
		return -EFAULT;
	}
	q = pqn->q;

	pdd = kfd_get_process_device_data(q->device, q->process);
	if (!pdd)
		return -ENODEV;

	vm = drm_priv_to_vm(pdd->drm_priv);
	err = amdgpu_bo_reserve(vm->root.bo, false);
	if (err)
		return err;

	if (addr) {
		uint64_t buf_byte_size;

		if (!num_records || !is_power_of_2(num_records) ||
		    !record_size_bytes) {
			amdgpu_bo_unreserve(vm->root.bo);
			return -EINVAL;
		}
		/* overflow guard */
		if (record_size_bytes > U32_MAX / num_records) {
			amdgpu_bo_unreserve(vm->root.bo);
			return -EINVAL;
		}
		buf_byte_size = (uint64_t)num_records * record_size_bytes;

		if (kfd_queue_buffer_get(vm, (void *)addr, &new_bo,
					 buf_byte_size)) {
			pr_debug("profiling buf 0x%llx size 0x%llx not GPU-mapped\n",
				 addr, buf_byte_size);
			amdgpu_bo_unreserve(vm->root.bo);
			return -EFAULT;
		}

		kfd_queue_unref_bo_va(vm, &q->properties.dispatch_record_bo);
		kfd_queue_buffer_put(&q->properties.dispatch_record_bo);
		q->properties.dispatch_record_bo = new_bo;
	} else if (q->properties.dispatch_record_bo) {
		kfd_queue_unref_bo_va(vm, &q->properties.dispatch_record_bo);
		kfd_queue_buffer_put(&q->properties.dispatch_record_bo);
	}

	amdgpu_bo_unreserve(vm->root.bo);

	q->properties.dispatch_record_buffer_addr = addr;
	q->properties.dispatch_record_buffer_size = num_records;

	return q->device->dqm->ops.update_queue(q->device->dqm, q, NULL);
}
```

**Lines to delete from `pqm_update_queue_properties`:**

* 609-655 (the entire dispatch_record `{ ... }` block).
* 662-665 (the two property-copy lines for
  `dispatch_record_buffer_addr/size`).

After this, `pqm_update_queue_properties` matches the `.orig` file
upstream-clean shape.

**Add prototype to `amd/amdkfd/kfd_priv.h`** alongside the other
`pqm_*` declarations.

### 2.4 New profiler sub-op handler in `amd/amdkfd/kfd_chardev.c`

**Sub-op design — combined SET vs separate ENABLE/SET.** Combined
"SET" semantics with `addr == 0` as the disable sentinel. Justification:

* Firmware enable is implicit — MEC checks `addr_lo != 0` per dispatch
  (see `PROFILING_RING_BUFFER.md`), so a separate enable bit would be
  cosmetic.
* Preserves the proven control flow that today's userspace already
  exercises through UPDATE_QUEUE.
* Keeps the new sub-op stateless inside the kernel.

No separate CLEAR sub-op — `addr=0` is the disable sentinel.

**Lock semantics.** `KFD_IOC_PROFILER_DISPATCH_LOG` does NOT require
the device-wide `kfd_profiler_pmc` lock (`kfd_chardev.c:3458-3474`).
Justification:

* Today UPDATE_QUEUE configures the ring without holding that lock.
* Ring buffer setup is per-queue (per-process) state, not contended
  device-wide HW state like PMC counters.
* rocprofiler-sdk's standalone drainer must coexist with PMC clients.

We do require `perfmon_capable()` — matches `kfd_profiler_pmc`'s policy
at `kfd_chardev.c:3487-3488`.

**New args struct in `include/uapi/linux/kfd_ioctl.h`** (added before
`struct kfd_ioctl_profiler_args` at line 1766):

```c
/**
 * kfd_ioctl_dispatch_log_args
 *
 * Sub-op of AMDKFD_IOC_PROFILER (KFD_IOC_PROFILER_DISPATCH_LOG).
 *
 *   SET (op=0):
 *     queue_id, addr, num_records, record_size_bytes -> configure ring;
 *     addr=0 disables. Requires CAP_PERFMON.
 *   GET_DESCRIPTOR (op=1):
 *     addr = userspace destination ptr;
 *     num_records (IN) = caller buffer bytes, (OUT) = bytes required;
 *     desc_version (OUT) = kernel-shipped descriptor version (matches
 *       the static blob compiled into the kernel module).
 *     Unprivileged: any process holding /dev/kfd may call this so
 *     non-profiler tools (crash dumpers, format inspectors, etc.) can
 *     decode rings they observe in core files or shared memory.
 *
 * gpu_id is required for all sub-ops.
 */
struct kfd_ioctl_dispatch_log_args {
	__u32 op;                  /* enum kfd_dispatch_log_op */
	__u32 gpu_id;
	__u32 queue_id;
	__u32 pad0;
	__u64 addr;
	__u32 num_records;
	__u32 record_size_bytes;
	__u32 desc_version;
	__u32 pad1;
};
```

**Offset/size table:**

| Offset | Size | Field             |
|--------|------|-------------------|
| 0x00   | 4    | op                |
| 0x04   | 4    | gpu_id            |
| 0x08   | 4    | queue_id          |
| 0x0C   | 4    | pad0              |
| 0x10   | 8    | addr              |
| 0x18   | 4    | num_records       |
| 0x1C   | 4    | record_size_bytes |
| 0x20   | 4    | desc_version      |
| 0x24   | 4    | pad1              |

Total: 0x28 bytes. 8-byte aligned for `addr`.

**New sub-op enums** (added at `kfd_ioctl.h:1751-1755`):

```c
enum kfd_profiler_ops {
	KFD_IOC_PROFILER_PMC          = 0,
	KFD_IOC_PROFILER_PC_SAMPLE    = 1,
	KFD_IOC_PROFILER_VERSION      = 2,
	KFD_IOC_PROFILER_DISPATCH_LOG = 3,	/* added in 1.23 */
};

enum kfd_dispatch_log_op {
	KFD_IOC_DISPATCH_LOG_SET            = 0,
	KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR = 1,
};
```

**Bump profiler version** at `kfd_ioctl.h:1750`:

```c
#define KFD_IOC_PROFILER_VERSION_NUM 2
```

**Extend the union** at `kfd_ioctl.h:1768-1772`:

```c
struct kfd_ioctl_profiler_args {
	__u32 op;					/* kfd_profiler_op */
	union {
		struct kfd_ioctl_pc_sample_args     pc_sample;
		struct kfd_ioctl_pmc_settings       pmc;
		struct kfd_ioctl_dispatch_log_args  dispatch_log;	/* op = 3 */
		__u32                               version;
	};
};
```

**New handler in `kfd_chardev.c`** (added after `kfd_profiler_pmc` at
~line 3503):

```c
static int kfd_profiler_dispatch_log(struct kfd_process *p,
				     struct kfd_ioctl_dispatch_log_args *args)
{
	struct kfd_process_device *pdd;
	int ret;

	if (args->pad0 || args->pad1)
		return -EINVAL;

	mutex_lock(&p->mutex);
	pdd = kfd_process_device_data_by_id(p, args->gpu_id);
	if (!pdd) {
		mutex_unlock(&p->mutex);
		return -EINVAL;
	}

	switch (args->op) {
	case KFD_IOC_DISPATCH_LOG_SET:
		if (!perfmon_capable()) {
			ret = -EPERM;
			break;
		}
		ret = pqm_update_dispatch_record(&p->pqm,
						 args->queue_id,
						 args->addr,
						 args->num_records,
						 args->record_size_bytes);
		break;
	case KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR:
		/* Unprivileged: any caller may read the static descriptor
		 * for the device's firmware variant. */
		ret = kfd_dispatch_log_get_descriptor(pdd,
						      (void __user *)args->addr,
						      &args->num_records,
						      &args->desc_version);
		break;
	default:
		ret = -EINVAL;
	}

	mutex_unlock(&p->mutex);
	return ret;
}
```

**Extend dispatch in `kfd_ioctl_profiler`** at `kfd_chardev.c:3505-3519`:

```c
static int kfd_ioctl_profiler(struct file *filep, struct kfd_process *p, void *data)
{
	struct kfd_ioctl_profiler_args *args = data;

	switch (args->op) {
	case KFD_IOC_PROFILER_VERSION:
		args->version = KFD_IOC_PROFILER_VERSION_NUM;
		return 0;
	case KFD_IOC_PROFILER_PC_SAMPLE:
		return kfd_ioctl_pc_sample(filep, p, &args->pc_sample);
	case KFD_IOC_PROFILER_PMC:
		return kfd_profiler_pmc(p, &args->pmc);
	case KFD_IOC_PROFILER_DISPATCH_LOG:
		return kfd_profiler_dispatch_log(p, &args->dispatch_log);
	}
	return -EINVAL;
}
```

### 2.5 What stays unchanged

* `kfd_mqd_manager_v9.c:379-392` — the MQD write block stays. Same code
  path runs from `dqm->ops.update_queue`, which
  `pqm_update_dispatch_record` invokes.
* `kfd_priv.h:623-625` — the `queue_properties` fields
  (`dispatch_record_buffer_addr`, `dispatch_record_buffer_size`,
  `dispatch_record_bo`) stay. Kernel-internal state, not UAPI.
* `kfd_queue.c:352, 392` — BO release/unref additions stay.

### 2.6 Disable semantics

`addr=0` continues to be the disable sentinel. No new "CLEAR" sub-op
needed. Matches the existing `kfd_mqd_manager_v9.c:387-392` else-branch
that zeros all four MQD DWords, and the `pqm_update_dispatch_record`
`else if (...dispatch_record_bo)` cleanup branch.

---

## 3. Task 2 — Self-describing JSON descriptor

### 3.A Format choice — JSON

Compared four candidates:

| Format | Pros | Cons | Kernel parser? | Wire size (16-byte record) | Future-proof? |
|---|---|---|---|---|---|
| YAML | Human-readable, comments | Heavy parsers, whitespace-sensitive | No | ~250 bytes | High |
| **JSON** | Universal libraries (rapidjson/nlohmann already in rocprofiler-sdk), strict grammar | No comments | No | ~280 bytes | High |
| Binary TLV | Compact (~40 bytes) | Bespoke schema, no library, harder to debug | No | ~40 bytes | Medium |
| Packed C struct | Smallest (~32 bytes), zero parse cost | ABI-fossilises immediately; defeats self-description | No | ~32 bytes | Low |

**Recommendation: JSON.**

* The kernel stays a dumb byte store in all four options, so format
  choice is determined by *userspace* ergonomics.
* rocprofiler-sdk already links a JSON library; YAML would force a new
  dependency.
* Schema evolution is trivial in JSON — add fields, leave
  `format_version`, ignore unknown keys.
* Wire size is irrelevant — descriptor queried once per queue at
  drainer init.
* Debuggability — developers can read the descriptor with `cat`.

**Where does the descriptor live?** Three options were considered:

1. **Userspace-only**, distributed in libhsakmt or rocprofiler-sdk.
   Rejected because the descriptor describes what *firmware* writes;
   firmware/driver/runtime versions drift independently, and any tool
   not co-built with the producer (crash dumpers, third-party
   profilers, CRIU restore tools) would have to maintain its own
   parallel copy.
2. **Userspace uploads to kernel** via a `SET_DESCRIPTOR` ioctl;
   kernel caches and serves it back. Rejected because (a) it requires
   coordinating who uploads first; (b) any tool that queries before
   the producer uploads gets `-ENOENT`; (c) the kernel becomes the
   storage layer for whatever JSON userspace decided to ship, which
   has no firmware-grounded source of truth.
3. **Kernel ships the descriptor as a static `const char[]`** compiled
   into the KFD module. Per-ASIC-family selection happens inside the
   kernel (via the device's MQD-manager family — gfx9, gfx10, etc.).
   Userspace queries via `GET_DESCRIPTOR`; the kernel `copy_to_user`s
   the static blob.

**Decision: option 3 — kernel-owned static descriptor.** Justification:

* Single source of truth, grounded in the same module that knows the
  firmware/MQD layout.
* Any tool that opens `/dev/kfd` and has a valid `gpu_id` can ask. No
  ordering dependency on the producer side.
* Zero descriptor storage overhead (it's `.rodata`).
* Bumping the descriptor requires a kernel module rebuild — exactly
  the cadence of the firmware/MQD format itself, so no false
  divergence.
* Removes the entire `SET_DESCRIPTOR` sub-op: half the API surface
  goes away.

### 3.B Record-type discrimination — descriptor declares size per type

The descriptor enumerates every known record_type with its fixed size.
The wire record format is **unchanged** from what the firmware writes
today:

```
DW0: ts_lo
DW1: ts_hi
DW2: record_type   (bare 1 = start, 2 = end; firmware unchanged)
DW3: dispatch_idx
```

The parser advances by looking up the size in the descriptor:

```cpp
while (have_records()) {
    uint32_t type = read_u32(ring_ptr + 8);   // DW2
    if (type == 0) break;                      // unwritten slot

    auto it = descriptor.types.find(type);
    if (it == descriptor.types.end()) {
        // Unknown record_type → we can't know its size.
        // Log a clear error and abort the ring scan for this queue.
        log_error("unknown record_type %u in queue %llu — descriptor stale",
                  type, queue_id);
        break;
    }
    process(it->second, ring_ptr);
    ring_ptr += it->second.size;
}
```

#### Why this scheme over an in-record size header

An earlier draft of this design proposed a 4-byte header in every
record `{u8 record_type, u8 hdr_version, u16 record_size_bytes}`,
letting any reader skip past unknown record types. **That was
rejected** for these reasons:

1. **Required a firmware change** to write the size into DW2 of every
   record. Firmware build/sign/deploy cycles are expensive (per
   `FIRMWARE_DESIGN_DOCUMENT.md` §15) and any firmware change carries
   GPU-hang risk.
2. **Failure mode of skew was silent.** If firmware wasn't updated, DW2
   size = 0, parser treats every record as "unwritten slot" → 100%
   data loss with no error logged.
3. **Forced firmware-kernel-userspace lockstep.** Every new record
   type still required descriptor + firmware coordination; the only
   thing the in-record header bought was forward-compat for unknown
   types — at the cost of forcing a header rev whenever the encoding
   itself needed to grow.

The descriptor-declares-size scheme has these properties instead:

* **No firmware change.** Firmware keeps writing exactly what it
  writes today.
* **Loud failure on skew.** Unknown `record_type` produces a logged
  error, not silent data loss.
* **Recovery is mechanical.** Descriptors are software-shippable
  (JSON blobs in libhsakmt or rocprofiler-sdk); a stale descriptor is
  fixed by a userspace release, not a firmware roll.
* **The descriptor will almost always be newer than the firmware.**
  Software rolls more often than firmware; the skew direction
  "firmware ahead of descriptor" is rare.

The single thing lost is the ability for old userspace to skip new
record types it doesn't know about. We accept that — old userspace
should be re-shipped with updated descriptor knowledge before the
firmware that emits new types is deployed.

### 3.C Concrete schema example (JSON)

**Descriptor for the current 16-byte START/END records:**

```json
{
  "format_version": "1.0",
  "ring_id": 0,
  "record_types": [
    {
      "type_id": 1,
      "name": "dispatch_start",
      "size": 16,
      "fields": [
        {"name": "ts_lo",        "offset": 0,  "size": 4, "type": "u32le"},
        {"name": "ts_hi",        "offset": 4,  "size": 4, "type": "u32le"},
        {"name": "record_type",  "offset": 8,  "size": 4, "type": "u32le"},
        {"name": "dispatch_idx", "offset": 12, "size": 4, "type": "u32le"}
      ]
    },
    {
      "type_id": 2,
      "name": "dispatch_end",
      "size": 16,
      "fields": [
        {"name": "ts_lo",        "offset": 0,  "size": 4, "type": "u32le"},
        {"name": "ts_hi",        "offset": 4,  "size": 4, "type": "u32le"},
        {"name": "record_type",  "offset": 8,  "size": 4, "type": "u32le"},
        {"name": "dispatch_idx", "offset": 12, "size": 4, "type": "u32le"}
      ]
    }
  ]
}
```

**Hypothetical future 32-byte record carrying `kernel_object`:**

```json
{
  "type_id": 3,
  "name": "dispatch_start_with_kobj",
  "size": 32,
  "fields": [
    {"name": "ts_lo",         "offset": 0,  "size": 4, "type": "u32le"},
    {"name": "ts_hi",         "offset": 4,  "size": 4, "type": "u32le"},
    {"name": "record_type",   "offset": 8,  "size": 4, "type": "u32le"},
    {"name": "dispatch_idx",  "offset": 12, "size": 4, "type": "u32le"},
    {"name": "kernel_object", "offset": 16, "size": 8, "type": "u64le"},
    {"name": "queue_id",      "offset": 24, "size": 8, "type": "u64le"}
  ]
}
```

A drainer holding an updated descriptor sees `record_type=3` in DW2,
finds it in the descriptor, and consumes a 32-byte record. A drainer
holding a stale descriptor that doesn't list `type_id=3` logs the
"unknown type" error and stops scanning the queue — a clear and
actionable failure, not silent corruption.

### 3.D UAPI surface (`GET_DESCRIPTOR` only)

The only descriptor-related sub-op is
`KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR` (1). It reuses
`struct kfd_ioctl_dispatch_log_args`:

* **GET_DESCRIPTOR:** `addr` = userspace destination pointer;
  `num_records` (in) = caller buffer bytes, (out) = bytes required;
  `desc_version` (out) = the kernel-shipped descriptor version (a
  `KFD_DISPATCH_LOG_DESC_VERSION` constant compiled into the module).
  Returns `-E2BIG` if caller buffer is too small (and `num_records` is
  set to the required size). `gpu_id` selects which device's descriptor
  to return — different ASIC families ship different descriptors.
  `queue_id` and `record_size_bytes` are ignored. **No CAP_PERFMON
  requirement** — any process holding `/dev/kfd` may read.

**Storage:** the descriptor is a `static const char[]` compiled into
the kernel module. No per-process or per-device dynamic allocation;
the kernel just `copy_to_user`s the static blob.

**Layout in kernel source** (proposed):

```
amd/amdkfd/
├── kfd_dispatch_log_descriptor.h   /* version macros, accessor decls */
├── kfd_dispatch_log_descriptor.c   /* static const char gfx9_desc[] = ...; */
```

`kfd_dispatch_log_descriptor.c` defines one `static const char[]` per
ASIC family that supports the firmware ring (today: gfx9 only — see
KFD investigation, `kfd_mqd_manager_v9.c:379-392`). A small accessor
selects the right blob:

```c
struct kfd_dispatch_log_blob {
	const char *bytes;
	u32         len;
	u32         version;
};

const struct kfd_dispatch_log_blob *
kfd_dispatch_log_get_blob(struct kfd_node *dev);
```

The accessor inspects `dev->kfd->mec_fw_version` (or whatever existing
field encodes the family/firmware combo) and returns the matching
blob. If the device's firmware does not support the dispatch ring,
`kfd_dispatch_log_get_blob` returns `NULL` and the ioctl returns
`-ENOTSUPP`.

**Maximum descriptor size:** No cap needed — the blob is part of
`.rodata` and its size is fixed at module-build time. The current
schema (§3.C) is ~1 KB. Comfortable headroom for additional record
types in future kernel rebuilds.

**Helper signature** (in `kfd_chardev.c`, near `kfd_profiler_dispatch_log`):

```c
/*
 * Copy the static dispatch-log descriptor for @pdd's device into
 * @user_buf. @inout_byte_len is in bytes (in: caller buffer size; out:
 * bytes actually required — written even on -E2BIG). @out_version is
 * the descriptor version constant compiled into this kernel module.
 *
 * Returns 0, -ENOTSUPP if the device's firmware doesn't support the
 * dispatch ring, -E2BIG if the caller buffer is too small, or
 * -EFAULT on copy_to_user failure.
 */
static int kfd_dispatch_log_get_descriptor(struct kfd_process_device *pdd,
					   void __user *user_buf,
					   u32 *inout_byte_len,
					   u32 *out_version);
```

Implementation sketch:

```c
static int kfd_dispatch_log_get_descriptor(struct kfd_process_device *pdd,
					   void __user *user_buf,
					   u32 *inout_byte_len,
					   u32 *out_version)
{
	const struct kfd_dispatch_log_blob *blob =
		kfd_dispatch_log_get_blob(pdd->dev);
	u32 caller_size = *inout_byte_len;

	if (!blob)
		return -ENOTSUPP;

	*inout_byte_len = blob->len;
	*out_version    = blob->version;

	if (caller_size < blob->len)
		return -E2BIG;
	if (!user_buf)
		return -EINVAL;
	if (copy_to_user(user_buf, blob->bytes, blob->len))
		return -EFAULT;
	return 0;
}
```

**No new fields in `struct kfd_process_device`.** The kernel does not
need to store the descriptor anywhere — `.rodata` is the storage.

**Encoding conventions** (still apply to the static blob):

* Encoding: UTF-8.
* No null-termination required; length is fixed at compile time.
* Endianness: irrelevant for the descriptor itself (text); the
  descriptor specifies `u32le`/`u16le` etc. for the on-wire records.

### 3.E Userspace flow (rocprofiler-sdk)

Files affected:

* `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.cpp`
* `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/kernel_dispatch/firmware_ring_drainer.hpp`
* `projects/rocprofiler-sdk/source/lib/rocprofiler-sdk/hsa/queue_controller.cpp`

**Drainer init flow:**

1. Open KFD device.
2. Issue `AMDKFD_IOC_PROFILER` with `op = KFD_IOC_PROFILER_VERSION` to
   confirm version ≥ 2.
3. If ≥ 2: issue
   `op = KFD_IOC_PROFILER_DISPATCH_LOG, sub_op = GET_DESCRIPTOR` with
   `gpu_id` for each device. First call with `num_records = 0` to learn
   the required size; second call with the right buffer to receive the
   blob. Parse JSON into a `RecordDescriptor` C++ struct (one entry
   per `type_id` holding `{size, fields}`).
4. If `-ENOTSUPP` (firmware doesn't support dispatch ring): fall back
   to hardcoded `mec_dispatch_record_16` parser if any older firmware
   ring records show up; otherwise simply skip drainer registration
   for this device. Log once.
5. If profiler version < 2: fall back to hardcoded parser. Log once.

**Per-record parse loop replacement.** Today's drainer (per
`firmware_ring_drainer.cpp` `mec_dispatch_record_16`) casts each 16-byte
slot to a hardcoded struct and dispatches on `record_type`. New loop:

```cpp
while (have_more_records()) {
    uint32_t type = read_u32(ring_ptr + 8);   // DW2
    if (type == 0) break;                      // unwritten slot

    auto it = descriptor.types.find(type);
    if (it == descriptor.types.end()) {
        log_error("unknown record_type %u in queue %llu — descriptor stale",
                  type, queue_id);
        break;  // abort ring scan; do not advance by guess
    }
    process_record(it->second, ring_ptr);
    ring_ptr += it->second.size;
}
```

The 16-byte fast path stays as a separate code path gated on the
fallback flag — small enough to keep, removes perf-regression risk
during migration.

**Migration path:**

* Phase 1 (this PR set): kernel ships descriptor as `.rodata`;
  rocprofiler-sdk reads it via `GET_DESCRIPTOR`. Producers
  (libhsakmt, ROCr) need no changes for the descriptor — they just
  call the new `SET` sub-op for ring buffer setup.
* Phase 2: once telemetry confirms all consumers go through the
  descriptor, remove the 16-byte fallback in rocprofiler-sdk.

### 3.F Versioning

Three independent levels:

1. **JSON document version** (`"format_version": "1.0"`). Userspace
   bumps the minor when adding optional fields; bumps major on
   breaking changes. Drainer warn-and-fallbacks on unsupported major.
2. **Kernel sub-op set version** (`KFD_IOC_PROFILER_VERSION_NUM` 1→2)
   indicates the `DISPATCH_LOG` sub-op family is available.
3. **`desc_version` field** (compiled-in constant
   `KFD_DISPATCH_LOG_DESC_VERSION` returned by `GET_DESCRIPTOR`). Bumped
   in lockstep with the static blob whenever the kernel rebuilds with a
   different schema. Userspace can cache parsed descriptors keyed on
   `(gpu_id, desc_version)` and reuse across processes if the version
   matches.

Userspace must check both (1) format_version major and (2) profiler
version ≥ 2 before issuing the new sub-op.

---

## 4. Combined UAPI changes — verbatim additions to `kfd_ioctl.h`

Showing the complete state of the changed regions after both tasks.

**Lines 30-53 (changelog + version):**

```c
/*
 * - 1.1 - initial version
 * - 1.3 - Add SMI events support
 * ...
 * - 1.22 - Add queue creation with metadata ring base address
 * - 1.23 - Move dispatch_record_buffer setup from UPDATE_QUEUE to PROFILER ioctl
 */
#define KFD_IOCTL_MAJOR_VERSION 1
#define KFD_IOCTL_MINOR_VERSION 23
```

**Lines 99-109 (UPDATE_QUEUE args, reverted):**

```c
struct kfd_ioctl_update_queue_args {
	__u64 ring_base_address;	/* to KFD */
	__u32 queue_id;			/* to KFD */
	__u32 ring_size;		/* to KFD */
	__u32 queue_percentage;		/* to KFD */
	__u32 queue_priority;		/* to KFD */
};
```

**Lines 1750-1773 (profiler args expanded):**

```c
#define KFD_IOC_PROFILER_VERSION_NUM 2
enum kfd_profiler_ops {
	KFD_IOC_PROFILER_PMC          = 0,
	KFD_IOC_PROFILER_PC_SAMPLE    = 1,
	KFD_IOC_PROFILER_VERSION      = 2,
	KFD_IOC_PROFILER_DISPATCH_LOG = 3,
};

enum kfd_dispatch_log_op {
	KFD_IOC_DISPATCH_LOG_SET            = 0,
	KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR = 1,
};

/* Compiled-in descriptor version. Bumped whenever the static blob in
 * kfd_dispatch_log_descriptor.c changes (new record_type, new field,
 * etc.). Userspace returns this in dispatch_log.desc_version on
 * GET_DESCRIPTOR. */
#define KFD_DISPATCH_LOG_DESC_VERSION 1

/**
 * Enables/Disables GPU Specific profiler settings
 */
struct kfd_ioctl_pmc_settings {
	__u32 gpu_id;             /* This is the user_gpu_id */
	__u32 lock;               /* Lock GPU for Profiling */
	__u32 perfcount_enable;   /* Force Perfcount Enable for queues on GPU */
};

struct kfd_ioctl_dispatch_log_args {
	__u32 op;                  /* enum kfd_dispatch_log_op */
	__u32 gpu_id;
	__u32 queue_id;
	__u32 pad0;
	__u64 addr;
	__u32 num_records;
	__u32 record_size_bytes;
	__u32 desc_version;
	__u32 pad1;
};

struct kfd_ioctl_profiler_args {
	__u32 op;					/* kfd_profiler_op */
	union {
		struct kfd_ioctl_pc_sample_args     pc_sample;
		struct kfd_ioctl_pmc_settings       pmc;
		struct kfd_ioctl_dispatch_log_args  dispatch_log;
		__u32                               version;
	};
};
```

No changes to `AMDKFD_IOC_PROFILER` definition at line 1981-1982 — the
ioctl number and macro stay; only the union grows.

---

## 5. Migration / compatibility analysis

| Component | Change required | Where | Breakage if not done |
|---|---|---|---|
| **KFD (kernel)** | Revert UPDATE_QUEUE fields, add new sub-op (SET + GET_DESCRIPTOR), add static `.rodata` descriptor blob + accessor | `kfd_ioctl.h`, `kfd_chardev.c`, `kfd_process_queue_manager.c`, `kfd_priv.h`, **new** `kfd_dispatch_log_descriptor.{c,h}` | — (this is the change) |
| **MEC firmware** | **None.** Wire format unchanged. | — | — |
| **libhsakmt** | Replace `hsaKmtSetQueueProfilingBuffer` with `hsaKmtSetQueueDispatchLog` calling new ioctl (SET only — no descriptor upload) | rocm-systems libhsakmt | Build failure: UPDATE_QUEUE no longer has dispatch_record fields |
| **ROCr** `AqlQueue::SetProfiling` | Switch to new thunk for ring buffer setup. Does NOT upload a descriptor (kernel owns it). | `rocr-runtime/.../amd_aql_queue.cpp` | Profiling silently no-ops |
| **rocprofiler-sdk** | (a) Query descriptor from kernel at drainer init via `GET_DESCRIPTOR` (§3.E). (b) Replace hardcoded `mec_dispatch_record_16` with descriptor-driven parser. (c) Keep 16-byte fallback for older kernels. | `firmware_ring_drainer.{cpp,hpp}`, `queue_controller.cpp` | Drainer reads correctly today (16-byte fast path) but blocks Phase 2 cleanup |
| **cpc_tracing branch consumers** | Audit any branch-private code that touches `properties.dispatch_record_buffer_addr` directly | Branch-local | Branch-private; trivial |
| **Firmware-onboarding test machine** (`gbt350-odcdh5-wbc1-b.png-odc.dcgpu`) | Re-deploy: amdgpu module (new UAPI + descriptor blob), libhsakmt (.so), libhsa-runtime64 (.so), rocprofiler-sdk (.so). Firmware untouched. | Remote | Profiling stops working until userspace + kernel match |
| **Older userspace running on new kernel** | Will trigger `_IOC_SIZE` mismatch in `kfd_ioctl()` (`kfd_chardev.c:3820-3823`). Behavior: ioctl works but tail bytes are ignored. | N/A | Acceptable on private branch |
| **Other tools (crash dumpers, third-party profilers)** | Can now ask KFD for the format directly via `GET_DESCRIPTOR` — no co-build with libhsakmt/ROCr required. | N/A | — (this is a new capability) |

Critically: **the firmware row used to read "MEC firmware: Write
0x00100001/0x00100002 to DW2 instead of bare 1/2" — that row is now
gone.** No firmware coordination needed.

---

## 6. Test plan

### 6.1 Build verification

* Kernel module builds cleanly. No warnings about unused
  `dispatch_record_buffer_addr/size` in `kfd_chardev.c`.
* `git diff` against `.orig` files shows: chardev / UAPI now match
  `.orig`; pqm has the new function added but the inline block at
  609-655 is gone.

### 6.2 Sanity test — dispatch ring still records

After kernel + userspace coordinated rebuild:

```
cd /home/djavady/proton
./test_profbuf_hsa_api
```

Expected: same "NON-ZERO DATA FOUND" output as today. DW2 of each
record still reads `0x00000001` (start) or `0x00000002` (end) — wire
format unchanged.

### 6.3 Profiler ioctl version probe

Open `/dev/kfd`, issue `AMDKFD_IOC_PROFILER` with
`op=KFD_IOC_PROFILER_VERSION`. Expect `version == 2`.

### 6.4 Disable round-trip

Set with `addr != 0`, then set with `addr == 0`. Verify via debug
logging or sysfs that DW43-47 zero in MQD. Re-set with `addr != 0`,
confirm records appear again.

### 6.5 Descriptor query — size probe

Issue `GET_DESCRIPTOR` with `num_records = 0` (no buffer). Expect
`-E2BIG`, `num_records` returned as the static blob size,
`desc_version` returned as `KFD_DISPATCH_LOG_DESC_VERSION`.

### 6.6 Descriptor query — full read

Allocate a buffer of size from the probe; call `GET_DESCRIPTOR` with
that buffer. Expect 0; bytes copied are byte-for-byte identical to
the static blob in `kfd_dispatch_log_descriptor.c`. Parse the JSON;
verify it contains `format_version`, `record_types`, etc.

### 6.7 Unsupported device

Run `GET_DESCRIPTOR` against a `gpu_id` whose firmware doesn't
support the dispatch ring (e.g., a gfx10 device on a multi-GPU
system). Expect `-ENOTSUPP`.

### 6.8 Unprivileged access to GET_DESCRIPTOR

Run `GET_DESCRIPTOR` as a regular (non-CAP_PERFMON) user. Expect
success — descriptor is read-only static data and should be available
to any tool inspecting ring buffers (e.g., a crash-dump decoder).

### 6.9 Privileged access to SET

Run `DISPATCH_LOG_SET` as a non-CAP_PERFMON user. Expect `-EPERM`.

### 6.10 Concurrent PMC + DISPATCH_LOG

Acquire PMC lock from one process; from another, issue
DISPATCH_LOG_SET on a different queue. Expect success — the PMC
profiler_lock must NOT block dispatch_log.

### 6.11 Negative tests

* Bad `op`: `-EINVAL`.
* `pad0` or `pad1` non-zero: `-EINVAL`.
* `gpu_id` invalid: `-EINVAL`.
* `queue_id` invalid (SET): `-EFAULT`.
* `num_records` not power-of-2 with `addr != 0`: `-EINVAL`.
* `record_size_bytes = 0` with `addr != 0`: `-EINVAL`.
* `num_records * record_size_bytes` overflows u32: `-EINVAL`.

### 6.12 rocprofiler-sdk integration

Run `rocprofv3 --kernel-trace ./test_profbuf_hsa_api`. Expected: same
kernel trace output as today; per-dispatch start/end timestamps in JSON.

Inject a fake `record_type=99` into the ring (via debug write) and
verify the drainer logs "unknown record_type 99 — descriptor stale"
and stops scanning that queue (does NOT advance and corrupt subsequent
records).

### 6.13 Userspace fallback path

Run an older rocprofiler-sdk (without descriptor support) against the
new kernel. It uses the hardcoded 16-byte parser and produces correct
output. Run a new rocprofiler-sdk against an older kernel (one without
profiler version ≥ 2): same fallback path, produces correct output.

### 6.14 Third-party tool reading the format

Stand-alone test program: open `/dev/kfd`, issue `GET_DESCRIPTOR`
without ever issuing SET. Confirm the JSON blob comes back. Validates
that any tool can decode the ring without coordinating with
libhsakmt/ROCr.

---

## 7. Open questions

1. **Is `pqm_update_dispatch_record` correct to call
   `dqm->ops.update_queue` unconditionally?** Today's
   `pqm_update_queue_properties` calls it once after both ring-bo and
   dispatch-record-bo work. By splitting them we now call
   `update_queue` from two paths if both UPDATE_QUEUE and
   DISPATCH_LOG fire on the same queue back-to-back. The MQD writes
   are idempotent so this is safe, but it doubles the HWS preemption
   latency. Acceptable? (Likely yes — DISPATCH_LOG is rare; profiling
   setup is one-shot per queue.)
2. **Should `pqm_update_dispatch_record` defend against the queue
   being mid-destroy?** `get_queue_by_qid` doesn't take a lock beyond
   `p->mutex`. Today's code has the same exposure.
3. **Per-ASIC-family blob selection.** The accessor
   `kfd_dispatch_log_get_blob` needs to map the device to a blob.
   Open: do we key on ASIC family (gfx9/gfx10/...) only, or on
   firmware version too? gfx9 covers MI200/MI300/MI350 — same family,
   but only MI350 firmware writes the dispatch ring today. If the
   blob is keyed by firmware version we can return `-ENOTSUPP` for
   non-supporting MI200 firmware automatically. Recommended:
   firmware-version-keyed, with a fallback to family for forward
   compatibility.
4. **The `record_size_bytes` field in SET is duplicate info** (also
   encoded in the descriptor). They can disagree → kernel sizes BO
   check against ioctl field, firmware writes records of descriptor's
   declared size, mismatch = corruption. For Phase 1: document that
   `record_size_bytes` is authoritative for kernel BO validation
   only, and userspace MUST keep it in sync with what it parsed from
   the descriptor. For Phase 2: kernel could parse the JSON enough to
   extract the size for type_id=1 (dispatch start) — but this brings
   parsing into the kernel, which we want to avoid.
5. **Should the static blob be per-`kfd_node` rather than per
   `kfd_dev`?** Multi-XCD MI300/MI350 expose multiple `kfd_node`s per
   `kfd_dev`. Open whether the descriptor needs per-node specialization
   (probably not — wire format is determined by firmware, which is
   per-`kfd_dev`).

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Static blob in kernel goes stale relative to firmware (firmware adds a new `record_type` that the kernel-shipped descriptor doesn't list) | Low (kernel rebuilds in lockstep with firmware drops) | Medium (drainer aborts queue scan with clear logged error — not silent corruption) | Bump `KFD_DISPATCH_LOG_DESC_VERSION` and update the blob whenever firmware adds a record type. Detect via "unknown record_type N" log line. Coordination point is the kernel/firmware release, not a userspace coordination. |
| R2 | libhsakmt + ROCr roll out before kernel — userspace tries new ioctl on old kernel | Medium | Profiling silently disabled | Userspace probes `KFD_IOC_PROFILER_VERSION` and falls back to legacy UPDATE_QUEUE path on version < 2. But UPDATE_QUEUE no longer accepts the fields in the new userspace build — fallback requires keeping the old code paths in libhsakmt/ROCr behind a runtime check. Document in PR. |
| R3 | Splitting `pqm_update_queue_properties` introduces a regression in the queue update path | Medium | High (any KFD queue update breaks) | Diff the new `pqm_update_queue_properties` against `pqm.c.orig` — should be byte-identical. CI: run `kfdtest` on the test machine. |
| R4 | `KFD_IOC_PROFILER_VERSION_NUM` bump breaks existing PMC/PC_SAMPLE callers that gate on `version == 1` | Low | Low | Convention is `>=`, not `==`. Audit; current callers use `>=`. |
| R5 | `record_size_bytes` field in SET disagrees with descriptor's declared size | Medium | Medium (kernel sizes BO check based on ioctl field; firmware writes per-its-knowledge; mismatch = corruption) | Phase 1: document as caller responsibility (userspace parses descriptor and passes the correct size). Phase 2: kernel could parse just enough JSON to derive size — but this brings parsing into kernel, which we want to avoid. |
| R6 | `GET_DESCRIPTOR` returns `-ENOTSUPP` on unsupported devices but caller doesn't distinguish from other errors | Low | Low | Document the error code. Userspace sample code shows the right pattern. |
| R7 | Descriptor blob increases kernel module image size | Low | Negligible | ~1 KB per ASIC family in `.rodata`. Module size growth is tiny. |

---

**End of plan.** No code changes pending. See `KNOWN_ISSUES.md` for the
broader context this fits into and `FIRMWARE_RING_HYBRID_DESIGN.md` for
the related rocprofiler-sdk work that consumes the descriptor.
