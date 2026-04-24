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
union-discriminated profiler ioctl, with sub-ops `SET`,
`SET_DESCRIPTOR`, `GET_DESCRIPTOR`. The handler reuses
`pqm_update_queue_properties`'s machinery via a new extracted helper
`pqm_update_dispatch_record`, so the MQD write at
`kfd_mqd_manager_v9.c:379-392` is unchanged.

**Task 2 — Self-describing JSON descriptor.** Userspace uploads a
JSON blob describing each record type (id, name, size, fields).
The kernel stores it opaquely per-`kfd_process_device` and returns it
on demand. The kernel never parses it. The drainer in rocprofiler-sdk
queries the descriptor at init and uses it as the source of truth for
record size and field offsets. **No firmware change is needed** — the
size for each `record_type` is declared in the descriptor, not encoded
in the wire record.

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
 *     addr=0 disables.
 *   SET_DESCRIPTOR (op=1):
 *     addr = userspace ptr to descriptor bytes (cast to __u64);
 *     num_records = byte length (max KFD_DISPATCH_LOG_DESC_MAX_BYTES);
 *     desc_version = monotonic version (caller's choice).
 *   GET_DESCRIPTOR (op=2):
 *     addr = userspace destination ptr;
 *     num_records (IN) = caller buffer bytes, (OUT) = bytes written;
 *     desc_version (OUT) = stored version.
 *
 * gpu_id is required for all sub-ops.
 * Requires CAP_PERFMON (perfmon_capable()).
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
	KFD_IOC_DISPATCH_LOG_SET_DESCRIPTOR = 1,
	KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR = 2,
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

	if (!perfmon_capable())
		return -EPERM;

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
		ret = pqm_update_dispatch_record(&p->pqm,
						 args->queue_id,
						 args->addr,
						 args->num_records,
						 args->record_size_bytes);
		break;
	case KFD_IOC_DISPATCH_LOG_SET_DESCRIPTOR:
		ret = kfd_dispatch_log_set_descriptor(pdd,
						      (void __user *)args->addr,
						      args->num_records,
						      args->desc_version);
		break;
	case KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR:
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

**Should the descriptor live entirely in userspace?** Considered.
Rejected because:

1. The descriptor describes what *firmware* writes. Firmware/driver/
   runtime versions can drift independently. The kernel knows which
   firmware loaded; it's the natural source of truth.
2. The drainer in rocprofiler-sdk doesn't necessarily share a build
   with the producer — test harnesses, CRIU restores, different SDK
   versions may attach.
3. Kernel storage costs are tiny (8 KB cap per `kfd_process_device`).

**Decision:** kernel stores the bytes; userspace sets/gets via the new
sub-ops.

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

### 3.D UAPI surface (descriptor sub-ops)

Sub-ops `KFD_IOC_DISPATCH_LOG_SET_DESCRIPTOR` (1) and
`KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR` (2) reuse
`struct kfd_ioctl_dispatch_log_args` with these field meanings:

* **SET_DESCRIPTOR:** `addr` = userspace pointer to descriptor bytes;
  `num_records` = byte length; `desc_version` = monotonic version.
  `queue_id` and `record_size_bytes` ignored.
* **GET_DESCRIPTOR:** `addr` = userspace destination pointer;
  `num_records` (in) = caller buffer size, (out) = bytes actually
  written; `desc_version` (out) = stored version. Returns `-ENOENT` if
  no descriptor stored, `-E2BIG` if caller buffer too small (and
  `num_records` is set to required size).

**Maximum descriptor size:** 8 KB
(`#define KFD_DISPATCH_LOG_DESC_MAX_BYTES 8192`).

* Example schema is ~1 KB; 8× headroom covers ~30 record types or
  richer per-field metadata.
* Kernel cap prevents userspace from forcing arbitrary kernel
  allocations.
* Fits in a `kvmalloc` — no high-order page allocation risk.

**Storage location:** per-`kfd_process_device`.

* Per-queue is too granular — descriptor describes wire format, which
  is the same across queues on the same device given the same firmware.
* Per-`kfd_dev` would prevent two cooperating processes from disagreeing
  during a userspace upgrade window.
* Per-process-per-device is the natural scope of "this process's view
  of this device's profiling format."

**Storage in `struct kfd_process_device`** (added in `kfd_priv.h` near
other per-device process state):

```c
	/* Dispatch-log format descriptor (opaque to kernel; set/get via
	 * KFD_IOC_PROFILER_DISPATCH_LOG. NULL until first SET_DESCRIPTOR.
	 * Protected by p->mutex. */
	void     *dispatch_log_descriptor;
	u32       dispatch_log_descriptor_len;
	u32       dispatch_log_descriptor_version;
```

Free in the existing `kfd_process_device` teardown path.

**Encoding conventions:**

* Encoding: UTF-8.
* No null-termination required; length carried in `num_records`.
* Endianness: irrelevant for the descriptor itself (text); the
  descriptor specifies `u32le`/`u16le` etc. for the on-wire records.

**Helper signatures** (in `kfd_chardev.c`, near `kfd_profiler_dispatch_log`):

```c
static int kfd_dispatch_log_set_descriptor(struct kfd_process_device *pdd,
					   void __user *user_buf,
					   u32 byte_len,
					   u32 version);

static int kfd_dispatch_log_get_descriptor(struct kfd_process_device *pdd,
					   void __user *user_buf,
					   u32 *inout_byte_len,
					   u32 *out_version);
```

* SET: validates `byte_len <= KFD_DISPATCH_LOG_DESC_MAX_BYTES`,
  `kvzalloc`s, `copy_from_user`s, swaps under `p->mutex` (caller already
  holds), `kvfree`s old buffer.
* GET: validates `*inout_byte_len >= dispatch_log_descriptor_len`
  (else `-E2BIG` and writes required size to `*inout_byte_len`),
  `copy_to_user`s, returns version.

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
   `op = KFD_IOC_PROFILER_DISPATCH_LOG, sub_op = GET_DESCRIPTOR`. On
   success, parse JSON into a `RecordDescriptor` C++ struct (one entry
   per `type_id` holding `{size, fields}`).
4. If `-ENOENT` (no descriptor stored): fall back to hardcoded
   `mec_dispatch_record_16` parser. Log a one-shot warning.
5. If profiler version < 2: same fallback. Log a one-shot warning.

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

* Phase 1 (this PR set): kernel stores descriptor; rocprofiler-sdk
  reads it; producers (libhsakmt or ROCr) start populating it via
  SET_DESCRIPTOR.
* Phase 2: once telemetry confirms all consumers go through the
  descriptor, remove the 16-byte fallback in rocprofiler-sdk.

### 3.F Versioning

Three independent levels:

1. **JSON document version** (`"format_version": "1.0"`). Userspace
   bumps the minor when adding optional fields; bumps major on
   breaking changes. Drainer warn-and-fallbacks on unsupported major.
2. **Kernel sub-op set version** (`KFD_IOC_PROFILER_VERSION_NUM` 1→2)
   indicates the `DISPATCH_LOG` sub-op family is available.
3. **`desc_version` field** (kernel-stored, monotonic, set by producer
   at SET_DESCRIPTOR time; returned by GET_DESCRIPTOR). Lets consumers
   detect re-uploads (after process re-init) without re-parsing.

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
	KFD_IOC_DISPATCH_LOG_SET_DESCRIPTOR = 1,
	KFD_IOC_DISPATCH_LOG_GET_DESCRIPTOR = 2,
};

#define KFD_DISPATCH_LOG_DESC_MAX_BYTES 8192

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
| **KFD (kernel)** | Revert UPDATE_QUEUE fields, add new sub-op + descriptor storage | `kfd_ioctl.h`, `kfd_chardev.c`, `kfd_process_queue_manager.c`, `kfd_priv.h` | — (this is the change) |
| **MEC firmware** | **None.** Wire format unchanged. | — | — |
| **libhsakmt** | Replace `hsaKmtSetQueueProfilingBuffer` with `hsaKmtSetQueueDispatchLog` calling new ioctl | rocm-systems libhsakmt | Build failure: UPDATE_QUEUE no longer has dispatch_record fields |
| **ROCr** `AqlQueue::SetProfiling` | Switch to new thunk; on first init also upload descriptor via SET_DESCRIPTOR | `rocr-runtime/.../amd_aql_queue.cpp` | Profiling silently no-ops |
| **rocprofiler-sdk** | (a) Query descriptor at drainer init (§3.E). (b) Replace hardcoded `mec_dispatch_record_16` with descriptor-driven parser. (c) Keep 16-byte fallback. | `firmware_ring_drainer.{cpp,hpp}`, `queue_controller.cpp` | Drainer reads correctly today (16-byte fast path) but blocks Phase 2 cleanup |
| **cpc_tracing branch consumers** | Audit any branch-private code that touches `properties.dispatch_record_buffer_addr` directly | Branch-local | Branch-private; trivial |
| **Firmware-onboarding test machine** (`gbt350-odcdh5-wbc1-b.png-odc.dcgpu`) | Re-deploy: amdgpu module (new UAPI), libhsakmt (.so), libhsa-runtime64 (.so), rocprofiler-sdk (.so). Firmware untouched. | Remote | Profiling stops working until userspace + kernel match |
| **Older userspace running on new kernel** | Will trigger `_IOC_SIZE` mismatch in `kfd_ioctl()` (`kfd_chardev.c:3820-3823`). Behavior: ioctl works but tail bytes are ignored. | N/A | Acceptable on private branch |

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

### 6.5 Descriptor round-trip

Set descriptor (a 1 KB JSON blob). Get it back. Compare bytes — should
be byte-for-byte identical. `desc_version` returned matches set value.

### 6.6 Descriptor size cap

* Set descriptor with `byte_len = 8192`: succeeds.
* Set with `byte_len = 8193`: `-EINVAL`.
* Set with `byte_len = 0`: `-EINVAL` (recommended; reserve "clear" for
  a future explicit op if needed).

### 6.7 GET_DESCRIPTOR with too-small buffer

Set 1 KB descriptor, call GET with `byte_len = 100`. Expect `-E2BIG`,
and `num_records` returned as 1024. Retry with correct size: success.

### 6.8 Permission test

Run dispatch-log SET as a non-CAP_PERFMON user. Expect `-EPERM`.

### 6.9 Concurrent PMC + DISPATCH_LOG

Acquire PMC lock from one process; from another, issue
DISPATCH_LOG_SET on a different queue. Expect success — the PMC
profiler_lock must NOT block dispatch_log.

### 6.10 Negative tests

* Bad `op`: `-EINVAL`.
* `pad0` or `pad1` non-zero: `-EINVAL`.
* `gpu_id` invalid: `-EINVAL`.
* `queue_id` invalid (SET): `-EFAULT`.
* `num_records` not power-of-2 with `addr != 0`: `-EINVAL`.
* `record_size_bytes = 0` with `addr != 0`: `-EINVAL`.
* `num_records * record_size_bytes` overflows u32: `-EINVAL`.

### 6.11 rocprofiler-sdk integration

Run `rocprofv3 --kernel-trace ./test_profbuf_hsa_api`. Expected: same
kernel trace output as today; per-dispatch start/end timestamps in JSON.

Inject a fake `record_type=99` into the ring (via debug write) and
verify the drainer logs "unknown record_type 99 — descriptor stale"
and stops scanning that queue (does NOT advance and corrupt subsequent
records).

### 6.12 Userspace fallback path

Run an older rocprofiler-sdk (without descriptor support) against the
new kernel. It will not call SET_DESCRIPTOR; the kernel returns
`-ENOENT` on GET_DESCRIPTOR. The drainer must fall back to hardcoded
16-byte parsing and produce correct output.

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
3. **Is per-`kfd_process_device` storage right, or should we put the
   descriptor in `struct kfd_dev` (per-device global, set by the first
   process and shared)?** Per-process is safer for multi-tenant;
   per-device avoids redundant uploads. Recommended per-process but
   flagging.
4. **Should the descriptor be queryable by an unprivileged process?**
   Today only `perfmon_capable()` can call DISPATCH_LOG. If a
   non-profiler tool wants to know the wire format (e.g., for crash
   dumps), it can't. Possible mitigation: split GET_DESCRIPTOR off as
   unprivileged, keep SET as `perfmon_capable()`-only.
5. **The `record_size_bytes` field in SET is duplicate info** (also
   encoded in the descriptor). They can disagree → kernel sizes BO
   check against ioctl field, firmware writes records of descriptor's
   declared size, mismatch = corruption. For Phase 1: document that
   `record_size_bytes` is authoritative for kernel BO validation
   only, and userspace MUST keep it in sync. For Phase 2: derive it
   from the descriptor at SET-time.

---

## 8. Risk register

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| R1 | Descriptor stale relative to firmware (firmware adds new record_type that descriptor doesn't know) | Low (firmware rolls slowly; descriptor ships with software) | Medium (drainer aborts queue scan with clear logged error — not silent corruption) | Ship descriptor in lockstep with firmware updates. Descriptor is software-shippable so this is a normal SDK release operation. Detect via "unknown record_type N" log line. |
| R2 | libhsakmt + ROCr roll out before kernel — userspace tries new ioctl on old kernel | Medium | Profiling silently disabled | Userspace probes `KFD_IOC_PROFILER_VERSION` and falls back to legacy UPDATE_QUEUE path on version < 2. But UPDATE_QUEUE no longer accepts the fields in the new userspace build — fallback requires keeping the old code paths in libhsakmt/ROCr behind a runtime check. Document in PR. |
| R3 | Descriptor JSON payload triggers OOM if cap is too high | Low | Medium (DoS by privileged process) | Cap at 8 KB; use `kvzalloc`; cap enforced before allocation |
| R4 | rocprofiler-sdk's standalone drainer registered under a context without CAP_PERFMON | Low | Drainer can't query descriptor → fallback to 16-byte hardcoded path | Document; track in telemetry whether descriptor query succeeded |
| R5 | Splitting `pqm_update_queue_properties` introduces a regression in the queue update path | Medium | High (any KFD queue update breaks) | Diff the new `pqm_update_queue_properties` against `pqm.c.orig` — should be byte-identical. CI: run `kfdtest` on the test machine. |
| R6 | `KFD_IOC_PROFILER_VERSION_NUM` bump breaks existing PMC/PC_SAMPLE callers that gate on `version == 1` | Low | Low | Convention is `>=`, not `==`. Audit; current callers use `>=`. |
| R7 | per-`kfd_process_device` descriptor storage leaks on process abnormal exit | Low | Low (small bounded leak per process) | Hook into existing `kfd_process_device` teardown; `kvfree(pdd->dispatch_log_descriptor)`. |
| R8 | `record_size_bytes` field in SET disagrees with descriptor's declared size | Medium | Medium (kernel sizes BO check based on ioctl field; firmware writes per-its-knowledge; mismatch = corruption) | Phase 1: document as caller responsibility. Phase 2: derive from descriptor at SET-time. |
| R9 | Descriptor not present at drainer init (older producer hasn't called SET_DESCRIPTOR) | Medium during transition | Low | Fall back to hardcoded 16-byte parser; log a warning. |

---

**End of plan.** No code changes pending. See `KNOWN_ISSUES.md` for the
broader context this fits into and `FIRMWARE_RING_HYBRID_DESIGN.md` for
the related rocprofiler-sdk work that consumes the descriptor.
