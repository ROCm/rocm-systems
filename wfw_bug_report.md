# Bug Report: Unit_hipStreamQuery_WithFinishedWork — APERTURE_VIOLATION on null stream dispatch

**Date:** 2026-08-04  
**Repro machine:** heliosp-1b114-c01-3.mnb.dcgpu (gfx1250, 1P1G Helios-P)  
**Affected builds:** 10.1.x nightlies (0803_dev `0888ca35`, 0803 nightly `10.1.0a20260804`)  
**Not affected:** 10.0.x (0729 nightly `10.0.0a20260730`)  
**Owner:** Saleel K (PR #7727 author)

**NOT reproduced on:** e21-14 (ctheliosp-aris-e21-14, Anacapa 1P4G) — `device_mem_ring_buf=0` on that platform, MOVDIR64B path not active

---

## Platform Specificity — Root Cause

Tested on both machines with `DEBUG_CLR_AQL_DEV_QUEUE=1` to force `device_mem_ring_buf=1`:
- **c01-3 (Helios-P, PCIe):** `device_mem_ring_buf 1` → **REPRODUCES 7/10**
- **e21-14 (Anacapa, IFoE):** `device_mem_ring_buf 1` (forced) → **CLEAN 30/30**

So `device_mem_ring_buf=1` alone is not the discriminator. The actual difference:

| Property | c01-3 Helios-P | e21-14 Anacapa |
|---|---|---|
| Interconnect | PCIe (non-coherent BAR) | IFoE fabric |
| HSA Profile | `HSA_PROFILE_BASE` | `HSA_PROFILE_FULL` |
| `hostUnifiedMemory_` | 0 | 1 |
| `Coherent Host Access` | FALSE | **TRUE** |

**Root cause:** On coherent platforms (`HSA_PROFILE_FULL`, Anacapa/IFoE), CPU writes to GPU memory via MOVDIR64B are immediately visible to the GPU — `skip_fence=true` is safe. On non-coherent PCIe (`HSA_PROFILE_BASE`, Helios-P), MOVDIR64B closes the CPU WC buffer (posts the write to PCIe) but does NOT guarantee delivery before the subsequent UC MMIO doorbell write. Without the sfence, the CP can ring the doorbell, fetch the ring-buffer slot, and read the old `kernel_object`.

The faulty assumption in `nontemporal.hpp` comment: *"MOVDIR64B already closed the WC buffer, so the dispatch packet is posted before the UC doorbell store reaches the GPU."* This holds for coherent fabric but NOT for non-coherent PCIe, where "posted" ≠ "delivered".

---

## Summary

`Unit_hipStreamQuery_WithFinishedWork` ("Null Stream" section) fails with
`HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION` when dispatching `hip::stream::empty_kernel`
to the null stream. The bug is **state-dependent** — it only triggers after specific
preceding stream tests leave the null stream's HW queue in a recycled state.
Failure rate: **7/10** with the trigger sequence, **0/20** in isolation.

The regression was introduced by CLR PR #7727 (`feat(clr): Use device-resident queue
ring buffers`, commit `f79ca3d3a9ee`, author Saleel K, landed 2026-07-21 in the 10.1
branch).

---

## Failure Signature

```
Warning: Queue error: HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION: The agent attempted
to access memory beyond the largest legal address.

VGPU(0x557f328f4b70) hang analysis:
SWq=0x7f41de348000, HWq=0x7f41d6c00000, id=1, priority=1,
Dispatch Header = 0xb00 (type=0, barrier=1, acquire=1, release=1),
cluster_count=[1, 1, 1], cluster_size=[1, 1, 1], workgroup=[1, 1, 1],
kernel_obj=0x7f41ea030580   ← host VA in GPU aperture = APERTURE_VIOLATION
kernarg_address=0x7f41d6600200

:0:rocdevice.cpp:4434: Callback: Queue 0x7f41d6c00000 aborting with error:
HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION code: 0x29
[host: heliosp-1b114-c01-3.mnb.dcgpu, GPU index: 0, kernel: hip::stream::empty_kernel()]
```

The failing line in the test (`hipStreamQuery.cc:34`):
```cpp
SECTION("Null Stream") {
    hip::stream::empty_kernel<<<dim3(1), dim3(1), 0, stream>>>();  // ← faults here
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipStreamQuery(stream));
}
```

---

## Root Cause

### PR #7727 introduced device-resident (VRAM) queue ring buffers

PR #7727 switched the CLR queue ring buffer from system memory to device memory (VRAM)
when `HSA_AMD_QUEUE_CREATE_DEVICE_MEM_RING_BUF` is set. It also introduced:
- `use_movdir64b_`: use MOVDIR64B (atomic 64-byte store) for AQL packet writes when ring
  buffer is in VRAM
- Direct doorbell (`doorbell_ptr_`) cached per-queue
- `ReleaseHwQueue()` optimization in `hipStreamQuery_common` to release idle stream queues
  back to the pool

### Queue recycling chain that corrupts the null stream

The bug requires the null stream's HW queue to be recycled before the WFW dispatch.
The trigger sequence:

**Step 1 — `Unit_hipStreamDestroy_WithFinishedWork`:**
```
hipStreamCreate(&stream)         → acquire Q1 from pool (device-resident ring buf)
setToOne<<<..., stream>>>()      → dispatch on Q1, write AQL packet to Q1's VRAM ring buf
checkDataSet()                   → calls hipStreamSynchronize(nullptr)
                                   → submits barrier/marker to NULL STREAM (Q_null)
hipStreamDestroy(stream)         → ~VirtualGPU() → releaseQueue(Q1) → Q1 back to pool
```

**Step 2 — `Unit_hipStreamQuery_WithNoWork` (Null Stream section):**
```
hipStreamQuery(nullptr)
  → hip_stream->getLastQueuedCommand() → returns the barrier from Step 1 (NOT null)
  → barrier IS complete
  → calls hip_stream->vdev()->ReleaseHwQueue()   ← KEY: releases Q_null to pool!
  → SetGpuQueue(nullptr), last_hwq_ = Q_null
```

**Step 3 — `Unit_hipStreamQuery_WithNoWork` (Created Stream section):**
```
hipStreamCreate(&stream)  → AcquireActiveQueue(priority_, Q_null preference)
                          → null stream's Q_null acquired by new stream!
hipStreamQuery(stream)    → nothing submitted → returns (no ReleaseHwQueue)
hipStreamDestroy(stream)  → releaseQueue(Q_null) → Q_null back to pool, ring buf dirty
```

**Step 4 — `Unit_hipStreamQuery_WithFinishedWork` (Null Stream section):**
```
empty_kernel<<<nullptr>>>()
  → null stream needs queue → AcquireActiveQueue → gets recycled Q_null
  → SetGpuQueue(Q_null):
      device_mem_ring_buf_ = true   (Q_null has VRAM ring buf)
      use_movdir64b_ = true         (CPU supports MOVDIR64B)
  → dispatchGenericAqlPacket:
      writePacketToRingBuffer (MOVDIR64B path):
        SetMetadata → writes to Q_null's VRAM metadata ring buf
        sfence
        nontemporalWriteAQL → MOVDIR64B writes AQL packet to Q_null's VRAM ring buf
      ringQueueDoorbell (skip_fence=true, doorbell_ptr_ set)
  → GPU reads packet → kernel_obj = 0x7f41ea030580 → APERTURE_VIOLATION
```

### Why `kernel_obj` is wrong on the recycled queue

The exact sub-bug in the MOVDIR64B submission path (after queue recycling) is not fully
isolated. Candidates:

1. **Stale VRAM metadata ring buffer**: Q_null's metadata ring buffer (in VRAM) was not
   cleared when the queue was released. When re-acquired, `metadata_preloader_.SetQueueBase`
   resets the base pointer but not the VRAM content. The CP prefetches metadata and may
   see stale entries with wrong `kernel_code_entry_byte_offset` values, leading to the
   wrong `kernel_obj` being used.

2. **`use_movdir64b_` state mismatch on recycled queue**: Between `SetGpuQueue(nullptr)`
   (sets `use_movdir64b_=false`) and `SetGpuQueue(Q_null)` (sets `use_movdir64b_=true`),
   if any submission happens in between using the wrong path, ring buffer state diverges.

3. **MOVDIR64B write ordering vs doorbell**: With `skip_fence=true` in `ringDoorbell`,
   the fence is skipped because "MOVDIR64B already closed the WC buffer". If the
   WC buffer close is not sufficient to guarantee ordering for BAR-mapped VRAM writes
   under all CPU microarchitectures, the CP may read the old ring buffer slot content.

### Why `AMD_LOG_LEVEL=7` masks the bug

Heavy AMD logging adds CPU-side latency between `SetGpuQueue` and `dispatchGenericAqlPacket`,
giving the VRAM ring buffer time to stabilize. This is consistent with a timing race.

### Why 10.0 (0729) is not affected

10.0 branch predates PR #7727 — queues use system memory ring buffers, `use_movdir64b_=false`,
no device-resident ring buffer recycling issue. `ReleaseHwQueue()` may also not have been
present in the 10.0 `hipStreamQuery_common`.

---

## Repro Script

Location on heliosp-1b114-c01-3.mnb.dcgpu: `/home/mingsun/wfw_repro.sh`

```bash
#!/bin/bash
# Run preceding stream tests then WFW to trigger the state-dependent bug
# $1 = build dir (e.g. 0803_dev or 0803)

BUILD=$1
L=/home/mingsun/mi450/$BUILD/libs
T=/home/mingsun/mi450/$BUILD/tests/share/hip/catch_tests
LIBPATH=$T/lib:$T/lib/rocm_sysdeps/lib:$L/lib:$L/lib/rocm_sysdeps/lib:$L/llvm/lib
BINPATH=$L/bin:$L/llvm/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

PREREQS="Unit_hipStreamDestroy_Negative_NullStream
Unit_hipStreamDestroy_WithFinishedWork
Unit_hipStreamQuery_WithNoWork"

run() {
  env -i HOME=/root PATH=$BINPATH LD_LIBRARY_PATH=$LIBPATH \
    ROCM_PATH=$L HIP_VISIBLE_DEVICES=0 "$@"
}

pass=0; fail=0
echo "=== WFW repro with preceding stream tests: $BUILD ==="
for i in $(seq 1 10); do
  for t in $PREREQS; do
    run timeout 15 $T/StreamTest $t > /dev/null 2>&1
  done
  r=$(run timeout 15 $T/StreamTest Unit_hipStreamQuery_WithFinishedWork 2>&1 \
      | grep -oE "1 passed|1 failed" | tail -1)
  [ "$r" = "1 passed" ] && pass=$((pass+1)) || fail=$((fail+1))
  echo "iter $i: $r  pass=$pass fail=$fail"
done
echo "SUMMARY $BUILD: pass=$pass/10 fail=$fail/10"
```

**Observed results:**
```
SUMMARY 0803_dev: pass=3/10 fail=7/10   (10.1 with PR #7727)
SUMMARY 0803:     pass=3/10 fail=7/10   (10.1 nightly, same failure rate)
```

NOTE: Running WFW in isolation (without prereqs) gives 0/20 fail — test appears to pass.
The preceding stream tests are required to trigger the null stream queue recycling.

---

## Ruled Out

| Env var | Result | Conclusion |
|---|---|---|
| `HSA_HOTSWAP_DISABLE=1` | Still fails 5/5 | Not hotswap/rocjitsu |
| `DEBUG_HIP_DYNAMIC_QUEUES=0` | Still fails 5/5 | Not RLC29 dynamic queue |
| `GPU_MAX_HW_QUEUES=16` | Still fails 10/10 | Not queue pool exhaustion |

---

## Fix

**File:** `projects/clr/rocclr/device/rocm/rocvirtual.cpp:1405`  
**Function:** `VirtualGPU::ringQueueDoorbell`

```cpp
// BEFORE (broken on non-coherent PCIe):
amd::ringDoorbell(doorbell_ptr_, index, use_movdir64b_);

// AFTER (correct):
// skip_fence is only safe when host-GPU memory access is coherent (HSA_PROFILE_FULL,
// e.g. IFoE/XGMI fabric). On non-coherent PCIe (HSA_PROFILE_BASE), MOVDIR64B closes
// the CPU WC buffer but does NOT guarantee the PCIe posted write reaches the device
// before the subsequent UC MMIO doorbell write. Without the sfence, the CP can ring
// the doorbell, fetch the ring-buffer slot, and read stale data (wrong kernel_object).
const bool skip_fence = use_movdir64b_ && roc_device_.info().hostUnifiedMemory_;
amd::ringDoorbell(doorbell_ptr_, index, skip_fence);
```

**Status:** Fix applied locally to `projects/clr/rocclr/device/rocm/rocvirtual.cpp`.
Needs build + verification on c01-3 (Helios-P, PCIe non-coherent).

---

## Key Source Locations

- `hipStreamQuery_common` + `ReleaseHwQueue()` call:  
  `projects/clr/hipamd/src/hip_stream.cpp:570-614`
- `SetGpuQueue` (ring buffer state setup on queue (re-)acquire):  
  `projects/clr/rocclr/device/rocm/rocvirtual.cpp:1231-1255`
- `AcquireQueueWithPreference` (null stream queue re-acquire path):  
  `projects/clr/rocclr/device/rocm/rocvirtual.cpp:1258-1264`
- `writePacketToRingBuffer` (MOVDIR64B vs non-MOVDIR64B path):  
  `projects/clr/rocclr/device/rocm/rocvirtual.cpp:1390-1402`
- `nontemporalWriteAQL` / `ringDoorbell` / `nontemporalStoreFence`:  
  `projects/clr/rocclr/utils/nontemporal.hpp`
- PR #7727 commit: `f79ca3d3a9ee387496edb4ba7aad4e25bca7205c`

---

## Session Context

This analysis was done on 2026-08-04 as part of a 10.1 nightly regression investigation
on c01-3. The broader context:
- 22 other 10.1 regressions (hotswap/ROCjitsu gfx1250 `s_sleep`) were confirmed fixed in
  `0888ca35` and the 0803 nightly
- The WFW bug is the last remaining issue in the 0803 nightly suite (1/4367)
- Repro assets on c01-3: `/home/mingsun/wfw_repro.sh`, builds in
  `/home/mingsun/mi450/0803_dev/` and `/home/mingsun/mi450/0803/`
