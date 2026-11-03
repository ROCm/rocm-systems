// Local POC: CLR side of runtime-contract section 11 (tables, registration, services, cookie and
// serialization tracking). Used by rocclr/device/rocm/rocvirtual.cpp and rocdevice.cpp.
#pragma once

#include "rocm_runtime_contract.h"

#include <hsa/hsa.h>

#include <cstdint>

namespace rtc {
// Registers the tables on first call (RC-PKT-0). Returns the hook table when both entries are set
// and its size covers them (RC-PKT-0b), else null. The emit entry is read with an acquire load.
const rocm_runtime_hooks_t* Hooks();
using emit_fn_t = void (*)(const rocm_dispatch_info_t*, rocm_dispatch_packets_t*);
emit_fn_t EmitFn(const rocm_runtime_hooks_t* hooks);

uint32_t GpuId(hsa_agent_t agent);  // KFD gpu_id, cached per agent

// Runtime-owned completion signals (RC-PKT-15), from a pool created at registration.
bool AcquireSignal(hsa_signal_t* out);

// Track one reservation's completion signal. cookie 0: no done call (serialization-only).
// serialize: counts toward RC-SER-2 until reported. dep: signal this dispatch waits on (released
// when this one completes).
void Track(const void* queue, uint32_t gpu_id, hsa_signal_t sig, uint64_t cookie, bool serialize,
           hsa_signal_t dep);
// Exactly one done call per non-zero cookie (RC-PKT-16).
void ReportNow(uint64_t cookie, uint32_t status);
// Report every completed signal; never blocks (RC-PKT-17/18). Called on dispatch and after CPU waits.
void Drain();
// Queue destruction: drain, then report the queue's outstanding cookies DROPPED (RC-PKT-17).
void DropQueue(const void* queue);

// RC-SER: serialization is active on a GPU while any SERIALIZE dispatch is outstanding.
bool SerializeActive(uint32_t gpu_id);
// The completion signal of the last kernel written on gpu_id while serializing (handle 0 if none);
// takes a reference that Track(dep=...) hands back on completion.
hsa_signal_t TakeLastSignal(uint32_t gpu_id);
void SetLastSignal(uint32_t gpu_id, hsa_signal_t sig);
}  // namespace rtc
