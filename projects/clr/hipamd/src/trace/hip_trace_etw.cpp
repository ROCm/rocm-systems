/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_trace_etw.hpp"

#include <atomic>
#include <cstddef>

// HSA headers resolve unqualified on the Windows backends and under hsa/ on Linux, as in
// rocrctx.hpp.
#if defined(ROCR_DYN_DLL) || defined(ROCR_STATIC_OPEN)
#include "hsa.h"
#include "hsa_ext_amd.h"
#else
#include "hsa/hsa.h"
#include "hsa/hsa_ext_amd.h"
#endif

#include "platform/activity.hpp"

// rocm_hip_tlg — ETW provider for HIP API tracepoints.
// GUID: 726f636d-6869-7000-746c-670000000002
TRACELOGGING_DEFINE_PROVIDER(rocm_hip_tlg, "rocm_hip_tlg",
                             (0x726f636d, 0x6869, 0x7000, 0x74, 0x6c, 0x67, 0x00, 0x00, 0x00, 0x00,
                              0x02));

namespace hip {
namespace trace {
namespace {

// A LoadLibrary/FreeLibrary cycle must be able to re-register, so this is a state
// machine rather than a once_flag.
enum class provider_state : int {
  kUnregistered = 0,
  kRegistering = 1,
  kRegistered = 2,
};

std::atomic<provider_state> g_state{provider_state::kUnregistered};

// Grid sizes in ext dispatch packets are in clusters: cluster count x cluster size (in
// workgroups) x workgroup size (in work-items). A product that does not fit in u32 is
// unrepresentable and reported as 0 (runtime contract RC-DISP-3).
inline uint32_t gridDim(uint64_t dim) { return dim <= UINT32_MAX ? static_cast<uint32_t>(dim) : 0; }

inline uint64_t extGridDim(uint32_t cluster_count, uint16_t cluster_size, uint16_t workgroup_size) {
  return static_cast<uint64_t>(cluster_count) * cluster_size * workgroup_size;
}

// Context is 0: CLR has no context identifier (runtime contract: 0 when there is none).
void writeEvent(uint32_t gpu_id, uint64_t ring_va, uint64_t packet_index, uint64_t kernel_object,
                const uint32_t (&grid)[3], const uint32_t (&workgroup)[3]) {
  TraceLoggingWrite(rocm_hip_tlg, "kernel_dispatch", TraceLoggingUInt32(gpu_id, "gpu_id"),
                    TraceLoggingUInt16(0, "context"), TraceLoggingHexUInt64(ring_va, "ring_va"),
                    TraceLoggingUInt64(packet_index, "packet_index"),
                    TraceLoggingHexUInt64(kernel_object, "kernel_object"),
                    TraceLoggingUInt32Array(grid, 3, "grid"),
                    TraceLoggingUInt32Array(workgroup, 3, "workgroup"));
}

void emitKernelDispatch(uint32_t gpu_id, uint64_t ring_va, uint64_t packet_index,
                        const hsa_kernel_dispatch_packet_t* packet) {
  const uint32_t grid[3] = {packet->grid_size_x, packet->grid_size_y, packet->grid_size_z};
  const uint32_t workgroup[3] = {packet->workgroup_size_x, packet->workgroup_size_y,
                                 packet->workgroup_size_z};
  writeEvent(gpu_id, ring_va, packet_index, packet->kernel_object, grid, workgroup);
}

void emitKernelDispatch(uint32_t gpu_id, uint64_t ring_va, uint64_t packet_index,
                        const hsa_amd_ext_kernel_dispatch_packet_t* packet) {
  const uint32_t grid[3] = {gridDim(extGridDim(packet->cluster_count_x, packet->cluster_size_x,
                                               packet->workgroup_size_x)),
                            gridDim(extGridDim(packet->cluster_count_y, packet->cluster_size_y,
                                               packet->workgroup_size_y)),
                            gridDim(extGridDim(packet->cluster_count_z, packet->cluster_size_z,
                                               packet->workgroup_size_z))};
  const uint32_t workgroup[3] = {packet->workgroup_size_x, packet->workgroup_size_y,
                                 packet->workgroup_size_z};
  writeEvent(gpu_id, ring_va, packet_index, packet->kernel_object, grid, workgroup);
}

// Sink for amd::activity_prof::report_kernel_dispatch. |aql_packet| is the packet about to
// be written to the ring; |ext| selects the ext dispatch packet layout.
void reportKernelDispatch(uint32_t gpu_id, uint64_t ring_va, uint64_t packet_index,
                          const void* aql_packet, bool ext) {
  // Nothing trace-only may run before the enable check (runtime contract RC-GEN-2).
  if (!TraceLoggingProviderEnabled(rocm_hip_tlg, 0, 0)) {
    return;
  }
  if (ext) {
    emitKernelDispatch(gpu_id, ring_va, packet_index,
                       static_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(aql_packet));
  } else {
    emitKernelDispatch(gpu_id, ring_va, packet_index,
                       static_cast<const hsa_kernel_dispatch_packet_t*>(aql_packet));
  }
}

// One kernel_dispatch per kernel packet of a graph batch. Bodies in the flat buffer hold
// the real packet fields (their headers are invalid); full_headers carries
// {header(16), setup or amd_format(16)} per packet.
void reportKernelDispatchBatch(uint32_t gpu_id, uint64_t ring_va, uint64_t first_packet_index,
                               const void* flat_packets, const uint32_t* full_headers,
                               size_t packet_count) {
  // Nothing trace-only may run before the enable check (runtime contract RC-GEN-2).
  if (!TraceLoggingProviderEnabled(rocm_hip_tlg, 0, 0)) {
    return;
  }
  // Both base and ext kernel dispatch packets place kernel_object at the same offset.
  static_assert(offsetof(hsa_kernel_dispatch_packet_t, kernel_object) ==
                    offsetof(hsa_amd_ext_kernel_dispatch_packet_t, kernel_object),
                "kernel_object offset mismatch between base and ext dispatch packets");
  static constexpr size_t kPacketSize = sizeof(hsa_kernel_dispatch_packet_t);
  const auto* packets = static_cast<const uint8_t*>(flat_packets);
  for (size_t i = 0; i < packet_count; ++i) {
    const uint32_t full = full_headers[i];
    const uint8_t type = static_cast<uint8_t>(full & 0xFF);
    const uint8_t amd_format = static_cast<uint8_t>((full >> 16) & 0xFF);
    const auto* base =
        reinterpret_cast<const hsa_kernel_dispatch_packet_t*>(packets + i * kPacketSize);
    if (type == HSA_PACKET_TYPE_KERNEL_DISPATCH) {
      emitKernelDispatch(gpu_id, ring_va, first_packet_index + i, base);
    } else if (type == HSA_PACKET_TYPE_VENDOR_SPECIFIC &&
               amd_format == HSA_AMD_PACKET_TYPE_EXT_KERNEL_DISPATCH) {
      emitKernelDispatch(gpu_id, ring_va, first_packet_index + i,
                         reinterpret_cast<const hsa_amd_ext_kernel_dispatch_packet_t*>(base));
    }
  }
}

}  // namespace

void initialize() {
  auto expected = provider_state::kUnregistered;
  if (!g_state.compare_exchange_strong(expected, provider_state::kRegistering)) {
    return;
  }
  TraceLoggingRegister(rocm_hip_tlg);
  // Published after the provider registration so no sink call can hit an unregistered
  // provider; cleared in finalize() before the unregistration for the same reason.
  amd::activity_prof::report_kernel_dispatch.store(&reportKernelDispatch,
                                                   std::memory_order_release);
  amd::activity_prof::report_kernel_dispatch_batch.store(&reportKernelDispatchBatch,
                                                         std::memory_order_release);
  g_state.store(provider_state::kRegistered);
}

void finalize() {
  auto expected = provider_state::kRegistered;
  if (!g_state.compare_exchange_strong(expected, provider_state::kUnregistered)) {
    return;
  }
  amd::activity_prof::report_kernel_dispatch.store(nullptr, std::memory_order_release);
  amd::activity_prof::report_kernel_dispatch_batch.store(nullptr, std::memory_order_release);
  TraceLoggingUnregister(rocm_hip_tlg);
}

void emit_api(uint32_t domain, uint32_t op_id, const char* op_name, uint64_t start_qpc,
              int32_t retval) {
  TraceLoggingWrite(rocm_hip_tlg, "hip_api", TraceLoggingUInt32(domain, "domain"),
                    TraceLoggingUInt32(op_id, "op_id"),
                    TraceLoggingString(op_name != nullptr ? op_name : "", "op_name"),
                    TraceLoggingUInt64(start_qpc, "start_qpc"),
                    TraceLoggingInt32(retval, "retval"));
}

}  // namespace trace
}  // namespace hip
