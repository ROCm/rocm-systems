// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/aql_packet_processor.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/amd_ext_aql_packet.h"
RJ_DIAGNOSTIC_POP

#include <bit>
#include <cstring>
#include <limits>
#include <optional>

namespace rocjitsu::amdgpu {
namespace {

constexpr uint32_t kSignalValueOffset = 8;

std::optional<uint32_t> checked_grid_size(uint32_t cluster_count, uint32_t cluster_size,
                                          uint32_t workgroup_size) {
  if (cluster_count == 0 || cluster_size == 0 || workgroup_size == 0)
    return std::nullopt;
  const uint64_t grid_size =
      static_cast<uint64_t>(cluster_count) * cluster_size * static_cast<uint64_t>(workgroup_size);
  if (grid_size > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(grid_size);
}

AqlPacketProcessResult blocked(AqlBlockedReason reason) {
  PacketProcessResult packet{
      .status = PacketProcessStatus::Blocked,
      .retirement_bytes = kAqlPacketBytes,
  };
  return {.packet = packet, .blocked_reason = reason};
}

AqlPacketProcessResult terminal(PacketProcessStatus status, AqlPacketDiagnostic diagnostic) {
  return {.packet = {.status = status, .retirement_bytes = kAqlPacketBytes},
          .diagnostic = diagnostic};
}

} // namespace

AqlPacketProcessResult AqlPacketProcessor::admit(const Request &request,
                                                 AqlPreparedPacket prepared) {
  if (!callbacks_.admit)
    return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::MissingAdmissionCallback);
  const AqlAdmissionResult admission = callbacks_.admit(request, std::move(prepared));
  switch (admission.status) {
  case AqlAdmissionStatus::Complete:
    return {.packet = {.status = PacketProcessStatus::Complete,
                       .retirement = PacketRetirement::Retire,
                       .retirement_bytes = kAqlPacketBytes}};
  case AqlAdmissionStatus::Blocked:
    return blocked(AqlBlockedReason::AdmissionUnavailable);
  case AqlAdmissionStatus::Faulted:
    return terminal(PacketProcessStatus::Faulted, admission.diagnostic);
  case AqlAdmissionStatus::Malformed:
    return terminal(PacketProcessStatus::Malformed, admission.diagnostic);
  case AqlAdmissionStatus::Unsupported:
    return terminal(PacketProcessStatus::Unsupported, admission.diagnostic);
  }
  return terminal(PacketProcessStatus::Malformed, admission.diagnostic);
}

AqlPacketProcessResult
AqlPacketProcessor::process_barrier(const Request &request,
                                    const hsa_kernel_dispatch_packet_t &packet, bool is_and) {
  hsa_barrier_and_packet_t barrier{};
  std::memcpy(&barrier, &packet, sizeof(barrier));

  bool dependencies_satisfied = is_and;
  bool has_dependencies = false;
  for (const hsa_signal_t dependency : barrier.dep_signal) {
    if (dependency.handle == 0)
      continue;
    has_dependencies = true;
    if (!callbacks_.load_signal)
      return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::MissingSignalReader);
    const AtomicLoadResult loaded =
        callbacks_.load_signal(request, dependency.handle + kSignalValueOffset);
    if (loaded.outcome == VmAccessOutcome::Unavailable)
      return blocked(AqlBlockedReason::MemoryUnavailable);
    if (loaded.outcome == VmAccessOutcome::Faulted)
      return terminal(PacketProcessStatus::Faulted, AqlPacketDiagnostic::None);
    if (loaded.outcome == VmAccessOutcome::Malformed)
      return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::None);

    const int64_t dependency_value = std::bit_cast<int64_t>(loaded.value);
    if (dependency_value > 0) {
      if (is_and) {
        dependencies_satisfied = false;
        break;
      }
    } else if (!is_and) {
      dependencies_satisfied = true;
      break;
    }
  }
  if (!has_dependencies)
    dependencies_satisfied = true;
  if (!dependencies_satisfied)
    return blocked(AqlBlockedReason::DependencyUnsatisfied);

  return admit(request, {.kind = AqlPreparedPacketKind::NonKernel,
                         .completion_signal = barrier.completion_signal.handle,
                         .barrier_bit = ((packet.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0});
}

AqlPacketProcessResult
AqlPacketProcessor::process_vendor(const Request &request,
                                   const hsa_kernel_dispatch_packet_t &packet) {
  AmdExtKernelDispatchPacket extension{};
  std::memcpy(&extension, &packet, sizeof(extension));
  if (extension.amd_format == kHsaAmdPacketTypeBarrierValue) {
    AmdBarrierValuePacket barrier{};
    std::memcpy(&barrier, &packet, sizeof(barrier));
    if (barrier.signal.handle != 0) {
      if (!callbacks_.load_signal)
        return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::MissingSignalReader);
      const AtomicLoadResult loaded =
          callbacks_.load_signal(request, barrier.signal.handle + kSignalValueOffset);
      if (loaded.outcome == VmAccessOutcome::Unavailable)
        return blocked(AqlBlockedReason::MemoryUnavailable);
      if (loaded.outcome == VmAccessOutcome::Faulted)
        return terminal(PacketProcessStatus::Faulted, AqlPacketDiagnostic::None);
      if (loaded.outcome == VmAccessOutcome::Malformed)
        return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::None);

      const int64_t signal_value = std::bit_cast<int64_t>(loaded.value);
      const int64_t masked_value = signal_value & barrier.mask;
      bool satisfied = false;
      switch (barrier.condition) {
      case HSA_SIGNAL_CONDITION_EQ:
        satisfied = masked_value == barrier.value;
        break;
      case HSA_SIGNAL_CONDITION_NE:
        satisfied = masked_value != barrier.value;
        break;
      case HSA_SIGNAL_CONDITION_LT:
        satisfied = masked_value < barrier.value;
        break;
      case HSA_SIGNAL_CONDITION_GTE:
        satisfied = masked_value >= barrier.value;
        break;
      default:
        return terminal(PacketProcessStatus::Unsupported,
                        AqlPacketDiagnostic::UnsupportedBarrierCondition);
      }
      if (!satisfied)
        return blocked(AqlBlockedReason::DependencyUnsatisfied);
    }
    return admit(request,
                 {.kind = AqlPreparedPacketKind::NonKernel,
                  .completion_signal = barrier.completion_signal.handle,
                  .barrier_bit = ((barrier.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0});
  }

  if (extension.amd_format == kHsaAmdPacketTypeExtKernelDispatch) {
    if (extension.dep_signal.handle != 0) {
      if (!callbacks_.load_signal)
        return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::MissingSignalReader);
      const AtomicLoadResult loaded =
          callbacks_.load_signal(request, extension.dep_signal.handle + kSignalValueOffset);
      if (loaded.outcome == VmAccessOutcome::Unavailable)
        return blocked(AqlBlockedReason::MemoryUnavailable);
      if (loaded.outcome == VmAccessOutcome::Faulted)
        return terminal(PacketProcessStatus::Faulted, AqlPacketDiagnostic::None);
      if (loaded.outcome == VmAccessOutcome::Malformed)
        return terminal(PacketProcessStatus::Malformed, AqlPacketDiagnostic::None);
      if (std::bit_cast<int64_t>(loaded.value) != 0)
        return blocked(AqlBlockedReason::DependencyUnsatisfied);
    }

    const std::optional<uint32_t> grid_size_x = checked_grid_size(
        extension.cluster_count_x, extension.cluster_size_x, extension.workgroup_size_x);
    const std::optional<uint32_t> grid_size_y = checked_grid_size(
        extension.cluster_count_y, extension.cluster_size_y, extension.workgroup_size_y);
    const std::optional<uint32_t> grid_size_z = checked_grid_size(
        extension.cluster_count_z, extension.cluster_size_z, extension.workgroup_size_z);
    if (!grid_size_x || !grid_size_y || !grid_size_z) {
      return terminal(PacketProcessStatus::Malformed,
                      AqlPacketDiagnostic::MalformedExtendedDispatchDimensions);
    }

    AqlPreparedPacket prepared{.kind = AqlPreparedPacketKind::KernelDispatch};
    prepared.kernel_dispatch.header = extension.header;
    prepared.kernel_dispatch.setup = extension.setup;
    prepared.kernel_dispatch.workgroup_size_x = extension.workgroup_size_x;
    prepared.kernel_dispatch.workgroup_size_y = extension.workgroup_size_y;
    prepared.kernel_dispatch.workgroup_size_z = extension.workgroup_size_z;
    prepared.kernel_dispatch.grid_size_x = *grid_size_x;
    prepared.kernel_dispatch.grid_size_y = *grid_size_y;
    prepared.kernel_dispatch.grid_size_z = *grid_size_z;
    prepared.kernel_dispatch.private_segment_size = extension.private_segment_size;
    prepared.kernel_dispatch.group_segment_size = extension.group_segment_size;
    prepared.kernel_dispatch.kernel_object = extension.kernel_object;
    prepared.kernel_dispatch.kernarg_address = extension.kernarg_address;
    prepared.kernel_dispatch.completion_signal = extension.completion_signal;
    prepared.cluster_shape = {.count_x = extension.cluster_count_x,
                              .count_y = extension.cluster_count_y,
                              .count_z = extension.cluster_count_z,
                              .size_x = extension.cluster_size_x,
                              .size_y = extension.cluster_size_y,
                              .size_z = extension.cluster_size_z};
    return admit(request, std::move(prepared));
  }

  if (extension.amd_format == kAmdAqlFormatPm4Ib) {
    return admit(request, {.kind = AqlPreparedPacketKind::NonKernel,
                           .completion_signal = extension.completion_signal.handle,
                           .barrier_bit = ((packet.header >> HSA_PACKET_HEADER_BARRIER) & 1) != 0});
  }

  return terminal(PacketProcessStatus::Unsupported, AqlPacketDiagnostic::UnsupportedVendorFormat);
}

AqlPacketProcessResult AqlPacketProcessor::process(Request request) {
  hsa_kernel_dispatch_packet_t packet{};
  std::memcpy(&packet, request.packet.data(), request.packet.size());
  const uint8_t packet_type = packet.header & 0xff;

  switch (packet_type) {
  case HSA_PACKET_TYPE_INVALID:
    return blocked(AqlBlockedReason::HeaderInvalid);
  case HSA_PACKET_TYPE_KERNEL_DISPATCH:
    return admit(request,
                 {.kind = AqlPreparedPacketKind::KernelDispatch, .kernel_dispatch = packet});
  case HSA_PACKET_TYPE_BARRIER_AND:
    return process_barrier(request, packet, true);
  case HSA_PACKET_TYPE_BARRIER_OR:
    return process_barrier(request, packet, false);
  case HSA_PACKET_TYPE_VENDOR_SPECIFIC:
    return process_vendor(request, packet);
  default:
    return terminal(PacketProcessStatus::Unsupported, AqlPacketDiagnostic::UnsupportedPacketType);
  }
}

} // namespace rocjitsu::amdgpu
