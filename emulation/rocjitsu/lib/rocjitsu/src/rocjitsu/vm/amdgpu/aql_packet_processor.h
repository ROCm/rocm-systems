// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file aql_packet_processor.h
/// @brief AQL packet decoding and admission into the command processor.

#pragma once

#include "rocjitsu/vm/amdgpu/aql_packet_types.h"
#include "rocjitsu/vm/amdgpu/gpu_vm.h"
#include "rocjitsu/vm/amdgpu/packet_processor.h"

#include "rocjitsu/base/rj_compiler.h"
#ifndef HSA_LARGE_MODEL
#define HSA_LARGE_MODEL 1
#endif
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/hsa.h"
RJ_DIAGNOSTIC_POP

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <utility>

namespace rocjitsu::amdgpu {

enum class AqlBlockedReason : uint8_t {
  None,
  HeaderInvalid,
  DependencyUnsatisfied,
  MemoryUnavailable,
  AdmissionUnavailable,
};

enum class AqlPacketDiagnostic : uint8_t {
  None,
  UnsupportedPacketType,
  UnsupportedVendorFormat,
  UnsupportedBarrierCondition,
  MalformedExtendedDispatchDimensions,
  UnsupportedKernelWaveSize,
  WgpTopologyUnavailable,
  LdsCapacityExceeded,
  InvalidClusterShape,
  MissingSignalReader,
  MissingAdmissionCallback,
};

enum class AqlPreparedPacketKind : uint8_t {
  KernelDispatch,
  NonKernel,
};

/// @brief Outcome of attempting to commit a decoded AQL action to its CP queue.
enum class AqlAdmissionStatus : uint8_t {
  Complete,
  Blocked,
  Faulted,
  Malformed,
  Unsupported,
};

struct AqlAdmissionResult {
  AqlAdmissionStatus status = AqlAdmissionStatus::Malformed;
  AqlPacketDiagnostic diagnostic = AqlPacketDiagnostic::None;
};

/// @brief Decoded AQL action ready for durable CP admission.
struct AqlPreparedPacket {
  AqlPreparedPacketKind kind = AqlPreparedPacketKind::NonKernel;
  hsa_kernel_dispatch_packet_t kernel_dispatch{};
  ClusterDispatchShape cluster_shape{};
  uint64_t completion_signal = 0;
  bool barrier_bit = false;
};

struct AqlPacketProcessRequest {
  std::span<const std::byte, kAqlPacketBytes> packet;
  /// Immutable address-space binding captured by the ring owner for this
  /// complete fetch/decode/admission transaction. The processor and callbacks
  /// use this synchronously; every queue must supply a valid binding.
  const GpuVmAccess &access;
  uint64_t registration_id = 0;
  uint32_t process_id = 0;
  uint32_t queue_id = 0;
  uint32_t ring_slot = 0;
  uint64_t packet_index = 0;
  uint64_t packet_address = 0;
};

struct AqlPacketProcessResult {
  [[nodiscard]] const PacketProcessResult &packet_result() const noexcept { return packet; }

  PacketProcessResult packet;
  AqlBlockedReason blocked_reason = AqlBlockedReason::None;
  AqlPacketDiagnostic diagnostic = AqlPacketDiagnostic::None;
};

class AqlPacketCallbacks {
public:
  using LoadSignal =
      std::function<AtomicLoadResult(const AqlPacketProcessRequest &, uint64_t address)>;
  /// Durably admit one decoded packet into the owning CP queue.
  ///
  /// A Complete result means the action has been committed and the ring owner
  /// may retire the packet. Every other result must mean that no queue entry was
  /// admitted, so retrying the same packet cannot duplicate an effect.
  using Admit =
      std::function<AqlAdmissionResult(const AqlPacketProcessRequest &, AqlPreparedPacket)>;

  LoadSignal load_signal;
  Admit admit;
};

/// @brief Decodes and durably admits one fixed-size AQL packet.
/// @details Queue scheduling, ring traversal, cursor publication, dispatch
/// execution, and completion remain owned by CommandProcessor.
class AqlPacketProcessor {
public:
  using Request = AqlPacketProcessRequest;
  using Result = AqlPacketProcessResult;

  explicit AqlPacketProcessor(AqlPacketCallbacks callbacks) : callbacks_(std::move(callbacks)) {}

  [[nodiscard]] Result process(Request request);

private:
  [[nodiscard]] Result admit(const Request &request, AqlPreparedPacket prepared);
  [[nodiscard]] Result process_barrier(const Request &request,
                                       const hsa_kernel_dispatch_packet_t &packet, bool is_and);
  [[nodiscard]] Result process_vendor(const Request &request,
                                      const hsa_kernel_dispatch_packet_t &packet);

  AqlPacketCallbacks callbacks_;
};

static_assert(PacketProcessor<AqlPacketProcessor>);

} // namespace rocjitsu::amdgpu
