// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file adapter.h
/// @brief Bridge between rocJitsu execution-plugin callbacks and the generic
/// data hazard engine.
///
/// This header intentionally does not include rocJitsu headers. The plugin
/// layer translates native Wavefront/Instruction callback payloads into the
/// small view structs below, then delegates to DataHazardAdapter, which turns
/// them into simulator-neutral ::hazard_core events. Keeping the translation
/// split this way lets the adapter be unit-tested without standing up a VM.

#pragma once

#include "hazard_events.h"
#include "simulator_api.h"

#include <array>
#include <cstdint>
#include <mutex>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu::plugins::data_hazard {

enum class RegisterClass {
  None,
  Scalar,
  Vector,
  AccumVector,
};

enum class WaitKind {
  None,
  Waitcnt,
  WaitLoadcnt,
  WaitStorecnt,
  WaitKmcnt,
  WaitDscnt,
  WaitLoadcntDscnt,
  WaitStorecntDscnt,
  WaitIdle,
  WaitVscnt,
  WaitTensorcnt,
  BarrierWait,
  /// s_sema_wait, the wavegroup-scope counterpart of BarrierWait.
  SemaphoreWait,
  AddressTranslation,
};

struct WaitInfo {
  WaitKind kind = WaitKind::None;
  uint32_t immediate = 0;
};

struct InstructionView {
  hazard_core::ExecutionKey execution;
  hazard_core::EntityId instruction_id = 0;
  uint64_t pc = 0;
  std::array<uint32_t, 4> raw_isa{};
  WaitInfo wait;
};

struct RegisterAccessView {
  InstructionView instruction;
  RegisterClass register_class = RegisterClass::None;
  uint32_t physical_reg = 0;
  uint32_t size_bytes = hazard_core::BYTES_PER_DWORD;
  bool is_read = false;
  bool is_write = false;
};

struct MemoryRouteView {
  InstructionView instruction;
  /// What the route targets: one of the memory kinds, or ScalarRegister for
  /// scalar memory, whose only tracked resource is the destination register.
  hazard_core::ResourceKind resource_kind = hazard_core::ResourceKind::Unknown;
  RegisterClass register_class = RegisterClass::None;
  uint64_t address = 0;
  uint64_t local_address = 0;
  std::vector<uint64_t> per_lane_addresses;
  std::vector<uint64_t> per_lane_local_addresses;
  uint32_t register_base = 0;
  uint32_t size_bytes = hazard_core::BYTES_PER_DWORD;
  uint32_t local_size_bytes = 0;
  bool is_load = false;
  bool is_store = false;
  bool is_atomic = false;
  bool is_flat = false;
  bool is_tensor = false;
  bool writes_local_memory = false;
  hazard_core::WaitCntType local_write_wait = hazard_core::WaitCntType::NONE;
  uint64_t exec_mask = 0;
};

/// @brief Translates rocJitsu-shaped views into ::hazard_core engine events.
///
/// Register-access callbacks arrive without instruction identity, so the
/// adapter remembers the instruction most recently seen on each wave and
/// attributes subsequent sparse callbacks to it.
class DataHazardAdapter {
public:
  explicit DataHazardAdapter(hazard_core::DataHazardSimulatorApi &engine);

  void reset();

  void on_dispatch_begin(hazard_core::EntityId dispatch_id);
  void on_dispatch_end(hazard_core::EntityId dispatch_id);
  void on_workgroup_begin(const hazard_core::ExecutionKey &workgroup);
  void on_workgroup_end(const hazard_core::ExecutionKey &workgroup);
  /// Wavegroup lifecycle, taking the wavegroup from ExecutionKey::wavegroup_id.
  void on_wavegroup_begin(const hazard_core::ExecutionKey &wavegroup);
  void on_wavegroup_end(const hazard_core::ExecutionKey &wavegroup);
  void on_wave_begin(const hazard_core::ExecutionKey &wave);
  void on_wave_end(const hazard_core::ExecutionKey &wave);
  void on_instruction(const InstructionView &instruction);
  void on_register_access(const RegisterAccessView &access);
  void on_memory_route(const MemoryRouteView &route);
  void on_workgroup_barrier(const hazard_core::ExecutionKey &wave);
  void on_local_memory_atomic_barrier(const hazard_core::ExecutionKey &wave, bool async);
  /// s_sema_signal on @p wave. Reported separately from the wait, which the
  /// instruction's WaitInfo carries, because only the signal needs the wave's
  /// LDS stores to have drained.
  void on_wavegroup_semaphore_signal(const hazard_core::ExecutionKey &wave);
  void on_shutdown();

private:
  hazard_core::InstructionDescriptor resolve_instruction(const InstructionView &instruction,
                                                         bool make_current);
  hazard_core::InstructionDescriptor
  current_or_resolved_instruction(const InstructionView &instruction);
  void erase_dispatch(hazard_core::EntityId dispatch_id);

  hazard_core::DataHazardSimulatorApi &api_;
  std::mutex mutex_;
  hazard_core::EntityId next_instruction_id_ = 1ull << 63;
  std::unordered_map<hazard_core::ExecutionKey, hazard_core::InstructionDescriptor,
                     hazard_core::ExecutionKeyHash>
      current_instruction_;
};

hazard_core::WaitAction make_wait_action(const WaitInfo &wait);
WaitKind make_wait_kind(std::string_view mnemonic);
hazard_core::RegisterKind make_register_kind(RegisterClass reg_class);
hazard_core::InstructionDescriptor make_instruction_descriptor(const InstructionView &instruction,
                                                               hazard_core::EntityId fallback_id);

} // namespace rocjitsu::plugins::data_hazard
