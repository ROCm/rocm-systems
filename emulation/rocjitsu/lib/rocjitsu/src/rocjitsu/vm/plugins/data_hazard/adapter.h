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
#include <optional>
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
  // The per-counter waits gfx10 and gfx11 drain a single counter with, each
  // naming the counter in its mnemonic rather than in a field of one immediate.
  WaitVscnt,
  WaitVmcnt,
  WaitLgkmcnt,
  WaitTensorcnt,
  BarrierWait,
  AddressTranslation,
};

/// The depths a wait instruction drains its counters to, carried as the
/// frontend read them rather than repacked into an immediate: the field widths
/// of s_waitcnt differ across GFX generations, so any single layout truncates
/// the counts of the families it was not written for.
struct WaitInfo {
  WaitKind kind = WaitKind::None;
  /// Depth for the counter the mnemonic names, and for the two waits naming a
  /// pair, the memory one: vmcnt for s_waitcnt, loadcnt or storecnt for the
  /// split waits. Empty when the instruction leaves that counter parked at its
  /// no-op maximum — which the disassembler prints by omitting the field, not by
  /// spelling a zero — so make_wait_action drains it only when a depth is set.
  std::optional<uint32_t> count;
  /// Depth for the second counter of those pairs — lgkmcnt for s_waitcnt,
  /// dscnt for the split waits. Empty on the single-counter kinds, and on a pair
  /// whose second field the instruction left parked.
  std::optional<uint32_t> paired_count;
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
  /// Whether the route reads LDS asynchronously, as a tensor store streaming
  /// out of it does. The read stays live until local_read_wait drains, and
  /// overwriting the LDS before then is a WAR hazard.
  bool reads_local_memory = false;
  hazard_core::WaitCntType local_read_wait = hazard_core::WaitCntType::NONE;
  /// Counter a store of this route is outstanding on: VMEM where one counter
  /// covers loads and stores, as on gfx9 and CDNA, and STORE where gfx10 and
  /// later count them apart. Read from the decoded instruction rather than
  /// assumed, and ignored when the route is not a store.
  hazard_core::WaitCntType store_wait = hazard_core::WaitCntType::STORE;
  /// Lanes of the route that reach memory. A zero mask means none of them do,
  /// as when EXEC is zero or every lane falls outside its buffer, and the route
  /// then addresses nothing. Left empty by routes with no lane dimension, such
  /// as scalar memory and tensor DMA, whose single address always counts.
  std::optional<uint64_t> exec_mask;
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
  void on_wave_begin(const hazard_core::ExecutionKey &wave);
  void on_wave_end(const hazard_core::ExecutionKey &wave);
  void on_instruction(const InstructionView &instruction);
  void on_register_access(const RegisterAccessView &access);
  void on_memory_route(const MemoryRouteView &route);
  void on_workgroup_barrier(const hazard_core::ExecutionKey &wave);
  void on_local_memory_atomic_barrier(const hazard_core::ExecutionKey &wave, bool async);
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
/// Reads the counts a wait of @p kind states in @p operand_text, the operand as
/// the disassembler printed it — `vmcnt(1) expcnt(0) lgkmcnt(3)` for s_waitcnt,
/// a bare count for the waits naming one counter.
WaitInfo make_wait_info(WaitKind kind, std::string_view operand_text);
/// Whether a wait of @p kind names a register before its count. The gfx10
/// per-counter waits are SOPK instructions — `s_waitcnt_vscnt null, 0` — so
/// their count is the second operand, where every other wait carries it in the
/// first; the register name read as a count would be the SGPR's own number.
bool wait_count_follows_register(WaitKind kind);
hazard_core::RegisterKind make_register_kind(RegisterClass reg_class);
/// The memory resource a scalar-memory route is tracked against. A scalar load
/// (s_load_*) reads global memory, so it is tracked there — letting the read be
/// seen racing a global write from another workgroup, a race no s_waitcnt can
/// close, only synchronization can. Its destination SGPR is still tracked on its
/// own, because the route keeps register_class = Scalar. A scalar store has no
/// global-shadow slot plumbing yet, so it stays tracked only through its
/// destination register.
hazard_core::ResourceKind scalar_memory_resource_kind(bool is_load);
hazard_core::InstructionDescriptor make_instruction_descriptor(const InstructionView &instruction,
                                                               hazard_core::EntityId fallback_id);

} // namespace rocjitsu::plugins::data_hazard
