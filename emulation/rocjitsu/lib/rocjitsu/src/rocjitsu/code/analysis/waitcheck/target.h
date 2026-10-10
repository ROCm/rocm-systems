// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/register_set.h"
#include "util/result.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu {
class Instruction;

namespace waitcheck_detail {
inline constexpr size_t kCounterCount = kWaitCounterCount;

enum class WaitEventKind {
  Unknown,
  VmemNoSamplerLoad,
  FlatLoad,
  VmemStore,
  FlatStore,
  Ds,
  Gds,
  Smem,
  Sample,
  Bvh,
  Export,
  SccWrite,
  SqMessage,
  GlobalInv,
  GlobalWb,
  LdsDirect,
  AsyncLdsLoad,
  AsyncLdsStore,
  AsyncBarrier,
  TensorLdsLoad,
  TensorLdsStore,
  Count,
};

inline constexpr size_t kWaitEventKindCount = static_cast<size_t>(WaitEventKind::Count);
inline constexpr uint8_t kNoPendingEventAge = std::numeric_limits<uint8_t>::max();

enum class TrackedRegisterSource {
  None,
  Defs,
  Uses,
  VectorUses,
  StoreDataUses,
};

struct ClassifiedEvent {
  ClassifiedEvent(WaitCounterKind counter = WaitCounterKind::Load,
                  WaitEventKind kind = WaitEventKind::Unknown,
                  TrackedRegisterSource registers = TrackedRegisterSource::Defs,
                  bool check_uses = true, bool check_defs = true, bool check_exec_defs = false,
                  std::optional<RegisterRef> special_reg = std::nullopt,
                  std::optional<int64_t> barrier_id = std::nullopt, bool check_memory_order = false,
                  bool check_program_end = false, bool check_counter_parity_order = false)
      : counter(counter), kind(kind), registers(registers), check_uses(check_uses),
        check_defs(check_defs), check_exec_defs(check_exec_defs), special_reg(special_reg),
        barrier_id(barrier_id), check_memory_order(check_memory_order),
        check_program_end(check_program_end),
        check_counter_parity_order(check_counter_parity_order) {}

  WaitCounterKind counter = WaitCounterKind::Load;
  WaitEventKind kind = WaitEventKind::Unknown;
  TrackedRegisterSource registers = TrackedRegisterSource::Defs;
  bool check_uses = true;
  bool check_defs = true;
  bool check_exec_defs = false;
  std::optional<RegisterRef> special_reg;
  std::optional<int64_t> barrier_id;
  bool check_memory_order = false;
  bool check_program_end = false;
  bool check_counter_parity_order = false;
};

// ISA classification and wait-field decoding have no pending-state or CFG
// dependencies. The checker composes these policies with its dataflow state.
// Each fallible entry point independently rejects unsupported architectures.
// Successful empty optionals/vectors represent absent counters, waits, or events.
struct WaitcheckTarget : WaitCounterPolicy {
  [[nodiscard]] static util::FailureOr<std::optional<WaitFields>>
  embedded_wait_fields(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static bool vm_vsrc_event_implied_by_wait(WaitEventKind kind,
                                                          WaitCounterKind counter);

  [[nodiscard]] static std::optional<WaitEventKind>
  normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind, WaitcntModel model);

  [[nodiscard]] static bool is_xcnt_vmem_kind(WaitEventKind kind);

  [[nodiscard]] static bool is_xcnt_drain(const Instruction &inst);

  [[nodiscard]] static std::optional<int64_t> first_barrier_id(const Instruction &inst);

  [[nodiscard]] static bool is_cdna4_mubuf_lds_load(const Instruction &inst, rj_code_arch_t arch);

  [[nodiscard]] static std::optional<uint32_t> vinterp_wait_exp(const Instruction &inst);

  [[nodiscard]] static bool is_dsdir(std::string_view mnemonic);

  // GFX11 WAITVDST and GFX12 WAIT_VA_VDST share bits 16-19.
  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_va_vdst(const Instruction &inst);

  // WAIT_VM_VSRC exists only on GFX12; bit 23 is reserved on GFX11.
  [[nodiscard]] static std::optional<uint32_t> dsdir_wait_vm_vsrc(const Instruction &inst);

  [[nodiscard]] static bool is_scalar_memory_op(std::string_view mnemonic);

  [[nodiscard]] static bool is_vmem_store(std::string_view mnemonic);

  [[nodiscard]] static bool is_vmem_atomic(std::string_view mnemonic);

  [[nodiscard]] static bool is_image_atomic(std::string_view mnemonic);

  [[nodiscard]] static bool uses_ds_wait_counter(std::string_view mnemonic);

  [[nodiscard]] static bool is_async_lds_load(std::string_view mnemonic);

  [[nodiscard]] static bool is_async_lds_store(std::string_view mnemonic);

  [[nodiscard]] static bool is_tensor_lds_load(std::string_view mnemonic);

  [[nodiscard]] static bool is_tensor_lds_store(std::string_view mnemonic);

  [[nodiscard]] static util::FailureOr<std::vector<ClassifiedEvent>>
  classify_events(const Instruction &inst, rj_code_arch_t arch);

  // Runtime issuers retain the output storage between instructions.
  [[nodiscard]] static util::Result classify_events_into(const Instruction &inst,
                                                         rj_code_arch_t arch,
                                                         std::vector<ClassifiedEvent> &events);
};

} // namespace waitcheck_detail
} // namespace rocjitsu
