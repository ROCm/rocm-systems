// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/analysis/waitcheck/target.h"

#include <cstdint>
#include <optional>
#include <string_view>
#include <vector>

namespace rocjitsu::waitcheck_detail {

// These runtime-only event IDs extend the develop checker enum without changing
// the checker files in the independently landable simulator change. The next
// checker change assigns these same IDs to its named enum members.
static_assert(static_cast<unsigned>(WaitEventKind::ExportDualSourceBlend) == 25);
static_assert(static_cast<unsigned>(WaitEventKind::Count) == 26);
inline constexpr auto kGfx10ScalarSource = static_cast<WaitEventKind>(26);
inline constexpr auto kValuCore = static_cast<WaitEventKind>(27);
inline constexpr auto kValuDp = static_cast<WaitEventKind>(28);
inline constexpr auto kValuMatrix = static_cast<WaitEventKind>(29);
inline constexpr auto kValuTrans = static_cast<WaitEventKind>(30);

// Runtime completion tracking extends the develop ISA classifier. Keep its
// base classification for ordinary instructions and override only the cache,
// async, and scalar-prefetch cases that require separate issue bookkeeping.
struct RuntimeWaitTarget : WaitcheckTarget {
  [[nodiscard]] static util::FailureOr<std::vector<ClassifiedEvent>>
  classify_events(const Instruction &inst, rj_code_arch_t arch);
  [[nodiscard]] static util::Result classify_events_into(const Instruction &inst,
                                                         rj_code_arch_t arch,
                                                         std::vector<ClassifiedEvent> &events);
  [[nodiscard]] static std::optional<ClassifiedEvent> gfx10_scalar_source(const Instruction &inst,
                                                                          rj_code_arch_t arch);
  [[nodiscard]] static bool releases_gfx10_scalar_sources(const Instruction &inst,
                                                          rj_code_arch_t arch);
  [[nodiscard]] static bool export_skips_with_empty_exec(const Instruction &inst,
                                                         rj_code_arch_t arch, uint64_t exec);
  [[nodiscard]] static std::optional<WaitEventKind> expert_valu_kind(const Instruction &inst,
                                                                     rj_code_arch_t arch);
  [[nodiscard]] static bool expert_valu_issues_without_active_lanes(const Instruction &inst,
                                                                    rj_code_arch_t arch,
                                                                    unsigned wave_size);
  [[nodiscard]] static uint8_t
  wmma_valu_wait_states(const Instruction &inst, rj_code_arch_t arch,
                        rj_code_target_id_t target = ROCJITSU_CODE_TARGET_INVALID);
  [[nodiscard]] static constexpr uint64_t wmma_core_wait_bound(uint64_t younger_core,
                                                               uint8_t window) {
    return window && younger_core > window ? younger_core - window - 1 : 0;
  }
  [[nodiscard]] static std::optional<WaitEventKind>
  normalized_hardware_event_kind(WaitCounterKind counter, WaitEventKind kind, WaitcntModel model);
  [[nodiscard]] static std::optional<uint32_t>
  completion_order_kind(WaitCounterKind counter, WaitEventKind kind, rj_code_arch_t arch);
  [[nodiscard]] static bool flat_issues_both_counters(rj_code_arch_t arch);
  [[nodiscard]] static uint32_t issue_units(const Instruction &inst, const ClassifiedEvent &event,
                                            rj_code_arch_t arch);
  [[nodiscard]] static bool is_async_transfer_kind(WaitEventKind kind);
  [[nodiscard]] static bool is_scalar_prefetch(std::string_view mnemonic) {
    return mnemonic.starts_with("s_prefetch_") || mnemonic.starts_with("s_buffer_prefetch_");
  }
};

} // namespace rocjitsu::waitcheck_detail
