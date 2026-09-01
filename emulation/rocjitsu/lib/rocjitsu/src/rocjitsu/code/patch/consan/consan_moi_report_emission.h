// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

#include <optional>
#include <vector>

namespace rocjitsu::consan_moi_detail {

class ConSanMoiRecordEmitter {
public:
  enum class MissingWorkgroupSource : uint8_t { Skip, StoreZero };

  ConSanMoiRecordEmitter(std::vector<uint32_t> &words, uint16_t address_vgpr,
                         uint16_t temporary_vgpr, rj_code_arch_t arch);

  [[nodiscard]] bool materialize_address(uint64_t address);
  [[nodiscard]] bool load(uint32_t offset, uint16_t destination_vgpr);
  [[nodiscard]] bool store_vgpr(uint32_t offset, uint16_t value_vgpr);
  [[nodiscard]] bool store_literal(uint32_t offset, uint32_t value);
  [[nodiscard]] bool store_sgpr(uint32_t offset, uint16_t value_sgpr);
  [[nodiscard]] bool store_private(uint32_t offset, uint32_t private_offset);
  [[nodiscard]] bool store_workgroup(uint32_t offset, const ConSanMoiWorkgroupSource &source,
                                     MissingWorkgroupSource missing = MissingWorkgroupSource::Skip);

private:
  std::vector<uint32_t> &words_;
  uint16_t address_vgpr_;
  uint16_t temporary_vgpr_;
  rj_code_arch_t arch_;
};

/// One already-authorized source for an inseparable 64-bit dispatch identity.
/// Register alternatives name the low word of a consecutive pair.
struct ConSanMoiReportDispatchIdSource {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;
  std::optional<uint32_t> private_offset;
  std::optional<uint64_t> literal;

  [[nodiscard]] bool is_well_formed() const {
    return static_cast<uint8_t>(sgpr.has_value()) + static_cast<uint8_t>(vgpr.has_value()) +
               static_cast<uint8_t>(private_offset.has_value()) +
               static_cast<uint8_t>(literal.has_value()) ==
           1u;
  }
};

struct ConSanMoiReportDispatchIdPlanningContext {
  const ConSanMoiOperatingPoint &point;
  const BoundRuntimeResources &resources;
  std::optional<uint32_t> private_offset = std::nullopt;
};

[[nodiscard]] ConSanMoiReportDispatchIdSource
moi_target_dispatch_id_sources(const ConSanMoiReportDispatchIdPlanningContext &context,
                               rj_code_arch_t arch);
[[nodiscard]] ConSanMoiReportDispatchIdSource
moi_bound_dispatch_id_sources(const ConSanMoiReportDispatchIdPlanningContext &context);
[[nodiscard]] bool
append_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                   const ConSanMoiReportDispatchIdSource &sources,
                                   uint16_t destination_vgpr, bool high_word, rj_code_arch_t arch);
[[nodiscard]] bool
append_moi_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                   const ConSanMoiReportDispatchIdSource &sources,
                                   uint16_t low_vgpr, uint16_t high_vgpr, rj_code_arch_t arch);
[[nodiscard]] bool append_compare_moi_report_dispatch_id_word(
    std::vector<uint32_t> &words, const ConSanMoiReportDispatchIdSource &sources,
    uint16_t value_vgpr, uint16_t clobberable_literal_temporary_vgpr, bool high_word,
    rj_code_arch_t arch);
[[nodiscard]] bool
append_store_moi_report_dispatch_id_pair(ConSanMoiRecordEmitter &record,
                                         const ConSanMoiReportDispatchIdSource &sources,
                                         uint32_t low_offset);

} // namespace rocjitsu::consan_moi_detail
