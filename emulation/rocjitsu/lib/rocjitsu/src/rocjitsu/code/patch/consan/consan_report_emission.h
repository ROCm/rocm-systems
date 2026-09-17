// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_dispatch_identity_source.h"
#include "rocjitsu/code/patch/consan/consan_identity_contracts.h"
#include "rocjitsu/code/patch/consan/consan_internal.h"

#include <optional>
#include <vector>

namespace rocjitsu::consan::detail {

class RecordEmitter {
public:
  enum class MissingWorkgroupSource : uint8_t { Skip, StoreZero };

  RecordEmitter(std::vector<uint32_t> &words, uint16_t address_vgpr, uint16_t temporary_vgpr,
                rj_code_arch_t arch);

  [[nodiscard]] bool materialize_address(uint64_t address);
  [[nodiscard]] bool load(uint32_t offset, uint16_t destination_vgpr);
  [[nodiscard]] bool store_vgpr(uint32_t offset, uint16_t value_vgpr);
  [[nodiscard]] bool store_literal(uint32_t offset, uint32_t value);
  [[nodiscard]] bool store_sgpr(uint32_t offset, uint16_t value_sgpr);
  [[nodiscard]] bool store_private(uint32_t offset, uint32_t private_offset);
  [[nodiscard]] bool store_workgroup(uint32_t offset, const WorkgroupSource &source,
                                     MissingWorkgroupSource missing = MissingWorkgroupSource::Skip);

private:
  std::vector<uint32_t> &words_;
  uint16_t address_vgpr_;
  uint16_t temporary_vgpr_;
  rj_code_arch_t arch_;
};

using ReportDispatchIdSource = DispatchIdentity;

struct ReportDispatchIdPlanningContext {
  const OperatingPoint &point;
  const BoundRuntimeResources &resources;
  std::optional<uint32_t> private_offset = std::nullopt;
};

[[nodiscard]] ReportDispatchIdSource
target_dispatch_id_sources(const ReportDispatchIdPlanningContext &context, rj_code_arch_t arch);
[[nodiscard]] ReportDispatchIdSource
bound_dispatch_id_sources(const ReportDispatchIdPlanningContext &context);
[[nodiscard]] bool append_report_dispatch_id_word(std::vector<uint32_t> &words,
                                                  const ReportDispatchIdSource &sources,
                                                  uint16_t destination_vgpr, bool high_word,
                                                  rj_code_arch_t arch);
[[nodiscard]] bool append_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                                  const ReportDispatchIdSource &sources,
                                                  uint16_t low_vgpr, uint16_t high_vgpr,
                                                  rj_code_arch_t arch);
[[nodiscard]] bool append_compare_report_dispatch_id_word(
    std::vector<uint32_t> &words, const ReportDispatchIdSource &sources, uint16_t value_vgpr,
    uint16_t clobberable_literal_temporary_vgpr, bool high_word, rj_code_arch_t arch);
[[nodiscard]] bool append_store_report_dispatch_id_pair(RecordEmitter &record,
                                                        const ReportDispatchIdSource &sources,
                                                        uint32_t low_offset);

} // namespace rocjitsu::consan::detail
