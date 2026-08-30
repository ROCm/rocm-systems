// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/patch/consan/consan_moi_engine_contracts.h"
#include "rocjitsu/code/patch/consan/consan_moi_internal.h"

#include <array>
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
  [[nodiscard]] bool store_workgroup(uint32_t offset, const ConSanMoiWorkgroupSource &source,
                                     MissingWorkgroupSource missing = MissingWorkgroupSource::Skip);

private:
  std::vector<uint32_t> &words_;
  uint16_t address_vgpr_;
  uint16_t temporary_vgpr_;
  rj_code_arch_t arch_;
};

struct ConSanMoiReportDispatchIdWordSource {
  std::optional<uint16_t> sgpr;
  std::optional<uint16_t> vgpr;
  uint32_t literal = 0;
};

using ConSanMoiReportDispatchIdSources = std::array<ConSanMoiReportDispatchIdWordSource, 2>;

enum class ConSanMoiLiteralDispatchIdPolicy : uint8_t { TargetDeclared, ExternalBindingAllowed };

[[nodiscard]] bool moi_permits_literal_dispatch_identity(ConSanMoiEngine engine,
                                                         rj_code_arch_t arch);
[[nodiscard]] ConSanMoiReportDispatchIdSources
moi_report_dispatch_id_sources(const ConSanMoiOperatingPoint &point,
                               const BoundRuntimeResources &resources);
[[nodiscard]] bool
moi_report_dispatch_id_source_permitted(const ConSanMoiReportDispatchIdWordSource &source,
                                        ConSanMoiLiteralDispatchIdPolicy policy,
                                        rj_code_arch_t arch);
[[nodiscard]] bool
append_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                   const ConSanMoiReportDispatchIdSources &sources,
                                   uint16_t destination_vgpr, bool high_word, rj_code_arch_t arch,
                                   ConSanMoiLiteralDispatchIdPolicy policy);
[[nodiscard]] bool
append_moi_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                   const ConSanMoiReportDispatchIdSources &sources,
                                   uint16_t low_vgpr, uint16_t high_vgpr, rj_code_arch_t arch,
                                   ConSanMoiLiteralDispatchIdPolicy policy);
[[nodiscard]] bool append_compare_moi_report_dispatch_id_word(
    std::vector<uint32_t> &words, const ConSanMoiReportDispatchIdSources &sources,
    uint16_t value_vgpr, uint16_t clobberable_literal_temporary_vgpr, bool high_word,
    rj_code_arch_t arch, ConSanMoiLiteralDispatchIdPolicy policy);
[[nodiscard]] bool append_store_moi_report_dispatch_id_pair(
    ConSanMoiRecordEmitter &record, const ConSanMoiReportDispatchIdSources &sources,
    uint32_t low_offset, rj_code_arch_t arch, ConSanMoiLiteralDispatchIdPolicy policy);

} // namespace rocjitsu::consan_moi_detail
