// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/consan/consan_moi_report_emission.h"

#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/patch/consan/consan_moi_memory_emission.h"
#include "rocjitsu/code/patch/instruction_sequence.h"
#include "rocjitsu/code/patch/instrumentation_builder.h"

#include <cassert>

namespace rocjitsu::consan_moi_detail {

ConSanMoiRecordEmitter::ConSanMoiRecordEmitter(std::vector<uint32_t> &words, uint16_t address_vgpr,
                                               uint16_t temporary_vgpr, rj_code_arch_t arch)
    : words_(words), address_vgpr_(address_vgpr), temporary_vgpr_(temporary_vgpr), arch_(arch) {}

bool ConSanMoiRecordEmitter::materialize_address(uint64_t address) {
  const auto low = instrumentation::build_v_mov_b32_literal(address_vgpr_,
                                                            static_cast<uint32_t>(address), arch_);
  const auto high = instrumentation::build_v_mov_b32_literal(
      static_cast<uint16_t>(address_vgpr_ + 1u), static_cast<uint32_t>(address >> 32u), arch_);
  InstructionSequence sequence(words_);
  return sequence.emit_all(low, high);
}

bool ConSanMoiRecordEmitter::load(uint32_t offset, uint16_t destination_vgpr) {
  return append_load_u32_vgpr_at_offset(words_, address_vgpr_, offset, destination_vgpr, arch_);
}

bool ConSanMoiRecordEmitter::store_vgpr(uint32_t offset, uint16_t value_vgpr) {
  return append_store_u32_vgpr_at_offset(words_, address_vgpr_, offset, value_vgpr, arch_);
}

bool ConSanMoiRecordEmitter::store_literal(uint32_t offset, uint32_t value) {
  const auto materialize = instrumentation::build_v_mov_b32_literal(temporary_vgpr_, value, arch_);
  InstructionSequence sequence(words_);
  return sequence.emit(materialize) && store_vgpr(offset, temporary_vgpr_);
}

bool ConSanMoiRecordEmitter::store_sgpr(uint32_t offset, uint16_t value_sgpr) {
  words_.push_back(build_v_mov_b32_e32(temporary_vgpr_, value_sgpr, arch_));
  return store_vgpr(offset, temporary_vgpr_);
}

bool ConSanMoiRecordEmitter::store_workgroup(uint32_t offset,
                                             const ConSanMoiWorkgroupSource &source,
                                             MissingWorkgroupSource missing) {
  if (!source.is_well_formed())
    return false;
  if (!source.has_value())
    return missing == MissingWorkgroupSource::Skip ? true : store_literal(offset, 0u);
  return consan_detail::append_workgroup_source_value(words_, source, temporary_vgpr_, arch_) &&
         store_vgpr(offset, temporary_vgpr_);
}

bool moi_permits_literal_dispatch_identity(ConSanMoiEngine engine, rj_code_arch_t arch) {
  return consan_arch_uses_literal_dispatch_identity(arch) || engine == ConSanMoiEngine::Sampled;
}

namespace {

ConSanMoiReportDispatchIdSource
plan_dispatch_id_sources(const ConSanMoiReportDispatchIdPlanningContext &context,
                         bool literal_permitted) {
  return {
      .sgpr = context.point.moi_dispatch_id_sgpr,
      .vgpr = context.point.moi_dispatch_id_vgpr,
      .literal = context.point.moi_dispatch_id_sgpr || context.point.moi_dispatch_id_vgpr ||
                         !literal_permitted
                     ? std::nullopt
                     : std::optional<uint64_t>{context.resources.moi_report_dispatch_id},
  };
}

} // namespace

ConSanMoiReportDispatchIdSource
moi_target_dispatch_id_sources(const ConSanMoiReportDispatchIdPlanningContext &context,
                               rj_code_arch_t arch) {
  return plan_dispatch_id_sources(context, consan_arch_uses_literal_dispatch_identity(arch));
}

ConSanMoiReportDispatchIdSource
moi_bound_dispatch_id_sources(const ConSanMoiReportDispatchIdPlanningContext &context) {
  return plan_dispatch_id_sources(context, true);
}

bool append_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                        const ConSanMoiReportDispatchIdSource &source,
                                        uint16_t destination_vgpr, bool high_word,
                                        rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  const uint16_t register_word = high_word ? 1u : 0u;
  if (source.sgpr) {
    words.push_back(build_v_mov_b32_e32(destination_vgpr,
                                        static_cast<uint16_t>(*source.sgpr + register_word), arch));
    return true;
  }
  if (source.vgpr) {
    words.push_back(build_v_mov_b32_e32(
        destination_vgpr, vector_source_vgpr(static_cast<uint16_t>(*source.vgpr + register_word)),
        arch));
    return true;
  }
  const uint32_t literal_word = static_cast<uint32_t>(*source.literal >> (high_word ? 32u : 0u));
  const auto materialize =
      instrumentation::build_v_mov_b32_literal(destination_vgpr, literal_word, arch);
  if (!materialize)
    return false;
  words.insert(words.end(), materialize->begin(), materialize->end());
  return true;
}

bool append_moi_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                        const ConSanMoiReportDispatchIdSource &source,
                                        uint16_t low_vgpr, uint16_t high_vgpr,
                                        rj_code_arch_t arch) {
  return append_moi_report_dispatch_id_word(words, source, low_vgpr,
                                            /*high_word=*/false, arch) &&
         append_moi_report_dispatch_id_word(words, source, high_vgpr,
                                            /*high_word=*/true, arch);
}

bool append_compare_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                                const ConSanMoiReportDispatchIdSource &source,
                                                uint16_t value_vgpr,
                                                uint16_t clobberable_literal_temporary_vgpr,
                                                bool high_word, rj_code_arch_t arch) {
  if (!source.is_well_formed())
    return false;
  uint16_t expected = 0u;
  const uint16_t register_word = high_word ? 1u : 0u;
  if (source.sgpr) {
    expected = static_cast<uint16_t>(*source.sgpr + register_word);
  } else if (source.vgpr) {
    expected = vector_source_vgpr(static_cast<uint16_t>(*source.vgpr + register_word));
  } else {
    assert(clobberable_literal_temporary_vgpr != value_vgpr);
    if (clobberable_literal_temporary_vgpr == value_vgpr ||
        !append_moi_report_dispatch_id_word(words, source, clobberable_literal_temporary_vgpr,
                                            high_word, arch)) {
      return false;
    }
    expected = vector_source_vgpr(clobberable_literal_temporary_vgpr);
  }
  const auto equal = instrumentation::build_v_cmp_eq_u32_vcc(expected, value_vgpr, arch);
  if (!equal)
    return false;
  words.push_back(*equal);
  return true;
}

bool append_store_moi_report_dispatch_id_pair(ConSanMoiRecordEmitter &record,
                                              const ConSanMoiReportDispatchIdSource &source,
                                              uint32_t low_offset) {
  if (!source.is_well_formed())
    return false;
  for (const bool high_word : {false, true}) {
    const uint32_t offset = low_offset + (high_word ? sizeof(uint32_t) : 0u);
    const uint16_t register_word = high_word ? 1u : 0u;
    if (source.sgpr
            ? !record.store_sgpr(offset, static_cast<uint16_t>(*source.sgpr + register_word))
        : source.vgpr
            ? !record.store_vgpr(offset, static_cast<uint16_t>(*source.vgpr + register_word))
            : !record.store_literal(
                  offset, static_cast<uint32_t>(*source.literal >> (high_word ? 32u : 0u)))) {
      return false;
    }
  }
  return true;
}

} // namespace rocjitsu::consan_moi_detail
