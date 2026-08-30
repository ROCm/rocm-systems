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

ConSanMoiReportDispatchIdSources
moi_report_dispatch_id_sources(const ConSanMoiOperatingPoint &point,
                               const BoundRuntimeResources &resources) {
  ConSanMoiReportDispatchIdSources sources;
  for (size_t index = 0; index < sources.size(); ++index) {
    if (point.moi_dispatch_id_sgpr) {
      sources[index].sgpr = static_cast<uint16_t>(*point.moi_dispatch_id_sgpr + index);
    } else if (point.moi_dispatch_id_vgpr) {
      sources[index].vgpr = static_cast<uint16_t>(*point.moi_dispatch_id_vgpr + index);
    } else {
      sources[index].literal =
          static_cast<uint32_t>(resources.moi_report_dispatch_id >> (index * 32u));
    }
  }
  return sources;
}

bool moi_report_dispatch_id_source_permitted(const ConSanMoiReportDispatchIdWordSource &source,
                                             ConSanMoiLiteralDispatchIdPolicy policy,
                                             rj_code_arch_t arch) {
  return source.sgpr || source.vgpr ||
         policy == ConSanMoiLiteralDispatchIdPolicy::ExternalBindingAllowed ||
         consan_arch_uses_literal_dispatch_identity(arch);
}

bool append_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                        const ConSanMoiReportDispatchIdSources &sources,
                                        uint16_t destination_vgpr, bool high_word,
                                        rj_code_arch_t arch,
                                        ConSanMoiLiteralDispatchIdPolicy policy) {
  const ConSanMoiReportDispatchIdWordSource &source = sources[high_word ? 1u : 0u];
  if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
    return false;
  if (source.sgpr) {
    words.push_back(build_v_mov_b32_e32(destination_vgpr, *source.sgpr, arch));
    return true;
  }
  if (source.vgpr) {
    words.push_back(build_v_mov_b32_e32(destination_vgpr, vector_source_vgpr(*source.vgpr), arch));
    return true;
  }
  const auto materialize =
      instrumentation::build_v_mov_b32_literal(destination_vgpr, source.literal, arch);
  if (!materialize)
    return false;
  words.insert(words.end(), materialize->begin(), materialize->end());
  return true;
}

bool append_moi_report_dispatch_id_pair(std::vector<uint32_t> &words,
                                        const ConSanMoiReportDispatchIdSources &sources,
                                        uint16_t low_vgpr, uint16_t high_vgpr, rj_code_arch_t arch,
                                        ConSanMoiLiteralDispatchIdPolicy policy) {
  return append_moi_report_dispatch_id_word(words, sources, low_vgpr,
                                            /*high_word=*/false, arch, policy) &&
         append_moi_report_dispatch_id_word(words, sources, high_vgpr,
                                            /*high_word=*/true, arch, policy);
}

bool append_compare_moi_report_dispatch_id_word(std::vector<uint32_t> &words,
                                                const ConSanMoiReportDispatchIdSources &sources,
                                                uint16_t value_vgpr,
                                                uint16_t clobberable_literal_temporary_vgpr,
                                                bool high_word, rj_code_arch_t arch,
                                                ConSanMoiLiteralDispatchIdPolicy policy) {
  const ConSanMoiReportDispatchIdWordSource &source = sources[high_word ? 1u : 0u];
  if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
    return false;
  uint16_t expected = 0u;
  if (source.sgpr) {
    expected = *source.sgpr;
  } else if (source.vgpr) {
    expected = vector_source_vgpr(*source.vgpr);
  } else {
    assert(clobberable_literal_temporary_vgpr != value_vgpr);
    if (clobberable_literal_temporary_vgpr == value_vgpr ||
        !append_moi_report_dispatch_id_word(words, sources, clobberable_literal_temporary_vgpr,
                                            high_word, arch, policy)) {
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
                                              const ConSanMoiReportDispatchIdSources &sources,
                                              uint32_t low_offset, rj_code_arch_t arch,
                                              ConSanMoiLiteralDispatchIdPolicy policy) {
  for (const bool high_word : {false, true}) {
    const ConSanMoiReportDispatchIdWordSource &source = sources[high_word ? 1u : 0u];
    if (!moi_report_dispatch_id_source_permitted(source, policy, arch))
      return false;
    const uint32_t offset = low_offset + (high_word ? sizeof(uint32_t) : 0u);
    if (source.sgpr   ? !record.store_sgpr(offset, *source.sgpr)
        : source.vgpr ? !record.store_vgpr(offset, *source.vgpr)
                      : !record.store_literal(offset, source.literal)) {
      return false;
    }
  }
  return true;
}

} // namespace rocjitsu::consan_moi_detail
