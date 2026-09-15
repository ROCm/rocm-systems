// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/coverage/plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/mem_state.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace rocjitsu::plugins::coverage {
namespace {

bool has_prefix(std::string_view mnemonic, std::string_view prefix) {
  return mnemonic.starts_with(prefix);
}

std::string json_escape(std::string_view value) {
  std::string escaped;
  escaped.reserve(value.size());
  constexpr char hex[] = "0123456789abcdef";
  for (const unsigned char c : value) {
    switch (c) {
    case '\"':
      escaped += "\\\"";
      break;
    case '\\':
      escaped += "\\\\";
      break;
    case '\b':
      escaped += "\\b";
      break;
    case '\f':
      escaped += "\\f";
      break;
    case '\n':
      escaped += "\\n";
      break;
    case '\r':
      escaped += "\\r";
      break;
    case '\t':
      escaped += "\\t";
      break;
    default:
      if (c < 0x20) {
        escaped += "\\u00";
        escaped += hex[c >> 4];
        escaped += hex[c & 0xf];
      } else {
        escaped += static_cast<char>(c);
      }
    }
  }
  return escaped;
}

} // namespace

CoveragePlugin::CoveragePlugin(const char * /*config_json*/) : ExecutionPlugin("coverage") {}

CoveragePlugin::~CoveragePlugin() { onShutdown(); }

InstructionFamily CoveragePlugin::classify(const Instruction &inst) {
  const std::string_view mnemonic = inst.mnemonic();

  if (inst.is_mfma() || has_prefix(mnemonic, "v_mfma_") || has_prefix(mnemonic, "v_smfmac_") ||
      has_prefix(mnemonic, "v_wmma_") || has_prefix(mnemonic, "v_swmmac_"))
    return InstructionFamily::Matrix;

  if (inst.is_memory_op()) {
    if (const auto *state = inst.data()) {
      if (state->tag() == amdgpu::LOCAL_MEM)
        return InstructionFamily::Lds;
      if (state->tag() == amdgpu::GLOBAL_MEM || state->tag() == amdgpu::SCALAR_MEM)
        return InstructionFamily::Global;
    }
    if (has_prefix(mnemonic, "ds_"))
      return InstructionFamily::Lds;
    return InstructionFamily::Global;
  }

  constexpr uint64_t control_flags = BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL |
                                     PROGRAM_TERMINATOR | WAITCNT | BARRIER;
  if ((inst.flags() & control_flags) != 0 || has_prefix(mnemonic, "s_nop") ||
      has_prefix(mnemonic, "s_sleep") || has_prefix(mnemonic, "s_delay"))
    return InstructionFamily::Control;
  if (has_prefix(mnemonic, "s_"))
    return InstructionFamily::Scalar;
  if (has_prefix(mnemonic, "v_"))
    return InstructionFamily::Vector;
  return InstructionFamily::Other;
}

std::string_view CoveragePlugin::family_name(InstructionFamily family) {
  constexpr std::array<std::string_view, kInstructionFamilyCount> names = {
      "scalar", "vector", "matrix", "lds", "global", "control", "other"};
  const size_t index = static_cast<size_t>(family);
  return index < names.size() ? names[index] : "other";
}

namespace {

/// Ordering over sightings of the same mnemonic, used to decide which one's
/// encoding fields the merged entry keeps.
///
/// Merging walks an unordered_map, so "whichever we saw first" is really
/// "whichever this run's hashing happened to visit first" -- and the report
/// exists to be diffed between runs. Ordering by a value tuple instead makes
/// the choice independent of iteration order, and therefore reproducible.
bool sighting_precedes(const MnemonicCoverage &lhs, const MnemonicCoverage &rhs) {
  return std::tie(lhs.first_dispatch_id, lhs.encoding_id, lhs.opcode) <
         std::tie(rhs.first_dispatch_id, rhs.encoding_id, rhs.opcode);
}

} // namespace

void CoveragePlugin::merge(CoverageMap &destination, const CoverageMap &source) {
  for (const auto &[mnemonic, entry] : source) {
    auto [iter, inserted] = destination.try_emplace(mnemonic, entry);
    if (inserted)
      continue;
    MnemonicCoverage &existing = iter->second;
    const uint64_t executions = existing.executions + entry.executions;
    if (sighting_precedes(entry, existing))
      existing = entry;
    existing.executions = executions;
  }
}

void CoveragePlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  dispatches_[info.dispatch_id].info = info;
}

void CoveragePlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  // Wavefront::reset() does not clear plugin state, so a recycled wave slot
  // still holds the previous wavefront's map. Replace it rather than reuse it.
  wf.set_plugin_state(slot_index(), std::make_unique<CoverageWavefrontState>());
}

void CoveragePlugin::onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                      amdgpu::Wavefront &wf) {
  auto *state = static_cast<CoverageWavefrontState *>(wf.plugin_state(slot_index()));
  const std::string_view mnemonic = inst.mnemonic();
  auto iter = state->counts.find(mnemonic);
  if (iter == state->counts.end()) {
    MnemonicCoverage entry;
    entry.encoding_id = inst.encoding_id();
    entry.opcode = inst.opcode();
    entry.encoding_bytes = static_cast<uint8_t>(std::max(inst.size(), 0));
    entry.family = classify(inst);
    entry.first_dispatch_id = wf.dispatch_id();
    iter = state->counts.emplace(std::string(mnemonic), entry).first;
  }
  ++iter->second.executions;
}

void CoveragePlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  auto *state = static_cast<CoverageWavefrontState *>(wf.plugin_state(slot_index()));
  merge(dispatches_[wf.dispatch_id()].counts, state->counts);
}

void CoveragePlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  auto iter = dispatches_.find(dispatch_id);
  if (iter == dispatches_.end())
    return;

  DispatchState &state = iter->second;
  state.info.dispatch_id = dispatch_id;
  emit_record("dispatch", &state.info, state.counts);
  merge(aggregate_, state.counts);
  ++completed_dispatches_;
  dispatches_.erase(iter);
}

void CoveragePlugin::onShutdown() {
  if (summary_emitted_)
    return;
  summary_emitted_ = true;
  emit_record("summary", nullptr, aggregate_);
}

void CoveragePlugin::emit_record(std::string_view record, const KernelDispatchInfo *info,
                                 const CoverageMap &counts) {
  // Sorted so two runs of the same workload produce byte-identical mnemonic
  // ordering and the JSONL diffs cleanly.
  std::vector<const CoverageMap::value_type *> sorted;
  sorted.reserve(counts.size());
  for (const auto &item : counts)
    sorted.push_back(&item);
  std::sort(sorted.begin(), sorted.end(),
            [](const auto *lhs, const auto *rhs) { return lhs->first < rhs->first; });

  std::array<uint64_t, kInstructionFamilyCount> family_executions{};
  std::array<uint64_t, kInstructionFamilyCount> family_mnemonics{};
  uint64_t executions = 0;
  for (const auto *item : sorted) {
    const size_t family = static_cast<size_t>(item->second.family);
    family_executions[family] += item->second.executions;
    ++family_mnemonics[family];
    executions += item->second.executions;
  }

  std::string output =
      std::format("{{\"schema\":\"rocjitsu.coverage.v1\",\"record\":\"{}\"", record);
  if (info) {
    output += std::format(",\"dispatch_id\":{},\"kernel_name\":\"{}\",\"kernel_symbol\":\"{}\""
                          ",\"grid\":[{},{},{}],\"workgroup\":[{},{},{}],\"workgroups\":{}"
                          ",\"waves_per_workgroup\":{}",
                          info->dispatch_id, json_escape(info->kernelNameOrUnknown()),
                          json_escape(info->kernelSymbolOrUnknown()), info->grid_size_x,
                          info->grid_size_y, info->grid_size_z, info->workgroup_size_x,
                          info->workgroup_size_y, info->workgroup_size_z, info->workgroup_count,
                          info->wfs_per_workgroup);
  } else {
    output += std::format(",\"dispatches\":{}", completed_dispatches_);
  }
  output += std::format(",\"wave_instructions\":{},\"unique_mnemonics\":{},\"families\":{{",
                        executions, sorted.size());
  for (size_t i = 0; i < kInstructionFamilyCount; ++i) {
    if (i != 0)
      output += ',';
    output += std::format("\"{}\":{{\"mnemonics\":{},\"executions\":{}}}",
                          family_name(static_cast<InstructionFamily>(i)), family_mnemonics[i],
                          family_executions[i]);
  }
  output += "},\"mnemonics\":{";
  bool first = true;
  for (const auto *item : sorted) {
    if (!first)
      output += ',';
    first = false;
    const MnemonicCoverage &entry = item->second;
    output +=
        std::format("\"{}\":{{\"executions\":{},\"encoding_id\":{},\"opcode\":{},"
                    "\"encoding_bytes\":{},\"family\":\"{}\",\"first_dispatch_id\":{}}}",
                    json_escape(item->first), entry.executions, entry.encoding_id, entry.opcode,
                    entry.encoding_bytes, family_name(entry.family), entry.first_dispatch_id);
  }
  output += "}}\n";
  sink().write(output);
}

} // namespace rocjitsu::plugins::coverage
