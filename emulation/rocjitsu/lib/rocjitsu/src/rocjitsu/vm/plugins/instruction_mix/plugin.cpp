// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/instruction_mix/plugin.h"

#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <format>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

namespace rocjitsu::plugins::instruction_mix {
namespace {

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

/// Ordering over sightings of the same mnemonic, used to decide which one's
/// encoding fields the merged entry keeps.
///
/// Merging walks an unordered_map, so "whichever we saw first" is really
/// "whichever this run's hashing happened to visit first" -- and the report
/// exists to be diffed between runs. Ordering by a value tuple instead makes
/// the choice independent of iteration order, and therefore reproducible.
///
/// Every field the record emits is in the tuple. Dropping one would leave
/// sightings that tie here but print differently, and the tie would again be
/// broken by merge order: VOP1 is the live case, since the generated
/// constructors set `encoding_id_` and `opcode_` before widening `size_` for
/// the DPP, SDWA and literal forms (e.g. generated/cdna4/encodings.cpp
/// Vop1::Vop1), so `v_mov_b32_e32` can be sighted at four and at eight bytes
/// with everything else equal.
bool sighting_precedes(const MnemonicStats &lhs, const MnemonicStats &rhs) {
  return std::tie(lhs.first_dispatch_id, lhs.encoding_id, lhs.opcode, lhs.encoding_bytes,
                  lhs.family) < std::tie(rhs.first_dispatch_id, rhs.encoding_id, rhs.opcode,
                                         rhs.encoding_bytes, rhs.family);
}

} // namespace

InstructionMixPlugin::InstructionMixPlugin(const char * /*config_json*/)
    : ExecutionPlugin("instruction-mix") {}

InstructionMixPlugin::~InstructionMixPlugin() { onShutdown(); }

InstructionFamily InstructionMixPlugin::classify(const Instruction &inst) {
  return classify_instruction(inst);
}

std::string_view InstructionMixPlugin::family_name(InstructionFamily family) {
  return instruction_family_name(family);
}

void InstructionMixPlugin::merge(MnemonicMap &destination, const MnemonicMap &source) {
  for (const auto &[mnemonic, entry] : source) {
    auto [iter, inserted] = destination.try_emplace(mnemonic, entry);
    if (inserted)
      continue;
    MnemonicStats &existing = iter->second;
    const uint64_t executions = existing.executions + entry.executions;
    if (sighting_precedes(entry, existing))
      existing = entry;
    existing.executions = executions;
  }
}

void InstructionMixPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  dispatches_[info.dispatch_id].info = info;
}

void InstructionMixPlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  // Wavefront::reset() does not clear plugin state, so a recycled wave slot
  // still holds the previous wavefront's map. Replace it rather than reuse it.
  wf.set_plugin_state(slot_index(), std::make_unique<InstructionMixWavefrontState>());
}

void InstructionMixPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/,
                                                            const Instruction &inst,
                                                            amdgpu::Wavefront &wf) {
  auto *state = static_cast<InstructionMixWavefrontState *>(wf.plugin_state(slot_index()));
  const std::string_view mnemonic = inst.mnemonic();
  auto iter = state->counts.find(mnemonic);
  if (iter == state->counts.end()) {
    MnemonicStats entry;
    entry.encoding_id = inst.encoding_id();
    entry.opcode = inst.opcode();
    entry.encoding_bytes = static_cast<uint8_t>(std::max(inst.size(), 0));
    entry.family = classify(inst);
    entry.first_dispatch_id = wf.dispatch_id();
    iter = state->counts.emplace(std::string(mnemonic), entry).first;
  }
  ++iter->second.executions;
}

void InstructionMixPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  auto *state = static_cast<InstructionMixWavefrontState *>(wf.plugin_state(slot_index()));
  merge(dispatches_[wf.dispatch_id()].counts, state->counts);
}

void InstructionMixPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
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

void InstructionMixPlugin::onShutdown() {
  if (summary_emitted_)
    return;
  summary_emitted_ = true;

  // Anything still in flight was executed, and the report's whole contract is
  // that a mnemonic absent from the summary was never executed -- so fold it
  // in rather than drop it. A bounded run (rj_vm_request_exit) stops with work
  // in flight routinely. These dispatches get no per-dispatch record: their
  // totals are not final, and only the union is trustworthy.
  //
  // Waves that never reached wavefront-halt are still missed: their counts
  // live in wavefront-local plugin state and the plugin has no way to
  // enumerate live wavefronts. `incomplete_dispatches` is what tells a reader
  // the summary may be a subset; a run that ends cleanly reports zero.
  incomplete_dispatches_ = dispatches_.size();
  for (const auto &[dispatch_id, state] : dispatches_)
    merge(aggregate_, state.counts);

  emit_record("summary", nullptr, aggregate_);
}

void InstructionMixPlugin::emit_record(std::string_view record, const KernelDispatchInfo *info,
                                       const MnemonicMap &counts) {
  // Sorted so two runs of the same workload produce byte-identical mnemonic
  // ordering and the JSONL diffs cleanly.
  std::vector<const MnemonicMap::value_type *> sorted;
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
      std::format("{{\"schema\":\"rocjitsu.instruction_mix.v1\",\"record\":\"{}\"", record);
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
    output += std::format(",\"dispatches\":{},\"incomplete_dispatches\":{},\"complete\":{}",
                          completed_dispatches_, incomplete_dispatches_,
                          incomplete_dispatches_ == 0 ? "true" : "false");
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
    const MnemonicStats &entry = item->second;
    output +=
        std::format("\"{}\":{{\"executions\":{},\"encoding_id\":{},\"opcode\":{},"
                    "\"encoding_bytes\":{},\"family\":\"{}\",\"first_dispatch_id\":{}}}",
                    json_escape(item->first), entry.executions, entry.encoding_id, entry.opcode,
                    entry.encoding_bytes, family_name(entry.family), entry.first_dispatch_id);
  }
  output += "}}\n";
  sink().write(output);
}

} // namespace rocjitsu::plugins::instruction_mix
