// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/plugins/throughput/plugin.h"

#include "rocjitsu/vm/amdgpu/mem_state.h"

#include <algorithm>
#include <format>
#include <memory>
#include <string>

namespace rocjitsu::plugins::throughput {
namespace {

constexpr size_t family_index(InstructionFamily family) { return static_cast<size_t>(family); }

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

double seconds_between(std::chrono::steady_clock::time_point begin,
                       std::chrono::steady_clock::time_point end) {
  return std::chrono::duration<double>(end - begin).count();
}

double mips(uint64_t instructions, double seconds) {
  return seconds > 0.0 ? static_cast<double>(instructions) / seconds / 1.0e6 : 0.0;
}

double nanoseconds_to_seconds(uint64_t nanoseconds) {
  return static_cast<double>(nanoseconds) / 1.0e9;
}

} // namespace

ThroughputPlugin::ThroughputPlugin(const char * /*config_json*/) : ExecutionPlugin("throughput") {}

ThroughputPlugin::~ThroughputPlugin() { onShutdown(); }

InstructionFamily ThroughputPlugin::classify(const Instruction &inst) {
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

    // These fallbacks keep synthetic/model-only instructions useful even when
    // they do not carry execution-pipeline state.
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

std::string_view ThroughputPlugin::family_name(InstructionFamily family) {
  constexpr std::array<std::string_view, kInstructionFamilyCount> names = {
      "scalar", "vector", "matrix", "lds", "global", "control", "other"};
  const size_t index = family_index(family);
  return index < names.size() ? names[index] : "other";
}

uint64_t ThroughputPlugin::total(const InstructionCounts &counts) {
  uint64_t result = 0;
  for (const uint64_t count : counts)
    result += count;
  return result;
}

void ThroughputPlugin::add(InstructionCounts &destination, const InstructionCounts &source) {
  for (size_t i = 0; i < destination.size(); ++i)
    destination[i] += source[i];
}

void ThroughputPlugin::finish_instruction(ThroughputWavefrontState &state, Clock::time_point end) {
  if (!state.instruction_active)
    return;
  const auto elapsed =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - state.instruction_begin).count();
  if (elapsed > 0)
    state.execution_nanoseconds[family_index(state.active_family)] +=
        static_cast<uint64_t>(elapsed);
  state.instruction_active = false;
}

void ThroughputPlugin::onAmdgpuDispatchPacketProcessed(const KernelDispatchInfo &info) {
  dispatches_[info.dispatch_id].info = info;
}

void ThroughputPlugin::onAmdgpuDispatchExecutionBegin(uint32_t dispatch_id) {
  auto &state = dispatches_[dispatch_id];
  state.info.dispatch_id = dispatch_id;
  state.begin = Clock::now();
  state.begun = true;
}

void ThroughputPlugin::onAmdgpuWavefrontDispatched(amdgpu::Wavefront &wf) {
  wf.set_plugin_state(slot_index(), std::make_unique<ThroughputWavefrontState>());
}

void ThroughputPlugin::onAmdgpuBeforeExecuteInstruction(uint64_t /*pc*/, const Instruction &inst,
                                                        amdgpu::Wavefront &wf) {
  auto *state = static_cast<ThroughputWavefrontState *>(wf.plugin_state(slot_index()));
  const InstructionFamily family = classify(inst);
  ++state->counts[family_index(family)];
  state->active_family = family;
  // Start after classification and accounting so their profiler cost is not
  // charged to the simulated instruction.
  state->instruction_begin = Clock::now();
  state->instruction_active = true;
}

void ThroughputPlugin::onAmdgpuAfterExecuteInstruction(uint64_t /*pc*/,
                                                       const Instruction & /*inst*/,
                                                       amdgpu::Wavefront &wf) {
  const Clock::time_point end = Clock::now();
  auto *state = static_cast<ThroughputWavefrontState *>(wf.plugin_state(slot_index()));
  finish_instruction(*state, end);
}

void ThroughputPlugin::onAmdgpuWavefrontHalted(amdgpu::Wavefront &wf) {
  const Clock::time_point end = Clock::now();
  auto *state = static_cast<ThroughputWavefrontState *>(wf.plugin_state(slot_index()));
  // Program terminators intentionally have no after-execute callback.
  finish_instruction(*state, end);
  auto &dispatch = dispatches_[wf.dispatch_id()];
  add(dispatch.counts, state->counts);
  add(dispatch.execution_nanoseconds, state->execution_nanoseconds);
}

void ThroughputPlugin::onAmdgpuDispatchExecutionEnd(uint32_t dispatch_id) {
  const Clock::time_point end = Clock::now();
  auto iter = dispatches_.find(dispatch_id);
  if (iter == dispatches_.end())
    return;

  DispatchState &state = iter->second;
  const double wall_seconds = state.begun ? seconds_between(state.begin, end) : 0.0;
  emit_record("dispatch", &state.info, state.counts, state.execution_nanoseconds, wall_seconds);

  add(aggregate_counts_, state.counts);
  add(aggregate_execution_nanoseconds_, state.execution_nanoseconds);
  dispatch_seconds_sum_ += wall_seconds;
  ++completed_dispatches_;
  if (!have_active_window_) {
    first_begin_ = state.begun ? state.begin : end;
    have_active_window_ = true;
  } else if (state.begun) {
    first_begin_ = std::min(first_begin_, state.begin);
  }
  last_end_ = std::max(last_end_, end);
  dispatches_.erase(iter);
}

void ThroughputPlugin::onShutdown() {
  if (summary_emitted_)
    return;
  summary_emitted_ = true;
  const double wall_seconds = have_active_window_ ? seconds_between(first_begin_, last_end_) : 0.0;
  emit_record("summary", nullptr, aggregate_counts_, aggregate_execution_nanoseconds_, wall_seconds,
              dispatch_seconds_sum_);
}

void ThroughputPlugin::emit_record(std::string_view record, const KernelDispatchInfo *info,
                                   const InstructionCounts &counts,
                                   const InstructionNanoseconds &execution_nanoseconds,
                                   double wall_seconds, double dispatch_seconds_sum) {
  const uint64_t instruction_count = total(counts);
  std::string output =
      std::format("{{\"schema\":\"rocjitsu.throughput.v2\",\"record\":\"{}\"", record);
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
    output += std::format(",\"dispatches\":{},\"dispatch_seconds_sum\":{:.9g}",
                          completed_dispatches_, dispatch_seconds_sum);
  }
  output += std::format(",\"wall_seconds\":{:.9g},\"wave_instructions\":{},\"mips\":{:.9g}"
                        ",\"families\":{{",
                        wall_seconds, instruction_count, mips(instruction_count, wall_seconds));

  for (size_t i = 0; i < counts.size(); ++i) {
    if (i != 0)
      output += ',';
    const double execution_seconds = nanoseconds_to_seconds(execution_nanoseconds[i]);
    output +=
        std::format("\"{}\":{{\"instructions\":{},\"execution_seconds\":{:.9g},"
                    "\"execution_mips\":{:.9g},\"dispatch_mips\":{:.9g}}}",
                    family_name(static_cast<InstructionFamily>(i)), counts[i], execution_seconds,
                    mips(counts[i], execution_seconds), mips(counts[i], wall_seconds));
  }
  output += "}}\n";
  sink().write(output);
}

} // namespace rocjitsu::plugins::throughput
