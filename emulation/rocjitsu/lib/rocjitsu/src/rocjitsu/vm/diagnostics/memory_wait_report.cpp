// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/diagnostics/memory_wait_report.h"

#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/vm/amdgpu/compute_unit.h"
#include "rocjitsu/vm/amdgpu/wavefront.h"
#include "rocjitsu/vm/diagnostics/hazard_sites.h"
#include "rocjitsu/vm/plugins/kernel_dispatch_info.h"
#include "util/diagnostic.h"

#include <algorithm>
#include <bit>
#include <format>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu::diagnostics {
namespace {

using amdgpu::MemoryWaitScoreboard;
using amdgpu::WaitCounterKind;

/// Instructions listed from the producer to the consumer, both included.
constexpr size_t kMaxTraceSteps = 32;
constexpr std::string_view kNotStraightLine =
    "producer and consumer are not on a straight-line path";
constexpr std::string_view kDisableWaitCheckingHint =
    "set top-level wait_checking to \"off\" (rocjitsu --wait-checking=off) to disable wait "
    "checking.";

const char *type_name(RegClass cls) {
  switch (cls) {
  case RegClass::VGPR:
    return "VGPR";
  case RegClass::SGPR:
    return "SGPR";
  case RegClass::TTMP:
    return "TTMP";
  case RegClass::ACC_VGPR:
    return "ACC";
  case RegClass::EXEC:
    return "EXEC";
  case RegClass::VCC:
    return "VCC";
  case RegClass::SCC:
    return "SCC";
  case RegClass::M0:
    return "M0";
  case RegClass::FLAT_SCRATCH:
    return "FLAT_SCRATCH";
  case RegClass::PC:
    return "PC";
  }
  return "REG";
}

std::string register_name(const RegisterRef &reg) {
  switch (reg.cls) {
  case RegClass::VGPR:
    return std::format("v{}", reg.index);
  case RegClass::SGPR:
    return std::format("s{}", reg.index);
  case RegClass::TTMP:
    return std::format("ttmp{}", reg.index);
  case RegClass::ACC_VGPR:
    return std::format("acc{}", reg.index);
  case RegClass::EXEC:
    return "exec";
  case RegClass::VCC:
    return "vcc";
  case RegClass::SCC:
    return "scc";
  case RegClass::M0:
    return "m0";
  case RegClass::FLAT_SCRATCH:
    return "flat_scratch";
  case RegClass::PC:
    return "pc";
  }
  return "unknown register";
}

std::string counter_name(rj_code_arch_t arch, WaitCounterKind counter) {
  const auto model = amdgpu::waitcnt_model(arch);
  if (model.succeeded() && amdgpu::uses_legacy_waitcnt(model.value())) {
    if (counter == WaitCounterKind::Load)
      return "vmcnt";
    if (counter == WaitCounterKind::Store)
      return "vscnt";
    if (counter == WaitCounterKind::Ds)
      return "lgkmcnt";
  }
  return std::string(amdgpu::wait_counter_name(counter));
}

/// @brief The first lane that the producer wrote and the consumer reads, or -1 for a
/// scalar register.
int32_t hazard_lane(const MemoryWaitScoreboard::Hazard &hazard, const amdgpu::Wavefront &wf) {
  if (hazard.reg.cls != RegClass::VGPR && hazard.reg.cls != RegClass::ACC_VGPR)
    return -1;
  uint64_t lanes = hazard.producer.lanes & wf.exec();
  if (lanes == 0)
    lanes = hazard.producer.lanes;
  return lanes == 0 ? 0 : std::countr_zero(lanes);
}

/// @brief The disassembly and encoded size of the instruction at @p pc.
/// @details Probes before reading, because a read of unmapped code reports a guest fault.
std::pair<std::string, size_t> disassemble(amdgpu::ComputeUnitCore &cu, const amdgpu::Wavefront &wf,
                                           uint64_t pc) {
  Decoder *decoder = cu.decoder();
  if (decoder == nullptr || pc == 0)
    return {};
  std::vector<rj_code_binary_inst_t> words(decoder->max_instruction_words());
  const std::span<std::byte> bytes = std::as_writable_bytes(std::span(words));
  if (const amdgpu::GpuVmAccess *access = wf.vm_access()) {
    constexpr auto kExecute = amdgpu::VmAccessKind::Execute;
    if (access->query_access(pc, bytes.size(), kExecute) != amdgpu::VmAccessOutcome::Complete ||
        access->read(pc, bytes, kExecute) != amdgpu::VmAccessOutcome::Complete)
      return {};
  } else if (const amdgpu::GpuMemory *memory = cu.memory()) {
    for (size_t word = 0; word < words.size(); ++word)
      words[word] = memory->fetch32(pc + word * sizeof(rj_code_binary_inst_t));
  } else {
    return {};
  }
  util::StringDiagnostic decode_error;
  const DecodeResult decoded = decoder->decode_window(words, 0, decode_error.emitter());
  if (decoded.failed() || !decoded.value())
    return {};
  return {decoded.value()->disassemble(), decoded.value()->size()};
}

/// @brief List the producer and the consumer, and the code between them when the
/// consumer follows the producer in straight-line code. Each PC is decoded once.
void list_code(HazardRecord &record, amdgpu::ComputeUnitCore &cu, const amdgpu::Wavefront &wf) {
  const auto annotate = [&](HazardSite &site) {
    auto [disasm, size] = disassemble(cu, wf, site.pc);
    site.disasm = std::move(disasm);
    return size;
  };
  const uint64_t producer = record.producer.pc;
  const uint64_t consumer = record.consumer.pc;
  std::vector<HazardSite> &trace = record.trace;
  if (producer != 0 && producer < consumer) {
    for (uint64_t pc = producer; pc <= consumer && trace.size() < kMaxTraceSteps;) {
      HazardSite &site = trace.emplace_back();
      site.pc = pc;
      const size_t size = annotate(site);
      site.marked = pc == producer || pc == consumer;
      site.note = pc == producer ? "producer" : pc == consumer ? "consumer" : "static";
      if (pc == consumer || size == 0)
        break;
      pc += size;
    }
  }
  const bool reached = !trace.empty() && trace.back().pc == consumer;
  if (!trace.empty())
    record.producer.disasm = trace.front().disasm;
  else
    annotate(record.producer);
  if (reached)
    record.consumer.disasm = trace.back().disasm;
  else if (consumer == producer)
    record.consumer.disasm = record.producer.disasm;
  else
    annotate(record.consumer);
  if (!reached) {
    trace.clear();
    if (producer != consumer)
      record.trace_note = kNotStraightLine;
  }
}

std::string summary(const HazardRecord &record, const MemoryWaitScoreboard::Hazard &hazard,
                    bool replay) {
  const std::string where =
      std::format("[workgroup ({}, {}, {}), wave {}{}]", record.wg[0], record.wg[1], record.wg[2],
                  record.wave, record.lane >= 0 ? std::format(", lane {}", record.lane) : "");
  if (replay)
    return std::format(
        "Hazard on {} {} {}: overwrite before the replay source from pc={:#x} is known safe to "
        "reuse. XNACK replay may need the original value. s_wait_xcnt <= {} is required; {}",
        record.type_name, register_name(hazard.reg), where, record.producer.pc,
        *record.required_wait, kDisableWaitCheckingHint);
  return std::format(
      "Hazard on {} {} {}: {} before memory result from pc={:#x} is known ready ({}). A wait "
      "threshold <= {} is required; {}",
      record.type_name, register_name(hazard.reg), where, hazard.write ? "overwrite" : "read",
      record.producer.pc, *record.counter, *record.required_wait, kDisableWaitCheckingHint);
}

} // namespace

void publish_memory_wait(HazardSites &sites, amdgpu::ComputeUnitCore &cu,
                         const amdgpu::Wavefront &wf, const MemoryWaitScoreboard::Hazard &hazard) {
  const bool replay = hazard.producer.counter == WaitCounterKind::X;
  const auto max_wait =
      amdgpu::WaitCounterPolicy::maximum_dependency_wait(cu.arch(), hazard.producer.counter);

  HazardRecord record;
  record.checker = replay ? "xcnt-wait" : "memory-wait";
  record.type_name = type_name(hazard.reg.cls);
  record.reg = hazard.reg.index;
  record.write = hazard.write;
  record.gpu = sites.gpu();
  record.dispatch = wf.dispatch_id();
  record.process = wf.process_id();
  record.wg = wf.wg_coord();
  record.wave = wf.wf_id();
  record.lane = hazard_lane(hazard, wf);
  record.cu = cu.full_path();
  record.counter = counter_name(cu.arch(), hazard.producer.counter);
  record.required_wait = max_wait.succeeded() ? std::min(hazard.required_wait, max_wait.value())
                                              : hazard.required_wait;
  record.producer.pc = hazard.producer.pc;
  record.consumer.pc = hazard.consumer_pc;
  const DispatchIdentity identity = sites.find_dispatch(record.dispatch);
  const bool known = identity.dispatch_id == record.dispatch && !identity.kernel.empty();
  record.kernel = known ? identity.kernel : kUnknownKernelIdentity;
  record.symbol = known ? identity.symbol : kUnknownKernelIdentity;
  list_code(record, cu, wf);
  record.summary = summary(record, hazard, replay);
  sites.publish(record);
}

} // namespace rocjitsu::diagnostics
