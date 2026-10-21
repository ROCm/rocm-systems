// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cassert>
#include <optional>
#include <span>
#include <utility>
#include <vector>

namespace rocjitsu::hooks {
namespace {

// The arches DBI's probe-call path is validated on end to end
// (docs/dbi-design.md).
[[nodiscard]] bool is_dbi_validated_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

// Whether patching grew @p original's private segment, as a spill does.
[[nodiscard]] bool grows_scratch(const KernelDescriptorInfo &original,
                                 std::span<const uint8_t> patched_elf) {
  const std::vector<KernelDescriptorInfo> kernels =
      scan_object_kernel_descriptors(AmdGpuCodeObject(patched_elf.data(), patched_elf.size()));
  assert(kernels.size() == 1 && "patch() keeps the one kernel");
  return kernels.front().descriptor.private_segment_fixed_size >
         original.descriptor.private_segment_fixed_size;
}

[[nodiscard]] rj_code_arch_t target_arch(rj_code_target_id_t target) {
  const IsaTargetDescriptor *descriptor = default_isa_target_registry().find(target);
  return descriptor == nullptr ? ROCJITSU_CODE_ARCH_INVALID : descriptor->architecture_id;
}

} // namespace

const char *dbi_load_decision_name(DbiLoadDecision decision) {
  switch (decision) {
  case DbiLoadDecision::Instrumented:
    return "instrumented";
  case DbiLoadDecision::NotAmdGpuObject:
    return "not-amdgpu-object";
  case DbiLoadDecision::UnsupportedArch:
    return "unsupported-arch";
  case DbiLoadDecision::ProbeTargetMismatch:
    return "probe-target-mismatch";
  case DbiLoadDecision::NotSingleKernel:
    return "not-single-kernel";
  case DbiLoadDecision::UnsupportedControlFlow:
    return "unsupported-control-flow";
  case DbiLoadDecision::PcRelativeAddress:
    return "pc-relative-address";
  case DbiLoadDecision::NoSites:
    return "no-sites";
  case DbiLoadDecision::PatchFailed:
    return "patch-failed";
  case DbiLoadDecision::GrowsScratch:
    return "grows-scratch";
  case DbiLoadDecision::Count:
    break;
  }
  return "unknown";
}

DbiInstrumentation instrument_each_block(std::span<const uint8_t> elf, const DbiProbe &probe) {
  assert(probe.object != nullptr && probe.object->is_valid());
  DbiInstrumentation out;
  const AmdGpuCodeObject object(elf.data(), elf.size());
  if (!object.is_valid()) {
    out.decision = DbiLoadDecision::NotAmdGpuObject;
    return out;
  }
  const rj_code_arch_t arch = target_arch(object.target_id());
  if (!is_dbi_validated_arch(arch)) {
    out.decision = DbiLoadDecision::UnsupportedArch;
    return out;
  }
  if (probe.object->target_id() != object.target_id()) {
    out.decision = DbiLoadDecision::ProbeTargetMismatch;
    return out;
  }
  // Checked before decoding: most HIP code objects hold many kernels, and
  // refusing those should not cost a decode of their .text.
  const std::vector<KernelDescriptorInfo> kernel_list = scan_object_kernel_descriptors(object);
  if (kernel_list.size() != 1) {
    out.decision = DbiLoadDecision::NotSingleKernel;
    return out;
  }

  Instrumentor instrumentor(object, arch);
  std::string error;
  const std::optional<std::vector<const BasicBlock *>> blocks =
      instrumentor.kernel_blocks(kernel_list.front().entry_text_offset, &error);
  if (!blocks) {
    out.decision = DbiLoadDecision::PatchFailed;
    out.errors.push_back(std::move(error));
    return out;
  }
  // The trampolines' liveness follows successors() only, so it would miss
  // registers read across a call or an unresolved edge.
  for (const BasicBlock *block : *blocks) {
    if (!block->call_edges().empty() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::None) {
      out.decision = DbiLoadDecision::UnsupportedControlFlow;
      return out;
    }
  }
  // Any PC read is refused, including one that addresses code or .rodata,
  // which do not move.
  for (const BasicBlock *block : *blocks) {
    if (std::ranges::any_of(instrumentor.block_instructions(*block),
                            [](const Instruction *inst) { return reads_pc(*inst); })) {
      out.decision = DbiLoadDecision::PcRelativeAddress;
      return out;
    }
  }
  for (const BasicBlock *block : *blocks) {
    const std::optional<uint64_t> anchor = instrumentor.first_anchor(*block);
    if (!anchor) {
      ++out.blocks_without_site;
      continue;
    }
    InstrumentationPoint point;
    point.anchor_offset = *anchor;
    point.probe_obj = probe.object;
    point.probe_symbol = probe.symbol;
    instrumentor.add_point(std::move(point));
    ++out.sites;
  }
  if (out.sites == 0) {
    out.decision = DbiLoadDecision::NoSites;
    return out;
  }

  InstrumentedCodeObject patched = instrumentor.patch();
  if (!patched.errors.empty()) {
    out.decision = DbiLoadDecision::PatchFailed;
    out.errors = std::move(patched.errors);
    return out;
  }
  if (grows_scratch(kernel_list.front(), patched.elf_bytes)) {
    out.decision = DbiLoadDecision::GrowsScratch;
    return out;
  }
  out.decision = DbiLoadDecision::Instrumented;
  out.elf_bytes = std::move(patched.elf_bytes);
  return out;
}

} // namespace rocjitsu::hooks
