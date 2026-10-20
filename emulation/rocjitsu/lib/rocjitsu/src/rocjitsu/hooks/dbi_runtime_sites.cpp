// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/hooks/dbi_runtime_sites.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/target_registry.h"

#include <cassert>
#include <iterator>
#include <optional>
#include <utility>

namespace rocjitsu::hooks {
namespace {

// The arches DBI's probe-call path is validated on end to end
// (docs/dbi-design.md).
[[nodiscard]] bool is_dbi_validated_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4 ||
         arch == ROCJITSU_CODE_ARCH_RDNA4;
}

[[nodiscard]] rj_code_arch_t target_arch(rj_code_target_id_t target) {
  const IsaTargetDescriptor *descriptor = default_isa_target_registry().find(target);
  return descriptor == nullptr ? ROCJITSU_CODE_ARCH_INVALID : descriptor->architecture_id;
}

[[nodiscard]] std::vector<KernelDescriptorInfo> kernels(const AmdGpuCodeObject &object) {
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(object.image_data()),
                                       object.image_size());
  std::vector<KernelDescriptorInfo> found;
  for (const Section *text : object.text_sections()) {
    std::vector<KernelDescriptorInfo> in_text =
        scan_kernel_descriptors(image, text->sectionOffset(), text->size());
    found.insert(found.end(), std::make_move_iterator(in_text.begin()),
                 std::make_move_iterator(in_text.end()));
  }
  return found;
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
  case DbiLoadDecision::NoSites:
    return "no-sites";
  case DbiLoadDecision::PatchFailed:
    return "patch-failed";
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
  const std::vector<KernelDescriptorInfo> kernel_list = kernels(object);
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
  out.decision = DbiLoadDecision::Instrumented;
  out.elf_bytes = std::move(patched.elf_bytes);
  return out;
}

} // namespace rocjitsu::hooks
