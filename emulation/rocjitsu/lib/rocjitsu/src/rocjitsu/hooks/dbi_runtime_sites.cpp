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
#include <memory>
#include <unordered_set>
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

// Blocks reachable from the block starting at @p entry along successors(), in
// .text order. Empty when no block starts there.
[[nodiscard]] std::vector<const BasicBlock *>
reachable_blocks(const std::vector<std::unique_ptr<BasicBlock>> &blocks, uint64_t entry) {
  std::unordered_set<const BasicBlock *> reachable;
  std::vector<const BasicBlock *> worklist;
  for (const std::unique_ptr<BasicBlock> &block : blocks) {
    if (block->start_offset() == entry) {
      reachable.insert(block.get());
      worklist.push_back(block.get());
    }
  }
  while (!worklist.empty()) {
    const BasicBlock *block = worklist.back();
    worklist.pop_back();
    for (const BasicBlock *successor : block->successors())
      if (reachable.insert(successor).second)
        worklist.push_back(successor);
  }
  std::vector<const BasicBlock *> ordered;
  for (const std::unique_ptr<BasicBlock> &block : blocks)
    if (reachable.contains(block.get()))
      ordered.push_back(block.get());
  return ordered;
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
  const std::vector<std::unique_ptr<BasicBlock>> *blocks = instrumentor.blocks(&error);
  if (blocks == nullptr) {
    out.decision = DbiLoadDecision::PatchFailed;
    out.errors.push_back(std::move(error));
    return out;
  }
  // Dead code (padding, unreferenced device functions) is left alone, and does
  // not get to veto the kernel.
  const std::vector<const BasicBlock *> reachable =
      reachable_blocks(*blocks, kernel_list.front().entry_text_offset);
  if (reachable.empty()) {
    out.decision = DbiLoadDecision::PatchFailed;
    out.errors.emplace_back("the kernel entry does not start a decoded block");
    return out;
  }
  // The trampolines' liveness follows successors() only, so it would miss
  // registers read across a call or an unresolved edge.
  for (const BasicBlock *block : reachable) {
    if (!block->call_edges().empty() ||
        block->successor_issue() != BasicBlock::SuccessorIssue::None) {
      out.decision = DbiLoadDecision::UnsupportedControlFlow;
      return out;
    }
  }
  for (const BasicBlock *block : reachable) {
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
