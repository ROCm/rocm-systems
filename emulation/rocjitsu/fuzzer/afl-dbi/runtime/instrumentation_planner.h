#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/patch/code_object_patcher.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu_fuzzer/afl_dbi_plan.h"
#include "vopd_liveness.h"

#include <stdint.h>

#include <array>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace rocjitsu {
class BasicBlock;
class CodeObjectPatcher;
class Instruction;
class LivenessAnalysis;
} // namespace rocjitsu

namespace rocjitsu::fuzzer::afl_dbi {

const char *arch_name(rj_code_arch_t arch);

uint32_t stable_bb_id(std::string_view kernel_name, uint64_t block_text_offset);
uint32_t stable_edge_id(std::string_view kernel_name, uint64_t pred_text_offset,
                        uint64_t target_text_offset);

std::vector<KernelSite> find_kernel_sites(std::span<const uint8_t> image);

InstrumentationPlan select_edge_sites(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                                      std::span<const KernelSite> kernels,
                                      const InstrumentationPlanOptions &options);

struct EntryProbeRegisterSelection {
  std::string kernel_name;
  Rdna4ProbeRegisters probe_registers;
  uint32_t liveness_probe_points = 0;
};

struct EdgeProbeRegisterSelection {
  Rdna4ProbeRegisters probe_registers;
  std::optional<ProbeScratchSpillPlan> scratch_spill_plan;
  bool uses_fresh_registers = false;
};

struct BlockEntryPatchPoint {
  const rocjitsu::Instruction *instruction = nullptr;
  uint64_t text_offset = 0;
};

struct BlockEntryPatchSkip {
  const rocjitsu::Instruction *instruction = nullptr;
  uint64_t text_offset = 0;
  std::string_view reason;
};

std::vector<EntryProbeRegisterSelection>
select_entry_probe_registers(const rocjitsu::AmdGpuCodeObject &co, rj_code_arch_t arch,
                             std::span<const KernelSite> kernels,
                             bool masked_entry_probe_registers,
                             bool stable_state_sgpr,
                             const Rdna4ProbeRegisters &base_registers);

std::optional<EdgeProbeRegisterSelection> select_edge_probe_registers_from_liveness(
    const KernelSite &kernel, rj_code_arch_t arch, const rocjitsu::LivenessAnalysis &liveness,
    std::span<const rocjitsu::Instruction *const> probe_points,
    bool previous_bb_probe_registers, bool stable_state_sgpr,
    const Rdna4ProbeRegisters &base_registers, bool allow_fresh_registers,
    bool allow_vgpr_scratch_spills, const char **failure_reason = nullptr,
    bool allow_direct_exec_fixed_counter_scratch_spills = false,
    bool allow_sgpr_scratch_spills = false,
    bool force_fresh_sgprs = false, bool force_saved_exec_sgpr_pair = false,
    bool force_fresh_vgprs = false, bool force_fresh_scratch_address_vgpr = false);

std::optional<BlockEntryPatchPoint>
select_block_entry_patch_point(const rocjitsu::BasicBlock &block,
                               const rocjitsu::Instruction &first,
                               std::string_view *skip_reason,
                               BlockEntryPatchSkip *skip = nullptr);

std::optional<KernelDescriptorResourceSummary>
plan_kernel_descriptor_resources(std::span<const uint8_t> image, const KernelSite &site,
                                 const ProbeRegisterRequirements &requirements,
                                 const char **failure_reason = nullptr);
struct AmdgpuMetadataPrivateSegmentPatch {
  enum class Kind {
    InPlaceBytes,
    RebuiltNoteSection,
  };

  Kind kind = Kind::InPlaceBytes;
  uint64_t file_offset = 0;
  std::array<uint8_t, 8> bytes{};
  size_t size = 0;
  uint64_t note_section_file_offset = 0;
  std::vector<uint8_t> note_section_bytes;
};

const char *
amdgpu_metadata_private_segment_patch_kind_name(AmdgpuMetadataPrivateSegmentPatch::Kind kind);

std::optional<AmdgpuMetadataPrivateSegmentPatch>
plan_amdgpu_metadata_private_segment_patch(std::span<const uint8_t> image,
                                           std::string_view kernel_name,
                                           uint32_t private_segment_fixed_size,
                                           const char **failure_reason = nullptr);
std::optional<AmdgpuMetadataPrivateSegmentPatch>
plan_amdgpu_metadata_sgpr_count_patch(std::span<const uint8_t> image,
                                      std::string_view kernel_name,
                                      uint32_t sgpr_count,
                                      const char **failure_reason = nullptr);

std::optional<KernelDescriptorResourceSummary>
patch_kernel_descriptor_for_requirements(rocjitsu::CodeObjectPatcher &patcher,
                                         const KernelSite &site,
                                         const ProbeRegisterRequirements &requirements);
bool patch_kernel_descriptor_resources(rocjitsu::CodeObjectPatcher &patcher,
                                       const KernelDescriptorResourceSummary &summary,
                                       const char **failure_reason = nullptr,
                                       std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind>
                                           *applied_private_segment_metadata_patch = nullptr,
                                       std::optional<AmdgpuMetadataPrivateSegmentPatch::Kind>
                                           *applied_sgpr_count_metadata_patch = nullptr);
std::optional<KernelDescriptorResourceSummary>
patch_kernel_descriptor_for_probe(rocjitsu::CodeObjectPatcher &patcher, const KernelSite &site,
                                  const Rdna4ProbeRegisters &probe_registers);

std::optional<ProbeScratchSpillPlan>
plan_probe_scratch_spills(rj_code_arch_t arch, uint8_t address_vgpr,
                          std::span<const uint8_t> spilled_vgprs,
                          std::span<const uint8_t> spilled_sgprs,
                          uint32_t original_private_segment_bytes,
                          bool wave32 = false);

std::optional<ProbeScratchSpillPlan>
plan_vgpr_scratch_spills(rj_code_arch_t arch, uint8_t address_vgpr,
                         std::span<const uint8_t> spilled_vgprs,
                         uint32_t original_private_segment_bytes,
                         bool wave32 = false);

enum class EdgeTrampolinePlacement {
  AppendedCave,
  LocalTextCave,
};

struct EdgeTrampoline {
  uint32_t patch_branch = 0;
  std::vector<uint32_t> cave_words;
};

struct EdgePatchResult {
  EdgeTrampolinePlacement placement = EdgeTrampolinePlacement::AppendedCave;
  uint64_t cave_text_offset = 0;
};

struct PlannedEdgeTrampoline {
  EdgeSite site;
  EdgeTrampoline trampoline;
  EdgePatchResult result;
};

struct PlannedEntryProbe {
  KernelSite site;
  rocjitsu::KernelEntryProloguePlan prologue;
  bool liveness_registers = false;
};

// Non-mutating patch plan assembled before descriptor, entry, or text bytes are
// changed. Installation consumes this object and reports are derived from it so
// planning and mutation stay visibly separated.
struct DeviceElfPatchPlan {
  std::vector<PlannedEntryProbe> entry_probes;
  std::vector<KernelSite> entry_backed_edge_sites;
  std::vector<KernelSite> edge_probe_sites;
  std::vector<KernelSite> self_contained_edge_sites;
  InstrumentationPlan edge_selection;
  std::vector<EdgeSite> edge_sites;
  std::vector<KernelDescriptorResourceSummary> descriptor_resources;
  std::vector<PlannedEdgeTrampoline> edge_trampolines;
  std::unordered_map<std::string, ProbeRegisterRequirements> kernel_probe_requirements;
  std::vector<uint8_t> text;
  LocalTextCaveSummary local_text_cave_summary;
  std::vector<EdgePatchFailure> sampled_edge_failures;
  uint64_t planned_entry_cave_body_size = 0;
  uint64_t planned_cave_body_size = 0;
  uint32_t entry_candidate_count = 0;
  uint32_t edge_patch_failures = 0;
  uint32_t branch_range_failures = 0;
  bool hybrid_edge_probes = false;
};

void record_patch_plan_summary(PatchDeviceElfReport &report,
                               const DeviceElfPatchPlan &plan);

uint32_t edge_count_for_site(const EdgeSite &site);

bool previous_bb_branch_site(const EdgeSite &site);

bool placement_failure_can_degrade_to_fixed(std::string_view reason);

void prime_fixed_counter_placement_tracker(FixedEdgeSlotTracker &tracker,
                                           std::span<const EdgeSite> sites);

EdgeSite make_stable_fixed_counter_fallback_site(const EdgeSite &site);

uint32_t record_fixed_counter_placement_slots(FixedEdgeSlotTracker &tracker,
                                              const EdgeSite &site);

bool placement_fixed_fallback_has_budget(const KernelEdgeSelectionSummary &summary,
                                         uint32_t edge_count);

bool placement_fixed_fallback_has_budget(const InstrumentationPlan &selection,
                                         const EdgeSite &site);

bool record_previous_bb_branch_placement_fallback(
    InstrumentationPlan &selection, const EdgeSite &site,
    uint32_t fixed_slot_collisions, std::string_view failure_reason);

std::optional<PlannedEdgeTrampoline>
plan_edge_trampoline(const EdgeSite &site, std::span<const uint8_t> text,
                     uint64_t appended_cave_body_size, uint64_t cave_start,
                     LocalTextCaveAllocator &local_caves, rj_code_arch_t arch,
                     uint64_t state_pointer, const char **failure_reason);

void install_planned_edge_trampoline(const PlannedEdgeTrampoline &planned,
                                     std::vector<uint8_t> &text,
                                     rocjitsu::CodeObjectPatcher &patcher,
                                     rj_code_arch_t arch);

} // namespace rocjitsu::fuzzer::afl_dbi
