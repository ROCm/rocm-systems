// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/ordinary_acquire.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/analysis/instruction_boundaries.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/fault_injection/fault_mutation_internal.h"
#include "rocjitsu/code/patch/fault_injection/ordinary_acquire_internal.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/code/relocation_function_table.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/cdna5/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna3/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/generated/rdna4/machine_insts.h"
#include "rocjitsu/isa/arch/amdgpu/shared/legacy_memory_displacement_encoded_fields.h"
#include "rocjitsu/isa/arch/amdgpu/shared/wait_counter_policy.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rocjitsu {

namespace {

enum class AcquireCacheKind { None, Single, PairPrefix, PairCompletion };

struct DecodedAcquireLoad {
  int32_t byte_offset = 0;
  std::optional<uint32_t> raw_scope;
};

template <typename Raw>
std::optional<DecodedAcquireLoad> decode_modern_acquire_load(const Instruction &instruction,
                                                             std::span<const uint8_t> bytes,
                                                             rj_code_target_id_t) {
  if (instruction.mnemonic() != "global_load_b32" || bytes.size() != sizeof(Raw) ||
      instruction.size() != static_cast<int>(sizeof(Raw)) || !instruction.src_operand(0) ||
      !instruction.dst_operand(0) || !instruction.src_operand(0)->decoded_vgpr() ||
      !instruction.dst_operand(0)->decoded_vgpr())
    return std::nullopt;
  Raw raw = {};
  std::memcpy(&raw, bytes.data(), sizeof(raw));
  const bool high_padding_clear = [&] {
    if constexpr (requires { raw.pad_40_48; })
      return raw.pad_40_48 == 0;
    else
      return raw.pad_40_47 == 0;
  }();
  if (raw.encoding != 0xeeu || raw.pad_8_13 != 0 || raw.pad_22_23 != 0 || !high_padding_clear ||
      raw.pad_63 != 0 || (raw.scope != 2 && raw.scope != 3))
    return std::nullopt;
  const uint32_t encoded = raw.ioffset;
  const int32_t offset = encoded >= 0x800000u ? static_cast<int32_t>(encoded) - 0x1000000
                                              : static_cast<int32_t>(encoded);
  return DecodedAcquireLoad{.byte_offset = offset, .raw_scope = static_cast<uint32_t>(raw.scope)};
}

template <typename Raw>
std::optional<DecodedAcquireLoad> decode_cdna34_acquire_load(const Instruction &instruction,
                                                             std::span<const uint8_t> bytes,
                                                             rj_code_target_id_t target) {
  if (instruction.mnemonic() != "global_load_dword" || bytes.size() != sizeof(Raw) ||
      instruction.size() != static_cast<int>(sizeof(Raw)))
    return std::nullopt;
  Raw raw = {};
  std::memcpy(&raw, bytes.data(), sizeof(raw));
  if (raw.encoding != 0x37u || raw.seg != 2u || raw.sc1 == 0)
    return std::nullopt;
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), bytes.data(), sizeof(words));
  const std::optional<int32_t> offset = amdgpu::read_legacy_memory_byte_offset(target, words);
  if (!offset)
    return std::nullopt;
  // Native SC1 admits Device/System load scope. These legacy controls are
  // not a literal GFX12 scope field; keep the public raw_scope absent.
  return DecodedAcquireLoad{.byte_offset = *offset, .raw_scope = std::nullopt};
}

std::optional<DecodedAcquireLoad> decode_rdna3_acquire_load(const Instruction &instruction,
                                                            std::span<const uint8_t> bytes,
                                                            rj_code_target_id_t target) {
  if (instruction.mnemonic() != "global_load_b32" ||
      bytes.size() != sizeof(rdna3::FlatGlobalMachineInst) ||
      instruction.size() != static_cast<int>(sizeof(rdna3::FlatGlobalMachineInst)))
    return std::nullopt;
  rdna3::FlatGlobalMachineInst raw = {};
  std::memcpy(&raw, bytes.data(), sizeof(raw));
  // RDNA3 GLC selects DEVICE scope for loads; SLC/DLC remain independent cache hints.
  if (raw.encoding != 0x37u || raw.seg != 2u || raw.pad_25 != 0 || raw.glc == 0)
    return std::nullopt;
  std::array<uint32_t, 2> words = {};
  std::memcpy(words.data(), bytes.data(), sizeof(words));
  const std::optional<int32_t> offset = amdgpu::read_legacy_memory_byte_offset(target, words);
  if (!offset)
    return std::nullopt;
  return DecodedAcquireLoad{.byte_offset = *offset, .raw_scope = std::nullopt};
}

AcquireCacheKind rdna3_acquire_cache(std::string_view form) {
  if (form == "s_dcache_inv")
    return AcquireCacheKind::Single;
  if (form == "buffer_gl1_inv")
    return AcquireCacheKind::PairPrefix;
  if (form == "buffer_gl0_inv")
    return AcquireCacheKind::PairCompletion;
  return AcquireCacheKind::None;
}

bool modern_acquire_wait(const Instruction &instruction, std::span<const uint8_t>) {
  return instruction.mnemonic().starts_with("s_wait");
}

template <IsaFamily Arch>
bool legacy_acquire_wait(const Instruction &instruction, std::span<const uint8_t> bytes) {
  if (instruction.mnemonic() != "s_waitcnt" || bytes.size() != sizeof(uint32_t) ||
      instruction.size() != static_cast<int>(sizeof(uint32_t)))
    return false;
  uint32_t word = 0;
  std::memcpy(&word, bytes.data(), sizeof(word));
  return amdgpu::decode_legacy_waitcnt(word & 0xffffu, Arch).vmcnt == 0;
}

AcquireCacheKind cdna34_acquire_cache(std::string_view form) {
  return form == "buffer_inv" || form == "s_dcache_inv" ? AcquireCacheKind::Single
                                                        : AcquireCacheKind::None;
}

AcquireCacheKind modern_acquire_cache(std::string_view form) {
  return form == "global_inv" || form == "s_dcache_inv" ? AcquireCacheKind::Single
                                                        : AcquireCacheKind::None;
}

struct AcquireProfile {
  IsaFamily arch;
  OrdinaryAcquireProfile id;
  std::optional<DecodedAcquireLoad> (*decode_load)(const Instruction &, std::span<const uint8_t>,
                                                   rj_code_target_id_t);
  bool (*qualifying_wait)(const Instruction &, std::span<const uint8_t>);
  AcquireCacheKind (*classify_cache)(std::string_view);
  std::string_view first_cache;
  std::string_view second_cache = {};
};

constexpr std::array<AcquireProfile, 5> kAcquireProfiles = {{
    {.arch = IsaFamily::Cdna5,
     .id = OrdinaryAcquireProfile::Cdna5GlobalB32,
     .decode_load = decode_modern_acquire_load<cdna5::VglobalMachineInst>,
     .qualifying_wait = modern_acquire_wait,
     .classify_cache = modern_acquire_cache,
     .first_cache = "global_inv"},
    {.arch = IsaFamily::Rdna4,
     .id = OrdinaryAcquireProfile::Rdna4GlobalB32,
     .decode_load = decode_modern_acquire_load<rdna4::VglobalMachineInst>,
     .qualifying_wait = modern_acquire_wait,
     .classify_cache = modern_acquire_cache,
     .first_cache = "global_inv"},
    {.arch = IsaFamily::Cdna4,
     .id = OrdinaryAcquireProfile::Cdna34GlobalDword,
     .decode_load = decode_cdna34_acquire_load<cdna4::FlatGlblMachineInst>,
     .qualifying_wait = legacy_acquire_wait<IsaFamily::Cdna4>,
     .classify_cache = cdna34_acquire_cache,
     .first_cache = "buffer_inv"},
    {.arch = IsaFamily::Cdna3,
     .id = OrdinaryAcquireProfile::Cdna34GlobalDword,
     .decode_load = decode_cdna34_acquire_load<cdna3::FlatGlblMachineInst>,
     .qualifying_wait = legacy_acquire_wait<IsaFamily::Cdna3>,
     .classify_cache = cdna34_acquire_cache,
     .first_cache = "buffer_inv"},
    {.arch = IsaFamily::Rdna3,
     .id = OrdinaryAcquireProfile::Rdna3GlobalB32Pair,
     .decode_load = decode_rdna3_acquire_load,
     .qualifying_wait = legacy_acquire_wait<IsaFamily::Rdna3>,
     .classify_cache = rdna3_acquire_cache,
     .first_cache = "buffer_gl1_inv",
     .second_cache = "buffer_gl0_inv"},
}};

// Native BUFFER_INV without SC1 is either a NOP or conditional on TG_SPLIT,
// which this implicit-Agent acquire profile does not establish.
template <typename Raw>
bool qualifying_cdna34_cache(const Instruction &instruction, std::span<const uint8_t> bytes) {
  if (instruction.mnemonic() != "buffer_inv" || bytes.size() != sizeof(Raw) ||
      instruction.size() != static_cast<int>(sizeof(Raw)))
    return false;
  Raw raw = {};
  std::memcpy(&raw, bytes.data(), sizeof(raw));
  return raw.encoding == 0x38u && raw.sc1 != 0;
}

bool qualifying_acquire_cache(const AcquireProfile &profile, const Instruction &instruction,
                              std::span<const uint8_t> bytes) {
  switch (profile.arch) {
  case IsaFamily::Cdna3:
    return qualifying_cdna34_cache<cdna3::MubufMachineInst>(instruction, bytes);
  case IsaFamily::Cdna4:
    return qualifying_cdna34_cache<cdna4::MubufMachineInst>(instruction, bytes);
  default:
    return true;
  }
}

struct AcquireContainer {
  uint64_t begin = 0;
  uint64_t size = 0;
  std::string name;
  bool operator==(const AcquireContainer &) const = default;
};

bool contains(const AcquireContainer &container, uint64_t offset, uint64_t size) {
  return size > 0 && offset >= container.begin && offset - container.begin <= container.size &&
         size <= container.size - (offset - container.begin);
}

// ELF function extents provide container identity independently of the
// requested dispatch owner. Unknown or overlapping containers cannot prove an
// association. Descriptor-only reachability does not invent an ELF extent.
std::vector<AcquireContainer> source_containers(const AmdGpuCodeObject &source) {
  const Section &text = *source.text_sections().front();
  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(source.image_data()),
                                       source.image_size());
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  std::vector<AcquireContainer> containers;
  for (const TextFunctionSymbolExtent &extent : discover_text_function_symbol_extents(source)) {
    const TextFunctionSymbolRange &range = extent.range;
    std::string name = "function@" + std::to_string(range.start_offset);
    for (const KernelDescriptorInfo &kernel : kernels)
      if (kernel.entry_text_offset == range.start_offset) {
        name = kernel.kernel_name;
        break;
      }
    containers.push_back(
        {.begin = range.start_offset, .size = range.size, .name = std::move(name)});
  }
  return containers;
}

std::optional<AcquireContainer> source_container(std::span<const AcquireContainer> containers,
                                                 const Instruction &instruction) {
  std::optional<AcquireContainer> result;
  const uint64_t offset = instruction.src_loc();
  const uint64_t size = static_cast<uint64_t>(instruction.size());
  for (const AcquireContainer &candidate : containers) {
    const bool overlaps = offset >= candidate.begin ? offset - candidate.begin < candidate.size
                                                    : candidate.begin - offset < size;
    if (!overlaps)
      continue;
    // A function boundary inside an instruction leaves its container ambiguous.
    if (!contains(candidate, offset, size))
      return std::nullopt;
    if (result && *result != candidate)
      return std::nullopt;
    result = candidate;
  }
  return result;
}

struct AcquirePoint {
  const Instruction *instruction = nullptr;
  const BasicBlock *block = nullptr;
  std::optional<AcquireContainer> container;
  std::vector<FaultMutationOwner> owners;
  std::vector<uint8_t> bytes;
  bool exact = false;
};

bool metadata_compatible_endpoints(const AcquirePoint &load, const AcquirePoint &cache) {
  return load.container && cache.container && load.container == cache.container &&
         !load.owners.empty() && load.owners == cache.owners;
}

bool compatible_endpoints(const AcquirePoint &load, const AcquirePoint &cache) {
  return load.exact && cache.exact && metadata_compatible_endpoints(load, cache);
}

bool fence_like_form(std::string_view form) {
  constexpr std::array<std::string_view, 10> kFencePrefixes = {
      "s_dcache",  "s_icache",   "global_wb",    "global_inv",     "global_wbinv",
      "buffer_wb", "buffer_inv", "buffer_wbinv", "buffer_gl0_inv", "buffer_gl1_inv"};
  return std::ranges::any_of(kFencePrefixes,
                             [&](std::string_view prefix) { return form.starts_with(prefix); });
}

bool bounded_bookkeeping(const Instruction &instruction) {
  constexpr uint64_t kControlFlow =
      BRANCH | COND_BRANCH | INDIRECT_BRANCH | INDIRECT_CALL | PROGRAM_TERMINATOR;
  const std::string_view form = instruction.mnemonic();
  return instruction.size() > 0 && (!instruction.is_memory_op() || form.starts_with("s_wait")) &&
         !instruction.is_barrier() && form.find("barrier") == std::string_view::npos &&
         !fence_like_form(form) && form != "s_clause" &&
         (instruction.flags() & kControlFlow) == 0 && !instruction.branch_offset_bytes();
}

struct AcquirePathProof {
  OrdinaryAcquirePath kind = OrdinaryAcquirePath::SameBlock;
  std::optional<uint64_t> loop_header;
  std::vector<OrdinaryAcquireMember> waits;
  std::vector<uint64_t> proof_offsets;
  bool saw_required_wait = false;
};

OrdinaryAcquireMember member_of(const Instruction &instruction) {
  return {.offset = instruction.src_loc(),
          .instruction_size = static_cast<uint32_t>(instruction.size()),
          .form = std::string(instruction.mnemonic())};
}

bool scan_bookkeeping(const BasicBlock &block, uint64_t begin, uint64_t end,
                      const AcquireContainer &container,
                      const std::map<uint64_t, AcquirePoint> &points, AcquirePathProof &proof,
                      const AcquireProfile &profile, bool collect_waits = true,
                      bool waits_only = false) {
  uint64_t expected = begin;
  for (const Instruction &instruction : block.instructions()) {
    const uint64_t offset = instruction.src_loc();
    if (offset < begin)
      continue;
    if (offset >= end)
      break;
    const std::map<uint64_t, AcquirePoint>::const_iterator point = points.find(offset);
    if (offset != expected || !bounded_bookkeeping(instruction) || point == points.end() ||
        !point->second.exact || point->second.container != container ||
        static_cast<uint64_t>(instruction.size()) > end - offset ||
        (waits_only && !instruction.mnemonic().starts_with("s_wait")))
      return false;
    proof.proof_offsets.push_back(offset);
    if (collect_waits && instruction.mnemonic().starts_with("s_wait")) {
      proof.waits.push_back(member_of(instruction));
      proof.saw_required_wait |= profile.qualifying_wait(instruction, point->second.bytes);
    }
    expected += static_cast<uint64_t>(instruction.size());
  }
  return expected == end;
}

std::optional<AcquirePathProof> prove_acquire_path(const AcquirePoint &load,
                                                   const AcquirePoint &cache,
                                                   const std::map<uint64_t, AcquirePoint> &points,
                                                   const AcquireProfile &profile) {
  constexpr uint64_t kMaximumBookkeepingBytes = 64;
  constexpr uint64_t kMaximumLoopPrefixBytes = 128;
  if (!compatible_endpoints(load, cache) || !load.block || !cache.block)
    return std::nullopt;
  const uint64_t load_begin = load.instruction->src_loc();
  const uint64_t load_end = load_begin + static_cast<uint64_t>(load.instruction->size());
  const uint64_t cache_begin = cache.instruction->src_loc();
  const BasicBlock &source = *load.block;
  const BasicBlock &boundary = *cache.block;
  if (load_end > cache_begin || cache_begin - load_end > kMaximumBookkeepingBytes ||
      !source.call_edges().empty() || !boundary.call_edges().empty())
    return std::nullopt;
  AcquirePathProof proof;
  if (&source == &boundary) {
    if (!scan_bookkeeping(source, load_end, cache_begin, *load.container, points, proof, profile) ||
        !proof.saw_required_wait)
      return std::nullopt;
    return proof;
  }
  if (!source.has_terminator() && source.end_offset() == boundary.start_offset() &&
      source.successors().size() == 1 && source.successors().front() == &boundary) {
    if (!scan_bookkeeping(source, load_end, source.end_offset(), *load.container, points, proof,
                          profile) ||
        !scan_bookkeeping(boundary, boundary.start_offset(), cache_begin, *load.container, points,
                          proof, profile) ||
        !proof.saw_required_wait)
      return std::nullopt;
    proof.kind = OrdinaryAcquirePath::ExactFallthroughJoin;
    return proof;
  }
  const Instruction *terminator = source.terminator();
  if (!terminator || (terminator->flags() & COND_BRANCH) == 0 ||
      !terminator->branch_offset_bytes() || load_begin < source.start_offset() ||
      load_begin - source.start_offset() > kMaximumLoopPrefixBytes ||
      source.end_offset() != boundary.start_offset() || source.successors().size() != 2 ||
      std::ranges::find(source.successors(), &source) == source.successors().end() ||
      std::ranges::find(source.successors(), &boundary) == source.successors().end() ||
      boundary.predecessors().size() != 1 || boundary.predecessors().front() != &source ||
      source.end_offset() > static_cast<uint64_t>(std::numeric_limits<int64_t>::max()))
    return std::nullopt;
  const int64_t delta = *terminator->branch_offset_bytes();
  const int64_t branch_base = static_cast<int64_t>(source.end_offset());
  if ((delta > 0 && branch_base > std::numeric_limits<int64_t>::max() - delta) ||
      (delta < 0 && branch_base < std::numeric_limits<int64_t>::min() - delta))
    return std::nullopt;
  const int64_t destination = branch_base + delta;
  if (destination < 0 || static_cast<uint64_t>(destination) != source.start_offset() ||
      terminator->src_loc() < load_end ||
      terminator->src_loc() + static_cast<uint64_t>(terminator->size()) != source.end_offset() ||
      !scan_bookkeeping(source, source.start_offset(), load_begin, *load.container, points, proof,
                        profile, false) ||
      !scan_bookkeeping(source, load_end, terminator->src_loc(), *load.container, points, proof,
                        profile) ||
      !scan_bookkeeping(boundary, boundary.start_offset(), cache_begin, *load.container, points,
                        proof, profile) ||
      !proof.saw_required_wait)
    return std::nullopt;
  proof.kind = OrdinaryAcquirePath::ExactSelfLoopExit;
  proof.loop_header = source.start_offset();
  proof.proof_offsets.push_back(terminator->src_loc());
  return proof;
}

bool valid_wave(IsaFamily arch, const KernelDescriptorInfo &kernel) {
  const uint32_t wave = kernel_wavefront_size(arch, kernel.descriptor);
  if (arch == IsaFamily::Cdna5)
    return wave == 32;
  if (arch == IsaFamily::Cdna3 || arch == IsaFamily::Cdna4)
    return wave == 64 && (kernel.descriptor.kernel_code_properties &
                          amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) == 0;
  return wave == 32 || wave == 64;
}

} // namespace

std::span<const OrdinaryAcquireSite> OrdinaryAcquireInventory::sites() const {
  return record_ ? std::span<const OrdinaryAcquireSite>(record_->sites)
                 : std::span<const OrdinaryAcquireSite>();
}

const FaultMutationOwner *OrdinaryAcquireSelection::owner() const {
  return record_ ? &record_->owner : nullptr;
}

FailureOr<OrdinaryAcquireInventory> discover_ordinary_acquires(const FaultMutationContext &context,
                                                               std::string_view entry,
                                                               FaultMutationError &error) {
  error = {};
  const rj_code_target_id_t target = context.source().target_id();
  const IsaTargetDescriptor *target_facts = default_isa_target_registry().find(target);
  if (!target_facts ||
      (target != ROCJITSU_CODE_TARGET_GFX1250 && target != ROCJITSU_CODE_TARGET_GFX1251 &&
       target != ROCJITSU_CODE_TARGET_GFX1200 && target != ROCJITSU_CODE_TARGET_GFX1201 &&
       target != ROCJITSU_CODE_TARGET_GFX950 && target != ROCJITSU_CODE_TARGET_GFX942 &&
       target != ROCJITSU_CODE_TARGET_GFX1100))
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedTarget,
                                 "target has no ordinary acquire query profile");
  const std::array<AcquireProfile, 5>::const_iterator profile =
      std::ranges::find(kAcquireProfiles, target_facts->family, &AcquireProfile::arch);
  if (profile == kAcquireProfiles.end())
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedTarget,
                                 "missing ordinary acquire profile");

  FailureOr<const FaultSourceFacts *> found = context.source_facts(entry, error);
  const KernelOwnership *ownership = nullptr;
  std::optional<std::span<const KernelDescriptorInfo>> owner_kernels;
  std::optional<KernelOwnerFlow> selected_flow;
  size_t selected_owner = 0;
  FaultSourceFacts shared_facts;
  if (found.failed()) {
    if (!context.state_->complete_ownership_needed)
      return Result::failure();
    const FailureOr<const KernelOwnership *> owner_result = context.execution_ownership(error);
    if (owner_result.failed())
      return Result::failure();
    ownership = owner_result.value();
    owner_kernels = ownership->kernels();
    if (!owner_kernels)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "expired acquire owner facts");
    const std::ptrdiff_t matches =
        std::ranges::count_if(*owner_kernels, [&](const KernelDescriptorInfo &kernel) {
          return kernel.kernel_name == entry;
        });
    if (matches != 1)
      return reject_fault_mutation(error,
                                   matches == 0 ? FaultMutationErrorCode::BadIdentity
                                                : FaultMutationErrorCode::AmbiguousSelection,
                                   "entry must match one exact kernel name");
    const std::span<const KernelDescriptorInfo>::iterator kernel =
        std::ranges::find_if(*owner_kernels, [&](const KernelDescriptorInfo &item) {
          return item.kernel_name == entry;
        });
    selected_owner = static_cast<size_t>(kernel - owner_kernels->begin());
    selected_flow = ownership->owner_flow(selected_owner);
    if (!selected_flow)
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "missing acquire owner flow");
    const Section &section = *context.source().text_sections().front();
    TextEditError edit_error;
    const InstructionBoundaries *boundaries =
        context.state_->instrumentor->instruction_boundaries(edit_error);
    if (!boundaries) {
      error = translate_fault_mutation_edit_error(edit_error);
      return Result::failure();
    }
    shared_facts = {.section = {.index = *section.sectionHeaderIndex(),
                                .file_offset = section.sectionOffset(),
                                .vaddr = section.vaddr(),
                                .size = section.size()},
                    .owner = {.entry = kernel->kernel_name,
                              .entry_offset = kernel->entry_text_offset,
                              .descriptor_file_offset = kernel->descriptor_file_offset},
                    .descriptor = *kernel,
                    .blocks = {selected_flow->blocks.begin(), selected_flow->blocks.end()},
                    .boundaries = boundaries};
  }
  error = {};
  const FaultSourceFacts &facts = ownership ? shared_facts : *found.value();
  if (!valid_wave(profile->arch, facts.descriptor))
    return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                 "unsupported ordinary acquire wave size");
  std::shared_ptr<OrdinaryAcquireRecord> record = std::make_shared<OrdinaryAcquireRecord>();
  record->source = context.source_;
  record->section = facts.section;
  record->owner = facts.owner;
  const std::span<const uint8_t> image = context.image();
  const std::vector<AcquireContainer> containers = source_containers(context.source());
  std::map<uint64_t, AcquirePoint> points;
  for (const BasicBlock *block : facts.blocks) {
    for (const Instruction &instruction : block->instructions()) {
      const uint64_t offset = instruction.src_loc();
      const int size = instruction.size();
      if (size <= 0 || size % 4 != 0 || offset > facts.section.size ||
          static_cast<uint64_t>(size) > facts.section.size - offset ||
          facts.section.file_offset > image.size() ||
          offset > image.size() - facts.section.file_offset ||
          static_cast<uint64_t>(size) > image.size() - facts.section.file_offset - offset)
        return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                     "invalid ordinary acquire source extent", offset);
      std::vector<FaultMutationOwner> affected;
      if (ownership) {
        const std::optional<std::span<const KernelExecutionOwner>> owners =
            ownership->owners_at(offset);
        if (!owners || owners->empty() ||
            std::ranges::none_of(*owners, [&](const KernelExecutionOwner &owner) {
              return owner.kernel_index == selected_owner;
            }))
          return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                       "incomplete ordinary acquire owner set", offset);
        for (const KernelExecutionOwner &owner : *owners) {
          if (owner.kernel_index >= owner_kernels->size())
            return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                         "ordinary acquire owner index is out of range", offset);
          const KernelDescriptorInfo &kernel = (*owner_kernels)[owner.kernel_index];
          if (!valid_wave(profile->arch, kernel))
            return reject_fault_mutation(error, FaultMutationErrorCode::UnsupportedForm,
                                         "unsupported affected ordinary acquire wave", offset);
          affected.push_back({.entry = kernel.kernel_name,
                              .entry_offset = kernel.entry_text_offset,
                              .descriptor_file_offset = kernel.descriptor_file_offset});
        }
      } else {
        affected.push_back(facts.owner);
      }
      std::ranges::sort(
          affected, [](const FaultMutationOwner &left, const FaultMutationOwner &right) {
            return std::tie(left.entry, left.entry_offset, left.descriptor_file_offset) <
                   std::tie(right.entry, right.entry_offset, right.descriptor_file_offset);
          });
      const InstructionBoundaries::Entry *boundary = facts.boundaries->find(offset);
      AcquirePoint point = {.instruction = &instruction,
                            .block = block,
                            .container = source_container(containers, instruction),
                            .owners = std::move(affected),
                            .bytes = {image.begin() + facts.section.file_offset + offset,
                                      image.begin() + facts.section.file_offset + offset + size},
                            .exact = boundary && boundary->size == static_cast<uint32_t>(size) &&
                                     !boundary->clause && !boundary->clause_member};
      if (!points.emplace(offset, std::move(point)).second)
        return reject_fault_mutation(error, FaultMutationErrorCode::AmbiguousSelection,
                                     "duplicate ordinary acquire source instruction", offset);
    }
  }

  struct Association {
    std::vector<const AcquirePoint *> caches;
    AcquirePathProof proof;
  };
  // Keep all fence events in source order, including metadata-incompatible
  // interrupters. Build this once: load-only kernels must not rescan their
  // remaining instruction suffix for each possible acquire load.
  std::vector<const AcquirePoint *> fences;
  if (!profile->second_cache.empty())
    for (const std::pair<const uint64_t, AcquirePoint> &item : points)
      if (profile->classify_cache(item.second.instruction->mnemonic()) != AcquireCacheKind::None ||
          fence_like_form(item.second.instruction->mnemonic()))
        fences.push_back(&item.second);
  std::unordered_set<uint64_t> claimed_caches;
  for (const std::pair<const uint64_t, AcquirePoint> &item : points) {
    const AcquirePoint &load = item.second;
    if (!load.exact || !load.container)
      continue;
    const std::optional<DecodedAcquireLoad> decoded =
        profile->decode_load(*load.instruction, load.bytes, target);
    if (!decoded)
      continue;
    const uint64_t load_end = item.first + static_cast<uint64_t>(load.instruction->size());
    std::vector<Association> single_matches;
    std::vector<Association> pair_matches;
    size_t single_metadata_matches = 0;
    for (std::map<uint64_t, AcquirePoint>::const_iterator candidate = points.lower_bound(load_end);
         candidate != points.end() && candidate->first - load_end <= 64; ++candidate) {
      const AcquirePoint &cache = candidate->second;
      if (profile->classify_cache(cache.instruction->mnemonic()) != AcquireCacheKind::Single ||
          !metadata_compatible_endpoints(load, cache))
        continue;
      ++single_metadata_matches;
      const std::optional<AcquirePathProof> proof =
          prove_acquire_path(load, cache, points, *profile);
      if (proof)
        single_matches.push_back({.caches = {&cache}, .proof = *proof});
    }
    // Original single-fence metadata takes precedence even when its path
    // fails. A nearby split pair must not bypass that candidate decision.
    if (single_metadata_matches == 0 && !profile->second_cache.empty()) {
      const std::vector<const AcquirePoint *>::const_iterator first_fence =
          std::ranges::lower_bound(fences, load_end, {}, [](const AcquirePoint *point) {
            return point->instruction->src_loc();
          });
      for (size_t index = static_cast<size_t>(first_fence - fences.cbegin());
           index + 1 < fences.size(); ++index) {
        const AcquirePoint &first = *fences[index];
        const AcquirePoint &tail = *fences[index + 1];
        if (first.instruction->src_loc() - load_end > 64)
          break;
        if (profile->classify_cache(first.instruction->mnemonic()) !=
                AcquireCacheKind::PairPrefix ||
            profile->classify_cache(tail.instruction->mnemonic()) !=
                AcquireCacheKind::PairCompletion ||
            !compatible_endpoints(load, first) || !compatible_endpoints(load, tail) ||
            first.block != tail.block)
          continue;
        std::optional<AcquirePathProof> proof = prove_acquire_path(load, first, points, *profile);
        const uint64_t first_end = first.instruction->src_loc() + first.instruction->size();
        // Pair members are exact consecutive fence events. Only waits may
        // separate them; the original 64-byte bound applies to the first.
        if (!proof || !scan_bookkeeping(*first.block, first_end, tail.instruction->src_loc(),
                                        *load.container, points, *proof, *profile, true, true))
          continue;
        pair_matches.push_back({.caches = {&first, &tail}, .proof = std::move(*proof)});
      }
    }
    std::optional<Association> association;
    if (single_matches.size() == 1)
      association = std::move(single_matches.front());
    else if (single_metadata_matches == 0 && single_matches.empty() && pair_matches.size() == 1)
      association = std::move(pair_matches.front());
    if (!association ||
        association->caches.front()->instruction->mnemonic() != profile->first_cache ||
        !qualifying_acquire_cache(*profile, *association->caches.front()->instruction,
                                  association->caches.front()->bytes) ||
        association->caches.size() != (profile->second_cache.empty() ? 1u : 2u) ||
        (association->caches.size() == 2 &&
         association->caches.back()->instruction->mnemonic() != profile->second_cache) ||
        std::ranges::any_of(association->caches, [&](const AcquirePoint *cache) {
          return claimed_caches.contains(cache->instruction->src_loc());
        }))
      continue;
    OrdinaryAcquireSite site = {.offset = item.first,
                                .instruction_size = static_cast<uint32_t>(load.instruction->size()),
                                .form = std::string(load.instruction->mnemonic()),
                                .profile = profile->id,
                                .byte_offset = decoded->byte_offset,
                                .raw_scope = decoded->raw_scope,
                                .path = association->proof.kind,
                                .container_entry = load.container->begin,
                                .container_size = load.container->size,
                                .container_name = std::string(load.container->name),
                                .polling_loop_header = association->proof.loop_header,
                                .cache_members = {},
                                .wait_members = association->proof.waits,
                                .affected_owners = load.owners};
    record->source_bytes.emplace(item.first, load.bytes);
    for (const AcquirePoint *cache : association->caches) {
      site.cache_members.push_back(member_of(*cache->instruction));
      claimed_caches.insert(cache->instruction->src_loc());
      record->source_bytes.emplace(cache->instruction->src_loc(), cache->bytes);
    }
    for (uint64_t offset : association->proof.proof_offsets) {
      const std::map<uint64_t, AcquirePoint>::const_iterator proof_point = points.find(offset);
      if (proof_point == points.end())
        return reject_fault_mutation(error, FaultMutationErrorCode::ValidationFailure,
                                     "ordinary acquire proof lost an exact source instruction",
                                     offset);
      record->source_bytes.emplace(offset, proof_point->second.bytes);
    }
    record->sites.push_back(std::move(site));
  }
  OrdinaryAcquireInventory inventory;
  inventory.record_ = std::move(record);
  return inventory;
}

FailureOr<OrdinaryAcquireSelection>
select_ordinary_acquire(const OrdinaryAcquireInventory &inventory,
                        const OrdinaryAcquireSelector &selector, FaultMutationError &error) {
  error = {};
  if (!inventory.record_ || inventory.record_->source.expired())
    return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                 "ordinary acquire source expired");
  if (selector.exact) {
    const OrdinaryAcquireSelection &exact = *selector.exact;
    if (!exact.record_ || exact.record_->source.lock() != inventory.record_->source.lock() ||
        std::ranges::none_of(inventory.record_->sites, [&](const OrdinaryAcquireSite &site) {
          return site == exact.site_ &&
                 std::ranges::find(site.affected_owners, inventory.record_->owner) !=
                     site.affected_owners.end();
        }))
      return reject_fault_mutation(error, FaultMutationErrorCode::BadIdentity,
                                   "exact acquire does not match source and affected owner");
    OrdinaryAcquireSelection selection;
    selection.record_ = inventory.record_;
    selection.site_ = exact.site_;
    return selection;
  }
  if (inventory.record_->sites.empty())
    return reject_fault_mutation(error, FaultMutationErrorCode::NoCandidates,
                                 "no owned qualified acquire load");
  if (selector.ordinal >= inventory.record_->sites.size())
    return reject_fault_mutation(error, FaultMutationErrorCode::OrdinalOutOfRange,
                                 "qualified acquire ordinal out of range");
  OrdinaryAcquireSelection selection;
  selection.record_ = inventory.record_;
  selection.site_ = inventory.record_->sites[selector.ordinal];
  return selection;
}

} // namespace rocjitsu
