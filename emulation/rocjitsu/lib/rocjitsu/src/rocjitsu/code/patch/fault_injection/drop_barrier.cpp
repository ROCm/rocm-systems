// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/fault_injection/drop_barrier.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/analysis/control_flow.h"
#include "rocjitsu/code/basic_block.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/code/kernel_scope.h"
#include "rocjitsu/code/patch/checked_text_edits.h"
#include "rocjitsu/code/patch/instrumentor.h"
#include "rocjitsu/isa/decoder.h"
#include "rocjitsu/isa/instruction.h"
#include "rocjitsu/isa/target_registry.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace rocjitsu {
namespace {

Result reject(FaultMutationError &error, FaultMutationErrorCode code, std::string message,
              uint64_t offset = 0) {
  error = {.code = code, .offset = offset, .message = std::move(message)};
  return Result::failure();
}

FaultMutationError translate_edit_error(const TextEditError &edit) {
  if (edit.code == TextEditErrorCode::UnsupportedTarget)
    return {.code = FaultMutationErrorCode::UnsupportedTarget,
            .offset = edit.offset,
            .message = edit.message};
  const bool unsupported = edit.code == TextEditErrorCode::UnsupportedReferences ||
                           edit.code == TextEditErrorCode::Clause ||
                           edit.code == TextEditErrorCode::InvalidInstruction ||
                           edit.code == TextEditErrorCode::InvalidImage;
  return {.code = unsupported ? FaultMutationErrorCode::UnsupportedForm
                              : FaultMutationErrorCode::ValidationFailure,
          .offset = edit.offset,
          .message = edit.message};
}

// Called only after AmdGpuCodeObject validates segment extents and VA arithmetic.
// Section payloads alone do not establish what a loader places at an address.
std::optional<uint64_t> resolve_file_backed_address(std::span<const Elf64_Phdr> segments,
                                                    uint64_t offset, uint64_t size,
                                                    uint32_t required_flags) {
  if (size == 0)
    return std::nullopt;
  const Elf64_Phdr *mapping = nullptr;
  uint64_t address = 0;
  for (const Elf64_Phdr &segment : segments) {
    if (segment.p_type != PT_LOAD || offset < segment.p_offset)
      continue;
    const uint64_t delta = offset - segment.p_offset;
    if (delta > segment.p_filesz || size > segment.p_filesz - delta)
      continue;
    if (mapping || (segment.p_flags & required_flags) != required_flags)
      return std::nullopt;
    mapping = &segment;
    address = segment.p_vaddr + delta;
  }
  if (!mapping)
    return std::nullopt;
  // A partial file alias can load these bytes at a disjoint address outside the
  // analyzed ownership scope. Reject it independently of virtual overlap, then
  // check memory ranges including zero-fill tails. Neither check depends on the
  // segment order or a particular loader's policy.
  for (const Elf64_Phdr &segment : segments) {
    if (&segment == mapping || segment.p_type != PT_LOAD)
      continue;
    if (segment.p_filesz != 0 && segment.p_offset < offset + size &&
        (segment.p_offset >= offset || segment.p_filesz > offset - segment.p_offset))
      return std::nullopt;
    if (segment.p_memsz != 0 && segment.p_vaddr < address + size &&
        (segment.p_vaddr >= address || segment.p_memsz > address - segment.p_vaddr))
      return std::nullopt;
  }
  return address;
}

// FI qualifies raw descriptor facts; it does not resolve load bias or symbols.
// Protect the complete descriptor object from the dynamic writes handled by
// ROCr's ApplyDynamicRelocation. Explicit-target/static metadata is not applied.
Result validate_descriptor_relocation_writes(std::span<const uint8_t> image,
                                             const Elf64_Ehdr &header,
                                             std::span<const Elf64_Shdr> sections,
                                             std::span<const uint64_t> descriptor_addresses,
                                             FaultMutationError &error) {
  for (const Elf64_Shdr &section : sections) {
    if (classify_rocr_relocation_section(header, sections, section) !=
        RocrRelocationSectionMode::Dynamic)
      continue;
    // ROCr materializes complete contiguous RELA records, including those before
    // a partial tail. Source admission has already validated their file extents.
    const std::optional<std::vector<Elf64_Rela>> relocations = read_elf_table<Elf64_Rela>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Rela), sizeof(Elf64_Rela));
    if (!relocations)
      return reject(error, FaultMutationErrorCode::UnsupportedForm,
                    "unreadable dynamic relocations for kernel descriptors");
    for (const Elf64_Rela &relocation : *relocations) {
      uint64_t write_size = 0;
      switch (elf_reloc_type(relocation.r_info)) {
      case R_AMDGPU_ABS32_LO:
      case R_AMDGPU_ABS32_HI:
      case R_AMDGPU_ABS32:
        write_size = sizeof(uint32_t);
        break;
      case R_AMDGPU_ABS64:
      case R_AMDGPU_RELATIVE64:
        write_size = sizeof(uint64_t);
        break;
      default:
        continue; // Other forms are not writes supported by the current dynamic loader.
      }
      for (const uint64_t address : descriptor_addresses) {
        const bool overlaps =
            relocation.r_offset >= address
                ? relocation.r_offset - address < sizeof(amdhsa::kernel_descriptor_t)
                : write_size > address - relocation.r_offset;
        if (overlaps)
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "dynamic relocation writes a kernel descriptor");
      }
    }
  }
  return Result::success();
}

} // namespace

// Cached per-kernel qualification, owned by the context and any returned inventories.
// Its weak source link lets a retained inventory detect context expiry.
struct FaultBarrierRecord {
  FaultMutationSection section;
  FaultMutationOwner owner;
  std::vector<StaticBarrier> barriers;
  std::weak_ptr<const AmdGpuCodeObject> source;
  FaultMutationError error; // Kernel-local qualification failure; siblings may still be usable.
};

class FaultMutationContext::State {
public:
  std::once_flag analysis_once;
  FaultMutationError error; // Whole-image admission/ownership failure, shared by every discovery.
  bool analysis_ok = false;
  std::vector<std::shared_ptr<const FaultBarrierRecord>> kernels;
  // Borrow the context's source; forks share these pristine facts, not attempt state.
  std::unique_ptr<Instrumentor> instrumentor;
};

FaultMutationContext::FaultMutationContext(std::span<const uint8_t> image)
    : source_(std::make_shared<const AmdGpuCodeObject>(image.data(), image.size())),
      state_(std::make_unique<State>()) {}
FaultMutationContext::~FaultMutationContext() = default;

std::span<const uint8_t> FaultMutationContext::image() const {
  return {reinterpret_cast<const uint8_t *>(source_->image_data()), source_->image_size()};
}

bool DropBarrierSelection::belongs_to(const FaultMutationContext &context) const {
  return source_.lock().get() == &context.source();
}

const FaultMutationSection &FaultBarrierInventory::section() const { return record_->section; }
const FaultMutationOwner &FaultBarrierInventory::owner() const { return record_->owner; }
std::span<const StaticBarrier> FaultBarrierInventory::barriers() const { return record_->barriers; }

FailureOr<DropBarrierSelection> select_drop_barrier(const FaultBarrierInventory &inventory,
                                                    const DropBarrierSelector &selector,
                                                    FaultMutationError &error) {
  error = {};
  const std::shared_ptr<const FaultBarrierRecord> &record = inventory.record_;
  if (!record || record->source.expired())
    return reject(error, FaultMutationErrorCode::BadIdentity, "expired or unbound inventory");
  // TODO(hanchung): Add a better barrier-selection heuristic while preserving explicit selection
  // and reproducibility. Currently, an exact selection takes precedence; otherwise we use the
  // requested zero-based logical ordinal in first-member source order, defaulting to zero. Repeated
  // default requests choose the same barrier; no race scoring is used.
  if (selector.exact) {
    if (selector.exact->record_.lock() != record)
      return reject(error, FaultMutationErrorCode::BadIdentity,
                    "exact selector belongs to another source or kernel");
    return *selector.exact;
  }
  if (record->barriers.empty())
    return reject(error, FaultMutationErrorCode::NoCandidates, "no complete qualified barriers");
  if (selector.ordinal >= record->barriers.size())
    return reject(error, FaultMutationErrorCode::OrdinalOutOfRange,
                  "logical barrier ordinal out of range");
  DropBarrierSelection selection;
  selection.owner_ = record->owner;
  selection.barrier_ = record->barriers[selector.ordinal];
  selection.source_ = record->source;
  selection.record_ = record;
  selection.ordinal_ = selector.ordinal;
  return selection;
}

std::vector<FaultMutationCapability> fault_mutation_capabilities() {
  const IsaTargetRegistry &registry = default_isa_target_registry();
  std::vector<FaultMutationCapability> capabilities;
  for (const IsaTargetDescriptor &family : registry.targets()) {
    for (const IsaGpuTargetDescription &binding : family.gpu_targets) {
      const std::optional<StaticBarrierPolicy> policy =
          static_barrier_policy(binding.public_id, registry);
      if (!policy || !Decoder::create(registry, binding.public_id))
        continue;
      for (const uint32_t wave : policy->wave_sizes)
        capabilities.push_back(
            {.kind = FaultMutationKind::DropBarrier,
             .form = policy->form,
             .architecture = family.architecture_id,
             .target = binding.public_id,
             .wave_size = wave,
             .admission = "shared DBI ELF admission; file-backed text/descriptors; exact kernel; "
                          "independent siblings; closed direct CFG; complete builtin barrier"});
    }
  }
  return capabilities;
}

Result FaultMutationContext::analyze() const {
  // Admit one immutable image and its source-derived concrete target. Family
  // policy limits mutation forms; it must not substitute an architecture decoder.
  FaultMutationError &error = state_->error;
  const std::span<const uint8_t> image = this->image();
  const AmdGpuCodeObject &object = source();
  const rj_code_target_id_t target = object.target_id();
  Elf64_Ehdr header = {};
  if (image.size() < sizeof(header))
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "truncated ELF header");
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "expected little-endian AMDHSA ET_DYN ELF64");
  const IsaTargetRegistry &registry = default_isa_target_registry();
  const IsaTargetDescriptor *target_facts = registry.find(target);
  const IsaGpuTargetDescription *binding = registry.find_gpu_target(target);
  const std::optional<StaticBarrierPolicy> policy = static_barrier_policy(target, registry);
  if (!target_facts || !binding || !policy ||
      (header.e_flags & EF_AMDGPU_MACH) != binding->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget,
                  "requires a registered complete-barrier target with matching ELF flags");
  if (!object.is_valid())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid code object");
  if (object.text_sections().size() != 1)
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires one executable .text; multi-text is unsupported");
  const Section &text = *object.text_sections().front();
  if ((text.flags() & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR))
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "selected .text must be allocated and executable");
  // Establish load correspondence for every hardware entry, including siblings.
  // Section-relative descriptor facts alone do not prove a loader uses those bytes.
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!segments)
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid program headers");
  const std::optional<uint64_t> text_address =
      resolve_file_backed_address(*segments, text.sectionOffset(), text.size(), PF_X);
  if (!text_address || *text_address != text.vaddr())
    return reject(error, FaultMutationErrorCode::UnsupportedForm,
                  "requires unambiguous file-backed executable .text at its declared address");
  // Loaded VAs of complete descriptors whose file facts define owner admission.
  std::vector<uint64_t> descriptor_addresses;
  descriptor_addresses.reserve(kernels.size());
  for (const KernelDescriptorInfo &owner : kernels) {
    const std::optional<uint64_t> descriptor_address = resolve_file_backed_address(
        *segments, owner.descriptor_file_offset, sizeof(owner.descriptor), 0);
    const std::optional<uint64_t> loaded_entry =
        descriptor_address
            ? add_signed_offset(*descriptor_address, owner.descriptor.kernel_code_entry_byte_offset)
            : std::nullopt;
    // The scanner has already checked each section-based entry extent.
    if (!loaded_entry || *loaded_entry != text.vaddr() + owner.entry_text_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm,
                    "descriptor load mapping does not resolve to its analyzed entry");
    if (kernarg_preload_length(owner.descriptor) != 0)
      return reject(error, FaultMutationErrorCode::UnsupportedForm,
                    "firmware preload ownership requires additional entry analysis");
    descriptor_addresses.push_back(*descriptor_address);
  }

  // Full-text structural facts cover even unreachable references. Independently
  // reconcile declared owners: a descriptor omitted by the scanner is unresolved,
  // not evidence that the owner does not exist.
  state_->instrumentor = std::make_unique<Instrumentor>(object, target_facts->architecture_id);
  TextEditError edit_error;
  const InstructionBoundaries *boundaries =
      state_->instrumentor->instruction_boundaries(edit_error);
  if (!boundaries) {
    error = translate_edit_error(edit_error);
    return Result::failure();
  }
  const std::optional<std::vector<Elf64_Shdr>> section_headers =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  if (!section_headers)
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid section headers");
  const std::span<const Elf64_Shdr> sections = *section_headers;
  if (validate_descriptor_relocation_writes(image, header, sections, descriptor_addresses, error)
          .failed())
    return Result::failure();
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM)
      continue;
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "malformed symbol table");
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    if (!symbols || section.sh_link >= sections.size())
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol table range");
    const Elf64_Shdr &strings = sections[section.sh_link];
    if (strings.sh_type != SHT_STRTAB || strings.sh_offset > image.size() ||
        strings.sh_size > image.size() - strings.sh_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol string table");
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_name >= strings.sh_size)
        return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid symbol name");
      const char *name =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const void *end = std::memchr(name, 0, strings.sh_size - symbol.st_name);
      if (!end)
        return reject(error, FaultMutationErrorCode::UnsupportedForm, "unterminated symbol name");
      const std::string_view symbol_name(name, static_cast<const char *>(end) - name);
      // The scanner omits malformed descriptors. Every declared owner must
      // resolve before all-entry boundaries can establish exclusive ownership.
      if (symbol_name.ends_with(".kd")) {
        if (symbol.st_shndx >= sections.size() ||
            symbol.st_size != sizeof(amdhsa::kernel_descriptor_t) ||
            elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
            elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal)
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "unresolved descriptor owner");
        const Elf64_Shdr &owner = sections[symbol.st_shndx];
        if (owner.sh_type == SHT_NOBITS || symbol.st_value < owner.sh_addr ||
            symbol.st_value - owner.sh_addr > owner.sh_size ||
            symbol.st_size > owner.sh_size - (symbol.st_value - owner.sh_addr) ||
            owner.sh_offset > image.size() || owner.sh_size > image.size() - owner.sh_offset)
          return reject(error, FaultMutationErrorCode::UnsupportedForm, "invalid descriptor owner");
        const uint64_t file_offset = owner.sh_offset + (symbol.st_value - owner.sh_addr);
        if (!std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &candidate) {
              return candidate.kernel_name == symbol_name.substr(0, symbol_name.size() - 3) &&
                     candidate.descriptor_file_offset == file_offset;
            }))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "descriptor symbol was omitted from the owner inventory");
      }
      const uint8_t type = elf_symbol_type(symbol.st_info);
      if (symbol.st_shndx != text.sectionHeaderIndex()) {
        if ((type == kElfSymbolTypeFunc || type == kElfSymbolTypeAmdGpuHsaKernel) &&
            symbol.st_shndx < sections.size() &&
            (sections[symbol.st_shndx].sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) ==
                (SHF_ALLOC | SHF_EXECINSTR))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "executable owners outside .text require broader ownership analysis");
        continue;
      }
      const bool external = elf_symbol_bind(symbol.st_info) != kElfSymbolBindLocal;
      if (type == kElfSymbolTypeObject || type == kElfSymbolTypeAmdGpuHsaKernel ||
          ((type == kElfSymbolTypeFunc || external) &&
           (symbol.st_value < text.vaddr() ||
            !std::ranges::any_of(kernels, [&](const KernelDescriptorInfo &candidate) {
              return symbol.st_value - text.vaddr() == candidate.entry_text_offset;
            }))))
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "separate function/data owners inside .text are outside this profile");
    }
  }
  // Build Reachable / ExplicitOnly CFG facts from all descriptor entries. Starting
  // only at the selected entry would hide cross-kernel or shared-block ownership;
  // these reachability facts cannot replace the full-text structural scan above.
  std::unique_ptr<Decoder> decoder = Decoder::create(registry, target);
  if (!decoder)
    return reject(error, FaultMutationErrorCode::UnsupportedTarget,
                  "concrete target decoder unavailable");
  std::vector<uint64_t> entries;
  std::unordered_set<uint64_t> all_entries;
  for (const KernelDescriptorInfo &owner : kernels) {
    if (!all_entries.insert(owner.entry_text_offset).second)
      return reject(error, FaultMutationErrorCode::AmbiguousSelection,
                    "kernel descriptors alias one hardware entry");
    entries.push_back(owner.entry_text_offset);
  }
  util::StringDiagnostic diagnostic;
  FailureOr<std::vector<std::unique_ptr<BasicBlock>>> blocks =
      BasicBlock::build_reachable(object, *decoder, target_facts->architecture_id, entries,
                                  diagnostic.emitter(), {}, {}, {}, target);
  if (blocks.failed())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, diagnostic.message());
  const BlockOffsetIndex index = build_block_offset_index(blocks.value());
  const BlockPositionIndex positions = build_block_position_index(blocks.value());
  std::unordered_map<const BasicBlock *, uint64_t> block_owners;
  for (const KernelDescriptorInfo &owner : kernels) {
    BasicBlock *root = block_for_offset(index, owner.entry_text_offset);
    if (!root || root->start_offset() != owner.entry_text_offset)
      return reject(error, FaultMutationErrorCode::UnsupportedForm, "entry is not a block start");
    const KernelScopeSpec scope_spec = {.kernel_entries = all_entries,
                                        .own_entries = {owner.entry_text_offset},
                                        .address_taken_entries = {}};
    std::vector<BasicBlock *> scope =
        reachable_kernel_blocks(blocks.value(), index, positions, *root, scope_spec);
    for (BasicBlock *block : scope) {
      const std::pair<std::unordered_map<const BasicBlock *, uint64_t>::iterator, bool> inserted =
          block_owners.emplace(block, owner.entry_text_offset);
      if (!inserted.second && inserted.first->second != owner.entry_text_offset)
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "shared reachable blocks require broader ownership analysis",
                      block->start_offset());
      if (block->successor_issue() != BasicBlock::SuccessorIssue::None ||
          block->has_implicit_terminator() || !block->call_edges().empty())
        return reject(error, FaultMutationErrorCode::UnsupportedForm,
                      "kernel ownership requires a closed direct CFG", block->start_offset());
      for (const BasicBlock *successor : block->successors()) {
        if (all_entries.contains(successor->start_offset()) &&
            successor->start_offset() != owner.entry_text_offset)
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "control flow crosses another kernel entry", block->start_offset());
      }
      for (const Instruction &instruction : block->instructions()) {
        if ((instruction.flags() & (INDIRECT_BRANCH | INDIRECT_CALL)) != 0 ||
            (is_control_flow_transfer(instruction) && !instruction.branch_offset_bytes()))
          return reject(error, FaultMutationErrorCode::UnsupportedForm,
                        "indirect owner is unsupported", instruction.src_loc());
      }
    }
    // Ownership failures reject the whole profile. Once this scope is exclusive,
    // cache its wave/form error or qualified barriers in an immutable record so
    // an unsupported sibling barrier does not disqualify an independent kernel.
    auto record = std::make_shared<FaultBarrierRecord>();
    record->section = {.index = *text.sectionHeaderIndex(),
                       .file_offset = text.sectionOffset(),
                       .vaddr = text.vaddr(),
                       .size = text.size()};
    record->owner = {.entry = owner.kernel_name,
                     .entry_offset = owner.entry_text_offset,
                     .descriptor_file_offset = owner.descriptor_file_offset};
    record->source = source_;
    const uint32_t wave = kernel_wavefront_size(target_facts->architecture_id, owner.descriptor);
    // The shared helper normalizes CDNA3/4 to Wave64 even if the raw Wave32 bit
    // is set. Check that bit against family policy, independently of barrier form:
    // RDNA3 full singletons admit both waves; CDNA3/4 full singletons do not.
    const bool descriptor_requests_wave32 =
        (owner.descriptor.kernel_code_properties &
         amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32) != 0;
    if (std::ranges::find(policy->wave_sizes, wave) == policy->wave_sizes.end() ||
        (descriptor_requests_wave32 &&
         std::ranges::find(policy->wave_sizes, 32u) == policy->wave_sizes.end())) {
      record->error = {.code = FaultMutationErrorCode::UnsupportedForm,
                       .offset = 0,
                       .message = "descriptor wave mode is not admitted by this barrier family"};
    } else {
      StaticBarrierError barrier_error;
      FailureOr<std::vector<StaticBarrier>> barriers =
          qualify_static_barriers(scope, target, *boundaries, barrier_error);
      if (barriers.failed()) {
        record->error = {.code = barrier_error.code == StaticBarrierErrorCode::IncompleteBarrier
                                     ? FaultMutationErrorCode::IncompleteSelection
                                     : FaultMutationErrorCode::UnsupportedForm,
                         .offset = barrier_error.offset,
                         .message = barrier_error.message};
      } else {
        record->barriers = std::move(barriers).value();
        std::ranges::sort(record->barriers, {}, [](const StaticBarrier &barrier) {
          return barrier.members.front().offset;
        });
      }
    }
    state_->kernels.push_back(std::move(record));
  }
  if (block_owners.size() != blocks.value().size())
    return reject(error, FaultMutationErrorCode::UnsupportedForm, "unresolved code ownership");
  return Result::success();
}

FailureOr<FaultBarrierInventory> discover_drop_barriers(const FaultMutationContext &context,
                                                        std::string_view entry,
                                                        FaultMutationError &error) {
  error = {};
  std::call_once(context.state_->analysis_once,
                 [&] { context.state_->analysis_ok = context.analyze().succeeded(); });
  if (!context.state_->analysis_ok) {
    error = context.state_->error;
    return Result::failure();
  }
  // First choose one exact kernel record; ambiguous names never select the first.
  // Logical-barrier ordering and exact-handle precedence apply within that record.
  const std::vector<std::shared_ptr<const FaultBarrierRecord>> &kernels = context.state_->kernels;
  const std::ptrdiff_t matches =
      std::ranges::count_if(kernels, [&](const std::shared_ptr<const FaultBarrierRecord> &record) {
        return record->owner.entry == entry;
      });
  if (matches > 1)
    return reject(error, FaultMutationErrorCode::AmbiguousSelection,
                  "exact entry name identifies multiple descriptors");
  if (entry.empty() || matches == 0)
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "entry must match one exact kernel name");
  const std::shared_ptr<const FaultBarrierRecord> record = *std::ranges::find_if(
      kernels, [&](const std::shared_ptr<const FaultBarrierRecord> &candidate) {
        return candidate->owner.entry == entry;
      });
  if (record->error.code != FaultMutationErrorCode::None) {
    error = record->error;
    return Result::failure();
  }
  FaultBarrierInventory inventory;
  inventory.record_ = record;
  return inventory;
}

FailureOr<DropBarrierResult> apply_drop_barrier(const FaultMutationContext &context,
                                                const DropBarrierSelection &selection,
                                                FaultMutationError &error) {
  error = {};
  if (!selection.belongs_to(context))
    return reject(error, FaultMutationErrorCode::BadIdentity,
                  "selection belongs to another or expired source context");
  const std::shared_ptr<const FaultBarrierRecord> record = selection.record_.lock();
  // Only selection can bind the private record and validated ordinal. The matching
  // live context retains that immutable record throughout apply.
  assert(record && selection.ordinal_ < record->barriers.size());
  const StaticBarrier &barrier = record->barriers[selection.ordinal_];
  const IsaTargetDescriptor *target =
      default_isa_target_registry().find(context.source().target_id());
  // Only successful analyze/discover can produce a bound selection. That path
  // admits the immutable source target and initializes the shared source facts.
  assert(target && context.state_->instrumentor);
  TextEditError edit_error;
  FailureOr<std::unique_ptr<Instrumentor>> fork = context.state_->instrumentor->fork(edit_error);
  if (fork.failed()) {
    error = translate_edit_error(edit_error);
    return Result::failure();
  }
  std::unique_ptr<Instrumentor> instrumentor = std::move(fork).value();
  const uint32_t nop = build_s_nop(0, target->architecture_id);
  for (const StaticBarrierMember &member : barrier.members) {
    InstructionReplacement replacement = {.expected_bytes = member.bytes,
                                          .replacement_bytes =
                                              std::vector<uint8_t>(member.bytes.size())};
    std::memcpy(replacement.replacement_bytes.data(), &nop, sizeof(nop));
    InstrumentationPoint point = {.anchor_offset = member.offset,
                                  .kind = InstrumentationKind::ReplaceWithInst,
                                  .probe_symbol = {},
                                  .probe_args = {},
                                  .replacement = std::move(replacement)};
    instrumentor->add_point(std::move(point));
  }
  InstrumentedCodeObject applied = instrumentor->patch();
  if (applied.edit_error) {
    error = translate_edit_error(*applied.edit_error);
    return Result::failure();
  }
  if (!applied.errors.empty()) {
    std::string message;
    for (const std::string &diagnostic : applied.errors) {
      if (!message.empty())
        message += "; ";
      message += diagnostic;
    }
    return reject(error, FaultMutationErrorCode::ValidationFailure, std::move(message));
  }
  DropBarrierResult result = {
      .selection = selection,
      .image = std::move(applied.elf_bytes),
      .assumptions = {
          "Static mutation affects every executed occurrence of the selected logical barrier.",
          "Same-block membership does not prove barrier convergence.",
          "Execution and detector qualification have not been performed."}};
  if (barrier.scope == StaticBarrierScope::Cluster)
    result.assumptions.emplace_back(
        "Cluster barrier -3 assumes a clustered dispatch with multiple workgroups per cluster; "
        "it is inert otherwise. A valid pristine barrier requires exactly one signal per workgroup "
        "after workgroup synchronization, with every wave waiting. Static qualification does not "
        "verify workgroup size or this protocol.");
  return result;
}
} // namespace rocjitsu
