// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdgpu_kernel_metadata.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include "hsa/AMDHSAKernelDescriptor.h" // Check SGPR allocation

#include <algorithm>
#include <concepts>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <utility>

namespace rocjitsu {

namespace {

class HsaHeader : public Header {
public:
  explicit HsaHeader(const Elf64_Ehdr &ehdr) : ehdr_(ehdr) {}

  uint64_t programHeaderOff() const override { return ehdr_.e_phoff; }
  int numProgramHeaders() const override { return static_cast<int>(ehdr_.e_phnum); }
  uint64_t sectionHeaderOff() const override { return ehdr_.e_shoff; }
  int numSectionHeaders() const override { return static_cast<int>(ehdr_.e_shnum); }
  int sectionHeaderStrIdx() const override { return static_cast<int>(ehdr_.e_shstrndx); }
  uint32_t flags() const override { return ehdr_.e_flags; }

private:
  Elf64_Ehdr ehdr_;
};

class HsaSection : public Section {
public:
  HsaSection(std::string name, std::unique_ptr<char[]> data, const Elf64_Shdr &shdr)
      : Section(std::move(name), std::move(data)), shdr_(shdr) {}

  std::size_t size() const override { return shdr_.sh_size; }
  uint64_t flags() const override { return shdr_.sh_flags; }
  uint64_t vaddr() const override { return shdr_.sh_addr; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
};

// The requested section objects and their mutually exclusive text/rodata
// classification slot must fit within the two-image section model. The runtime
// capacity check below additionally covers standard-library over-allocation.
static_assert(sizeof(HsaSection) + sizeof(std::unique_ptr<Section>) + sizeof(const Section *) <=
              2 * sizeof(Elf64_Shdr));

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

using detail::fits_in_bounds;

/*
 * \NPI new GPU: map its MACH value and its gfxNNNN triple to a target id in \
 * both target_from_machine_flags() and target_from_triple() below.
 */

struct FunctionSymbolInfo {
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t text_size = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
};

struct FunctionEntryEvidence {
  uint64_t explicit_code_size = 0;
  bool conflicting_explicit_sizes = false;
};

using FunctionEntryKey = std::pair<uint64_t, uint64_t>;

using amdgpu_code_object_detail::KernelMetadata;
using amdgpu_code_object_detail::KernelMetadataVisitStatus;

struct KernelMetadataEntry {
  KernelMetadata metadata;
  uint64_t source_note = 0;
};

using KernelMetadataMap = std::unordered_map<std::string_view, KernelMetadataEntry>;
static_assert(kAmdGpuCodeObjectKernelMetadataEntryChargeBytes >=
                  sizeof(KernelMetadataMap::value_type) +
                      amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes,
              "update kKernelMetadataEntryBytes to cover KernelMetadataMap::value_type");
static_assert(kAmdGpuCodeObjectKernelMetadataEntryChargeBytes <=
                  sizeof(KernelMetadataMap::value_type) +
                      amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes +
                      2 * alignof(void *),
              "kKernelMetadataEntryBytes has drifted above the real entry layout");

[[nodiscard]] uint64_t align4(uint64_t value) { return (value + 3u) & ~uint64_t{3}; }

struct KernelMetadataReadResult {
  KernelMetadataMap metadata;
  bool budget_exceeded = false;
  std::optional<byte_accounting::ChargeOutcome> parse_work_failure;
  bool replay_failed = false;
};

template <typename RetainDerivedStateBytes>
  requires std::invocable<RetainDerivedStateBytes &, uint64_t> &&
           std::convertible_to<std::invoke_result_t<RetainDerivedStateBytes &, uint64_t>, bool>
[[nodiscard]] KernelMetadataReadResult
read_kernel_metadata(std::span<const uint8_t> image, const Elf64_Ehdr &header,
                     RetainDerivedStateBytes &&retain_derived_state_bytes) {
  KernelMetadataReadResult result;
  uint64_t next_source_note = 0;
  byte_accounting::CheckedByteBudget metadata_parse_work(byte_accounting::saturating_multiply(
      kAmdGpuCodeObjectMetadataParseWorkImageUnits, image.size()));
  if (header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phoff > image.size() ||
      static_cast<uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr) > image.size() - header.e_phoff) {
    return result;
  }
  for (uint16_t index = 0; index < header.e_phnum; ++index) {
    Elf64_Phdr program_header{};
    std::memcpy(&program_header,
                image.data() + header.e_phoff + static_cast<uint64_t>(index) * sizeof(Elf64_Phdr),
                sizeof(program_header));
    if (program_header.p_type != PT_NOTE || program_header.p_offset > image.size() ||
        program_header.p_filesz > image.size() - program_header.p_offset) {
      continue;
    }
    uint64_t cursor = program_header.p_offset;
    const uint64_t end = cursor + program_header.p_filesz;
    while (cursor <= end && sizeof(Elf64_Nhdr) <= end - cursor) {
      Elf64_Nhdr note{};
      std::memcpy(&note, image.data() + cursor, sizeof(note));
      cursor += sizeof(note);
      const uint64_t name_bytes = align4(note.n_namesz);
      const uint64_t desc_bytes = align4(note.n_descsz);
      if (name_bytes > end - cursor || desc_bytes > end - cursor - name_bytes)
        break;
      const uint64_t desc_offset = cursor + name_bytes;
      cursor = desc_offset + desc_bytes;
      if (note.n_type != NT_AMDGPU_METADATA || note.n_descsz > image.size() - desc_offset)
        continue;
      const auto payload = image.subspan(static_cast<size_t>(desc_offset), note.n_descsz);
      // Account both passes before parsing. Valid non-overlapping payloads fit
      // within one image; repeated or overlapping note references cannot
      // multiply parser work beyond the image-proportional bound.
      const byte_accounting::ChargeResult parse_work_charge =
          metadata_parse_work.charge_allocation(payload.size(), 2);
      if (!parse_work_charge) {
        result.parse_work_failure = parse_work_charge.outcome;
        return result;
      }
      // Validate the entire note before retaining any part of it. The second
      // pass deliberately re-parses the bounded payload, then charges each
      // unique entry before its map allocation.
      if (amdgpu_code_object_detail::visit_kernel_metadata_payload(
              payload, [](std::string_view, const KernelMetadata &) { return true; }) !=
          KernelMetadataVisitStatus::Complete) {
        continue;
      }
      const uint64_t source_note = next_source_note++;
      const KernelMetadataVisitStatus parsed =
          amdgpu_code_object_detail::visit_kernel_metadata_payload(
              payload, [&](std::string_view name, const KernelMetadata &metadata) {
                if (auto existing = result.metadata.find(name); existing != result.metadata.end()) {
                  // Preserve the old last-record-wins rule within one note while
                  // retaining the old first-note-wins rule across notes.
                  if (existing->second.source_note == source_note)
                    existing->second.metadata = metadata;
                  return true;
                }
                if (!retain_derived_state_bytes(kAmdGpuCodeObjectKernelMetadataEntryChargeBytes)) {
                  result.budget_exceeded = true;
                  return false;
                }
                result.metadata.emplace(name, KernelMetadataEntry{metadata, source_note});
                return true;
              });
      if (result.budget_exceeded)
        return result;
      // The first validation pass consumed the same immutable payload, and the
      // only visitor failure is handled as budget exhaustion above.
      if (parsed != KernelMetadataVisitStatus::Complete) {
        result.replay_failed = true;
        return result;
      }
    }
  }
  return result;
}

rj_code_target_id_t target_from_machine_flags(uint32_t flags) {
  uint32_t mach = flags & EF_AMDGPU_MACH;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX90A)
    return ROCJITSU_CODE_TARGET_GFX90A;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX942)
    return ROCJITSU_CODE_TARGET_GFX942;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX950)
    return ROCJITSU_CODE_TARGET_GFX950;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1100)
    return ROCJITSU_CODE_TARGET_GFX1100;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1150)
    return ROCJITSU_CODE_TARGET_GFX1150;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1151)
    return ROCJITSU_CODE_TARGET_GFX1151;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1200)
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1201)
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1250)
    return ROCJITSU_CODE_TARGET_GFX1250;
  return ROCJITSU_CODE_TARGET_INVALID;
}

rj_code_target_id_t target_from_triple(const std::string &triple) {
  if (triple == "gfx90a")
    return ROCJITSU_CODE_TARGET_GFX90A;
  if (triple == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (triple == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
  if (triple == "gfx1100")
    return ROCJITSU_CODE_TARGET_GFX1100;
  if (triple == "gfx1150")
    return ROCJITSU_CODE_TARGET_GFX1150;
  if (triple == "gfx1151")
    return ROCJITSU_CODE_TARGET_GFX1151;
  if (triple == "gfx1200")
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (triple == "gfx1201")
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (triple == "gfx1250")
    return ROCJITSU_CODE_TARGET_GFX1250;
  return ROCJITSU_CODE_TARGET_INVALID;
}

} // namespace

AmdGpuCodeObject::AmdGpuCodeObject(AmdGpuCodeObject &&other) noexcept
    : target_id_(other.target_id_), offload_kind_(std::move(other.offload_kind_)),
      target_triple_(std::move(other.target_triple_)) {
  is_valid_ = other.is_valid_;
  image_ = std::move(other.image_);
  header_ = std::move(other.header_);
  sections_ = std::move(other.sections_);
  text_sections_ = std::move(other.text_sections_);
  rodata_sections_ = std::move(other.rodata_sections_);
  kd_offsets_ = std::move(other.kd_offsets_);
  kernels_ = std::move(other.kernels_);
  functions_ = std::move(other.functions_);
}

AmdGpuCodeObject::AmdGpuCodeObject(const std::string &elf_path) {
  try {
    image_ = detail::read_file_bytes(elf_path);
  } catch (const std::exception &) {
    is_valid_ = false;
    return;
  }

  if (image_.size() < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size,
                                   std::string offload_kind, std::string target_triple)
    : offload_kind_(std::move(offload_kind)), target_triple_(std::move(target_triple)) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_triple(target_triple_);
}

AmdGpuCodeObject::~AmdGpuCodeObject() = default;

void AmdGpuCodeObject::load_sections() {
  const auto shoff = header_->sectionHeaderOff();
  const int num_shdrs = header_->numSectionHeaders();
  if (num_shdrs <= 0 ||
      !fits_in_bounds(shoff, static_cast<uint64_t>(num_shdrs) * sizeof(Elf64_Shdr),
                      image_.size())) {
    is_valid_ = false;
    return;
  }

  const size_t section_count = static_cast<size_t>(num_shdrs);
  const auto section_header = [&](size_t index) -> std::optional<Elf64_Shdr> {
    if (index >= section_count)
      return std::nullopt;
    Elf64_Shdr shdr{};
    std::memcpy(&shdr, image_.data() + shoff + index * sizeof(Elf64_Shdr), sizeof(shdr));
    return shdr;
  };

  int shstrndx = header_->sectionHeaderStrIdx();
  if (shstrndx < 0 || static_cast<size_t>(shstrndx) >= section_count) {
    is_valid_ = false;
    return;
  }

  const std::optional<Elf64_Shdr> shstrtab_header = section_header(static_cast<size_t>(shstrndx));
  if (!shstrtab_header) {
    is_valid_ = false;
    return;
  }
  const Elf64_Shdr shstrtab = *shstrtab_header;
  if (!fits_in_bounds(shstrtab.sh_offset, shstrtab.sh_size, image_.size())) {
    is_valid_ = false;
    return;
  }
  const char *shstrtab_data = image_.data() + shstrtab.sh_offset;
  const auto section_name = [&](const Elf64_Shdr &shdr) -> std::string_view {
    if (shdr.sh_name >= shstrtab.sh_size)
      return {};
    const size_t max_len = static_cast<size_t>(shstrtab.sh_size - shdr.sh_name);
    return {shstrtab_data + shdr.sh_name, strnlen(shstrtab_data + shdr.sh_name, max_len)};
  };

  byte_accounting::CheckedByteBudget retained_derived_state(byte_accounting::saturating_multiply(
      kAmdGpuCodeObjectRetainedDerivedStateImageUnits, image_.size()));
  const auto retain_derived_state_bytes = [&](uint64_t charge) {
    if (!retained_derived_state.charge(charge)) {
      is_valid_ = false;
      return false;
    }
    return true;
  };

  // Bound aggregate string ownership before materializing HsaSection names.
  // Duplicate offsets and long unterminated suffixes must not amplify one
  // string table into a quadratic parser working set.
  uint64_t copied_section_name_bytes = 0;
  for (size_t i = 0; i < section_count; ++i) {
    const size_t name_len = section_name(*section_header(i)).size();
    if (name_len > image_.size() - copied_section_name_bytes) {
      is_valid_ = false;
      return;
    }
    copied_section_name_bytes += name_len;
  }

  // Section payloads are copied into owning Section objects. Bound their
  // aggregate before allocating so duplicate or overlapping headers cannot
  // amplify one input image into an unbounded parser working set.
  uint64_t copied_section_bytes = 0;
  size_t materialized_section_count = 0;
  size_t text_section_count = 0;
  size_t rodata_section_count = 0;
  const auto section_classification_bytes =
      amdgpu_code_object_detail::section_classification_charge(section_count);
  // The in-image section table makes this charge smaller than the image for
  // the current Elf64 layout. Keep the checked charge as a portability
  // backstop and to account it in the shared ledger before allocation.
  if (!section_classification_bytes || !retain_derived_state_bytes(*section_classification_bytes)) {
    is_valid_ = false;
    return;
  }
  const size_t section_classification_word_count =
      (*section_classification_bytes / sizeof(uint64_t));
  auto text_section_classification_words =
      std::make_unique<uint64_t[]>(section_classification_word_count);
  const std::span<uint64_t> text_section_classification(text_section_classification_words.get(),
                                                        section_classification_word_count);
  for (size_t i = 0; i < section_count; ++i) {
    const Elf64_Shdr shdr = *section_header(i);
    const std::string_view sec_name = section_name(shdr);
    if (shdr.sh_type == SHT_NULL || shdr.sh_type == SHT_NOBITS || sec_name.empty()) {
      continue;
    }
    if (!fits_in_bounds(shdr.sh_offset, shdr.sh_size, image_.size()) ||
        shdr.sh_size > image_.size() - copied_section_bytes) {
      is_valid_ = false;
      return;
    }
    copied_section_bytes += shdr.sh_size;
    // Only materialized, in-image sections participate in text classification.
    // NOBITS has no file bytes and cannot provide patchable function ranges.
    if (sec_name == ".text" &&
        !amdgpu_code_object_detail::set_section_classification(text_section_classification, i)) {
      is_valid_ = false;
      return;
    }
    ++materialized_section_count;
    text_section_count += sec_name == ".text";
    rodata_section_count += sec_name == ".rodata";
  }
  const auto is_materialized_text_section = [&](size_t index) {
    return index <= std::numeric_limits<Elf_Half>::max() &&
           is_regular_elf_section_index(static_cast<Elf_Half>(index)) &&
           amdgpu_code_object_detail::test_section_classification(text_section_classification,
                                                                  index);
  };

  // The in-image section table bounds the three slot-array reservations before
  // this check. Charge the actual capacities returned by the standard library
  // before materializing section objects. Exact reserve behavior fits by
  // construction; the runtime limit is the portability backstop for capacity
  // over-allocation or a larger private section type.
  sections_.reserve(materialized_section_count);
  text_sections_.reserve(text_section_count);
  rodata_sections_.reserve(rodata_section_count);
  byte_accounting::CheckedByteBudget section_collections(
      byte_accounting::saturating_multiply(image_.size(), 2u));
  if (!section_collections.charge_allocation(materialized_section_count, sizeof(HsaSection)) ||
      !section_collections.charge_allocation(sections_.capacity(),
                                             sizeof(std::unique_ptr<Section>)) ||
      !section_collections.charge_allocation(text_sections_.capacity(), sizeof(const Section *)) ||
      !section_collections.charge_allocation(rodata_sections_.capacity(),
                                             sizeof(const Section *))) {
    is_valid_ = false;
    return;
  }

  for (size_t i = 0; i < section_count; ++i) {
    const Elf64_Shdr shdr = *section_header(i);
    if (shdr.sh_type == SHT_NULL || shdr.sh_type == SHT_NOBITS)
      continue;
    const std::string_view sec_name = section_name(shdr);
    if (sec_name.empty())
      continue;

    auto sec_data = std::make_unique<char[]>(shdr.sh_size);
    std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
    sections_.emplace_back(
        std::make_unique<HsaSection>(std::string(sec_name), std::move(sec_data), shdr));

    if (sec_name == ".text") {
      text_sections_.push_back(sections_.back().get());
    } else if (sec_name == ".rodata") {
      rodata_sections_.push_back(sections_.back().get());
    }
  }

  // Parse both SHT_SYMTAB and SHT_DYNSYM. Stripped code objects may only
  // contain the latter. Transient maps retain string views into image_, while
  // the final object owns two copies of each kernel name (kd_offsets_ and
  // kernels_) and one of each function name. Charging each distinct retained
  // role's copied name plus conservative aggregate fixed state therefore bounds
  // names, public records, and overlapping transient structures within the
  // exported derived-state budget. Aggregate accounting matters because a
  // kernel, function, and dynamic-stack symbol commonly share one logical name.
  std::unordered_map<std::string_view, uint64_t> kernel_descriptor_offsets;
  std::unordered_map<std::string_view, uint64_t> descriptor_file_offsets;
  std::unordered_map<std::string_view, FunctionSymbolInfo> function_symbols;
  std::unordered_map<std::string_view, bool> dynamic_stack_symbols;
  constexpr uint64_t kAssociativeEntryBookkeepingBytes =
      amdgpu_code_object_detail::kAssociativeEntryBookkeepingBytes;
  static_assert(kAmdGpuCodeObjectKernelEntryChargeBytes >=
                sizeof(AmdGpuKernelInfo) + sizeof(decltype(kd_offsets_)::value_type) +
                    sizeof(decltype(kernel_descriptor_offsets)::value_type) +
                    sizeof(decltype(descriptor_file_offsets)::value_type) +
                    3 * kAssociativeEntryBookkeepingBytes);
  static_assert(kAmdGpuCodeObjectKernelAndTransientEntryChargeBytes >=
                sizeof(AmdGpuKernelInfo) + sizeof(decltype(kd_offsets_)::value_type) +
                    sizeof(decltype(kernel_descriptor_offsets)::value_type) +
                    sizeof(decltype(descriptor_file_offsets)::value_type) +
                    sizeof(decltype(dynamic_stack_symbols)::value_type) +
                    4 * kAssociativeEntryBookkeepingBytes);
  static_assert(kAmdGpuCodeObjectFunctionEntryChargeBytes >=
                sizeof(AmdGpuFunctionInfo) + sizeof(decltype(function_symbols)::value_type) +
                    sizeof(std::map<FunctionEntryKey, FunctionEntryEvidence>::value_type) +
                    2 * kAssociativeEntryBookkeepingBytes);
  static_assert(kAmdGpuCodeObjectFunctionAndTransientEntryChargeBytes >=
                sizeof(AmdGpuFunctionInfo) + sizeof(decltype(function_symbols)::value_type) +
                    sizeof(std::map<FunctionEntryKey, FunctionEntryEvidence>::value_type) +
                    sizeof(decltype(dynamic_stack_symbols)::value_type) +
                    3 * kAssociativeEntryBookkeepingBytes);
  static_assert(kAmdGpuCodeObjectTransientSymbolEntryChargeBytes >=
                sizeof(decltype(dynamic_stack_symbols)::value_type) +
                    kAssociativeEntryBookkeepingBytes);
  static_assert(kAmdGpuCodeObjectKernelAndTransientEntryChargeBytes ==
                kAmdGpuCodeObjectKernelEntryChargeBytes +
                    kAmdGpuCodeObjectTransientSymbolEntryChargeBytes);
  static_assert(kAmdGpuCodeObjectFunctionAndTransientEntryChargeBytes ==
                kAmdGpuCodeObjectFunctionEntryChargeBytes +
                    kAmdGpuCodeObjectTransientSymbolEntryChargeBytes);

  enum class SymbolRetentionRole {
    Kernel,
    Function,
    DynamicStack,
  };
  using amdgpu_code_object_detail::retained_symbol_role_charge;
  const auto retain_symbol_role = [&](std::string_view name, SymbolRetentionRole role) {
    bool has_kernel = kernel_descriptor_offsets.contains(name);
    bool has_function = function_symbols.contains(name);
    bool has_dynamic_stack = dynamic_stack_symbols.contains(name);
    const bool already_retained = role == SymbolRetentionRole::Kernel     ? has_kernel
                                  : role == SymbolRetentionRole::Function ? has_function
                                                                          : has_dynamic_stack;
    if (already_retained)
      return true;

    const uint64_t old_fixed_charge =
        retained_symbol_role_charge(has_kernel, has_function, has_dynamic_stack);
    has_kernel |= role == SymbolRetentionRole::Kernel;
    has_function |= role == SymbolRetentionRole::Function;
    has_dynamic_stack |= role == SymbolRetentionRole::DynamicStack;
    const uint64_t new_fixed_charge =
        retained_symbol_role_charge(has_kernel, has_function, has_dynamic_stack);
    const uint64_t copied_name_count = role == SymbolRetentionRole::Kernel     ? 2u
                                       : role == SymbolRetentionRole::Function ? 1u
                                                                               : 0u;
    if (copied_name_count != 0 &&
        name.size() > std::numeric_limits<uint64_t>::max() / copied_name_count) {
      is_valid_ = false;
      return false;
    }
    const uint64_t copied_name_bytes = copied_name_count * name.size();
    // Roles only transition from absent to present, and the exported helper is
    // additive in each role, so this subtraction is monotone by construction.
    const uint64_t fixed_charge_increment = new_fixed_charge - old_fixed_charge;
    if (copied_name_bytes > std::numeric_limits<uint64_t>::max() - fixed_charge_increment) {
      is_valid_ = false;
      return false;
    }
    return retain_derived_state_bytes(fixed_charge_increment + copied_name_bytes);
  };
  const auto symbol_file_offset = [&](const Elf64_Sym &sym) -> std::optional<uint64_t> {
    const std::optional<Elf64_Shdr> section = section_header(sym.st_shndx);
    if (!section)
      return std::nullopt;
    if (sym.st_value < section->sh_addr)
      return std::nullopt;
    const uint64_t section_delta = sym.st_value - section->sh_addr;
    if (section_delta > section->sh_size)
      return std::nullopt;
    return section->sh_offset + section_delta;
  };
  for (size_t i = 0; i < section_count; ++i) {
    const Elf64_Shdr symtab_shdr = *section_header(i);
    if (symtab_shdr.sh_type != SHT_SYMTAB && symtab_shdr.sh_type != SHT_DYNSYM)
      continue;
    if (symtab_shdr.sh_entsize < sizeof(Elf64_Sym) ||
        !fits_in_bounds(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_.size()) ||
        symtab_shdr.sh_link >= section_count) {
      continue;
    }
    const std::optional<Elf64_Shdr> strtab_header = section_header(symtab_shdr.sh_link);
    if (!strtab_header)
      continue;
    const Elf64_Shdr strtab_shdr = *strtab_header;
    if (!fits_in_bounds(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_.size()))
      continue;
    const char *sym_strtab = image_.data() + strtab_shdr.sh_offset;
    const size_t num_syms = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    const char *symtab_data = image_.data() + symtab_shdr.sh_offset;

    for (size_t sym_index = 0; sym_index < num_syms; ++sym_index) {
      const char *sym_data = symtab_data + sym_index * symtab_shdr.sh_entsize;
      Elf64_Sym sym{};
      std::memcpy(&sym, sym_data, sizeof(sym));
      if (sym.st_name >= strtab_shdr.sh_size)
        continue;
      const size_t max_len = static_cast<size_t>(strtab_shdr.sh_size - sym.st_name);
      const std::string_view sym_name(sym_strtab + sym.st_name,
                                      strnlen(sym_strtab + sym.st_name, max_len));
      constexpr std::string_view kDynamicStackSuffix = ".has_dyn_sized_stack";
      if (sym.st_shndx == SHN_ABS && sym_name.ends_with(kDynamicStackSuffix)) {
        const std::string_view kernel_name =
            sym_name.substr(0, sym_name.size() - kDynamicStackSuffix.size());
        auto dynamic_stack = dynamic_stack_symbols.find(kernel_name);
        if (dynamic_stack == dynamic_stack_symbols.end()) {
          if (!retain_symbol_role(kernel_name, SymbolRetentionRole::DynamicStack))
            return;
          dynamic_stack_symbols.emplace(kernel_name, sym.st_value != 0);
        } else {
          dynamic_stack->second = sym.st_value != 0;
        }
        continue;
      }
      // AMDHSA kernel descriptors have a ".kd" suffix symbol.
      if (sym_name.size() > 3 && sym_name.ends_with(".kd")) {
        const std::string_view kernel_name = sym_name.substr(0, sym_name.size() - 3);
        auto descriptor = kernel_descriptor_offsets.find(kernel_name);
        if (descriptor == kernel_descriptor_offsets.end()) {
          if (!retain_symbol_role(kernel_name, SymbolRetentionRole::Kernel))
            return;
          kernel_descriptor_offsets.emplace(kernel_name, sym.st_value);
        } else {
          descriptor->second = sym.st_value;
        }
        if (auto file_offset = symbol_file_offset(sym))
          descriptor_file_offsets[kernel_name] = *file_offset;
        continue;
      }

      if (elf_symbol_type(sym.st_info) == kElfSymbolTypeFunc &&
          is_materialized_text_section(sym.st_shndx)) {
        const std::optional<Elf64_Shdr> text_header = section_header(sym.st_shndx);
        if (!text_header)
          continue;
        const Elf64_Shdr text = *text_header;
        if (sym.st_value >= text.sh_addr && sym.st_value - text.sh_addr <= text.sh_size) {
          FunctionSymbolInfo info;
          info.entry_text_offset = sym.st_value - text.sh_addr;
          info.text_file_offset = text.sh_offset;
          info.text_size = text.sh_size;
          info.code_size = sym.st_size;
          const auto existing = function_symbols.find(sym_name);
          if (existing == function_symbols.end()) {
            if (!retain_symbol_role(sym_name, SymbolRetentionRole::Function))
              return;
            function_symbols.emplace(sym_name, info);
          } else if (existing->second.code_size == 0 && info.code_size != 0) {
            existing->second = info;
          }
        }
      }
    }
  }

  std::map<FunctionEntryKey, FunctionEntryEvidence> function_entries;
  for (const auto &[name, function] : function_symbols) {
    (void)name;
    auto &entry = function_entries[{function.text_file_offset, function.entry_text_offset}];
    if (function.code_size == 0)
      continue;
    if (entry.explicit_code_size != 0 && entry.explicit_code_size != function.code_size)
      entry.conflicting_explicit_sizes = true;
    else
      entry.explicit_code_size = function.code_size;
  }

  // Assembly-produced code objects may leave STT_FUNC sizes at zero. Prefer an
  // unambiguous explicit size from another symbol at the same entry: generated
  // device libraries commonly publish one real symbol plus many zero-sized
  // target-selection aliases. Only when no such ELF evidence exists, infer a
  // conservative range from the next distinct function entry in the same text
  // section, or from the section end for the final function.
  for (auto &[name, function] : function_symbols) {
    (void)name;
    if (function.code_size != 0 || function.entry_text_offset >= function.text_size)
      continue;
    const FunctionEntryKey key{function.text_file_offset, function.entry_text_offset};
    const auto evidence = function_entries.find(key);
    if (evidence != function_entries.end() && evidence->second.explicit_code_size != 0 &&
        !evidence->second.conflicting_explicit_sizes) {
      function.code_size = evidence->second.explicit_code_size;
      continue;
    }

    uint64_t end = function.text_size;
    const auto next =
        evidence == function_entries.end() ? function_entries.end() : std::next(evidence);
    if (next != function_entries.end() && next->first.first == function.text_file_offset)
      end = std::min(end, next->first.second);
    function.code_size_inferred_from_zero = true;
    function.code_size = end - function.entry_text_offset;
  }

  Elf64_Ehdr elf_header{};
  std::memcpy(&elf_header, image_.data(), sizeof(elf_header));
  auto metadata_result = read_kernel_metadata(
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(image_.data()), image_.size()),
      elf_header, retain_derived_state_bytes);
  if (metadata_result.budget_exceeded || metadata_result.parse_work_failure ||
      metadata_result.replay_failed) {
    is_valid_ = false;
    return;
  }
  const KernelMetadataMap &metadata = metadata_result.metadata;

  kernels_.clear();
  kernels_.reserve(kernel_descriptor_offsets.size());
  const auto excess_kernel_capacity = byte_accounting::excess_capacity_charge(
      kernels_.capacity(), kernel_descriptor_offsets.size(), sizeof(AmdGpuKernelInfo));
  if (!excess_kernel_capacity || !retain_derived_state_bytes(*excess_kernel_capacity)) {
    is_valid_ = false;
    return;
  }
  for (const auto &entry : kernel_descriptor_offsets) {
    const std::string_view kernel_name = entry.first;
    AmdGpuKernelInfo kernel;
    kernel.name = kernel_name;
    kd_offsets_[kernel.name] = entry.second;
    if (auto file_offset = descriptor_file_offsets.find(kernel_name);
        file_offset != descriptor_file_offsets.end())
      kernel.descriptor_file_offset = file_offset->second;

    if (auto func = function_symbols.find(kernel_name); func != function_symbols.end()) {
      kernel.entry_text_offset = func->second.entry_text_offset;
      kernel.text_file_offset = func->second.text_file_offset;
      kernel.text_size = func->second.text_size;
      kernel.code_size = func->second.code_size;
      kernel.code_size_inferred_from_zero = func->second.code_size_inferred_from_zero;
      kernel.has_text_range = true;
    } else {
      kernel.code_size = 0;
      kernel.has_text_range = false;
    }
    if (auto metadata_entry = metadata.find(std::string_view(kernel.name));
        metadata_entry != metadata.end()) {
      kernel.has_dynamic_lds = metadata_entry->second.metadata.has_dynamic_lds;
      kernel.uses_dynamic_stack = metadata_entry->second.metadata.uses_dynamic_stack;
      kernel.sgpr_count = metadata_entry->second.metadata.sgpr_count;
      kernel.required_workgroup_size = metadata_entry->second.metadata.required_workgroup_size;
    }
    if (!kernel.uses_dynamic_stack) {
      auto dynamic_stack = dynamic_stack_symbols.find(kernel_name);
      if (dynamic_stack != dynamic_stack_symbols.end())
        kernel.uses_dynamic_stack = dynamic_stack->second;
    }
    kernels_.push_back(std::move(kernel));
  }
  std::sort(kernels_.begin(), kernels_.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.name < rhs.name; });

  functions_.clear();
  functions_.reserve(function_symbols.size());
  const auto excess_function_capacity = byte_accounting::excess_capacity_charge(
      functions_.capacity(), function_symbols.size(), sizeof(AmdGpuFunctionInfo));
  if (!excess_function_capacity || !retain_derived_state_bytes(*excess_function_capacity)) {
    is_valid_ = false;
    return;
  }
  for (const auto &entry : function_symbols) {
    AmdGpuFunctionInfo function;
    function.name = entry.first;
    function.entry_text_offset = entry.second.entry_text_offset;
    function.text_file_offset = entry.second.text_file_offset;
    function.text_size = entry.second.text_size;
    function.code_size = entry.second.code_size;
    function.code_size_inferred_from_zero = entry.second.code_size_inferred_from_zero;
    functions_.push_back(std::move(function));
  }
  std::sort(functions_.begin(), functions_.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.entry_text_offset != rhs.entry_text_offset)
      return lhs.entry_text_offset < rhs.entry_text_offset;
    return lhs.name < rhs.name;
  });

  is_valid_ = true;
}

uint64_t AmdGpuCodeObject::kernel_descriptor_offset(const std::string &kernel_name) const {
  auto it = kd_offsets_.find(kernel_name);
  return it != kd_offsets_.end() ? it->second : 0;
}

namespace {

// CDNA targets encode the wavefront SGPR count in the descriptor even when the
// granulated field is 0; RDNA-style targets treat a granulated 0 as "use the
// fixed per-wave SGPR pool". Mirrors the command processor's
// sgpr_count_is_descriptor_encoded(); kept as the short, stable CDNA
// list so a new non-CDNA family falls through to the RDNA-style branch by
// default. (Canonical copy: code/dbt/kernel_descriptor_translator.cpp.)
[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

} // namespace

uint32_t amdgpu_kernel_descriptor_sgpr_count(uint32_t granulated, rj_code_arch_t arch) {
  // Descriptor-encoded (granulated != 0, or a CDNA target): (granulated + 1) * 8.
  // Otherwise the field is an RDNA-style sentinel and the wave owns the fixed
  // per-wave SGPR pool.
  if (granulated != 0 || is_cdna_arch(arch))
    return (granulated + 1) * 8;
  return amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF;
}

std::optional<uint32_t> AmdGpuCodeObject::min_kernel_sgpr_count(rj_code_arch_t arch) const {
  namespace kd = rocr::llvm::amdhsa;
  using KD = kd::kernel_descriptor_t;

  std::optional<uint32_t> min_count;
  for (const auto &[name, kd_vaddr] : kd_offsets_) {
    // Locate the section whose address range covers the .kd symbol, then read the
    // descriptor out of that section's own bytes (no ELF re-walk).
    for (const auto &section : all_sections()) {
      const uint64_t base = section->vaddr();
      if (base == 0 || kd_vaddr < base)
        continue;
      const uint64_t off = kd_vaddr - base;
      if (off + sizeof(KD) > section->size())
        continue;
      KD desc;
      std::memcpy(&desc, section->data() + off, sizeof(desc));
      const uint32_t granulated = AMDHSA_BITS_GET(
          desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
      const uint32_t count = amdgpu_kernel_descriptor_sgpr_count(granulated, arch);
      min_count = min_count ? std::min(*min_count, count) : count;
      break;
    }
  }
  return min_count;
}

} // namespace rocjitsu
