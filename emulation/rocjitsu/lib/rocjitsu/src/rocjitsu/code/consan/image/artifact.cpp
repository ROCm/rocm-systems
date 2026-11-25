// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/artifact.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdhsa_metadata.h"
#include "rocjitsu/code/consan/contract_internal.h"
#include "rocjitsu/code/consan/image/metadata.h"
#include "rocjitsu/code/consan/target.h"
#include "rocjitsu/isa/arch/amdgpu/generated/shared/isa_properties.h"
#include "rocjitsu/isa/isa_family.h"
#include "util/bit.h"
#include "util/sha256.h"

#include <algorithm>
#include <cstring>
#include <fstream>

namespace rocjitsu::consan {
namespace {
struct InspectedImage {
  Elf64_Shdr text = {};
  std::vector<KernelDescriptorInfo> kernels;
  amdhsa::MetadataDocument metadata;
};

const amdhsa::MetadataValue *metadata_entry(const amdhsa::MetadataDocument &document,
                                            std::string_view name) {
  const amdhsa::MetadataValue *kernels = document.root().field("amdhsa.kernels");
  if (kernels == nullptr) {
    return nullptr;
  }
  for (const amdhsa::MetadataValue &entry : kernels->array) {
    if (entry.text_field_is(".name", name)) {
      return &entry;
    }
  }
  return nullptr;
}

bool equal_metadata(const amdhsa::MetadataValue &first, const amdhsa::MetadataValue &second) {
  if (first.kind != second.kind || first.integer != second.integer ||
      first.string != second.string || first.array.size() != second.array.size() ||
      first.map.size() != second.map.size()) {
    return false;
  }
  for (size_t index = 0; index < first.array.size(); ++index) {
    if (!equal_metadata(first.array[index], second.array[index])) {
      return false;
    }
  }
  for (const std::pair<const std::string_view, amdhsa::MetadataValue> &field : first.map) {
    const amdhsa::MetadataValue *other = second.field(field.first);
    if (other == nullptr || !equal_metadata(field.second, *other)) {
      return false;
    }
  }
  return true;
}

Result verify_descriptor_symbols(const ImageFacts &facts, std::span<const uint8_t> image,
                                 std::span<const Elf64_Shdr> sections, bool prepared,
                                 util::DiagnosticEmitter diagnostic) {
  // Descriptor discovery deduplicates .symtab/.dynsym by file offset. Check the
  // symbols too so another .kd name cannot silently alias an admitted binding.
  for (const Elf64_Shdr &section : sections) {
    if (section.sh_type != SHT_SYMTAB && section.sh_type != SHT_DYNSYM) {
      continue;
    }
    if (section.sh_entsize != sizeof(Elf64_Sym) || section.sh_size % sizeof(Elf64_Sym) != 0 ||
        section.sh_link >= sections.size()) {
      return diagnostic.emit() << "invalid artifact symbol table";
    }
    const std::optional<std::vector<Elf64_Sym>> symbols = read_elf_table<Elf64_Sym>(
        image, section.sh_offset, section.sh_size / sizeof(Elf64_Sym), section.sh_entsize);
    const Elf64_Shdr &strings = sections[section.sh_link];
    if (!symbols || !util::is_range_in_bounds(strings.sh_offset, strings.sh_size, image.size())) {
      return diagnostic.emit() << "invalid artifact symbol strings";
    }
    for (const Elf64_Sym &symbol : *symbols) {
      if (symbol.st_shndx == 0) {
        continue;
      }
      if (symbol.st_name >= strings.sh_size) {
        return diagnostic.emit() << "invalid artifact symbol name";
      }
      const char *begin =
          reinterpret_cast<const char *>(image.data() + strings.sh_offset + symbol.st_name);
      const size_t available = strings.sh_size - symbol.st_name;
      const void *end = std::memchr(begin, 0, available);
      if (end == nullptr) {
        return diagnostic.emit() << "unterminated artifact symbol name";
      }
      const std::string_view name(begin, static_cast<const char *>(end) - begin);
      if (!name.ends_with(".kd")) {
        continue;
      }
      if (elf_symbol_type(symbol.st_info) != kElfSymbolTypeObject ||
          elf_symbol_bind(symbol.st_info) != kElfSymbolBindGlobal || symbol.st_size != 64 ||
          symbol.st_shndx >= sections.size()) {
        return diagnostic.emit() << "unsupported descriptor symbol or alias";
      }
      const Elf64_Shdr &owner = sections[symbol.st_shndx];
      if (symbol.st_value < owner.sh_addr ||
          !util::is_range_in_bounds(owner.sh_offset, owner.sh_size, image.size()) ||
          !util::is_range_in_bounds(symbol.st_value - owner.sh_addr, 64, owner.sh_size)) {
        return diagnostic.emit() << "descriptor symbol is outside its section";
      }
      const uint64_t offset = owner.sh_offset + symbol.st_value - owner.sh_addr;
      const std::vector<ImageEntryFacts>::const_iterator binding =
          std::ranges::find_if(facts.entries, [&](const ImageEntryFacts &entry) {
            return name == entry.binding.name + ".kd";
          });
      if (binding == facts.entries.end() ||
          offset != (prepared ? binding->binding.prepared_descriptor_offset : binding->key)) {
        return diagnostic.emit() << "unknown descriptor symbol or alias";
      }
    }
  }
  return Result::success();
}

FailureOr<InspectedImage> inspect_image(const ImageFacts &facts, std::span<const uint8_t> image,
                                        bool prepared, util::DiagnosticEmitter diagnostic) {
  const TargetProfile *target = target_profile(facts.target);
  if (target == nullptr) {
    return diagnostic.emit() << "unsupported artifact target";
  }
  Elf64_Ehdr header = {};
  if (image.size() < sizeof(header)) {
    return diagnostic.emit() << "artifact has truncated ELF header";
  }
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0 ||
      header.e_ident[EI_CLASS] != ELFCLASS64 || header.e_ident[EI_DATA] != ELFDATA2LSB ||
      header.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA ||
      header.e_ident[EI_ABIVERSION] != ELFABIVERSION_AMDGPU_HSA_V5 || header.e_type != ET_DYN ||
      header.e_machine != EM_AMDGPU || (header.e_flags & EF_AMDGPU_MACH) != target->elf_machine ||
      (header.e_flags & ~uint32_t{0xfff}) != 0 || header.e_shentsize != sizeof(Elf64_Shdr) ||
      header.e_phentsize != sizeof(Elf64_Phdr) || header.e_shstrndx >= header.e_shnum) {
    return diagnostic.emit() << "artifact ELF target, ABI or tables are inconsistent";
  }
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  const std::optional<std::vector<Elf64_Phdr>> segments =
      read_elf_table<Elf64_Phdr>(image, header.e_phoff, header.e_phnum, header.e_phentsize);
  if (!sections || !segments) {
    return diagnostic.emit() << "artifact ELF target, ABI or tables are inconsistent";
  }
  if (verify_descriptor_symbols(facts, image, *sections, prepared, diagnostic).failed()) {
    return Result::failure();
  }
  const Elf64_Shdr &strings = (*sections)[header.e_shstrndx];
  if (!util::is_range_in_bounds(strings.sh_offset, strings.sh_size, image.size())) {
    return diagnostic.emit() << "artifact section names are truncated";
  }
  std::optional<Elf64_Shdr> text;
  size_t text_index = 0;
  for (size_t index = 0; index < header.e_shnum; ++index) {
    const Elf64_Shdr &section = (*sections)[index];
    if ((section.sh_flags & (SHF_EXECINSTR | SHF_ALLOC)) != (SHF_EXECINSTR | SHF_ALLOC)) {
      continue;
    }
    if (text || section.sh_name >= strings.sh_size || section.sh_type != SHT_PROGBITS ||
        !util::is_range_in_bounds(section.sh_offset, section.sh_size, image.size()) ||
        section.sh_size > UINT64_MAX - section.sh_addr) {
      return diagnostic.emit() << "artifact executable section is ambiguous";
    }
    const std::span<const uint8_t> name =
        image.subspan(strings.sh_offset + section.sh_name, strings.sh_size - section.sh_name);
    if (name.size() < 6 || std::memcmp(name.data(), ".text", 6) != 0) {
      return diagnostic.emit() << "artifact executable section is not .text";
    }
    text = section;
    text_index = index;
  }
  if (!text ||
      text->sh_size != (prepared ? facts.prepared_text_bytes : facts.original_text_bytes)) {
    return diagnostic.emit() << "artifact text extent differs from facts";
  }
  bool loaded_text = false;
  for (const Elf64_Phdr &segment : *segments) {
    if (segment.p_type != PT_LOAD) {
      continue;
    }
    if (!util::is_range_in_bounds(segment.p_offset, segment.p_filesz, image.size()) ||
        segment.p_filesz > segment.p_memsz || segment.p_memsz > UINT64_MAX - segment.p_vaddr) {
      return diagnostic.emit() << "artifact load segment has invalid extent";
    }
    loaded_text |= text->sh_offset >= segment.p_offset &&
                   util::is_range_in_bounds(text->sh_offset, text->sh_size,
                                            segment.p_offset + segment.p_filesz) &&
                   (segment.p_flags & PF_X) != 0 &&
                   segment.p_vaddr + (text->sh_offset - segment.p_offset) == text->sh_addr;
  }
  if (!loaded_text) {
    return diagnostic.emit()
           << "artifact executable .text is not fully file-backed at its declared address";
  }
  InspectedImage inspection;
  inspection.text = *text;
  inspection.kernels = scan_kernel_descriptors(image, text->sh_offset, text->sh_size, text_index);
  if (inspection.kernels.size() != facts.entries.size()) {
    return diagnostic.emit() << "artifact entry inventory does not match facts";
  }
  FailureOr<amdhsa::MetadataDocument> document = amdhsa::MetadataDocument::parse(image);
  if (document.failed()) {
    return diagnostic.emit() << "artifact metadata cannot be parsed";
  }
  const amdhsa::MetadataValue *metadata_kernels = document.value().root().field("amdhsa.kernels");
  if (metadata_kernels == nullptr || metadata_kernels->kind != amdhsa::MetadataValue::Kind::Array ||
      metadata_kernels->array.size() != facts.entries.size()) {
    return diagnostic.emit() << "artifact metadata inventory differs from facts";
  }
  inspection.metadata = std::move(document.value());
  for (const ImageEntryFacts &row : facts.entries) {
    const EntryFacts &entry = row.binding;
    const uint64_t descriptor_offset = prepared ? entry.prepared_descriptor_offset : row.key;
    const uint64_t entry_offset =
        prepared ? entry.prepared_entry_offset : entry.original_entry_offset;
    const std::vector<KernelDescriptorInfo>::const_iterator found = std::ranges::find(
        inspection.kernels, descriptor_offset, &KernelDescriptorInfo::descriptor_file_offset);
    if (found == inspection.kernels.end() || found->kernel_name != entry.name) {
      return diagnostic.emit() << "artifact entry/symbol does not match facts";
    }
    const KernelDescriptorInfo &kernel = *found;
    if (row.selected && row.selected->scope == ScopeKind::ExecutionTrace &&
        RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc2,
                           amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X) == 0)
      return diagnostic.emit() << "trace descriptor lacks its original workgroup-X input";
    uint32_t kernarg = entry.original_kernarg_bytes;
    if (prepared && row.disposition == EntryDisposition::Selected) {
      const std::optional<KernargExtensionLayout> wrapper =
          make_kernarg_extension_layout(entry.original_kernarg_bytes, kContextPointerPayloads);
      if (!wrapper) {
        return diagnostic.emit() << "artifact kernarg wrapper overflows";
      }
      kernarg = wrapper->wrapper_size;
    }
    if (kernel.entry_text_offset != entry_offset || entry_offset > UINT64_MAX - text->sh_addr ||
        ((text->sh_addr + entry_offset) % 256) != 0 ||
        kernel_wavefront_size(target->arch, kernel.descriptor) != facts.wave_size ||
        kernel.descriptor.kernarg_size != kernarg ||
        kernel.descriptor.private_segment_fixed_size !=
            (prepared ? entry.prepared_private_bytes : entry.original_private_bytes) ||
        kernel.descriptor.group_segment_fixed_size !=
            (prepared ? entry.prepared_group_bytes : entry.original_group_bytes) ||
        (target->id == 1100 && facts.wave_size == 32 &&
         kernel_descriptor_initial_sgpr_count(target->arch, kernel.descriptor) < 16)) {
      return diagnostic.emit() << "artifact descriptor differs from its binding facts";
    }
    const uint32_t workgroup_id_scalar =
        RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc2,
                           amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
            ? (isa_properties(target->arch).uses_ttmp_workgroup_ids
                   ? kWorkgroupIdTtmp9Selector
                   : kernel_descriptor_user_sgpr_count(target->arch, kernel.descriptor))
            : UINT32_MAX;
    if (workgroup_id_scalar != entry.workgroup_id_scalar) {
      return diagnostic.emit() << "artifact workgroup-X input differs from its binding facts";
    }
    bool loaded_descriptor = false;
    for (const Elf64_Phdr &segment : *segments) {
      if (segment.p_type != PT_LOAD || descriptor_offset < segment.p_offset ||
          !util::is_range_in_bounds(descriptor_offset, 64, segment.p_offset + segment.p_filesz)) {
        continue;
      }
      const uint64_t descriptor_vaddr = segment.p_vaddr + (descriptor_offset - segment.p_offset);
      const std::optional<uint64_t> loaded_entry =
          add_signed_offset(descriptor_vaddr, kernel.descriptor.kernel_code_entry_byte_offset);
      loaded_descriptor |= loaded_entry && *loaded_entry == text->sh_addr + entry_offset;
    }
    if (!loaded_descriptor) {
      return diagnostic.emit() << "artifact entry/descriptor is not loaded";
    }
    std::string metadata_error;
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(image, kernel, &metadata_error,
                         row.disposition == EntryDisposition::Native ? MetadataPolicy::Native
                                                                     : MetadataPolicy::Selected);
    if (metadata.failed()) {
      return diagnostic.emit() << metadata_error;
    }
    const uint32_t expected_sgprs = prepared ? entry.prepared_sgprs : row.original_sgprs;
    const uint32_t expected_vgprs = prepared ? entry.prepared_vgprs : row.original_vgprs;
    if (metadata.value().maximum_workgroup_size != entry.maximum_workgroup_size) {
      return diagnostic.emit() << "kernel launch bound differs from facts";
    }
    if (metadata.value().required_workgroup_size != entry.required_workgroup_size ||
        metadata.value().kernarg_alignment != entry.kernarg_alignment) {
      return diagnostic.emit() << "kernel argument alignment or required shape differs from facts";
    }
    if (metadata.value().sgprs != expected_sgprs || metadata.value().vgprs != expected_vgprs) {
      return diagnostic.emit() << "artifact metadata resources differ from facts";
    }
    if (prepared && row.disposition == EntryDisposition::Selected &&
        (target->arch == IsaFamily::Cdna3 || target->arch == IsaFamily::Cdna4)) {
      const uint32_t allocated_sgprs =
          (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc1,
                              amdhsa::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT) +
           1) *
          8;
      if (ordinary_sgpr_bound(target->arch, allocated_sgprs) < entry.prepared_sgprs) {
        return diagnostic.emit()
               << "prepared descriptor does not allocate its declared ordinary SGPRs";
      }
    }
  }
  return inspection;
}

FailureOr<std::vector<uint8_t>> read(const std::filesystem::path &path, uint64_t maximum,
                                     util::DiagnosticEmitter diagnostic) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file || file.tellg() < 0 || static_cast<uint64_t>(file.tellg()) > maximum) {
    return diagnostic.emit() << "missing or oversized artifact file: " + path.string();
  }
  std::vector<uint8_t> bytes(static_cast<size_t>(file.tellg()));
  file.seekg(0);
  file.read(reinterpret_cast<char *>(bytes.data()), bytes.size());
  if (!file) {
    return diagnostic.emit() << "short artifact file read: " + path.string();
  }
  return bytes;
}
} // namespace

FailureOr<LoadedNativeImage> load_native_image(const std::filesystem::path &path,
                                               util::DiagnosticEmitter diagnostic) {
  FailureOr<std::vector<uint8_t>> original = read(path, ContractLimits{}.image_bytes, diagnostic);
  if (original.failed()) {
    return Result::failure();
  }
  if (original.value().size() < sizeof(Elf64_Ehdr)) {
    return diagnostic.emit() << "native image has a truncated header";
  }
  Elf64_Ehdr header = {};
  std::memcpy(&header, original.value().data(), sizeof(header));
  const std::string_view target = elf_mach_name(header.e_flags);
  FailureOr<NativeImageInventory> inventory =
      inspect_native_image(original.value(), target, diagnostic);
  if (inventory.failed()) {
    return Result::failure();
  }
  return LoadedNativeImage{.inventory = std::move(inventory.value()),
                           .original = std::move(original.value())};
}

FailureOr<NativeImageInventory> inspect_native_image(std::span<const uint8_t> original,
                                                     std::string_view concrete_target,
                                                     util::DiagnosticEmitter diagnostic) {
  const TargetProfile *target = target_profile(concrete_target);
  if (target == nullptr || original.size() < sizeof(Elf64_Ehdr) ||
      original.size() > ContractLimits{}.image_bytes) {
    return diagnostic.emit() << "invalid native image target or extent";
  }
  Elf64_Ehdr header = {};
  std::memcpy(&header, original.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(original, header.e_shoff, header.e_shnum, header.e_shentsize);
  if (!sections) {
    return diagnostic.emit() << "invalid native image section table";
  }
  if (header.e_shstrndx != SHN_UNDEF) {
    if (header.e_shstrndx >= sections->size()) {
      return diagnostic.emit() << "invalid native image section-name table";
    }
    const Elf64_Shdr &strings = (*sections)[header.e_shstrndx];
    if (strings.sh_type != SHT_STRTAB ||
        !util::is_range_in_bounds(strings.sh_offset, strings.sh_size, original.size())) {
      return diagnostic.emit() << "invalid native image section-name strings";
    }
    for (const Elf64_Shdr &section : *sections) {
      if (section.sh_name >= strings.sh_size) {
        return diagnostic.emit() << "invalid native image section name";
      }
      const char *begin =
          reinterpret_cast<const char *>(original.data() + strings.sh_offset + section.sh_name);
      const size_t available = strings.sh_size - section.sh_name;
      const void *end = std::memchr(begin, 0, available);
      if (end == nullptr) {
        return diagnostic.emit() << "unterminated native image section name";
      }
      if (std::string_view(begin, static_cast<const char *>(end) - begin) == ".rocjitsu.kernarg") {
        return diagnostic.emit()
               << "native image carries a reserved prepared kernarg wrapper section";
      }
    }
  }
  std::optional<Elf64_Shdr> text;
  size_t text_index = 0;
  for (size_t index = 0; index < sections->size(); ++index) {
    const Elf64_Shdr &section = (*sections)[index];
    if ((section.sh_flags & (SHF_ALLOC | SHF_EXECINSTR)) != (SHF_ALLOC | SHF_EXECINSTR)) {
      continue;
    }
    if (text || !util::is_range_in_bounds(section.sh_offset, section.sh_size, original.size())) {
      return diagnostic.emit() << "native executable section is ambiguous or truncated";
    }
    text = section;
    text_index = index;
  }
  if (!text) {
    return diagnostic.emit() << "native image has no executable text";
  }
  const std::vector<KernelDescriptorInfo> kernels =
      scan_kernel_descriptors(original, text->sh_offset, text->sh_size, text_index);
  if (kernels.empty() || kernels.size() > kImageEntryLimit) {
    return diagnostic.emit() << "native entry inventory is empty or oversized";
  }
  ImageFacts bindings;
  bindings.original = {.sha256 = digest(original), .bytes = original.size()};
  bindings.prepared = bindings.original;
  bindings.target = target->id;
  bindings.wave_size = kernel_wavefront_size(target->arch, kernels.front().descriptor);
  bindings.code_object_abi = 5;
  bindings.original_text_bytes = text->sh_size;
  bindings.prepared_text_bytes = text->sh_size;
  for (const KernelDescriptorInfo &kernel : kernels) {
    std::string message;
    const FailureOr<MetadataResources> metadata =
        inspect_metadata(original, kernel, &message, MetadataPolicy::Native);
    if (metadata.failed()) {
      return diagnostic.emit() << message;
    }
    ImageEntryFacts entry;
    entry.key = kernel.descriptor_file_offset;
    entry.original_sgprs = metadata.value().sgprs;
    entry.original_vgprs = metadata.value().vgprs;
    entry.binding = {
        .name = kernel.kernel_name,
        .original_descriptor_offset = entry.key,
        .prepared_descriptor_offset = entry.key,
        .original_entry_offset = kernel.entry_text_offset,
        .prepared_entry_offset = kernel.entry_text_offset,
        .original_kernarg_bytes = kernel.descriptor.kernarg_size,
        .kernarg_alignment = metadata.value().kernarg_alignment,
        .original_group_bytes = kernel.descriptor.group_segment_fixed_size,
        .prepared_group_bytes = kernel.descriptor.group_segment_fixed_size,
        .original_private_bytes = kernel.descriptor.private_segment_fixed_size,
        .prepared_private_bytes = kernel.descriptor.private_segment_fixed_size,
        .prepared_sgprs = metadata.value().sgprs,
        .prepared_vgprs = metadata.value().vgprs,
        .maximum_workgroup_size = metadata.value().maximum_workgroup_size,
        .workgroup_id_scalar =
            RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc2,
                               amdhsa::COMPUTE_PGM_RSRC2_ENABLE_SGPR_WORKGROUP_ID_X)
                ? (isa_properties(target->arch).uses_ttmp_workgroup_ids
                       ? kWorkgroupIdTtmp9Selector
                       : kernel_descriptor_user_sgpr_count(target->arch, kernel.descriptor))
                : UINT32_MAX,
        .required_workgroup_size = metadata.value().required_workgroup_size};
    bindings.entries.push_back(std::move(entry));
  }
  std::ranges::sort(bindings.entries, {}, &ImageEntryFacts::key);
  const FailureOr<InspectedImage> checked = inspect_image(bindings, original, false, diagnostic);
  if (checked.failed()) {
    return Result::failure();
  }
  return NativeImageInventory{.identity = bindings.original,
                              .target = bindings.target,
                              .wave_size = bindings.wave_size,
                              .entries = std::move(bindings.entries)};
}

Result verify_artifact_images(const ImageFacts &facts, std::span<const uint8_t> original,
                              std::span<const uint8_t> prepared,
                              util::DiagnosticEmitter diagnostic) {
  if (match_image_facts(facts, original, prepared, facts.selection_profile, diagnostic).failed()) {
    return Result::failure();
  }
  const FailureOr<InspectedImage> original_image =
      inspect_image(facts, original, false, diagnostic);
  const FailureOr<InspectedImage> prepared_image = inspect_image(facts, prepared, true, diagnostic);
  if (original_image.failed() || prepared_image.failed()) {
    return Result::failure();
  }
  for (const ImageEntryFacts &entry : facts.entries) {
    if (entry.disposition != EntryDisposition::Native) {
      continue;
    }
    const EntryFacts &binding = entry.binding;
    const std::vector<KernelDescriptorInfo>::const_iterator original_kernel = std::ranges::find(
        original_image.value().kernels, entry.key, &KernelDescriptorInfo::descriptor_file_offset);
    const std::vector<KernelDescriptorInfo>::const_iterator prepared_kernel =
        std::ranges::find(prepared_image.value().kernels, binding.prepared_descriptor_offset,
                          &KernelDescriptorInfo::descriptor_file_offset);
    amdhsa::kernel_descriptor_t first = original_kernel->descriptor;
    amdhsa::kernel_descriptor_t second = prepared_kernel->descriptor;
    first.kernel_code_entry_byte_offset = 0;
    second.kernel_code_entry_byte_offset = 0;
    if (std::memcmp(&first, &second, sizeof(first)) != 0) {
      return diagnostic.emit() << "native descriptor changed beyond entry relocation";
    }
    for (const NativeCodeRange &range : entry.native_code) {
      if (!std::ranges::equal(
              original.subspan(original_image.value().text.sh_offset + range.original_offset,
                               range.bytes),
              prepared.subspan(prepared_image.value().text.sh_offset + range.prepared_offset,
                               range.bytes))) {
        return diagnostic.emit() << "native entry body changed";
      }
    }
    if (entry.no_probes_needed) {
      util::Sha256 hash;
      hash.update("consan-no-lds-code-v1");
      for (const NativeCodeRange &range : entry.native_code) {
        std::array<uint8_t, 16> coordinates{};
        wire::write_u64(coordinates, 0, range.original_offset);
        wire::write_u64(coordinates, 8, range.bytes);
        hash.update(coordinates.data(), coordinates.size());
        const std::span<const uint8_t> bytes = original.subspan(
            original_image.value().text.sh_offset + range.original_offset, range.bytes);
        hash.update(bytes.data(), bytes.size());
      }
      if (hash.finish() != entry.no_probes_needed->code_sha256)
        return diagnostic.emit() << "no-probes-needed certificate differs from original code";
    }
    const amdhsa::MetadataValue *first_metadata =
        metadata_entry(original_image.value().metadata, binding.name);
    const amdhsa::MetadataValue *second_metadata =
        metadata_entry(prepared_image.value().metadata, binding.name);
    if (first_metadata == nullptr || second_metadata == nullptr ||
        !equal_metadata(*first_metadata, *second_metadata)) {
      return diagnostic.emit() << "native metadata changed";
    }
  }
  return Result::success();
}

Result verify_artifact_images(const Manifest &manifest, std::span<const uint8_t> original,
                              std::span<const uint8_t> prepared,
                              util::DiagnosticEmitter diagnostic) {
  if (match_images(manifest, original, prepared, diagnostic).failed()) {
    return Result::failure();
  }
  KernelDescriptorInfo descriptor;
  descriptor.kernel_name = manifest.entry.name;
  descriptor.descriptor_file_offset = manifest.entry.original_descriptor_offset;
  descriptor.entry_text_offset = manifest.entry.original_entry_offset;
  std::memcpy(&descriptor.descriptor, original.data() + descriptor.descriptor_file_offset,
              sizeof(descriptor.descriptor));
  const FailureOr<MetadataResources> metadata = inspect_metadata(original, descriptor);
  if (metadata.failed()) {
    return diagnostic.emit() << "unsupported original metadata";
  }
  return verify_artifact_images(
      single_entry_image_facts(manifest, metadata.value().sgprs, metadata.value().vgprs), original,
      prepared, diagnostic);
}

FailureOr<LoadedArtifact> load_artifact(const std::filesystem::path &directory,
                                        std::span<const uint8_t> original,
                                        std::string_view concrete_target,
                                        const Digest &expected_selection_profile,
                                        util::DiagnosticEmitter diagnostic) {
  const TargetProfile *target = target_profile(concrete_target);
  if (target == nullptr) {
    return diagnostic.emit() << "ConSan artifact requires a supported concrete GPU agent";
  }
  const FailureOr<std::vector<uint8_t>> ready =
      read(directory / "ready", kProducerContract.size(), diagnostic);
  if (ready.failed()) {
    return Result::failure();
  }
  if (!std::ranges::equal(ready.value(), kProducerContract)) {
    return diagnostic.emit() << "artifact ready marker has wrong contract";
  }
  const ContractLimits limits;
  const FailureOr<std::vector<uint8_t>> bytes =
      read(directory / "facts.bin", limits.metadata_bytes, diagnostic);
  if (bytes.failed()) {
    return Result::failure();
  }
  FailureOr<ImageFacts> facts = decode_image_facts(bytes.value(), diagnostic);
  if (facts.failed()) {
    return Result::failure();
  }
  if (facts.value().target != target->id) {
    return diagnostic.emit() << "concrete agent differs from artifact target";
  }
  if (facts.value().selection_profile != expected_selection_profile) {
    return diagnostic.emit() << "configured selection profile differs from artifact";
  }
  FailureOr<std::vector<uint8_t>> image =
      read(directory / "prepared.hsaco", limits.image_bytes, diagnostic);
  if (image.failed() ||
      verify_artifact_images(facts.value(), original, image.value(), diagnostic).failed()) {
    return Result::failure();
  }
  return LoadedArtifact{.facts = std::move(facts.value()), .prepared = std::move(image.value())};
}
} // namespace rocjitsu::consan
