// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file amdgpu_code_object_test.cpp
/// @brief Unit tests for AmdGpuCodeObject queries, ownership admission, and
///        payload-level kernel metadata parsing.

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/amdgpu_kernel_metadata.h"
#include "rocjitsu/code/rj_code.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include "hsa/AMDHSAKernelDescriptor.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace rocjitsu {
namespace {

namespace kd = rocr::llvm::amdhsa;
using KD = kd::kernel_descriptor_t;

constexpr uint16_t kKernelMetadataFixtureSectionCount = 2;
constexpr uint16_t kDerivedStateFixtureSectionCount = 6;

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

void append_bytes(std::vector<uint8_t> &bytes, const void *source, size_t size) {
  const auto *begin = static_cast<const uint8_t *>(source);
  bytes.insert(bytes.end(), begin, begin + size);
}

void append_msgpack_string(std::vector<uint8_t> &bytes, std::string_view value) {
  if (value.size() > 31) {
    ADD_FAILURE() << "metadata fixture string is too large for a msgpack fixstr";
    return;
  }
  bytes.push_back(static_cast<uint8_t>(0xa0u | value.size()));
  append_bytes(bytes, value.data(), value.size());
}

void append_msgpack_unsigned(std::vector<uint8_t> &bytes, uint64_t value) {
  if (value <= 0x7fu) {
    bytes.push_back(static_cast<uint8_t>(value));
    return;
  }
  const unsigned byte_count = value <= std::numeric_limits<uint8_t>::max()    ? 1u
                              : value <= std::numeric_limits<uint16_t>::max() ? 2u
                              : value <= std::numeric_limits<uint32_t>::max() ? 4u
                                                                              : 8u;
  bytes.push_back(byte_count == 1u   ? 0xccu
                  : byte_count == 2u ? 0xcdu
                  : byte_count == 4u ? 0xceu
                                     : 0xcfu);
  for (int shift = static_cast<int>((byte_count - 1u) * 8u); shift >= 0; shift -= 8)
    bytes.push_back(static_cast<uint8_t>((value >> shift) & 0xffu));
}

std::vector<uint8_t>
make_single_kernel_metadata_field_payload(std::string_view field,
                                          std::span<const uint8_t> encoded_value) {
  std::vector<uint8_t> payload;
  payload.push_back(0x81u); // one-entry root map
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u); // one kernel
  payload.push_back(0x82u); // name plus the requested field
  append_msgpack_string(payload, ".name");
  append_msgpack_string(payload, "k");
  append_msgpack_string(payload, field);
  payload.insert(payload.end(), encoded_value.begin(), encoded_value.end());
  return payload;
}

std::vector<uint8_t> make_complete_kernel_metadata_payload() {
  std::vector<uint8_t> payload;
  payload.push_back(0x81u); // one-entry root map
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u); // one kernel
  payload.push_back(0x85u); // name plus four retained metadata fields
  append_msgpack_string(payload, ".name");
  append_msgpack_string(payload, "kernel");
  append_msgpack_string(payload, ".args");
  payload.push_back(0x91u); // one argument
  payload.push_back(0x81u); // one argument field
  append_msgpack_string(payload, ".value_kind");
  append_msgpack_string(payload, "hidden_dynamic_lds_size");
  append_msgpack_string(payload, ".uses_dynamic_stack");
  payload.push_back(0xc3u);
  append_msgpack_string(payload, ".sgpr_count");
  append_msgpack_unsigned(payload, 65535u);
  append_msgpack_string(payload, ".reqd_workgroup_size");
  payload.push_back(0x93u);
  append_msgpack_unsigned(payload, 1u);
  append_msgpack_unsigned(payload, 256u);
  append_msgpack_unsigned(payload, std::numeric_limits<uint32_t>::max());
  return payload;
}

bool accept_kernel_metadata(std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
  return true;
}

struct MutableKernelMetadataVisitor {
  size_t visit_count = 0;

  bool operator()(std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
    ++visit_count;
    return true;
  }
};

void append_kernel_metadata_payload_header(std::vector<uint8_t> &payload, size_t kernel_count) {
  if (kernel_count > std::numeric_limits<uint32_t>::max()) {
    ADD_FAILURE() << "metadata fixture kernel count is outside the msgpack array range";
    return;
  }
  payload.push_back(0x81u); // one-entry root map
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0xddu); // array32
  const uint32_t count = static_cast<uint32_t>(kernel_count);
  for (int shift = 24; shift >= 0; shift -= 8)
    payload.push_back(static_cast<uint8_t>((count >> shift) & 0xffu));
}

bool append_kernel_metadata_record(std::vector<uint8_t> &payload, char name, uint8_t sgpr_count) {
  if (sgpr_count >= 0x80u) {
    ADD_FAILURE() << "metadata fixture sgpr_count is outside the msgpack fixint range";
    return false;
  }
  payload.push_back(0x82u); // .name plus one retained metadata field
  append_msgpack_string(payload, ".name");
  append_msgpack_string(payload, std::string_view(&name, 1));
  append_msgpack_string(payload, ".sgpr_count");
  payload.push_back(sgpr_count);
  return true;
}

std::vector<uint8_t> make_kernel_metadata_payload(size_t kernel_count) {
  std::vector<uint8_t> payload;
  append_kernel_metadata_payload_header(payload, kernel_count);
  if (payload.empty())
    return {};
  for (size_t index = 0; index < kernel_count; ++index) {
    const char name = static_cast<char>(1u + index % 255u);
    if (!append_kernel_metadata_record(payload, name, 8u))
      return {};
  }
  return payload;
}

std::vector<uint8_t>
make_named_kernel_metadata_payload(const std::vector<std::pair<char, uint8_t>> &kernels) {
  std::vector<uint8_t> payload;
  append_kernel_metadata_payload_header(payload, kernels.size());
  if (payload.empty())
    return {};
  for (const auto &[name, sgpr_count] : kernels) {
    if (!append_kernel_metadata_record(payload, name, sgpr_count))
      return {};
  }
  return payload;
}

std::vector<uint8_t>
make_metadata_payload_with_unknown_root_value(std::span<const uint8_t> encoded_value,
                                              bool append_kernel_array = true) {
  std::vector<uint8_t> payload;
  payload.push_back(append_kernel_array ? 0x82u : 0x81u);
  append_msgpack_string(payload, "future.root.field");
  payload.insert(payload.end(), encoded_value.begin(), encoded_value.end());
  if (!append_kernel_array)
    return payload;
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u); // one kernel
  if (!append_kernel_metadata_record(payload, 'k', 8u))
    return {};
  return payload;
}

std::vector<std::pair<char, uint8_t>> make_distinct_kernel_metadata_records(size_t count) {
  if (count > 255) {
    ADD_FAILURE() << "metadata fixture has too many distinct one-byte names";
    return {};
  }
  std::vector<std::pair<char, uint8_t>> kernels;
  kernels.reserve(count);
  for (size_t index = 0; index < count; ++index)
    kernels.emplace_back(static_cast<char>(index + 1), 8u);
  return kernels;
}

std::vector<uint8_t> make_kernel_metadata_note_from_payload(const std::vector<uint8_t> &payload) {
  if (payload.empty())
    return {};

  std::vector<uint8_t> note;
  Elf64_Nhdr header{};
  header.n_namesz = 7;
  header.n_descsz = static_cast<uint32_t>(payload.size());
  header.n_type = NT_AMDGPU_METADATA;
  append_bytes(note, &header, sizeof(header));
  constexpr std::array<char, 8> kNoteName{'A', 'M', 'D', 'G', 'P', 'U', '\0', '\0'};
  append_bytes(note, kNoteName.data(), kNoteName.size());
  append_bytes(note, payload.data(), payload.size());
  note.resize(align_up(note.size(), 4), 0);
  return note;
}

std::vector<uint8_t> make_kernel_metadata_note(size_t kernel_count) {
  return make_kernel_metadata_note_from_payload(make_kernel_metadata_payload(kernel_count));
}

bool install_kernel_metadata_notes(std::vector<uint8_t> &image,
                                   const std::vector<std::vector<uint8_t>> &notes,
                                   uint64_t note_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Phdr),
                                   size_t program_header_count = 1) {
  if (image.size() < sizeof(Elf64_Ehdr))
    return false;
  const uint64_t program_header_offset = sizeof(Elf64_Ehdr);
  if (program_header_count == 0 || program_header_count > std::numeric_limits<uint16_t>::max()) {
    return false;
  }
  const uint64_t program_header_bytes = program_header_count * sizeof(Elf64_Phdr);
  uint64_t note_bytes = 0;
  for (const auto &note : notes) {
    if (note.empty() || note.size() > std::numeric_limits<uint64_t>::max() - note_bytes)
      return false;
    note_bytes += note.size();
  }
  if (notes.empty() || note_offset < program_header_offset + program_header_bytes ||
      note_offset > image.size() || note_bytes > image.size() - note_offset) {
    return false;
  }
  const auto region_is_zero = [&](uint64_t offset, uint64_t size) {
    return std::all_of(image.begin() + static_cast<ptrdiff_t>(offset),
                       image.begin() + static_cast<ptrdiff_t>(offset + size),
                       [](uint8_t byte) { return byte == 0; });
  };
  if (!region_is_zero(program_header_offset, program_header_bytes) ||
      !region_is_zero(note_offset, note_bytes)) {
    return false;
  }

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_phoff = program_header_offset;
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = static_cast<uint16_t>(program_header_count);
  std::memcpy(image.data(), &header, sizeof(header));

  Elf64_Phdr program_header{};
  program_header.p_type = PT_NOTE;
  program_header.p_offset = note_offset;
  program_header.p_filesz = note_bytes;
  for (size_t index = 0; index < program_header_count; ++index) {
    std::memcpy(image.data() + program_header_offset + index * sizeof(Elf64_Phdr), &program_header,
                sizeof(program_header));
  }
  uint64_t cursor = note_offset;
  for (const auto &note : notes) {
    std::memcpy(image.data() + cursor, note.data(), note.size());
    cursor += note.size();
  }
  return true;
}

bool install_kernel_metadata_note(std::vector<uint8_t> &image, size_t kernel_count,
                                  uint64_t note_offset = sizeof(Elf64_Ehdr) + sizeof(Elf64_Phdr)) {
  return install_kernel_metadata_notes(image, {make_kernel_metadata_note(kernel_count)},
                                       note_offset);
}

std::vector<uint8_t>
make_elf_with_kernel_metadata_payloads(const std::vector<std::vector<uint8_t>> &payloads,
                                       size_t fixed_image_size = 0,
                                       size_t program_header_count = 1) {
  std::vector<std::vector<uint8_t>> notes;
  uint64_t note_bytes = 0;
  for (const auto &payload : payloads) {
    auto note = make_kernel_metadata_note_from_payload(payload);
    if (note.empty() || note.size() > std::numeric_limits<uint64_t>::max() - note_bytes)
      return {};
    note_bytes += note.size();
    notes.push_back(std::move(note));
  }
  if (notes.empty() || program_header_count == 0 ||
      program_header_count > std::numeric_limits<uint16_t>::max()) {
    return {};
  }

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  const uint64_t note_offset = sizeof(Elf64_Ehdr) + program_header_count * sizeof(Elf64_Phdr);
  const uint64_t shstrtab_offset = note_offset + note_bytes;
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  std::vector<uint8_t> image(shoff + kKernelMetadataFixtureSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_shoff = shoff;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kKernelMetadataFixtureSectionCount;
  header.e_shstrndx = 1;
  std::memcpy(image.data(), &header, sizeof(header));
  if (!install_kernel_metadata_notes(image, notes, note_offset, program_header_count)) {
    ADD_FAILURE() << "metadata fixture note does not fit in its ELF image";
    return {};
  }

  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());
  std::array<Elf64_Shdr, kKernelMetadataFixtureSectionCount> sections{};
  sections[1].sh_name = shstrtab_name;
  sections[1].sh_type = SHT_STRTAB;
  sections[1].sh_offset = shstrtab_offset;
  sections[1].sh_size = shstrtab.size();
  sections[1].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sizeof(sections));
  if (fixed_image_size != 0) {
    if (image.size() > fixed_image_size) {
      ADD_FAILURE() << "metadata fixture does not fit in its fixed-size image";
      return {};
    }
    image.resize(fixed_image_size, 0);
  }
  return image;
}

std::vector<uint8_t> make_elf_with_kernel_metadata_note(size_t kernel_count,
                                                        size_t fixed_image_size = 0) {
  return make_elf_with_kernel_metadata_payloads({make_kernel_metadata_payload(kernel_count)},
                                                fixed_image_size);
}

// A 64-byte kernel descriptor whose wavefront SGPR granulation field is
// `granulated`; everything else zero.
KD make_kd(uint32_t granulated) {
  KD desc{};
  AMDHSA_BITS_SET(desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT,
                  granulated);
  return desc;
}

// A minimal gfx950 code object exporting one `<name>.kd` object symbol per entry
// in `kernels`, each pointing at a kernel descriptor with the given granulated
// SGPR count. The descriptors live in an SHF_ALLOC .rodata section with a real
// sh_addr, and each .kd symbol's st_value is that descriptor's virtual address,
// so min_kernel_sgpr_count() can locate and decode them via Section::vaddr().
// Sections: [1]=.text [2]=.rodata [3]=.strtab [4]=.symtab [5]=.shstrtab.
std::vector<uint8_t>
make_elf_with_kds(const std::vector<std::pair<std::string, uint32_t>> &kernels) {
  constexpr uint64_t kTextAddr = 0x1000;
  constexpr uint64_t kRodataAddr = 0x2000;

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  // .rodata holds one 64-byte KD per kernel; .symtab gets a matching `.kd`
  // symbol whose st_value is the KD's virtual address.
  std::vector<uint8_t> rodata(kernels.size() * sizeof(KD), 0);
  std::vector<uint8_t> strtab{'\0'};
  std::vector<Elf64_Sym> syms(1); // mandatory null symbol
  for (size_t i = 0; i < kernels.size(); ++i) {
    const KD desc = make_kd(kernels[i].second);
    std::memcpy(rodata.data() + i * sizeof(KD), &desc, sizeof(KD));
    Elf64_Sym sym{};
    sym.st_name = add_elf_name(strtab, kernels[i].first + ".kd");
    sym.st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeObject); // global object
    sym.st_shndx = 2;                                                     // .rodata
    sym.st_value = kRodataAddr + i * sizeof(KD);
    sym.st_size = sizeof(KD);
    syms.push_back(sym);
  }

  const uint32_t text_word = 0xbf800000u; // s_nop 0
  const uint64_t text_offset = 0x100;
  const uint64_t text_size = sizeof(text_word);
  const uint64_t rodata_offset = align_up(text_offset + text_size, 8);
  const uint64_t strtab_offset = rodata_offset + rodata.size();
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const uint64_t shstrtab_offset = symtab_offset + syms.size() * sizeof(Elf64_Sym);
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  constexpr uint16_t section_count = 6;

  std::vector<uint8_t> image(shoff + section_count * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_REL;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_shoff = shoff;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = section_count;
  ehdr.e_shstrndx = 5;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  std::memcpy(image.data() + text_offset, &text_word, text_size);
  if (!rodata.empty())
    std::memcpy(image.data() + rodata_offset, rodata.data(), rodata.size());
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + symtab_offset, syms.data(), syms.size() * sizeof(Elf64_Sym));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, section_count> shdrs{};
  shdrs[1].sh_name = text_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  shdrs[1].sh_addr = kTextAddr;
  shdrs[1].sh_offset = text_offset;
  shdrs[1].sh_size = text_size;
  shdrs[1].sh_addralign = sizeof(uint32_t);

  shdrs[2].sh_name = rodata_name;
  shdrs[2].sh_type = SHT_PROGBITS;
  shdrs[2].sh_flags = SHF_ALLOC;
  shdrs[2].sh_addr = kRodataAddr;
  shdrs[2].sh_offset = rodata_offset;
  shdrs[2].sh_size = rodata.size();
  shdrs[2].sh_addralign = 8;

  shdrs[3].sh_name = strtab_name;
  shdrs[3].sh_type = SHT_STRTAB;
  shdrs[3].sh_offset = strtab_offset;
  shdrs[3].sh_size = strtab.size();
  shdrs[3].sh_addralign = 1;

  shdrs[4].sh_name = symtab_name;
  shdrs[4].sh_type = SHT_SYMTAB;
  shdrs[4].sh_offset = symtab_offset;
  shdrs[4].sh_size = syms.size() * sizeof(Elf64_Sym);
  shdrs[4].sh_link = 3; // .strtab
  shdrs[4].sh_info = 1; // index of first global symbol
  shdrs[4].sh_entsize = sizeof(Elf64_Sym);
  shdrs[4].sh_addralign = 8;

  shdrs[5].sh_name = shstrtab_name;
  shdrs[5].sh_type = SHT_STRTAB;
  shdrs[5].sh_offset = shstrtab_offset;
  shdrs[5].sh_size = shstrtab.size();
  shdrs[5].sh_addralign = 1;

  std::memcpy(image.data() + shoff, shdrs.data(), shdrs.size() * sizeof(Elf64_Shdr));
  return image;
}

bool set_aggregate_copied_section_bytes(std::vector<uint8_t> &image, uint64_t aggregate_bytes) {
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_shnum <= 1)
    return false;

  std::vector<Elf64_Shdr> sections(header.e_shnum);
  std::memcpy(sections.data(), image.data() + header.e_shoff, sections.size() * sizeof(Elf64_Shdr));
  uint64_t other_bytes = 0;
  for (size_t i = 2; i < sections.size(); ++i) {
    if (sections[i].sh_type != SHT_NULL && sections[i].sh_type != SHT_NOBITS &&
        sections[i].sh_name != 0) {
      other_bytes += sections[i].sh_size;
    }
  }
  if (aggregate_bytes < other_bytes || aggregate_bytes - other_bytes > image.size())
    return false;
  sections[1].sh_offset = 0;
  sections[1].sh_size = aggregate_bytes - other_bytes;
  std::memcpy(image.data() + header.e_shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  return true;
}

enum class SymbolNameBudget {
  AtBoundary,
  OneByteOver,
};

enum class RetainedSymbolKind {
  Functions,
  KernelDescriptors,
  KernelsAndFunctions,
};

constexpr size_t kDerivedStateBoundaryImageBytes = 4098;
constexpr size_t kRetainedFunctionBoundaryCount = 48;
constexpr size_t kRetainedKernelBoundaryCount = 34;
constexpr size_t kRetainedCombinedBoundaryCount = 20;
constexpr size_t kDerivedStateBudgetBytes =
    kAmdGpuCodeObjectRetainedDerivedStateImageUnits * kDerivedStateBoundaryImageBytes;
constexpr size_t kDerivedStateFixtureClassificationBytes =
    *amdgpu_code_object_detail::section_classification_charge(kDerivedStateFixtureSectionCount);
static_assert(kDerivedStateFixtureClassificationBytes < kDerivedStateBudgetBytes);
constexpr size_t kDerivedStateFixtureRoleBudgetBytes =
    kDerivedStateBudgetBytes - kDerivedStateFixtureClassificationBytes;
// Short function names are copied once and short kernel names twice.
constexpr size_t kMaximumShortRetainedFunctionEntries =
    kDerivedStateFixtureRoleBudgetBytes / (kAmdGpuCodeObjectFunctionEntryChargeBytes + 1);
constexpr size_t kMaximumShortRetainedKernelEntries =
    kDerivedStateFixtureRoleBudgetBytes / (kAmdGpuCodeObjectKernelEntryChargeBytes + 2);
static_assert(kMaximumShortRetainedFunctionEntries <= 64);
static_assert(kMaximumShortRetainedKernelEntries <= 64);
static_assert((kMaximumShortRetainedFunctionEntries + 1) *
                  (kAmdGpuCodeObjectFunctionEntryChargeBytes + 1) >
              kDerivedStateFixtureRoleBudgetBytes);
static_assert((kMaximumShortRetainedKernelEntries + 1) *
                  (kAmdGpuCodeObjectKernelEntryChargeBytes + 2) >
              kDerivedStateFixtureRoleBudgetBytes);

struct DerivedStateBoundaryOptions {
  SymbolNameBudget budget = SymbolNameBudget::AtBoundary;
  RetainedSymbolKind retained_kind = RetainedSymbolKind::Functions;
  uint8_t symbol_type = kElfSymbolTypeFunc;
  uint16_t symbol_section_index = 1;
  size_t short_retained_name_count = 0;
  bool duplicate_symbol_table = false;
  bool include_dynamic_stack_symbol = false;
  bool dynamic_stack_symbol_first = false;
  uint32_t code_section_type = SHT_PROGBITS;
  std::string_view code_section_name = ".text";
  bool code_section_offset_out_of_image = false;
};

std::vector<uint8_t>
make_elf_at_retained_derived_state_boundary(const DerivedStateBoundaryOptions &options = {}) {
  constexpr uint64_t kTextAddress = 0x1000;
  const bool retain_functions = options.retained_kind != RetainedSymbolKind::KernelDescriptors;
  const bool retain_kernels = options.retained_kind != RetainedSymbolKind::Functions;
  const size_t retained_name_count =
      options.short_retained_name_count != 0                   ? options.short_retained_name_count
      : options.retained_kind == RetainedSymbolKind::Functions ? kRetainedFunctionBoundaryCount
      : options.retained_kind == RetainedSymbolKind::KernelDescriptors
          ? kRetainedKernelBoundaryCount
          : kRetainedCombinedBoundaryCount;
  const uint64_t function_entry_charge = options.include_dynamic_stack_symbol
                                             ? kAmdGpuCodeObjectFunctionAndTransientEntryChargeBytes
                                             : kAmdGpuCodeObjectFunctionEntryChargeBytes;
  const uint64_t kernel_entry_charge = options.include_dynamic_stack_symbol && !retain_functions
                                           ? kAmdGpuCodeObjectKernelAndTransientEntryChargeBytes
                                           : kAmdGpuCodeObjectKernelEntryChargeBytes;
  const uint64_t per_name_entry_charge =
      (retain_functions ? function_entry_charge : 0) + (retain_kernels ? kernel_entry_charge : 0);
  const uint64_t charged_name_copies = retain_functions + 2 * retain_kernels;
  size_t base_name_length = 1;
  size_t last_name_extra = 0;
  if (options.short_retained_name_count == 0) {
    const uint64_t fixed_entry_charge = retained_name_count * per_name_entry_charge;
    if (fixed_entry_charge >= kDerivedStateFixtureRoleBudgetBytes ||
        (kDerivedStateFixtureRoleBudgetBytes - fixed_entry_charge) % charged_name_copies != 0) {
      ADD_FAILURE() << "boundary fixture charges do not divide the derived-state budget";
      return {};
    }
    size_t aggregate_name_bytes = static_cast<size_t>(
        (kDerivedStateFixtureRoleBudgetBytes - fixed_entry_charge) / charged_name_copies);
    if (options.budget == SymbolNameBudget::OneByteOver)
      ++aggregate_name_bytes;
    const size_t distinct_length_delta = retained_name_count * (retained_name_count - 1) / 2;
    if (aggregate_name_bytes <= distinct_length_delta) {
      ADD_FAILURE() << "boundary fixture has too few name bytes for distinct symbols";
      return {};
    }
    base_name_length = (aggregate_name_bytes - distinct_length_delta) / retained_name_count;
    last_name_extra = (aggregate_name_bytes - distinct_length_delta) % retained_name_count;
  } else if (retained_name_count > 64) {
    ADD_FAILURE() << "short-name fixture supports at most 64 distinct leading characters";
    return {};
  }

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, options.code_section_name);
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t dynsym_name = add_elf_name(shstrtab, ".dynsym");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");

  std::vector<uint8_t> strtab{'\0'};
  std::vector<Elf64_Sym> symbols;
  const auto append_symbol = [&](std::string name, uint8_t symbol_type) {
    Elf64_Sym symbol{};
    symbol.st_name = add_elf_name(strtab, name);
    symbol.st_info = static_cast<uint8_t>((1u << 4) | symbol_type);
    symbol.st_shndx = options.symbol_section_index;
    symbol.st_value = kTextAddress;
    symbol.st_size = sizeof(uint32_t);
    symbols.push_back(symbol);
  };
  const auto append_dynamic_stack_symbol = [&](const std::string &name) {
    Elf64_Sym symbol{};
    symbol.st_name = add_elf_name(strtab, name + ".has_dyn_sized_stack");
    symbol.st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeNone);
    symbol.st_shndx = SHN_ABS;
    symbol.st_value = 1;
    symbols.push_back(symbol);
  };
  for (size_t i = 0; i < retained_name_count; ++i) {
    size_t name_length =
        options.short_retained_name_count == 0 ? base_name_length + i : base_name_length;
    if (options.short_retained_name_count == 0 && i + 1 == retained_name_count)
      name_length += last_name_extra;
    std::string name(name_length, 'x');
    name.front() = static_cast<char>('a' + i);
    if (options.include_dynamic_stack_symbol && options.dynamic_stack_symbol_first)
      append_dynamic_stack_symbol(name);
    if (retain_functions)
      append_symbol(name, options.symbol_type);
    if (retain_kernels)
      append_symbol(name + ".kd", kElfSymbolTypeObject);
    if (options.include_dynamic_stack_symbol && !options.dynamic_stack_symbol_first)
      append_dynamic_stack_symbol(name);
  }

  constexpr uint64_t text_offset = 0x100;
  const uint64_t strtab_offset = text_offset + sizeof(uint32_t);
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), 8);
  const uint64_t symbol_bytes = symbols.size() * sizeof(Elf64_Sym);
  const uint64_t shstrtab_offset = symtab_offset + symbol_bytes;
  const uint64_t shoff = align_up(shstrtab_offset + shstrtab.size(), 8);
  if (shoff + kDerivedStateFixtureSectionCount * sizeof(Elf64_Shdr) >
      kDerivedStateBoundaryImageBytes) {
    ADD_FAILURE() << "boundary fixture metadata does not fit in its fixed-size image";
    return {};
  }

  std::vector<uint8_t> image(kDerivedStateBoundaryImageBytes, 0);
  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_shoff = shoff;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kDerivedStateFixtureSectionCount;
  header.e_shstrndx = 5;
  std::memcpy(image.data(), &header, sizeof(header));

  const uint32_t text_word = 0xbf800000u;
  std::memcpy(image.data() + text_offset, &text_word, sizeof(text_word));
  std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
  std::memcpy(image.data() + symtab_offset, symbols.data(), symbol_bytes);
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());

  std::array<Elf64_Shdr, kDerivedStateFixtureSectionCount> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = options.code_section_type;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = kTextAddress;
  sections[1].sh_offset = options.code_section_offset_out_of_image ? image.size() + 1 : text_offset;
  sections[1].sh_size = sizeof(text_word);
  sections[1].sh_addralign = alignof(uint32_t);

  sections[2].sh_name = strtab_name;
  sections[2].sh_type = SHT_STRTAB;
  sections[2].sh_offset = strtab_offset;
  sections[2].sh_size = strtab.size();
  sections[2].sh_addralign = 1;

  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = symbol_bytes;
  sections[3].sh_link = 2;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[3].sh_addralign = alignof(Elf64_Sym);

  if (options.duplicate_symbol_table) {
    sections[4] = sections[3];
    sections[4].sh_name = dynsym_name;
    sections[4].sh_type = SHT_DYNSYM;
  }

  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = shstrtab.size();
  sections[5].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sizeof(sections));
  return image;
}

std::vector<uint8_t>
make_zero_sized_section_dense_elf(size_t section_count,
                                  std::optional<Elf_Half> function_section_index = std::nullopt) {
  if (section_count < 3 || section_count > std::numeric_limits<uint16_t>::max()) {
    ADD_FAILURE() << "dense-section fixture count is outside the ELF header range";
    return {};
  }
  const size_t shstrtab_index = section_count - 1;
  const size_t strtab_index = section_count - 2;
  const size_t symtab_index = section_count - 3;
  if (function_section_index &&
      (*function_section_index == SHN_UNDEF || *function_section_index >= symtab_index)) {
    ADD_FAILURE() << "dense-section function index overlaps fixture metadata";
    return {};
  }

  std::vector<uint8_t> shstrtab{'\0'};
  const uint32_t text_name = add_elf_name(shstrtab, ".text");
  const uint32_t rodata_name = add_elf_name(shstrtab, ".rodata");
  const uint32_t strtab_name = add_elf_name(shstrtab, ".strtab");
  const uint32_t symtab_name = add_elf_name(shstrtab, ".symtab");
  const uint32_t shstrtab_name = add_elf_name(shstrtab, ".shstrtab");
  std::vector<uint8_t> strtab{'\0'};
  const uint32_t function_name = add_elf_name(strtab, "f");
  std::array<Elf64_Sym, 2> symbols{};
  symbols[1].st_name = function_name;
  symbols[1].st_info = static_cast<uint8_t>((1u << 4) | kElfSymbolTypeFunc);
  symbols[1].st_shndx = function_section_index.value_or(SHN_UNDEF);
  const uint64_t shoff = sizeof(Elf64_Ehdr);
  const uint64_t shstrtab_offset = shoff + section_count * sizeof(Elf64_Shdr);
  const uint64_t strtab_offset = shstrtab_offset + shstrtab.size();
  const uint64_t symtab_offset = align_up(strtab_offset + strtab.size(), alignof(Elf64_Sym));
  const uint64_t image_size =
      function_section_index ? symtab_offset + sizeof(symbols) : shstrtab_offset + shstrtab.size();
  std::vector<uint8_t> image(image_size, 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_REL;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_shoff = shoff;
  header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX950;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = static_cast<uint16_t>(section_count);
  header.e_shstrndx = static_cast<uint16_t>(shstrtab_index);
  std::memcpy(image.data(), &header, sizeof(header));

  std::vector<Elf64_Shdr> sections(section_count);
  for (size_t i = 1; i + 1 < section_count; ++i) {
    if (function_section_index)
      sections[i].sh_name = i == *function_section_index ? text_name : rodata_name;
    else
      sections[i].sh_name = i % 2 == 0 ? text_name : rodata_name;
    sections[i].sh_type = SHT_PROGBITS;
  }
  if (function_section_index) {
    sections[strtab_index].sh_name = strtab_name;
    sections[strtab_index].sh_type = SHT_STRTAB;
    sections[strtab_index].sh_offset = strtab_offset;
    sections[strtab_index].sh_size = strtab.size();
    sections[strtab_index].sh_addralign = 1;
    sections[symtab_index].sh_name = symtab_name;
    sections[symtab_index].sh_type = SHT_SYMTAB;
    sections[symtab_index].sh_offset = symtab_offset;
    sections[symtab_index].sh_size = sizeof(symbols);
    sections[symtab_index].sh_link = strtab_index;
    sections[symtab_index].sh_info = 1;
    sections[symtab_index].sh_entsize = sizeof(Elf64_Sym);
    sections[symtab_index].sh_addralign = alignof(Elf64_Sym);
  }
  sections[shstrtab_index].sh_name = shstrtab_name;
  sections[shstrtab_index].sh_type = SHT_STRTAB;
  sections[shstrtab_index].sh_offset = shstrtab_offset;
  sections[shstrtab_index].sh_size = shstrtab.size();
  sections[shstrtab_index].sh_addralign = 1;
  std::memcpy(image.data() + shoff, sections.data(), sections.size() * sizeof(Elf64_Shdr));
  std::memcpy(image.data() + shstrtab_offset, shstrtab.data(), shstrtab.size());
  if (function_section_index) {
    std::memcpy(image.data() + strtab_offset, strtab.data(), strtab.size());
    std::memcpy(image.data() + symtab_offset, symbols.data(), sizeof(symbols));
  }
  return image;
}

// CDNA: a granulated field of 0 encodes a real 8-SGPR allocation.
TEST(AmdGpuCodeObjectSgpr, CdnaGranulatedZeroIsEightSgprs) {
  const auto image = make_elf_with_kds({{"k", 0}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  const auto count = obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, 8u);
}

// CDNA: (granulated + 1) * 8. granulated 3 -> 32, exactly enough to own s[30:31].
TEST(AmdGpuCodeObjectSgpr, CdnaGranulatedDecodesTimesEight) {
  const auto image = make_elf_with_kds({{"k", 3}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2), std::optional<uint32_t>(32u));
}

// RDNA: a granulated field of 0 is a sentinel; the wave owns the fixed per-wave
// SGPR pool, not 8. This branch is unreachable on CI hardware, so this unit test
// is the only coverage for it.
TEST(AmdGpuCodeObjectSgpr, RdnaGranulatedZeroIsFixedPool) {
  const auto image = make_elf_with_kds({{"k", 0}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  const auto count = obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_RDNA4);
  ASSERT_TRUE(count.has_value());
  EXPECT_EQ(*count, amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF);
  EXPECT_GE(*count, 32u); // so the fixed link pair s[30:31] always fits on RDNA
}

// Without an anchor->kernel map, the smallest kernel bounds every anchor.
TEST(AmdGpuCodeObjectSgpr, ReturnsMinAcrossKernels) {
  const auto image = make_elf_with_kds({{"big", 7}, {"small", 0}}); // 64 and 8
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2), std::optional<uint32_t>(8u));
}

// No kernel descriptor -> nullopt, so the caller falls back permissively.
TEST(AmdGpuCodeObjectSgpr, NoKernelDescriptorReturnsNullopt) {
  const auto image = make_elf_with_kds({}); // no .kd symbols
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_FALSE(obj.min_kernel_sgpr_count(ROCJITSU_CODE_ARCH_CDNA2).has_value());
}

TEST(AmdGpuCodeObjectAccounting, SymbolRoleChargeCombinesAllRoleCosts) {
  using amdgpu_code_object_detail::retained_symbol_role_charge;

  EXPECT_EQ(retained_symbol_role_charge(false, false, false), 0u);
  EXPECT_EQ(retained_symbol_role_charge(true, true, true),
            kAmdGpuCodeObjectKernelEntryChargeBytes + kAmdGpuCodeObjectFunctionEntryChargeBytes +
                kAmdGpuCodeObjectTransientSymbolEntryChargeBytes);
}

TEST(AmdGpuCodeObjectAccounting, SectionClassificationCrossesWordBoundaries) {
  std::array<uint64_t, 2> words{};
  for (const uint64_t index : {1u, 63u, 64u, 70u})
    ASSERT_TRUE(amdgpu_code_object_detail::set_section_classification(words, index));
  for (const uint64_t index : {1u, 63u, 64u, 70u})
    EXPECT_TRUE(amdgpu_code_object_detail::test_section_classification(words, index));
  EXPECT_FALSE(amdgpu_code_object_detail::test_section_classification(words, 62));
  EXPECT_FALSE(amdgpu_code_object_detail::set_section_classification(words, 128));
  EXPECT_FALSE(amdgpu_code_object_detail::test_section_classification(words, 128));
}

TEST(AmdGpuElfSectionIndex, DistinguishesRegularAndReservedIndices) {
  EXPECT_TRUE(is_regular_elf_section_index(1));
  EXPECT_TRUE(is_regular_elf_section_index(SHN_LORESERVE - 1));
  EXPECT_FALSE(is_regular_elf_section_index(SHN_UNDEF));
  EXPECT_FALSE(is_regular_elf_section_index(SHN_LORESERVE));
  EXPECT_FALSE(is_regular_elf_section_index(SHN_ABS));
  EXPECT_FALSE(is_regular_elf_section_index(std::numeric_limits<uint16_t>::max()));
}

TEST(AmdGpuKernelMetadataPayload, VisitsAllSupportedRetainedFields) {
  const auto payload = make_complete_kernel_metadata_payload();
  std::string visited_name;
  std::optional<amdgpu_code_object_detail::KernelMetadata> visited_metadata;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload,
      [&](std::string_view name, const amdgpu_code_object_detail::KernelMetadata &metadata) {
        visited_name = name;
        visited_metadata = metadata;
        return true;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(visited_name, "kernel");
  ASSERT_TRUE(visited_metadata);
  EXPECT_TRUE(visited_metadata->has_dynamic_lds);
  EXPECT_EQ(visited_metadata->uses_dynamic_stack, std::optional<bool>(true));
  EXPECT_EQ(visited_metadata->sgpr_count, std::optional<uint16_t>(65535u));
  EXPECT_EQ(visited_metadata->required_workgroup_size,
            (std::optional<std::array<uint32_t, 3>>(
                std::array<uint32_t, 3>{1u, 256u, std::numeric_limits<uint32_t>::max()})));
}

TEST(AmdGpuKernelMetadataPayload, PreservesExplicitFalseDynamicStackValue) {
  const std::array<uint8_t, 1> encoded_false{0xc2u};
  const auto payload =
      make_single_kernel_metadata_field_payload(".uses_dynamic_stack", encoded_false);
  std::optional<bool> uses_dynamic_stack;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload, [&](std::string_view, const amdgpu_code_object_detail::KernelMetadata &metadata) {
        uses_dynamic_stack = metadata.uses_dynamic_stack;
        return true;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(uses_dynamic_stack, std::optional<bool>(false));
}

TEST(AmdGpuKernelMetadataPayload, RejectsMalformedTagsAndTruncation) {
  const std::array<uint8_t, 1> invalid_boolean{0xc0u};
  const std::array<uint8_t, 2> truncated_sgpr{0xcdu, 0xffu};
  struct MalformedCase {
    std::string_view description;
    std::vector<uint8_t> payload;
  };
  const std::array<MalformedCase, 4> malformed = {
      MalformedCase{"empty payload", {}},
      MalformedCase{"truncated root map", {0x81u}},
      MalformedCase{"invalid boolean tag", make_single_kernel_metadata_field_payload(
                                               ".uses_dynamic_stack", invalid_boolean)},
      MalformedCase{"truncated integer",
                    make_single_kernel_metadata_field_payload(".sgpr_count", truncated_sgpr)},
  };

  for (const auto &malformed_case : malformed) {
    SCOPED_TRACE(malformed_case.description);
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  malformed_case.payload,
                  [](std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
                    return true;
                  }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  }
}

TEST(AmdGpuKernelMetadataPayload, EnforcesSgprCountRange) {
  std::vector<uint8_t> maximum;
  append_msgpack_unsigned(maximum, std::numeric_limits<uint16_t>::max());
  std::vector<uint8_t> one_too_many;
  append_msgpack_unsigned(one_too_many,
                          static_cast<uint64_t>(std::numeric_limits<uint16_t>::max()) + 1u);
  std::optional<uint16_t> visited_count;
  const auto maximum_payload = make_single_kernel_metadata_field_payload(".sgpr_count", maximum);

  EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                maximum_payload,
                [&](std::string_view, const amdgpu_code_object_detail::KernelMetadata &metadata) {
                  visited_count = metadata.sgpr_count;
                  return true;
                }),
            amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(visited_count, std::optional<uint16_t>(std::numeric_limits<uint16_t>::max()));

  const auto invalid_payload =
      make_single_kernel_metadata_field_payload(".sgpr_count", one_too_many);
  EXPECT_EQ(
      amdgpu_code_object_detail::visit_kernel_metadata_payload(
          invalid_payload,
          [](std::string_view, const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
      amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
}

TEST(AmdGpuKernelMetadataPayload, RequiresThreeNonzeroWorkgroupDimensions) {
  struct InvalidDimensions {
    std::string_view description;
    std::vector<uint8_t> encoded;
  };
  const std::array<InvalidDimensions, 3> invalid_values = {
      InvalidDimensions{"two dimensions", {0x92u, 1u, 1u}},
      InvalidDimensions{"zero dimension", {0x93u, 1u, 0u, 1u}},
      InvalidDimensions{"dimension exceeds uint32",
                        {0x93u, 1u, 1u, 0xcfu, 0u, 0u, 0u, 1u, 0u, 0u, 0u, 0u}},
  };

  for (const auto &invalid : invalid_values) {
    SCOPED_TRACE(invalid.description);
    const auto payload =
        make_single_kernel_metadata_field_payload(".reqd_workgroup_size", invalid.encoded);
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  payload, [](std::string_view,
                              const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  }
}

TEST(AmdGpuKernelMetadataPayload, ReportsVisitorRefusalSeparatelyFromMalformedInput) {
  const auto payload = make_complete_kernel_metadata_payload();
  size_t visit_count = 0;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload, [&](std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
        ++visit_count;
        return false;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::VisitorRejected);
  EXPECT_EQ(visit_count, 1u);
}

TEST(AmdGpuKernelMetadataPayload, MalformationAfterRefusalTakesPrecedence) {
  std::vector<uint8_t> payload;
  payload.push_back(0x81u);
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x92u);
  ASSERT_TRUE(append_kernel_metadata_record(payload, 'k', 8u));
  payload.push_back(0x81u);
  append_msgpack_string(payload, ".sgpr_count");
  payload.push_back(0xcdu); // truncated uint16
  size_t visit_count = 0;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload, [&](std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
        ++visit_count;
        return false;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  EXPECT_EQ(visit_count, 1u);
}

TEST(AmdGpuKernelMetadataPayload, SupportsFreeFunctionsAndMutableVisitors) {
  const auto payload = make_complete_kernel_metadata_payload();
  EXPECT_EQ(
      amdgpu_code_object_detail::visit_kernel_metadata_payload(payload, accept_kernel_metadata),
      amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);

  MutableKernelMetadataVisitor visitor;
  EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(payload, visitor),
            amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(visitor.visit_count, 1u);
}

TEST(AmdGpuKernelMetadataPayload, HandlesExtensionValueBoundariesBeforeFirstKernelArray) {
  struct EncodedValueCase {
    std::string_view description;
    std::vector<uint8_t> encoded;
  };
  const std::array<EncodedValueCase, 8> values = {
      EncodedValueCase{"ext8", {0xc7u, 0x02u, 0x01u, 0xaau, 0xbbu}},
      EncodedValueCase{"ext16", {0xc8u, 0x00u, 0x02u, 0x01u, 0xaau, 0xbbu}},
      EncodedValueCase{"ext32", {0xc9u, 0x00u, 0x00u, 0x00u, 0x02u, 0x01u, 0xaau, 0xbbu}},
      EncodedValueCase{"fixext1", {0xd4u, 0x01u, 0xaau}},
      EncodedValueCase{"fixext2", {0xd5u, 0x01u, 0xaau, 0xbbu}},
      EncodedValueCase{"fixext4", {0xd6u, 0x01u, 0xaau, 0xbbu, 0xccu, 0xddu}},
      EncodedValueCase{"fixext8",
                       {0xd7u, 0x01u, 0xaau, 0xbbu, 0xccu, 0xddu, 0xeeu, 0xf0u, 0x11u, 0x22u}},
      EncodedValueCase{"fixext16",
                       {0xd8u, 0x01u, 0x00u, 0x01u, 0x02u, 0x03u, 0x04u, 0x05u, 0x06u, 0x07u, 0x08u,
                        0x09u, 0x0au, 0x0bu, 0x0cu, 0x0du, 0x0eu, 0x0fu}},
  };

  for (const auto &value : values) {
    SCOPED_TRACE(value.description);
    const auto payload = make_metadata_payload_with_unknown_root_value(value.encoded);
    std::vector<std::string> names;
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  payload,
                  [&](std::string_view name, const amdgpu_code_object_detail::KernelMetadata &) {
                    names.emplace_back(name);
                    return true;
                  }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
    EXPECT_EQ(names, std::vector<std::string>{"k"});
    std::vector<uint8_t> truncated = value.encoded;
    ASSERT_FALSE(truncated.empty());
    truncated.pop_back();
    // Omit the following root entry so its bytes cannot accidentally satisfy
    // the shortened extension payload.
    const auto truncated_payload =
        make_metadata_payload_with_unknown_root_value(truncated, /*append_kernel_array=*/false);
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  truncated_payload,
                  [](std::string_view, const amdgpu_code_object_detail::KernelMetadata &) {
                    return true;
                  }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  }
}

TEST(AmdGpuKernelMetadataPayload, EnforcesNestedUnknownValueDepthBeforeFirstKernelArray) {
  const auto make_nested_array = [](size_t depth, bool declared_count) {
    std::vector<uint8_t> encoded;
    encoded.reserve(depth * (declared_count ? 3u : 1u) + 1u);
    for (size_t level = 0; level < depth; ++level) {
      if (declared_count)
        encoded.insert(encoded.end(), {0xdcu, 0x00u, 0x01u});
      else
        encoded.push_back(0x91u);
    }
    encoded.push_back(0u);
    return encoded;
  };

  for (const auto &[description, declared_count] : std::array<std::pair<std::string_view, bool>, 2>{
           std::pair{"fixarray", false},
           std::pair{"array16", true},
       }) {
    SCOPED_TRACE(description);
    const auto maximum = make_metadata_payload_with_unknown_root_value(make_nested_array(
        amdgpu_code_object_detail::kMaximumKernelMetadataNestingDepth, declared_count));
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  maximum, [](std::string_view,
                              const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);

    const auto too_deep = make_metadata_payload_with_unknown_root_value(make_nested_array(
        amdgpu_code_object_detail::kMaximumKernelMetadataNestingDepth + 1u, declared_count));
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  too_deep, [](std::string_view,
                               const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  }
}

TEST(AmdGpuKernelMetadataPayload, SkipsDeclaredCollectionsBeforeFirstKernelArray) {
  struct EncodedValueCase {
    std::string_view description;
    std::vector<uint8_t> encoded;
  };
  const std::array<EncodedValueCase, 4> values = {
      EncodedValueCase{"array16", {0xdcu, 0x00u, 0x01u, 0u}},
      EncodedValueCase{"array32", {0xddu, 0x00u, 0x00u, 0x00u, 0x01u, 0u}},
      EncodedValueCase{"map16", {0xdeu, 0x00u, 0x01u, 0xa1u, 'k', 0u}},
      EncodedValueCase{"map32", {0xdfu, 0x00u, 0x00u, 0x00u, 0x01u, 0xa1u, 'k', 0u}},
  };

  for (const auto &value : values) {
    SCOPED_TRACE(value.description);
    const auto payload = make_metadata_payload_with_unknown_root_value(value.encoded);
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  payload, [](std::string_view,
                              const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  }
}

TEST(AmdGpuKernelMetadataPayload, RejectsTruncatedDeclaredCollectionsBeforeFirstKernelArray) {
  struct EncodedValueCase {
    std::string_view description;
    std::vector<uint8_t> encoded;
  };
  const std::array<EncodedValueCase, 4> values = {
      EncodedValueCase{"array16", {0xdcu, 0x00u, 0x01u}},
      EncodedValueCase{"array32", {0xddu, 0xffu, 0xffu, 0xffu, 0xffu}},
      EncodedValueCase{"map16", {0xdeu, 0x00u, 0x01u, 0xa1u, 'k'}},
      EncodedValueCase{"map32", {0xdfu, 0xffu, 0xffu, 0xffu, 0xffu}},
  };

  for (const auto &value : values) {
    SCOPED_TRACE(value.description);
    const auto payload =
        make_metadata_payload_with_unknown_root_value(value.encoded,
                                                      /*append_kernel_array=*/false);
    EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                  payload, [](std::string_view,
                              const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
              amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
  }
}

TEST(AmdGpuKernelMetadataPayload, RejectsUnsupportedValueBeforeFirstKernelArray) {
  const std::array<uint8_t, 1> unsupported{0xc1u};
  const auto payload = make_metadata_payload_with_unknown_root_value(unsupported);

  EXPECT_EQ(amdgpu_code_object_detail::visit_kernel_metadata_payload(
                payload, [](std::string_view,
                            const amdgpu_code_object_detail::KernelMetadata &) { return true; }),
            amdgpu_code_object_detail::KernelMetadataVisitStatus::Malformed);
}

TEST(AmdGpuKernelMetadataPayload, IgnoresRootEntriesAndBytesAfterFirstKernelArray) {
  std::vector<uint8_t> payload;
  payload.push_back(0x82u);
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u);
  ASSERT_TRUE(append_kernel_metadata_record(payload, 'k', 8u));
  append_msgpack_string(payload, "future.root.field");
  payload.push_back(0xc1u); // unsupported value outside the retained kernel array
  payload.push_back(0u);    // bytes outside the declared root map
  std::vector<std::string> names;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload, [&](std::string_view name, const amdgpu_code_object_detail::KernelMetadata &) {
        names.emplace_back(name);
        return true;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(names, std::vector<std::string>{"k"});
}

TEST(AmdGpuKernelMetadataPayload, RetainsFirstKernelArrayForDuplicateRootKeys) {
  std::vector<uint8_t> payload;
  payload.push_back(0x82u);
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u);
  ASSERT_TRUE(append_kernel_metadata_record(payload, 'a', 8u));
  append_msgpack_string(payload, "amdhsa.kernels");
  payload.push_back(0x91u);
  ASSERT_TRUE(append_kernel_metadata_record(payload, 'b', 16u));
  std::vector<std::string> names;

  const auto status = amdgpu_code_object_detail::visit_kernel_metadata_payload(
      payload, [&](std::string_view name, const amdgpu_code_object_detail::KernelMetadata &) {
        names.emplace_back(name);
        return true;
      });

  EXPECT_EQ(status, amdgpu_code_object_detail::KernelMetadataVisitStatus::Complete);
  EXPECT_EQ(names, std::vector<std::string>{"a"});
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateCopiedSectionsLargerThanImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));

  // Each range is individually in bounds, but copying both overlapping
  // sections would retain twice the full input image.
  for (const size_t section_index : {size_t{1}, size_t{2}}) {
    const size_t header_offset = header.e_shoff + section_index * sizeof(Elf64_Shdr);
    Elf64_Shdr section{};
    std::memcpy(&section, image.data() + header_offset, sizeof(section));
    section.sh_offset = 0;
    section.sh_size = image.size();
    std::memcpy(image.data() + header_offset, &section, sizeof(section));
  }

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsAggregateCopiedSectionsEqualToImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  ASSERT_TRUE(set_aggregate_copied_section_bytes(image, image.size()));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 0u);
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateCopiedSectionsOneByteOverImage) {
  auto image = make_elf_with_kds({{"k", 0}});
  ASSERT_TRUE(set_aggregate_copied_section_bytes(image, image.size() + 1));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsAggregateSectionNameAmplificationBeforeAllocation) {
  constexpr size_t kImageBytes = 4096;
  constexpr size_t kSectionCount = 32;
  constexpr size_t kSectionTableOffset = sizeof(Elf64_Ehdr);
  constexpr size_t kStringTableOffset = kSectionTableOffset + kSectionCount * sizeof(Elf64_Shdr);
  static_assert(kStringTableOffset < kImageBytes);

  const auto seed = make_elf_with_kds({});
  Elf64_Ehdr header{};
  std::memcpy(&header, seed.data(), sizeof(header));
  header.e_shoff = kSectionTableOffset;
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 0;

  std::vector<uint8_t> image(kImageBytes, 0);
  std::memcpy(image.data(), &header, sizeof(header));
  std::fill(image.begin() + kStringTableOffset, image.end(), static_cast<uint8_t>('x'));

  std::array<Elf64_Shdr, kSectionCount> sections{};
  sections[0].sh_type = SHT_STRTAB;
  sections[0].sh_offset = kStringTableOffset;
  sections[0].sh_size = kImageBytes - kStringTableOffset;
  for (Elf64_Shdr &section : sections)
    section.sh_name = 1;
  std::memcpy(image.data() + kSectionTableOffset, sections.data(), sizeof(sections));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsMaximumDirectlyEncodedElfSectionCount) {
  // Direct e_shnum values stop before the ELF reserved-index range.
  constexpr size_t kSectionCount = SHN_LORESERVE - 1;
  const auto image = make_zero_sized_section_dense_elf(kSectionCount);
  AmdGpuCodeObject obj(image.data(), image.size());

  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.all_sections().size(), kSectionCount - 1);
  EXPECT_EQ(obj.text_sections().size(), (kSectionCount - 2) / 2);
}

TEST(AmdGpuCodeObjectValidation,
     SafelyHandlesMaximumSectionTableWithoutResolvingReservedFunctionIndex) {
  constexpr size_t kSectionCount = std::numeric_limits<uint16_t>::max();
  const auto image = make_zero_sized_section_dense_elf(kSectionCount, SHN_LORESERVE);
  AmdGpuCodeObject obj(image.data(), image.size());

  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.all_sections().size(), kSectionCount - 1);
  EXPECT_EQ(obj.text_sections().size(), 1u);
  EXPECT_TRUE(obj.functions().empty());
}

constexpr size_t kKernelMetadataBoundaryImageBytes = 4096;
constexpr size_t kKernelMetadataFixtureClassificationBytes =
    *amdgpu_code_object_detail::section_classification_charge(kKernelMetadataFixtureSectionCount);
constexpr size_t kMaximumKernelMetadataEntries =
    (kAmdGpuCodeObjectRetainedDerivedStateImageUnits * kKernelMetadataBoundaryImageBytes -
     kKernelMetadataFixtureClassificationBytes) /
    kAmdGpuCodeObjectKernelMetadataEntryChargeBytes;
static_assert(kMaximumKernelMetadataEntries < 255);
static_assert(kKernelMetadataFixtureClassificationBytes +
                  kMaximumKernelMetadataEntries * kAmdGpuCodeObjectKernelMetadataEntryChargeBytes <=
              kAmdGpuCodeObjectRetainedDerivedStateImageUnits * kKernelMetadataBoundaryImageBytes);
static_assert(kKernelMetadataFixtureClassificationBytes +
                  (kMaximumKernelMetadataEntries + 1) *
                      kAmdGpuCodeObjectKernelMetadataEntryChargeBytes >
              kAmdGpuCodeObjectRetainedDerivedStateImageUnits * kKernelMetadataBoundaryImageBytes);

TEST(AmdGpuCodeObjectValidation, AcceptsMaximumKernelMetadataEntriesWithinBudget) {
  const auto image = make_elf_with_kernel_metadata_note(kMaximumKernelMetadataEntries,
                                                        kKernelMetadataBoundaryImageBytes);
  ASSERT_EQ(image.size(), kKernelMetadataBoundaryImageBytes);

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RejectsOneTooManyKernelMetadataEntries) {
  const auto image = make_elf_with_kernel_metadata_note(kMaximumKernelMetadataEntries + 1,
                                                        kKernelMetadataBoundaryImageBytes);
  ASSERT_EQ(image.size(), kKernelMetadataBoundaryImageBytes);

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, DoesNotChargeDuplicateMetadataWithinOneNote) {
  auto records = make_distinct_kernel_metadata_records(kMaximumKernelMetadataEntries);
  ASSERT_EQ(records.size(), kMaximumKernelMetadataEntries);
  records.push_back(records.front());
  const auto image = make_elf_with_kernel_metadata_payloads(
      {make_named_kernel_metadata_payload(records)}, kKernelMetadataBoundaryImageBytes);
  ASSERT_EQ(image.size(), kKernelMetadataBoundaryImageBytes);

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, DoesNotChargeDuplicateMetadataAcrossNotes) {
  const auto records = make_distinct_kernel_metadata_records(kMaximumKernelMetadataEntries);
  ASSERT_EQ(records.size(), kMaximumKernelMetadataEntries);
  const auto image = make_elf_with_kernel_metadata_payloads(
      {make_named_kernel_metadata_payload(records),
       make_named_kernel_metadata_payload({records.front()})},
      kKernelMetadataBoundaryImageBytes);
  ASSERT_EQ(image.size(), kKernelMetadataBoundaryImageBytes);

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, ReportsMalformedNoteWithoutConsumingMetadataStateBudget) {
  const auto records = make_distinct_kernel_metadata_records(kMaximumKernelMetadataEntries);
  ASSERT_EQ(records.size(), kMaximumKernelMetadataEntries);
  const auto image = make_elf_with_kernel_metadata_payloads(
      {{0x81u}, make_named_kernel_metadata_payload(records)}, kKernelMetadataBoundaryImageBytes);
  ASSERT_EQ(image.size(), kKernelMetadataBoundaryImageBytes);

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 1u);
  EXPECT_FALSE(obj.kernel_metadata_is_trustworthy());
}

TEST(AmdGpuCodeObjectValidation, MalformedNoteTrustStateSurvivesMoveConstruction) {
  const auto image = make_elf_with_kernel_metadata_payloads({{0x81u}});
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  ASSERT_EQ(obj.malformed_kernel_metadata_note_count(), 1u);
  ASSERT_FALSE(obj.kernel_metadata_is_trustworthy());

  AmdGpuCodeObject moved(std::move(obj));
  EXPECT_TRUE(moved.is_valid());
  EXPECT_EQ(moved.malformed_kernel_metadata_note_count(), 1u);
  EXPECT_FALSE(moved.kernel_metadata_is_trustworthy());
}

TEST(AmdGpuCodeObjectValidation, RejectsAllMetadataFromPartiallyMalformedNote) {
  auto image = make_elf_with_kds({{"k", 0}});
  auto payload = make_named_kernel_metadata_payload({{'k', 8u}, {'j', 8u}});
  ASSERT_FALSE(payload.empty());
  payload.back() = 0xc1u;
  ASSERT_TRUE(
      install_kernel_metadata_notes(image, {make_kernel_metadata_note_from_payload(payload)}));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 1u);
  EXPECT_FALSE(obj.kernel_metadata_is_trustworthy());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_FALSE(obj.kernels().front().sgpr_count);
}

TEST(AmdGpuCodeObjectValidation, CountsMalformedNotesAndRetainsLaterValidNote) {
  auto image = make_elf_with_kds({{"k", 0}});
  const auto malformed_a = make_kernel_metadata_note_from_payload({0x81u});
  const auto malformed_b = make_kernel_metadata_note_from_payload({0x81u, 0xa1u, 'x'});
  const auto valid =
      make_kernel_metadata_note_from_payload(make_named_kernel_metadata_payload({{'k', 8u}}));
  ASSERT_TRUE(install_kernel_metadata_notes(image, {malformed_a, malformed_b, valid}));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 2u);
  EXPECT_FALSE(obj.kernel_metadata_is_trustworthy());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_EQ(obj.kernels().front().sgpr_count, std::optional<uint16_t>(8u));
}

TEST(AmdGpuCodeObjectValidation, ReportsMalformedMetadataNoteFraming) {
  auto image = make_elf_with_kds({{"k", 0}});
  const auto note =
      make_kernel_metadata_note_from_payload(make_named_kernel_metadata_payload({{'k', 8u}}));
  ASSERT_TRUE(install_kernel_metadata_notes(image, {note}));
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, image.data() + header.e_phoff, sizeof(note_segment));
  Elf64_Nhdr note_header{};
  std::memcpy(&note_header, image.data() + note_segment.p_offset, sizeof(note_header));
  note_header.n_descsz = std::numeric_limits<uint32_t>::max();
  std::memcpy(image.data() + note_segment.p_offset, &note_header, sizeof(note_header));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 1u);
  EXPECT_FALSE(obj.kernel_metadata_is_trustworthy());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_FALSE(obj.kernels().front().sgpr_count);
}

TEST(AmdGpuCodeObjectValidation, ReportsOutOfRangeNoteSegmentAsIncompleteMetadataScan) {
  auto image = make_elf_with_kds({{"k", 0}});
  ASSERT_TRUE(install_kernel_metadata_note(image, 1u));
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  Elf64_Phdr note_segment{};
  std::memcpy(&note_segment, image.data() + header.e_phoff, sizeof(note_segment));
  note_segment.p_filesz = image.size();
  std::memcpy(image.data() + header.e_phoff, &note_segment, sizeof(note_segment));

  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 0u);
  EXPECT_FALSE(obj.kernel_metadata_is_trustworthy());
}

TEST(AmdGpuCodeObjectValidation, RejectsRepeatedMetadataPayloadsAboveParseWorkBudget) {
  constexpr size_t kDuplicateRecordCount = 200;
  constexpr size_t kRepeatedProgramHeaderCount = 3;
  const std::vector<std::pair<char, uint8_t>> records(kDuplicateRecordCount, {'k', 8u});
  const auto payload = make_named_kernel_metadata_payload(records);
  ASSERT_FALSE(payload.empty());
  const auto single_reference = make_elf_with_kernel_metadata_payloads({payload});
  ASSERT_FALSE(single_reference.empty());
  ASSERT_LE(2 * payload.size(),
            kAmdGpuCodeObjectMetadataParseWorkImageUnits * single_reference.size());
  AmdGpuCodeObject accepted(single_reference.data(), single_reference.size());
  ASSERT_TRUE(accepted.is_valid());

  const auto repeated_references =
      make_elf_with_kernel_metadata_payloads({payload}, 0, kRepeatedProgramHeaderCount);
  ASSERT_FALSE(repeated_references.empty());
  ASSERT_GT(2 * payload.size() * kRepeatedProgramHeaderCount,
            kAmdGpuCodeObjectMetadataParseWorkImageUnits * repeated_references.size());
  AmdGpuCodeObject rejected(repeated_references.data(), repeated_references.size());
  EXPECT_FALSE(rejected.is_valid());
}

TEST(AmdGpuCodeObjectValidation, RetainsLastRecordForDuplicateKernelNamesWithinOneNote) {
  auto image = make_elf_with_kds({{"k", 0}});
  const auto payload = make_named_kernel_metadata_payload({{'k', 8u}, {'k', 16u}});
  const auto note = make_kernel_metadata_note_from_payload(payload);
  ASSERT_TRUE(install_kernel_metadata_notes(image, {note}));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 0u);
  EXPECT_TRUE(obj.kernel_metadata_is_trustworthy());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_EQ(obj.kernels().front().sgpr_count, std::optional<uint16_t>(16u));
}

TEST(AmdGpuCodeObjectValidation, RetainsFirstRecordForDuplicateKernelNamesAcrossNotes) {
  auto image = make_elf_with_kds({{"k", 0}});
  const auto first_note =
      make_kernel_metadata_note_from_payload(make_named_kernel_metadata_payload({{'k', 8u}}));
  const auto second_note =
      make_kernel_metadata_note_from_payload(make_named_kernel_metadata_payload({{'k', 16u}}));
  ASSERT_TRUE(install_kernel_metadata_notes(image, {first_note, second_note}));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_EQ(obj.kernels().front().sgpr_count, std::optional<uint16_t>(8u));
}

TEST(AmdGpuCodeObjectValidation, RetainsMetadataWhenLaterRootBytesAreUnsupported) {
  auto image = make_elf_with_kds({{"k", 0}});
  auto payload = make_named_kernel_metadata_payload({{'k', 8u}});
  payload.push_back(0xc1u);
  const auto note = make_kernel_metadata_note_from_payload(payload);
  ASSERT_TRUE(install_kernel_metadata_notes(image, {note}));

  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.malformed_kernel_metadata_note_count(), 0u);
  EXPECT_TRUE(obj.kernel_metadata_is_trustworthy());
  ASSERT_EQ(obj.kernels().size(), 1u);
  EXPECT_EQ(obj.kernels().front().sgpr_count, std::optional<uint16_t>(8u));
}

TEST(AmdGpuCodeObjectValidation, AcceptsCombinedSymbolAndMetadataStateBelowBudget) {
  constexpr size_t kShortFunctionEntryCharge = kAmdGpuCodeObjectFunctionEntryChargeBytes + 1;
  static_assert(kDerivedStateFixtureClassificationBytes +
                    kMaximumShortRetainedFunctionEntries * kShortFunctionEntryCharge +
                    kAmdGpuCodeObjectKernelMetadataEntryChargeBytes <=
                kDerivedStateBudgetBytes);

  auto image = make_elf_at_retained_derived_state_boundary({
      .short_retained_name_count = kMaximumShortRetainedFunctionEntries,
  });
  ASSERT_TRUE(install_kernel_metadata_note(image, 1));
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, SharesDerivedStateBudgetBetweenSymbolsAndMetadata) {
  {
    const auto image = make_elf_with_kernel_metadata_note(1);
    AmdGpuCodeObject metadata_only(image.data(), image.size());
    ASSERT_TRUE(metadata_only.is_valid());
  }
  {
    const auto image = make_elf_at_retained_derived_state_boundary();
    AmdGpuCodeObject symbol_only(image.data(), image.size());
    ASSERT_TRUE(symbol_only.is_valid());
  }
  auto image = make_elf_at_retained_derived_state_boundary();
  ASSERT_TRUE(install_kernel_metadata_note(image, 1));
  AmdGpuCodeObject combined(image.data(), image.size());
  EXPECT_FALSE(combined.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsRetainedFunctionStateEqualToBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary();
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.functions().size(), kRetainedFunctionBoundaryCount);
}

TEST(AmdGpuCodeObjectValidation, RejectsRetainedFunctionStateOverBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsMaximumShortRetainedFunctionEntries) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .short_retained_name_count = kMaximumShortRetainedFunctionEntries,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.functions().size(), kMaximumShortRetainedFunctionEntries);
}

TEST(AmdGpuCodeObjectValidation, RejectsOneTooManyShortRetainedFunctionEntries) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .short_retained_name_count = kMaximumShortRetainedFunctionEntries + 1,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, IgnoresUnretainedNamesAboveRetainedNameBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .symbol_type = kElfSymbolTypeNone,
      .symbol_section_index = SHN_UNDEF,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.kernels().empty());
}

TEST(AmdGpuCodeObjectValidation, IgnoresFunctionSymbolWithOutOfRangeSectionIndex) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .symbol_section_index = 42,
      .short_retained_name_count = 1,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
}

TEST(AmdGpuCodeObjectValidation, DeduplicatesRetainedNamesAcrossSymbolTables) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .duplicate_symbol_table = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.functions().size(), kRetainedFunctionBoundaryCount);
}

TEST(AmdGpuCodeObjectValidation, AcceptsRetainedKernelStateEqualToBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), kRetainedKernelBoundaryCount);
}

TEST(AmdGpuCodeObjectValidation, RejectsRetainedKernelStateOverBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AcceptsMaximumShortRetainedKernelEntries) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
      .short_retained_name_count = kMaximumShortRetainedKernelEntries,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), kMaximumShortRetainedKernelEntries);
}

TEST(AmdGpuCodeObjectValidation, RejectsOneTooManyShortRetainedKernelEntries) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
      .short_retained_name_count = kMaximumShortRetainedKernelEntries + 1,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, DeduplicatesRetainedKernelStateAcrossSymbolTables) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelDescriptors,
      .duplicate_symbol_table = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), kRetainedKernelBoundaryCount);
}

TEST(AmdGpuCodeObjectValidation, ChargesKernelAndFunctionRetentionIndependently) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_TRUE(obj.is_valid());
  EXPECT_EQ(obj.kernels().size(), kRetainedCombinedBoundaryCount);
  EXPECT_EQ(obj.functions().size(), kRetainedCombinedBoundaryCount);
}

TEST(AmdGpuCodeObjectValidation, RejectsCombinedKernelAndFunctionStateOverBudget) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .budget = SymbolNameBudget::OneByteOver,
      .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  EXPECT_FALSE(obj.is_valid());
}

TEST(AmdGpuCodeObjectValidation, AccountsSharedDynamicStackStateIndependentOfSymbolOrder) {
  for (const bool dynamic_stack_symbol_first : {false, true}) {
    const auto image = make_elf_at_retained_derived_state_boundary({
        .retained_kind = RetainedSymbolKind::KernelsAndFunctions,
        .short_retained_name_count = 1,
        .include_dynamic_stack_symbol = true,
        .dynamic_stack_symbol_first = dynamic_stack_symbol_first,
    });
    AmdGpuCodeObject obj(image.data(), image.size());

    ASSERT_TRUE(obj.is_valid()) << "dynamic_stack_symbol_first=" << dynamic_stack_symbol_first;
    ASSERT_EQ(obj.kernels().size(), 1u);
    EXPECT_EQ(obj.kernels().front().uses_dynamic_stack, std::optional<bool>(true));
    EXPECT_EQ(obj.functions().size(), 1u);
  }
}

TEST(AmdGpuCodeObjectValidation, DoesNotReportFunctionsFromInBoundsNoBitsTextSection) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .code_section_type = SHT_NOBITS,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.text_sections().empty());
}

TEST(AmdGpuCodeObjectValidation, DoesNotReportFunctionsFromOutOfImageNoBitsTextSection) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .code_section_type = SHT_NOBITS,
      .code_section_offset_out_of_image = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.text_sections().empty());
}

TEST(AmdGpuCodeObjectValidation, DoesNotTreatOtherNoBitsSectionAsText) {
  const auto image = make_elf_at_retained_derived_state_boundary({
      .code_section_type = SHT_NOBITS,
      .code_section_name = ".bss",
      .code_section_offset_out_of_image = true,
  });
  AmdGpuCodeObject obj(image.data(), image.size());
  ASSERT_TRUE(obj.is_valid());
  EXPECT_TRUE(obj.functions().empty());
  EXPECT_TRUE(obj.text_sections().empty());
}

} // namespace
} // namespace rocjitsu
