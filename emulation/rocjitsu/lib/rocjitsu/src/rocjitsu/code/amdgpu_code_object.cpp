// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"

#include <cstring>
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

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

bool is_executable_section(const Elf64_Shdr &shdr) {
  return shdr.sh_type == SHT_PROGBITS && (shdr.sh_flags & SHF_EXECINSTR) != 0;
}

using detail::fits_in_bounds;

rj_code_target_id_t target_from_machine_flags(uint32_t flags) {
  uint32_t mach = flags & EF_AMDGPU_MACH;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX942)
    return ROCJITSU_CODE_TARGET_GFX942;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX950)
    return ROCJITSU_CODE_TARGET_GFX950;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1200)
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1201)
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1250)
    return ROCJITSU_CODE_TARGET_GFX1250;
  return ROCJITSU_CODE_TARGET_INVALID;
}

rj_code_target_id_t target_from_triple(const std::string &triple) {
  if (triple == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (triple == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
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
  code_sections_ = std::move(other.code_sections_);
  rodata_sections_ = std::move(other.rodata_sections_);
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
  if (num_shdrs < 0 || !fits_in_bounds(shoff, static_cast<uint64_t>(num_shdrs) * sizeof(Elf64_Shdr),
                                       image_.size())) {
    is_valid_ = false;
    return;
  }

  std::vector<Elf64_Shdr> section_hdrs(static_cast<size_t>(num_shdrs));
  std::memcpy(section_hdrs.data(), image_.data() + shoff, section_hdrs.size() * sizeof(Elf64_Shdr));

  int shstrndx = header_->sectionHeaderStrIdx();
  if (shstrndx < 0 || static_cast<size_t>(shstrndx) >= section_hdrs.size()) {
    is_valid_ = false;
    return;
  }

  auto &shstrtab = section_hdrs[shstrndx];
  if (!fits_in_bounds(shstrtab.sh_offset, shstrtab.sh_size, image_.size())) {
    is_valid_ = false;
    return;
  }
  const char *shstrtab_data = image_.data() + shstrtab.sh_offset;

  for (const auto &shdr : section_hdrs) {
    if (shdr.sh_type == SHT_NULL || shdr.sh_type == SHT_NOBITS)
      continue;
    if (shdr.sh_name >= shstrtab.sh_size)
      continue;
    if (!fits_in_bounds(shdr.sh_offset, shdr.sh_size, image_.size())) {
      is_valid_ = false;
      return;
    }

    size_t max_len = shstrtab.sh_size - shdr.sh_name;
    std::string sec_name(shstrtab_data + shdr.sh_name,
                         strnlen(shstrtab_data + shdr.sh_name, max_len));

    auto sec_data = std::make_unique<char[]>(shdr.sh_size);
    std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
    sections_.emplace_back(std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr));

    if (sec_name == ".text")
      text_sections_.push_back(sections_.back().get());
    else if (sec_name == ".rodata")
      rodata_sections_.push_back(sections_.back().get());

    if (is_executable_section(shdr))
      code_sections_.push_back(sections_.back().get());
  }

  // Parse symbol table for kernel descriptor offsets.
  // Scan both SHT_SYMTAB and SHT_DYNSYM — stripped code objects may
  // only have the latter.
  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    if (section_hdrs[i].sh_type != SHT_SYMTAB && section_hdrs[i].sh_type != SHT_DYNSYM)
      continue;
    auto &symtab_shdr = section_hdrs[i];
    if (symtab_shdr.sh_entsize == 0)
      continue;
    if (symtab_shdr.sh_entsize < sizeof(Elf64_Sym))
      continue;
    if (!fits_in_bounds(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_.size()))
      continue;

    // Read the string table linked to this symtab.
    if (symtab_shdr.sh_link >= section_hdrs.size())
      continue;
    auto &strtab_shdr = section_hdrs[symtab_shdr.sh_link];
    if (!fits_in_bounds(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_.size()))
      continue;
    const char *sym_strtab = image_.data() + strtab_shdr.sh_offset;

    // Read symbols.
    size_t num_syms = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    const char *symtab_data = image_.data() + symtab_shdr.sh_offset;

    for (size_t sym_index = 0; sym_index < num_syms; ++sym_index) {
      const char *sym_data = symtab_data + sym_index * symtab_shdr.sh_entsize;
      Elf64_Sym sym;
      std::memcpy(&sym, sym_data, sizeof(sym));
      if (sym.st_name >= strtab_shdr.sh_size)
        continue;
      std::string sym_name(sym_strtab + sym.st_name,
                           strnlen(sym_strtab + sym.st_name, strtab_shdr.sh_size - sym.st_name));
      // AMDHSA kernel descriptors have a ".kd" suffix symbol.
      if (sym_name.size() > 3 && sym_name.substr(sym_name.size() - 3) == ".kd") {
        std::string kernel_name = sym_name.substr(0, sym_name.size() - 3);
        kd_offsets_[kernel_name] = sym.st_value;
      }
    }
  }

  is_valid_ = true;
}

uint64_t AmdGpuCodeObject::kernel_descriptor_offset(const std::string &kernel_name) const {
  auto it = kd_offsets_.find(kernel_name);
  return it != kd_offsets_.end() ? it->second : 0;
}

} // namespace rocjitsu
