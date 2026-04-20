// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"

#include "rocjitsu/code/amdgpu_elf.h"

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
  uint64_t vaddr() const override { return shdr_.sh_addr; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

rj_code_target_id_t target_from_machine_flags(uint32_t flags) {
  uint32_t mach = flags & EF_AMDGPU_MACH;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX942)
    return ROCJITSU_CODE_TARGET_GFX942;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX950)
    return ROCJITSU_CODE_TARGET_GFX950;
  return ROCJITSU_CODE_TARGET_INVALID;
}

rj_code_target_id_t target_from_triple(const std::string &triple) {
  if (triple == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (triple == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
  return ROCJITSU_CODE_TARGET_INVALID;
}

} // namespace

AmdGpuCodeObject::AmdGpuCodeObject(AmdGpuCodeObject &&other) noexcept
    : target_id_(other.target_id_), offload_kind_(std::move(other.offload_kind_)),
      target_triple_(std::move(other.target_triple_)), fatbin_offset_(other.fatbin_offset_) {
  is_valid_ = other.is_valid_;
  image_ = std::move(other.image_);
  header_ = std::move(other.header_);
  sections_ = std::move(other.sections_);
  text_sections_ = std::move(other.text_sections_);
  rodata_sections_ = std::move(other.rodata_sections_);
}

AmdGpuCodeObject::AmdGpuCodeObject(const std::string &elf_path) {
  std::ifstream elf_file(elf_path, std::ios::ate | std::ios::binary);
  if (!elf_file) {
    is_valid_ = false;
    return;
  }
  auto file_size = static_cast<std::size_t>(elf_file.tellg());

  if (file_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  elf_file.seekg(0, std::ios::beg);
  image_.resize(file_size, 0);
  elf_file.read(image_.data(), static_cast<std::streamsize>(file_size));
  if (!elf_file) {
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
  load_sections(elf_file);
  target_id_ = target_from_machine_flags(header_->flags());
  elf_file.close();
}

AmdGpuCodeObject::AmdGpuCodeObject(uint64_t size, std::ifstream &elf_file, std::string offload_kind,
                                   std::string target_triple, int64_t fatbin_offset)
    : offload_kind_(std::move(offload_kind)), target_triple_(std::move(target_triple)),
      fatbin_offset_(fatbin_offset) {
  elf_file.seekg(fatbin_offset_, std::ios::beg);

  if (size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.resize(size, 0);
  elf_file.read(image_.data(), static_cast<std::streamsize>(size));
  if (!elf_file) {
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
  load_sections(elf_file);
  target_id_ = target_from_triple(target_triple_);
}

AmdGpuCodeObject::~AmdGpuCodeObject() = default;

void AmdGpuCodeObject::load_sections(std::ifstream &elf_file) {
  elf_file.seekg(static_cast<std::streamoff>(header_->sectionHeaderOff() + fatbin_offset_),
                 std::ios::beg);
  std::vector<Elf64_Shdr> section_hdrs(header_->numSectionHeaders());
  elf_file.read(reinterpret_cast<char *>(section_hdrs.data()),
                static_cast<std::streamsize>(section_hdrs.size() * sizeof(Elf64_Shdr)));
  if (!elf_file) {
    is_valid_ = false;
    return;
  }

  int shstrndx = header_->sectionHeaderStrIdx();
  if (shstrndx < 0 || static_cast<size_t>(shstrndx) >= section_hdrs.size()) {
    is_valid_ = false;
    return;
  }

  auto &shstrtab = section_hdrs[shstrndx];
  std::vector<char> shstrtab_data(shstrtab.sh_size);
  elf_file.seekg(static_cast<std::streamoff>(shstrtab.sh_offset + fatbin_offset_), std::ios::beg);
  elf_file.read(shstrtab_data.data(), static_cast<std::streamsize>(shstrtab_data.size()));
  if (!elf_file) {
    is_valid_ = false;
    return;
  }

  for (const auto &shdr : section_hdrs) {
    if (shdr.sh_type == SHT_NULL)
      continue;
    if (shdr.sh_name >= shstrtab_data.size())
      continue;

    size_t max_len = shstrtab_data.size() - shdr.sh_name;
    std::string sec_name(&shstrtab_data[shdr.sh_name],
                         strnlen(&shstrtab_data[shdr.sh_name], max_len));
    elf_file.seekg(static_cast<std::streamoff>(shdr.sh_offset + fatbin_offset_), std::ios::beg);
    if (shdr.sh_offset + shdr.sh_size > image_.size()) {
      is_valid_ = false;
      return;
    }

    auto sec_data = std::make_unique<char[]>(shdr.sh_size);
    elf_file.read(sec_data.get(), static_cast<std::streamsize>(shdr.sh_size));
    if (!elf_file) {
      is_valid_ = false;
      return;
    }
    sections_.emplace_back(std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr));

    if (sec_name == ".text")
      text_sections_.push_back(sections_.back().get());
    else if (sec_name == ".rodata")
      rodata_sections_.push_back(sections_.back().get());
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

    // Read the string table linked to this symtab.
    if (symtab_shdr.sh_link >= section_hdrs.size())
      continue;
    auto &strtab_shdr = section_hdrs[symtab_shdr.sh_link];
    std::vector<char> sym_strtab(strtab_shdr.sh_size);
    elf_file.seekg(static_cast<std::streamoff>(strtab_shdr.sh_offset + fatbin_offset_),
                   std::ios::beg);
    elf_file.read(sym_strtab.data(), static_cast<std::streamsize>(sym_strtab.size()));
    if (!elf_file)
      break;

    // Read symbols.
    size_t num_syms = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    std::vector<Elf64_Sym> syms(num_syms);
    elf_file.seekg(static_cast<std::streamoff>(symtab_shdr.sh_offset + fatbin_offset_),
                   std::ios::beg);
    elf_file.read(reinterpret_cast<char *>(syms.data()),
                  static_cast<std::streamsize>(symtab_shdr.sh_size));
    if (!elf_file)
      break;

    for (const auto &sym : syms) {
      if (sym.st_name >= sym_strtab.size())
        continue;
      std::string sym_name(&sym_strtab[sym.st_name],
                           strnlen(&sym_strtab[sym.st_name], sym_strtab.size() - sym.st_name));
      // AMDHSA kernel descriptors have a ".kd" suffix symbol.
      if (sym_name.size() > 3 && sym_name.substr(sym_name.size() - 3) == ".kd") {
        std::string kernel_name = sym_name.substr(0, sym_name.size() - 3);
        kd_offsets_[kernel_name] = sym.st_value;
      }
    }
  }
}

uint64_t AmdGpuCodeObject::kernel_descriptor_offset(const std::string &kernel_name) const {
  auto it = kd_offsets_.find(kernel_name);
  return it != kd_offsets_.end() ? it->second : 0;
}

} // namespace rocjitsu
