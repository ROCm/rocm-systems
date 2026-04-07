// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/executable.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "util/debug_print.h"

#include <cstring>
#include <ranges>
#include <utility>

namespace rocjitsu {

namespace {

constexpr char CLANG_OFFLOAD_KIND_HOST[] = "host";
constexpr char CLANG_OFFLOAD_KIND_HIP[] = "hip";
constexpr char CLANG_OFFLOAD_KIND_HIPV4[] = "hipv4";

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
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

} // namespace

Executable::Executable(const std::string &path) {
  std::ifstream elf_file(path, std::ios::ate | std::ios::binary);
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
  elf_file.read(image_.data(), file_size);
  if (!elf_file) {
    is_valid_ = false;
    return;
  }

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr)) {
    is_valid_ = false;
    return;
  }

  if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) {
    is_valid_ = false;
    return;
  }

  // Standalone AMD GPU HSA device ELF.
  if (ehdr.e_ident[EI_OSABI] == ELFOSABI_AMDGPU_HSA) {
    elf_file.close();
    load_device_elf(path);
    return;
  }

  // x86 host ELF (fat binary with .hip_fatbin section).
  if (ehdr.e_ident[EI_OSABI] == ELFOSABI_NONE) {
    header_ = std::make_unique<HsaHeader>(ehdr);
    load_fat_binary(elf_file);
    elf_file.close();
    return;
  }

  is_valid_ = false;
}

Executable::~Executable() = default;

uint32_t Executable::num_code_objects(rj_code_target_id_t target) const {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end())
    return 0;
  return static_cast<uint32_t>(it->second.size());
}

AmdGpuCodeObject *Executable::code_object(rj_code_target_id_t target, uint32_t index) {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end() || index >= it->second.size())
    return nullptr;
  return it->second[index];
}

const AmdGpuCodeObject *Executable::code_object(rj_code_target_id_t target, uint32_t index) const {
  auto it = code_objs_by_target_.find(target);
  if (it == code_objs_by_target_.end() || index >= it->second.size())
    return nullptr;
  return it->second[index];
}

void Executable::load_device_elf(const std::string &path) {
  auto co = std::make_unique<AmdGpuCodeObject>(path);
  if (!co->is_valid()) {
    is_valid_ = false;
    return;
  }
  register_code_object(std::move(co));
}

void Executable::load_fat_binary(std::ifstream &elf_file) {
  // Parse section headers to find .hip_fatbin.
  elf_file.seekg(static_cast<std::streamoff>(header_->sectionHeaderOff()), std::ios::beg);
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
  elf_file.seekg(static_cast<std::streamoff>(shstrtab.sh_offset), std::ios::beg);
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

    if (sec_name == ".hip_fatbin") {
      std::unique_ptr<char[]> sec_data(std::make_unique<char[]>(shdr.sh_size));
      elf_file.seekg(static_cast<std::streamoff>(shdr.sh_offset), std::ios::beg);
      elf_file.read(sec_data.get(), static_cast<std::streamsize>(shdr.sh_size));
      if (!elf_file) {
        is_valid_ = false;
        return;
      }
      auto section = std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr);
      load_hip_fatbin(*section, elf_file);
      sections_.emplace_back(std::move(section));
      if (!is_valid_)
        return;
    }
  }
}

void Executable::load_hip_fatbin(const Section &fatbin_section, std::ifstream &elf_file) {
  ClangOffloadBundleHeader bundle_hdr;
  elf_file.seekg(static_cast<std::streamoff>(fatbin_section.sectionOffset()), std::ios::beg);
  elf_file.read(bundle_hdr.magic, sizeof(bundle_hdr.magic));
  if (!elf_file) {
    is_valid_ = false;
    return;
  }

  if (std::memcmp(bundle_hdr.magic, CLANG_OFFLOAD_MAGIC_STR, CLANG_OFFLOAD_MAGIC_STR_SIZE)) {
    is_valid_ = false;
    return;
  }

  elf_file.read(reinterpret_cast<char *>(&bundle_hdr.num_code_objs),
                sizeof(bundle_hdr.num_code_objs));
  if (!elf_file) {
    is_valid_ = false;
    return;
  }

  util::debug::print(__func__, ": Found .hip_fatbin with ", bundle_hdr.num_code_objs,
                     " code objects");

  std::vector<ClangOffloadBundleInfo> infos(bundle_hdr.num_code_objs);
  for (auto &info : infos) {
    elf_file.read(reinterpret_cast<char *>(&info.offset), sizeof(info.offset));
    if (!elf_file) {
      is_valid_ = false;
      return;
    }
    elf_file.read(reinterpret_cast<char *>(&info.size), sizeof(info.size));
    if (!elf_file) {
      is_valid_ = false;
      return;
    }
    elf_file.read(reinterpret_cast<char *>(&info.bundle_entry_id_size),
                  sizeof(info.bundle_entry_id_size));
    if (!elf_file) {
      is_valid_ = false;
      return;
    }
    info.bundle_entry_id.resize(info.bundle_entry_id_size, '\0');
    elf_file.read(info.bundle_entry_id.data(),
                  static_cast<std::streamsize>(info.bundle_entry_id.size()));
    if (!elf_file) {
      is_valid_ = false;
      return;
    }

    auto parts_view = info.bundle_entry_id | std::views::split('-');
    std::vector<std::string> parts;
    for (const auto &v : parts_view)
      parts.emplace_back(v.begin(), v.end());

    if (parts.size() <= 5) {
      is_valid_ = false;
      return;
    }

    if (parts[0] == CLANG_OFFLOAD_KIND_HIP || parts[0] == CLANG_OFFLOAD_KIND_HIPV4) {
      util::debug::print(__func__, ": Fatbin has target triple ", parts[5]);
      int64_t fatbin_offset = static_cast<int64_t>(fatbin_section.sectionOffset() + info.offset);

      auto co = std::make_unique<AmdGpuCodeObject>(info.size, elf_file, parts[0], parts[5],
                                                   fatbin_offset);
      if (!co->is_valid()) {
        is_valid_ = false;
        return;
      }
      register_code_object(std::move(co));
    } else if (parts[0] == CLANG_OFFLOAD_KIND_HOST) {
      continue;
    } else {
      is_valid_ = false;
      return;
    }
  }
}

void Executable::register_code_object(std::unique_ptr<AmdGpuCodeObject> co) {
  auto target = co->target_id();
  code_objs_by_target_[target].push_back(co.get());
  owned_code_objs_.push_back(std::move(co));
}

} // namespace rocjitsu
