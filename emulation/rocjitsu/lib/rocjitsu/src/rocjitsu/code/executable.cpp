// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/executable.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "util/log.h"

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
  uint64_t flags() const override { return shdr_.sh_flags; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

using detail::fits_in_bounds;
using detail::read_value;

} // namespace

Executable::Executable(const std::string &path) {
  try {
    image_ = detail::read_file_bytes(path);
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
    load_device_elf();
    return;
  }

  // x86 host ELF (fat binary with .hip_fatbin section).
  if (ehdr.e_ident[EI_OSABI] == ELFOSABI_NONE) {
    header_ = std::make_unique<HsaHeader>(ehdr);
    load_fat_binary();
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

void Executable::load_device_elf() {
  auto co = std::make_unique<AmdGpuCodeObject>(reinterpret_cast<const uint8_t *>(image_.data()),
                                               image_.size());
  if (!co->is_valid()) {
    is_valid_ = false;
    return;
  }
  register_code_object(std::move(co));
}

void Executable::load_fat_binary() {
  // Parse section headers to find .hip_fatbin.
  const int num_shdrs = header_->numSectionHeaders();
  if (num_shdrs < 0 ||
      !fits_in_bounds(header_->sectionHeaderOff(),
                      static_cast<uint64_t>(num_shdrs) * sizeof(Elf64_Shdr), image_.size())) {
    is_valid_ = false;
    return;
  }
  std::vector<Elf64_Shdr> section_hdrs(static_cast<size_t>(num_shdrs));
  std::memcpy(section_hdrs.data(), image_.data() + header_->sectionHeaderOff(),
              section_hdrs.size() * sizeof(Elf64_Shdr));

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
    if (shdr.sh_type == SHT_NULL)
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

    if (sec_name == ".hip_fatbin") {
      std::unique_ptr<char[]> sec_data(std::make_unique<char[]>(shdr.sh_size));
      std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
      auto section = std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr);
      load_hip_fatbin(*section);
      sections_.emplace_back(std::move(section));
      if (!is_valid_)
        return;
    }
  }
}

void Executable::load_hip_fatbin(const Section &fatbin_section) {
  ClangOffloadBundleHeader bundle_hdr;
  const char *fatbin_data = fatbin_section.data();
  const size_t fatbin_size = fatbin_section.size();
  size_t cursor = 0;
  if (!fits_in_bounds(cursor, sizeof(bundle_hdr.magic), fatbin_size)) {
    is_valid_ = false;
    return;
  }
  std::memcpy(bundle_hdr.magic, fatbin_data + cursor, sizeof(bundle_hdr.magic));
  cursor += sizeof(bundle_hdr.magic);

  if (std::memcmp(bundle_hdr.magic, CLANG_OFFLOAD_MAGIC_STR, CLANG_OFFLOAD_MAGIC_STR_SIZE)) {
    is_valid_ = false;
    return;
  }

  if (!read_value(fatbin_data, fatbin_size, cursor, bundle_hdr.num_code_objs)) {
    is_valid_ = false;
    return;
  }

  util::Logger::debug_print(__func__, ": Found .hip_fatbin with ", bundle_hdr.num_code_objs,
                            " code objects");

  std::vector<ClangOffloadBundleInfo> infos(bundle_hdr.num_code_objs);
  for (auto &info : infos) {
    if (!read_value(fatbin_data, fatbin_size, cursor, info.offset)) {
      is_valid_ = false;
      return;
    }
    if (!read_value(fatbin_data, fatbin_size, cursor, info.size)) {
      is_valid_ = false;
      return;
    }
    if (!read_value(fatbin_data, fatbin_size, cursor, info.bundle_entry_id_size)) {
      is_valid_ = false;
      return;
    }
    if (!fits_in_bounds(cursor, info.bundle_entry_id_size, fatbin_size)) {
      is_valid_ = false;
      return;
    }
    info.bundle_entry_id.resize(info.bundle_entry_id_size, '\0');
    std::memcpy(info.bundle_entry_id.data(), fatbin_data + cursor, info.bundle_entry_id_size);
    cursor += static_cast<size_t>(info.bundle_entry_id_size);

    auto parts_view = info.bundle_entry_id | std::views::split('-');
    std::vector<std::string> parts;
    for (const auto &v : parts_view)
      parts.emplace_back(v.begin(), v.end());

    if (parts.size() <= 5) {
      is_valid_ = false;
      return;
    }

    if (parts[0] == CLANG_OFFLOAD_KIND_HIP || parts[0] == CLANG_OFFLOAD_KIND_HIPV4) {
      util::Logger::debug_print(__func__, ": Fatbin has target triple ", parts[5]);
      const uint64_t fatbin_offset = fatbin_section.sectionOffset() + info.offset;
      if (!fits_in_bounds(fatbin_offset, info.size, image_.size())) {
        is_valid_ = false;
        return;
      }

      auto co = std::make_unique<AmdGpuCodeObject>(
          reinterpret_cast<const uint8_t *>(image_.data() + fatbin_offset),
          static_cast<size_t>(info.size), parts[0], parts[5]);
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
