/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_code.hpp"

#include <cstring>
#include <elf.h>

#include "core/inc/amd_aie_section.h"
#include "core/inc/amd_elf_image.hpp"

namespace rocr {
namespace AMD {

namespace {
constexpr const char* kArchSectionNames[] = {"aie2", "aie2p"};

// Returns the arch section (by name) if present, else nullptr, and sets out_name.
amd::elf::Section* FindArchSection(amd::elf::Image* elf, std::string* out_name) {
  for (size_t i = 0; i < elf->sectionCount(); ++i) {
    amd::elf::Section* sec = elf->section(i);
    if (!sec) continue;
    const std::string name = sec->Name();
    for (const char* arch : kArchSectionNames) {
      if (name == arch) {
        *out_name = name;
        return sec;
      }
    }
  }
  return nullptr;
}
}  // namespace

bool AieCode::IsAieCodeObject(const void* data, size_t size) {
  if (!data || size < sizeof(Elf64_Ehdr)) return false;
  const auto* ehdr = static_cast<const Elf64_Ehdr*>(data);
  if (memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0) return false;

  auto img = std::unique_ptr<amd::elf::Image>(amd::elf::NewElf64Image());
  if (!img || !img->initAsBuffer(data, size)) return false;
  std::string name;
  return FindArchSection(img.get(), &name) != nullptr;
}

std::unique_ptr<AieCode> AieCode::Create(const void* data, size_t size) {
  if (!data || size == 0) return nullptr;
  auto code = std::unique_ptr<AieCode>(new AieCode());
  code->elf_base_ = static_cast<const uint8_t*>(data);
  code->elf_size_ = size;
  code->elf_.reset(amd::elf::NewElf64Image());
  // initAsBuffer keeps a pointer into the caller's buffer (no copy), which is
  // required since AieKernelInfo::insts_data/pdi_data point into that buffer.
  if (!code->elf_ || !code->elf_->initAsBuffer(data, size)) return nullptr;
  if (!code->Parse()) return nullptr;
  return code;
}

bool AieCode::Parse() {
  amd::elf::Section* sec = FindArchSection(elf_.get(), &arch_section_name_);
  if (!sec) return false;

  section_size_ = sec->size();
  if (section_size_ < sizeof(aie_section_header)) return false;
  // Section has no direct data() accessor; compute the pointer into the ELF
  // buffer from the section's file offset. Validate the section's [offset, size)
  // lies within the caller's buffer before dereferencing — sec->offset()/size()
  // come straight from the (possibly malformed) section header.
  const uint64_t sec_offset = sec->offset();
  if (sec_offset > elf_size_ || section_size_ > elf_size_ - sec_offset) return false;
  section_base_ = elf_base_ + sec_offset;

  const auto* hdr = reinterpret_cast<const aie_section_header*>(section_base_);
  if (hdr->magic != kAieSectionMagic) return false;
  if (hdr->version_major != kAieSectionVersionMajor) return false;
  if (hdr->header_size + static_cast<uint64_t>(hdr->kernel_count) * hdr->kernel_entry_size >
      section_size_) {
    return false;
  }
  if (hdr->kernel_entry_size < sizeof(aie_kernel_entry)) return false;

  auto in_section = [&](uint64_t off, uint64_t len) {
    return len == 0 ? off <= section_size_ : (off < section_size_ && off + len <= section_size_);
  };

  for (uint32_t i = 0; i < hdr->kernel_count; ++i) {
    const auto* e = reinterpret_cast<const aie_kernel_entry*>(
        section_base_ + hdr->header_size + static_cast<uint64_t>(i) * hdr->kernel_entry_size);

    if (e->insts_size == 0) return false;
    // Instructions are 32-bit words; the driver submits insts_size / 4 as the
    // dword count, so a non-multiple-of-4 size would silently truncate the stream.
    if (e->insts_size % sizeof(uint32_t) != 0) return false;
    if (!in_section(e->insts_offset, e->insts_size)) return false;
    if (e->pdi_size != 0 && !in_section(e->pdi_offset, e->pdi_size)) return false;
    if (e->pdi_size == 0 && e->pdi_offset != 0) return false;  // PDI absent iff both are 0

    const uint64_t name_abs = static_cast<uint64_t>(hdr->string_table_offset) + e->name_offset;
    if (name_abs >= section_size_) return false;
    const char* nm = reinterpret_cast<const char*>(section_base_ + name_abs);
    const uint64_t max_len = section_size_ - name_abs;
    if (::strnlen(nm, max_len) == max_len) return false;  // unterminated

    AieKernelInfo info;
    info.name = nm;
    info.insts_data = section_base_ + e->insts_offset;
    info.insts_size = e->insts_size;
    info.pdi_data = e->pdi_size ? section_base_ + e->pdi_offset : nullptr;
    info.pdi_size = e->pdi_size;
    info.kernarg_size = e->kernarg_size;
    info.num_cols = e->num_cols;
    if (kernels_.count(info.name)) return false;  // duplicate within one object
    kernels_[info.name] = info;
  }
  return !kernels_.empty();
}

std::vector<std::string> AieCode::GetKernelNames() const {
  std::vector<std::string> names;
  names.reserve(kernels_.size());
  for (const auto& kv : kernels_) names.push_back(kv.first);
  return names;
}

const AieKernelInfo* AieCode::GetKernel(const std::string& name) const {
  auto it = kernels_.find(name);
  return it == kernels_.end() ? nullptr : &it->second;
}

}  // namespace AMD
}  // namespace rocr
