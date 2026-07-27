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
/// @brief Returns the arch section if present, else @c nullptr, and sets out_name if not
/// @c nullptr.
///
/// The section is identified structurally by its @ref aie_section_header magic, not by a
/// hardcoded name allowlist: the section's name IS the arch name, and which arch names are
/// acceptable is the AIE agent's decision (validated in the loader against AieAgent::arch_name),
/// not the parser's. This keeps a single source of truth for the accepted arch.
///
/// @param elf ELF image to search (buffer-backed via initAsBuffer).
/// @param out_name Set to the matched section's name (the arch name) on success.
/// @return The matched section, or @c nullptr if no section carries a valid AIE header.
amd::elf::Section* FindArchSection(amd::elf::Image* elf, std::string* out_name) {
  const auto* elf_base = reinterpret_cast<const uint8_t*>(elf->data());
  const uint64_t elf_size = elf->size();
  for (size_t i = 0; i < elf->sectionCount(); ++i) {
    amd::elf::Section* sec = elf->section(i);
    if (!sec) continue;
    const uint64_t off = sec->offset();
    const uint64_t sz = sec->size();
    // The magic lives at the section start; require the header to fit within the buffer before
    // reading it, since offset()/size() come from the (possibly malformed) section header.
    if (sz < sizeof(aie_section_header)) continue;
    if (off > elf_size || sz > elf_size - off) continue;
    if (reinterpret_cast<const aie_section_header*>(elf_base + off)->magic != kAieSectionMagic)
      continue;
    if (out_name) {
      *out_name = sec->Name();
    }
    return sec;
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
  return FindArchSection(img.get(), nullptr) != nullptr;
}

std::unique_ptr<AieCode> AieCode::Create(const void* data, size_t size) {
  if (!data || size == 0) return nullptr;
  auto code = std::unique_ptr<AieCode>(new AieCode());
  code->elf_.reset(amd::elf::NewElf64Image());
  // initAsBuffer keeps a pointer into the caller's buffer (no copy), which is
  // required since AieKernelInfo::insts_data/pdi_data point into that buffer.
  if (!code->elf_ || !code->elf_->initAsBuffer(data, size)) return nullptr;
  if (!code->Parse()) return nullptr;
  return code;
}

bool AieCode::Parse() {
  // FindArchSection has already bounds-checked the section header and matched the magic.
  amd::elf::Section* sec = FindArchSection(elf_.get(), &arch_section_name_);
  if (!sec) return false;

  section_size_ = sec->size();
  section_base_ = reinterpret_cast<const uint8_t*>(elf_->data()) + sec->offset();

  const auto* hdr = reinterpret_cast<const aie_section_header*>(section_base_);
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
