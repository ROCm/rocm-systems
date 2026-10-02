/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "core/inc/amd_aie_code.hpp"

#include <cstring>
#include <libelf.h>

#include "core/inc/amd_aie_section.h"
#include "core/inc/amd_elf_image.hpp"

namespace rocr {
namespace AMD {

namespace {
/// @brief Returns the AIE section for @p arch if present, else @c nullptr.
///
/// The section is identified structurally by its @ref aie_section_header magic, not by a
/// hardcoded name allowlist: the section's name IS the arch name, and which arch names are
/// acceptable is the AIE agent's decision (the caller passes AieAgent::arch_name), not the
/// parser's. This keeps a single source of truth for the accepted arch.
///
/// One hsaco can carry a section per arch, so the search continues past an AIE section for a
/// different arch rather than stopping at the first one.
///
/// @param elf ELF image to search (buffer-backed via initAsBuffer).
/// @param arch Arch name to match; empty matches any AIE section.
/// @return The matched section, or @c nullptr if no section carries a valid AIE header and the
/// name @p arch.
amd::elf::Section* FindArchSection(amd::elf::Image* elf, std::string_view arch) {
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
    if (!arch.empty() && sec->Name() != arch) continue;
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
  return FindArchSection(img.get(), {}) != nullptr;
}

hsa_status_t AieCode::Create(const void* data, size_t size, std::string_view arch,
                             std::unique_ptr<AieCode>* out) {
  if (!out) return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  if (!data || size == 0 || arch.empty()) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  auto code = std::unique_ptr<AieCode>(new AieCode());
  code->elf_.reset(amd::elf::NewElf64Image());
  // initAsBuffer keeps a pointer into the caller's buffer (no copy), which is
  // required since AieKernelInfo::insts_data/pdi_data point into that buffer.
  if (!code->elf_ || !code->elf_->initAsBuffer(data, size)) {
    return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  }
  // FindArchSection bounds-checks the section header and matches the magic.
  amd::elf::Section* sec = FindArchSection(code->elf_.get(), arch);
  if (!sec) return HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS;
  code->arch_section_name_ = arch;
  if (!code->Parse(sec)) return HSA_STATUS_ERROR_INVALID_CODE_OBJECT;
  *out = std::move(code);
  return HSA_STATUS_SUCCESS;
}

bool AieCode::Parse(amd::elf::Section* sec) {
  const uint64_t section_size = sec->size();
  const uint8_t* const section_base =
      reinterpret_cast<const uint8_t*>(elf_->data()) + sec->offset();

  const auto* hdr = reinterpret_cast<const aie_section_header*>(section_base);
  if (hdr->version_major != kAieSectionVersionMajor) return false;
  if (hdr->header_size < sizeof(aie_section_header)) return false;
  if (hdr->kernel_entry_size < sizeof(aie_kernel_entry)) return false;

  // The regions must follow each other in the declared order, not merely lie within the section:
  // blob_pool_offset is the lower bound the blob checks below trust, and an overlapping string
  // table would read kernel-table bytes as names. Same rules as mlir-aie's aie-hsaco-dump.
  const uint64_t table_end =
      hdr->header_size + static_cast<uint64_t>(hdr->kernel_count) * hdr->kernel_entry_size;
  const uint64_t string_table_end =
      static_cast<uint64_t>(hdr->string_table_offset) + hdr->string_table_size;
  if (table_end > section_size) return false;
  if (hdr->string_table_offset < table_end) return false;
  if (string_table_end > section_size) return false;
  if (hdr->blob_pool_offset < string_table_end) return false;
  if (hdr->blob_pool_offset > section_size) return false;

  // A blob bounded only from above could point back at the metadata and still parse.
  auto in_pool = [&](uint64_t off, uint64_t len) {
    return off >= hdr->blob_pool_offset && off + len <= section_size;
  };

  for (uint32_t i = 0; i < hdr->kernel_count; ++i) {
    const auto* e = reinterpret_cast<const aie_kernel_entry*>(
        section_base + hdr->header_size + static_cast<uint64_t>(i) * hdr->kernel_entry_size);

    if (e->insts_size == 0) return false;
    // For PdiInsts, instructions are 32-bit words and the driver submits
    // insts_size / 4 as the dword count, so a non-multiple-of-4 size would
    // silently truncate the stream. FullElf's insts_* instead locate a nested
    // ELF, which carries no such alignment constraint.
    if (e->kind == AieKernelKind::PdiInsts && e->insts_size % sizeof(uint32_t) != 0) return false;
    if (!in_pool(e->insts_offset, e->insts_size)) return false;
    if (e->pdi_size != 0 && !in_pool(e->pdi_offset, e->pdi_size)) return false;
    if (e->pdi_size == 0 && e->pdi_offset != 0) return false;  // PDI absent iff both are 0
    // A full ELF carries its own PDI; a separate one would never be loaded.
    if (e->kind == AieKernelKind::FullElf && e->pdi_size != 0) return false;

    // Bounded by the string table, not the section: an offset past the table would otherwise read
    // a name out of the blob pool.
    if (e->name_offset >= hdr->string_table_size) return false;
    const char* nm =
        reinterpret_cast<const char*>(section_base + hdr->string_table_offset + e->name_offset);
    const uint64_t max_len = hdr->string_table_size - e->name_offset;
    if (::strnlen(nm, max_len) == max_len) return false;  // unterminated

    AieKernelInfo info;
    info.name = nm;
    info.insts_data = section_base + e->insts_offset;
    info.insts_size = e->insts_size;
    info.pdi_data = e->pdi_size ? section_base + e->pdi_offset : nullptr;
    info.pdi_size = e->pdi_size;
    info.kernarg_size = e->kernarg_size;
    info.num_cols = e->num_cols;
    info.kind = e->kind;
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
