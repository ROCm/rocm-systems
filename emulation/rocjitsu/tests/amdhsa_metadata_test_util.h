// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/code_object_patcher.h"

#include <array>
#include <cassert>
#include <cstring>
#include <initializer_list>
#include <string>
#include <string_view>
#include <vector>

namespace rocjitsu::test {

struct MetadataKernel {
  std::string_view name;
  uint32_t kernarg_bytes;
  uint32_t sgprs;
  uint32_t vgprs;
  uint32_t private_bytes = 4096;
};

inline void append_metadata_text(std::vector<uint8_t> &bytes, std::string_view text) {
  assert(text.size() < 32);
  bytes.push_back(static_cast<uint8_t>(0xa0 | text.size()));
  bytes.insert(bytes.end(), text.begin(), text.end());
}

inline void append_metadata_integer(std::vector<uint8_t> &bytes, uint32_t value, bool narrow) {
  if (narrow && value < 128) {
    bytes.push_back(static_cast<uint8_t>(value));
    return;
  }
  bytes.push_back(0xce);
  for (unsigned shift : {24u, 16u, 8u, 0u})
    bytes.push_back(static_cast<uint8_t>(value >> shift));
}

inline std::vector<uint8_t> resource_metadata_payload(std::initializer_list<MetadataKernel> kernels,
                                                      bool narrow = false) {
  assert(kernels.size() < 16);
  std::vector<uint8_t> bytes{0x83};
  append_metadata_text(bytes, "amdhsa.version");
  bytes.insert(bytes.end(), {0x92, 1, 2});
  append_metadata_text(bytes, "amdhsa.kernels");
  bytes.push_back(static_cast<uint8_t>(0x90 | kernels.size()));
  for (const MetadataKernel &kernel : kernels) {
    bytes.push_back(0x86);
    append_metadata_text(bytes, ".name");
    append_metadata_text(bytes, kernel.name);
    append_metadata_text(bytes, ".symbol");
    append_metadata_text(bytes, std::string(kernel.name) + ".kd");
    append_metadata_text(bytes, ".kernarg_segment_size");
    append_metadata_integer(bytes, kernel.kernarg_bytes, narrow);
    append_metadata_text(bytes, ".sgpr_count");
    append_metadata_integer(bytes, kernel.sgprs, narrow);
    append_metadata_text(bytes, ".vgpr_count");
    append_metadata_integer(bytes, kernel.vgprs, narrow);
    // Keep a nonzero scratch baseline to check that unrelated kernarg/GPR edits
    // preserve the existing private-segment metadata.
    append_metadata_text(bytes, ".private_segment_fixed_size");
    append_metadata_integer(bytes, kernel.private_bytes, narrow);
  }
  append_metadata_text(bytes, "vendor.extra");
  append_metadata_text(bytes, "unchanged");
  return bytes;
}

inline std::vector<uint8_t> with_metadata_note(const std::vector<uint8_t> &original,
                                               std::span<const uint8_t> payload) {
  std::vector<uint8_t> note;
  const std::array<uint32_t, 3> framing{7, static_cast<uint32_t>(payload.size()),
                                        NT_AMDGPU_METADATA};
  note.resize(sizeof(framing));
  std::memcpy(note.data(), framing.data(), sizeof(framing));
  note.insert(note.end(), {'A', 'M', 'D', 'G', 'P', 'U', 0, 0});
  note.insert(note.end(), payload.begin(), payload.end());
  while (note.size() % 4 != 0)
    note.push_back(0);
  const AmdGpuCodeObject object(original.data(), original.size());
  CodeObjectPatcher patcher(object);
  if (!patcher.append_nonalloc_section(".note.amdgpu", note, 4))
    return {};
  std::vector<uint8_t> image = patcher.emit();
  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  const std::optional<std::vector<Elf64_Shdr>> sections =
      read_elf_table<Elf64_Shdr>(image, header.e_shoff, header.e_shnum, header.e_shentsize);
  if (!sections || sections->empty())
    return {};
  Elf64_Shdr section = sections->back();
  section.sh_type = SHT_NOTE;
  std::memcpy(image.data() + header.e_shoff + (header.e_shnum - 1) * sizeof(section), &section,
              sizeof(section));
  return image;
}

inline std::vector<uint8_t> with_resource_metadata(const std::vector<uint8_t> &image,
                                                   std::initializer_list<MetadataKernel> kernels,
                                                   bool narrow = false) {
  return with_metadata_note(image, resource_metadata_payload(kernels, narrow));
}

} // namespace rocjitsu::test
