// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "../dbi_test_util.h"
#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/builders/instruction_builder.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/target_registry.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

namespace rocjitsu::test {

inline uint32_t checked_encoding(const FailureOr<uint32_t> &encoding) {
  EXPECT_TRUE(encoding.succeeded());
  if (encoding.failed())
    return 0;
  return encoding.value();
}

constexpr uint32_t kSignal = 0xbe804ec1u;
constexpr uint32_t kWait = 0xbf94ffffu;
constexpr uint32_t kWaitDscnt = 0xbfc60000u;
constexpr uint32_t kMov = 0xbe800081u;

struct DropTargetRow {
  const char *name;
  rj_code_target_id_t target;
  rj_code_arch_t arch;
  uint32_t wave_size;
  bool split;
};
constexpr DropTargetRow kRdna3DropRows[] = {
    {.name = "Gfx1100Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 32,
     .split = false},
    {.name = "Gfx1100Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1100,
     .arch = ROCJITSU_CODE_ARCH_RDNA3,
     .wave_size = 64,
     .split = false},
};
constexpr DropTargetRow kDropRows[] = {
    {.name = "Gfx1200Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1200Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1200,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1201Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1201Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX1201,
     .arch = ROCJITSU_CODE_ARCH_RDNA4,
     .wave_size = 64,
     .split = true},
    {.name = "Gfx1250Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1250,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx1251Wave32",
     .target = ROCJITSU_CODE_TARGET_GFX1251,
     .arch = ROCJITSU_CODE_ARCH_CDNA5,
     .wave_size = 32,
     .split = true},
    {.name = "Gfx942Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX942,
     .arch = ROCJITSU_CODE_ARCH_CDNA3,
     .wave_size = 64,
     .split = false},
    {.name = "Gfx950Wave64",
     .target = ROCJITSU_CODE_TARGET_GFX950,
     .arch = ROCJITSU_CODE_ARCH_CDNA4,
     .wave_size = 64,
     .split = false},
    kRdna3DropRows[0],
    kRdna3DropRows[1],
};

inline std::vector<uint32_t> make_barrier_words(const DropTargetRow &row) {
  // RDNA3 generated test_encodings.h: s_waitcnt and full s_barrier differ
  // from CDNA. Offset 4/16 barriers surround unchanged waits and s_mov_b32.
  if (row.arch == ROCJITSU_CODE_ARCH_RDNA3)
    return {0xbf890000u, 0xbfbd0000u, kMov,
            0xbf890000u, 0xbfbd0000u, checked_encoding(build_s_endpgm(row.arch))};
  const uint32_t wait = row.split ? kWaitDscnt : 0xbf8c0000u;
  if (row.split)
    return {wait,  kSignal, kMov,
            kWait, wait,    kSignal,
            kMov,  kWait,   checked_encoding(build_s_endpgm(row.arch))};
  return {wait, 0xbf8a0000u, kMov, wait, 0xbf8a0000u, checked_encoding(build_s_endpgm(row.arch))};
}

inline std::vector<uint8_t> make_barrier_elf_for_target(const DropTargetRow &row,
                                                        std::vector<uint32_t> words,
                                                        bool siblings = false) {
  constexpr uint32_t kSecondEntryByteOffset = 256;
  const IsaGpuTargetDescription *binding =
      default_isa_target_registry().find_gpu_target(row.target);
  if (!binding)
    return {};
  if (siblings) {
    const std::vector<uint32_t> body = words;
    words.resize(kSecondEntryByteOffset / sizeof(uint32_t),
                 checked_encoding(build_s_nop(0, row.arch)));
    words.insert(words.end(), body.begin(), body.end());
  }
  std::vector<uint8_t> image =
      siblings ? test::make_gfx950_two_kernel_elf(words, 0)
               : test::make_amdgpu_kernel_elf(words, 0, 3, binding->elf_machine);
  Elf64_Ehdr header = {};
  std::memcpy(&header, image.data(), sizeof(header));
  header.e_ident[EI_DATA] = ELFDATA2LSB;
  header.e_ident[EI_ABIVERSION] = ELFABIVERSION_AMDGPU_HSA_V5;
  header.e_flags = binding->elf_machine;
  std::memcpy(image.data(), &header, sizeof(header));
  const AmdGpuCodeObject object(image.data(), image.size());
  const Section &text = *object.text_sections().front();
  const std::vector<KernelDescriptorInfo> descriptors =
      scan_kernel_descriptors(image, text.sectionOffset(), text.size(), text.sectionHeaderIndex());
  for (const KernelDescriptorInfo &kernel : descriptors) {
    amdhsa::kernel_descriptor_t descriptor = kernel.descriptor;
    if (row.wave_size == 32)
      descriptor.kernel_code_properties |= amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    else
      descriptor.kernel_code_properties &= ~amdhsa::KERNEL_CODE_PROPERTY_ENABLE_WAVEFRONT_SIZE32;
    if (siblings && kernel.kernel_name == "test_kernel1")
      descriptor.kernel_code_entry_byte_offset += kSecondEntryByteOffset;
    std::memcpy(image.data() + kernel.descriptor_file_offset, &descriptor, sizeof(descriptor));
  }
  return image;
}

} // namespace rocjitsu::test
