// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file executable_test.cpp
/// @brief Malformed-input tests for executable and HIP fat-binary loading.

#include "rocjitsu/code/executable.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "scoped_temp.h"

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <limits>
#include <string_view>
#include <vector>

namespace rocjitsu {
namespace {

uint32_t add_elf_name(std::vector<uint8_t> &names, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(names.size());
  names.insert(names.end(), name.begin(), name.end());
  names.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

std::vector<uint8_t> make_standalone_amdgpu_elf() {
  constexpr uint64_t kPayloadOffset = 0x100;
  constexpr uint64_t kPayloadVaddr = 0x1000;
  constexpr uint32_t kPayload = 0xa5a55a5a;

  std::vector<uint8_t> names{'\0'};
  const uint32_t names_name = add_elf_name(names, ".shstrtab");
  const uint64_t names_offset = kPayloadOffset + sizeof(kPayload);
  const uint64_t section_offset = align_up(names_offset + names.size(), alignof(Elf64_Shdr));
  std::vector<uint8_t> image(section_offset + 2 * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  ehdr.e_type = ET_DYN;
  ehdr.e_machine = EM_AMDGPU;
  ehdr.e_version = 1;
  ehdr.e_phoff = sizeof(Elf64_Ehdr);
  ehdr.e_shoff = section_offset;
  ehdr.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
  ehdr.e_ehsize = sizeof(Elf64_Ehdr);
  ehdr.e_phentsize = sizeof(Elf64_Phdr);
  ehdr.e_phnum = 1;
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 2;
  ehdr.e_shstrndx = 1;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  Elf64_Phdr phdr{};
  phdr.p_type = PT_LOAD;
  phdr.p_offset = kPayloadOffset;
  phdr.p_vaddr = kPayloadVaddr;
  phdr.p_filesz = sizeof(kPayload);
  phdr.p_memsz = 2 * sizeof(kPayload);
  std::memcpy(image.data() + ehdr.e_phoff, &phdr, sizeof(phdr));
  std::memcpy(image.data() + kPayloadOffset, &kPayload, sizeof(kPayload));
  std::memcpy(image.data() + names_offset, names.data(), names.size());

  std::array<Elf64_Shdr, 2> shdrs{};
  shdrs[1].sh_name = names_name;
  shdrs[1].sh_type = SHT_STRTAB;
  shdrs[1].sh_offset = names_offset;
  shdrs[1].sh_size = names.size();
  shdrs[1].sh_addralign = 1;
  std::memcpy(image.data() + section_offset, shdrs.data(), sizeof(shdrs));
  return image;
}

struct RecordedLoad {
  std::size_t size;
  uint64_t address;
};

struct RecordingMemory {
  void load_image(const uint8_t *data, std::size_t size, uint64_t address) {
    EXPECT_NE(data, nullptr);
    loads.push_back({size, address});
  }

  std::vector<RecordedLoad> loads;
};

std::vector<uint8_t> make_host_elf_with_fatbin(const std::vector<uint8_t> &fatbin) {
  constexpr uint64_t kFatbinOffset = 0x100;
  std::vector<uint8_t> names{'\0'};
  const uint32_t fatbin_name = add_elf_name(names, ".hip_fatbin");
  const uint32_t names_name = add_elf_name(names, ".shstrtab");
  const uint64_t names_offset = kFatbinOffset + fatbin.size();
  const uint64_t section_offset = align_up(names_offset + names.size(), alignof(Elf64_Shdr));

  std::vector<uint8_t> image(section_offset + 3 * sizeof(Elf64_Shdr), 0);
  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_NONE;
  ehdr.e_machine = EM_X86_64;
  ehdr.e_version = 1;
  ehdr.e_shoff = section_offset;
  ehdr.e_ehsize = sizeof(ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  ehdr.e_shnum = 3;
  ehdr.e_shstrndx = 2;
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));

  if (!fatbin.empty())
    std::memcpy(image.data() + kFatbinOffset, fatbin.data(), fatbin.size());
  std::memcpy(image.data() + names_offset, names.data(), names.size());

  std::array<Elf64_Shdr, 3> shdrs{};
  shdrs[1].sh_name = fatbin_name;
  shdrs[1].sh_type = SHT_PROGBITS;
  shdrs[1].sh_offset = kFatbinOffset;
  shdrs[1].sh_size = fatbin.size();
  shdrs[1].sh_addralign = 8;
  shdrs[2].sh_name = names_name;
  shdrs[2].sh_type = SHT_STRTAB;
  shdrs[2].sh_offset = names_offset;
  shdrs[2].sh_size = names.size();
  shdrs[2].sh_addralign = 1;
  std::memcpy(image.data() + section_offset, shdrs.data(), sizeof(shdrs));
  return image;
}

std::vector<uint8_t> make_fatbin_header(uint64_t count) {
  constexpr std::array<uint8_t, 24> magic = {'_', '_', 'C', 'L', 'A', 'N', 'G', '_',
                                             'O', 'F', 'F', 'L', 'O', 'A', 'D', '_',
                                             'B', 'U', 'N', 'D', 'L', 'E', '_', '_'};
  std::vector<uint8_t> fatbin(magic.size() + sizeof(count));
  std::memcpy(fatbin.data(), magic.data(), magic.size());
  std::memcpy(fatbin.data() + magic.size(), &count, sizeof(count));
  return fatbin;
}

TEST(ExecutableInputValidation, RejectsEmptySectionTable) {
  std::vector<uint8_t> image(sizeof(Elf64_Ehdr), 0);
  Elf64_Ehdr ehdr{};
  std::memcpy(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  ehdr.e_ident[EI_CLASS] = ELFCLASS64;
  ehdr.e_ident[EI_OSABI] = ELFOSABI_NONE;
  ehdr.e_machine = EM_X86_64;
  ehdr.e_version = 1;
  ehdr.e_ehsize = sizeof(ehdr);
  ehdr.e_shentsize = sizeof(Elf64_Shdr);
  std::memcpy(image.data(), &ehdr, sizeof(ehdr));
  test::ScopedTempFile file("rj-empty-section-table-");
  file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));

  const Executable executable(file.path());
  EXPECT_FALSE(executable.is_valid());
}

TEST(ExecutableInputValidation, RejectsImpossibleBundleCountBeforeAllocation) {
  const std::vector<uint8_t> image =
      make_host_elf_with_fatbin(make_fatbin_header(std::numeric_limits<uint64_t>::max()));
  test::ScopedTempFile file("rj-bundle-count-");
  file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));

  const Executable executable(file.path());
  EXPECT_FALSE(executable.is_valid());
}

TEST(ExecutableInputValidation, RejectsBundlePayloadOutsideItsSection) {
  constexpr std::string_view kTarget = "hip-amdgcn-amd-amdhsa--gfx1250";
  auto fatbin = make_fatbin_header(1);
  const size_t info_offset = fatbin.size();
  fatbin.resize(fatbin.size() + 3 * sizeof(uint64_t));
  const uint64_t id_size = kTarget.size();
  std::memcpy(fatbin.data() + info_offset + 2 * sizeof(uint64_t), &id_size, sizeof(id_size));
  fatbin.insert(fatbin.end(), kTarget.begin(), kTarget.end());
  for (const bool wrapping : {false, true}) {
    SCOPED_TRACE(wrapping);
    auto image = make_host_elf_with_fatbin(fatbin);
    constexpr uint64_t kFatbinOffset = 0x100;
    uint64_t payload_offset, payload_size;
    if (wrapping) {
      // A valid minimal GPU ELF in the host's unused header padding. Adding
      // this crafted bundle offset to 0x100 wraps to the embedded ELF at 0x40.
      constexpr uint64_t kDeviceOffset = sizeof(Elf64_Ehdr);
      std::array<uint8_t, sizeof(Elf64_Ehdr) + sizeof(Elf64_Shdr)> device{};
      Elf64_Ehdr header{};
      std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
      header.e_ident[EI_CLASS] = ELFCLASS64;
      header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
      header.e_type = ET_DYN;
      header.e_machine = EM_AMDGPU;
      header.e_version = 1;
      header.e_flags = EF_AMDGPU_MACH_AMDGCN_GFX1250;
      header.e_ehsize = sizeof(Elf64_Ehdr);
      header.e_shoff = sizeof(Elf64_Ehdr);
      header.e_shentsize = sizeof(Elf64_Shdr);
      header.e_shnum = 1;
      std::memcpy(device.data(), &header, sizeof(header));
      std::memcpy(image.data() + kDeviceOffset, device.data(), device.size());
      payload_offset = kDeviceOffset - kFatbinOffset;
      payload_size = device.size();
    } else {
      // A valid GPU ELF remains inside the host image but past the fatbin's
      // declared extent. Image-only bounds checks would accept this payload.
      const auto device = make_standalone_amdgpu_elf();
      payload_offset = image.size() - kFatbinOffset;
      payload_size = device.size();
      image.insert(image.end(), device.begin(), device.end());
    }
    std::memcpy(image.data() + kFatbinOffset + info_offset, &payload_offset,
                sizeof(payload_offset));
    std::memcpy(image.data() + kFatbinOffset + info_offset + sizeof(uint64_t), &payload_size,
                sizeof(payload_size));
    test::ScopedTempFile file("rj-bundle-payload-");
    file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));
    const Executable executable(file.path());
    EXPECT_FALSE(executable.is_valid());
  }
}

TEST(ExecutableInputValidation, LoadsStandaloneDeviceElfByValidatedSegments) {
  const std::vector<uint8_t> image = make_standalone_amdgpu_elf();
  test::ScopedTempFile file("rj-standalone-device-elf-");
  file.write(std::string_view(reinterpret_cast<const char *>(image.data()), image.size()));

  const Executable executable(file.path());
  ASSERT_TRUE(executable.is_valid());
  ASSERT_EQ(executable.num_code_objects(ROCJITSU_CODE_TARGET_GFX1250), 1u);
  const AmdGpuCodeObject *code_object = executable.code_object(ROCJITSU_CODE_TARGET_GFX1250, 0);
  ASSERT_NE(code_object, nullptr);
  EXPECT_EQ(executable.mem_size(), code_object->mem_size());
  EXPECT_EQ(executable.mem_size(), 0x1008u);

  RecordingMemory memory;
  executable.load_to_memory(&memory, 0x4000);
  ASSERT_EQ(memory.loads.size(), 1u);
  EXPECT_EQ(memory.loads[0].size, sizeof(uint32_t));
  EXPECT_EQ(memory.loads[0].address, 0x5000u);
}

} // namespace
} // namespace rocjitsu
