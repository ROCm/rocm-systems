// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "code_object_fuzz_core.h"

#include "rocjitsu/code/amdgpu_code_object.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/executable.h"
#include "rocjitsu/code/kernel_descriptor_scan.h"
#include "rocjitsu/isa/target_registry.h"

#include "rocjitsu/base/rj_compiler.h"
RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string_view>

namespace rocjitsu::code_object_fuzz {
namespace {

class SplitMix64 {
public:
  explicit SplitMix64(uint64_t seed) : state_(seed) {}

  uint64_t next() {
    uint64_t value = (state_ += 0x9e3779b97f4a7c15ull);
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ull;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebull;
    return value ^ (value >> 31);
  }

  std::size_t below(std::size_t bound) {
    return bound == 0 ? 0 : static_cast<std::size_t>(next() % bound);
  }

private:
  uint64_t state_;
};

uint32_t add_name(std::vector<uint8_t> &table, std::string_view name) {
  const uint32_t offset = static_cast<uint32_t>(table.size());
  table.insert(table.end(), name.begin(), name.end());
  table.push_back('\0');
  return offset;
}

uint64_t align_up(uint64_t value, uint64_t alignment) {
  const uint64_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

template <typename T> void write_at(std::vector<uint8_t> &image, uint64_t offset, const T &value) {
  if (offset > image.size() || sizeof(value) > image.size() - offset)
    throw std::logic_error("code-object fuzz seed write exceeds image");
  std::memcpy(image.data() + offset, &value, sizeof(value));
}

void write_bytes(std::vector<uint8_t> &image, uint64_t offset, const void *data, std::size_t size) {
  if (offset > image.size() || size > image.size() - offset)
    throw std::logic_error("code-object fuzz seed write exceeds image");
  if (size != 0)
    std::memcpy(image.data() + offset, data, size);
}

std::vector<uint8_t> make_device_elf(uint32_t machine_flag) {
  using KernelDescriptor = rocr::llvm::amdhsa::kernel_descriptor_t;
  constexpr uint64_t kTextOffset = 0x100;
  constexpr uint64_t kTextAddress = 0x1100;
  constexpr uint64_t kRodataAddress = 0x3000;
  constexpr std::array<uint32_t, 2> kText = {0xbf800000u, 0xbf810000u};

  std::vector<uint8_t> section_names{'\0'};
  const uint32_t text_name = add_name(section_names, ".text");
  const uint32_t rodata_name = add_name(section_names, ".rodata");
  const uint32_t symtab_name = add_name(section_names, ".symtab");
  const uint32_t strtab_name = add_name(section_names, ".strtab");
  const uint32_t shstrtab_name = add_name(section_names, ".shstrtab");

  std::vector<uint8_t> symbol_names{'\0'};
  const uint32_t kernel_name = add_name(symbol_names, "fuzz_kernel.kd");

  const uint64_t text_size = sizeof(kText);
  const uint64_t rodata_offset = align_up(kTextOffset + text_size, 8);
  const uint64_t strtab_offset = rodata_offset + sizeof(KernelDescriptor);
  const uint64_t symtab_offset = align_up(strtab_offset + symbol_names.size(), 8);
  constexpr uint64_t kSymbolCount = 2;
  const uint64_t shstrtab_offset = symtab_offset + kSymbolCount * sizeof(Elf64_Sym);
  const uint64_t section_offset = align_up(shstrtab_offset + section_names.size(), 8);
  constexpr uint16_t kSectionCount = 6;
  std::vector<uint8_t> image(section_offset + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = 1;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_OSABI] = ELFOSABI_AMDGPU_HSA;
  header.e_type = ET_DYN;
  header.e_machine = EM_AMDGPU;
  header.e_version = 1;
  header.e_phoff = sizeof(Elf64_Ehdr);
  header.e_shoff = section_offset;
  header.e_flags = machine_flag;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_phentsize = sizeof(Elf64_Phdr);
  header.e_phnum = 2;
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 5;
  write_at(image, 0, header);

  std::array<Elf64_Phdr, 2> programs{};
  programs[0].p_type = PT_LOAD;
  programs[0].p_flags = 0x5; // PF_R | PF_X
  programs[0].p_offset = kTextOffset;
  programs[0].p_vaddr = kTextAddress;
  programs[0].p_paddr = kTextAddress;
  programs[0].p_filesz = text_size;
  programs[0].p_memsz = text_size;
  programs[0].p_align = 0x1000;
  programs[1].p_type = PT_LOAD;
  programs[1].p_flags = 0x4; // PF_R
  programs[1].p_offset = rodata_offset;
  programs[1].p_vaddr = kRodataAddress;
  programs[1].p_paddr = kRodataAddress;
  programs[1].p_filesz = sizeof(KernelDescriptor);
  programs[1].p_memsz = sizeof(KernelDescriptor);
  programs[1].p_align = 0x1000;
  write_bytes(image, header.e_phoff, programs.data(), sizeof(programs));
  write_bytes(image, kTextOffset, kText.data(), sizeof(kText));

  KernelDescriptor descriptor{};
  descriptor.kernel_code_entry_byte_offset =
      static_cast<int64_t>(kTextAddress) - static_cast<int64_t>(kRodataAddress);
  write_at(image, rodata_offset, descriptor);
  write_bytes(image, strtab_offset, symbol_names.data(), symbol_names.size());

  std::array<Elf64_Sym, kSymbolCount> symbols{};
  symbols[1].st_name = kernel_name;
  symbols[1].st_info = elf_symbol_info(kElfSymbolBindGlobal, kElfSymbolTypeObject);
  symbols[1].st_shndx = 2;
  symbols[1].st_value = kRodataAddress;
  symbols[1].st_size = sizeof(KernelDescriptor);
  write_bytes(image, symtab_offset, symbols.data(), sizeof(symbols));
  write_bytes(image, shstrtab_offset, section_names.data(), section_names.size());

  std::array<Elf64_Shdr, kSectionCount> sections{};
  sections[1].sh_name = text_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_flags = SHF_ALLOC | SHF_EXECINSTR;
  sections[1].sh_addr = kTextAddress;
  sections[1].sh_offset = kTextOffset;
  sections[1].sh_size = text_size;
  sections[1].sh_addralign = sizeof(uint32_t);
  sections[2].sh_name = rodata_name;
  sections[2].sh_type = SHT_PROGBITS;
  sections[2].sh_flags = SHF_ALLOC;
  sections[2].sh_addr = kRodataAddress;
  sections[2].sh_offset = rodata_offset;
  sections[2].sh_size = sizeof(KernelDescriptor);
  sections[2].sh_addralign = 8;
  sections[3].sh_name = symtab_name;
  sections[3].sh_type = SHT_SYMTAB;
  sections[3].sh_offset = symtab_offset;
  sections[3].sh_size = sizeof(symbols);
  sections[3].sh_link = 4;
  sections[3].sh_info = 1;
  sections[3].sh_addralign = 8;
  sections[3].sh_entsize = sizeof(Elf64_Sym);
  sections[4].sh_name = strtab_name;
  sections[4].sh_type = SHT_STRTAB;
  sections[4].sh_offset = strtab_offset;
  sections[4].sh_size = symbol_names.size();
  sections[4].sh_addralign = 1;
  sections[5].sh_name = shstrtab_name;
  sections[5].sh_type = SHT_STRTAB;
  sections[5].sh_offset = shstrtab_offset;
  sections[5].sh_size = section_names.size();
  sections[5].sh_addralign = 1;
  write_bytes(image, section_offset, sections.data(), sizeof(sections));
  return image;
}

std::vector<uint8_t> make_host_bundle(std::span<const uint8_t> device, std::string_view bundle_id) {
  constexpr std::string_view kMagic = "__CLANG_OFFLOAD_BUNDLE__";
  const uint64_t bundle_header_size =
      kMagic.size() + sizeof(uint64_t) + 3 * sizeof(uint64_t) + bundle_id.size();
  std::vector<uint8_t> bundle(bundle_header_size + device.size(), 0);
  uint64_t cursor = 0;
  write_bytes(bundle, cursor, kMagic.data(), kMagic.size());
  cursor += kMagic.size();
  const uint64_t count = 1;
  write_at(bundle, cursor, count);
  cursor += sizeof(count);
  const uint64_t object_offset = bundle_header_size;
  const uint64_t object_size = device.size();
  const uint64_t id_size = bundle_id.size();
  write_at(bundle, cursor, object_offset);
  cursor += sizeof(object_offset);
  write_at(bundle, cursor, object_size);
  cursor += sizeof(object_size);
  write_at(bundle, cursor, id_size);
  cursor += sizeof(id_size);
  write_bytes(bundle, cursor, bundle_id.data(), bundle_id.size());
  write_bytes(bundle, object_offset, device.data(), device.size());

  std::vector<uint8_t> names{'\0'};
  const uint32_t fatbin_name = add_name(names, ".hip_fatbin");
  const uint32_t shstrtab_name = add_name(names, ".shstrtab");
  constexpr uint64_t kBundleOffset = 0x100;
  const uint64_t names_offset = kBundleOffset + bundle.size();
  const uint64_t section_offset = align_up(names_offset + names.size(), 8);
  constexpr uint16_t kSectionCount = 3;
  std::vector<uint8_t> image(section_offset + kSectionCount * sizeof(Elf64_Shdr), 0);

  Elf64_Ehdr header{};
  std::memcpy(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE);
  header.e_ident[EI_CLASS] = ELFCLASS64;
  header.e_ident[EI_DATA] = 1;
  header.e_ident[EI_VERSION] = 1;
  header.e_ident[EI_OSABI] = ELFOSABI_NONE;
  header.e_type = ET_DYN;
  header.e_machine = EM_X86_64;
  header.e_version = 1;
  header.e_shoff = section_offset;
  header.e_ehsize = sizeof(Elf64_Ehdr);
  header.e_shentsize = sizeof(Elf64_Shdr);
  header.e_shnum = kSectionCount;
  header.e_shstrndx = 2;
  write_at(image, 0, header);
  write_bytes(image, kBundleOffset, bundle.data(), bundle.size());
  write_bytes(image, names_offset, names.data(), names.size());

  std::array<Elf64_Shdr, kSectionCount> sections{};
  sections[1].sh_name = fatbin_name;
  sections[1].sh_type = SHT_PROGBITS;
  sections[1].sh_offset = kBundleOffset;
  sections[1].sh_size = bundle.size();
  sections[1].sh_addralign = 8;
  sections[2].sh_name = shstrtab_name;
  sections[2].sh_type = SHT_STRTAB;
  sections[2].sh_offset = names_offset;
  sections[2].sh_size = names.size();
  sections[2].sh_addralign = 1;
  write_bytes(image, section_offset, sections.data(), sizeof(sections));
  return image;
}

class LoadSink {
public:
  explicit LoadSink(std::span<const uint8_t> image) : image_(image) {}

  void load_image(const uint8_t *data, std::size_t size, uint64_t address) {
    if (!load_segment_in_bounds(image_, reinterpret_cast<std::uintptr_t>(data), size)) {
      throw std::logic_error("code object exposed an out-of-bounds load segment");
    }
    if (size != 0) {
      checksum_ ^= data[0];
      checksum_ ^= data[size - 1];
    }
    checksum_ ^= address;
    ++loads_;
  }

  uint64_t loads() const { return loads_; }
  uint64_t checksum() const { return checksum_; }

private:
  std::span<const uint8_t> image_;
  uint64_t loads_ = 0;
  uint64_t checksum_ = 0;
};

void inspect_object(const AmdGpuCodeObject &object, RunStats &stats) {
  if (!object.is_valid())
    return;
  ++stats.code_objects_valid;
  stats.sections += object.all_sections().size();
  stats.text_sections += object.text_sections().size();

  const std::span<const uint8_t> image(reinterpret_cast<const uint8_t *>(object.image_data()),
                                       object.image_size());
  for (const Section *section : object.text_sections()) {
    const std::vector<KernelDescriptorInfo> found = scan_kernel_descriptors(
        image, section->sectionOffset(), section->size(), section->sectionHeaderIndex());
    stats.descriptors += found.size();
  }
  // Exercise metadata queries whose results are not otherwise needed by the
  // loader campaign.
  const IsaTargetDescriptor *target = default_isa_target_registry().find(object.target_id());
  if (target != nullptr)
    (void)object.min_kernel_sgpr_count(target->architecture_id);
  (void)object.mem_size();
  LoadSink sink(image);
  object.load_to_memory(&sink, 0x100000);
  stats.load_segments += sink.loads();
  // Keep the memory reads observable in optimized sanitizer campaign builds.
  volatile uint64_t checksum = sink.checksum();
  (void)checksum;
}

void inspect_executable(std::span<const uint8_t> bytes, RunStats &stats) {
  Executable executable(bytes);
  if (!executable.is_valid())
    return;
  ++stats.executables_valid;
  stats.sections += executable.all_sections().size();
  for (int32_t raw_target = 0; raw_target < ROCJITSU_CODE_TARGET_NUM_TARGETS; ++raw_target) {
    const auto target = static_cast<rj_code_target_id_t>(raw_target);
    const uint32_t count = executable.num_code_objects(target);
    stats.bundled_code_objects += count;
    for (uint32_t index = 0; index < count; ++index) {
      const AmdGpuCodeObject *object = executable.code_object(target, index);
      if (object == nullptr || !object->is_valid() || object->target_id() != target)
        throw std::logic_error("executable exposed an inconsistent code object");
      inspect_object(*object, stats);
    }
    if (executable.code_object(target, count) != nullptr)
      throw std::logic_error("executable accepted an out-of-range code-object index");
  }
  const rj_code_target_id_t invalid_target = ROCJITSU_CODE_TARGET_INVALID;
  const uint32_t invalid_count = executable.num_code_objects(invalid_target);
  stats.bundled_code_objects += invalid_count;
  for (uint32_t index = 0; index < invalid_count; ++index) {
    const AmdGpuCodeObject *object = executable.code_object(invalid_target, index);
    if (object == nullptr || !object->is_valid() || object->target_id() != invalid_target)
      throw std::logic_error("executable exposed an inconsistent code object");
    inspect_object(*object, stats);
  }
  if (executable.code_object(invalid_target, invalid_count) != nullptr)
    throw std::logic_error("executable accepted an out-of-range code-object index");
}

void overwrite_integer(std::vector<uint8_t> &bytes, SplitMix64 &random, std::size_t width) {
  if (bytes.size() < width)
    return;
  const std::size_t offset = random.below(bytes.size() - width + 1);
  const uint64_t value = random.next();
  std::memcpy(bytes.data() + offset, &value, width);
}

void mutate(std::vector<uint8_t> &bytes, SplitMix64 &random) {
  const std::size_t edits = 1 + random.below(8);
  for (std::size_t edit = 0; edit < edits; ++edit) {
    switch (random.below(8)) {
    case 0:
      if (!bytes.empty())
        bytes[random.below(bytes.size())] ^= static_cast<uint8_t>(1u << random.below(8));
      break;
    case 1:
      if (!bytes.empty())
        bytes[random.below(bytes.size())] = static_cast<uint8_t>(random.next());
      break;
    case 2:
      overwrite_integer(bytes, random, 2);
      break;
    case 3:
      overwrite_integer(bytes, random, 4);
      break;
    case 4:
      overwrite_integer(bytes, random, 8);
      break;
    case 5:
      if (bytes.size() > 1)
        bytes.resize(1 + random.below(bytes.size()));
      break;
    case 6: {
      const std::size_t room = kMaxInputSize - bytes.size();
      const std::size_t count = std::min<std::size_t>(room, 1 + random.below(32));
      const std::size_t at = random.below(bytes.size() + 1);
      bytes.insert(bytes.begin() + static_cast<std::ptrdiff_t>(at), count,
                   static_cast<uint8_t>(random.next()));
      break;
    }
    case 7:
      if (!bytes.empty()) {
        const std::size_t at = random.below(bytes.size());
        const std::size_t count = std::min<std::size_t>(bytes.size() - at, 1 + random.below(32));
        for (std::size_t i = 0; i < count; ++i)
          bytes[at + i] = static_cast<uint8_t>(random.next());
      }
      break;
    }
  }
}

} // namespace

bool load_segment_in_bounds(std::span<const uint8_t> image, std::uintptr_t data_address,
                            std::size_t size) {
  const std::uintptr_t image_begin = reinterpret_cast<std::uintptr_t>(image.data());
  if (data_address < image_begin)
    return false;
  const std::uintptr_t offset = data_address - image_begin;
  return offset <= image.size() && size <= image.size() - offset;
}

RunStats &operator+=(RunStats &left, const RunStats &right) {
  left.inputs += right.inputs;
  left.code_objects_valid += right.code_objects_valid;
  left.executables_valid += right.executables_valid;
  left.bundled_code_objects += right.bundled_code_objects;
  left.sections += right.sections;
  left.text_sections += right.text_sections;
  left.descriptors += right.descriptors;
  left.load_segments += right.load_segments;
  return left;
}

RunStats run_input(std::span<const uint8_t> bytes) {
  if (bytes.size() > kMaxInputSize)
    throw std::invalid_argument("code-object fuzz input exceeds size limit");
  RunStats stats{.inputs = 1};
  AmdGpuCodeObject object(bytes.data(), bytes.size());
  inspect_object(object, stats);
  inspect_executable(bytes, stats);
  return stats;
}

std::vector<std::vector<uint8_t>> seed_corpus() {
  constexpr std::string_view kGfx1250BundleId = "hip-amdgcn-amd-amdhsa--gfx1250";
  std::vector<std::vector<uint8_t>> seeds;
  seeds.push_back(make_device_elf(EF_AMDGPU_MACH_AMDGCN_GFX1250));
  seeds.push_back(make_host_bundle(seeds.front(), kGfx1250BundleId));
  seeds.push_back(std::vector<uint8_t>{0x7f, 'E', 'L', 'F'});
  return seeds;
}

std::vector<uint8_t> generate_input(uint64_t seed) {
  SplitMix64 random(seed);
  if ((random.next() & 15) == 0) {
    std::vector<uint8_t> bytes(1 + random.below(4096));
    for (uint8_t &byte : bytes)
      byte = static_cast<uint8_t>(random.next());
    return bytes;
  }
  auto seeds = seed_corpus();
  std::vector<uint8_t> bytes = seeds[random.below(seeds.size())];
  if ((random.next() & 31) != 0)
    mutate(bytes, random);
  return bytes;
}

} // namespace rocjitsu::code_object_fuzz
