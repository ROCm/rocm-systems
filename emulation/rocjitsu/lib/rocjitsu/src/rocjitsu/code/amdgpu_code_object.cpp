// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/amdgpu_code_object.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/file_io.h"
#include "rocjitsu/isa/arch/amdgpu/shared/rdna_isa_base.h"

#include "hsa/AMDHSAKernelDescriptor.h" // Check SGPR allocation

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <span>
#include <string_view>
#include <unordered_map>
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
  uint64_t flags() const override { return shdr_.sh_flags; }
  uint64_t vaddr() const override { return shdr_.sh_addr; }
  uint32_t sectionHeaderNameIdx() const override { return shdr_.sh_name; }
  uint64_t sectionOffset() const override { return shdr_.sh_offset; }

private:
  Elf64_Shdr shdr_;
};

bool is_elf(const Elf64_Ehdr &ehdr) { return !std::memcmp(ehdr.e_ident, EI_MAGIC, EI_MAGIC_SIZE); }

using detail::fits_in_bounds;

/*
 * \NPI new GPU: map its MACH value and its gfxNNNN triple to a target id in \
 * both target_from_machine_flags() and target_from_triple() below.
 */

struct FunctionSymbolInfo {
  uint64_t entry_text_offset = 0;
  uint64_t text_file_offset = 0;
  uint64_t text_size = 0;
  uint64_t code_size = 0;
  bool code_size_inferred_from_zero = false;
};

[[nodiscard]] std::optional<uint64_t> symbol_file_offset(const Elf64_Sym &sym,
                                                         const std::vector<Elf64_Shdr> &shdrs) {
  if (sym.st_shndx >= shdrs.size())
    return std::nullopt;
  const auto &section = shdrs[sym.st_shndx];
  if (sym.st_value < section.sh_addr)
    return std::nullopt;
  const uint64_t section_delta = sym.st_value - section.sh_addr;
  if (section_delta > section.sh_size)
    return std::nullopt;
  return section.sh_offset + section_delta;
}

struct MetadataCursor {
  std::span<const uint8_t> bytes;
  size_t offset = 0;
};

[[nodiscard]] bool skip_metadata_bytes(MetadataCursor &cursor, uint64_t count) {
  if (cursor.offset > cursor.bytes.size() || count > cursor.bytes.size() - cursor.offset)
    return false;
  cursor.offset += static_cast<size_t>(count);
  return true;
}

[[nodiscard]] bool read_metadata_be(MetadataCursor &cursor, unsigned byte_count, uint64_t &value) {
  if (byte_count > sizeof(value) || cursor.offset > cursor.bytes.size() ||
      byte_count > cursor.bytes.size() - cursor.offset) {
    return false;
  }
  value = 0;
  for (unsigned i = 0; i < byte_count; ++i)
    value = (value << 8u) | cursor.bytes[cursor.offset++];
  return true;
}

[[nodiscard]] bool read_metadata_collection_count(MetadataCursor &cursor, bool map,
                                                  uint32_t &count) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  if (map && (tag & 0xf0u) == 0x80u) {
    count = tag & 0x0fu;
    return true;
  }
  if (!map && (tag & 0xf0u) == 0x90u) {
    count = tag & 0x0fu;
    return true;
  }
  uint64_t wide_count = 0;
  if (tag == (map ? 0xdeu : 0xdcu)) {
    if (!read_metadata_be(cursor, 2u, wide_count))
      return false;
  } else if (tag == (map ? 0xdfu : 0xddu)) {
    if (!read_metadata_be(cursor, 4u, wide_count))
      return false;
  } else {
    return false;
  }
  if (wide_count > std::numeric_limits<uint32_t>::max())
    return false;
  count = static_cast<uint32_t>(wide_count);
  return true;
}

[[nodiscard]] bool read_metadata_string(MetadataCursor &cursor, std::string_view &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  uint64_t length = 0;
  if ((tag & 0xe0u) == 0xa0u) {
    length = tag & 0x1fu;
  } else if (tag == 0xd9u) {
    if (!read_metadata_be(cursor, 1u, length))
      return false;
  } else if (tag == 0xdau) {
    if (!read_metadata_be(cursor, 2u, length))
      return false;
  } else if (tag == 0xdbu) {
    if (!read_metadata_be(cursor, 4u, length))
      return false;
  } else {
    return false;
  }
  if (cursor.offset > cursor.bytes.size() || length > cursor.bytes.size() - cursor.offset)
    return false;
  value = std::string_view(reinterpret_cast<const char *>(cursor.bytes.data() + cursor.offset),
                           static_cast<size_t>(length));
  cursor.offset += static_cast<size_t>(length);
  return true;
}

[[nodiscard]] bool read_metadata_unsigned(MetadataCursor &cursor, uint64_t &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  if (tag <= 0x7fu) {
    value = tag;
    return true;
  }
  switch (tag) {
  case 0xccu:
    return read_metadata_be(cursor, 1u, value);
  case 0xcdu:
    return read_metadata_be(cursor, 2u, value);
  case 0xceu:
    return read_metadata_be(cursor, 4u, value);
  case 0xcfu:
    return read_metadata_be(cursor, 8u, value);
  default:
    return false;
  }
}

[[nodiscard]] bool skip_metadata_value(MetadataCursor &cursor, unsigned depth = 0) {
  if (depth > 64u || cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset];
  if (tag <= 0x7fu || tag >= 0xe0u || tag == 0xc0u || tag == 0xc2u || tag == 0xc3u)
    return skip_metadata_bytes(cursor, 1u);
  if ((tag & 0xe0u) == 0xa0u)
    return skip_metadata_bytes(cursor, 1u + (tag & 0x1fu));
  if ((tag & 0xf0u) == 0x90u || (tag & 0xf0u) == 0x80u) {
    ++cursor.offset;
    const uint32_t elements =
        (tag & 0xf0u) == 0x80u ? static_cast<uint32_t>(tag & 0x0fu) * 2u : tag & 0x0fu;
    for (uint32_t i = 0; i < elements; ++i) {
      if (!skip_metadata_value(cursor, depth + 1u))
        return false;
    }
    return true;
  }

  ++cursor.offset;
  uint64_t length = 0;
  const auto skip_length_prefixed = [&](unsigned length_bytes, uint64_t extra) {
    return read_metadata_be(cursor, length_bytes, length) &&
           skip_metadata_bytes(cursor, length + extra);
  };
  switch (tag) {
  case 0xc4u:
  case 0xd9u:
    return skip_length_prefixed(1u, 0u);
  case 0xc5u:
  case 0xdau:
    return skip_length_prefixed(2u, 0u);
  case 0xc6u:
  case 0xdbu:
    return skip_length_prefixed(4u, 0u);
  case 0xc7u:
    return skip_length_prefixed(1u, 1u);
  case 0xc8u:
    return skip_length_prefixed(2u, 1u);
  case 0xc9u:
    return skip_length_prefixed(4u, 1u);
  case 0xcau:
  case 0xceu:
  case 0xd2u:
    return skip_metadata_bytes(cursor, 4u);
  case 0xcbu:
  case 0xcfu:
  case 0xd3u:
    return skip_metadata_bytes(cursor, 8u);
  case 0xccu:
  case 0xd0u:
    return skip_metadata_bytes(cursor, 1u);
  case 0xcdu:
  case 0xd1u:
    return skip_metadata_bytes(cursor, 2u);
  case 0xd4u:
    return skip_metadata_bytes(cursor, 2u);
  case 0xd5u:
    return skip_metadata_bytes(cursor, 3u);
  case 0xd6u:
    return skip_metadata_bytes(cursor, 5u);
  case 0xd7u:
    return skip_metadata_bytes(cursor, 9u);
  case 0xd8u:
    return skip_metadata_bytes(cursor, 17u);
  case 0xdcu:
  case 0xddu:
  case 0xdeu:
  case 0xdfu: {
    const bool map = tag == 0xdeu || tag == 0xdfu;
    const unsigned count_bytes = tag == 0xdcu || tag == 0xdeu ? 2u : 4u;
    if (!read_metadata_be(cursor, count_bytes, length) ||
        (map && length > std::numeric_limits<uint64_t>::max() / 2u)) {
      return false;
    }
    const uint64_t elements = map ? length * 2u : length;
    for (uint64_t i = 0; i < elements; ++i) {
      if (!skip_metadata_value(cursor, depth + 1u))
        return false;
    }
    return true;
  }
  default:
    return false;
  }
}

struct KernelMetadata {
  bool has_dynamic_lds = false;
  std::optional<bool> uses_dynamic_stack;
  std::optional<uint16_t> sgpr_count;
  std::optional<std::array<uint32_t, 3>> required_workgroup_size;
};

[[nodiscard]] bool read_kernel_args_metadata(MetadataCursor &cursor, bool &has_dynamic_lds) {
  uint32_t argument_count = 0;
  if (!read_metadata_collection_count(cursor, /*map=*/false, argument_count))
    return false;
  for (uint32_t argument_index = 0; argument_index < argument_count; ++argument_index) {
    uint32_t argument_entries = 0;
    if (!read_metadata_collection_count(cursor, /*map=*/true, argument_entries))
      return false;
    for (uint32_t argument_entry = 0; argument_entry < argument_entries; ++argument_entry) {
      std::string_view argument_key;
      if (!read_metadata_string(cursor, argument_key))
        return false;
      if (argument_key == ".value_kind") {
        std::string_view value_kind;
        if (!read_metadata_string(cursor, value_kind))
          return false;
        has_dynamic_lds |= value_kind == "hidden_dynamic_lds_size";
      } else if (!skip_metadata_value(cursor)) {
        return false;
      }
    }
  }
  return true;
}

[[nodiscard]] std::unordered_map<std::string, KernelMetadata>
parse_kernel_metadata(std::span<const uint8_t> payload) {
  std::unordered_map<std::string, KernelMetadata> result;
  MetadataCursor root{payload};
  uint32_t root_entries = 0;
  if (!read_metadata_collection_count(root, /*map=*/true, root_entries))
    return result;
  for (uint32_t entry = 0; entry < root_entries; ++entry) {
    std::string_view key;
    if (!read_metadata_string(root, key))
      return {};
    if (key != "amdhsa.kernels") {
      if (!skip_metadata_value(root))
        return {};
      continue;
    }
    uint32_t kernel_count = 0;
    if (!read_metadata_collection_count(root, /*map=*/false, kernel_count))
      return {};
    for (uint32_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
      uint32_t kernel_entries = 0;
      if (!read_metadata_collection_count(root, /*map=*/true, kernel_entries))
        return {};
      std::optional<std::string> name;
      KernelMetadata metadata;
      for (uint32_t kernel_entry = 0; kernel_entry < kernel_entries; ++kernel_entry) {
        std::string_view kernel_key;
        if (!read_metadata_string(root, kernel_key))
          return {};
        if (kernel_key == ".name") {
          std::string_view parsed_name;
          if (!read_metadata_string(root, parsed_name))
            return {};
          name = parsed_name;
        } else if (kernel_key == ".args") {
          if (!read_kernel_args_metadata(root, metadata.has_dynamic_lds))
            return {};
        } else if (kernel_key == ".uses_dynamic_stack") {
          if (root.offset >= root.bytes.size())
            return {};
          const uint8_t tag = root.bytes[root.offset++];
          if (tag != 0xc2u && tag != 0xc3u)
            return {};
          metadata.uses_dynamic_stack = tag == 0xc3u;
        } else if (kernel_key == ".sgpr_count") {
          uint64_t count = 0;
          if (!read_metadata_unsigned(root, count) ||
              count > std::numeric_limits<uint16_t>::max()) {
            return {};
          }
          metadata.sgpr_count = static_cast<uint16_t>(count);
        } else if (kernel_key == ".reqd_workgroup_size") {
          uint32_t dimension_count = 0;
          if (!read_metadata_collection_count(root, /*map=*/false, dimension_count) ||
              dimension_count != 3u) {
            return {};
          }
          std::array<uint32_t, 3> dimensions{};
          for (uint32_t dimension = 0; dimension < dimension_count; ++dimension) {
            uint64_t value = 0;
            if (!read_metadata_unsigned(root, value) || value == 0u ||
                value > std::numeric_limits<uint32_t>::max()) {
              return {};
            }
            dimensions[dimension] = static_cast<uint32_t>(value);
          }
          metadata.required_workgroup_size = dimensions;
        } else if (!skip_metadata_value(root)) {
          return {};
        }
      }
      if (name && (metadata.has_dynamic_lds || metadata.uses_dynamic_stack || metadata.sgpr_count ||
                   metadata.required_workgroup_size))
        result[*name] = metadata;
    }
    break;
  }
  return result;
}

[[nodiscard]] uint64_t align4(uint64_t value) { return (value + 3u) & ~uint64_t{3}; }

[[nodiscard]] std::unordered_map<std::string, KernelMetadata>
read_kernel_metadata(std::span<const uint8_t> image, const Elf64_Ehdr &header) {
  std::unordered_map<std::string, KernelMetadata> result;
  if (header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phoff > image.size() ||
      static_cast<uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr) > image.size() - header.e_phoff) {
    return result;
  }
  for (uint16_t index = 0; index < header.e_phnum; ++index) {
    Elf64_Phdr program_header{};
    std::memcpy(&program_header,
                image.data() + header.e_phoff + static_cast<uint64_t>(index) * sizeof(Elf64_Phdr),
                sizeof(program_header));
    if (program_header.p_type != PT_NOTE || program_header.p_offset > image.size() ||
        program_header.p_filesz > image.size() - program_header.p_offset) {
      continue;
    }
    uint64_t cursor = program_header.p_offset;
    const uint64_t end = cursor + program_header.p_filesz;
    while (cursor <= end && sizeof(Elf64_Nhdr) <= end - cursor) {
      Elf64_Nhdr note{};
      std::memcpy(&note, image.data() + cursor, sizeof(note));
      cursor += sizeof(note);
      const uint64_t name_bytes = align4(note.n_namesz);
      const uint64_t desc_bytes = align4(note.n_descsz);
      if (name_bytes > end - cursor || desc_bytes > end - cursor - name_bytes)
        break;
      const uint64_t desc_offset = cursor + name_bytes;
      cursor = desc_offset + desc_bytes;
      if (note.n_type != NT_AMDGPU_METADATA || note.n_descsz > image.size() - desc_offset)
        continue;
      auto parsed =
          parse_kernel_metadata(image.subspan(static_cast<size_t>(desc_offset), note.n_descsz));
      result.insert(parsed.begin(), parsed.end());
    }
  }
  return result;
}

rj_code_target_id_t target_from_machine_flags(uint32_t flags) {
  uint32_t mach = flags & EF_AMDGPU_MACH;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX90A)
    return ROCJITSU_CODE_TARGET_GFX90A;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX942)
    return ROCJITSU_CODE_TARGET_GFX942;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX950)
    return ROCJITSU_CODE_TARGET_GFX950;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1100)
    return ROCJITSU_CODE_TARGET_GFX1100;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1150)
    return ROCJITSU_CODE_TARGET_GFX1150;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1151)
    return ROCJITSU_CODE_TARGET_GFX1151;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1200)
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1201)
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (mach == EF_AMDGPU_MACH_AMDGCN_GFX1250)
    return ROCJITSU_CODE_TARGET_GFX1250;
  return ROCJITSU_CODE_TARGET_INVALID;
}

rj_code_target_id_t target_from_triple(const std::string &triple) {
  if (triple == "gfx90a")
    return ROCJITSU_CODE_TARGET_GFX90A;
  if (triple == "gfx942")
    return ROCJITSU_CODE_TARGET_GFX942;
  if (triple == "gfx950")
    return ROCJITSU_CODE_TARGET_GFX950;
  if (triple == "gfx1100")
    return ROCJITSU_CODE_TARGET_GFX1100;
  if (triple == "gfx1150")
    return ROCJITSU_CODE_TARGET_GFX1150;
  if (triple == "gfx1151")
    return ROCJITSU_CODE_TARGET_GFX1151;
  if (triple == "gfx1200")
    return ROCJITSU_CODE_TARGET_GFX1200;
  if (triple == "gfx1201")
    return ROCJITSU_CODE_TARGET_GFX1201;
  if (triple == "gfx1250")
    return ROCJITSU_CODE_TARGET_GFX1250;
  return ROCJITSU_CODE_TARGET_INVALID;
}

} // namespace

AmdGpuCodeObject::AmdGpuCodeObject(AmdGpuCodeObject &&other) noexcept
    : target_id_(other.target_id_), offload_kind_(std::move(other.offload_kind_)),
      target_triple_(std::move(other.target_triple_)) {
  is_valid_ = other.is_valid_;
  image_ = std::move(other.image_);
  header_ = std::move(other.header_);
  sections_ = std::move(other.sections_);
  text_sections_ = std::move(other.text_sections_);
  rodata_sections_ = std::move(other.rodata_sections_);
  kd_offsets_ = std::move(other.kd_offsets_);
  kernels_ = std::move(other.kernels_);
  functions_ = std::move(other.functions_);
}

AmdGpuCodeObject::AmdGpuCodeObject(const std::string &elf_path) {
  try {
    image_ = detail::read_file_bytes(elf_path);
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

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_machine_flags(header_->flags());
}

AmdGpuCodeObject::AmdGpuCodeObject(const uint8_t *elf_bytes, size_t elf_size,
                                   std::string offload_kind, std::string target_triple)
    : offload_kind_(std::move(offload_kind)), target_triple_(std::move(target_triple)) {
  if (elf_size < sizeof(Elf64_Ehdr)) {
    is_valid_ = false;
    return;
  }

  image_.assign(reinterpret_cast<const char *>(elf_bytes),
                reinterpret_cast<const char *>(elf_bytes) + elf_size);

  Elf64_Ehdr ehdr;
  std::memcpy(&ehdr, image_.data(), sizeof(Elf64_Ehdr));

  if (!is_elf(ehdr) || ehdr.e_ident[EI_CLASS] != ELFCLASS64 ||
      ehdr.e_ident[EI_OSABI] != ELFOSABI_AMDGPU_HSA) {
    is_valid_ = false;
    return;
  }

  header_ = std::make_unique<HsaHeader>(ehdr);
  load_sections();
  if (!is_valid_)
    return;
  target_id_ = target_from_triple(target_triple_);
}

AmdGpuCodeObject::~AmdGpuCodeObject() = default;

void AmdGpuCodeObject::load_sections() {
  const auto shoff = header_->sectionHeaderOff();
  const int num_shdrs = header_->numSectionHeaders();
  if (num_shdrs <= 0 ||
      !fits_in_bounds(shoff, static_cast<uint64_t>(num_shdrs) * sizeof(Elf64_Shdr),
                      image_.size())) {
    is_valid_ = false;
    return;
  }

  std::vector<Elf64_Shdr> section_hdrs(static_cast<size_t>(num_shdrs));
  std::memcpy(section_hdrs.data(), image_.data() + shoff, section_hdrs.size() * sizeof(Elf64_Shdr));

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

  std::vector<std::string> section_names(section_hdrs.size());
  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    const auto &shdr = section_hdrs[i];
    if (shdr.sh_name >= shstrtab.sh_size)
      continue;
    size_t max_len = shstrtab.sh_size - shdr.sh_name;
    section_names[i] =
        std::string(shstrtab_data + shdr.sh_name, strnlen(shstrtab_data + shdr.sh_name, max_len));
  }

  // Section payloads are copied into owning Section objects. Bound their
  // aggregate before allocating so duplicate or overlapping headers cannot
  // amplify one input image into an unbounded parser working set.
  uint64_t copied_section_bytes = 0;
  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    const auto &shdr = section_hdrs[i];
    if (shdr.sh_type == SHT_NULL || shdr.sh_type == SHT_NOBITS || section_names[i].empty())
      continue;
    if (!fits_in_bounds(shdr.sh_offset, shdr.sh_size, image_.size()) ||
        shdr.sh_size > image_.size() - copied_section_bytes) {
      is_valid_ = false;
      return;
    }
    copied_section_bytes += shdr.sh_size;
  }

  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    const auto &shdr = section_hdrs[i];
    if (shdr.sh_type == SHT_NULL || shdr.sh_type == SHT_NOBITS)
      continue;
    if (section_names[i].empty())
      continue;

    const std::string &sec_name = section_names[i];

    auto sec_data = std::make_unique<char[]>(shdr.sh_size);
    std::memcpy(sec_data.get(), image_.data() + shdr.sh_offset, shdr.sh_size);
    sections_.emplace_back(std::make_unique<HsaSection>(sec_name, std::move(sec_data), shdr));

    if (sec_name == ".text")
      text_sections_.push_back(sections_.back().get());
    else if (sec_name == ".rodata")
      rodata_sections_.push_back(sections_.back().get());
  }

  // Parse symbol table for kernel descriptor offsets.
  // Scan both SHT_SYMTAB and SHT_DYNSYM — stripped code objects may
  // only have the latter.
  std::unordered_map<std::string, uint64_t> descriptor_file_offsets;
  std::unordered_map<std::string, FunctionSymbolInfo> function_symbols;
  std::unordered_map<std::string, bool> dynamic_stack_symbols;
  for (size_t i = 0; i < section_hdrs.size(); ++i) {
    if (section_hdrs[i].sh_type != SHT_SYMTAB && section_hdrs[i].sh_type != SHT_DYNSYM)
      continue;
    auto &symtab_shdr = section_hdrs[i];
    if (symtab_shdr.sh_entsize == 0)
      continue;
    if (symtab_shdr.sh_entsize < sizeof(Elf64_Sym))
      continue;
    if (!fits_in_bounds(symtab_shdr.sh_offset, symtab_shdr.sh_size, image_.size()))
      continue;

    // Read the string table linked to this symtab.
    if (symtab_shdr.sh_link >= section_hdrs.size())
      continue;
    auto &strtab_shdr = section_hdrs[symtab_shdr.sh_link];
    if (!fits_in_bounds(strtab_shdr.sh_offset, strtab_shdr.sh_size, image_.size()))
      continue;
    const char *sym_strtab = image_.data() + strtab_shdr.sh_offset;

    // Read symbols.
    size_t num_syms = symtab_shdr.sh_size / symtab_shdr.sh_entsize;
    const char *symtab_data = image_.data() + symtab_shdr.sh_offset;

    for (size_t sym_index = 0; sym_index < num_syms; ++sym_index) {
      const char *sym_data = symtab_data + sym_index * symtab_shdr.sh_entsize;
      Elf64_Sym sym;
      std::memcpy(&sym, sym_data, sizeof(sym));
      if (sym.st_name >= strtab_shdr.sh_size)
        continue;
      std::string sym_name(sym_strtab + sym.st_name,
                           strnlen(sym_strtab + sym.st_name, strtab_shdr.sh_size - sym.st_name));
      constexpr std::string_view kDynamicStackSuffix = ".has_dyn_sized_stack";
      if (sym.st_shndx == SHN_ABS && sym_name.ends_with(kDynamicStackSuffix)) {
        const std::string kernel_name =
            sym_name.substr(0, sym_name.size() - kDynamicStackSuffix.size());
        dynamic_stack_symbols[kernel_name] = sym.st_value != 0;
        continue;
      }
      // AMDHSA kernel descriptors have a ".kd" suffix symbol.
      if (sym_name.size() > 3 && sym_name.substr(sym_name.size() - 3) == ".kd") {
        std::string kernel_name = sym_name.substr(0, sym_name.size() - 3);
        kd_offsets_[kernel_name] = sym.st_value;
        if (auto file_offset = symbol_file_offset(sym, section_hdrs))
          descriptor_file_offsets[kernel_name] = *file_offset;
        continue;
      }

      if (elf_symbol_type(sym.st_info) == kElfSymbolTypeFunc &&
          sym.st_shndx < section_hdrs.size() && section_names[sym.st_shndx] == ".text") {
        const auto &text = section_hdrs[sym.st_shndx];
        if (sym.st_value >= text.sh_addr && sym.st_value - text.sh_addr <= text.sh_size) {
          FunctionSymbolInfo info;
          info.entry_text_offset = sym.st_value - text.sh_addr;
          info.text_file_offset = text.sh_offset;
          info.text_size = text.sh_size;
          info.code_size = sym.st_size;
          const auto existing = function_symbols.find(sym_name);
          if (existing == function_symbols.end() ||
              (existing->second.code_size == 0 && info.code_size != 0))
            function_symbols[sym_name] = info;
        }
      }
    }
  }

  struct FunctionEntryEvidence {
    uint64_t explicit_code_size = 0;
    bool conflicting_explicit_sizes = false;
  };
  using FunctionEntryKey = std::pair<uint64_t, uint64_t>;
  std::map<FunctionEntryKey, FunctionEntryEvidence> function_entries;
  for (const auto &[name, function] : function_symbols) {
    (void)name;
    auto &entry = function_entries[{function.text_file_offset, function.entry_text_offset}];
    if (function.code_size == 0)
      continue;
    if (entry.explicit_code_size != 0 && entry.explicit_code_size != function.code_size)
      entry.conflicting_explicit_sizes = true;
    else
      entry.explicit_code_size = function.code_size;
  }

  // Assembly-produced code objects may leave STT_FUNC sizes at zero. Prefer an
  // unambiguous explicit size from another symbol at the same entry: generated
  // device libraries commonly publish one real symbol plus many zero-sized
  // target-selection aliases. Only when no such ELF evidence exists, infer a
  // conservative range from the next distinct function entry in the same text
  // section, or from the section end for the final function.
  for (auto &[name, function] : function_symbols) {
    (void)name;
    if (function.code_size != 0 || function.entry_text_offset >= function.text_size)
      continue;
    const FunctionEntryKey key{function.text_file_offset, function.entry_text_offset};
    const auto evidence = function_entries.find(key);
    if (evidence != function_entries.end() && evidence->second.explicit_code_size != 0 &&
        !evidence->second.conflicting_explicit_sizes) {
      function.code_size = evidence->second.explicit_code_size;
      continue;
    }

    uint64_t end = function.text_size;
    const auto next =
        evidence == function_entries.end() ? function_entries.end() : std::next(evidence);
    if (next != function_entries.end() && next->first.first == function.text_file_offset)
      end = std::min(end, next->first.second);
    function.code_size_inferred_from_zero = true;
    function.code_size = end - function.entry_text_offset;
  }

  Elf64_Ehdr elf_header{};
  std::memcpy(&elf_header, image_.data(), sizeof(elf_header));
  const auto metadata = read_kernel_metadata(
      std::span<const uint8_t>(reinterpret_cast<const uint8_t *>(image_.data()), image_.size()),
      elf_header);

  kernels_.clear();
  kernels_.reserve(kd_offsets_.size());
  for (const auto &entry : kd_offsets_) {
    const std::string &kernel_name = entry.first;
    AmdGpuKernelInfo kernel;
    kernel.name = kernel_name;
    if (auto file_offset = descriptor_file_offsets.find(kernel_name);
        file_offset != descriptor_file_offsets.end())
      kernel.descriptor_file_offset = file_offset->second;

    if (auto func = function_symbols.find(kernel_name); func != function_symbols.end()) {
      kernel.entry_text_offset = func->second.entry_text_offset;
      kernel.text_file_offset = func->second.text_file_offset;
      kernel.text_size = func->second.text_size;
      kernel.code_size = func->second.code_size;
      kernel.code_size_inferred_from_zero = func->second.code_size_inferred_from_zero;
      kernel.has_text_range = true;
    } else {
      kernel.code_size = 0;
      kernel.has_text_range = false;
    }
    if (auto metadata_entry = metadata.find(kernel_name); metadata_entry != metadata.end()) {
      kernel.has_dynamic_lds = metadata_entry->second.has_dynamic_lds;
      kernel.uses_dynamic_stack = metadata_entry->second.uses_dynamic_stack;
      kernel.sgpr_count = metadata_entry->second.sgpr_count;
      kernel.required_workgroup_size = metadata_entry->second.required_workgroup_size;
    }
    if (!kernel.uses_dynamic_stack) {
      auto dynamic_stack = dynamic_stack_symbols.find(kernel_name);
      if (dynamic_stack != dynamic_stack_symbols.end())
        kernel.uses_dynamic_stack = dynamic_stack->second;
    }
    kernels_.push_back(std::move(kernel));
  }
  std::sort(kernels_.begin(), kernels_.end(),
            [](const auto &lhs, const auto &rhs) { return lhs.name < rhs.name; });

  functions_.clear();
  functions_.reserve(function_symbols.size());
  for (const auto &entry : function_symbols) {
    AmdGpuFunctionInfo function;
    function.name = entry.first;
    function.entry_text_offset = entry.second.entry_text_offset;
    function.text_file_offset = entry.second.text_file_offset;
    function.text_size = entry.second.text_size;
    function.code_size = entry.second.code_size;
    function.code_size_inferred_from_zero = entry.second.code_size_inferred_from_zero;
    functions_.push_back(std::move(function));
  }
  std::sort(functions_.begin(), functions_.end(), [](const auto &lhs, const auto &rhs) {
    if (lhs.entry_text_offset != rhs.entry_text_offset)
      return lhs.entry_text_offset < rhs.entry_text_offset;
    return lhs.name < rhs.name;
  });

  is_valid_ = true;
}

uint64_t AmdGpuCodeObject::kernel_descriptor_offset(const std::string &kernel_name) const {
  auto it = kd_offsets_.find(kernel_name);
  return it != kd_offsets_.end() ? it->second : 0;
}

namespace {

// CDNA targets encode the wavefront SGPR count in the descriptor even when the
// granulated field is 0; RDNA-style targets treat a granulated 0 as "use the
// fixed per-wave SGPR pool". Mirrors the command processor's
// sgpr_count_is_descriptor_encoded(); kept as the short, stable CDNA
// list so a new non-CDNA family falls through to the RDNA-style branch by
// default. (Canonical copy: code/dbt/kernel_descriptor_translator.cpp.)
[[nodiscard]] bool is_cdna_arch(rj_code_arch_t arch) {
  return arch == ROCJITSU_CODE_ARCH_CDNA1 || arch == ROCJITSU_CODE_ARCH_CDNA2 ||
         arch == ROCJITSU_CODE_ARCH_CDNA3 || arch == ROCJITSU_CODE_ARCH_CDNA4;
}

} // namespace

uint32_t amdgpu_kernel_descriptor_sgpr_count(uint32_t granulated, rj_code_arch_t arch) {
  // Descriptor-encoded (granulated != 0, or a CDNA target): (granulated + 1) * 8.
  // Otherwise the field is an RDNA-style sentinel and the wave owns the fixed
  // per-wave SGPR pool.
  if (granulated != 0 || is_cdna_arch(arch))
    return (granulated + 1) * 8;
  return amdgpu::RdnaIsaBase::MAX_SGPRS_PER_WF;
}

std::optional<uint32_t> AmdGpuCodeObject::min_kernel_sgpr_count(rj_code_arch_t arch) const {
  namespace kd = rocr::llvm::amdhsa;
  using KD = kd::kernel_descriptor_t;

  std::optional<uint32_t> min_count;
  for (const auto &[name, kd_vaddr] : kd_offsets_) {
    // Locate the section whose address range covers the .kd symbol, then read the
    // descriptor out of that section's own bytes (no ELF re-walk).
    for (const auto &section : all_sections()) {
      const uint64_t base = section->vaddr();
      if (base == 0 || kd_vaddr < base)
        continue;
      const uint64_t off = kd_vaddr - base;
      if (off + sizeof(KD) > section->size())
        continue;
      KD desc;
      std::memcpy(&desc, section->data() + off, sizeof(desc));
      const uint32_t granulated = AMDHSA_BITS_GET(
          desc.compute_pgm_rsrc1, kd::COMPUTE_PGM_RSRC1_GRANULATED_WAVEFRONT_SGPR_COUNT);
      const uint32_t count = amdgpu_kernel_descriptor_sgpr_count(granulated, arch);
      min_count = min_count ? std::min(*min_count, count) : count;
      break;
    }
  }
  return min_count;
}

} // namespace rocjitsu
