// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/consan/image/metadata.h"

#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/consan/target.h"

#include <algorithm>
#include <cstring>
#include <limits>
#include <map>
#include <optional>
#include <vector>

namespace rocjitsu::consan {
namespace {

// The bounded integer/string/container cursor follows the prototype metadata
// parser's algorithm. This pass additionally requires the entire payload,
// rejects duplicate keys and owns locations for in-place resource updates.
struct Value {
  enum class Kind { Integer, String, Array, Map, Boolean, Nil } kind = Kind::Nil;
  uint64_t integer = 0;
  std::string_view string;
  std::vector<Value> array;
  std::map<std::string_view, Value> map;
  // Byte offset of the MessagePack tag within the metadata payload, used when
  // patching its existing encoding in place.
  size_t offset = 0;
  // Integer payload bytes after the tag; zero means a positive fixint in the tag.
  unsigned integer_bytes = 0;
};

class Cursor {
public:
  explicit Cursor(std::span<const uint8_t> bytes) : bytes_(bytes) {}

  bool read(Value &value, unsigned depth = 0) {
    if (depth > 32 || ++nodes_ > 65536 || offset_ >= bytes_.size())
      return false;
    value.offset = offset_;
    const uint8_t tag = bytes_[offset_++];
    uint64_t count = 0;
    if (tag < 128 || (tag >= 0xcc && tag <= 0xcf)) {
      value.kind = Value::Kind::Integer;
      value.integer_bytes = tag < 128 ? 0 : 1u << (tag - 0xcc);
      if (tag < 128) {
        value.integer = tag;
        return true;
      }
      return big_endian(value.integer_bytes, value.integer);
    }
    if ((tag & 0xe0) == 0xa0 || tag == 0xd9 || tag == 0xda || tag == 0xdb) {
      value.kind = Value::Kind::String;
      if ((tag & 0xe0) == 0xa0)
        count = tag & 31;
      else if (!big_endian(1u << (tag - 0xd9), count))
        return false;
      if (count > bytes_.size() - offset_)
        return false;
      value.string = {reinterpret_cast<const char *>(bytes_.data() + offset_),
                      static_cast<size_t>(count)};
      offset_ += count;
      return true;
    }
    const bool map = (tag & 0xf0) == 0x80 || tag == 0xde || tag == 0xdf;
    const bool array = (tag & 0xf0) == 0x90 || tag == 0xdc || tag == 0xdd;
    if (map || array) {
      value.kind = map ? Value::Kind::Map : Value::Kind::Array;
      if ((tag & 0xe0) == 0x80)
        count = tag & 15;
      else if (!big_endian((tag == 0xdc || tag == 0xde) ? 2 : 4, count))
        return false;
      if (count > 65536 || count > bytes_.size() - offset_)
        return false;
      for (uint64_t index = 0; index < count; ++index) {
        Value key;
        Value child;
        if (map && (!read(key, depth + 1) || key.kind != Value::Kind::String))
          return false;
        if (!read(child, depth + 1))
          return false;
        if (map) {
          if (!value.map.emplace(key.string, std::move(child)).second)
            return false;
        } else {
          value.array.push_back(std::move(child));
        }
      }
      return true;
    }
    if (tag == 0xc2 || tag == 0xc3) {
      value.kind = Value::Kind::Boolean;
      value.integer = tag == 0xc3;
      return true;
    }
    return tag == 0xc0;
  }

  bool at_end() const { return offset_ == bytes_.size(); }

private:
  bool big_endian(unsigned count, uint64_t &value) {
    if (count > 8 || count > bytes_.size() - offset_)
      return false;
    value = 0;
    for (unsigned index = 0; index < count; ++index)
      value = (value << 8) | bytes_[offset_++];
    return true;
  }

  std::span<const uint8_t> bytes_;
  size_t offset_ = 0;
  size_t nodes_ = 0;
};

const Value *field(const Value &value, std::string_view name) {
  const auto found = value.map.find(name);
  return found == value.map.end() ? nullptr : &found->second;
}

std::optional<uint32_t> integer(const Value &value, std::string_view name) {
  const Value *child = field(value, name);
  if (child == nullptr || child->kind != Value::Kind::Integer ||
      child->integer > std::numeric_limits<uint32_t>::max())
    return std::nullopt;
  return static_cast<uint32_t>(child->integer);
}

bool text_equals(const Value &value, std::string_view name, std::string_view expected) {
  const Value *child = field(value, name);
  return child != nullptr && child->kind == Value::Kind::String && child->string == expected;
}

bool matches_target(const Value &root, std::string_view processor) {
  const Value *target = field(root, "amdhsa.target");
  if (target == nullptr || target->kind != Value::Kind::String)
    return false;
  // An omitted triple environment and "unknown" are equivalent. Keep the
  // complete processor match: feature suffixes need their own ELF ABI checks.
  for (const std::string_view prefix : {"amdgcn-amd-amdhsa--", "amdgcn-amd-amdhsa-unknown-"})
    if (target->string.starts_with(prefix) && target->string.substr(prefix.size()) == processor)
      return true;
  return false;
}

std::optional<std::span<const uint8_t>> note(std::span<const uint8_t> image) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return std::nullopt;
  std::memcpy(&header, image.data(), sizeof(header));
  if (header.e_shentsize != sizeof(Elf64_Shdr) || header.e_shoff > image.size() ||
      uint64_t{header.e_shnum} * sizeof(Elf64_Shdr) > image.size() - header.e_shoff)
    return std::nullopt;
  std::optional<std::span<const uint8_t>> result;
  for (uint32_t index = 0; index < header.e_shnum; ++index) {
    Elf64_Shdr section{};
    std::memcpy(&section, image.data() + header.e_shoff + index * sizeof(section), sizeof(section));
    if (section.sh_type != SHT_NOTE)
      continue;
    if (section.sh_offset > image.size() || section.sh_size > image.size() - section.sh_offset)
      return std::nullopt;
    const auto bytes = image.subspan(section.sh_offset, section.sh_size);
    for (size_t offset = 0; offset < bytes.size();) {
      if (bytes.size() - offset < 12)
        return std::nullopt;
      std::array<uint32_t, 3> framing{};
      std::memcpy(framing.data(), bytes.data() + offset, 12);
      const uint64_t name_bytes = (uint64_t{framing[0]} + 3) & ~uint64_t{3};
      const uint64_t data_bytes = (uint64_t{framing[1]} + 3) & ~uint64_t{3};
      if (name_bytes + data_bytes > bytes.size() - offset - 12)
        return std::nullopt;
      if (framing[2] == NT_AMDGPU_METADATA) {
        if (result || framing[0] != 7 || std::memcmp(bytes.data() + offset + 12, "AMDGPU", 7) != 0)
          return std::nullopt;
        result = bytes.subspan(offset + 12 + name_bytes, framing[1]);
      }
      offset += 12 + name_bytes + data_bytes;
    }
  }
  return result;
}

struct Parsed {
  std::span<const uint8_t> payload;
  Value root;
  const Value *kernel = nullptr;
  MetadataResources resources;
};

bool parse(std::span<const uint8_t> image, const KernelDescriptorInfo &kernel, Parsed &parsed) {
  Elf64_Ehdr header{};
  if (image.size() < sizeof(header))
    return false;
  std::memcpy(&header, image.data(), sizeof(header));
  const auto *target = target_profile(elf_mach_name(header.e_flags));
  if (target == nullptr)
    return false;
  const uint32_t wave_size = kernel_wavefront_size(target->arch, kernel.descriptor);
  const uint32_t granule = descriptor_vgpr_granularity_for_wavefront(target->arch, wave_size);
  const auto payload = note(image);
  if (!payload)
    return false;
  parsed.payload = *payload;
  Cursor cursor(*payload);
  if (!cursor.read(parsed.root) || !cursor.at_end() || parsed.root.kind != Value::Kind::Map ||
      !matches_target(parsed.root, target->name))
    return false;
  const Value *version = field(parsed.root, "amdhsa.version");
  if (version == nullptr || version->array.size() != 2 ||
      version->array[0].kind != Value::Kind::Integer || version->array[0].integer != 1 ||
      version->array[1].kind != Value::Kind::Integer || version->array[1].integer != 2)
    return false;
  const Value *kernels = field(parsed.root, "amdhsa.kernels");
  if (kernels == nullptr || kernels->kind != Value::Kind::Array || kernels->array.size() != 1)
    return false;
  parsed.kernel = &kernels->array.front();
  const Value &entry = *parsed.kernel;
  const auto kernarg = integer(entry, ".kernarg_segment_size");
  const auto alignment = integer(entry, ".kernarg_segment_align");
  const auto sgprs = integer(entry, ".sgpr_count");
  const auto vgprs = integer(entry, ".vgpr_count");
  if (target->arch == ROCJITSU_CODE_ARCH_CDNA3 || target->arch == ROCJITSU_CODE_ARCH_CDNA4) {
    const uint32_t accumulator_base =
        (RJ_AMDHSA_BITS_GET(kernel.descriptor.compute_pgm_rsrc3,
                            rocjitsu::amdhsa::COMPUTE_PGM_RSRC3_GFX90A_ACCUM_OFFSET) +
         1) *
        4;
    if (!vgprs || *vgprs > accumulator_base)
      return false;
  }
  if (!text_equals(entry, ".name", kernel.kernel_name) ||
      !text_equals(entry, ".symbol", kernel.kernel_name + ".kd") || !kernarg ||
      *kernarg != kernel.descriptor.kernarg_size || !alignment || *alignment != 8 || !sgprs ||
      *sgprs == 0 || *sgprs > 106 || !vgprs || *vgprs == 0 ||
      ((*vgprs + granule - 1) / granule) != 1 + (kernel.descriptor.compute_pgm_rsrc1 & 63) ||
      integer(entry, ".wavefront_size") != wave_size || !supports_wave(*target, wave_size) ||
      integer(entry, ".group_segment_fixed_size") != kernel.descriptor.group_segment_fixed_size ||
      integer(entry, ".private_segment_fixed_size") != 0)
    return false;
  const auto maximum = integer(entry, ".max_flat_workgroup_size");
  if (!maximum || *maximum == 0 || *maximum > 1024)
    return false;
  const Value *dynamic_stack = field(entry, ".uses_dynamic_stack");
  if ((dynamic_stack != nullptr &&
       (dynamic_stack->kind != Value::Kind::Boolean || dynamic_stack->integer != 0)) ||
      field(entry, ".reqd_workgroup_size") != nullptr)
    return false;
  const Value *arguments = field(entry, ".args");
  if (arguments == nullptr || arguments->kind != Value::Kind::Array)
    return false;
  uint64_t previous_end = 0;
  for (const Value &argument : arguments->array) {
    const auto size = integer(argument, ".size");
    const auto offset = integer(argument, ".offset");
    if (!size || *size == 0 || !offset || *offset < previous_end ||
        uint64_t{*offset} + *size > *kernarg ||
        (!text_equals(argument, ".value_kind", "by_value") &&
         !text_equals(argument, ".value_kind", "global_buffer")))
      return false;
    if (text_equals(argument, ".value_kind", "global_buffer") &&
        (*size != 8 || (*offset % 8) != 0 || !text_equals(argument, ".address_space", "global")))
      return false;
    previous_end = uint64_t{*offset} + *size;
  }
  parsed.resources = {*kernarg, *alignment, *sgprs, *vgprs, *maximum};
  return true;
}

Result fail(std::string *error, const char *message) {
  if (error != nullptr)
    *error = message;
  return Result::failure();
}

} // namespace

FailureOr<MetadataResources> inspect_metadata(std::span<const uint8_t> image,
                                              const KernelDescriptorInfo &kernel,
                                              std::string *error) {
  Parsed parsed;
  if (!parse(image, kernel, parsed))
    return fail(error, "unsupported or inconsistent AMDHSA metadata/argument ABI");
  return parsed.resources;
}

Result update_metadata(std::span<uint8_t> image, const KernelDescriptorInfo &kernel,
                       const MetadataResources &resources, std::string *error) {
  Parsed parsed;
  if (!parse(image, kernel, parsed))
    return fail(error, "cannot update inconsistent AMDHSA metadata");
  const std::array<std::pair<std::string_view, uint32_t>, 3> updates = {{
      {".kernarg_segment_size", resources.kernarg_bytes},
      {".sgpr_count", resources.sgprs},
      {".vgpr_count", resources.vgprs},
  }};
  for (const auto &[name, number] : updates) {
    const Value &value = *field(*parsed.kernel, name);
    if ((value.integer_bytes == 0 && number > 127) ||
        (value.integer_bytes != 0 && value.integer_bytes < 4 &&
         number >= (uint64_t{1} << (8 * value.integer_bytes))))
      return fail(error, "resource growth needs a wider metadata integer encoding");
  }
  const size_t payload_offset = parsed.payload.data() - image.data();
  for (const auto &[name, number] : updates) {
    const Value &value = *field(*parsed.kernel, name);
    if (value.integer_bytes == 0) {
      image[payload_offset + value.offset] = static_cast<uint8_t>(number);
    } else {
      for (unsigned index = 0; index < value.integer_bytes; ++index)
        image[payload_offset + value.offset + 1 + index] =
            static_cast<uint8_t>(uint64_t{number} >> (8 * (value.integer_bytes - index - 1)));
    }
  }
  return Result::success();
}

} // namespace rocjitsu::consan
