// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/spill_manager.h"

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/code/amdgpu_elf.h"
#include "rocjitsu/code/patch/instruction_builder.h"
#include "util/bit.h"

RJ_DIAGNOSTIC_PUSH
RJ_DIAGNOSTIC_IGNORE_PEDANTIC
#include "hsa/AMDHSAKernelDescriptor.h"
RJ_DIAGNOSTIC_POP

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <limits>
#include <string_view>
#include <utility> // for std::pair

namespace rocjitsu {

namespace {

/// Per-class hardware bound. Indices >= this are not representable in
/// RegisterSet's bitsets and indicate either a programming error or a class
/// that RegisterSet doesn't track (EXEC, VCC, etc.).
[[nodiscard]] size_t per_class_max(RegClass cls) {
  switch (cls) {
  case RegClass::SGPR:
    return REGISTER_SET_MAX_SGPRS;
  case RegClass::VGPR:
    return REGISTER_SET_MAX_VGPRS;
  case RegClass::ACC_VGPR:
    return REGISTER_SET_MAX_ACC_VGPRS;
  default:
    return 0; // class not tracked — every index rejected
  }
}

struct MsgPackCursor {
  std::span<uint8_t> bytes;
  size_t offset = 0;
};

[[nodiscard]] bool skip_bytes(MsgPackCursor &cursor, uint64_t count) {
  if (count > cursor.bytes.size() - cursor.offset)
    return false;
  cursor.offset += static_cast<size_t>(count);
  return true;
}

[[nodiscard]] bool read_big_endian(MsgPackCursor &cursor, unsigned byte_count, uint64_t &value) {
  if (byte_count > sizeof(value) || byte_count > cursor.bytes.size() - cursor.offset)
    return false;
  value = 0;
  for (unsigned i = 0; i < byte_count; ++i)
    value = (value << 8u) | cursor.bytes[cursor.offset++];
  return true;
}

[[nodiscard]] bool read_collection_count(MsgPackCursor &cursor, bool map, uint32_t &count) {
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
    if (!read_big_endian(cursor, 2, wide_count))
      return false;
  } else if (tag == (map ? 0xdfu : 0xddu)) {
    if (!read_big_endian(cursor, 4, wide_count))
      return false;
  } else {
    return false;
  }
  count = static_cast<uint32_t>(wide_count);
  return true;
}

[[nodiscard]] bool read_msgpack_string(MsgPackCursor &cursor, std::string_view &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset++];
  uint64_t length = 0;
  if ((tag & 0xe0u) == 0xa0u) {
    length = tag & 0x1fu;
  } else if (tag == 0xd9u) {
    if (!read_big_endian(cursor, 1, length))
      return false;
  } else if (tag == 0xdau) {
    if (!read_big_endian(cursor, 2, length))
      return false;
  } else if (tag == 0xdbu) {
    if (!read_big_endian(cursor, 4, length))
      return false;
  } else {
    return false;
  }
  if (length > cursor.bytes.size() - cursor.offset)
    return false;
  value = std::string_view(reinterpret_cast<const char *>(cursor.bytes.data() + cursor.offset),
                           static_cast<size_t>(length));
  cursor.offset += static_cast<size_t>(length);
  return true;
}

struct MsgPackUnsigned {
  uint64_t value = 0;
  size_t token_offset = 0;
  uint8_t tag = 0;
};

[[nodiscard]] bool read_msgpack_unsigned(MsgPackCursor &cursor, MsgPackUnsigned &value) {
  if (cursor.offset >= cursor.bytes.size())
    return false;
  value.token_offset = cursor.offset;
  value.tag = cursor.bytes[cursor.offset++];
  if (value.tag <= 0x7fu) {
    value.value = value.tag;
    return true;
  }
  unsigned byte_count = 0;
  switch (value.tag) {
  case 0xccu:
    byte_count = 1;
    break;
  case 0xcdu:
    byte_count = 2;
    break;
  case 0xceu:
    byte_count = 4;
    break;
  case 0xcfu:
    byte_count = 8;
    break;
  default:
    return false;
  }
  return read_big_endian(cursor, byte_count, value.value);
}

[[nodiscard]] bool write_msgpack_unsigned(std::span<uint8_t> bytes, const MsgPackUnsigned &encoded,
                                          uint64_t value) {
  if (encoded.token_offset >= bytes.size())
    return false;
  if (encoded.tag <= 0x7fu) {
    if (value > 0x7fu)
      return false;
    bytes[encoded.token_offset] = static_cast<uint8_t>(value);
    return true;
  }
  unsigned byte_count = 0;
  switch (encoded.tag) {
  case 0xccu:
    byte_count = 1;
    break;
  case 0xcdu:
    byte_count = 2;
    break;
  case 0xceu:
    byte_count = 4;
    break;
  case 0xcfu:
    byte_count = 8;
    break;
  default:
    return false;
  }
  if (byte_count < sizeof(value) && value >= (uint64_t{1} << (byte_count * 8u)))
    return false;
  if (encoded.token_offset + 1u + byte_count > bytes.size())
    return false;
  for (unsigned i = 0; i < byte_count; ++i) {
    const unsigned shift = (byte_count - i - 1u) * 8u;
    bytes[encoded.token_offset + 1u + i] = static_cast<uint8_t>(value >> shift);
  }
  return true;
}

[[nodiscard]] bool skip_msgpack_value(MsgPackCursor &cursor, unsigned depth = 0) {
  if (depth > 64 || cursor.offset >= cursor.bytes.size())
    return false;
  const uint8_t tag = cursor.bytes[cursor.offset];
  if (tag <= 0x7fu || tag >= 0xe0u || tag == 0xc0u || tag == 0xc2u || tag == 0xc3u)
    return skip_bytes(cursor, 1);
  if ((tag & 0xe0u) == 0xa0u)
    return skip_bytes(cursor, 1u + (tag & 0x1fu));
  if ((tag & 0xf0u) == 0x90u) {
    ++cursor.offset;
    for (uint32_t i = 0; i < (tag & 0x0fu); ++i) {
      if (!skip_msgpack_value(cursor, depth + 1))
        return false;
    }
    return true;
  }
  if ((tag & 0xf0u) == 0x80u) {
    ++cursor.offset;
    for (uint32_t i = 0; i < (tag & 0x0fu) * 2u; ++i) {
      if (!skip_msgpack_value(cursor, depth + 1))
        return false;
    }
    return true;
  }

  ++cursor.offset;
  uint64_t length = 0;
  auto skip_length_prefixed = [&](unsigned length_bytes, uint64_t extra) {
    return read_big_endian(cursor, length_bytes, length) && skip_bytes(cursor, length + extra);
  };
  switch (tag) {
  case 0xc4u:
  case 0xd9u:
    return skip_length_prefixed(1, 0);
  case 0xc5u:
  case 0xdau:
    return skip_length_prefixed(2, 0);
  case 0xc6u:
  case 0xdbu:
    return skip_length_prefixed(4, 0);
  case 0xc7u:
    return skip_length_prefixed(1, 1);
  case 0xc8u:
    return skip_length_prefixed(2, 1);
  case 0xc9u:
    return skip_length_prefixed(4, 1);
  case 0xcau:
  case 0xceu:
  case 0xd2u:
    return skip_bytes(cursor, 4);
  case 0xcbu:
  case 0xcfu:
  case 0xd3u:
    return skip_bytes(cursor, 8);
  case 0xccu:
  case 0xd0u:
    return skip_bytes(cursor, 1);
  case 0xcdu:
  case 0xd1u:
    return skip_bytes(cursor, 2);
  case 0xd4u:
    return skip_bytes(cursor, 2);
  case 0xd5u:
    return skip_bytes(cursor, 3);
  case 0xd6u:
    return skip_bytes(cursor, 5);
  case 0xd7u:
    return skip_bytes(cursor, 9);
  case 0xd8u:
    return skip_bytes(cursor, 17);
  case 0xdcu:
  case 0xddu: {
    const unsigned count_bytes = tag == 0xdcu ? 2u : 4u;
    if (!read_big_endian(cursor, count_bytes, length))
      return false;
    for (uint64_t i = 0; i < length; ++i) {
      if (!skip_msgpack_value(cursor, depth + 1))
        return false;
    }
    return true;
  }
  case 0xdeu:
  case 0xdfu: {
    const unsigned count_bytes = tag == 0xdeu ? 2u : 4u;
    if (!read_big_endian(cursor, count_bytes, length))
      return false;
    if (length > std::numeric_limits<uint64_t>::max() / 2u)
      return false;
    for (uint64_t i = 0; i < length * 2u; ++i) {
      if (!skip_msgpack_value(cursor, depth + 1))
        return false;
    }
    return true;
  }
  default:
    return false;
  }
}

enum class MetadataSearchResult : uint8_t {
  Updated,
  Unchanged,
  KernelNotFound,
  Invalid,
  Unencodable,
};

[[nodiscard]] MetadataSearchResult update_metadata_payload(std::span<uint8_t> payload,
                                                           std::string_view kernel_name,
                                                           uint32_t required_private_bytes) {
  MsgPackCursor root{payload};
  uint32_t root_entries = 0;
  if (!read_collection_count(root, /*map=*/true, root_entries))
    return MetadataSearchResult::Invalid;
  for (uint32_t entry = 0; entry < root_entries; ++entry) {
    std::string_view key;
    if (!read_msgpack_string(root, key))
      return MetadataSearchResult::Invalid;
    if (key != "amdhsa.kernels") {
      if (!skip_msgpack_value(root))
        return MetadataSearchResult::Invalid;
      continue;
    }

    uint32_t kernel_count = 0;
    if (!read_collection_count(root, /*map=*/false, kernel_count))
      return MetadataSearchResult::Invalid;
    for (uint32_t kernel_index = 0; kernel_index < kernel_count; ++kernel_index) {
      uint32_t kernel_entries = 0;
      if (!read_collection_count(root, /*map=*/true, kernel_entries))
        return MetadataSearchResult::Invalid;
      std::optional<std::string_view> name;
      std::optional<MsgPackUnsigned> private_bytes;
      for (uint32_t kernel_entry = 0; kernel_entry < kernel_entries; ++kernel_entry) {
        std::string_view kernel_key;
        if (!read_msgpack_string(root, kernel_key))
          return MetadataSearchResult::Invalid;
        if (kernel_key == ".name") {
          std::string_view parsed_name;
          if (!read_msgpack_string(root, parsed_name))
            return MetadataSearchResult::Invalid;
          name = parsed_name;
        } else if (kernel_key == ".private_segment_fixed_size") {
          MsgPackUnsigned parsed_private;
          if (!read_msgpack_unsigned(root, parsed_private))
            return MetadataSearchResult::Invalid;
          private_bytes = parsed_private;
        } else if (!skip_msgpack_value(root)) {
          return MetadataSearchResult::Invalid;
        }
      }
      if (!name || *name != kernel_name)
        continue;
      if (!private_bytes)
        return MetadataSearchResult::Invalid;
      if (private_bytes->value >= required_private_bytes)
        return MetadataSearchResult::Unchanged;
      return write_msgpack_unsigned(payload, *private_bytes, required_private_bytes)
                 ? MetadataSearchResult::Updated
                 : MetadataSearchResult::Unencodable;
    }
    return MetadataSearchResult::KernelNotFound;
  }
  return MetadataSearchResult::KernelNotFound;
}

[[nodiscard]] uint64_t align4(uint64_t value) { return (value + 3u) & ~uint64_t{3}; }

} // namespace

SpillManager::SpillManager(uint32_t original_private_bytes, uint32_t per_lane_scratch_limit)
    : base_offset_(util::align_up(original_private_bytes, kDbiZoneAlignment)),
      total_bytes_(base_offset_), limit_(per_lane_scratch_limit), next_offset_(base_offset_) {}

std::optional<uint32_t> SpillManager::allocate_slot(RegisterRef reg) {
  // Reject indices past the per-class hardware bound (or unsupported classes
  // like EXEC/VCC). The cache lookup happens first so an idempotent re-alloc
  // of an already-cached register cannot fail this check (it never could have
  // been cached without passing the check the first time).
  const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
  auto it = reg_to_offset_.find(key);
  if (it != reg_to_offset_.end()) {
    return it->second;
  }
  if (reg.index >= per_class_max(reg.cls)) {
    return std::nullopt;
  }
  // Overflow-safe equivalent of `next_offset_ + kSlotBytes > limit_`.
  if (static_cast<uint64_t>(next_offset_) + kSlotBytes > limit_) {
    return std::nullopt;
  }
  const uint32_t offset = next_offset_;
  next_offset_ += kSlotBytes;
  total_bytes_ = next_offset_;
  reg_to_offset_.emplace(key, offset);
  return offset;
}

std::optional<uint32_t> SpillManager::allocate_slots(RegisterRef reg, unsigned width) {
  if (width == 0)
    return std::nullopt;
  // Reject ranges that would wrap the uint16_t register index space.
  if (static_cast<uint32_t>(reg.index) + width - 1 > std::numeric_limits<uint16_t>::max()) {
    return std::nullopt;
  }
  // RegisterRef::width is uint8_t; reject anything that wouldn't fit.
  if (width > std::numeric_limits<uint8_t>::max()) {
    return std::nullopt;
  }
  // Reject ranges that would extend past the per-class hardware bound —
  // RegisterSet::expand would silently truncate them, leaving the caller
  // with a short allocation and no error.
  if (static_cast<size_t>(reg.index) + width > per_class_max(reg.cls)) {
    return std::nullopt;
  }

  // Build a width-N range and reserve
  RegisterSet set;
  set.expand(RegisterRef{reg.cls, reg.index, static_cast<uint8_t>(width)});
  if (!reserve(set))
    return std::nullopt;
  // width=1 is irrelevant here; offset_for keys on (cls, index) only.
  return offset_for(RegisterRef{reg.cls, reg.index, 1});
}

bool SpillManager::reserve(const RegisterSet &set) {
  // Count NEW registers (cache misses) so we can size-check upfront.
  unsigned num_new = 0;
  set.for_each([&](RegisterRef reg) {
    const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
    if (!reg_to_offset_.contains(key))
      ++num_new;
  });
  if (num_new > 0 && static_cast<uint64_t>(next_offset_) + kSlotBytes * num_new > limit_) {
    return false;
  }

  // Capacity check passed — no failure possible from here. Every reg from
  // for_each is within per-class bounds (the bitset itself enforces that),
  // and we just verified there's enough room.
  set.for_each([this](RegisterRef reg) {
    [[maybe_unused]] auto off = allocate_slot(reg);
    assert(off.has_value() && "allocate_slot failed after capacity check");
  });
  return true;
}

std::optional<uint32_t> SpillManager::offset_for(RegisterRef reg) const {
  const std::pair<RegClass, uint16_t> key{reg.cls, reg.index};
  auto it = reg_to_offset_.find(key);
  if (it == reg_to_offset_.end()) {
    return std::nullopt;
  }
  return it->second;
}

std::optional<VgprSpillSequence> build_vgpr_spill_sequence(SpillManager &manager,
                                                           uint16_t vgpr_base, uint16_t vgpr_count,
                                                           rj_code_arch_t arch) {
  if ((arch != ROCJITSU_CODE_ARCH_RDNA4 && arch != ROCJITSU_CODE_ARCH_CDNA4) || vgpr_count == 0 ||
      static_cast<uint32_t>(vgpr_base) + vgpr_count > REGISTER_SET_MAX_VGPRS) {
    return std::nullopt;
  }

  SpillManager planned_manager = manager;
  const auto first_offset =
      planned_manager.allocate_slots(RegisterRef{RegClass::VGPR, vgpr_base, 1}, vgpr_count);
  const uint32_t max_private_bytes = arch == ROCJITSU_CODE_ARCH_CDNA4
                                         ? kMaxCdna4AddressFreeScratchPrivateBytes
                                         : kMaxAddressFreeScratchPrivateBytes;
  if (!first_offset || planned_manager.total_private_bytes() > max_private_bytes) {
    return std::nullopt;
  }

  const auto wait_store = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                           : build_s_wait_storecnt0(arch);
  const auto wait_load = arch == ROCJITSU_CODE_ARCH_CDNA4 ? build_cdna4_s_wait_vmcnt0(arch)
                                                          : build_s_wait_loadcnt0(arch);
  const auto wait_guest_load = build_s_wait_flat_load0(arch);
  const auto wait_guest_lds = build_s_wait_lds0(arch);
  if (!wait_store || !wait_load || !wait_guest_load || !wait_guest_lds)
    return std::nullopt;

  VgprSpillSequence sequence;
  sequence.vgpr_base = vgpr_base;
  sequence.vgpr_count = vgpr_count;
  sequence.slot_offsets.reserve(vgpr_count);
  sequence.save_words.reserve(static_cast<size_t>(vgpr_count) * 3u + 3u);
  sequence.restore_words.reserve(static_cast<size_t>(vgpr_count) * 3u + 1u);
  // A live VGPR can still have an outstanding asynchronous guest producer.
  // Saving it before that producer retires snapshots the old value; the probe
  // then clobbers the completed value and restore silently resurrects stale
  // data. Drain both vector/FLAT and LDS load paths before taking the snapshot.
  sequence.save_words.push_back(*wait_guest_load);
  if (arch == ROCJITSU_CODE_ARCH_RDNA4)
    sequence.save_words.push_back(*wait_guest_lds);
  for (uint16_t i = 0; i < vgpr_count; ++i) {
    const uint16_t vgpr = static_cast<uint16_t>(vgpr_base + i);
    const auto offset = planned_manager.offset_for(RegisterRef{RegClass::VGPR, vgpr, 1});
    if (!offset)
      return std::nullopt;
    sequence.slot_offsets.push_back(*offset);
    if (arch == ROCJITSU_CODE_ARCH_CDNA4) {
      const auto store = build_cdna4_address_free_scratch_store_b32(vgpr, *offset, arch);
      const auto load = build_cdna4_address_free_scratch_load_b32(vgpr, *offset, arch);
      if (!store || !load)
        return std::nullopt;
      sequence.save_words.insert(sequence.save_words.end(), store->begin(), store->end());
      sequence.restore_words.insert(sequence.restore_words.end(), load->begin(), load->end());
    } else {
      const auto store = build_address_free_scratch_store_b32(vgpr, *offset, arch);
      const auto load = build_address_free_scratch_load_b32(vgpr, *offset, arch);
      if (!store || !load)
        return std::nullopt;
      sequence.save_words.insert(sequence.save_words.end(), store->begin(), store->end());
      sequence.restore_words.insert(sequence.restore_words.end(), load->begin(), load->end());
    }
  }
  sequence.save_words.push_back(*wait_store);
  sequence.restore_words.push_back(*wait_load);
  sequence.total_private_bytes = planned_manager.total_private_bytes();
  manager = std::move(planned_manager);
  return sequence;
}

SpillDescriptorUpdate update_kernel_descriptor_for_spills(std::span<uint8_t> image,
                                                          uint64_t descriptor_file_offset,
                                                          uint32_t required_private_bytes,
                                                          bool uses_dynamic_stack) {
  using KD = rocr::llvm::amdhsa::kernel_descriptor_t;
  namespace kd = rocr::llvm::amdhsa;
  if (uses_dynamic_stack)
    return SpillDescriptorUpdate::DynamicStack;
  if (required_private_bytes == 0 || required_private_bytes > kMaxAddressFreeScratchPrivateBytes) {
    return SpillDescriptorUpdate::InvalidPrivateSize;
  }
  if (descriptor_file_offset > image.size() || sizeof(KD) > image.size() - descriptor_file_offset) {
    return SpillDescriptorUpdate::InvalidDescriptor;
  }

  KD descriptor{};
  std::memcpy(&descriptor, image.data() + descriptor_file_offset, sizeof(descriptor));
  const uint32_t grown_private_bytes =
      std::max(descriptor.private_segment_fixed_size, required_private_bytes);
  const bool private_segment_enabled =
      AMDHSA_BITS_GET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT) !=
      0;
  if (grown_private_bytes == descriptor.private_segment_fixed_size && private_segment_enabled)
    return SpillDescriptorUpdate::Unchanged;

  descriptor.private_segment_fixed_size = grown_private_bytes;
  AMDHSA_BITS_SET(descriptor.compute_pgm_rsrc2, kd::COMPUTE_PGM_RSRC2_ENABLE_PRIVATE_SEGMENT, 1u);
  std::memcpy(image.data() + descriptor_file_offset, &descriptor, sizeof(descriptor));
  return SpillDescriptorUpdate::Updated;
}

SpillMetadataUpdate update_amdgpu_metadata_for_spills(std::span<uint8_t> image,
                                                      std::string_view kernel_name,
                                                      uint32_t required_private_bytes) {
  if (image.size() < sizeof(Elf64_Ehdr) || kernel_name.empty() || required_private_bytes == 0)
    return SpillMetadataUpdate::InvalidMetadata;

  Elf64_Ehdr header{};
  std::memcpy(&header, image.data(), sizeof(header));
  if (std::memcmp(header.e_ident, EI_MAGIC, EI_MAGIC_SIZE) != 0)
    return SpillMetadataUpdate::InvalidMetadata;
  if (header.e_phnum == 0)
    return SpillMetadataUpdate::NoMetadata;
  if (header.e_phentsize != sizeof(Elf64_Phdr) || header.e_phoff > image.size() ||
      static_cast<uint64_t>(header.e_phnum) * sizeof(Elf64_Phdr) > image.size() - header.e_phoff) {
    return SpillMetadataUpdate::InvalidMetadata;
  }

  bool saw_metadata = false;
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
        return SpillMetadataUpdate::InvalidMetadata;
      const uint64_t desc_offset = cursor + name_bytes;
      cursor = desc_offset + desc_bytes;
      if (note.n_type != NT_AMDGPU_METADATA)
        continue;
      saw_metadata = true;
      if (note.n_descsz > image.size() - desc_offset)
        return SpillMetadataUpdate::InvalidMetadata;
      const MetadataSearchResult update =
          update_metadata_payload(image.subspan(static_cast<size_t>(desc_offset), note.n_descsz),
                                  kernel_name, required_private_bytes);
      switch (update) {
      case MetadataSearchResult::Updated:
        return SpillMetadataUpdate::Updated;
      case MetadataSearchResult::Unchanged:
        return SpillMetadataUpdate::Unchanged;
      case MetadataSearchResult::KernelNotFound:
        break;
      case MetadataSearchResult::Invalid:
        return SpillMetadataUpdate::InvalidMetadata;
      case MetadataSearchResult::Unencodable:
        return SpillMetadataUpdate::UnencodableGrowth;
      }
    }
  }
  return saw_metadata ? SpillMetadataUpdate::KernelNotFound : SpillMetadataUpdate::NoMetadata;
}

void PrivateDispatchRequirements::note_kernel(uint64_t executable, std::string_view kernel_name,
                                              uint32_t required_private_bytes) {
  if (required_private_bytes == 0)
    return;
  const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
    return candidate.executable == executable && candidate.kernel_name == kernel_name;
  });
  if (pending == pending_.end()) {
    pending_.push_back({executable, std::string(kernel_name), required_private_bytes});
  } else {
    pending->required_private_bytes =
        std::max(pending->required_private_bytes, required_private_bytes);
  }
}

std::optional<uint32_t> PrivateDispatchRequirements::bind_symbol(uint64_t executable,
                                                                 std::string_view symbol_name,
                                                                 uint64_t symbol,
                                                                 uint64_t kernel_object) {
  const auto normalize = [](std::string_view name) {
    if (name.ends_with(".kd"))
      name.remove_suffix(3);
    return name;
  };
  const std::string_view normalized = normalize(symbol_name);
  const auto pending = std::ranges::find_if(pending_, [&](const Pending &candidate) {
    return candidate.executable == executable && normalize(candidate.kernel_name) == normalized;
  });
  if (pending == pending_.end())
    return std::nullopt;

  const auto bound = std::ranges::find_if(
      bound_, [&](const Bound &candidate) { return candidate.symbol == symbol; });
  if (bound == bound_.end()) {
    bound_.push_back({symbol, kernel_object, pending->required_private_bytes});
  } else {
    bound->kernel_object = kernel_object;
    bound->required_private_bytes =
        std::max(bound->required_private_bytes, pending->required_private_bytes);
  }
  return pending->required_private_bytes;
}

std::optional<uint32_t> PrivateDispatchRequirements::required_for_symbol(uint64_t symbol) const {
  const auto bound = std::ranges::find_if(
      bound_, [&](const Bound &candidate) { return candidate.symbol == symbol; });
  return bound == bound_.end() ? std::nullopt
                               : std::optional<uint32_t>(bound->required_private_bytes);
}

std::optional<uint32_t>
PrivateDispatchRequirements::required_for_kernel_object(uint64_t kernel_object) const {
  const auto bound = std::ranges::find_if(
      bound_, [&](const Bound &candidate) { return candidate.kernel_object == kernel_object; });
  return bound == bound_.end() ? std::nullopt
                               : std::optional<uint32_t>(bound->required_private_bytes);
}

void PrivateDispatchRequirements::clear() {
  pending_.clear();
  bound_.clear();
}

uint32_t required_dispatch_private_segment_size(uint32_t packet_private_bytes,
                                                std::optional<uint32_t> required_private_bytes) {
  return required_private_bytes ? std::max(packet_private_bytes, *required_private_bytes)
                                : packet_private_bytes;
}

} // namespace rocjitsu
