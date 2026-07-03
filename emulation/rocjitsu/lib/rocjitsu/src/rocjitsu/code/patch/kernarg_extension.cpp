// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/kernarg_extension.h"

#include <cstring>
#include <limits>

namespace rocjitsu {
namespace {

[[nodiscard]] bool is_power_of_two(uint32_t value) {
  return value != 0 && (value & (value - 1u)) == 0;
}

[[nodiscard]] std::optional<uint32_t> align_up_u32(uint32_t value, uint32_t alignment) {
  if (!is_power_of_two(alignment))
    return std::nullopt;
  const uint32_t mask = alignment - 1u;
  if (value > std::numeric_limits<uint32_t>::max() - mask)
    return std::nullopt;
  return (value + mask) & ~mask;
}

} // namespace

std::optional<KernargExtensionLayout>
make_kernarg_extension_layout(uint32_t original_kernarg_size,
                              std::span<const KernargExtensionPayloadLayout> payloads) {
  KernargExtensionLayout layout;
  layout.original_kernarg_size = original_kernarg_size;

  // The saved original pointer is part of the extension area. Keep it naturally
  // aligned so the entry prologue can load it with one scalar dwordx2 SMEM op.
  auto cursor = align_up_u32(original_kernarg_size, alignof(uint64_t));
  if (!cursor)
    return std::nullopt;
  layout.original_kernarg_pointer_offset = *cursor;
  if (*cursor > std::numeric_limits<uint32_t>::max() - sizeof(uint64_t))
    return std::nullopt;
  *cursor += static_cast<uint32_t>(sizeof(uint64_t));

  layout.payload_offsets.reserve(payloads.size());
  for (const KernargExtensionPayloadLayout &payload : payloads) {
    cursor = align_up_u32(*cursor, payload.alignment);
    if (!cursor || *cursor > std::numeric_limits<uint32_t>::max() - payload.size)
      return std::nullopt;
    layout.payload_offsets.push_back(*cursor);
    *cursor += payload.size;
  }

  layout.wrapper_size = *cursor;
  return layout;
}

bool write_kernarg_extension_wrapper(std::span<uint8_t> wrapper,
                                     const KernargExtensionLayout &layout,
                                     const void *original_kernarg,
                                     uint64_t original_kernarg_pointer,
                                     std::span<const KernargExtensionPayloadWrite> payloads) {
  if (wrapper.size() != layout.wrapper_size)
    return false;
  if (payloads.size() != layout.payload_offsets.size())
    return false;
  if (layout.wrapper_size < sizeof(uint64_t))
    return false;
  if (layout.original_kernarg_size != 0 && original_kernarg == nullptr)
    return false;
  if (layout.original_kernarg_pointer_offset >
      layout.wrapper_size - static_cast<uint32_t>(sizeof(uint64_t))) {
    return false;
  }

  std::memset(wrapper.data(), 0, wrapper.size());
  if (layout.original_kernarg_size != 0)
    std::memcpy(wrapper.data(), original_kernarg, layout.original_kernarg_size);
  std::memcpy(wrapper.data() + layout.original_kernarg_pointer_offset, &original_kernarg_pointer,
              sizeof(original_kernarg_pointer));

  for (size_t i = 0; i < payloads.size(); ++i) {
    const KernargExtensionPayloadWrite &payload = payloads[i];
    if (payload.size == 0)
      continue;
    if (payload.data == nullptr)
      return false;
    const uint32_t offset = layout.payload_offsets[i];
    if (offset > layout.wrapper_size || payload.size > layout.wrapper_size - offset)
      return false;
    std::memcpy(wrapper.data() + offset, payload.data, payload.size);
  }
  return true;
}

} // namespace rocjitsu
