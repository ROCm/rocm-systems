// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/code/patch/trampoline_builder.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>

namespace {

uint64_t take_u64(std::span<const uint8_t> bytes, size_t offset) {
  uint64_t value = 0;
  const size_t count = std::min(sizeof(value), bytes.size() - offset);
  std::memcpy(&value, bytes.data() + offset, count);
  return value;
}

} // namespace

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  if (size < 8)
    return 0;
  const std::span<const uint8_t> bytes(data, size);
  const uint64_t appended_offset = (take_u64(bytes, 0) & 0xFFFFCu);
  rocjitsu::DbiPatchPlacementPlanner planner(ROCJITSU_CODE_ARCH_RDNA4, appended_offset);

  for (size_t offset = 8; offset < size && offset < 4096; offset += 32) {
    const size_t available = std::min<size_t>(32, size - offset);
    const std::span<const uint8_t> record = bytes.subspan(offset, available);
    rocjitsu::DbiPatchPlacementRequest request;
    request.anchor_offset = take_u64(record, 0) & 0xFFFFCu;
    request.original_size = (take_u64(record, std::min<size_t>(8, available - 1)) & 0x3FCu);
    request.body_size = (take_u64(record, std::min<size_t>(16, available - 1)) & 0x7FCu);
    request.inline_capacity = (take_u64(record, std::min<size_t>(24, available - 1)) & 0x7FCu);
    request.allow_appended_cave = (record.front() & 1u) != 0;
    if ((record.front() & 2u) != 0) {
      request.local_cave = rocjitsu::DbiPatchLocalCave{
          (request.anchor_offset + 4u + (request.body_size & 0xFFCu)) & 0xFFFFCu,
          request.body_size + (record.front() & 0x3Cu)};
    }
    (void)planner.plan(request);
  }
  return 0;
}
