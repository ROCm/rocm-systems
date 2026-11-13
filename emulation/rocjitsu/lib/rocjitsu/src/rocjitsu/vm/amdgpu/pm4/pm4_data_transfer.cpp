// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocjitsu/vm/amdgpu/pm4/pm4_data_transfer.h"

#include <utility>

namespace rocjitsu::amdgpu {

Pm4DataTransfer::Pm4DataTransfer(std::optional<uint64_t> source, uint64_t destination,
                                 std::vector<std::byte> data, uint64_t root_cursor,
                                 uint32_t root_ring_bytes, const char *read_failure,
                                 const char *write_failure)
    : root_cursor(root_cursor), root_ring_bytes(root_ring_bytes), source_(source),
      destination_(destination), data_(std::move(data)), read_failure_(read_failure),
      write_failure_(write_failure) {}

VmAccessOutcome Pm4DataTransfer::resume(const GpuVmAccess &access) {
  if (source_) {
    const auto outcome = access.read(*source_, data_, completed_bytes_, token_);
    if (outcome != VmAccessOutcome::Complete)
      return outcome;
    source_.reset();
    completed_bytes_ = 0;
  }
  return access.write(destination_, data_, completed_bytes_, token_);
}

} // namespace rocjitsu::amdgpu
