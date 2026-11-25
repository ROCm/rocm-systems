// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "rocjitsu/vm/amdgpu/gpu_vm_access.h"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace rocjitsu::amdgpu {

/// @brief Submission-owned bytes and progress for one PM4 data-movement packet.
/// @details A copy finishes its read before starting the write. A write or fill
/// supplies its payload directly. Unavailable retains both the committed prefix
/// and outstanding request; destruction cancels that request before releasing
/// staging. The caller retires the root packet only after Complete.
class Pm4DataTransfer {
public:
  Pm4DataTransfer(std::optional<uint64_t> source, uint64_t destination, std::vector<std::byte> data,
                  uint64_t root_cursor, uint32_t root_ring_bytes, const char *read_failure,
                  const char *write_failure);

  [[nodiscard]] VmAccessOutcome resume(const GpuVmAccess &access);

  [[nodiscard]] const char *failure_message() const {
    return source_ ? read_failure_ : write_failure_;
  }

  /// Root cursor after the packet, in dwords; independent of nested IB state.
  uint64_t root_cursor;
  /// Root ring extent in bytes, or zero for an indirect/submission buffer.
  uint32_t root_ring_bytes;

private:
  /// Present until the complete source has been read into owned staging.
  std::optional<uint64_t> source_;
  /// Destination GPU virtual address for the complete payload.
  uint64_t destination_;
  /// Owned source result or captured immediate/register/timestamp payload.
  std::vector<std::byte> data_;
  /// Committed prefix of the current read or write phase, in bytes.
  std::size_t completed_bytes_ = 0;
  /// Static diagnostic for a failed source read.
  const char *read_failure_;
  /// Static diagnostic for a failed destination write.
  const char *write_failure_;
  /// Outstanding request, destroyed before the storage it may populate.
  VmTransferToken token_;
};

} // namespace rocjitsu::amdgpu
