/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "streamCommon.hh"

namespace hip {

inline namespace internal {

bool checkStreamPriority_(hipStream_t stream, bool checkPriority = false, int priority_ = 0) {
  int priority{0};
  HIP_CHECK(hipStreamGetPriority(stream, &priority));
  if (checkPriority) {
    if (priority_ != priority) {
      UNSCOPED_INFO("Priority Mismatch, Expected Priority: " << priority_
                                                             << " Actual Priority: " << priority);
      return false;
    }
  } else {
    int priority_low{0}, priority_high{0};
    HIP_CHECK(hipDeviceGetStreamPriorityRange(&priority_low, &priority_high));
    if (priority_low < priority || priority_high > priority) {
      UNSCOPED_INFO("Priority Mismatch, Expected Priority Range: "
                    << priority_low << " - " << priority_high << " Actual Priority: " << priority);
      return false;
    }
  }
  return true;
}

bool checkStreamFlags_(hipStream_t stream, bool checkFlags = false, unsigned flags_ = 0) {
  unsigned flags{0};
  HIP_CHECK(hipStreamGetFlags(stream, &flags));
  if (checkFlags) {
    if (flags_ != flags) {
      UNSCOPED_INFO("Flags Mismatch, Expected Flag: " << flags_ << " Actual Flag: " << flags);
      return false;
    }
  } else {
    if (flags != hipStreamDefault && flags != hipStreamNonBlocking) {
      UNSCOPED_INFO("Flags Mismatch, Expected Flag: " << hipStreamDefault << " or "
                                                      << hipStreamNonBlocking
                                                      << " Actual Flag: " << flags);
      return false;
    }
  }
  return true;
}
}  // namespace internal

inline namespace stream {

/* Empty kernel to ensure work finishes on the stream quickly */
__global__ void empty_kernel() {}

bool checkStream(hipStream_t stream) {
  {  // Check default flags
    auto res = checkStreamFlags_(stream, true, hipStreamDefault);
    if (!res) return false;
  }

  {  // Check default Priority
    auto res = checkStreamPriority_(stream);
    if (!res) return false;
  }

  return true;
}

bool checkStreamPriorityAndFlags(hipStream_t stream, int priority, unsigned int flags) {
  {  // Check flags
    auto res = checkStreamFlags_(stream, true, flags);
    if (!res) return false;
  }

  {  // Check priority
    auto res = checkStreamPriority_(stream, true, priority);
    if (!res) return false;
  }

  return true;
}

}  // namespace stream
}  // namespace hip
