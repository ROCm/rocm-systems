/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

/*! \addtogroup GPU GPU Device Implementation
 *  @{
 */

namespace amd::roc {

class Device;

//! Per-stream assert fault record buffer (flag + site id), printf-shaped helper.
class AssertFaultBuffer {
 public:
  static constexpr size_t kSize = 8;
  static constexpr size_t kAlign = 8;

  explicit AssertFaultBuffer(Device& device);
  ~AssertFaultBuffer();

  //! Allocates the fixed-size assert fault buffer on first use.
  bool allocate();

  //! Returns the device buffer pointer, or nullptr if never allocated.
  address faultBuffer() const { return faultBuffer_; }

 protected:
  address faultBuffer_;       //!< Buffer to hold assert fault record
  size_t faultBuffer_size_;   //!< Size of the assert fault buffer
  Device& gpuDevice_;         //!< GPU device object

  //! Gets GPU device object
  Device& dev() const { return gpuDevice_; }

 private:
  AssertFaultBuffer(const AssertFaultBuffer&) = delete;
  AssertFaultBuffer& operator=(const AssertFaultBuffer&) = delete;
};

}  // namespace amd::roc
