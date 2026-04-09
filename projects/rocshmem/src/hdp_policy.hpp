/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal in the Software without restriction, including without limitation the
 * rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
 * IN THE SOFTWARE.
 *****************************************************************************/

#ifndef LIBRARY_SRC_HDP_POLICY_HPP_
#define LIBRARY_SRC_HDP_POLICY_HPP_

#include <hip/hip_runtime.h>
#include <hsa/hsa_ext_amd.h>

#include "rocshmem/rocshmem_config.h"  // NOLINT(build/include_subdir)
#include "memory/hip_allocator.hpp"
#include "util.hpp"

namespace rocshmem {

class HdpHostSideFlushRocmPolicy {
 public:
  HdpHostSideFlushRocmPolicy() { set_hdp_flush_ptr(); }
  ~HdpHostSideFlushRocmPolicy() {}

  /**
   * @brief Host-side API to set the flush polling ptr
   * @param hdp_gpu_cpu_flush_flag A pointer allocated on the symmetric heap
   */
  __host__ void set_flush_polling_ptr(unsigned int* hdp_gpu_cpu_flush_flag) {
    hdp_gpu_cpu_flush_flag_ = hdp_gpu_cpu_flush_flag;
    if (hdp_gpu_cpu_flush_flag_ != nullptr) {
      *hdp_gpu_cpu_flush_flag_ =
          static_cast<std::underlying_type_t<hdp_poll_flag>>(
              hdp_poll_flag::NO_FLUSH);
    }
  }

  /**
   * @brief Host-side API that performs the flush.
   */
  __host__ void hdp_flush() { *hdp_flush_ptr_ = HDP_FLUSH_VAL; }

  /**
   * @brief Device-side API that signal to the CPU spinning thread to do the
   * flush on behalf of the GPU.
   *
   * @param hdp_ptr A pointer to use as the HDP flush ptr
   */
  __device__ static void hdp_flush(unsigned int* hdp_ptr) {
    STORE(hdp_ptr, static_cast<unsigned int>(hdp_poll_flag::FLUSH));
  }

  /**
   * @brief Device-side API that signal to the CPU spinning thread to do the
   * flush on behalf of the GPU.
   */
  __device__ void hdp_flush() { hdp_flush(hdp_gpu_cpu_flush_flag_); }

  __device__ void flushCoherency() { hdp_flush(); }

  /**
   * @brief Get the hdp flush signal flag.
   *
   * @return A pointer to an object that stores a signal for the CPU thread to
   * perform the flush on the host.
   */
  __host__ unsigned int* get_hdp_flush_ptr() const {
    return hdp_gpu_cpu_flush_flag_;
  }

  /**
   * @brief Check if there is an active request to flush the hdp cache.
   *
   * @return A boolean indicating whether there is a flush request.
   */
  __host__ bool has_active_flush_request() const {
    if (hdp_gpu_cpu_flush_flag_ == nullptr) {
      return false;
    }
    auto device_flag_value = __hip_atomic_load(
        hdp_gpu_cpu_flush_flag_, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
    const auto flush_value = static_cast<std::underlying_type_t<hdp_poll_flag>>(
        hdp_poll_flag::FLUSH);
    return device_flag_value == flush_value;
  }

  /**
   * @brief Clears the flag that signals hdp cache flush.
   */
  __host__ void clear_active_flush_flag() {
    const auto no_flush_value =
        static_cast<std::underlying_type_t<hdp_poll_flag>>(
            hdp_poll_flag::NO_FLUSH);
    __hip_atomic_store(hdp_gpu_cpu_flush_flag_, no_flush_value,
                       __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
  }

  static const int HDP_FLUSH_VAL{0x01};

  enum class hdp_poll_flag : unsigned int {
    NO_FLUSH = 0x0,
    FLUSH = 0x1,
  };

 private:
  void set_hdp_flush_ptr() {
    int hip_dev_id{};
    CHECK_HIP(hipGetDevice(&hip_dev_id));
    CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                                    hipDeviceAttributeHdpMemFlushCntl,
                                    hip_dev_id));
  }

  unsigned int* hdp_flush_ptr_{nullptr};
  unsigned int* hdp_gpu_cpu_flush_flag_{nullptr};
};

class HdpDeviceSideFlushRocmPolicy {
 public:
  HdpDeviceSideFlushRocmPolicy() { set_hdp_flush_ptr(); }
  ~HdpDeviceSideFlushRocmPolicy() {}

  /**
   * @brief Flush the HDP by setting the flush control signal
   */
  __host__ __device__ void hdp_flush() {
    STORE(hdp_flush_ptr_, static_cast<unsigned int>(HDP_FLUSH_VAL));
  }
  __device__ void flushCoherency() { hdp_flush(); }

  /**
   * @brief Get the hdp flush signal flag.
   *
   * @return A pointer to the HDP flush control signal
   */
  __host__ unsigned int* get_hdp_flush_ptr() const { return hdp_flush_ptr_; }

  static const int HDP_FLUSH_VAL{0x01};

 private:
  void set_hdp_flush_ptr() {
    int hip_dev_id{};
    CHECK_HIP(hipGetDevice(&hip_dev_id));
    CHECK_HIP(hipDeviceGetAttribute(reinterpret_cast<int*>(&hdp_flush_ptr_),
                                    hipDeviceAttributeHdpMemFlushCntl,
                                    hip_dev_id));
  }

  unsigned int* hdp_flush_ptr_{nullptr};
};

class NoHdpPolicy {
 public:
  NoHdpPolicy() = default;

  __host__ void hdp_flush() {}

  __host__ unsigned int* get_hdp_flush_ptr() const { return nullptr; }

  __device__ void hdp_flush() {}

  __device__ void flushCoherency() { __roc_flush(); }
};

/*
 * Select which one of our HDP policies to use at compile time.
 */
#if defined USE_HDP_FLUSH
// Only when we are using the IB conduit, we have to use a polling thread to
// flush the HDP cache on the GPU's behalf.
#if defined USE_HDP_FLUSH_HOST_SIDE
typedef HdpHostSideFlushRocmPolicy HdpPolicy;
#else
typedef HdpDeviceSideFlushRocmPolicy HdpPolicy;
#endif
#else
typedef NoHdpPolicy HdpPolicy;
#endif

}  // namespace rocshmem

#endif  // LIBRARY_SRC_HDP_POLICY_HPP_
