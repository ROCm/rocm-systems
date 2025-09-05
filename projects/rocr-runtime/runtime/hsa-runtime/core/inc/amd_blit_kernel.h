////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
// 
// Copyright (c) 2014-2020, Advanced Micro Devices, Inc. All rights reserved.
// 
// Developed by:
// 
//                 AMD Research and AMD HSA Software Development
// 
//                 Advanced Micro Devices, Inc.
// 
//                 www.amd.com
// 
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
// 
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
// 
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

#ifndef HSA_RUNTIME_CORE_INC_AMD_BLIT_KERNEL_H_
#define HSA_RUNTIME_CORE_INC_AMD_BLIT_KERNEL_H_

#include <map>
#include <mutex>
#include <vector>
#include <atomic>
#include <stdint.h>

#include "core/inc/blit.h"

namespace rocr {
namespace AMD {
class BlitKernel : public core::Blit {
 public:
  explicit BlitKernel(core::Queue* queue);
  virtual ~BlitKernel() override;

  /// @brief Initialize a blit kernel object.
  ///
  /// @param agent Pointer to the agent that will execute the AQL packets.
  ///
  /// @return hsa_status_t
  hsa_status_t Initialize(const core::Agent& agent);

  /// @brief Marks the blit kernel object as invalid and uncouples its link with
  /// the underlying AQL kernel queue. Use of the blit object
  /// once it has been release is illegal and any behavior is indeterminate
  ///
  /// @note: The call will block until all AQL packets have been executed.
  ///
  /// @param agent Agent passed to Initialize.
  ///
  /// @return hsa_status_t
  virtual hsa_status_t Destroy(const core::Agent& agent) override;

  /// @brief Submit an AQL packet to perform vector copy. The call is blocking
  /// until the command execution is finished.
  ///
  /// @param dst Memory address of the copy destination.
  /// @param src Memory address of the copy source.
  /// @param size Size of the data to be copied.
  virtual hsa_status_t SubmitLinearCopyCommand(void* dst, const void* src,
                                               size_t size) override;

  /// @brief Submit a linear copy command to the the underlying compute device's
  /// control block. The call is non blocking. The memory transfer will start
  /// after all dependent signals are satisfied. After the transfer is
  /// completed, the out signal will be decremented.
  ///
  /// @param dst Memory address of the copy destination.
  /// @param src Memory address of the copy source.
  /// @param size Size of the data to be copied.
  /// @param dep_signals Arrays of dependent signal.
  /// @param out_signal Output signal.
  /// @param gang_signals Array of gang signals.
  virtual hsa_status_t SubmitLinearCopyCommand(
      void* dst, const void* src, size_t size,
      std::vector<core::Signal*>& dep_signals,
      core::Signal& out_signal, std::vector<core::Signal*>& gang_signals) override;

  /// @brief Submit an AQL packet to perform memory fill. The call is blocking
  /// until the command execution is finished.
  ///
  /// @param ptr Memory address of the fill destination.
  /// @param value Value to be set.
  /// @param count Number of uint32_t element to be set to the value.
  virtual hsa_status_t SubmitLinearFillCommand(void* ptr, uint32_t value,
                                               size_t count) override;

  virtual hsa_status_t EnableProfiling(bool enable) override;

  virtual uint64_t PendingBytes() override;

  virtual void GangLeader(bool gang_leader) override {}
  virtual bool GangLeader() const override { return false; }

  const uint16_t kInvalidPacketHeader = HSA_PACKET_TYPE_INVALID;
 private:
  union KernelArgs {
    struct __ALIGNED__(16) {
      uint64_t phase1_src_start;
      uint64_t phase1_dst_start;
      uint64_t phase2_src_start;
      uint64_t phase2_dst_start;
      uint64_t phase3_src_start;
      uint64_t phase3_dst_start;
      uint64_t phase4_src_start;
      uint64_t phase4_dst_start;
      uint64_t phase4_src_end;
      uint64_t phase4_dst_end;
      uint32_t num_workitems;
    } copy_aligned;

    struct __ALIGNED__(16) {
      uint64_t phase1_src_start;
      uint64_t phase1_dst_start;
      uint64_t phase2_src_start;
      uint64_t phase2_dst_start;
      uint64_t phase2_src_end;
      uint64_t phase2_dst_end;
      uint32_t num_workitems;
    } copy_misaligned;

    struct __ALIGNED__(16) {
      uint64_t phase1_dst_start;
      uint64_t phase2_dst_start;
      uint64_t phase2_dst_end;
      uint32_t fill_value;
      uint32_t num_workitems;
    } fill;
  };

  // Index after which bytes will have been written.
  struct BytesWritten {
    uint64_t index;
    uint64_t bytes;
  };

  /// Reserve a slot in the queue buffer. The call will wait until the queue
  /// buffer has a room.
  uint64_t AcquireWriteIndex(uint32_t num_packet);

  /// Update the queue doorbell register with ::write_index. This
  /// function also serializes concurrent doorbell update to ensure that the
  /// packet processor doesn't get invalid packet.
  void ReleaseWriteIndex(uint64_t write_index, uint32_t num_packet);

  void PopulateQueue(uint64_t index, uint64_t code_handle, void* args,
                     uint32_t grid_size_x, hsa_signal_t completion_signal);

  KernelArgs* ObtainAsyncKernelCopyArg();

  void RecordBlitHistory(uint64_t size, uint64_t index);

  /// AQL code object and size for each kernel.
  enum class KernelType {
    CopyAligned,
    CopyMisaligned,
    Fill,
  };

  struct KernelCode {
    void* code_buf_;
    size_t code_buf_size_;
  };

  std::map<KernelType, KernelCode> kernels_;

  /// AQL queue for submitting the vector copy kernel.
  core::Queue* queue_;
  uint32_t queue_bitmask_;

  /// Pointer to the kernel argument buffer.
  KernelArgs* kernarg_async_;
  uint32_t kernarg_async_mask_;
  volatile uint32_t kernarg_async_counter_;

  /// Completion signal for every kernel dispatched.
  hsa_signal_t completion_signal_;

  /// Bytes moved by commands < index.
  /// Any record's byte value may be inexact by the size of concurrently issued operations.
  std::vector<BytesWritten> bytes_written_;

  /// Total bytes written by all commands issued.
  uint64_t bytes_queued_;

  /// Index where most recent blit operation queued.
  uint64_t last_queued_;

  /// Orders command indices and bytes_queued_ updates
  std::mutex reservation_lock_;

  /// Search resume index
  std::atomic<uint64_t> pending_search_index_;

  /// Lock to synchronize access to kernarg_ and completion_signal_
  std::mutex lock_;

  /// Number of CUs on the underlying agent.
  int num_cus_;
};
}  // namespace amd
}  // namespace rocr

#endif  // header guard
