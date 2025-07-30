////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_AMD_BLIT_SDMA_H_
#define HSA_RUNTIME_CORE_INC_AMD_BLIT_SDMA_H_

#include <mutex>
#include <stdint.h>
#include <vector>

#include "core/inc/amd_gpu_agent.h"
#include "core/inc/blit.h"
#include "core/inc/runtime.h"
#include "core/inc/signal.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

class BlitSdmaBase : public core::Blit {
 public:
  static const size_t kQueueSize;
  static const size_t kCopyPacketSize;
  static const size_t kMaxSingleCopySize;
  static const size_t kMaxSingleFillSize;
  virtual bool isSDMA() const override { return true; }
  virtual hsa_status_t Initialize(const core::Agent& agent, bool use_xgmi,
                                  size_t linear_copy_size_override, int rec_engine) = 0;
  virtual hsa_status_t SubmitCopyRectCommand(const hsa_pitched_ptr_t* dst,
                                             const hsa_dim3_t* dst_offset,
                                             const hsa_pitched_ptr_t* src,
                                             const hsa_dim3_t* src_offset, const hsa_dim3_t* range,
                                             std::vector<core::Signal*>& dep_signals,
                                             core::Signal& out_signal) = 0;
};

// RingIndexTy: 32/64-bit monotonic ring index, counting in bytes.
// HwIndexMonotonic: true if SDMA HW index is monotonic, false if it wraps at end of ring.
// SizeToCountOffset: value added to size (in bytes) to form SDMA command count field.
template <typename RingIndexTy, bool HwIndexMonotonic, int SizeToCountOffset, bool useGCR>
class BlitSdma : public BlitSdmaBase {
 public:
  BlitSdma();

  virtual ~BlitSdma() override;

  /// @brief Initialize a User Mode SDMA Queue object. Input parameters specify
  /// properties of queue being created.
  ///
  /// @param agent Pointer to the agent that will execute the PM4 commands.
  ///
  /// @return hsa_status_t
  virtual hsa_status_t Initialize(const core::Agent& agent, bool use_xgmi,
                                  size_t linear_copy_size_override, int rec_eng) override;

  /// @brief Marks the queue object as invalid and uncouples its link with
  /// the underlying compute device's control block. Use of queue object
  /// once it has been release is illegal and any behavior is indeterminate
  ///
  /// @note: The call will block until all packets have executed.
  ///
  /// @param agent Agent passed to Initialize.
  ///
  /// @return hsa_status_t
  virtual hsa_status_t Destroy(const core::Agent& agent) override;

  /// @brief Submit a linear copy command to the queue buffer.
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

  virtual hsa_status_t SubmitCopyRectCommand(const hsa_pitched_ptr_t* dst,
                                             const hsa_dim3_t* dst_offset,
                                             const hsa_pitched_ptr_t* src,
                                             const hsa_dim3_t* src_offset, const hsa_dim3_t* range,
                                             std::vector<core::Signal*>& dep_signals,
                                             core::Signal& out_signal) override;

  /// @brief Submit a linear fill command to the queue buffer
  ///
  /// @param ptr Memory address of the fill destination.
  /// @param value Value to be set.
  /// @param count Number of uint32_t element to be set to the value.
  virtual hsa_status_t SubmitLinearFillCommand(void* ptr, uint32_t value,
                                               size_t count) override;

  virtual hsa_status_t EnableProfiling(bool enable) override;

  virtual uint64_t PendingBytes() override;
  virtual void GangLeader(bool gang_leader) override { gang_leader_ = gang_leader; }
  virtual bool GangLeader() const override { return gang_leader_; }

 private:
  /// @brief Acquires the address into queue buffer where a new command
  /// packet of specified size could be written. The address that is
  /// returned is guaranteed to be unique even in a multi-threaded access
  /// scenario. This function is guaranteed to return a pointer for writing
  /// data into the queue buffer.
  ///
  /// @param cmd_size Command packet size in bytes.
  ///
  /// @param curr_index (output) Index to pass to ReleaseWriteAddress.
  ///
  /// @return pointer into the queue buffer where a PM4 packet of specified size
  /// could be written. NULL if input size is greater than the size of queue
  /// buffer.

  char* AcquireWriteAddress(uint32_t cmd_size, RingIndexTy& curr_index);

  void UpdateWriteAndDoorbellRegister(RingIndexTy curr_index, RingIndexTy new_index);

  /// @brief Updates the Write Register of compute device to the end of
  /// SDMA packet written into queue buffer. The update to Write Register
  /// will be safe under multi-threaded usage scenario. Furthermore, updates
  /// to Write Register are blocking until all prior updates are completed
  /// i.e. if two threads T1 & T2 were to call release, then updates by T2
  /// will block until T1 has completed its update (assumes T1 acquired the
  /// write address first).
  ///
  /// @param curr_index Index passed back from AcquireWriteAddress.
  ///
  /// @param cmd_size Command packet size in bytes.
  void ReleaseWriteAddress(RingIndexTy curr_index, uint32_t cmd_size);

  /// @brief Writes NO-OP words into queue buffer in case writing a command
  /// causes the queue buffer to wrap.
  ///
  /// @param curr_index Index to begin padding from.
  void PadRingToEnd(RingIndexTy curr_index);

  uint32_t WrapIntoRing(RingIndexTy index);
  bool CanWriteUpto(RingIndexTy upto_index);

  /// @brief Build fence command
  void BuildFenceCommand(char* fence_command_addr, uint32_t* fence,
                         uint32_t fence_value);

  /// @brief Build Hdp Flush command
  void BuildHdpFlushCommand(char* cmd_addr);

  void BuildCopyCommand(char* cmd_addr, uint32_t num_copy_command, void* dst,
                        const void* src, size_t size);

  void BuildCopyRectCommand(const std::function<void*(size_t)>& append,
                            const hsa_pitched_ptr_t* dst, const hsa_dim3_t* dst_offset,
                            const hsa_pitched_ptr_t* src, const hsa_dim3_t* src_offset,
                            const hsa_dim3_t* range);

  void BuildFillCommand(char* cmd_addr, uint32_t num_fill_command, void* ptr, uint32_t value,
                        size_t count);

  void BuildPollCommand(char* cmd_addr, void* addr, uint32_t reference);

  void BuildAtomicDecrementCommand(char* cmd_addr, void* addr);

  void BuildGetGlobalTimestampCommand(char* cmd_addr, void* write_address);

  void BuildTrapCommand(char* cmd_addr, uint32_t event_id);

  void BuildGCRCommand(char* cmd_addr, bool invalidate);

  hsa_status_t SubmitCommand(const void* cmds, size_t cmd_size, uint64_t size,
                             const std::vector<core::Signal*>& dep_signals,
                             core::Signal& out_signal, std::vector<core::Signal*>& gang_signals);

  hsa_status_t SubmitBlockingCommand(const void* cmds, size_t cmd_size, uint64_t size);

  // Agent object owning the SDMA engine.
  GpuAgent* agent_;

  /// Base address of the Queue buffer at construction time.
  char* queue_start_addr_;

  // Pending bytes tracking
  // bytes_written_ is indexed with wrapped command queue indices (which are in bytes).
  // The data_ index corresponding to a command queue index is the first uint64_t index which begins
  // in the packet area.  All packets have a header & at least one address so must be larger than 12
  // bytes, thus this index always exists.
  std::mutex reservation_lock_;
  uint64_t bytes_queued_;
  class {
   public:
    // Indexed by wrapped command queue indices (offsets).
    uint64_t& operator[](uint32_t index) { return data_[convert(index)]; }

    void resize(size_t size) { data_.resize(convert(size)); }

    void fill(uint32_t start, uint32_t stop, uint64_t value) {
      for (uint32_t i = convert(start); i < convert(stop); i++) {
        data_[i] = value;
      }
    }

   private:
    uint32_t convert(uint32_t index) { return (index + sizeof(uint64_t) - 1) / sizeof(uint64_t); }

    std::vector<uint64_t> data_;
  } bytes_written_;

  // Internal signals for blocking APIs
  core::unique_signal_ptr signals_[2];
  KernelMutex lock_;
  bool parity_;

  /// Queue resource descriptor for doorbell, read
  /// and write indices
  HsaQueueResource queue_resource_;

  // Monotonic ring indices, in bytes, tracking written and submitted commands.
  RingIndexTy cached_reserve_index_;
  RingIndexTy cached_commit_index_;

  static const uint32_t linear_copy_command_size_;

  static const uint32_t fill_command_size_;

  static const uint32_t fence_command_size_;

  static const uint32_t poll_command_size_;

  static const uint32_t flush_command_size_;

  static const uint32_t atomic_command_size_;

  static const uint32_t timestamp_command_size_;

  static const uint32_t trap_command_size_;

  static const uint32_t gcr_command_size_;

  // Max copy size of a single linear copy command packet.
  size_t max_single_linear_copy_size_;

  /// Max total copy size supported by the queue.
  size_t max_total_linear_copy_size_;

  /// Max count of uint32_t of a single fill command packet.
  size_t max_single_fill_size_;

  /// Max total fill count supported by the queue.
  size_t max_total_fill_size_;

  /// True if platform atomic is supported.
  bool platform_atomic_support_;

  /// True if sDMA supports HDP flush
  bool hdp_flush_support_;

  /// True if SDMA blit is gang leader
  bool gang_leader_;

  /// True if SDMA blit is ganged
  bool is_ganged_;

  /// Minimum submission size in bytes.
  size_t min_submission_size_;
};

// Ring indices are 32-bit.
// HW ring indices are not monotonic (wrap at end of ring).
// Count fields of SDMA commands are 0-based.
typedef BlitSdma<uint32_t, false, 0, false> BlitSdmaV2V3;

// Ring indices are 64-bit.
// HW ring indices are monotonic (do not wrap at end of ring).
// Count fields of SDMA commands are 1-based.
typedef BlitSdma<uint64_t, true, -1, false> BlitSdmaV4;

// Ring indices are 64-bit.
// HW ring indices are monotonic (do not wrap at end of ring).
// Count fields of SDMA commands are 1-based.
// SDMA is connected to gL2.
typedef BlitSdma<uint64_t, true, -1, true> BlitSdmaV5;

}  // namespace amd
}  // namespace rocr

#endif  // header guard
