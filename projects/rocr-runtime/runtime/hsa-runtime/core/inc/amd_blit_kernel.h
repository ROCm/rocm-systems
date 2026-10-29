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

#include <functional>
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
  /// @return hsa_status_t
  virtual hsa_status_t Destroy() override;

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

  /// @brief The compute-shader blit is the swap fallback wherever the matching
  /// shader is built for the agent's ISA. Initialize() registers a kernel only
  /// when its shader exists, so presence is the capability.
  bool SupportsSwapCopyCommand() const override {
    return kernels_.count(KernelType::SwapCopy) != 0;
  }

  /// @brief Submit a batch of swap copy commands. Entries and oversized
  /// entries are split across dispatches; all of them share @p out_signal.
  virtual hsa_status_t SubmitSwapCopyCommand(
      uint32_t num_entries, void* const* addr_a_list, void* const* addr_b_list,
      const size_t* size_list, std::vector<core::Signal*>& dep_signals,
      core::Signal& out_signal) override;

  /// Keep the base class' single-pair convenience overload visible; declaring
  /// the N-entry override above would otherwise hide it for callers holding a
  /// BlitKernel rather than a core::Blit.
  using core::Blit::SubmitSwapCopyCommand;

  const uint16_t kInvalidPacketHeader = HSA_PACKET_TYPE_INVALID;

  /// Bytes one swap dispatch exchanges. KernelArgs::swap::swap_size is a
  /// uint32_t, and the bound is rounded down to a multiple of the shader's
  /// 16-byte chunk so every chunk boundary stays 16-byte aligned - the shader
  /// splits its grid into a 16-byte phase and a trailing-byte phase, and a
  /// boundary that was not a multiple of 16 would give the tail of one chunk
  /// and the head of the next inconsistent phase assignments.
  static constexpr size_t kMaxSwapBytesPerDispatch = size_t(UINT32_MAX) & ~size_t(15);

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

    struct __ALIGNED__(16) {
      uint64_t addr_a;     // First buffer address
      uint64_t addr_b;     // Second buffer address
      uint32_t swap_size;  // Size to swap in bytes
    } swap;
  };

  // Index after which bytes will have been written.
  struct BytesWritten {
    uint64_t index;
    uint64_t bytes;
  };

  /// True when the ring is large enough to hold num_packet packets at once.
  /// A reservation that fails this can never be satisfied, so the submission
  /// has to be rejected rather than waited on.
  bool ReservationFits(uint32_t num_packet) const;

  /// Reserve a slot in the queue buffer. The call will wait until the queue
  /// buffer has a room. Must only be called for a reservation that satisfies
  /// ReservationFits.
  uint64_t AcquireWriteIndex(uint32_t num_packet);

  /// Update the queue doorbell register with ::write_index. This
  /// function also serializes concurrent doorbell update to ensure that the
  /// packet processor doesn't get invalid packet.
  void ReleaseWriteIndex(uint64_t write_index, uint32_t num_packet);

  void PopulateQueue(uint64_t index, uint64_t code_handle, void* args,
                     uint32_t grid_size_x, hsa_signal_t completion_signal);

  /// Emit BARRIER_AND packets to gate the following dispatch on @p dep_signals.
  /// Packets are written starting at @p barrier_start_index (one packet per up
  /// to five dependent signals). Shared by the linear and swap copy submit
  /// paths. @p completion_signal is attached to the last barrier packet;
  /// pass a zero handle when a following dispatch carries the signal instead.
  void SubmitDependentSignalBarriers(uint64_t barrier_start_index,
                                     std::vector<core::Signal*>& dep_signals,
                                     hsa_signal_t completion_signal = {0});

  /// Complete a zero-byte copy. Still honors @p dep_signals: @p out_signal is
  /// only released once they have fired.
  hsa_status_t SubmitEmptyCopyCommand(std::vector<core::Signal*>& dep_signals,
                                      core::Signal& out_signal);

  /// Dispatch packets that still fit in one queue reservation once
  /// @p num_barrier_packet barrier packets have taken their place in it.
  /// AcquireWriteIndex asserts the whole reservation fits in the queue, so a
  /// request needing more dispatches than this is split across reservations.
  /// Never returns zero.
  uint32_t MaxDispatchesPerReservation(uint32_t num_barrier_packet) const;

  /// Run @p submit on a runtime handler thread once every signal in
  /// @p pending_deps has reached zero, and return immediately.
  ///
  /// AcquireWriteIndex blocks the calling thread until the packet processor
  /// retires enough earlier packets to free queue space. That is only safe
  /// while those packets can retire unaided. Dependency barriers break that:
  /// they sit at the head of the first reservation waiting on signals that
  /// only the caller can satisfy, so a batch large enough to need a second
  /// reservation would spin forever inside the very call the caller has to
  /// return from in order to satisfy them.
  ///
  /// Deferring keeps the two mutually exclusive. A batch that fits in one
  /// reservation carries its barriers in it as before; one that does not waits
  /// off-queue instead and is then submitted with nothing left to gate it, so
  /// its reservations drain on GPU progress alone. @p submit must therefore
  /// not carry dependencies, and must already be past every failure path -
  /// there is no caller left to return a status to.
  hsa_status_t DeferUntilDepsResolved(const std::vector<core::Signal*>& pending_deps,
                                      std::function<void()> submit);

  KernelArgs* ObtainAsyncKernelCopyArg();

  void RecordBlitHistory(uint64_t size, uint64_t index);

  /// AQL code object and size for each kernel.
  enum class KernelType {
    CopyAligned,
    CopyMisaligned,
    Fill,
    SwapCopy,
  };

  struct KernelCode {
    void* code_buf_;
    size_t code_buf_size_;
  };

  std::map<KernelType, KernelCode> kernels_;

  const core::Agent* agent_;

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
