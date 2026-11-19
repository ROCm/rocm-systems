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

#ifndef HSA_RUNTIME_CORE_INC_BLIT_H_
#define HSA_RUNTIME_CORE_INC_BLIT_H_

#include <stdint.h>

#include "core/inc/agent.h"

namespace rocr {
namespace core {
class Blit {
 public:
  explicit Blit() {}
  virtual ~Blit() {}

  /// @brief Marks the blit object as invalid and uncouples its link with
  /// the underlying compute device's control block. Use of blit object
  /// once it has been release is illegal and any behavior is indeterminate
  ///
  /// @note: The call will block until all commands have executed.
  ///
  /// @param agent Agent passed to Initialize.
  ///
  /// @return hsa_status_t
  virtual hsa_status_t Destroy() = 0;

  /// @brief Submit a linear copy command to the the underlying compute device's
  /// control block. The call is blocking until the command execution is
  /// finished.
  ///
  /// @param dst Memory address of the copy destination.
  /// @param src Memory address of the copy source.
  /// @param size Size of the data to be copied.
  virtual hsa_status_t SubmitLinearCopyCommand(void* dst, const void* src,
                                               size_t size) = 0;

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
      std::vector<core::Signal*>& dep_signals, core::Signal& out_signal,
      std::vector<core::Signal*>& gang_signals) = 0;

  /// @brief Submit a linear fill command to the the underlying compute device's
  /// control block. The call is blocking until the command execution is
  /// finished.
  ///
  /// @param ptr Memory address of the fill destination.
  /// @param value Value to be set.
  /// @param num Number of uint32_t element to be set to the value.
  virtual hsa_status_t SubmitLinearFillCommand(void* ptr, uint32_t value,
                                               size_t num) = 0;

  /// @brief Enable profiling of the asynchronous copy command. The timestamp
  /// of each copy request will be stored in the completion signal structure.
  ///
  /// @param enable True to enable profiling. False to disable profiling.
  ///
  /// @return HSA_STATUS_SUCCESS if the request to enable/disable profiling is
  /// successful.
  virtual hsa_status_t EnableProfiling(bool enable) = 0;

  /// @brief Blit operations use SDMA.
  virtual bool isSDMA() const { return false; }

  /// @Brief Reports the approximate number of remaining bytes to copy or fill.  Any return of zero
  /// must be exact.
  virtual uint64_t PendingBytes() = 0;

  virtual void GangLeader(bool gang_leader) = 0;
  virtual bool GangLeader() const { return false; };

  /// @brief Whether this blit implements SubmitSwapCopyCommand below. Lets the
  /// copy router pick between a capable engine and a per-entry fallback without
  /// switching on the concrete type.
  ///
  /// @note Deliberately not named SwapSupported(): BlitSdmaBase already uses
  /// that name for "this SDMA engine has the swap packet", which is a different
  /// question - the SDMA engine reaches its swap packet through SubmitBodies,
  /// not through SubmitSwapCopyCommand.
  virtual bool SupportsSwapCopyCommand() const { return false; }

  /// @brief Submit a batch of swap copy commands, exchanging the contents of
  /// each address pair. After execution entry i's buffer A holds the original
  /// contents of its buffer B and vice versa. All entries share @p out_signal,
  /// which is released once every entry has completed.
  ///
  /// @note N-entry rather than single-pair because that is the shape the HSA
  /// batch-copy API and its HIP caller use; a swap batch must not have to be
  /// unrolled into N ops by the caller just to reach a blit.
  ///
  /// @param num_entries Number of address pairs, >= 1.
  /// @param addr_a_list Array of @p num_entries first-buffer addresses.
  /// @param addr_b_list Array of @p num_entries second-buffer addresses.
  /// @param size_list Array of @p num_entries swap sizes in bytes.
  /// @param dep_signals Arrays of dependent signals.
  /// @param out_signal Output signal.
  virtual hsa_status_t SubmitSwapCopyCommand(
      uint32_t num_entries, void* const* addr_a_list, void* const* addr_b_list,
      const size_t* size_list, std::vector<core::Signal*>& dep_signals,
      core::Signal& out_signal) {
    return HSA_STATUS_ERROR_INVALID_ARGUMENT;
  }

  /// @brief Single-pair form of the above.
  ///
  /// @note Deliberately non-virtual: the N-entry overload is the only
  /// overridable entry point, so an implementation cannot accidentally support
  /// one shape and silently reject the other.
  hsa_status_t SubmitSwapCopyCommand(void* addr_a, void* addr_b, size_t size,
                                     std::vector<core::Signal*>& dep_signals,
                                     core::Signal& out_signal) {
    void* const addr_a_list[1] = {addr_a};
    void* const addr_b_list[1] = {addr_b};
    const size_t size_list[1] = {size};
    return SubmitSwapCopyCommand(1, addr_a_list, addr_b_list, size_list,
                                 dep_signals, out_signal);
  }
};
}  // namespace core
}  // namespace rocr

#endif  // header guard
