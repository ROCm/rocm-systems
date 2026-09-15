////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef HSA_RUNTIME_CORE_INC_AMD_KFD_LIFECYCLE_H_
#define HSA_RUNTIME_CORE_INC_AMD_KFD_LIFECYCLE_H_

#include <functional>
#include <utility>

#include "inc/hsa.h"

namespace rocr {
namespace AMD {

/// @brief The thunk calls ShutDown() has to be able to make, as data.
///
/// @details Only the unwind is bound up front, because ShutDown() has to run it
/// with no help from whoever is unwinding - that is the whole point. The three
/// acquiring steps are passed at their call sites instead: each has its own
/// arguments and its own idea of which thunk statuses count as success, and
/// none of them is ever called from here.
struct KfdLifecycleOps {
  std::function<hsa_status_t()> disable_runtime;
  std::function<hsa_status_t()> release_snapshot;
  std::function<hsa_status_t()> close;

  /// @brief Reports the calling process id.
  ///
  /// @details Injected rather than called directly so a test can drive the
  /// fork path without forking, and so this header stays clear of the os
  /// layer. KfdDriver wires it to rocr::os::GetProcessId(), which is getpid()
  /// on Linux and _getpid() on Windows. An instance built without one records
  /// no owning process and never takes the fork path.
  std::function<int()> get_pid;
};

/// @brief Staged ownership of what a KFD driver has taken from the thunk.
///
/// @details A KFD driver acquires in three steps - open the device, take the
/// topology snapshot, enable the runtime - and initialization can fail between
/// any two of them. Those failures unwind out through AMD::Load() and
/// Runtime::Load() to Runtime::Acquire(), which only drops the runtime
/// reference count: Unload() never runs, so nothing gives back what the
/// earlier steps took and the thunk's session stays up for the life of the
/// process.
///
/// This class is the single owner of those references. It records which steps
/// actually succeeded and ShutDown() gives back exactly those, in the same
/// order a fully initialized runtime uses, whatever stage the failure happened
/// at. That is what makes ShutDown() correct on a half-built driver and
/// idempotent afterwards, so an unwind on the failure path and a later normal
/// shutdown cannot both release the same reference.
class KfdLifecycle {
 public:
  explicit KfdLifecycle(KfdLifecycleOps ops)
      : ops_(std::move(ops)), owner_pid_(ops_.get_pid ? ops_.get_pid() : 0) {}

  KfdLifecycle(const KfdLifecycle&) = delete;
  KfdLifecycle& operator=(const KfdLifecycle&) = delete;

  /// @brief Open the device, at most once.
  ///
  /// @param open The thunk call that opens it.
  ///
  /// @details A second open would take a second reference that nothing gives
  /// back, so once the open is owned this reports success without calling the
  /// thunk again.
  hsa_status_t Open(const std::function<hsa_status_t()>& open);

  /// @brief Enable the KFD runtime.
  ///
  /// @param enable The thunk call that enables it.
  ///
  /// @details Ownership is recorded even when the thunk reports the call
  /// unsupported and @p enable maps that to success, because a fully
  /// initialized runtime disables unconditionally on the way out and this has
  /// to match it.
  hsa_status_t EnableRuntime(const std::function<hsa_status_t()>& enable);

  /// @brief Take the topology snapshot reference, at most once.
  ///
  /// @param acquire The thunk call that takes the reference and fills in the
  /// caller's HsaSystemProperties.
  ///
  /// @details The reference is long-lived by design: it keeps the snapshot
  /// alive for as long as the runtime is up. Releasing and re-acquiring it
  /// would tear down the FMM apertures and then fail to re-acquire the VM,
  /// because the kernel-side VM binding persists. So there is exactly one per
  /// driver, and a repeat call reports success without taking a second.
  ///
  /// Two callers rely on that: Init() takes the reference before the runtime
  /// enable, whose DXG debug probe reads the device list the snapshot owns,
  /// and BuildTopology() then reads the same list through the short-circuited
  /// second call. A caller whose output argument the short-circuit skips has
  /// to answer it from what the first acquire reported.
  hsa_status_t AcquireSnapshot(const std::function<hsa_status_t()>& acquire);

  /// @brief Give back the open reference, and nothing else.
  ///
  /// @details The inverse of Open(), not a teardown: core::Driver documents
  /// Close() as closing a connection to an open driver, and a caller asking
  /// for that must not also lose the runtime enable and the topology snapshot.
  /// ShutDown() is the method that gives back everything.
  ///
  /// Idempotent, and safe to call on a driver that never opened: owning no
  /// open reference is success, because there is nothing to give back. Like
  /// every stage of ShutDown(), the ownership is dropped before the call
  /// rather than after, so a failed close is not retried into a double close.
  ///
  /// In a forked child the inherited open reference is the parent's, so this
  /// drops the claim without closing. See InheritedAcrossFork().
  ///
  /// @return What the thunk's close reported, or HSA_STATUS_SUCCESS if this
  /// owned no open reference.
  hsa_status_t Close();

  /// @brief Give back every reference this driver owns, and only those.
  ///
  /// @details Runs every owned step even if an earlier one fails - stopping at
  /// the first error would strand the references the remaining steps give
  /// back, which is the leak this class exists to prevent. Each step's
  /// ownership is dropped as it runs, so a failed release is not retried into
  /// a double release later. The last of those steps is Close().
  ///
  /// In a forked child every inherited claim is the parent's, so this gives
  /// back nothing at all and reports success. See InheritedAcrossFork().
  ///
  /// @return The first error any step reported, or HSA_STATUS_SUCCESS. Owning
  /// nothing is success: there is nothing to fail.
  hsa_status_t ShutDown();

 private:
  /// @brief Whether this instance's ownership was inherited across a fork.
  ///
  /// @details The flags below are plain bools, so fork() copies them into a
  /// child that holds none of the references they describe. The thunk zeroes
  /// its own counters and reallocates *dxg_runtime from hsaKmtOpenKFD()'s
  /// is_forked_child() path, and nothing under core/ installs a
  /// pthread_atfork handler, so a child's flags describe the parent's
  /// session. Giving those back would release references the parent still
  /// holds, and on WSL would tear the thunk down under another live consumer.
  ///
  /// Recording the owning pid mirrors what the thunk does with parent_pid,
  /// and for the same reason it avoids atfork: a handler cannot be
  /// uninstalled, and this class can have more than one instance.
  bool InheritedAcrossFork() const;

  KfdLifecycleOps ops_;

  /// @brief The process that took whatever the flags below claim.
  int owner_pid_ = 0;

  bool owns_open_ = false;
  bool owns_runtime_enable_ = false;
  bool owns_snapshot_ = false;
};

}  // namespace AMD
}  // namespace rocr

#endif  // header guard
