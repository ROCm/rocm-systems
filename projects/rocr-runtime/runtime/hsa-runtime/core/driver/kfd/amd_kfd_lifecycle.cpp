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

#include "core/inc/amd_kfd_lifecycle.h"
#include "core/util/utils.h"

namespace rocr {
namespace AMD {

hsa_status_t KfdLifecycle::Open(const std::function<hsa_status_t()>& open) {
  if (owns_open_) return HSA_STATUS_SUCCESS;

  const hsa_status_t status = open();
  if (status == HSA_STATUS_SUCCESS) owns_open_ = true;

  return status;
}

hsa_status_t KfdLifecycle::EnableRuntime(const std::function<hsa_status_t()>& enable) {
  const hsa_status_t status = enable();
  if (status == HSA_STATUS_SUCCESS) owns_runtime_enable_ = true;

  return status;
}

hsa_status_t KfdLifecycle::AcquireSnapshot(const std::function<hsa_status_t()>& acquire) {
  if (owns_snapshot_) return HSA_STATUS_SUCCESS;

  const hsa_status_t status = acquire();
  if (status == HSA_STATUS_SUCCESS) owns_snapshot_ = true;

  return status;
}

bool KfdLifecycle::InheritedAcrossFork() const {
  return ops_.get_pid && ops_.get_pid() != owner_pid_;
}

hsa_status_t KfdLifecycle::Close() {
  // The open reference belongs to the process that took it, so a child drops
  // the inherited claim instead of closing the parent's reference.
  if (InheritedAcrossFork()) {
    owns_open_ = false;
    return HSA_STATUS_SUCCESS;
  }

  if (!owns_open_) return HSA_STATUS_SUCCESS;

  owns_open_ = false;
  return ops_.close();
}

hsa_status_t KfdLifecycle::ShutDown() {
  // Nothing a child inherited is the child's to give back. Dropping the
  // claims and calling no op leaves the parent's references with the parent.
  if (InheritedAcrossFork()) {
    owns_runtime_enable_ = false;
    owns_snapshot_ = false;
    owns_open_ = false;
    return HSA_STATUS_SUCCESS;
  }

  hsa_status_t status = HSA_STATUS_SUCCESS;
  // Say which stage failed. The caller often discards this status, and only
  // the first error survives below, so without a diagnostic a failed release
  // is indistinguishable from a clean shutdown. The names are lifecycle
  // stages rather than thunk entry points because this class deliberately
  // does not know which thunk call implements each one.
  auto record = [&status](const char* stage, hsa_status_t err) {
    if (err == HSA_STATUS_SUCCESS) return;

    debug_print("KfdLifecycle::ShutDown() failed to %s: 0x%x\n", stage, static_cast<unsigned>(err));
    if (status == HSA_STATUS_SUCCESS) status = err;
  };

  // Ownership is dropped before the call, not after: a step that fails has
  // still had its one chance, and retrying it from a later ShutDown() would
  // release a reference twice.
  if (owns_runtime_enable_) {
    owns_runtime_enable_ = false;
    record("disable runtime", ops_.disable_runtime());
  }

  if (owns_snapshot_) {
    owns_snapshot_ = false;
    record("release topology snapshot", ops_.release_snapshot());
  }

  // The close stage is Close(), which carries the same ownership discipline
  // and reports success when there is no open reference to give back.
  record("close KFD", Close());

  return status;
}

}  // namespace AMD
}  // namespace rocr
