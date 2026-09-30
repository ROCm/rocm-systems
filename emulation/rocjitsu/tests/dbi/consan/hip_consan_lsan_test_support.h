// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#if !defined(__HIP_DEVICE_COMPILE__)
// ROCR keeps host Signal objects alive through driver-managed ABI storage and
// retains queue bookkeeping after the simulated dispatch queues are destroyed.
// Match those owning allocation sites, not the runtime DSO, so leaks from
// application callbacks invoked by ROCR remain visible.
extern "C" __attribute__((visibility("default"))) const char *__lsan_default_suppressions() {
  return "leak:rocr::AMD::hsa_amd_signal_create\n"
         "leak:rocr::AMD::AqlQueue::AqlQueue\n";
}
#endif
