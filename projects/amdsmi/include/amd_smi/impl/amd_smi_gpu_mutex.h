// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_GPU_MUTEX_H_
#define AMD_SMI_INCLUDE_AMD_SMI_GPU_MUTEX_H_

// NOTE: This header requires internal rocm_smi headers which are not installed.
// It is intended for GPU code paths only (amd_smi.cc, amd_smi_utils.cc).
// For non-GPU (NIC/CPU) code paths - TBD if needed.
#include "rocm_smi/rocm_smi.h"
#include "rocm_smi/rocm_smi_main.h"
#include "rocm_smi/rocm_smi_utils.h"

// GPU mutex: consults RocmSMI::init_options() to determine whether to use a
// blocking or non-blocking lock. RSMI_INIT_FLAG_RESRV_TEST1 switches to
// non-blocking mode so that tests can intentionally trigger AMDSMI_STATUS_BUSY.
#define SMIGPUDEVICE_MUTEX(MUTEX)                                                    \
  amd::smi::pthread_wrap _pw(*(MUTEX));                                              \
  amd::smi::RocmSMI& _smigpu = amd::smi::RocmSMI::getInstance();                     \
  bool _gpu_blocking =                                                               \
      !(_smigpu.init_options() & static_cast<uint64_t>(RSMI_INIT_FLAG_RESRV_TEST1)); \
  amd::smi::ScopedPthread _lock(_pw, _gpu_blocking);                                 \
  if (_lock.mutex_not_acquired()) {                                                  \
    return AMDSMI_STATUS_BUSY;                                                       \
  }

#endif  // AMD_SMI_INCLUDE_AMD_SMI_GPU_MUTEX_H_
