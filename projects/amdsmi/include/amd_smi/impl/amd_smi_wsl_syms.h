/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 */

// Internal header: dlsym-resolved function pointers for the WSL backend.
// No link-time dependency on hsakmt or rocdxg — all symbols are resolved at
// runtime by load_rocdxg() in amd_smi_system.cc.

#ifndef AMD_SMI_WSL_SYMS_H_
#define AMD_SMI_WSL_SYMS_H_

#ifdef ENABLE_WSL_BACKEND

#include <hsakmt/hsakmt.h>
#include <hsakmt/rocdxg_smi.h>

namespace amd::smi {

// Function pointer table populated by load_rocdxg().
// All pointers are null until dlopen succeeds.
struct WslSyms {
  // hsakmt
  HSAKMT_STATUS (*hsaKmtOpenKFD)() = nullptr;
  HSAKMT_STATUS (*hsaKmtCloseKFD)() = nullptr;
  HSAKMT_STATUS (*hsaKmtAcquireSystemProperties)(HsaSystemProperties*) = nullptr;
  HSAKMT_STATUS (*hsaKmtReleaseSystemProperties)() = nullptr;
  HSAKMT_STATUS (*hsaKmtGetNodeProperties)(HSAuint32, HsaNodeProperties*) = nullptr;

  // rocdxg_smi
  // Static device info (populated once per device via rocdxg_smi_get_device_info)
  HSAKMT_STATUS (*rocdxg_smi_get_device_info)(uint32_t, rocdxg_smi_device_info_t*) = nullptr;
  // Dynamic / per-call queries
  HSAKMT_STATUS (*rocdxg_smi_get_vram_usage)(uint32_t, rocdxg_smi_vram_usage_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_power_info)(uint32_t, rocdxg_smi_power_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_temperature)(uint32_t, uint32_t, uint32_t, int64_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_clock_info)(uint32_t, uint32_t,
                                             rocdxg_smi_clock_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_pcie_info)(uint32_t, rocdxg_smi_pcie_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_get_gpu_metrics_info)(uint32_t,
                                                   rocdxg_smi_gpu_metrics_info_t*) = nullptr;
  HSAKMT_STATUS (*rocdxg_smi_enum_processes)(uint32_t, uint32_t*,
                                             rocdxg_smi_process_info_t*) = nullptr;
};

// Defined in amd_smi_system.cc; valid after load_rocdxg() returns true.
extern WslSyms g_wsl_syms;

}  // namespace amd::smi

#endif  // ENABLE_WSL_BACKEND
#endif  // AMD_SMI_WSL_SYMS_H_
