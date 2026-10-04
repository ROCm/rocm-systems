// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <cstdint>

#include "amd_smi/amdsmi.h"

// Library-local test seam for the PLX reading behind amdsmi_get_temp_metric().
// Not amdsmi_-prefixed, so the linker version script keeps it out of
// libamd_smi.so; tests reach it through the static archive. Shared by the
// definition (src/amd_smi/amd_smi.cc) and the unit tests so the signature stays
// in sync.
//
// PLX temperature comes from gpu_metrics rather than hwmon. Returns
// AMDSMI_STATUS_NOT_SUPPORTED when the reading carries the not-applicable
// sentinel (e.g. an APU with no PLX/VRSOC sensor), otherwise writes
// *temperature and returns AMDSMI_STATUS_SUCCESS.
amdsmi_status_t smi_amdgpu_plx_temp_from_metrics(const amdsmi_gpu_metrics_t& metrics,
                                                 int64_t* temperature);
