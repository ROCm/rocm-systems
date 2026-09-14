// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"

/**
 *  @brief Populate physical_acc_id for a single GPU via UALoE. Implemented in
 *  amd_smi_ualoe.cc; keeps ualoe_lib types out of amd_smi.cc.
 *
 *  @note Unlike sibling UALoE functions, this does not acquire the device's
 *  SMIGPUDEVICE_MUTEX internally; the caller must already hold it.
 *
 *  @param[in]  device GPU device to query.
 *  @param[out] phys_id Physical accelerator ID; sentinel-filled on any
 *  NOT_SUPPORTED/error path.
 *
 *  @retval ::AMDSMI_STATUS_SUCCESS on success, ::AMDSMI_STATUS_NOT_SUPPORTED if the
 *  device has no active UALoE session.
 */
amdsmi_status_t get_physical_acc_id_from_ualoe(amd::smi::AMDSmiGPUDevice* device,
                                               uint32_t* phys_id);
