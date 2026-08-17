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
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#pragma once

#include "amd_smi/amdsmi.h"
#include "amd_smi/impl/amd_smi_gpu_device.h"

/**
 *  @brief Populate physical_acc_id for a single GPU via UALoE. Implemented in
 *  amd_smi_ualoe.cc; keeps ualoe_lib types out of amd_smi.cc.
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
