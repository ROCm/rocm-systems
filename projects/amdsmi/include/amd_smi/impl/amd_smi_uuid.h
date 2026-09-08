// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef GPUVSMI_UUID_H_
#define GPUVSMI_UUID_H_

#include <cstdint>

#include "amd_smi/amdsmi.h"

/**
 *  \brief  Generates uuid for device with specified parameters
 *
 *  \param [out] str      String buffer where to output generated uuid
 *
 *  \param [in]  serial   Asic serial
 *
 *  \param [in]  did      Device ID
 *
 *  \param [in]  idx      PF/VF/Partition index
 *
 *  \return SMI_RET_CODE indicating result.
 */
amdsmi_status_t amdsmi_uuid_gen(char* str, uint64_t serial, uint16_t did, uint8_t idx);

#endif
