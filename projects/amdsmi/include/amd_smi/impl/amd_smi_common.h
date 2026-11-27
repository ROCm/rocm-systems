// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_
#define AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_

#include <map>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi.h"

#ifdef ENABLE_ESMI_LIB
extern "C" {
#include <e_smi/e_smi.h>

#include <cstdint>
}
#endif

// Included bare: the header opens its own extern "C", and wrapping it again
// would extend C linkage over the C++-only declarations at its tail.
#include "amd_smi/impl/nic/amdsmi_unified/interface/smi_nic_interface.h"

namespace amd::smi {

const std::map<unsigned, amdsmi_vram_type_t> vram_type_map = {
    {0, AMDSMI_VRAM_TYPE_UNKNOWN}, {1, AMDSMI_VRAM_TYPE_GDDR1},  {2, AMDSMI_VRAM_TYPE_DDR2},
    {3, AMDSMI_VRAM_TYPE_GDDR3},   {4, AMDSMI_VRAM_TYPE_GDDR4},  {5, AMDSMI_VRAM_TYPE_GDDR5},
    {6, AMDSMI_VRAM_TYPE_HBM},     {7, AMDSMI_VRAM_TYPE_DDR3},   {8, AMDSMI_VRAM_TYPE_DDR4},
    {9, AMDSMI_VRAM_TYPE_GDDR6},   {10, AMDSMI_VRAM_TYPE_DDR5},  {11, AMDSMI_VRAM_TYPE_LPDDR4},
    {12, AMDSMI_VRAM_TYPE_LPDDR5}, {13, AMDSMI_VRAM_TYPE_HBM3E},
};

amdsmi_status_t rsmi_to_amdsmi_status(rsmi_status_t status);

amdsmi_vram_type_t vram_type_value(unsigned type);

#ifdef ENABLE_ESMI_LIB
// Define a map of esmi status codes to amdsmi status codes
const std::map<esmi_status_t, amdsmi_status_t> esmi_status_map = {
    {ESMI_SUCCESS, AMDSMI_STATUS_SUCCESS},
    {ESMI_INITIALIZED, AMDSMI_STATUS_SUCCESS},
    {ESMI_INVALID_INPUT, AMDSMI_STATUS_INVAL},
    {ESMI_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED},
    {ESMI_PERMISSION, AMDSMI_STATUS_NO_PERM},
    {ESMI_INTERRUPTED, AMDSMI_STATUS_INTERRUPT},
    {ESMI_IO_ERROR, AMDSMI_STATUS_IO},
    {ESMI_FILE_ERROR, AMDSMI_STATUS_FILE_ERROR},
    {ESMI_NO_MEMORY, AMDSMI_STATUS_OUT_OF_RESOURCES},
    {ESMI_DEV_BUSY, AMDSMI_STATUS_BUSY},
    {ESMI_NOT_INITIALIZED, AMDSMI_STATUS_NOT_INIT},
    {ESMI_UNEXPECTED_SIZE, AMDSMI_STATUS_UNEXPECTED_SIZE},
    {ESMI_UNKNOWN_ERROR, AMDSMI_STATUS_UNKNOWN_ERROR},
    {ESMI_NO_ENERGY_DRV, AMDSMI_STATUS_NO_ENERGY_DRV},
    {ESMI_NO_MSR_DRV, AMDSMI_STATUS_NO_MSR_DRV},
    {ESMI_NO_HSMP_DRV, AMDSMI_STATUS_NO_HSMP_DRV},
    {ESMI_NO_HSMP_SUP, AMDSMI_STATUS_NO_HSMP_SUP},
    {ESMI_NO_DRV, AMDSMI_STATUS_NO_DRV},
    {ESMI_FILE_NOT_FOUND, AMDSMI_STATUS_FILE_NOT_FOUND},
    {ESMI_ARG_PTR_NULL, AMDSMI_STATUS_ARG_PTR_NULL},
    {ESMI_HSMP_TIMEOUT, AMDSMI_STATUS_HSMP_TIMEOUT},
    {ESMI_NO_HSMP_MSG_SUP, AMDSMI_STATUS_NO_HSMP_MSG_SUP},
};

amdsmi_status_t esmi_to_amdsmi_status(esmi_status_t status);
#endif

// Define a map of smi nic status codes to amdsmi status codes
const std::map<smi_nic_status_t, amdsmi_status_t> ainic_status_map = {
    {SMI_NIC_STATUS_SUCCESS, AMDSMI_STATUS_SUCCESS},
    {SMI_NIC_STATUS_ERROR, AMDSMI_STATUS_API_FAILED},
    {SMI_NIC_STATUS_WRONG_PARAM, AMDSMI_STATUS_INVAL},
    {SMI_NIC_STATUS_NOT_FOUND, AMDSMI_STATUS_NOT_FOUND},
    {SMI_NIC_STATUS_NO_RESOURCE, AMDSMI_STATUS_OUT_OF_RESOURCES},
    {SMI_NIC_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_SUPPORTED},
    {SMI_NIC_STATUS_NOT_INIT, AMDSMI_STATUS_NOT_INIT},
    {SMI_NIC_STATUS_NO_DATA, AMDSMI_STATUS_NO_DATA},
    {SMI_NIC_STATUS_DRIVER_NOT_LOADED, AMDSMI_STATUS_DRIVER_NOT_LOADED}};
amdsmi_status_t ainic_to_amdsmi_status(smi_nic_status_t status);

/**
 *  AMDSMI Library init reference count (amdsmi_init / amdsmi_shut_down)
 *      - Lives in amd_smi_common.cc
 */
bool amdsmi_library_initialized();
void amdsmi_library_init_ref_acquire();

/**
 *  AMDSMI Decrements init ref; should run (count reached zero).
 *      - Returns true if AMDSmiSystem::cleanup()
 *
 */

bool amdsmi_library_init_ref_release();

}  // namespace amd::smi

// Verifies AMD SMI is initialized; returns AMDSMI_STATUS_NOT_INIT from the enclosing function.
#ifndef AMDSMI_CHECK_INIT
#define AMDSMI_CHECK_INIT()                        \
  do {                                             \
    if (!amd::smi::amdsmi_library_initialized()) { \
      return AMDSMI_STATUS_NOT_INIT;               \
    }                                              \
  } while (0)
#endif

#endif  // AMD_SMI_INCLUDE_AMD_SMI_COMMON_H_
