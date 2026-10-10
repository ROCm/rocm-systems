////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2024, Advanced Micro Devices, Inc. All rights reserved.
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

/**
 * @file hsa_ven_amd_launch_descriptor.h
 * @brief HSA AMD Vendor-Specific Launch Descriptor APIs
 *
 * This header defines AMD-specific launch descriptor types and APIs for
 * advanced kernel dispatch scheduling control. Launch descriptors provide
 * fine-grained control over:
 * - Compute Unit (CU) enable masks
 * - Wave dispatch limits and granularity
 * - Threadgroup chunk sizes
 * - L2 cache data prefetch regions
 * - Dispatch priority and power management hints
 */

#ifndef HSA_VEN_AMD_LAUNCH_DESCRIPTOR_H
#define HSA_VEN_AMD_LAUNCH_DESCRIPTOR_H

#include "hsa.h"
#include "amd_launch_descriptor.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @defgroup launch-descriptor Launch Descriptor
 *  @{
 */

/**
 * @brief Launch descriptor field identifiers for use with
 * hsa_ven_amd_launch_descriptor_set().
 */
typedef enum {
  /**
   * Version field. Specifies the descriptor format version.
   * Type: uint8_t
   * Valid values: AMD_LAUNCH_DESCRIPTOR_VERSION_*
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_VERSION = 0,

  /**
   * Priority field. Dispatch priority hint.
   * Type: uint8_t
   * Default: 0 (normal priority)
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_PRIORITY = 1,

  /**
   * Power management hint field.
   * Type: uint8_t
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_PM_HINT = 2,

  /**
   * CU start index. First CU index within each Shader Engine.
   * Type: uint32_t (4 bits)
   * Range: 0-15
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_CU_START = 3,

  /**
   * CU count. Number of contiguous CUs to enable per SE.
   * Type: uint32_t (4 bits)
   * Range: 0-15
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_CU_COUNT = 4,

  /**
   * Shader Engine enable mask.
   * Type: uint32_t (2 bits)
   * Range: 0-3
   * Bit 0: SE0 enable
   * Bit 1: SE1 enable
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_SE_EN = 5,

  /**
   * Dispatch granularity limiter. Maximum dispatch units resident.
   * Type: uint16_t
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_DISPATCH_GRANULARITY_LIMITER = 6,

  /**
   * Threadgroup chunk size. Number of workgroups per XCC.
   * Type: uint16_t
   */
  HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_TG_CHUNK_SIZE = 7
} hsa_ven_amd_launch_descriptor_field_t;

/**
 * @brief Create a launch descriptor.
 *
 * Allocates and initializes a new launch descriptor. The descriptor is
 * zero-initialized. The application must call
 * hsa_ven_amd_launch_descriptor_destroy() to free the descriptor when done.
 *
 * @param[out] launch_descriptor Pointer to receive the descriptor handle.
 *
 * @retval HSA_STATUS_SUCCESS The descriptor was created successfully.
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime is not initialized.
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT launch_descriptor is NULL.
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES Failed to allocate memory.
 */
hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_create(
    amd_launch_descriptor_t* launch_descriptor);

/**
 * @brief Destroy a launch descriptor.
 *
 * Frees the memory associated with a launch descriptor. The handle becomes
 * invalid after this call. The application must not use the descriptor in
 * any pending or future kernel dispatches after calling this function.
 *
 * @param[in] launch_descriptor The descriptor handle to destroy.
 *
 * @retval HSA_STATUS_SUCCESS The descriptor was destroyed successfully.
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime is not initialized.
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid descriptor handle.
 */
hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_destroy(
    amd_launch_descriptor_t launch_descriptor);

/**
 * @brief Set a field in a launch descriptor.
 *
 * Configures a single field of the launch descriptor. Use the
 * HSA_VEN_AMD_LAUNCH_DESCRIPTOR_FIELD_* enum values to specify which field
 * to set. The value is automatically cast to the appropriate field type.
 *
 * @param[in] launch_descriptor The descriptor handle.
 * @param[in] field Field identifier (hsa_ven_amd_launch_descriptor_field_t).
 * @param[in] value Value to set. Will be cast to the field's type.
 *
 * @retval HSA_STATUS_SUCCESS The field was set successfully.
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime is not initialized.
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid descriptor handle, unknown
 *         field identifier, or value out of range for the specified field.
 */
hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_set(
    amd_launch_descriptor_t launch_descriptor,
    uint32_t field,
    uint64_t value);

/**
 * @brief Configure an L2 cache prefetch region.
 *
 * Specifies a memory region to prefetch into L2 cache before kernel execution.
 * The hardware may prefetch the specified region to improve memory access
 * performance. Pass NULL as the prefetch parameter to disable the region.
 *
 * @param[in] launch_descriptor The descriptor handle.
 * @param[in] index Prefetch region index (0 or 1).
 * @param[in] prefetch Pointer to prefetch configuration, or NULL to disable.
 *
 * @retval HSA_STATUS_SUCCESS The prefetch region was configured successfully.
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime is not initialized.
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT Invalid descriptor handle, index
 *         out of range, or invalid prefetch configuration.
 */
hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_set_prefetch(
    amd_launch_descriptor_t launch_descriptor,
    uint32_t index,
    const amd_data_prefetch_t* prefetch);

/** @} */

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HSA_VEN_AMD_LAUNCH_DESCRIPTOR_H
