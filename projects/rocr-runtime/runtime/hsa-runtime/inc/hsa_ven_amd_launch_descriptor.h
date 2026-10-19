// Copyright © Advanced Micro Devices, Inc., or its affiliates.
//
// SPDX-License-Identifier: MIT

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
   * Version field. Read-only; automatically set by
   * hsa_ven_amd_launch_descriptor_create() based on the target agent.
   * Attempts to set this field will return HSA_STATUS_ERROR_INVALID_ARGUMENT.
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
 * @brief Create a launch descriptor for a specific agent.
 *
 * Allocates and initializes a new launch descriptor whose version is
 * determined by the target agent's GPU architecture. The descriptor is
 * zero-initialized with the version field set automatically.
 * The application must call hsa_ven_amd_launch_descriptor_destroy() to
 * free the descriptor when done.
 *
 * @param[in] agent GPU agent that this descriptor targets. The runtime
 *            uses the agent's ISA to select the appropriate descriptor
 *            version. Must be a GPU agent.
 * @param[out] launch_descriptor Pointer to receive the descriptor handle.
 *
 * @retval HSA_STATUS_SUCCESS The descriptor was created successfully.
 * @retval HSA_STATUS_ERROR_NOT_INITIALIZED The HSA runtime is not initialized.
 * @retval HSA_STATUS_ERROR_INVALID_AGENT The agent is not a valid GPU agent
 *         or does not support launch descriptors.
 * @retval HSA_STATUS_ERROR_INVALID_ARGUMENT launch_descriptor is NULL.
 * @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES Failed to allocate memory.
 */
hsa_status_t HSA_API hsa_ven_amd_launch_descriptor_create(
    hsa_agent_t agent,
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

/**
 * @brief Version 1.00 function pointer table for the launch descriptor
 * vendor extension.
 *
 * Obtained via hsa_system_get_major_extension_table() with
 * HSA_EXTENSION_AMD_LAUNCH_DESCRIPTOR.
 */
#define hsa_ven_amd_launch_descriptor_1_00

typedef struct hsa_ven_amd_launch_descriptor_1_00_pfn_s {
  hsa_status_t (*hsa_ven_amd_launch_descriptor_create)(
      hsa_agent_t agent,
      amd_launch_descriptor_t* launch_descriptor);

  hsa_status_t (*hsa_ven_amd_launch_descriptor_destroy)(
      amd_launch_descriptor_t launch_descriptor);

  hsa_status_t (*hsa_ven_amd_launch_descriptor_set)(
      amd_launch_descriptor_t launch_descriptor,
      uint32_t field,
      uint64_t value);

  hsa_status_t (*hsa_ven_amd_launch_descriptor_set_prefetch)(
      amd_launch_descriptor_t launch_descriptor,
      uint32_t index,
      const amd_data_prefetch_t* prefetch);
} hsa_ven_amd_launch_descriptor_1_00_pfn_t;

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // HSA_VEN_AMD_LAUNCH_DESCRIPTOR_H
