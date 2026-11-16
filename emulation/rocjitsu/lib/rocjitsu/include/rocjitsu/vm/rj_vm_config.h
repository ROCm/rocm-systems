// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm_config.h
/// @brief Passive GPU discovery from an owned VM configuration snapshot.

#ifndef ROCJITSU_VM_RJ_VM_CONFIG_H_
#define ROCJITSU_VM_RJ_VM_CONFIG_H_

#include "rocjitsu/vm/rj_vm.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @brief Immutable JSON, node selection, and GPU metadata without a simulated machine.
/// @details The handle owns its input snapshot. Queries and VM creation may run
/// concurrently; destruction requires exclusive access. The library must remain
/// loaded until the handle and every VM created from it have been destroyed.
typedef struct rj_vm_config_t rj_vm_config_t;

/// @brief Copy JSON and discover its selected node's GPUs without activating them.
/// @details Uses JSON local_node, or the sole node when no selector is configured.
/// Parses metadata without constructing simulator components, creating workers,
/// opening a simulated process, or generating sysfs files. Full VM construction
/// may still reject unsupported execution configuration. Failure leaves @p config
/// unchanged and creates no cleanup obligation.
/// @param[in] json Null-terminated JSON, borrowed only during this call.
/// @param[out] config Newly owned passive configuration handle.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL, or parsing
/// or node selection failed.
/// @retval ROCJITSU_STATUS_UNSUPPORTED The node has no discoverable KFD GPU identity
/// or XCD topology.
/// @retval ROCJITSU_STATUS_OUT_OF_RESOURCES Configuration allocation failed.
RJ_API_EXPORT rj_status_t rj_vm_config_create_from_string(const char *json,
                                                          rj_vm_config_t **config);

/// @brief Copy JSON and discover the GPUs in one named system node.
/// @details Node selection follows rj_vm_create_from_string_for_node. The node
/// must exist and agree with JSON local_node when present. NULL or empty uses
/// local_node; absent both selectors, the graph must contain exactly one node.
/// Both input strings are borrowed only during this call. Failure semantics
/// match rj_vm_config_create_from_string.
RJ_API_EXPORT rj_status_t rj_vm_config_create_from_string_for_node(const char *json,
                                                                   const char *node_id,
                                                                   rj_vm_config_t **config);

/// @brief Destroy a passive configuration; NULL is a no-op.
RJ_API_EXPORT void rj_vm_config_destroy(rj_vm_config_t *config);

/// @brief Return the selected node's GPU count without allocation or activation.
/// @details Invalid arguments return ROCJITSU_STATUS_INVALID_ARGUMENT and leave @p count unchanged.
RJ_API_EXPORT rj_status_t rj_vm_config_gpu_count(const rj_vm_config_t *config, uint32_t *count);

/// @brief Copy metadata for one zero-based GPU ordinal in the selected node.
/// @details Performs no allocation or activation. An invalid argument or index
/// leaves @p info unchanged and returns ROCJITSU_STATUS_INVALID_ARGUMENT.
RJ_API_EXPORT rj_status_t rj_vm_config_gpu_info_at(const rj_vm_config_t *config, uint32_t ordinal,
                                                   rj_vm_gpu_info_t *info);

/// @brief Create a VM from the retained JSON and node selection.
/// @details The VM owns its construction state independently of @p config;
/// the configuration may be destroyed after this call returns. Mode and VM
/// lifetime follow rj_vm_create_from_string_for_node. Failure leaves @p vm unchanged.
RJ_API_EXPORT rj_status_t rj_vm_create_from_config(const rj_vm_config_t *config, rj_vm_mode_t mode,
                                                   rj_vm_t **vm);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VM_RJ_VM_CONFIG_H_
