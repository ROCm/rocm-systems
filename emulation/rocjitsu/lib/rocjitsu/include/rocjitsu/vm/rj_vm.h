// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm.h
/// @brief Public C API for creating and running a rocjitsu virtual machine.

#ifndef ROCJITSU_VM_RJ_VM_H_
#define ROCJITSU_VM_RJ_VM_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup vm
/// @{

/// @brief Opaque VM handle.
///
/// @details Represents a fully configured virtual machine including topology,
/// memory, loaded programs, and pending dispatches. All configuration is driven
/// by the JSON config passed to rj_vm_create or rj_vm_create_from_string.
typedef struct rj_vm_t rj_vm_t;

/// @brief Create a VM from a JSON configuration file.
///
/// @details Parses the JSON against the embedded FlatBuffers schema, constructs
/// the full component hierarchy, and loads any program binaries.
/// @param[in] json_path Path to the JSON config file.
/// @param[out] vm The newly created VM handle.
/// @retval ROCJITSU_STATUS_SUCCESS VM was created successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL.
/// @retval ROCJITSU_STATUS_INVALID_FILE The JSON file could not be opened.
/// @retval ROCJITSU_STATUS_ERROR Parsing or construction failed.
RJ_API_EXPORT rj_status_t rj_vm_create(const char *json_path, rj_vm_t **vm);

/// @brief Create a VM from a JSON configuration string.
///
/// @param[in] json JSON configuration string.
/// @param[out] vm The newly created VM handle.
/// @retval ROCJITSU_STATUS_SUCCESS VM was created successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL.
/// @retval ROCJITSU_STATUS_ERROR Parsing or construction failed.
RJ_API_EXPORT rj_status_t rj_vm_create_from_string(const char *json, rj_vm_t **vm);

/// @brief Increment the VM's reference count.
///
/// @details Use this to share a VM handle across multiple owners. Each call
/// must be balanced by a corresponding rj_vm_release.
/// @param[in] vm VM handle (may be NULL, in which case this is a no-op).
RJ_API_EXPORT void rj_vm_retain(rj_vm_t *vm);

/// @brief Decrement the VM's reference count.
///
/// @details If the VM has been destroyed (via rj_vm_destroy) and the reference
/// count reaches 0, the backing memory is freed.
/// @param[in] vm VM handle (may be NULL, in which case this is a no-op).
RJ_API_EXPORT void rj_vm_release(rj_vm_t *vm);

/// @brief Mark a VM for destruction.
///
/// @details If the reference count is already 0, frees immediately. Otherwise,
/// the VM is freed when the last rj_vm_release drops the reference count to 0.
/// @param[in] vm VM to destroy (may be NULL).
RJ_API_EXPORT void rj_vm_destroy(rj_vm_t *vm);

/// @brief Step the entire VM by one tick.
///
/// @details Processes all simulation events at the next timestamp, advancing the
/// simulation by one tick.
/// @param[in] vm VM handle.
/// @param[out] active Non-zero if any wavefront is still executing.
/// @retval ROCJITSU_STATUS_SUCCESS Step completed successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p vm is NULL.
RJ_API_EXPORT rj_status_t rj_vm_step(rj_vm_t *vm, int *active);

/// @brief Run the VM to completion or until max_ticks is reached.
///
/// @details Runs the simulation to completion. Terminates when all primary
/// components signal completion, the tick limit from the configuration is
/// reached, or quiescence is detected.
/// @param[in] vm VM handle.
/// @param[out] ticks_executed Number of ticks actually executed (may be NULL).
/// @retval ROCJITSU_STATUS_SUCCESS Simulation completed successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p vm is NULL.
RJ_API_EXPORT rj_status_t rj_vm_run(rj_vm_t *vm, uint64_t *ticks_executed);

/// @brief Save a VM checkpoint to disk.
/// @param[in] vm VM handle.
/// @param[in] path Output file path for the checkpoint.
/// @param[in] tick Simulation tick to record as the checkpoint timestamp.
/// @retval ROCJITSU_STATUS_SUCCESS Checkpoint saved successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT @p vm or @p path is NULL.
/// @retval ROCJITSU_STATUS_ERROR Serialization or I/O failed.
RJ_API_EXPORT rj_status_t rj_vm_save_checkpoint(const rj_vm_t *vm, const char *path, uint64_t tick);

/// @brief Restore a VM from a checkpoint file.
/// @param[in] path Path to the checkpoint file.
/// @param[out] vm The restored VM handle.
/// @retval ROCJITSU_STATUS_SUCCESS VM restored successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL.
/// @retval ROCJITSU_STATUS_INVALID_FILE The checkpoint file could not be opened.
/// @retval ROCJITSU_STATUS_ERROR Deserialization failed.
RJ_API_EXPORT rj_status_t rj_vm_restore_checkpoint(const char *path, rj_vm_t **vm);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VM_RJ_VM_H_
