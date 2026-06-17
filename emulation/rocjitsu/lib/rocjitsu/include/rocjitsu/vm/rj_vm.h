// Copyright (c) 2025-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_vm.h
/// @brief Public C API for creating and running a rocjitsu virtual machine.

#ifndef ROCJITSU_VM_RJ_VM_H_
#define ROCJITSU_VM_RJ_VM_H_

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#include <stddef.h>
#include <stdint.h>
#include <sys/types.h>

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

/// @brief Platform-specific handle type (fd on Linux, HANDLE on Windows).
typedef int rj_handle_t;

/// @brief VM creation mode.
typedef enum rj_vm_mode_t {
  /// @brief Standalone simulation. Engine runs with configured tick limit.
  /// Topology and driver are not initialized — the caller drives the
  /// simulation via rj_vm_step or rj_vm_run.
  RJ_VM_MODE_DEFAULT = 0,
  /// @brief Single-process serving (LD_PRELOAD interposer). Engine runs
  /// indefinitely, topology is generated, and the driver is opened so that
  /// a host runtime (e.g., ROCR) can issue ioctls through the interposer.
  RJ_VM_MODE_LOCAL = 1,
  /// @brief Multi-process serving (daemon). Same as LOCAL, plus all GPU
  /// allocations are backed by memfds for cross-process sharing via
  /// SCM_RIGHTS. Used by the rocjitsu CLI in daemon mode to host the
  /// simulator on behalf of remote client processes.
  RJ_VM_MODE_DAEMON = 2,
} rj_vm_mode_t;

/// @brief Device command descriptor for rj_vm_execute.
typedef struct rj_vm_cmd_t {
  uint32_t cmd;              ///< Platform-specific command number.
  void *buf;                 ///< Command arguments buffer (with inlined arrays).
  size_t buf_size;           ///< Total size of the arguments buffer.
  int32_t result;            ///< [out] Return code (0 on success, negative errno on failure).
  rj_handle_t shared_handle; ///< [out] Backing handle for shareable allocations, or -1.
} rj_vm_cmd_t;

/// @brief Device memory mapping descriptor.
typedef struct rj_vm_map_t {
  uint64_t addr;        ///< Requested mapping address.
  uint64_t length;      ///< Length in bytes to map.
  int64_t offset;       ///< Platform-specific offset encoding.
  uint32_t prot;        ///< Memory protection flags.
  uint32_t flags;       ///< Mapping flags.
  uint64_t mapped_addr; ///< [out] Address the mapping was placed at.
} rj_vm_map_t;

/// @brief Device memory unmapping descriptor.
typedef struct rj_vm_unmap_t {
  uint64_t addr;   ///< Address of the mapping to unmap.
  uint64_t length; ///< Length in bytes to unmap.
} rj_vm_unmap_t;

/// @brief Simulated GPU metadata needed by daemon clients.
typedef struct rj_vm_gpu_info_t {
  uint32_t present; ///< Non-zero when this payload was populated.

  uint32_t gpu_id;
  uint32_t gfx_target_version;
  uint32_t vendor_id;
  uint32_t device_id;
  uint32_t family_id;
  uint64_t unique_id;
  uint32_t location_id;
  uint32_t domain;
  uint64_t hive_id;
  uint32_t drm_render_minor;
  uint32_t revision_id;
  uint32_t pci_revision_id;

  uint32_t simd_count;
  uint32_t max_waves_per_simd;
  uint32_t num_shader_engines;
  uint32_t num_shader_arrays_per_engine;
  uint32_t num_cu_per_sh;
  uint32_t simd_per_cu;
  uint32_t wave_front_size;
  uint32_t num_xcc;
  uint32_t max_slots_scratch_cu;

  uint64_t local_mem_size;
  uint32_t vram_type;
  uint32_t lds_size_kb;
  uint32_t mem_width;
  uint32_t mem_clk_max;

  uint32_t l1_size_kb;
  uint32_t l1_line_size;
  uint32_t l1_assoc;
  uint32_t l2_size_kb;
  uint32_t l2_line_size;
  uint32_t l2_assoc;

  uint32_t num_sdma_engines;
  uint32_t num_sdma_xgmi_engines;
  uint32_t num_cp_queues;
  uint32_t max_engine_clk_fcompute;

  uint32_t capability;
  uint32_t capability2;
  uint64_t debug_prop;

  uint32_t fw_version;
  uint32_t sdma_fw_version;

  char marketing_name[128];
} rj_vm_gpu_info_t;

/// @brief Create a VM from a JSON configuration file.
///
/// @details Parses the JSON against the embedded FlatBuffers schema, constructs
/// the full component hierarchy, and loads any program binaries. The mode
/// parameter controls initialization depth — see rj_vm_mode_t.
/// @param[in] json_path Path to the JSON config file.
/// @param[in] mode VM creation mode (DEFAULT, LOCAL, or DAEMON).
/// @param[out] vm The newly created VM handle.
/// @retval ROCJITSU_STATUS_SUCCESS VM was created successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL.
/// @retval ROCJITSU_STATUS_INVALID_FILE The JSON file could not be opened.
/// @retval ROCJITSU_STATUS_ERROR Parsing or construction failed.
RJ_API_EXPORT rj_status_t rj_vm_create(const char *json_path, rj_vm_mode_t mode, rj_vm_t **vm);

/// @brief Create a VM from a JSON configuration string.
///
/// @param[in] json JSON configuration string.
/// @param[in] mode VM creation mode (DEFAULT, LOCAL, or DAEMON).
/// @param[out] vm The newly created VM handle.
/// @retval ROCJITSU_STATUS_SUCCESS VM was created successfully.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL.
/// @retval ROCJITSU_STATUS_ERROR Parsing or construction failed.
RJ_API_EXPORT rj_status_t rj_vm_create_from_string(const char *json, rj_vm_mode_t mode,
                                                   rj_vm_t **vm);

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

/// @brief Execute a device command (local mode — uses the local process).
/// @param[in] vm VM handle.
/// @param[in,out] cmd Command descriptor.
RJ_API_EXPORT rj_status_t rj_vm_execute(rj_vm_t *vm, rj_vm_cmd_t *cmd);

/// @brief Execute a device command for a specific process (daemon mode).
/// @param[in] vm VM handle.
/// @param[in] process_id The target process ID.
/// @param[in,out] cmd Command descriptor.
RJ_API_EXPORT rj_status_t rj_vm_execute_as(rj_vm_t *vm, uint32_t process_id, rj_vm_cmd_t *cmd);

/// @brief Open the VM's simulated device and create (or reuse) a KFD process.
/// @param[in] vm VM handle.
/// @param[in] client_pid Connecting client's OS PID for daemon mode (enables
///   cross-process memory access and process reuse); pass 0 in local mode where
///   there is no separate client process. If a process already exists for a
///   nonzero client_pid, it is reused.
/// @param[out] process_id The simulator's process handle (may be NULL).
RJ_API_EXPORT rj_status_t rj_vm_device_open(rj_vm_t *vm, pid_t client_pid, uint32_t *process_id);

/// @brief Close a specific KFD process by ID.
/// @param[in] vm VM handle.
/// @param[in] process_id The process ID to close (0 closes the local process).
RJ_API_EXPORT rj_status_t rj_vm_device_close(rj_vm_t *vm, uint32_t process_id);

/// @brief Map device memory (local mode).
/// @param[in] vm VM handle.
/// @param[in,out] map Mapping descriptor; mapped_addr is set on success.
RJ_API_EXPORT rj_status_t rj_vm_device_map(rj_vm_t *vm, rj_vm_map_t *map);

/// @brief Map device memory for a specific process (daemon mode).
RJ_API_EXPORT rj_status_t rj_vm_device_map_as(rj_vm_t *vm, uint32_t process_id, rj_vm_map_t *map);

/// @brief Unmap device memory (local mode).
/// @param[in] vm VM handle.
/// @param[in] unmap Unmapping descriptor.
RJ_API_EXPORT rj_status_t rj_vm_device_unmap(rj_vm_t *vm, rj_vm_unmap_t *unmap);

/// @brief Unmap device memory for a specific process (daemon mode).
RJ_API_EXPORT rj_status_t rj_vm_device_unmap_as(rj_vm_t *vm, uint32_t process_id,
                                                rj_vm_unmap_t *unmap);

/// @brief Get the KFD gpu_id for the simulated device.
/// @param[in] vm VM handle.
/// @param[out] gpu_id The gpu_id value.
RJ_API_EXPORT rj_status_t rj_vm_gpu_id(rj_vm_t *vm, uint32_t *gpu_id);

/// @brief Get the sysfs topology directory path.
/// @param[in] vm VM handle.
/// @param[out] path Pointer to the topology path string (owned by the VM).
RJ_API_EXPORT rj_status_t rj_vm_topology_path(rj_vm_t *vm, const char **path);

/// @brief Get the DRM sysfs directory path.
/// @param[in] vm VM handle.
/// @param[out] path Pointer to the DRM path string (owned by the VM).
RJ_API_EXPORT rj_status_t rj_vm_drm_path(rj_vm_t *vm, const char **path);

/// @brief Get simulated GPU metadata.
/// @param[in] vm VM handle.
/// @param[out] info Simulated GPU metadata.
RJ_API_EXPORT rj_status_t rj_vm_gpu_info(rj_vm_t *vm, rj_vm_gpu_info_t *info);

/// @brief Get the backing memory handle (local mode).
RJ_API_EXPORT rj_status_t rj_vm_get_shared_mem(rj_vm_t *vm, int64_t offset, rj_handle_t *handle);

/// @brief Get the backing memory handle for a specific process (daemon mode).
RJ_API_EXPORT rj_status_t rj_vm_get_shared_mem_as(rj_vm_t *vm, uint32_t process_id, int64_t offset,
                                                  rj_handle_t *handle);

/// @brief Request the simulation engine to stop.
///
/// @details Thread-safe. Signals the engine's run() loop to exit at the next
/// opportunity. Use this before joining an engine thread to ensure clean
/// shutdown without use-after-free.
/// @param[in] vm VM handle.
/// @param[in] reason Human-readable reason for stopping (may be NULL).
RJ_API_EXPORT void rj_vm_request_exit(rj_vm_t *vm, const char *reason);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_VM_RJ_VM_H_
