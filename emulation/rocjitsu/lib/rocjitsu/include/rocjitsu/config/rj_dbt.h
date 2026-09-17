// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file rj_dbt.h
/// @brief Public C API for publishing the DBT per-invocation runtime handoff.

#ifndef ROCJITSU_CONFIG_RJ_DBT_H_
#define ROCJITSU_CONFIG_RJ_DBT_H_

#include <stdint.h>

#include "rocjitsu/base/rj_compiler.h"
#include "rocjitsu/base/rj_status.h"

#ifdef __cplusplus
extern "C" {
#endif

/// @addtogroup config
/// @{

/// @brief Atomically publish the per-invocation runtime config handoff.
///
/// @details The KMD interposer and the HSA tools hook both discover their
/// configuration by reading a `config_path` file out of the runtime directory
/// named by `$ROCJITSU_RUNTIME_DIR` (or `$ROCJITSU_INVOCATION_DIR`), rather
/// than from a configuration environment variable. In DBT guest mode they must
/// additionally agree on which physical GPU the launcher selected, because
/// re-deriving it per process would let two runtime layers in the same process
/// resolve `host_gpu_id: 0` to different devices.
///
/// This entry point exists so a launcher written in another language does not
/// have to reimplement the file's format. It is the same writer the in-tree
/// launcher uses; the only difference is that the destination directory is
/// passed in rather than derived from the caller's PID.
///
/// The write is atomic: the contents land in a temporary file in the same
/// directory and are renamed into place, so a concurrent reader sees either the
/// previous handoff or the complete new one, never a partial line.
///
/// @param[in] runtime_dir Directory to publish into. Created if absent.
/// @param[in] config_path Configuration path to record, verbatim.
/// @param[in] host_gpu_id Resolved DBT host KFD `gpu_id`. Zero means the
/// invocation is not in DBT guest mode and no GPU line is written; a DBT guest
/// invocation must pass the nonzero GPU it resolved.
/// @retval ROCJITSU_STATUS_SUCCESS The handoff is published.
/// @retval ROCJITSU_STATUS_INVALID_ARGUMENT A required argument is NULL or
/// empty.
/// @retval ROCJITSU_STATUS_ERROR The directory could not be created or the
/// handoff could not be written or renamed into place.
RJ_API_EXPORT rj_status_t rj_dbt_write_handoff(const char *runtime_dir, const char *config_path,
                                               uint32_t host_gpu_id);

/// @}

#ifdef __cplusplus
} // extern "C"
#endif

#endif // ROCJITSU_CONFIG_RJ_DBT_H_
