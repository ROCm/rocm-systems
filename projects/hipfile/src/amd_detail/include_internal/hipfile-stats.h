/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include "hipfile.h"

#ifdef __cplusplus
extern "C" {
#endif

/** @brief Opaque pointer to a stats collection context */
typedef struct hipFileStatsContext hipFileStatsContext_t;

/**
 * @enum hipFileStatsError
 * @brief Error codes returned by stats collection API functions
 */
typedef enum hipFileStatsError {
    hipFileStatsSuccess,                    /**< Operation completed successfully */
    hipFileStatsInvalidArgument,            /**< Invalid argument passed to function */
    hipFileStatsTargetProcessNotFound,      /**< Target process with given PID not found */
    hipFileStatsTargetProcessNotAccessible, /**< Cannot access target process */
    hipFileStatsReportGenerationFailed,     /**< Failed to generate or write report */
} hipFileStatsError_t;

/**
 * @brief Create a new stats collection context for a target process
 * @param[out] context Pointer to store the created context handle
 * @param[in] targetPid Process ID of the target process to monitor
 * @return #hipFileStatsSuccess on success, error code otherwise
 *
 * Creates a new statistics collection context for the specified target process.
 * The returned context must be freed with hipFileStatsCloseContext().
 */
HIPFILE_API hipFileStatsError_t hipFileStatsCreateContext(hipFileStatsContext_t **context, int targetPid);

/**
 * @brief Close and free a stats collection context
 * @param[in] context Stats context handle to close (may be NULL)
 *
 * Closes the specified stats context and releases all associated resources.
 * Safe to call with NULL pointer.
 */
HIPFILE_API void hipFileStatsCloseContext(hipFileStatsContext_t *context);

/**
 * @brief Connect the stats context to its target process
 * @param[in] context Stats context handle
 * @return #hipFileStatsSuccess on success, error code otherwise
 *
 * Establishes connection to the target process for statistics collection.
 * Must be called before hipFileStatsGenerateReport().
 */
HIPFILE_API hipFileStatsError_t hipFileStatsConnectToTargetProcess(hipFileStatsContext_t *context);

/**
 * @brief Poll the target process for updated statistics
 * @param[in] context Stats context handle
 * @param[in] block Whether to block until the target process completes
 * @return #hipFileStatsSuccess on success, error code otherwise
 *
 * Polls the target process for completion.
 * If block is true, waits indefinitely.
 */
HIPFILE_API hipFileStatsError_t hipFileStatsPollTargetProcess(const hipFileStatsContext_t *context,
                                                              bool                         block);

/**
 * @brief Generate a statistics report and write it to a file descriptor
 * @param[in] context Stats context handle
 * @param[in] fd File descriptor to write the report to
 * @return #hipFileStatsSuccess on success, error code otherwise
 *
 * Generates a formatted statistics report from collected data and writes it to
 * the specified file descriptor.
 */
HIPFILE_API hipFileStatsError_t hipFileStatsGenerateReport(const hipFileStatsContext_t *context, int fd);
#ifdef __cplusplus
}
#endif
