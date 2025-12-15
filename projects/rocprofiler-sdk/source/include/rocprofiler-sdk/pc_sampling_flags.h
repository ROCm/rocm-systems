// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/defines.h>
#include <rocprofiler-sdk/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup PC_SAMPLING_SERVICE PC Sampling
 * @brief Enabling PC (Program Counter) Sampling for GPU Activity
 * @{
 */

/**
 * @brief Enumeration for version of the rocprofiler_pc_sampling_record_v*_t struct
 *
 * This enum allows users to specify which PC sampling record format they want to receive.
 * Different versions are optimized for different GPU architectures and sampling methods:
 * - VERSION_0: Invalid sample (error indicator, contains no meaningful data)
 * - VERSION_1: Basic host-trap sampling for GFX9 and Navi4x (96 bytes)
 * - VERSION_2: Stochastic sampling for MI300/MI350 with snapshot information (104 bytes)
 * - VERSION_3: Host-Trap sampling on FUTURE_GEN_1_2. (112 bytes)
 * - VERSION_4: Stochastic sampling on FUTURE_GEN_1_2 (136 bytes)
 * - VERSION_5: Host-Trap sampling on FUTURE_GEN_3/FUTURE_GEN_4 (112 bytes)
 * - VERSION_6: Stochastic sampling on FUTURE_GEN_3/FUTURE_GEN_4 (136 bytes)
 */
typedef enum rocprofiler_pc_sampling_record_version_t
{
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_NONE = 0,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0    = 1,  ///< invalid record
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1    = 2,  ///< host-trap for GFX9/Navi4x (96B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2    = 3,  ///< stochastic for MI300/MI350 (104B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_3    = 4,  ///< host-trap on FUTURE_GEN_1_2 (112B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_4    = 5,  ///< stochastic on FUTURE_GEN_1_2 (136B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_5    = 6,  ///< host-trap on FUTURE_GEN_3/FUTURE_GEN_4 (136B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_6    = 7,  ///< stochastic on FUTURE_GEN_3/FUTURE_GEN_4 (136B)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_LAST,
} rocprofiler_pc_sampling_record_version_t;

/**
 * @brief Configuration flags for PC sampling record delivery.
 *
 * These flags control which types of PC sampling records the user wants to receive
 * in their buffer. By default (flags=0), only valid samples are delivered. Users
 * can opt-in to receive invalid/error records and future partial records.
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_config_flags_t
{
    ROCPROFILER_PC_SAMPLING_CONFIG_NONE = 0,  ///< Default: deliver only valid samples

    /// @brief Include invalid/error records (v0) in addition to valid samples.
    ///
    /// When set, the buffer will receive rocprofiler_pc_sampling_record_v0_t records
    /// indicating that a sampling error occurred. These records contain minimal information
    /// (just the size field) and serve as error indicators. Without this flag, sampling
    /// errors are silently discarded.
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES = (1 << 0),

    /// @brief Include partially valid records (future functionality).
    ///
    /// When implemented, this flag will enable delivery of records where some fields
    /// contain valid data and others are invalid/unavailable. This is useful for
    /// debugging or understanding sampling limitations. Reserved for future use.
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES = (1 << 1),

    ROCPROFILER_PC_SAMPLING_CONFIG_LAST
} rocprofiler_pc_sampling_config_flags_t;

/**
 * @brief (experimental) Function used to configure the PC sampling service on the GPU agent with @p
 * agent_id.
 *
 * Prerequisites are the following:
 * - The client must create a context and supply its @p context_id. By using this context,
 *   the client can start/stop PC sampling on the agent. For more information,
 *   please @see rocprofiler_start_context/rocprofiler_stop_context.
 * - The user must create a buffer and supply its @p buffer_id. Rocprofiler-SDK uses the buffer
 *   to deliver the PC samples to the client. For more information about the data delivery,
 *   please @see rocprofiler_create_buffer and @see rocprofiler_buffer_tracing_cb_t.
 *
 * Before calling this function, we recommend querying PC sampling configurations
 * supported by the GPU agent via the @see rocprofiler_query_pc_sampling_agent_configurations.
 * The client chooses the @p method, @p unit, and @p interval to match one of the
 * available configurations. Note that the @p interval must belong to the range of values
 * [available_config.min_interval, available_config.max_interval],
 * where available_config is the instance of the @see rocprofiler_pc_sampling_configuration_s
 * supported/available at the moment.
 *
 * PC Sampling Record Versioning:
 * The client must specify which record format to receive by providing @p record_version and
 * @p record_size parameters. Different versions are optimized for different hardware architectures
 * and sampling methods:
 * - VERSION_0: Invalid record (error indicator, minimal data)
 * - VERSION_1: Basic host-trap sampling for GFX9 and Navi4x (96 bytes)
 * - VERSION_2: Stochastic sampling for MI300/MI350 with snapshot information (104 bytes)
 * - VERSION_3: Host-Trap sampling on FUTURE_GEN_1_2 (112 bytes)
 * - VERSION_4: Stochastic sampling on FUTURE_GEN_1_2 (136 bytes)
 * - VERSION_5: Host-Trap sampling on FUTURE_GEN_3/FUTURE_GEN_4 (112 bytes)
 * - VERSION_6: Stochastic sampling on FUTURE_GEN_3/FUTURE_GEN_4 (136 bytes)
 *
 * Record Filtering via Flags:
 * The @p flags parameter controls which record types are delivered to your buffer:
 * - flags=0 (ROCPROFILER_PC_SAMPLING_CONFIG_NONE): Receive only valid samples in the
 *   format specified by @p record_version. Invalid/error samples are discarded silently.
 * - flags=ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES: Receive both valid samples
 *   (in the specified version format) AND invalid samples (as v0 records). This allows you
 *   to track sampling errors and understand data quality.
 * - Future: ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES will enable partially
 *   valid records.
 *
 * Note: The @p record_version parameter specifies the format for VALID samples only.
 * Invalid samples (v0) always use rocprofiler_pc_sampling_record_v0_t format, regardless
 * of the specified version.
 *
 * Example 1 - Basic usage (valid samples only):
 * @code
 * // Receive only valid samples in v1 format, discard errors
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
 *     sizeof(rocprofiler_pc_sampling_record_v1_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIG_NONE);
 * @endcode
 *
 * Example 2 - Include error tracking:
 * @code
 * // Receive valid samples in v2 format AND invalid samples (v0)
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
 *     sizeof(rocprofiler_pc_sampling_record_v2_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES);
 *
 * // In your buffer callback, handle both types:
 * void buffer_callback(rocprofiler_record_header_t** headers, size_t num) {
 *     for(size_t i = 0; i < num; ++i) {
 *         if(headers[i]->kind == ROCPROFILER_BUFFER_CATEGORY_PC_SAMPLING) {
 *             auto* record = (rocprofiler_pc_sampling_record_v2_t*)(headers[i]);
 *             if(record->size == sizeof(rocprofiler_pc_sampling_record_v0_t)) {
 *                 // This is an invalid sample (error occurred)
 *                 handle_sampling_error();
 *             } else {
 *                 // This is a valid v2 sample
 *                 process_sample(record);
 *             }
 *         }
 *     }
 * }
 * @endcode
 *
 * Example 3 - Future partial samples:
 * @code
 * // When partial sampling is implemented:
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_3,
 *     sizeof(rocprofiler_pc_sampling_record_v3_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES |
 *     ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES);
 * @endcode
 *
 * Rocprofiler-SDK checks whether the requested configuration is actually supported
 * at the moment of calling this function. If the answer is yes, it returns
 * the @see ROCPROFILER_STATUS_SUCCESS. Otherwise, it notifies the client about the
 * rejection reason via the returned status code. For more information
 * about the status codes, please @see rocprofiler_status_t.
 *
 * There are a few constraints a client's code needs to be aware of.
 *
 * Constraint1: A GPU agent can be configured to support at most one running PC sampling
 * configuration at any time, which implies some of the consequences described below.
 * After the tool configures the PC sampling with one of the available configurations,
 * rocprofiler-SDK guarantees that this configuration will be valid for the tool's
 * lifetime. The tool can start and stop the configured PC sampling service whenever convenient.
 *
 * Constraint2: Since the same GPU agent can be used by multiple processes concurrently,
 * Rocprofiler-SDK cannot guarantee the exclusive access to the PC sampling capability.
 * The consequence is the following scenario. The tool TA that belongs to the process PA,
 * calls the @see rocprofiler_query_pc_sampling_agent_configurations that returns the
 * two supported configurations CA and CB by the agent. Then the tool TB of the process PB,
 * configures the PC sampling on the same agent by using the configuration CB.
 * Subsequently, the TA tries configuring the CA on the agent, and it fails.
 * To point out that this case happened, we introduce a special status code
 * @see ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE.
 * When this status code is observed by the tool TA, it queries all available configurations again
 * by calling @see rocprofiler_query_pc_sampling_agent_configurations,
 * that returns only CB this time. The tool TA can choose CB, so that both
 * TA and TB use the PC sampling capability in the separate processes.
 * Both TA and TB receives samples generated by the kernels launched by the
 * corresponding processes PA and PB, respectively.
 *
 * Constraint3: Rocprofiler-SDK allows only one context to contain the configured PC sampling
 * service within the process, that implies that at most one of the loaded tools can use PC
 * sampling. One context can contains multiple PC sampling services configured for different GPU
 * agents.
 *
 * Constraint4: PC sampling feature is not available within the ROCgdb.
 *
 * Constraint5: PC sampling service cannot be used simultaneously with
 * counter collection service.
 *
 * @param [in] context_id - id of the context used for starting/stopping PC sampling service
 * @param [in] agent_id   - id of the agent on which caller tries using PC sampling capability
 * @param [in] method     - the type of PC sampling the caller tries to use on the agent.
 * @param [in] unit       - The unit appropriate to the PC sampling type/method.
 * @param [in] interval   - frequency at which PC samples are generated
 * @param [in] buffer_id  - id of the buffer used for delivering PC samples
 * @param [in] record_version - enum specifying which PC sampling record format to use for VALID samples
 * @param [in] record_size - should be set to sizeof(rocprofiler_pc_sampling_record_vN_t) where N matches the version
 * @param [in] flags      - Configuration flags controlling which record types to deliver.
 *                          Use ROCPROFILER_PC_SAMPLING_CONFIG_NONE for valid samples only, or
 *                          bitwise OR flags like ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES
 *                          to also receive error indicators. @see rocprofiler_pc_sampling_config_flags_t
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS PC sampling service configured successfully
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE One of the scenarios is present:
 * 1. PC sampling is already configured with configuration different than requested,
 * 2. PC sampling is requested from a process that runs within the ROCgdb.
 * 3. HSA runtime does not support PC sampling.
 * 4. GPU device does not support requested PC sampling method.
 * @retval ::ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL the amdgpu driver installed on the system
 * does not support the PC sampling feature
 * @retval ::ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_ABI record_size does not match the size of the
 * requested record version struct, or the record version is not supported
 * @retval ::ROCPROFILER_STATUS_ERROR a general error caused by the amdgpu driver
 * @retval ::ROCPROFILER_STATUS_ERROR_CONTEXT_CONFLICT counter collection service already
 * setup in the context
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT function invoked with an invalid argument,
 * invalid record_version value, record_size mismatch, or invalid flags combination
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t                   context_id,
                                          rocprofiler_agent_id_t                     agent_id,
                                          rocprofiler_pc_sampling_method_t           method,
                                          rocprofiler_pc_sampling_unit_t             unit,
                                          uint64_t                                   interval,
                                          rocprofiler_buffer_id_t                    buffer_id,
                                          rocprofiler_pc_sampling_record_version_t   record_version,
                                          size_t                                     record_size,
                                          uint32_t                                   flags) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
