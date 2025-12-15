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
 * @brief (experimental) Function used to configure the PC sampling service on the GPU agent with @p
 * agent_id.
 *
 * MULTIPLE CONFIGURATION CALLS APPROACH:
 * This API follows a pattern similar to rocprofiler_configure_buffer_tracing_service where
 * multiple calls can be made to the same function for the same agent to configure different
 * aspects of the service. This allows users to selectively enable record types.
 *
 * VERSION LOCKING SEMANTICS:
 * - The FIRST call with a NON-V0 version (VERSION_1 through VERSION_6) LOCKS the record format
 *   for all valid samples. This determines the struct layout and size for valid PC samples.
 * - Subsequent calls with DIFFERENT non-v0 versions will be REJECTED with
 *   ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED.
 * - Additional calls with VERSION_0 can be made AFTER the format is locked to enable
 *   delivery of invalid/error samples.
 * - The order of calls matters: you must lock the valid record format before enabling v0.
 * - Once VERSION_0 is enabled, it cannot be disabled (would require context restart).
 *
 * AMBIGUITY WITH V0-ONLY CONFIGURATION:
 * There is an inherent ambiguity if a user only calls this function with VERSION_0:
 *
 * Option A - REJECT V0-ONLY (Recommended):
 *   Calling with VERSION_0 only is an error. A valid record version (v1-v6) must be
 *   configured first to establish the record format, then VERSION_0 can optionally be
 *   enabled. This ensures the buffer knows what size records to allocate.
 *   Status code: ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT
 *
 * Option B - ALLOW V0-ONLY (Problematic):
 *   If only VERSION_0 is configured, the user receives ONLY invalid/error records.
 *   This creates problems:
 *   - What should record_size be? sizeof(rocprofiler_pc_sampling_record_v0_t)?
 *   - The buffer would only contain error indicators with no actual sample data
 *   - This scenario provides no useful profiling information
 *   - Valid and invalid samples would be treated identically (both "requested")
 *
 * RECOMMENDATION: Implement Option A. Require a valid version to be locked first.
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
 * Example 1 - Valid samples only (no invalid samples):
 * @code
 * // Configure to receive only valid samples in v1 format
 * // Invalid samples will be silently discarded
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
 *     sizeof(rocprofiler_pc_sampling_record_v1_t),
 *     0);
 * @endcode
 *
 * Example 2 - Valid samples + invalid samples:
 * @code
 * // Step 1: Lock the valid record format to v2
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
 *     sizeof(rocprofiler_pc_sampling_record_v2_t),
 *     0);
 *
 * // Step 2: Enable invalid sample delivery
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     0);
 *
 * // Now buffer receives both:
 * // - Valid samples as rocprofiler_pc_sampling_record_v2_t
 * // - Invalid samples as rocprofiler_pc_sampling_record_v0_t
 * @endcode
 *
 * Example 3 - Invalid: V0-only configuration (REJECTED):
 * @code
 * // This will FAIL with ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT
 * // because no valid record version has been locked
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,  // ERROR: Must configure valid version first
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     0);
 * @endcode
 *
 * Example 4 - Invalid: Changing locked version (REJECTED):
 * @code
 * // Step 1: Lock to v1
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
 *     sizeof(rocprofiler_pc_sampling_record_v1_t),
 *     0);
 *
 * // Step 2: Try to change to v3 - REJECTED
 * // Returns ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_3,  // ERROR: Version already locked to v1
 *     sizeof(rocprofiler_pc_sampling_record_v3_t),
 *     0);
 * @endcode
 *
 * Example 5 - Duplicate v0 calls (allowed, idempotent):
 * @code
 * // Step 1: Lock valid format
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
 *     sizeof(rocprofiler_pc_sampling_record_v2_t),
 *     0);
 *
 * // Step 2: Enable invalid samples
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     0);
 *
 * // Step 3: Call with v0 again - this is idempotent, returns SUCCESS
 * // Invalid samples remain enabled, no state change
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     0);
 * @endcode
 *
 * IMPLEMENTATION CONSIDERATIONS:
 *
 * 1. State Tracking:
 *    - Track whether valid version is locked (and which version)
 *    - Track whether v0 delivery is enabled
 *    - Reject attempts to change locked version
 *
 * 2. Buffer Management:
 *    - Buffer must be sized for the larger of v0 and locked valid version
 *    - Or use variable-size records (more complex)
 *
 * 3. V0-Only Prevention:
 *    - Check if valid version is locked before allowing v0 configuration
 *    - Return ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if not locked
 *
 * 4. Idempotency:
 *    - Calling with the same locked version again: returns SUCCESS (idempotent)
 *    - Calling with v0 multiple times: returns SUCCESS (idempotent)
 *
 * 5. Multiple Agents:
 *    - Each agent can have different locked versions
 *    - Each agent can independently enable/disable v0
 *
 * COMPARISON WITH BUFFER TRACING:
 * - Buffer tracing: Each "kind" is a different service (HIP API, HSA API, etc.)
 * - PC Sampling: Each "version" is a different FORMAT of the SAME data
 * - Buffer tracing: Multiple kinds can coexist naturally
 * - PC Sampling: Multiple versions are mutually exclusive (except v0)
 * - Buffer tracing: Clear semantics for multiple calls
 * - PC Sampling: Requires special locking semantics and ambiguity resolution
 *
 * POTENTIAL ISSUES:
 * - User confusion: "Why do I need to call configure twice?"
 * - Error-prone: Easy to forget the ordering requirement
 * - Unclear semantics: What if I call with v1, then v0, then v1 again?
 *   (Answer: v1 second time is idempotent, returns SUCCESS)
 * - Documentation burden: Complex rules to explain
 *
 * Constraints (same as single-call approach):
 * - Constraint1: At most one PC sampling configuration per GPU agent
 * - Constraint2: Multi-process access not guaranteed (ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE)
 * - Constraint3: Only one context can use PC sampling per process
 * - Constraint4: Not available within ROCgdb
 * - Constraint5: Cannot be used with counter collection service
 *
 * @param [in] context_id - id of the context used for starting/stopping PC sampling service
 * @param [in] agent_id   - id of the agent on which caller tries using PC sampling capability
 * @param [in] method     - the type of PC sampling the caller tries to use on the agent.
 * @param [in] unit       - The unit appropriate to the PC sampling type/method.
 * @param [in] interval   - frequency at which PC samples are generated
 * @param [in] buffer_id  - id of the buffer used for delivering PC samples
 * @param [in] record_version - enum specifying which PC sampling record format to use.
 *                              First non-v0 call locks the valid record format.
 *                              Subsequent v0 calls enable invalid sample delivery.
 * @param [in] record_size - should be set to sizeof(rocprofiler_pc_sampling_record_vN_t) where N matches the version
 * @param [in] flags      - for future use, currently must be 0
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
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT One of:
 *   - Invalid record_version value
 *   - record_size mismatch
 *   - Attempting to configure VERSION_0 before locking a valid version
 *   - Invalid argument values
 * @retval ::ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED Attempting to change the locked
 * valid record version (e.g., configuring v3 after v1 was already locked)
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
                                          int                                        flags) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
