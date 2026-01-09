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
 * @brief (experimental) Flags for configuring PC sampling service behavior
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_configure_flags_t
{
    ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE                    = 0,
    ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION = 1 << 0,

    /// @var ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION
    /// @brief Allow overriding a previously configured valid record version (VERSION_1-7).
    /// Without this flag, attempting to configure a different valid version after one has
    /// already been configured will return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED.
    /// With this flag, the new version will replace the existing valid version configuration.
    /// This flag has no effect on VERSION_0 (invalid samples) configuration.
} rocprofiler_pc_sampling_configure_flags_t;

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
 * and sampling methods, please @see ::rocprofiler_pc_sampling_record_kind_t
 *
 * Multiple Configuration Calls:
 * This function can be called multiple times for the same agent, but with important restrictions:
 * - At most ONE valid version (VERSION_0 through VERSION_5) can be configured per agent
 * - Once a valid version is configured, attempting to configure a different valid version
 *   will return ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED (unless the
 *   ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION flag is used)
 * - VERSION_0 (invalid samples) can be configured independently, either alone or alongside
 *   one valid version
 * - Attempting to configure the same version multiple times will return
 *   ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
 *
 * Overriding Valid Versions:
 * The ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION flag allows replacing
 * a previously configured valid version with a different one. This is useful when you need
 * to switch between different sampling record formats without restarting the context.
 *
 * Valid use cases for multiple calls:
 * - Enabling one valid sample version (VERSION_0-5) and invalid samples (VERSION_0)
 * - Configuring VERSION_0 alone to receive only invalid/error samples for debugging
 * - Overriding a previously configured valid version with a different one using the OVERRIDE flag
 *
 * The order of configuration calls does not matter - VERSION_0 can be configured before or after
 * the valid version.
 *
 * Example 1 - Configure only valid samples:
 * @code
 * // Configure to receive only valid samples in v0 format
 * // Invalid samples will be silently discarded
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 * @endcode
 *
 * Example 2 - Configure valid samples AND invalid samples:
 * @code
 * // Step 1: Configure valid samples (VERSION_0)
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 *
 * // Step 2: Enable invalid sample delivery
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE,
 *     sizeof(rocprofiler_pc_sampling_record_invalid_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 *
 * // Now buffer receives both:
 * // - Valid samples as rocprofiler_pc_sampling_record_v0_t
 * // - Invalid samples as rocprofiler_pc_sampling_record_invalid_t
 * @endcode
 *
 * Example 3 - Configure ONLY invalid samples:
 * @code
 * // Configure only VERSION_0 to receive only invalid/error samples
 * // Useful for debugging sampling failures
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_INVALID_SAMPLE,
 *     sizeof(rocprofiler_pc_sampling_record_invalid_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 * @endcode
 *
 * Example 4 - INVALID: Duplicate version configuration:
 * @code
 * // Step 1: Configure VERSION_0
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 *
 * // Step 2: Try to configure VERSION_0 again - REJECTED
 * // Returns ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,  // ERROR: Already configured
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 * @endcode
 *
 * Example 5 - INVALID: Configuring different valid versions:
 * @code
 * // Step 1: Configure VERSION_0
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 *
 * // Step 2: Try to configure VERSION_1 - REJECTED
 * // Returns ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED
 * // Only ONE valid version can be configured per agent
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,  // ERROR: Different valid version
 *     sizeof(rocprofiler_pc_sampling_record_v1_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 * @endcode
 *
 * Example 6 - Overriding a valid version with the OVERRIDE flag:
 * @code
 * // Step 1: Configure VERSION_0
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
 *     sizeof(rocprofiler_pc_sampling_record_v0_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE);
 *
 * // Step 2: Override with VERSION_1 using the OVERRIDE flag - ACCEPTED
 * rocprofiler_configure_pc_sampling_service(
 *     context_id, agent_id, method, unit, interval, buffer_id,
 *     ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
 *     sizeof(rocprofiler_pc_sampling_record_v1_t),
 *     ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION);
 *
 * // Now VERSION_0 is replaced with VERSION_1
 * // Buffer will receive samples in v1 format instead of v0
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
 * Constraint3: One context per agent can be configured for PC sampling, but multiple contexts
 * for different agents can be configured within the same process.
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
 * @param [in] record_version - enum specifying which PC sampling record format to use.
 *                              Each version can only be configured once per agent.
 * @param [in] record_size - should be set to sizeof(rocprofiler_pc_sampling_record_vN_t) where N
 * matches the version
 * @param [in] flags      - configuration flags from rocprofiler_pc_sampling_configure_flags_t.
 *                          Use ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_NONE for default behavior,
 *                          or ROCPROFILER_PC_SAMPLING_CONFIGURE_FLAGS_OVERRIDE_RECORD_VERSION
 *                          to replace a previously configured valid version
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
 * invalid record_version value, or record_size mismatch
 * @retval ::ROCPROFILER_STATUS_ERROR_SERVICE_ALREADY_CONFIGURED Attempting to configure the same
 * record version more than once for the same agent
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service_(rocprofiler_context_id_t                  context_id,
                                           rocprofiler_agent_id_t                    agent_id,
                                           rocprofiler_pc_sampling_method_t          method,
                                           rocprofiler_pc_sampling_unit_t            unit,
                                           uint64_t                                  interval,
                                           rocprofiler_buffer_id_t                   buffer_id,
                                           rocprofiler_pc_sampling_record_kind_t     record_version,
                                           size_t                                    record_size,
                                           rocprofiler_pc_sampling_configure_flags_t flags)
    ROCPROFILER_API;

/**
 * @brief (experimental) Enumeration describing values of flags of
 * ::rocprofiler_pc_sampling_configuration_t.
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_configuration_flags_t
{
    ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_NONE = 0,
    ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2,
    ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_LAST

    /// @var ROCPROFILER_PC_SAMPLING_CONFIGURATION_FLAGS_INTERVAL_POW2
    /// @brief The interval value must be a power of 2.
} rocprofiler_pc_sampling_configuration_flags_t;

/**
 * @brief (experimental) PC sampling configuration supported by a GPU agent.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_configuration_t
{
    uint64_t                         size;  ///< Size of this struct
    rocprofiler_pc_sampling_method_t method;
    rocprofiler_pc_sampling_unit_t   unit;
    size_t                           min_interval;
    size_t                           max_interval;
    uint64_t flags;  ///< take values from ::rocprofiler_pc_sampling_configuration_flags_t

    /// @var method
    /// @brief Sampling method supported by the GPU agent.
    /// Currently, it can take one of the following two values:
    /// - ::ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP: a background host kernel thread
    /// periodically interrupts waves execution on the GPU to generate PC samples
    /// - ::ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC: performance monitoring hardware
    /// on the GPU periodically interrupts waves to generate PC samples.
    /// @var unit
    /// @brief A unit used to specify the interval of the @ref method for samples generation.
    /// @var min_interval
    /// @brief the highest possible frequencey for generating samples using @ref method.
    /// @var max_interval
    /// @brief the lowest possible frequency for generating samples using @ref method

} rocprofiler_pc_sampling_configuration_t;

/**
 * @brief (experimental) Rocprofiler SDK's callback function to deliver the list of available PC
 * sampling configurations upon the call to the
 * ::rocprofiler_query_pc_sampling_agent_configurations.
 *
 * @param[out] configs - The array of PC sampling configurations supported by the agent
 * at the moment of invoking ::rocprofiler_query_pc_sampling_agent_configurations.
 * @param[out] num_config - The number of configurations contained in the underlying array
 * @p configs.
 * In case the GPU agent does not support PC sampling, the value is 0.
 * @param[in] user_data - client's private data passed via
 * ::rocprofiler_query_pc_sampling_agent_configurations
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef rocprofiler_status_t (*rocprofiler_available_pc_sampling_configurations_cb_t)(
    const rocprofiler_pc_sampling_configuration_t* configs,
    size_t                                         num_config,
    void*                                          user_data);

/**
 * @brief (experimental) Query PC Sampling Configuration.
 *
 * Lists PC sampling configurations a GPU agent with @p agent_id supports at the moment
 * of invoking the function. Delivers configurations via @p cb.
 * In case the PC sampling is configured on the GPU agent, the @p cb delivers information
 * about the active PC sampling configuration.
 * In case the GPU agent does not support PC sampling capability,
 * the @p cb delivers none PC sampling configurations.
 *
 * @param [in] agent_id  - id of the agent for which available configurations will be listed
 * @param [in] cb        - User callback that delivers the available PC sampling configurations
 * @param [in] user_data - passed to the @p cb
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE One of the scenarios is present:
 * 1. PC sampling is requested from a process that runs within the ROCgdb.
 * 2. HSA runtime does not support PC sampling.
 * @retval ::ROCPROFILER_STATUS_ERROR_INCOMPATIBLE_KERNEL the amdgpu driver installed on the system
 * does not support the PC sampling feature.
 * @retval ::ROCPROFILER_STATUS_ERROR a general error caused by the amdgpu driver
 * @retval ::ROCPROFILER_STATUS_SUCCESS @p cb successfully finished
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t                                agent_id,
    rocprofiler_available_pc_sampling_configurations_cb_t cb,
    void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2, 3);

/**
 * @brief (experimental) Information about the GPU part where wave was executing
 * at the moment of sampling.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_hw_id_v0_t
{
    uint64_t chiplet          : 6;  ///< chiplet index (3 bits allocated by the ROCr runtime)
    uint64_t wave_id          : 7;  ///< wave slot index
    uint64_t simd_id          : 2;  ///< SIMD index
    uint64_t pipe_id          : 4;  ///< pipe index
    uint64_t cu_or_wgp_id     : 4;
    uint64_t shader_array_id  : 1;   ///< Shared array index
    uint64_t shader_engine_id : 5;   ///< shared engine index
    uint64_t workgroup_id     : 7;   ///< thread_group index on GFX9, and workgroup index on GFX10+
    uint64_t vm_id            : 6;   ///< virtual memory ID
    uint64_t queue_id         : 4;   ///< queue id
    uint64_t microengine_id   : 2;   ///< ACE (microengine) index
    uint64_t reserved0        : 16;  ///< Reserved for the future use

    /// @var cu_or_wgp_id
    /// @brief Compute unit index on GFX9 or workgroup processor index on GFX10+.
} rocprofiler_pc_sampling_hw_id_v0_t;

/**
 * @brief (experimental) Sampled program counter.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_t
{
    uint64_t code_object_id;
    uint64_t code_object_offset;

    /// @var code_object_id
    /// @brief id of the loaded code object instance that contains sampled PC.
    /// This fields holds the value ::ROCPROFILER_CODE_OBJECT_ID_NONE
    /// if the code object cannot be determined
    /// (e.g., sampled PC belongs to code generated by self modifying code).
    /// @var code_object_offset
    /// @brief If @ref code_object_id is different than ::ROCPROFILER_CODE_OBJECT_ID_NONE,
    /// then this field contains the offset of the sampled PC relative to the
    /// ::rocprofiler_callback_tracing_code_object_load_data_t.load_base
    /// of the code object instance with @ref code_object_id.
    /// To calculate the original virtual address of the sampled PC, one can add the value
    /// of this field to the ::rocprofiler_callback_tracing_code_object_load_data_t.load_base.
    /// The value of @ref code_object_offset matches
    /// the virtual address of the sampled instruction (PC), only if the
    /// @ref code_object_id is equal to the ::ROCPROFILER_CODE_OBJECT_ID_NONE.
} rocprofiler_pc_t;

/**
 * @brief (experimental) ROCProfiler Host-Trap PC Sampling Record.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_host_trap_v0_t
{
    uint64_t                           size;         ///< Size of this struct
    rocprofiler_pc_sampling_hw_id_v0_t hw_id;        ///< @see ::rocprofiler_pc_sampling_hw_id_v0_t
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;
    rocprofiler_dim3_t                 workgroup_id;  ///< wave coordinates within the workgroup
    uint32_t wave_in_group : 8;                       ///< wave position within the workgroup (0-31)
    uint32_t reserved0     : 24;                      ///< wave position within the workgroup (0-31)

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_host_trap_v0_t;

/**
 * @brief (experimental) The header of the ::rocprofiler_pc_sampling_record_stochastic_v0_t,
 * indicating what fields of the ::rocprofiler_pc_sampling_record_stochastic_v0_t instance are
 * meaningful for the sample.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_stochastic_header_t
{
    uint8_t has_memory_counter : 1;  ///< pc sample provides memory counters information
                                     ///< via ::rocprofiler_pc_sampling_memory_counters_t
    uint8_t reserved_type : 7;
} rocprofiler_pc_sampling_record_stochastic_header_t;

/**
 * @brief (experimental) Enumeration describing type of sampled issued instruction.
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_instruction_type_t
{
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NONE = 0,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_VALU,        ///< vector ALU instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MATRIX,      ///< matrix instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_SCALAR,      ///< scalar (memory) instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_TEX,         ///< texture memory instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS,         ///< LDS memory instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LDS_DIRECT,  ///< LDS direct memory instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_FLAT,        ///< flat memory instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_EXPORT,      ///< export instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_MESSAGE,     ///< message instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BARRIER,     ///< barrier instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_JUMP,       ///< jump instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_OTHER,      ///< other types of instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_NO_INST,    ///< no instruction issued
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_DUAL_VALU,  /// dual VALU instruction
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_LAST

    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_NOT_TAKEN
    /// @brief Instruction representing a branch not being taken.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_TYPE_BRANCH_TAKEN
    /// @brief Instruction representing a taken branch.
} rocprofiler_pc_sampling_instruction_type_t;

/**
 * @brief (experimental) Enumeration describing reason for not issuing an instruction.
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_instruction_not_issued_reason_t
{
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NONE = 0,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT,  ///< waitcnt dependency
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_BARRIER_WAIT,  ///< waiting on a barrier
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT,
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_SLEEP_WAIT,  ///< wave was sleeping
    ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_LAST

    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_NO_INSTRUCTION_AVAILABLE
    /// @brief No instruction available in the instruction cache.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY
    /// @brief ALU dependency not resolved.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_INTERNAL_INSTRUCTION
    /// @brief Wave executes an internal instruction.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_NOT_WIN
    /// @brief The instruction did not win the arbiter.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ARBITER_WIN_EX_STALL
    /// @brief Arbiter issued an instruction, but the execution pipe pushed it back from execution.
    /// @var ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_OTHER_WAIT
    /// @brief Other types of wait (e.g., wait for XNACK acknowledgment).
} rocprofiler_pc_sampling_instruction_not_issued_reason_t;

/**
 * @brief (experimental) Data provided by stochastic sampling hardware.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_snapshot_v0_t
{
    uint32_t reason_not_issued          : 4;
    uint32_t reserved0                  : 1;  ///< reserved for future use
    uint32_t arb_state_issue_valu       : 1;  ///< arbiter issued a VALU instruction
    uint32_t arb_state_issue_matrix     : 1;  ///< arbiter issued a matrix instruction
    uint32_t arb_state_issue_lds        : 1;  ///< arbiter issued a LDS instruction
    uint32_t arb_state_issue_lds_direct : 1;  ///< arbiter issued a LDS direct instruction
    uint32_t arb_state_issue_scalar     : 1;  ///< arbiter issued a scalar (SALU/SMEM) instruction
    uint32_t arb_state_issue_vmem_tex   : 1;  ///< arbiter issued a texture instruction
    uint32_t arb_state_issue_flat       : 1;  ///< arbiter issued a FLAT instruction
    uint32_t arb_state_issue_exp        : 1;  ///< arbiter issued a export instruction
    uint32_t arb_state_issue_misc       : 1;  ///< arbiter issued a miscellaneous instruction
    uint32_t arb_state_issue_brmsg      : 1;  ///< arbiter issued a branch/message instruction
    uint32_t arb_state_issue_reserved   : 1;  ///< reserved for the future use
    uint32_t arb_state_stall_valu       : 1;
    uint32_t arb_state_stall_matrix     : 1;  ///< matrix instruction was stalled
    uint32_t arb_state_stall_lds        : 1;  ///< LDS instruction was stalled
    uint32_t arb_state_stall_lds_direct : 1;  ///< LDS direct instruction was stalled
    uint32_t arb_state_stall_scalar     : 1;  ///< Scalar (SALU/SMEM) instruction was stalled
    uint32_t arb_state_stall_vmem_tex   : 1;  ///< texture instruction was stalled
    uint32_t arb_state_stall_flat       : 1;  ///< flat instruction was stalled
    uint32_t arb_state_stall_exp        : 1;  ///< export instruction was stalled
    uint32_t arb_state_stall_misc       : 1;  ///< miscellaneous instruction was stalled
    uint32_t arb_state_stall_brmsg      : 1;  ///< branch/message instruction was stalled
    uint32_t arb_state_state_reserved   : 1;  ///< reserved for the future use
    // We have two reserved bits
    uint32_t dual_issue_valu : 1;
    uint32_t reserved1       : 1;  ///< reserved for the future use
    uint32_t reserved2       : 3;  ///< reserved for the future use

    /// @var reason_not_issued
    /// @brief The reason for not issuing an instruction. The field takes one of the value defined
    /// in ::rocprofiler_pc_sampling_instruction_not_issued_reason_t
    /// @var arb_state_stall_valu
    /// @brief VALU instruction was stalled when a sample was generated
    /// @var dual_issue_valu
    /// @brief Two VALU instructions were issued for coexecution (MI3xx specific)
} rocprofiler_pc_sampling_snapshot_v0_t;

/**
 * @brief (experimental) Counters of issued but not yet completed instructions.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_t
{
    uint32_t load_cnt   : 6;
    uint32_t store_cnt  : 6;
    uint32_t bvh_cnt    : 3;
    uint32_t sample_cnt : 6;
    uint32_t ds_cnt     : 6;
    uint32_t km_cnt     : 5;

    /// @var load_cnt
    /// @brief Counts the number of VMEM load instructions issued but not yet completed.
    /// @var store_cnt
    /// @brief Counts the number of VMEM store instructions issued but not yet completed.
    /// @var bvh_cnt
    /// @brief Counts the number of VMEM BVH instructions issued but not yet completed.
    /// @var sample_cnt
    /// @brief Counts the number of VMEM sample instructions issued but not yet completed.
    /// @var ds_cnt
    /// @brief Counts the number of LDS instructions issued but not yet completed.
    /// @var km_cnt
    /// @brief Counts the number of scalar memory reads and memory instructions issued but not yet
    /// completed.
} rocprofiler_pc_sampling_memory_counters_t;

/**
 * @brief (experimental) ROCProfiler Stochastic PC Sampling Record.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_stochastic_v0_t
{
    uint64_t                                           size;  ///< Size of this struct
    rocprofiler_pc_sampling_record_stochastic_header_t flags;
    uint8_t                                            wave_in_group;
    uint8_t                                            wave_issued : 1;
    uint8_t                                            inst_type   : 5;
    uint8_t                                            reserved    : 2;
    rocprofiler_pc_sampling_hw_id_v0_t                 hw_id;
    rocprofiler_pc_t                                   pc;
    uint64_t                                           exec_mask;
    rocprofiler_dim3_t                                 workgroup_id;
    uint32_t                                           wave_count;
    uint64_t                                           timestamp;
    uint64_t                                           dispatch_id;
    rocprofiler_async_correlation_id_t                 correlation_id;
    rocprofiler_pc_sampling_snapshot_v0_t              snapshot;
    rocprofiler_pc_sampling_memory_counters_t          memory_counters;

    /// @var flags
    /// @brief Defines what fields are meaningful for the sample.
    /// @var wave_in_group
    /// @brief wave position within the workgroup (0-15)
    /// @var wave_issued
    /// @brief wave issued the instruction represented with the PC
    /// @var inst_type
    /// @brief instruction type, takes a value defined in @ref
    /// ::rocprofiler_pc_sampling_instruction_type_t
    /// @var reserved
    /// @brief reserved 2 bits must be zero
    /// @var hw_id
    /// @brief @see ::rocprofiler_pc_sampling_hw_id_v0_t
    /// @var pc
    /// @brief information about sampled program counter
    /// @var exec_mask
    /// @brief active SIMD lanes at the moment of sampling
    /// @var workgroup_id
    /// @brief wave coordinates within the workgroup
    /// @var wave_count
    /// @brief active waves on the CU at the moment of sampling
    /// @var timestamp
    /// @brief timestamp when sample is generated
    /// @var dispatch_id
    /// @brief originating kernel dispatch ID
    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
    /// @var snapshot
    /// @brief Data provided by stochastic sampling hardware. @see
    /// ::rocprofiler_pc_sampling_snapshot_v0_t
    /// @var memory_counters
    /// @brief Counters of issued but not yet completed instructions. @see
    /// ::rocprofiler_pc_sampling_memory_counters_t
} rocprofiler_pc_sampling_record_stochastic_v0_t;

/**
 * @brief (experimental) Record representing an invalid PC Sampling Record.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_invalid_t
{
    uint64_t size;  ///< Size of the struct
} rocprofiler_pc_sampling_record_invalid_t;

/**
 * @brief (experimental) Return the string encoding of ::rocprofiler_pc_sampling_instruction_type_t
 * value
 * @param [in] instruction_type instruction type enum value
 * @param [out] name pointer to store the name string
 * @param [out] name_len pointer to store the length of the name string
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS if valid instruction_type is provided
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if invalid/unsupported instruction_type is
 * provided
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_get_pc_sampling_instruction_type_name_(
    rocprofiler_pc_sampling_instruction_type_t instruction_type,
    const char**                               name,
    uint64_t*                                  name_len) ROCPROFILER_API;

/**
 * @brief (experimental) Return the string encoding of
 * ::rocprofiler_pc_sampling_instruction_not_issued_reason_t value
 * @param [in] not_issued_reason no issue reason enum value
 * @param [out] name pointer to store the name string
 * @param [out] name_len pointer to store the length of the name string
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS if valid not_issued_reason is provided
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if invalid/unsupported not_issued_reason is
 * provided
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_get_pc_sampling_instruction_not_issued_reason_name_(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t not_issued_reason,
    const char**                                            name,
    uint64_t*                                               name_len) ROCPROFILER_API;

/**
 * @brief Information provided by snapshot block (relevant for stochastic PC sampling only)
 *
 * 8B in total.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_snapshot_information_v0_t
{
    uint8_t wave_issued;       ///< 1 - wave issued an instruction at the moment of sampling;
                               ///< 0 - wave didn't issue an instruction at the moment of sampling.
    uint8_t instruction_type;  ///< if wave_issued=1, type of issued instruction (see
                               ///< ::rocprofiler_pc_sampling_instruction_type_t); otherwise might
                               ///< be irrelevant
    uint8_t
        no_issue_reason;  ///< if wave_issue=0, reason for not issuing the instruction (see
                          ///< ::rocprofiler_pc_sampling_no_issue_reason_t); otherwise irrelevant
    uint8_t wave_count;  ///< number of concurrently running waves on CU (on GFX9) or SIMD (GFX10+)
                         ///< at the moment of sampling
    uint32_t arbiter_state;  ///< arbiter state bitfield. To decode, use
                             ///< ::rocprofiler_pc_sampling_arbiter_state_field_id_t and helper
                             ///< functions.
} rocprofiler_pc_sampling_snapshot_information_v0_t;

ROCPROFILER_CXX_CODE(
    static_assert(sizeof(rocprofiler_pc_sampling_snapshot_information_v0_t) == 8,
                  "Increasing the size of the rocprofiler_pc_sampling_snapshot_information_v0_t is "
                  "not permitted");)

/**
 * @brief (experimental) Memory counters for GFX12 architectures.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_v0_t
{
    uint8_t load_count;   ///< Number of VMEM load instructions issued but not yet completed
    uint8_t store_count;  ///< Number of VMEM store instructions issued but not yet completed
    uint8_t ds_count;     ///< Number of LDS instructions issued but not yet completed
    uint8_t km_count;   ///< Number of scalar memory reads/instructions issued but not yet completed
    uint8_t bvh_count;  ///< Number of VMEM BVH instructions issued but not yet completed
    uint8_t sample_count;  ///< Number of VMEM sample instructions issued but not yet completed
    uint8_t reserved[2];
} rocprofiler_pc_sampling_memory_counters_v0_t;

ROCPROFILER_CXX_CODE(
    static_assert(sizeof(rocprofiler_pc_sampling_memory_counters_v0_t) == 6 + 2,
                  "Increasing the size of the rocprofiler_pc_sampling_memory_counters_v0_t is not "
                  "permitted");)

/**
 * @brief (experimental) Reserved for the future architectures.
 *
 * Total size must not exceed 16 bytes.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_v1_t
{
    uint8_t load_count;   ///< Number of VMEM load instructions issued but not yet completed
    uint8_t store_count;  ///< Number of VMEM store instructions issued but not yet completed
    uint8_t ds_count;     ///< Number of LDS instructions issued but not yet completed
    uint8_t km_count;   ///< Number of scalar memory reads/instructions issued but not yet completed
    uint8_t bvh_count;  ///< Number of VMEM BVH instructions issued but not yet completed
    uint8_t sample_count;  ///< Number of VMEM sample instructions issued but not yet completed
    // For descriptive purpose only. Mimics counters that will be introduced in the future.
    uint8_t reserved0;  ///< Reserved for future use
    uint8_t reserved1;  ///< Reserved for future use
    uint8_t reserved2;  ///< Reserved for future use
    uint8_t reserved[7];
} rocprofiler_pc_sampling_memory_counters_v1_t;

ROCPROFILER_CXX_CODE(
    static_assert(sizeof(rocprofiler_pc_sampling_memory_counters_v1_t) == 9 + 7,
                  "Increasing the size of the rocprofiler_pc_sampling_memory_counters_v1_t is not "
                  "permitted");)

/**
 * @brief (experimental) Reserved for the future architectures
 *
 * Total size must not exceed 16 bytes.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_memory_counters_v2_t
{
    uint8_t load_count;   ///< Number of VMEM load instructions issued but not yet completed
    uint8_t store_count;  ///< Number of VMEM store instructions issued but not yet completed
    uint8_t ds_count;     ///< Number of LDS instructions issued but not yet completed
    uint8_t km_count;  ///< Number of scalar memory reads/instructions issued but not yet completed
    uint8_t sample_count;  ///< Number of VMEM sample instructions issued but not yet completed
    // For descriptive purpose only. Mimics counters that will be introduced in the future.
    uint8_t reserved0;  ///< Reserved for future use
    uint8_t reserved1;  ///< Reserved for future use
    uint8_t reserved2;  ///< Reserved for future use
    uint8_t reserved3;  ///< Reserved for future use
    uint8_t reserved4;  ///< Reserved for future use
    uint8_t reserved[6];
} rocprofiler_pc_sampling_memory_counters_v2_t;

ROCPROFILER_CXX_CODE(
    static_assert(sizeof(rocprofiler_pc_sampling_memory_counters_v2_t) == 10 + 6,
                  "Increasing the size of the rocprofiler_pc_sampling_memory_counters_v2_t is not "
                  "permitted");)

/**
 * @brief Information about where was running when sampled.
 *
 * If size of the records is not that important, we should use this version as it's simpler.
 * If however we need to save some more space in PC sampling record, then we can fall back
 * to packing this information into 64-bit and use bit-offsets.
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_hw_id_v1_t
{
    uint8_t chiplet;  ///< chiplet index (3 bits allocated by the ROCr runtime)
    uint8_t wave_id;  ///< wave slot index
    uint8_t simd_id;  ///< SIMD index
    uint8_t pipe_id;  ///< pipe index
    uint8_t cu_or_wgp_id;
    uint8_t shader_array_id;   ///< Shared array index
    uint8_t shader_engine_id;  ///< shared engine index
    uint8_t workgroup_id;      ///< thread_group index on GFX9, and workgroup index on GFX10+
    uint8_t vm_id;             ///< virtual memory ID
    uint8_t queue_id;          ///< queue id
    uint8_t microengine_id;    ///< ACE (microengine) index
    uint8_t reserved[5];
    /// @var cu_or_wgp_id
    /// @brief Compute unit index on GFX9 or workgroup processor index on GFX10+.
} rocprofiler_pc_sampling_hw_id_v1_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_hw_id_v1_t) == 11 + 5,
        "Increasing the size of the rocprofiler_pc_sampling_hw_id_v1_t is not permitted");)

/**
 * @brief 64B in total (experimental) Minimal PC sampling record acceptable on all architectures
 * (e.g., MI200, MI300, MI350)
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v0_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v0_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v0_t) == 56,
        "Increasing the size of the rocprofiler_pc_sampling_record_v0_t is not permitted");)

/**
 * @brief 88B in total (experimental) PC sampling records tailored for host-trap on GFX9 and Navi4x
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v1_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 11B (maybe padded to 16B)
    rocprofiler_pc_sampling_hw_id_v1_t hw_id;
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 4B for padding (1B is suffient)
    uint32_t wave_in_group;  ///< wave position in the workgroup

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v1_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v1_t) == 88,
        "Increasing the size of the rocprofiler_pc_sampling_record_v1_t is not permitted");)

/**
 * @brief 96B in total (experimental) PC Sampling Record tailored for stochastic sampling on
 * MI300/MI350
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v2_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 11B (padded to 16B)
    rocprofiler_pc_sampling_hw_id_v1_t
        hw_id;  ///< 8B if we use ::rocprofiler_pc_sampling_hw_id_record_packed_t
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 4B for padding (1B is suffient)
    uint32_t wave_in_group;  ///< wave position in the workgroup
    // 8B
    rocprofiler_pc_sampling_snapshot_information_v0_t snapshot_information;

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v2_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v2_t) == 96,
        "Increasing the size of the rocprofiler_pc_sampling_record_v2_t is not permitted");)

/**
 * @brief 104B in total (experimental) PC sampling record tailored for host-trap sampling
 * on future gen architectures.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v3_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 11B (padded to 16B)
    rocprofiler_pc_sampling_hw_id_v1_t
        hw_id;  ///< 8B if we use ::rocprofiler_pc_sampling_hw_id_record_packed_t
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 7B (will probably be padded to 8B)
    uint8_t wave_in_group;  ///< wave position in the workgroup
    uint8_t reserved0;      ///< must be zero
    uint8_t reserved1;      ///< must be zero
    uint8_t reserved2;      ///< must be zero
    uint8_t reserved3;      ///< must be zero
    uint8_t reserved4;      ///< must be zero
    uint8_t reserved5;      ///< must be zero
    // 12B
    rocprofiler_dim3_t reserved6;  ///< reserved for the future use (must be zero)

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
} rocprofiler_pc_sampling_record_v3_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v3_t) == 104,
        "Increasing the size of the rocprofiler_pc_sampling_record_v3_t is not permitted");)

/**
 * @brief 128B in total (experimental) PC sampling record tailored for stochastic on future gen
 * architectures.
 *
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v4_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 11B (padded to 16B)
    rocprofiler_pc_sampling_hw_id_v1_t
        hw_id;  ///< 8B if we use ::rocprofiler_pc_sampling_hw_id_record_packed_t
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 7B (will probably be padded to 8B)
    uint8_t wave_in_group;  ///< wave position in the workgroup
    uint8_t reserved0;      ///< must be zero
    uint8_t reserved1;      ///< must be zero
    uint8_t reserved2;      ///< must be zero
    uint8_t reserved3;      ///< must be zero
    uint8_t reserved4;      ///< must be zero
    uint8_t reserved5;      ///< must be zero
    // 12B
    rocprofiler_dim3_t reserved6;  ///< reserved for the future use (must be zero)
    // 8B
    rocprofiler_pc_sampling_snapshot_information_v0_t snapshot_information;
    // 9B (probably padded to 16B)
    rocprofiler_pc_sampling_memory_counters_v1_t memory_counters;

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
    /// @var flags
    /// @var memory_counters
    /// @brief Memory counters (@see ::rocprofiler_pc_sampling_memory_counters_v1_t).
} rocprofiler_pc_sampling_record_v4_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v4_t) == 128,
        "Increasing the size of the rocprofiler_pc_sampling_record_v4_t is not permitted");)

/**
 * @brief 128B in total (experimental) PC sampling record tailored for stochastic future gen arch
 */
typedef struct ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_record_v5_t
{
    rocprofiler_pc_t                   pc;           ///< information about sampled program counter
    uint64_t                           exec_mask;    ///< active SIMD lanes when sampled
    uint64_t                           timestamp;    ///< timestamp when sample is generated
    uint64_t                           dispatch_id;  ///< originating kernel dispatch ID
    rocprofiler_async_correlation_id_t correlation_id;

    // 11B (padded to 16B)
    rocprofiler_pc_sampling_hw_id_v1_t
        hw_id;  ///< 8B if we use ::rocprofiler_pc_sampling_hw_id_record_packed_t
    // 12B
    rocprofiler_dim3_t workgroup_position;  ///< work group position in 3D grid
    // 7B (will probably be padded to 8B)
    uint8_t wave_in_group;  ///< wave position in the workgroup
    uint8_t reserved0;      ///< reserved for the future use (must be zero)
    uint8_t reserved1;      ///< reserved for the future use (must be zero)
    uint8_t reserved2;      ///< reserved for the future use (must be zero)
    uint8_t reserved3;      ///< reserved for the future use (must be zero)
    uint8_t reserved4;      ///< reserved for the future use (must be zero)
    uint8_t reserved5;      ///< reserved for the future use (must be zero)
    // 12B
    rocprofiler_dim3_t reserved6;  ///< reserved for the future use (must be zero)
    // 8B
    rocprofiler_pc_sampling_snapshot_information_v0_t snapshot_information;
    // 10B (padded to 16B)
    rocprofiler_pc_sampling_memory_counters_v2_t memory_counters;

    /// @var correlation_id
    /// @brief API launch call id that matches dispatch ID
    /// @var memory_counters
    /// @brief Memory counters (@see ::rocprofiler_pc_sampling_memory_counters_v2_t).
} rocprofiler_pc_sampling_record_v5_t;

ROCPROFILER_CXX_CODE(
    static_assert(
        sizeof(rocprofiler_pc_sampling_record_v5_t) == 128,
        "Increasing the size of the rocprofiler_pc_sampling_record_v5_t is not permitted");)

/**
 * @brief (experimental) IDs of arbiter_state field.
 *
 * TODO: Think about ordering based on commonalities among architectures.
 */
typedef enum
{
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_NONE = 0,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_MATRIX,  ///< GFX9 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_LDS_DIRECT,  ///< GFX12 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_FLAT,  ///< GFX9 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_ISSUE_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VALU,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_MATRIX,  ///< GFX9 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_LDS_DIRECT,  ///< GFX12 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_SCALAR,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_VMEM_TEX,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_FLAT,  ///< GFX9 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_EXP,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_STALL_BRMSG_MISC,
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_DUAL_ISSUE_VALU,  ///< GFX9 specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED0,        ///< future gen specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED1,        ///< future gen specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_RESERVED2,        ///< future gen specific
    ROCPROFILER_PC_SAMPLING_ARBITER_STATE_FIELD_ID_LAST
} rocprofiler_pc_sampling_arbiter_state_field_id_t;

/**
 * @brief (experimental) (Optional) Return the string encoding of
 * ::rocprofiler_pc_sampling_arbiter_state_field_id_t value
 *
 * @param [in] field_id Arbiter state field enum value
 * @param [out] name pointer to store the name string
 * @param [out] name_len pointer to store the length of the name string
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS if valid field_id is provided
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT if invalid/unsupported field_id is provided
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_get_pc_sampling_arbiter_state_field_name(
    rocprofiler_pc_sampling_arbiter_state_field_id_t field_id,
    const char**                                     name,
    uint64_t*                                        name_len) ROCPROFILER_API;

/**
 * @brief (experimental) Callback function to deliver the list of supported arbiter state fields
 * for a specific GPU agent.
 *
 * @param[out] fields - Array of arbiter state field IDs supported by the agent
 * @param[out] num_fields - Number of fields in the array. May be 0 if arbiter state
 *                          is not supported on this agent.
 * @param[in] user_data - Client's private data passed via
 *                        ::rocprofiler_query_pc_sampling_arbiter_fields
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef rocprofiler_status_t (*rocprofiler_pc_sampling_arbiter_fields_cb_t)(
    const rocprofiler_pc_sampling_arbiter_state_field_id_t* fields,
    size_t                                                  num_fields,
    void*                                                   user_data);

/**
 * @brief (experimental) Query supported arbiter state fields for a GPU agent.
 *
 * Queries which arbiter state fields are supported by the GPU agent with @p agent_id
 * and delivers the list via the callback @p cb. Different GPU architectures support
 * different subsets of arbiter state fields.
 *
 * This function allows clients to determine at runtime which fields from
 * ::rocprofiler_pc_sampling_arbiter_state_field_id_t are meaningful for a given agent.
 * To extract field values from the arbiter_state bitfield, use
 * ::rocprofiler_pc_sampling_get_arbiter_state_fields.
 *
 * @param[in] agent_id - ID of the agent to query
 * @param[in] cb - User callback that receives the supported field IDs
 * @param[in] user_data - Passed through to @p cb
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS @p cb successfully finished
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT invalid agent_id or null callback
 * @retval ::ROCPROFILER_STATUS_ERROR_NOT_AVAILABLE agent does not support PC sampling
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_query_pc_sampling_arbiter_fields(rocprofiler_agent_id_t                      agent_id,
                                             rocprofiler_pc_sampling_arbiter_fields_cb_t cb,
                                             void* user_data) ROCPROFILER_API
    ROCPROFILER_NONNULL(2, 3);

/**
 * @brief (experimental) Callback to receive extracted arbiter state field values.
 *
 * @param[out] field_ids - Array of field IDs that were requested
 * @param[out] values - Array of extracted values corresponding to field_ids
 * @param[out] num_fields - Number of fields/values in the arrays
 * @param[in] user_data - Client's private data
 * @return ::rocprofiler_status_t
 */
ROCPROFILER_SDK_EXPERIMENTAL
typedef rocprofiler_status_t (*rocprofiler_pc_sampling_arbiter_field_values_cb_t)(
    const rocprofiler_pc_sampling_arbiter_state_field_id_t* field_ids,
    const uint32_t*                                         values,
    size_t                                                  num_fields,
    void*                                                   user_data);

/**
 * @brief (experimental) Extract multiple arbiter state field values from the bitfield.
 *
 * Helper function to extract multiple field values from the arbiter_state bitfield in
 * ::rocprofiler_pc_sampling_snapshot_information_t. The extraction uses hardware-specific
 * bit offsets and widths.
 *
 * IMPORTANT: To minimize overhead, this function does NOT validate that the requested
 * field_ids are supported. Users MUST first call ::rocprofiler_query_pc_sampling_arbiter_fields
 * to obtain the list of supported fields for their agent, then pass those field IDs to
 * this function. Passing unsupported field IDs results in undefined behavior.
 *
 * Typical usage pattern:
 * 1. Call ::rocprofiler_query_pc_sampling_arbiter_fields to get supported fields for an agent
 * 2. For each PC sample from that agent, call this function with the supported field IDs
 *    to extract their values from the arbiter_state bitfield
 *
 * @param[in] arbiter_state - The arbiter_state bitfield from a PC sampling record
 * @param[in] field_ids - Array of field IDs to extract (must be supported fields)
 * @param[in] num_fields - Number of fields to extract
 * @param[in] cb - Callback to receive the extracted values
 * @param[in] user_data - Passed through to @p cb
 * @return ::rocprofiler_status_t
 * @retval ::ROCPROFILER_STATUS_SUCCESS all fields extracted successfully and callback completed
 * @retval ::ROCPROFILER_STATUS_ERROR_INVALID_ARGUMENT null pointers or num_fields is 0
 */
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_pc_sampling_get_arbiter_state_fields(
    uint32_t                                                arbiter_state,
    const rocprofiler_pc_sampling_arbiter_state_field_id_t* field_ids,
    size_t                                                  num_fields,
    rocprofiler_pc_sampling_arbiter_field_values_cb_t       cb,
    void* user_data) ROCPROFILER_API ROCPROFILER_NONNULL(2, 4, 5);

// TODO: get rid of the size from all records
// TODO: ensure the size of the records, once you get rid of the `size` field.

/// NOTE: the following code is kept so that we could quickly compile the existing code before full
/// refactoring
ROCPROFILER_SDK_EXPERIMENTAL
rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_id_t           agent_id,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id,
                                          int                              flags) ROCPROFILER_API;

ROCPROFILER_SDK_EXPERIMENTAL
const char*
rocprofiler_get_pc_sampling_instruction_type_name(
    rocprofiler_pc_sampling_instruction_type_t instruction_type) ROCPROFILER_API;

ROCPROFILER_SDK_EXPERIMENTAL const char*
rocprofiler_get_pc_sampling_instruction_not_issued_reason_name(
    rocprofiler_pc_sampling_instruction_not_issued_reason_t not_issued_reason) ROCPROFILER_API;

/** @} */

ROCPROFILER_EXTERN_C_FINI
