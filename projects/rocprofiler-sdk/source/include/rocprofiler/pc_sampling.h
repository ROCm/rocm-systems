// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include <rocprofiler/agent.h>
#include <rocprofiler/defines.h>
#include <rocprofiler/fwd.h>

ROCPROFILER_EXTERN_C_INIT

/**
 * @defgroup PC_SAMPLING_SERVICE PC Sampling
 * @brief Enabling PC (Program Counter) Sampling for GPU Activity
 * @{
 */

/**
 * @brief Function used to configure the PC sampling service on the GPU agent with @p agent_id.
 * @brief Function used to configure the PC sampling service on the GPU agent with @p agent_id.
 * Prerequisites are the following:
 * - The user must create a context and supply its @p context_id. By using this context,
 * - The user must create a context and supply its @p context_id. By using this context,
 *   the user can start/stop PC sampling on the agent. For more information,
 *   please @see `rocprofiler_start_context`/`rocprofiler_stop_context`.
 * - The user must create a buffer and supply its @p buffer_id. Rocprofiler uses the buffer
 * - The user must create a buffer and supply its @p buffer_id. Rocprofiler uses the buffer
 *   to deliver the PC samples to the user. For more information about the data delivery,
 *   please @see `rocprofiler_create_buffer` and `rocprofiler_buffer_tracing_cb_t`.
 *
 * Before calling this function, we recommend querying PC sampling configurations
 * supported by the GPU agent via the `rocprofiler_query_pc_sampling_agent_configurations`.
 * The user then chooses the @p method, @p unit, and @p interval to match one of the
 * available configurations. Note that the @p interval must belong to the range of values
 * The user then chooses the @p method, @p unit, and @p interval to match one of the
 * available configurations. Note that the @p interval must belong to the range of values
 * [available_config.min_interval, available_config.max_interval],
 * where available_config is the instance of the `rocprofiler_pc_sampling_configuration_s`
 * supported at the moment.
 *
 * Rocprofiler checks whether the requsted configuration is actually supported
 * at the moment of calling this function. If the answer is yes, it returns
 * the ROCPROFILER_STATUS_SUCCESS. Otherwise, notifies the caller about the
 * rejection reason via the returned status code. For more information
 * about the status codes, please @see rocprofiler_status_t.
 *
 * Constraint1: A GPU agent can be configured to support at most one running PC sampling
 * configuration at any time, which implies some of the consequences described below.
 * After the tool configures the PC sampling with one of the available configurations,
 * rocprofiler guarantees that this configuration will be valid for the tool's
 * lifetime. The tool can start and stop the configured PC sampling service whenever convenient.
 *
 * Constraint2: Since the same GPU agent can be used by multiple processes concurrently,
 * Rocprofiler cannot guarantee the exclusive access to the PC sampling capability.
 * The consequence is the following scenario. The tool TA that belongs to the process PA,
 * calls the `rocprofiler_query_pc_sampling_agent_configurations` that returns the
 * two supported configurations CA and CB by the agent. Then the toolb TB of the process PB,
 * configures the PC sampling on the same agent by using the configuration CB.
 * Subsequently, the TA tries configuring the CA on the agent, and it fails.
 * To point out that this case happened, we introduce a special status code (TODO: ARE WE)?
 * When this status code is observed by the tool TA, it queties all available configurations again
 * by calling `rocprofiler_query_pc_sampling_agent_configurations`,
 * that returns only CB this time. The tool TA can choose CB, so that both
 * TA and TB use the PC sampling capability in the separate processes.
 *
 * Constraints3: We allow only one context to contain the configured PC sampling service
 * within the process, that implies that at most one of the loaded tools can use PC sampling.
 * One context can contains multiple PC sampling services configured for different GPU agents.
 *
 * @param [in] context_id - id of the context used for starting/stopping PC sampling service
 * @param [in] agent_id   - id of the agent on which caller tries using PC sampling capability
 * @param [in] method     - the type of PC sampling the caller tries to use on the agent.
 * @param [in] unit       - The unit appropriate to the PC sampling type/method.
 * @param [in] interval   - frequency at which PC samples are generated
 * @param [in] buffer_id  - id of the buffer used for delivering PC samples
 * @return ::rocprofiler_status_t
 *
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_configure_pc_sampling_service(rocprofiler_context_id_t         context_id,
                                          rocprofiler_agent_id_t           agent_id,
                                          rocprofiler_pc_sampling_method_t method,
                                          rocprofiler_pc_sampling_unit_t   unit,
                                          uint64_t                         interval,
                                          rocprofiler_buffer_id_t          buffer_id);

/**
 * @brief PC sampling configuration supported by a GPU agent.
 * @var rocprofiler_pc_sampling_configuration_s::method
 * Sampling method supported by the GPU
 * agent. Currenlty, it can take one of the following two values:
 * - ROCPROFILER_PC_SAMPLING_METHOD_HOST_TRAP: a background host thread
 * periodically interrupts waves execution on the GPU to generate PC samples
 * - ROCPROFILER_PC_SAMPLING_METHOD_STOCHASTIC: performance monitoring hardware
 * on the GPU periodically interrupts waves to generate PC samples.
 * @var rocprofiler_pc_sampling_configuration_s::unit
 * A unit used to specify the period of the
 * @ref method for samples generation.
 * @var rocprofiler_pc_sampling_configuration_s::min_interval
 * the highest possible frequencey for
 * generating samples using @ref method.
 * @var rocprofiler_pc_sampling_configuration_s::max_interval
 * the lowest possible frequency for
 * generating samples using @ref method
 * @var rocprofiler_pc_sampling_configuration_s::flags
 * TODO: ???
 */
struct rocprofiler_pc_sampling_configuration_s
{
    rocprofiler_pc_sampling_method_t method;
    rocprofiler_pc_sampling_unit_t   unit;
    size_t                           min_interval;
    size_t                           max_interval;
    uint64_t                         flags;
};

/**
 * @brief The rocprofiler calls the tool's callback to deliver the list
 * of available configurations upon the calls to the @ref
 * rocprofiler_query_pc_sampling_agent_configurations.
 *
 * @param[out] configs - The list of PC sampling configurations supported by the agent of the
 * moment of invoking @ref rocprofiler_query_pc_sampling_agent_configurations.
 * @param[out] num_config - The number of configuration contained in the underlying
 * In case the GPU agent does not support PC sampling, the value is 0.
 * @param[in] user_data - A pointer passed as the last argument of the
 * @ref rocprofiler_query_pc_sampling_agent_configurations
 */
typedef rocprofiler_status_t (*rocprofiler_available_pc_sampling_configurations_cb_t)(
    const rocprofiler_pc_sampling_configuration_t* configs,
    size_t                                         num_config,
    void*                                          user_data);

/**
 * @brief Query PC Sampling Configuration.
 *
 * @param [in] agent_id  - id of the agent for which available configuration will be listed
 * @param [in] cb        - User callback that delivers the available PC sampling configurations
 * @param [in] user_data - passed to the @p cb
 * @return ::rocprofiler_status_t
 */
rocprofiler_status_t ROCPROFILER_API
rocprofiler_query_pc_sampling_agent_configurations(
    rocprofiler_agent_id_t                                agent_id,
    rocprofiler_available_pc_sampling_configurations_cb_t cb,
    void*                                                 user_data) ROCPROFILER_NONNULL(2, 3);

/**
 * @brief The header of the @ref rocprofiler_pc_sampling_record_s, indicating
 * what fields of the @ref rocprofiler_pc_sampling_record_s instance are meaningful
 * @brief The header of the @ref rocprofiler_pc_sampling_record_s, indicating
 * what fields of the @ref rocprofiler_pc_sampling_record_s instance are meaningful
 * for the sample.
 * @var rocprofiler_pc_sampling_header_v1_t::valid
 * the sample is valid
 * @var rocprofiler_pc_sampling_header_v1_t::type
 * The following values are possible:
 * - 0 - reserved
 * - 1 - host trap pc sample
 * - 2 - stochastic pc sample
 * - 3 - perfcounter (unsupported at the moment)
 * - other values does not mean anything at the moment
 * @var rocprofiler_pc_sampling_header_v1_t::has_stall_reason
 * whether the sample contains
 * information about the stall reason. If so, please @see rocprofiler_pc_sampling_snapshot_v1_t.
 * @var rocprofiler_pc_sampling_header_v1_t::has_wave_cnt
 * whether the @ref rocprofiler_pc_sampling_record_s::wave_count contains
 * meaningful value
 * @var rocprofiler_pc_sampling_header_v1_t::has_memory_counter
 * whether the content of the @ref
 * rocprofiler_pc_sampling_memorycounters_v1_t is meaningful
 */
typedef struct
{
    uint8_t valid : 1;
    uint8_t type  : 4;  // 0=reserved, 1=hosttrap, 2=stochastic, 3=perfcounter, >=4 possible v2?
    uint8_t has_stall_reason   : 1;
    uint8_t has_wave_cnt       : 1;
    uint8_t has_memory_counter : 1;
} rocprofiler_pc_sampling_header_v1_t;

/**
 * @brief TODO: provide the description
 */
typedef struct
{
    uint32_t dual_issue_valu   : 1;
    uint32_t inst_type         : 4;
    uint32_t reason_not_issued : 7;
    uint32_t arb_state_issue   : 10;
    uint32_t arb_state_stall   : 10;
} rocprofiler_pc_sampling_snapshot_v1_t;

/**
 * @brief TODO: provide the description
 */
typedef union
{
    struct
    {
        uint32_t load_cnt   : 6;
        uint32_t store_cnt  : 6;
        uint32_t bvh_cnt    : 3;
        uint32_t sample_cnt : 6;
        uint32_t ds_cnt     : 6;
        uint32_t km_cnt     : 5;
    };
    uint32_t raw;
} rocprofiler_pc_sampling_memorycounters_v1_t;

// TODO: The definition of this structure might change over time
// to reduce the space needed to represent a single sample.
/**
 * @brief ROCProfiler PC Sampling Record corresponding to the interrupted wave.
 * @var rocprofiler_pc_sampling_record_s::flags
 * header that indicates what fields are meaningful
 * for the PC sample. The values depend on what the underlying GPU agent architecture supports.
 * @var rocprofiler_pc_sampling_record_s::chiplet
 * chiplet index
 * @var rocprofiler_pc_sampling_record_s::wave_id
 * wave identifier within the workgroup
 * @var rocprofiler_pc_sampling_record_s::wave_issued
 * a flags indicated whether the wave is
 * issueing the instruction' represented by the @ref pc at the moment of interruption.
 * @var rocprofiler_pc_sampling_record_s::reserved
 * FIXME: reserved 7 bits, must be zero.
 * @var rocprofiler_pc_sampling_record_s::hw_id
 * compute unit identifier
 * @var rocprofiler_pc_sampling_record_s::pc
 * The current program counter of the wave at the moment
 * of interruption
 * @var rocprofiler_pc_sampling_record_s::exec_mask
 * shows how many SIMD lanes of the wave were
 * executing the instruction represented by the @ref pc. Useful to understand thread-divergance
 * within the wave
 * @var rocprofiler_pc_sampling_record_s::workgroup_id_x
 * the x coordinate of the wave within the workgroup
 * @var rocprofiler_pc_sampling_record_s::workgroup_id_y
 * the y coordinate of the wave within the workgroup
 * @var rocprofiler_pc_sampling_record_s::workgroup_id_z
 * the y coordinate of the wave within the workgroup
 * @var rocprofiler_pc_sampling_record_s::wave_count
 * FIXME: number of waves active at the CU at the moment of sample generation???
 * @var rocprofiler_pc_sampling_record_s::timestamp
 * represents the GPU timestamp when the sample is generated
 * @var rocprofiler_pc_sampling_record_s::correlation_id
 * correlation id of the API call that
 * initiated kernel laucnh. The interrupted wave is executed as part of the kernel.
 * @var rocprofiler_pc_sampling_record_s::snapshot
 * TODO:
 * @var rocprofiler_pc_sampling_record_s::memory_counters
 * TODO:
 */
struct rocprofiler_pc_sampling_record_s
{
    rocprofiler_pc_sampling_header_v1_t         flags;
    uint8_t                                     chiplet;
    uint8_t                                     wave_id;
    uint8_t                                     wave_issued : 1;
    uint8_t                                     reserved    : 7;
    uint32_t                                    hw_id;
    uint64_t                                    pc;
    uint64_t                                    exec_mask;
    uint32_t                                    workgroup_id_x;
    uint32_t                                    workgroup_id_y;
    uint32_t                                    workgroup_id_z;
    uint32_t                                    wave_count;
    uint64_t                                    timestamp;
    rocprofiler_correlation_id_t                correlation_id;
    rocprofiler_pc_sampling_snapshot_v1_t       snapshot;
    rocprofiler_pc_sampling_memorycounters_v1_t memory_counters;
};

/** @} */

ROCPROFILER_EXTERN_C_FINI