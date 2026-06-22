// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

#include <profiler-hub/storage.hpp>
#include <profiler-hub/writer_types.hpp>

#include <memory>

namespace profiler_hub
{

struct writer_t
{
    explicit writer_t(std::unique_ptr<profiler_hub::storage_t> storage);
    ~writer_t();
    writer_t()                           = delete;
    writer_t(const writer_t&)            = delete;
    writer_t& operator=(const writer_t&) = delete;
    writer_t(writer_t&&)                 = delete;
    writer_t& operator=(writer_t&&)      = delete;

    /**
     * @brief Insert node info into rocpd
     * @param node_info Node info which will be inserted into rocpd
     */
    void register_node_info(const writer_types::node_info_t& node_info);

    /**
     * @brief Insert process info into rocpd
     * @param process_info Process info which will be inserted into rocpd
     */
    void register_process_info(const writer_types::process_info_t& process_info);

    /**
     * @brief Insert agent info into rocpd
     * @param agent Agent info which will be inserted into rocpd
     */
    void register_agent_info(const writer_types::agent_info_t& agent);

    /**
     * @brief Insert pmc info into rocpd
     * @param pmc_info Pmc info which will be inserted into rocpd
     */
    void register_pmc_info(const writer_types::pmc_info_t& pmc_info);

    /**
     * @brief Insert thread info into rocpd
     * @param thread_info Thread info which will be inserted into rocpd
     */
    void register_thread_info(const writer_types::thread_info_t& thread_info);

    /**
     * @brief Insert stream info into rocpd
     * @param stream_info Stream info which will be inserted into rocpd
     */
    void register_stream_info(const writer_types::stream_info_t& stream_info);

    /**
     * @brief Insert queue info into rocpd
     * @param queue_info Queue info which will be inserted into rocpd
     */
    void register_queue_info(const writer_types::queue_info_t& queue_info);

    /**
     * @brief Insert code object info into rocpd
     * @param code_object Code object which will be inserted into rocpd
     */
    void register_code_object_info(const writer_types::code_object_info_t& code_object);

    /**
     * @brief Insert kernel symbol info into rocpd
     * @param kernel_symbol Kernel symbol which will be inserted into rocpd
     */
    void register_kernel_symbol_info(
        const writer_types::kernel_symbol_info_t& kernel_symbol);

    /**
     * @brief Insert track info into rocpd
     * @param track Track info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     */
    void register_track_info(const writer_types::track_info_t& track);

    /**
     * @brief Insert string into rocpd
     * @param str String which will be inserted into rocpd
     */
    void register_string(std::string_view str);

    /**
     * @brief Insert region data into rocpd and create sample which will reference the
     * track
     * @param region_data Region data which will be inserted into rocpd
     * @param trace_environment Trace environment which is connected to the region
     */
    void insert_region_data(const writer_types::region_data_t&       region_data,
                            const writer_types::trace_environment_t& trace_environment);

    /**
     * @brief Insert pmc event data into rocpd
     * @param pmc_event_data Pmc event data which will be inserted into rocpd
     * @param pmc_unique_id Pmc unique id which will uniquely identify the pmc
     */
    void insert_pmc_event_data(const writer_types::pmc_event_data_t&     pmc_event_data,
                               const writer_types::pmc_info_unique_id_t& pmc_unique_id);

    /**
     * @brief Insert kernel dispatch data into rocpd
     * @param kernel_dispatch_data Kernel dispatch data which will be inserted into rocpd
     * @param trace_environment Trace environment which is connected to the kernel
     * dispatch
     */
    void insert_kernel_dispatch_data(
        const writer_types::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_types::trace_environment_t&    trace_environment);

    /**
     * @brief Insert memory copy data into rocpd
     * @param memory_copy_data Memory copy data which will be inserted into rocpd
     * @param trace_environment Trace environment which is connected to the memory copy
     */
    void insert_memory_copy_data(
        const writer_types::memory_copy_data_t&  memory_copy_data,
        const writer_types::trace_environment_t& trace_environment);

    /**
     * @brief Insert memory alloc data into rocpd
     * @param memory_alloc_data Memory alloc data which will be inserted into rocpd
     * @param trace_environment Trace environment which is connected to the memory alloc
     */
    void insert_memory_alloc_data(
        const writer_types::memory_alloc_data_t& memory_alloc_data,
        const writer_types::trace_environment_t& trace_environment);

    /**
     * @brief Flush in-memory data to disk
     * @note This function is only used with in-memory database option
     */
    void flush_in_memory_data_to_disk();

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace profiler_hub
