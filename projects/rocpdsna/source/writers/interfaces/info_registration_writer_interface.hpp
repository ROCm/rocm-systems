// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#pragma once

#include "rocpdsna/writer_types.hpp"
#include "writers/interfaces/api_writer_base.hpp"

namespace rocpdsna
{

template <typename Derived>
class info_registration_writer_interface : public api_writer_base<Derived>
{
public:
    // =========================================================================
    // Core info registration methods (v3+)
    // =========================================================================

    void register_node_info(const writer_types::node_info_t& node_info)
    {
        this->self().register_node_info_impl(node_info);
    }

    void register_process_info(const writer_types::process_info_t& process_info)
    {
        this->self().register_process_info_impl(process_info);
    }

    void register_agent_info(const writer_types::agent_info_t& agent_info)
    {
        this->self().register_agent_info_impl(agent_info);
    }

    void register_pmc_info(const writer_types::pmc_info_t& pmc_info)
    {
        this->self().register_pmc_info_impl(pmc_info);
    }

    void register_thread_info(const writer_types::thread_info_t& thread_info)
    {
        this->self().register_thread_info_impl(thread_info);
    }

    void register_stream_info(const writer_types::stream_info_t& stream_info)
    {
        this->self().register_stream_info_impl(stream_info);
    }

    void register_queue_info(const writer_types::queue_info_t& queue_info)
    {
        this->self().register_queue_info_impl(queue_info);
    }

    void register_code_object_info(
        const writer_types::code_object_info_t& code_object_info)
    {
        this->self().register_code_object_info_impl(code_object_info);
    }

    void register_kernel_symbol_info(
        const writer_types::kernel_symbol_info_t& kernel_symbol_info)
    {
        this->self().register_kernel_symbol_info_impl(kernel_symbol_info);
    }

    void register_track_info(const writer_types::track_info_t& track)
    {
        this->self().register_track_info_impl(track);
    }

    void register_string(std::string_view str) { this->self().register_string_impl(str); }

    // =========================================================================
    // v4+ specific info registration methods
    // =========================================================================

    /**
     * @brief Register a category (v4+ only)
     * @note In v3, categories are stored as strings via register_string()
     */
    void register_category_info(const writer_types::category_info_t& category_info)
    {
        this->self().register_category_info_impl(category_info);
    }

    /**
     * @brief Register an address range info (v4+ only)
     * @note In v3, address ranges are embedded in call_stack/line_info JSONB
     */
    void register_address_range_info(
        const writer_types::address_range_info_t& address_range)
    {
        this->self().register_address_range_info_impl(address_range);
    }

    /**
     * @brief Register source code info (v4+ only)
     * @note In v3, source code is embedded in line_info JSONB column
     */
    void register_source_code_info(const writer_types::source_code_info_t& source_code)
    {
        this->self().register_source_code_info_impl(source_code);
    }

    /**
     * @brief Register program counter info (v4+ only)
     * @note In v3, PC info is embedded in call_stack/line_info JSONB
     */
    void register_pc_info(const writer_types::pc_info_t& pc_info)
    {
        this->self().register_pc_info_impl(pc_info);
    }
};

}  // namespace rocpdsna
