// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#pragma once

#include "writer_data.hpp"
#include <memory>
#include <optional>

namespace rocm
{
class storage;
}

namespace rocstorage
{
namespace data_storage
{
class database;
}

struct writer
{
    friend class rocm::storage;

private:
    explicit writer(std::shared_ptr<data_storage::database> database, std::string uuid);

public:
    virtual ~writer();

    writer()                          = delete;
    writer(const writer&)             = delete;
    writer& operator=(const writer&)  = delete;
    writer(const writer&&)            = delete;
    writer& operator=(const writer&&) = delete;

    // --------------------- Info Tables ---------------------

    /***
     * @brief Insert node info into rocpd
     * @param node_info Node info which will be inserted into rocpd
     */
    void register_node_info(const writer_api::node_info_t& node_info);

    /***
     * @brief Insert process info into rocpd
     * @param process_info Process info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     */
    void register_process_info(const writer_api::process_info_t& process_info,
                               const writer_api::node_id_t       node_id_value);

    /***
     * @brief Insert agent info into rocpd
     * @param agent Agent info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     */
    void register_agent_info(const writer_api::agent_info_t& agent,
                             const writer_api::node_id_t     node_id_value,
                             const writer_api::process_id_t  process_id_value);

    /***
     * @brief Insert pmc info into rocpd
     * @param pmc_info Pmc info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param agent_id_value Agent Id Value (agent type and logical index) - which will
     * uniquely identify the agent
     */
    void register_pmc_info(const writer_api::pmc_info_t&        pmc_info,
                           const writer_api::node_id_t          node_id_value,
                           const writer_api::process_id_t       process_id_value,
                           const writer_api::agent_unique_id_t& agent_id_value);

    /***
     * @brief Insert thread info into rocpd
     * @param thread_info Thread info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     */
    void register_thread_info(const writer_api::thread_info_t& thread_info,
                              const writer_api::node_id_t      node_id_value,
                              const writer_api::process_id_t   process_id_value);

    /***
     * @brief Insert stream info into rocpd
     * @param stream_info Stream info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     */
    void register_stream_info(const writer_api::stream_info_t& stream_info,
                              const writer_api::node_id_t      node_id_value,
                              const writer_api::process_id_t   process_id_value);

    /***
     * @brief Insert queue info into rocpd
     * @param queue_info Queue info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     */
    void register_queue_info(const writer_api::queue_info_t& queue_info,
                             const writer_api::node_id_t     node_id_value,
                             const writer_api::process_id_t  process_id_value);

    /***
     * @brief Insert code object info into rocpd
     * @param code_object Code object which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param agent_id_value Agent Id Value (agent type and logical index) - which will
     * uniquely identify the agent
     */
    void register_code_object_info(const writer_api::code_object_info_t& code_object,
                                   const writer_api::node_id_t           node_id_value,
                                   const writer_api::process_id_t        process_id_value,
                                   const writer_api::agent_unique_id_t&  agent_id_value);

    /***
     * @brief Insert kernel symbol info into rocpd
     * @param kernel_symbol Kernel symbol which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param code_obj_id_value Code Object Id Value - which will uniquely identify the
     * code object
     */
    void register_kernel_symbol_info(
        const writer_api::kernel_symbol_info_t& kernel_symbol,
        const writer_api::node_id_t             node_id_value,
        const writer_api::process_id_t          process_id_value,
        const writer_api::code_obj_id_t         code_obj_id_value);

    /***
     * @brief Insert track info into rocpd
     * @param track Track info which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     */
    void register_track_info(
        const writer_api::track_info_t& track, const writer_api::node_id_t node_id_value,
        const writer_api::process_id_t               process_id_value,
        const std::optional<writer_api::thread_id_t> thread_id_value);

    /***
     * @brief Insert string into rocpd
     * @param str String which will be inserted into rocpd
     */
    void register_string(const std::string& str);

    // --------------------- Data Tables ---------------------

    /***
     * @brief Insert region data into rocpd and create sample which will reference the
     * track
     * @param region_data Region data which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     * @param track_name_value Track Name Value - which will uniquely identify the track
     */
    void insert_region_data(const writer_api::region_data_t& region_data,
                            const writer_api::node_id_t      node_id_value,
                            const writer_api::process_id_t   process_id_value,
                            const writer_api::thread_id_t    thread_id_value,
                            // new
                            const writer_api::track_name_t& track_name_value);

    /***
     * @brief Insert pmc event data into rocpd
     * @param pmc_event_data Pmc event data which will be inserted into rocpd
     * @param pmc_name_value Pmc Name Value - which will uniquely identify the pmc
     */
    void insert_pmc_event_data(const writer_api::pmc_event_data_t&       pmc_event_data,
                               const writer_api::pmc_description_name_t& pmc_name_value);

    /***
     * @brief Insert kernel dispatch data into rocpd
     * @param kernel_dispatch_data Kernel dispatch data which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     * @param agent_id_value Agent Id Value (agent type and logical index) - which will
     * uniquely identify the agent
     */
    void insert_kernel_dispatch_data(
        const writer_api::kernel_dispatch_data_t& kernel_dispatch_data,
        const writer_api::node_id_t               node_id_value,
        const writer_api::process_id_t            process_id_value,
        const writer_api::thread_id_t             thread_id_value,
        const writer_api::agent_unique_id_t&      agent_id_value,
        const writer_api::kernel_symbol_id_t      kernel_symbol_id_value,
        const writer_api::stream_id_t             stream_id_value,
        const writer_api::queue_id_t              queue_id_value,
        // new
        const writer_api::track_name_t& track_name_value);

    /***
     * @brief Insert memory copy data into rocpd
     * @param memory_copy_data Memory copy data which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     * @param src_agent_id_value Source Agent Id Value (agent type and logical index) -
     * which will uniquely identify the source agent
     */
    void insert_memory_copy_data(const writer_api::memory_copy_data_t& memory_copy_data,
                                 const writer_api::node_id_t           node_id_value,
                                 const writer_api::process_id_t        process_id_value,
                                 const writer_api::thread_id_t         thread_id_value,
                                 const writer_api::agent_unique_id_t&  src_agent_id_value,
                                 const writer_api::agent_unique_id_t&  dst_agent_id_value,
                                 const writer_api::stream_id_t         stream_id_value,
                                 const writer_api::queue_id_t          queue_id_value,
                                 // new
                                 const writer_api::track_name_t& track_name_value);

    /***
     * @brief Insert memory alloc data into rocpd
     * @param memory_alloc_data Memory alloc data which will be inserted into rocpd
     * @param node_id_value Node Id Value - which will uniquely identify the node
     * @param process_id_value Process Id Value - which will uniquely identify the process
     * @param thread_id_value Thread Id Value - which will uniquely identify the thread
     * @param queue_id_value Queue Id Value - which will uniquely identify the queue
     * @param stream_id_value Stream Id Value - which will uniquely identify the stream
     * @param agent_id_value Agent Id Value (agent type and logical index) - which will
     * uniquely identify the agent
     */
    void insert_memory_alloc_data(
        const writer_api::memory_alloc_data_t&              memory_alloc_data,
        const writer_api::node_id_t                         node_id_value,
        const writer_api::process_id_t                      process_id_value,
        const writer_api::thread_id_t                       thread_id_value,
        const writer_api::queue_id_t                        queue_id_value,
        const writer_api::stream_id_t                       stream_id_value,
        const std::optional<writer_api::agent_unique_id_t>& agent_id_value,
        // new
        const writer_api::track_name_t& track_name_value);

    /***
     * @brief Flush in-memory data to disk
     * @note This function is only used with in-memory database option
     */
    void flush_in_memory_data_to_disk();

private:
    struct impl;
    std::unique_ptr<impl> m_impl;
};

}  // namespace rocstorage
