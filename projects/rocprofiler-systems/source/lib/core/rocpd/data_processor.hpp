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

#include <cstdint>
#include <functional>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>

namespace rocprofsys
{
namespace rocpd
{
struct data_processor
{
    using insert_event_stmt =
        std::function<void(const char*, size_t, size_t, size_t, size_t, const char*,
                           const char*, const char*)>;
    using insert_pmc_event_stms =
        std::function<void(const char*, size_t, size_t, double, const char*)>;
    using insert_sample_stmt =
        std::function<void(const char*, size_t, uint64_t, size_t, const char*)>;
    using insert_region_stmt =
        std::function<void(const char*, size_t, size_t, size_t, uint64_t, uint64_t,
                           size_t, size_t, const char*)>;
    using insert_kernel_dispatch_stmt       = std::function<void(
        const char*, size_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
        uint64_t, uint64_t, size_t, size_t, size_t, size_t, size_t, size_t, size_t,
        size_t, size_t, size_t, const char*)>;
    using insert_memory_copy_stmt           = std::function<void(
        const char*, size_t, size_t, size_t, uint64_t, uint64_t, size_t, size_t, size_t,
        size_t, size_t, size_t, size_t, size_t, size_t, size_t, const char*)>;
    using insert_memory_alloc_stmt          = std::function<void(
        const char*, size_t, size_t, size_t, size_t, const char*, const char*, uint64_t,
        uint64_t, size_t, size_t, size_t, size_t, size_t, const char*)>;
    using insert_memory_alloc_no_agent_stmt = std::function<void(
        const char*, size_t, size_t, size_t, const char*, const char*, uint64_t, uint64_t,
        size_t, size_t, size_t, size_t, size_t, const char*)>;
    using insert_kernel_symbol_stmt =
        std::function<void(size_t, const char*, size_t, size_t, uint64_t, const char*,
                           const char*, uint64_t, uint32_t, uint32_t, uint32_t, uint32_t,
                           uint32_t, uint32_t, uint32_t, const char*)>;
    using insert_code_object_stmt =
        std::function<void(size_t, const char*, size_t, size_t, size_t, const char*,
                           uint64_t, uint64_t, uint64_t, const char*, const char*)>;
    using insert_args_stmt = std::function<void(const char*, size_t, size_t, const char*,
                                                const char*, const char*, const char*)>;

private:
    struct track_name_map
    {
        size_t track_id;
        size_t name_id;
    };

    struct pmc_identifier
    {
        size_t      agent_id;
        std::string name;
    };

    struct pmc_identifier_hash
    {
        std::size_t operator()(const pmc_identifier& pmc) const noexcept
        {
            std::size_t h1 = std::hash<size_t>{}(pmc.agent_id);
            std::size_t h2 = std::hash<std::string>{}(pmc.name);
            return h1 ^ (h2 << 1);
        }
    };

    struct pmc_identifier_equal
    {
        bool operator()(const pmc_identifier& lhs,
                        const pmc_identifier& rhs) const noexcept
        {
            return lhs.agent_id == rhs.agent_id && lhs.name == rhs.name;
        }
    };

public:
    static data_processor& get_instance();

    size_t insert_string(const char* str);

    void insert_node_info(size_t node_id, size_t hash, const char* machine_id,
                          const char* system_name, const char* hostname,
                          const char* release, const char* version,
                          const char* hardware_name, const char* domain_name);

    void insert_process_info(size_t node_id, size_t ppid, size_t pid, size_t init,
                             size_t fini, size_t start, size_t end, const char* command,
                             const char* environment = "{}", const char* extdata = "{}");

    size_t insert_agent(size_t node_id, size_t pid, const char* agent_type,
                        size_t absolute_index, size_t logical_index, size_t type_index,
                        uint64_t uuid, const char* name, const char* model_name,
                        const char* vendor_name, const char* product_name,
                        const char* user_name, const char* extdata = "{}");

    void insert_track(const char* track_name, size_t node_id, size_t process_id,
                      std::optional<size_t> thread_id, const char* extdata = "{}");

    size_t insert_event(size_t category_id, size_t stack_id, size_t parent_stack_id,
                        size_t correlation_id, const char* call_stack = "{}",
                        const char* line_info = "{}", const char* extdata = "{}");

    void insert_pmc_event(size_t event_id, size_t agent_id, const char* pmc_descriptor,
                          double value, const char* extdata = "{}");

    void insert_pmc_description(size_t node_id, size_t process_id, size_t agent_id,
                                const char* target_arch, size_t event_code,
                                size_t instance_id, const char* name, const char* symbol,
                                const char* description, const char* long_description,
                                const char* component, const char* units,
                                const char* value_type, const char* block,
                                const char* expression, uint32_t is_constant,
                                uint32_t is_derived, const char* extdata = "{}");

    void insert_sample(const char* track, uint64_t timestamp, size_t event_id,
                       const char* extdata = "{}");

    void insert_category(size_t category_id, const char* name);

    void insert_region(size_t node_id, size_t process_id, size_t thread_id,
                       uint64_t start, uint64_t end, size_t name_id, size_t event_id,
                       const char* extdata = "{}");

    size_t insert_thread_info(size_t node_id, size_t parent_process_id, size_t process_id,
                              size_t thread_id, const char* name, uint64_t start = 0,
                              uint64_t end = 0, const char* extdata = "{}");

    void insert_stream_info(size_t stream_id, size_t node_id, size_t process_id,
                            const char* name, const char* extdata = "{}");
    void insert_queue_info(size_t queue_id, size_t node_id, size_t process_id,
                           const char* name, const char* extdata = "{}");

    void insert_kernel_dispatch(size_t node_id, size_t process_id, size_t thread_id,
                                size_t agent_id, size_t kernel_id, size_t dispatch_id,
                                size_t queue_id, size_t stream_id, uint64_t start,
                                uint64_t end, size_t private_segment_size,
                                size_t group_segment_size, size_t workgroup_size_x,
                                size_t workgroup_size_y, size_t workgroup_size_z,
                                size_t grid_size_x, size_t grid_size_y,
                                size_t grid_size_z, size_t region_name_id,
                                size_t event_id, const char* extdata = "{}");

    void insert_memory_copy(size_t node_id, size_t process_id, size_t thread_id,
                            uint64_t start, uint64_t end, size_t name_id,
                            size_t dst_agent_id, size_t dst_addr, size_t src_agent_id,
                            size_t src_addr, size_t size, size_t queue_id,
                            size_t stream_id, size_t region_name_id, size_t event_id,
                            const char* extdata = "{}");

    void insert_kernel_symbol(size_t id, size_t node_id, size_t process_id,
                              uint64_t code_obj_id, const char* name,
                              const char* display_name, uint32_t kernel_obj,
                              uint32_t kernarg_segmnt_size,
                              uint32_t kernarg_segment_alignment,
                              uint32_t group_segment_size, uint32_t private_segment_size,
                              uint32_t sgrp_count, uint32_t arch_vgrp_count,
                              uint32_t accum_vgrp_count, const char* extdata = "{}");

    void insert_code_object(size_t id, size_t node_id, size_t process_id, size_t agent_id,
                            const char* uri, uint64_t ld_base, uint64_t ld_size,
                            uint64_t ld_delta, const char* storage_type,
                            const char* extdata = "{}");

    void insert_args(size_t event_id, size_t position, const char* type, const char* name,
                     const char* value, const char* extdata = "{}");

    void insert_memory_alloc(size_t node_id, size_t process_id, size_t thread_id,
                             std::optional<size_t> agent_id, const char* type,
                             const char* level, uint64_t start, uint64_t end,
                             size_t address, size_t size, size_t queue_id,
                             size_t stream_id, size_t event_id,
                             const char* extdata = "{}");

    void flush();

private:
    data_processor();
    data_processor(data_processor&)                  = delete;
    data_processor& operator=(const data_processor&) = delete;

    void initialize_pmc_event_stmt();
    void initialize_event_stmt();
    void initialize_sample_stmt();
    void initialize_region_stmt();
    void initialize_kernel_dispatch_stmt();
    void initialize_memory_copy_stmt();
    void initialize_kernel_symbol_stmt();
    void initialize_code_object_stmt();
    void initialize_metadata();
    void initialize_args_stmt();
    void initialize_memory_alloc_stmt();

private:
    std::unordered_map<std::string, track_name_map> _tracks;
    std::unordered_map<pmc_identifier, size_t, pmc_identifier_hash, pmc_identifier_equal>
                                            _pmc_descriptor_map;
    std::unordered_map<size_t, size_t>      _thread_id_map;
    std::unordered_map<size_t, size_t>      _category_map;
    std::unordered_map<std::string, size_t> _string_map;

    std::set<uint64_t> _code_object_ids;
    std::set<uint64_t> _kernel_sym_ids;
    std::set<uint64_t> _stream_ids;
    std::set<uint64_t> _queue_ids;

    insert_event_stmt                 _insert_event_statement;
    insert_pmc_event_stms             _insert_pmc_event_statement;
    insert_sample_stmt                _insert_sample_statement;
    insert_region_stmt                _insert_region_statement;
    insert_kernel_dispatch_stmt       _insert_kernel_dispatch_statement;
    insert_memory_copy_stmt           _insert_memory_copy_statement;
    insert_kernel_symbol_stmt         _insert_kernel_symbol_statement;
    insert_code_object_stmt           _insert_code_object_statement;
    insert_args_stmt                  _insert_args_statement;
    insert_memory_alloc_stmt          _insert_memory_alloc_statement;
    insert_memory_alloc_no_agent_stmt _insert_memory_alloc_no_agent_statement;

    std::string _upid{};

    std::mutex _data_mutex;
};

}  // namespace rocpd
}  // namespace rocprofsys
