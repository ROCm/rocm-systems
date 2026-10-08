// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#pragma once

// Standalone copies of the profiler-hub record types so the writer mock does not depend
// on profiler-hub headers. Layout and field names must stay in sync with
// profiler-hub's shared_types.hpp / writer_types.hpp.

#include <cstddef>
#include <deque>
#include <optional>
#include <string_view>
#include <vector>

namespace rocprofsys::trace_cache::test::shared_types
{

using timestamp_ns_t = size_t;

// --------------------- Call Stack & Line Info Abstract Data Types ------------------

struct address_range_info_t
{
    size_t           address_base;
    size_t           address_low;
    size_t           address_high;
    std::string_view extdata = "{}";
};

struct program_counter_info_t
{
    std::optional<std::string_view> function;
    std::optional<std::string_view> filename;
    std::optional<size_t>           line_number;
    std::string_view                extdata = "{}";
};

struct stack_frame_t
{
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;
    std::string_view                      extdata = "{}";
};

using call_stack_t = std::deque<stack_frame_t>;

struct source_code_info_t
{
    std::optional<std::string_view> filename;
    std::optional<size_t>           starting_line_number;
    std::vector<std::string_view>   source_code_lines;
    std::vector<std::string_view>   assembly_instruction_lines;
    std::string_view                extdata = "{}";
};

struct line_info_entry_t
{
    std::optional<source_code_info_t>     source_code;
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;
};

using source_context_list_t = std::vector<line_info_entry_t>;

}  // namespace rocprofsys::trace_cache::test::shared_types

namespace rocprofsys::trace_cache::test::writer_types
{

using node_id_t = size_t;

using process_id_t = size_t;

using thread_id_t = size_t;

using code_object_id_t = size_t;

using kernel_symbol_id_t = size_t;

using pmc_description_name_t = std::string_view;

using stream_id_t = size_t;

using queue_id_t = size_t;

using track_name_t = std::string_view;

using timestamp_ns_t = size_t;

constexpr std::string_view empty_json = "{}";

struct agent_unique_id_t
{
    std::optional<std::string_view> agent_type;
    size_t                          type_index;

    bool operator==(const agent_unique_id_t& other) const noexcept
    {
        return agent_type == other.agent_type && type_index == other.type_index;
    }
};

struct trace_environment_t
{
    std::optional<node_id_t>    node_id;
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    std::optional<agent_unique_id_t> agent_id;
    std::optional<stream_id_t>       stream_id;
    std::optional<queue_id_t>        queue_id;

    std::optional<track_name_t> track_name;
};

// --------------------- Info Tables ---------------------

struct node_info_t
{
    node_id_t        node_id;
    size_t           hash;
    std::string_view machine_id;

    std::optional<std::string_view> system_name;
    std::optional<std::string_view> hostname;
    std::optional<std::string_view> release;
    std::optional<std::string_view> version;
    std::optional<std::string_view> hardware_name;
    std::optional<std::string_view> domain_name;
};

struct process_info_t
{
    size_t       ppid{};
    process_id_t pid{};
    size_t       init{};
    size_t       fini{};
    size_t       start{};
    size_t       end{};

    std::optional<std::string_view> command;
    std::string_view                environment = empty_json;
    std::string_view                extdata     = empty_json;

    node_id_t node_id{};
};

struct agent_info_t
{
    agent_unique_id_t unique_id{};

    size_t absolute_index{};
    size_t logical_index{};
    size_t uuid{};

    std::optional<std::string_view> name;
    std::optional<std::string_view> model_name;
    std::optional<std::string_view> vendor_name;
    std::optional<std::string_view> product_name;
    std::optional<std::string_view> user_name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct pmc_info_unique_id_t
{
    pmc_description_name_t           name;
    std::optional<agent_unique_id_t> agent_id;

    bool operator==(const pmc_info_unique_id_t& other) const noexcept
    {
        const bool are_names_same = name == other.name;
        if(agent_id.has_value() && other.agent_id.has_value())
        {
            return are_names_same && (agent_id.value() == other.agent_id.value());
        }
        return are_names_same;
    }
};

struct pmc_info_t
{
    pmc_info_unique_id_t unique_id;

    std::optional<std::string_view> target_arch;
    size_t                          event_code{};
    size_t                          instance_id{};
    std::string_view                symbol;
    std::optional<std::string_view> description;
    std::optional<std::string_view> long_description;
    std::optional<std::string_view> component;
    std::optional<std::string_view> units;
    std::optional<std::string_view> value_type;
    std::optional<std::string_view> block;
    std::optional<std::string_view> expression;
    size_t                          is_constant{};
    size_t                          is_derived{};
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct thread_info_t
{
    size_t      parent_process_id{};
    thread_id_t thread_id{};

    std::optional<std::string_view> name;
    size_t                          start{};
    size_t                          end{};
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct stream_info_t
{
    stream_id_t stream_id{};

    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct queue_info_t
{
    queue_id_t queue_id{};

    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;

    node_id_t    node_id{};
    process_id_t process_id{};
};

struct code_object_info_t
{
    code_object_id_t id{};

    std::optional<std::string_view> uri;
    size_t                          load_base{};
    size_t                          load_size{};
    size_t                          load_delta{};
    std::optional<std::string_view> storage_type;
    std::string_view                extdata = empty_json;

    node_id_t                        node_id{};
    process_id_t                     process_id{};
    std::optional<agent_unique_id_t> agent_id;
};

struct kernel_symbol_info_t
{
    kernel_symbol_id_t id{};

    std::optional<std::string_view> name;
    std::optional<std::string_view> display_name;
    size_t                          kernel_object{};
    size_t                          kernarg_segment_size{};
    size_t                          kernarg_segment_alignment{};
    size_t                          group_segment_size{};
    size_t                          private_segment_size{};
    size_t                          sgpr_count{};
    size_t                          arch_vgpr_count{};
    size_t                          accum_vgpr_count{};
    std::string_view                extdata = empty_json;

    node_id_t        node_id{};
    process_id_t     process_id{};
    code_object_id_t code_obj_id{};
};

struct track_info_t
{
    std::optional<track_name_t> name;
    std::string_view            extdata = empty_json;

    node_id_t                   node_id{};
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    bool operator==(const track_info_t& other) const noexcept
    {
        return name == other.name && node_id == other.node_id &&
               process_id == other.process_id && thread_id == other.thread_id;
    }
};

// --------------------- Data Tables ---------------------

struct arg_data_t
{
    size_t           position{};
    std::string_view type;
    std::string_view name;

    std::optional<std::string_view> value;
    std::string_view                extdata = empty_json;
};

struct event_data_t
{
    std::optional<size_t> stack_id;
    std::optional<size_t> parent_stack_id;
    std::optional<size_t> correlation_id;

    shared_types::call_stack_t          call_stack;
    shared_types::source_context_list_t line_info_list;

    std::optional<std::string_view> event_category;
    std::string_view                extdata = empty_json;
};

struct region_data_t
{
    std::optional<event_data_t> event;

    timestamp_ns_t   start_timestamp;
    timestamp_ns_t   end_timestamp;
    std::string_view name;
    std::string_view extdata = empty_json;

    std::vector<arg_data_t> args;
};

struct sample_data_t
{
    timestamp_ns_t   timestamp{};
    track_info_t     track;
    std::string_view extdata = empty_json;
};

struct pmc_event_data_t
{
    std::optional<event_data_t> event;
    double                      value{};
    std::string_view            extdata = empty_json;
    sample_data_t               sample;
};

struct kernel_dispatch_data_t
{
    std::optional<event_data_t> event;
    size_t                      dispatch_id{};
    timestamp_ns_t              start_timestamp{};
    timestamp_ns_t              end_timestamp{};
    kernel_symbol_id_t          kernel_symbol_id{};
    code_object_id_t            code_object_id{};
    size_t                      private_segment_size{};
    size_t                      group_segment_size{};
    size_t                      workgroup_size_x{};
    size_t                      workgroup_size_y{};
    size_t                      workgroup_size_z{};
    size_t                      grid_size_x{};
    size_t                      grid_size_y{};
    size_t                      grid_size_z{};

    std::optional<std::string_view> name;
    std::string_view                extdata = empty_json;
};

struct memory_copy_data_t
{
    std::optional<event_data_t>      event;
    timestamp_ns_t                   start_timestamp{};
    timestamp_ns_t                   end_timestamp{};
    std::optional<agent_unique_id_t> dst_agent_id;
    std::optional<size_t>            dst_address;
    std::optional<agent_unique_id_t> src_agent_id;
    std::optional<size_t>            src_address;
    size_t                           size{};

    std::string_view                name;
    std::optional<std::string_view> region_name;
    std::string_view                extdata = empty_json;
};

struct memory_alloc_data_t
{
    std::optional<event_data_t> event;

    std::optional<std::string_view> type;
    std::optional<std::string_view> level;
    timestamp_ns_t                  start_timestamp{};
    timestamp_ns_t                  end_timestamp{};
    std::optional<size_t>           address;
    size_t                          size{};
    std::string_view                extdata = empty_json;
};

}  // namespace rocprofsys::trace_cache::test::writer_types
