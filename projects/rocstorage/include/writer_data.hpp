#pragma once

#include <cstddef>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace rocstorage
{
namespace writer_api
{

using node_id_t              = size_t;
using process_id_t           = size_t;
using thread_id_t            = size_t;
using agent_id_t             = size_t;
using code_obj_id_t          = size_t;
using kernel_symbol_id_t     = size_t;
using pmc_description_name_t = std::string;
using stream_id_t            = size_t;
using queue_id_t             = size_t;
using track_name_t           = std::string;

struct execution_context_t
{
    node_id_t                   node_id;
    std::optional<process_id_t> process_id;
    std::optional<thread_id_t>  thread_id;

    std::optional<agent_id_t>  agent_id;
    std::optional<stream_id_t> stream_id;
    std::optional<queue_id_t>  queue_id;

    std::optional<track_name_t> track_name;
};

// --------------------- Info Tables ---------------------

struct node_info_t
{
    node_id_t   node_id;
    size_t      hash;
    std::string machine_id;
    std::string system_name;
    std::string hostname;
    std::string release;
    std::string version;
    std::string hardware_name;
    std::string domain_name;
};

struct process_info_t
{
    size_t       ppid;
    process_id_t pid;
    size_t       init;
    size_t       fini;
    size_t       start;
    size_t       end;
    std::string  command;
    std::string  environment;
    std::string  extdata;
};

struct agent_unique_id_t
{
    size_t      logical_index;
    std::string agent_type;
};

struct agent_info_t
{
    agent_unique_id_t unique_id;

    size_t      absolute_index;
    size_t      type_index;
    size_t      uuid;
    std::string name;
    std::string model_name;
    std::string vendor_name;
    std::string product_name;
    std::string user_name;
    std::string extdata;
};

struct pmc_info_t
{
    std::string            target_arch;
    size_t                 event_code;
    size_t                 instance_id;
    pmc_description_name_t name;
    std::string            symbol;
    std::string            description;
    std::string            long_description;
    std::string            component;
    std::string            units;
    std::string            value_type;
    std::string            block;
    std::string            expression;
    size_t                 is_constant;
    size_t                 is_derived;
    std::string            extdata;
};

struct thread_info_t
{
    size_t      parent_process_id;
    thread_id_t thread_id;
    std::string name;
    size_t      start;
    size_t      end;
    std::string extdata;
};

struct stream_info_t
{
    stream_id_t stream_id;
    std::string name;
    std::string extdata;
};

struct queue_info_t
{
    queue_id_t  queue_id;
    std::string name;
    std::string extdata;
};

struct code_object_info_t
{
    code_obj_id_t id;
    std::string   uri;
    size_t        ld_base;
    size_t        ld_size;
    size_t        ld_delta;
    std::string   storage_type;
    std::string   extdata;
};

struct kernel_symbol_info_t
{
    kernel_symbol_id_t id;
    std::string        name;
    std::string        display_name;
    size_t             kernel_obj;
    size_t             kernarg_segmnt_size;
    size_t             kernarg_segment_alignment;
    size_t             group_segment_size;
    size_t             private_segment_size;
    size_t             sgrp_count;
    size_t             arch_vgrp_count;
    size_t             accum_vgrp_count;
    std::string        extdata;
};

struct track_info_t
{
    track_name_t name;
    std::string  extdata;
};

// --------------------- Call Stack & Line Info Abstract Data Types ------------------

struct address_range_info_t
{
    size_t      address_base;
    size_t      address_low;
    size_t      address_high;
    std::string extdata;
};

struct program_counter_info_t
{
    std::string function;
    std::string filename;
    size_t      line_number;
    std::string extdata;
};

struct stack_frame_t
{
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;
    std::string                           extdata;
};

// implicit depth, zero for the top frame etc.
using call_stack_t = std::deque<stack_frame_t>;

struct source_code_info_t
{
    std::optional<std::string> filename;
    std::optional<size_t>      starting_line_number;
    std::vector<std::string>   source_code_lines;
    std::vector<std::string>   assembly_instruction_lines;
    std::string                extdata;
};

struct line_info_entry_t
{
    std::optional<source_code_info_t>     source_code;
    std::optional<program_counter_info_t> program_counter;
    std::optional<address_range_info_t>   address_range;  // connect to program_counter
};

using source_context_list_t = std::vector<line_info_entry_t>;

// --------------------- Data Tables ---------------------

struct arg_data_t
{
    size_t      position;
    std::string type;
    std::string name;
    std::string value;
    std::string extdata;
};

struct event_data_t
{
    size_t stack_id;
    size_t parent_stack_id;
    size_t correlation_id;

    // v3: Serialize to JSON
    // v4: Put inside tables
    call_stack_t                   call_stack;
    std::vector<line_info_entry_t> line_info_list;

    // v3: table rocpd_string
    // v4: table rocpd_category
    std::string event_category;

    std::string extdata;
};

struct region_data_t
{
    event_data_t event;

    size_t      start_timestamp;
    size_t      end_timestamp;
    std::string name;
    std::string extdata;

    std::vector<arg_data_t> args;
};

struct sample_data_t
{
    size_t timestamp;

    std::string extdata;
};

struct pmc_event_data_t
{
    event_data_t event;
    double       value;
    std::string  extdata;

    sample_data_t sample;
};

struct kernel_dispatch_data_t
{
    event_data_t event;
    size_t       dispatch_id;
    size_t       start_timestamp;
    size_t       end_timestamp;
    size_t       private_segment_size;
    size_t       group_segment_size;
    size_t       workgroup_size_x;
    size_t       workgroup_size_y;
    size_t       workgroup_size_z;
    size_t       grid_size_x;
    size_t       grid_size_y;
    size_t       grid_size_z;
    std::string  name;
    std::string  extdata;
};

struct memory_copy_data_t
{
    event_data_t event;
    size_t       start_timestamp;
    size_t       end_timestamp;
    size_t       dst_address;
    size_t       src_address;
    size_t       size;
    std::string  name;
    std::string  extdata;
};

struct memory_alloc_data_t
{
    event_data_t event;
    std::string  type;
    std::string  level;
    size_t       start_timestamp;
    size_t       end_timestamp;
    size_t       address;
    size_t       size;
    std::string  extdata;
};

}  // namespace writer_api
}  // namespace rocstorage
