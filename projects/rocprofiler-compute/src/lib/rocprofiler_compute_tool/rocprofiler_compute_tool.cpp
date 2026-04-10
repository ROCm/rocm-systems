// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "sdk_callbacks.h"
#include "input_parameters.h"
#include "rocprofiler_compute_tool.h"
#include "sdk_wrapper.h"

#include <unistd.h>

#include <fstream>
#include <iostream>

using namespace rocprof_compute_tool;

std::shared_ptr<InputParameters> g_input_parameters = std::make_shared<EnvInputParameters>();

void test_knobs::set_input_parameters(std::shared_ptr<InputParameters> input_parameters)
{
    g_input_parameters = input_parameters;
}

namespace
{

using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;

rocprofiler_context_id_t& get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

iteration_multiplexing_mode_t iteration_multiplexing_mode(const std::string& mode)
{
    if (mode == "kernel")
        return iteration_multiplexing_mode_t::KERNEL;
    else if (mode == "kernel_launch_params")
        return iteration_multiplexing_mode_t::LAUNCH;
    else
        return iteration_multiplexing_mode_t::DISABLED;
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t*                record_data,
                     size_t                                       record_count,
                     rocprofiler_user_data_t /* user_data */,
                     void* callback_data_args)
{
    auto*        tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data_args);
    tool_data_t* tool;
    {
        std::lock_guard<std::mutex> lock(tool_data_ptr->get()->mut);
        tool = tool_data_ptr->get();
    }

    // For each counter, write: dispatch_id, counter_id, counter_name,
    // counter_value
    for (size_t i = 0; i < record_count; ++i)
    {
        rocprofiler_counter_id_t counter_id{};
        ROCPROFILER_CALL(rocprofiler_query_record_counter_id(record_data[i].id, &counter_id),
                         "query record counter id");

        // Store the counter info record in tool_data
        counter_info_record_t record{dispatch_data.dispatch_info.dispatch_id,
                                     dispatch_data.dispatch_info.agent_id.handle,
                                     dispatch_data.dispatch_info.kernel_id,
                                     dispatch_data.dispatch_info.group_segment_size,
                                     counter_id.handle,
                                     tool->counter_id_name_map[counter_id.handle],
                                     record_data[i].counter_value};
        {
            std::lock_guard<std::mutex> lock(tool->mut);
            tool->counter_records.push_back(std::move(record));
        }
    }
}

/**
 * Callback from rocprofiler when a code object is loaded.
 * We use this to get record kernel names as they are registered.
 */
void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* /*user_data*/,
                           void* callback_data)
{
    if (record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD &&
        record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
        record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* data            = static_cast<kernel_symbol_data_t*>(record.payload);
        int   demangle_status = 0;
        auto  kernel_name     = helper_utils::cxa_demangle(data->kernel_name, &demangle_status);
        kernel_name           = helper_utils::truncate_name(kernel_name);

        // check if regex can be found in kernel name matches regex from tool data,
        // if matches store kernel id
        auto* tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(callback_data);
        auto* tool          = tool_data_ptr->get();
        // Lock before modifying target_kernel_ids
        std::lock_guard<std::mutex> lock(tool->mut);
        if (!tool->kernel_filter_include_regex.empty())
        {
            try
            {
                std::regex re(tool->kernel_filter_include_regex);
                if (!kernel_name.empty() && std::regex_search(kernel_name, re))
                {
                    tool->target_kernel_ids.insert(data->kernel_id);
                }
            }
            catch (const std::regex_error& e)
            {
                std::cerr << "[rocprofiler-compute] [" << __FUNCTION__
                          << "] ERROR: Invalid regex in ROCPROF_KERNEL_FILTER_INCLUDE_REGEX: "
                          << tool->kernel_filter_include_regex << " : " << e.what() << std::endl;
            }
        }
        // If no regex specified, collect for all kernels
        else
        {
            tool->target_kernel_ids.insert(data->kernel_id);
        }
    }
}


int tool_init(rocprofiler_client_finalize_t, void* user_data)
{
    std::clog << "[rocprofiler-compute] In tool init\n";
    ROCPROFILER_CALL(rocprofiler_create_context(&get_client_ctx()), "context creation");

    ROCPROFILER_CALL(rocprofiler_configure_callback_dispatch_counting_service(get_client_ctx(),
                                                                              dispatch_callback,
                                                                              user_data,
                                                                              record_callback,
                                                                              user_data),
                     "setup counting service");
    ROCPROFILER_CALL(rocprofiler_configure_callback_tracing_service(get_client_ctx(),
                                                                    ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                                    nullptr,
                                                                    0,
                                                                    tool_tracing_callback,
                                                                    user_data),
                     "setup code object tracing service");
    ROCPROFILER_CALL(rocprofiler_start_context(get_client_ctx()), "start context");

    return 0;
}

void generate_output(tool_data_t* tool_data)
{
    // Dispatches before the kernel to be filtered was registered may have been
    // profiled. Remove any records whose kernel id does not match the
    // target_kernel_ids
    if (!tool_data->target_kernel_ids.empty())
    {
        tool_data->counter_records.erase(std::remove_if(tool_data->counter_records.begin(),
                                                        tool_data->counter_records.end(),
                                                        [tool_data](const counter_info_record_t& record)
                                                        {
                                                            return tool_data->target_kernel_ids.find(
                                                                       record.kernel_id) ==
                                                                   tool_data->target_kernel_ids.end();
                                                        }),
                                         tool_data->counter_records.end());
    }
    if (tool_data->counter_records.empty())
    {
        return;
    }
    // Write collected counter records and clean up
    if (!tool_data->output_filename.empty())
    {
        std::ofstream ofs(tool_data->output_filename);
        if (!ofs.is_open())
        {
            std::cerr << "Failed to open output file: " << tool_data->output_filename << std::endl;
            return;
        }
        // Write header at the beginning of the file
        ofs << "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,"
               "counter_id,counter_name,counter_value\n";
        for (const auto& r : tool_data->counter_records)
            ofs << r.dispatch_id << ',' << r.agent_id << "," << r.kernel_id << ',' << r.LDS_memory_size
                << ',' << r.counter_id << ',' << r.counter_name << ',' << r.counter_value << '\n';
        ofs.flush();
        std::clog << "[rocprofiler-compute] [" << __FUNCTION__
                  << "] Counter collection data has been written to: " << tool_data->output_filename
                  << std::endl;
    }
}

void tool_fini(void* user_data)
{
    assert(user_data);
    std::clog << "[rocprofiler-compute] In tool fini\n";
    rocprofiler_stop_context(get_client_ctx());

    auto* tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(user_data);
    generate_output(tool_data_ptr->get());

    delete tool_data_ptr;
}

}  // namespace

std::unique_ptr<tool_data_t> create_tool_data(rocprofiler_client_id_t* /*id*/)
{
    auto tool_data = std::make_unique<tool_data_t>();

    // Generate a unique output filename using the process ID
    std::string base_filename = std::to_string(getpid()) + "_native_counter_collection.csv";

    // Require ROCPROF_OUTPUT_PATH to be set, otherwise error out
    std::string filename;
    const char* output_path = g_input_parameters->get_output_path();
    if (!output_path || !*output_path)
    {
        throw std::runtime_error("ROCPROF_OUTPUT_PATH environment variable must be set");
    }
    filename = output_path;
    if (filename.back() != '/')
        filename += '/';
    // Use the generated base filename along with ROCPROF_OUTPUT_PATH
    filename += base_filename;

    tool_data->output_filename = filename;

    // Store ROCPROF env. vars. in tool_data

    // ROCPROF_COUNTERS env. var. is a string like "pmc: counter1 counter2 ..."
    if (const char* v = g_input_parameters->get_requested_counters())
        tool_data->requested_counters = v;

    if (const char* v = g_input_parameters->get_iteration_multiplexing_mode())
        tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode(v);

    // ROCPROF_KERNEL_FILTER_INCLUDE_REGEX env. var. is a regex string like
    // kernel_name_1|kernel_name_2|... Used to collect counters only for kernels
    // with names matching the regex
    if (const char* v = g_input_parameters->get_kernel_filter_include_regex())
        tool_data->kernel_filter_include_regex = v;

    // ROCPROF_KERNEL_FILTER_RANGE env. var. is a string like "[4,7-9,...]"
    if (const char* v = g_input_parameters->get_kernel_filter_range())
    {
        // Remove square brackets at the ends if present
        std::string v_str = v;
        if (!v_str.empty() && v_str.front() == '[')
            v_str.erase(0, 1);
        if (!v_str.empty() && v_str.back() == ']')
            v_str.pop_back();
        v = v_str.c_str();
        // Parse the range string into vector of pairs
        std::istringstream ss(v);
        for (std::string token; std::getline(ss, token, ',');)
        {
            size_t dash_pos = token.find('-');
            try
            {
                if (dash_pos == std::string::npos)
                {
                    // single number
                    uint64_t num = std::stoull(token);
                    tool_data->kernel_filter_ranges.emplace_back(num, num);
                }
                else
                {
                    // range of numbers
                    uint64_t start = std::stoull(token.substr(0, dash_pos));
                    uint64_t end   = std::stoull(token.substr(dash_pos + 1));
                    tool_data->kernel_filter_ranges.emplace_back(start, end);
                }
            }
            catch (const std::invalid_argument&)
            {
                std::cerr << "[rocprofiler-compute] [" << __FUNCTION__
                          << "] ERROR: Invalid entry in ROCPROF_KERNEL_FILTER_RANGE: " << token
                          << std::endl;
            }
        }
    }

    return tool_data;
}

rocprofiler_tool_configure_result_t* rocprofiler_configure(uint32_t                 version,
                                                           const char*              runtime_version,
                                                           uint32_t                 priority,
                                                           rocprofiler_client_id_t* id)
{
    // set the client name
    id->name = "[rocprofiler-compute]";

    // compute major/minor/patch version info
    uint32_t major = version / 10000;
    uint32_t minor = (version % 10000) / 100;
    uint32_t patch = version % 100;

    // generate info string
    auto info = std::stringstream{};
    info << id->name << " [" << __FUNCTION__ << "] (priority=" << priority
         << ") is using rocprofiler-sdk v" << major << "." << minor << "." << patch << " ("
         << runtime_version << ")";

    std::clog << info.str() << std::endl;

    // init tool data
    auto tool_data = create_tool_data(id);

    // create configure data
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t),
        &tool_init,
        &tool_fini,
        static_cast<void*>(new std::unique_ptr<tool_data_t>(std::move(tool_data)))};

    // return pointer to configure data
    return &cfg;
}
