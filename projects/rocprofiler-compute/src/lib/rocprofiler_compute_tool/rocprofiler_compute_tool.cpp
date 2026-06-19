// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprofiler_compute_tool.h"

#include "counters_writer.h"
#include "input_parameters.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"

#include <unistd.h>

#include <atomic>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string_view>

using namespace rocprofiler_compute_tool;

// Heap-allocated and never destroyed on purpose: rocprofiler-sdk calls our
// callbacks and tool_fini from its own _dl_fini, after this library's static
// destructors would have freed them. Process lifetime avoids a teardown
// use-after-destruction.
static std::shared_ptr<InputParameters>& g_input_parameters = *new std::shared_ptr<InputParameters>(
    std::make_shared<EnvInputParameters>());
static std::shared_ptr<SdkWrapper>& g_sdk_wrapper = *new std::shared_ptr<SdkWrapper>(
    std::make_shared<SdkWrapperImpl>());
static std::shared_ptr<SdkCallbacks>& g_sdk_callbacks = *new std::shared_ptr<SdkCallbacks>(
    std::make_shared<SdkCallbacksImpl>(g_sdk_wrapper));
static std::shared_ptr<CountersWriter>& g_counters_writer = *new std::shared_ptr<CountersWriter>(
    std::make_shared<CsvCountersWriter>());
static std::shared_ptr<rocprofiler_tool_configure_result_t>& g_cfg =
    *new std::shared_ptr<rocprofiler_tool_configure_result_t>();
static std::atomic<bool> g_tool_shutting_down{false};
static std::atomic<bool> g_hsa_intercept_done{false};

void test_knobs::set_input_parameters(const std::shared_ptr<InputParameters>& input_parameters)
{
    g_input_parameters = input_parameters;
}

void test_knobs::set_sdk_wrapper(const std::shared_ptr<SdkWrapper>& sdk_wrapper)
{
    g_sdk_wrapper = sdk_wrapper;
}

void test_knobs::set_csv_writer(const std::shared_ptr<CountersWriter>& csv_writer)
{
    g_counters_writer = csv_writer;
}

void test_knobs::reset_cfg()
{
    g_cfg.reset();
    g_tool_shutting_down.store(false, std::memory_order_release);
    g_hsa_intercept_done.store(false, std::memory_order_release);
}

namespace rocprofiler_compute_tool
{
static rocprofiler_context_id_t& get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

iteration_multiplexing_mode_t iteration_multiplexing_mode(std::string_view mode)
{
    if (mode == "kernel")
        return iteration_multiplexing_mode_t::KERNEL;
    else if (mode == "kernel_launch_params")
        return iteration_multiplexing_mode_t::LAUNCH;
    else
        return iteration_multiplexing_mode_t::DISABLED;
}

void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                       rocprofiler_counter_config_id_t*             config,
                       rocprofiler_user_data_t* /*user_data*/,
                       void* callback_data_args)
{
    g_sdk_callbacks->dispatch_callback(dispatch_data, config, callback_data_args);
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t*                record_data,
                     size_t                                       record_count,
                     rocprofiler_user_data_t /* user_data */,
                     void* callback_data_args)
{
    g_sdk_callbacks->record_callback(dispatch_data, record_data, record_count, callback_data_args);
}

void tool_tracing_callback(rocprofiler_callback_tracing_record_t record,
                           rocprofiler_user_data_t* /*user_data*/,
                           void* callback_data)
{
    g_sdk_callbacks->tool_tracing_callback(record, callback_data);
}

void on_hsa_runtime_loaded(rocprofiler_intercept_table_t /*type*/,
                           uint64_t /*lib_version*/,
                           uint64_t /*lib_instance*/,
                           void** /*tables*/,
                           uint64_t /*num_tables*/,
                           void* /*user_data*/)
{
    // Defer start_context until HSA loads: it starts HSA worker threads, which
    // would deadlock on fork() in non-GPU LD_PRELOAD'd shells.
    if (g_tool_shutting_down.load(std::memory_order_acquire))
        return;

    if (g_hsa_intercept_done.exchange(true, std::memory_order_acq_rel))
        return;

    g_sdk_wrapper->start_context(get_client_ctx());
}

int tool_init(rocprofiler_client_finalize_t, void* user_data)
{
    assert(user_data);
    std::clog << "[rocprofiler-compute] In tool init\n";
    g_sdk_wrapper->create_context(&get_client_ctx());

    g_sdk_wrapper->configure_callback_tracing_service(get_client_ctx(),
                                                      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                      nullptr,
                                                      0,
                                                      tool_tracing_callback,
                                                      user_data);

    // Declare counters before HSA loads so the SDK picks the legacy intercept path;
    // queue interposition (the default) cannot collect counters.
    auto* tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(user_data);
    if (!tool_data_ptr->get()->requested_counters.empty())
    {
        g_sdk_wrapper->configure_callback_dispatch_counting_service(get_client_ctx(),
                                                                    dispatch_callback,
                                                                    user_data,
                                                                    record_callback,
                                                                    user_data);
    }
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
    if (!tool_data->counter_records.empty() && !tool_data->output_filename.empty())
    {
        g_counters_writer->write_counters(tool_data);
    }

    if (tool_data->pc_sampling.enabled())
        tool_data->pc_sampling.finalize();
}

void tool_fini(void* user_data)
{
    g_tool_shutting_down.store(true, std::memory_order_release);
    assert(user_data);
    std::clog << "[rocprofiler-compute] In tool fini\n";
    rocprofiler_stop_context(get_client_ctx());

    auto* tool_data_ptr = static_cast<std::unique_ptr<tool_data_t>*>(user_data);
    generate_output(tool_data_ptr->get());

    delete tool_data_ptr;
}

}  // namespace rocprofiler_compute_tool

static std::string generate_output_filename(std::string_view output_path, std::string_view suffix)
{
    std::string filename{output_path};
    if (filename.back() != '/')
        filename += '/';
    filename += std::to_string(getpid());
    filename.append(suffix);
    return filename;
}

std::unique_ptr<tool_data_t> create_tool_data(rocprofiler_client_id_t* /*id*/)
{
    auto tool_data = std::make_unique<tool_data_t>();

    const auto output_path = g_input_parameters->get_output_path();
    tool_data->output_filename = generate_output_filename(output_path, "_native_counter_collection.csv");

    if (!g_input_parameters->get_pc_sampling_beta_enabled().empty())
    {
        const auto pc_mode = parse_pc_sampling_mode(
            std::string{g_input_parameters->get_pc_sampling_method()});
        tool_data->pc_sampling =
            pc_sampling_feature_t{pc_mode, generate_output_filename(output_path, "_code_obj_info.json")};
    }

    // ROCPROF_COUNTERS env. var. is a string like "pmc: counter1 counter2 ..."
    tool_data->requested_counters = std::string{g_input_parameters->get_requested_counters()};

    tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode(
        g_input_parameters->get_iteration_multiplexing_mode());

    // ROCPROF_KERNEL_FILTER_INCLUDE_REGEX env. var. is a regex string like
    // kernel_name_1|kernel_name_2|... Used to collect counters only for kernels
    // with names matching the regex
    tool_data->kernel_filter_include_regex = std::string{
        g_input_parameters->get_kernel_filter_include_regex()};

    // ROCPROF_KERNEL_FILTER_RANGE env. var. is a string like "[4,7-9,...]"
    std::string v_str{g_input_parameters->get_kernel_filter_range()};
    // Remove square brackets at the ends if present
    if (!v_str.empty() && v_str.front() == '[')
        v_str.erase(0, 1);
    if (!v_str.empty() && v_str.back() == ']')
        v_str.pop_back();
    // Parse the range string into vector of pairs
    std::istringstream ss{v_str};
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
    if (!g_cfg)
    {
        auto* tool_data_ptr = new std::unique_ptr<tool_data_t>(std::move(tool_data));
        g_cfg               = std::make_shared<rocprofiler_tool_configure_result_t>(
            rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                                &tool_init,
                                                &tool_fini,
                                                static_cast<void*>(tool_data_ptr)});

        // Defer HSA-touching counter setup until HSA actually loads in the
        // process. See on_hsa_runtime_loaded for rationale.
        g_sdk_wrapper->at_intercept_table_registration_hsa(&rocprofiler_compute_tool::on_hsa_runtime_loaded,
                                                           static_cast<void*>(tool_data_ptr));
    }

    return g_cfg.get();
}
