// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#include "rocprofiler_compute_tool.h"
#include "rocprofiler-sdk/cxx/codeobj/code_printing.hpp"

#include "env_parameters.h"
#include "gsl_assert.h"
#include "sdk_callbacks.h"
#include "sdk_wrapper.h"

#include <unistd.h>

#include <iostream>
#include <regex>
#include <sstream>

using namespace rocm_compute;

static std::shared_ptr<env_parameters_t> g_input_parameters = std::make_shared<env_parameters_impl_t>();
static std::shared_ptr<sdk_wrapper_t> g_sdk_wrapper = std::make_shared<sdk_wrapper_impl_t>();
static std::shared_ptr<counters_writer_t> g_counters_writer = std::make_shared<csv_counters_writer_t>();
static std::shared_ptr<rocprofiler_tool_configure_result_t> g_cfg;

void test_knobs::set_env_parameters(const std::shared_ptr<env_parameters_t>& input_parameters)
{
    g_input_parameters = input_parameters;
}

void test_knobs::set_sdk_wrapper(const std::shared_ptr<sdk_wrapper_t>& sdk_wrapper)
{
    g_sdk_wrapper = sdk_wrapper;
}

void test_knobs::set_csv_writer(const std::shared_ptr<counters_writer_t>& csv_writer)
{
    g_counters_writer = csv_writer;
}

void test_knobs::reset_cfg()
{
    g_cfg.reset();
}

namespace rocm_compute
{
static rocprofiler_context_id_t& get_client_ctx()
{
    static rocprofiler_context_id_t ctx{0};
    return ctx;
}

IterationMultiplexingMode iteration_multiplexing_mode(const std::string& mode)
{
    if (mode == "kernel")
        return IterationMultiplexingMode::Kernel;
    if (mode == "kernel_launch_params")
        return IterationMultiplexingMode::Launch;
    return IterationMultiplexingMode::Disabled;
}

PcSamplingMode pc_sampling_mode(const std::string& mode)
{
    if (mode == "stochastic")
        return PcSamplingMode::Stochastic;
    if (mode == "host_trap")
        return PcSamplingMode::HostTrap;
    return PcSamplingMode::Disabled;
}

void dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                       rocprofiler_counter_config_id_t*             config,
                       rocprofiler_user_data_t* /*user_data*/,
                       void* callback_data_args)
{
    Expects(callback_data_args);
    auto* tool_data = static_cast<tool_data_t*>(callback_data_args);
    tool_data->sdk_callbacks->dispatch_callback(dispatch_data, config, *tool_data);
}

void record_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                     rocprofiler_counter_record_t*                record_data,
                     size_t                                       record_count,
                     rocprofiler_user_data_t /* user_data */,
                     void* callback_data_args)
{
    Expects(callback_data_args);
    auto* tool_data = static_cast<tool_data_t*>(callback_data_args);
    tool_data->sdk_callbacks->record_callback(dispatch_data, record_data, record_count, *tool_data);
}

rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate codeobjTranslate;
using code_obj_load_data_t = rocprofiler_callback_tracing_code_object_load_data_t;
using kernel_symbol_data_t = rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t;
using kernel_symbol_map_t  = std::unordered_map<std::string, std::pair<uint64_t, size_t>>;

using Instruction             = rocprofiler::sdk::codeobj::disassembly::Instruction;
using CodeobjAddressTranslate = rocprofiler::sdk::codeobj::disassembly::CodeobjAddressTranslate;

rocprofiler_client_id_t*      client_id          = nullptr;
rocprofiler_client_finalize_t client_fini_func   = nullptr;
rocprofiler_context_id_t      client_ctx         = {0};
kernel_symbol_map_t           registered_kernels = {};

#define OUTPUT_OFSTREAM "code_obj_isa_decode.log"
std::ostream&
output_stream()
{
    static std::ofstream file(OUTPUT_OFSTREAM);

    static bool file_is_open_check = [&]() {
        if(!file.is_open())
            std::cout << "Could not open log file: " << OUTPUT_OFSTREAM << ", writing to stdout\n";
        else
            std::cout << "Writing code-object-isa-decode log to: " << OUTPUT_OFSTREAM << std::endl;
        return file.is_open();
    }();

    if(!file_is_open_check) return std::cout;
    return file;
};

void code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                                  rocprofiler_user_data_t* /*user_data*/,
                                  void* callback_data)
{
    Expects(callback_data);
    auto* tool_data = static_cast<tool_data_t*>(callback_data);
    if (record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
        record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
    {
        switch (record.operation)
        {
        case ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER:
        {
            tool_data->sdk_callbacks->tool_tracing_callback(record, *tool_data);
        }
        break;
        case ROCPROFILER_CODE_OBJECT_LOAD:
        {
            if (pc_sampling_mode(g_input_parameters->get_pc_sampling_mode()) != PcSamplingMode::Disabled)
            {
                Expects(record.payload);
                auto* obj_data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(
                    record.payload);
                tool_data->pc_sampling_collector.rlock([&](const pc_sampling_collector_t::ptr& collector)
                                                       { collector->on_code_object_load(*obj_data); });
            }
        }
        break;
        default:
            break;
        }
    }

    if (record.kind != ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT)
        return;
    if (record.phase != ROCPROFILER_CALLBACK_PHASE_LOAD)
        return;

    if (record.operation == ROCPROFILER_CODE_OBJECT_LOAD)
    {
        auto* data = static_cast<rocprofiler_callback_tracing_code_object_load_data_t*>(record.payload);

        if (std::string_view(data->uri).find("file:///") == 0)
        {
            codeobjTranslate.addDecoder(data->uri, data->code_object_id, data->load_delta, data->load_size);
        }
        else
        {
            codeobjTranslate.addDecoder(reinterpret_cast<const void*>(data->memory_base),
                                        data->memory_size,
                                        data->code_object_id,
                                        data->load_delta,
                                        data->load_size);
        }

        auto symbolmap = codeobjTranslate.getSymbolMap();
        for (auto& [vaddr, symbol] : symbolmap)
            registered_kernels.insert({symbol.name, {vaddr, vaddr + symbol.mem_size}});
    }
    else if (record.operation == ROCPROFILER_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        output_stream() << std::hex;
        auto* data        = static_cast<kernel_symbol_data_t*>(record.payload);
        auto  kernel_name = std::regex_replace(data->kernel_name, std::regex{"(\\.kd)$"}, "");

        if (registered_kernels.find(kernel_name) == registered_kernels.end())
        {
            output_stream() << "Not Found: " << kernel_name << " in codeobj." << std::endl;
            return;
        }

        auto& begin_end = registered_kernels.at(kernel_name);

        output_stream() << std::hex << "Found: " << kernel_name << " at addr: 0x" << begin_end.first
                        << std::dec << ". Printing first 64 bytes:" << std::endl;

        std::unordered_set<std::string> references{};

        int num_waitcnts = 0;
        int num_scalar   = 0;
        int num_vector   = 0;
        int num_other    = 0;

        size_t vaddr = begin_end.first;
        while (vaddr < begin_end.second)
        {
            auto inst = codeobjTranslate.get(vaddr);
            assert(inst != nullptr);
            if (inst->comment.size())
            {
                std::string_view source = inst->comment;
                if (source.rfind('/') < source.size())
                    source = source.substr(source.rfind('/'));
                if (vaddr < begin_end.first + 64)
                    output_stream() << '\t' << inst->inst << '\n';

                if (source.rfind(':') < source.size())
                    source = source.substr(0, source.rfind(':'));

                references.insert(std::string(source));
            }
            if (inst->inst.find("v_") == 0)
                num_vector++;
            else if (inst->inst.find("s_waitcnt") == 0)
                num_waitcnts++;
            else if (inst->inst.find("s_") == 0)
                num_scalar++;
            else
                num_other++;

            vaddr += inst->size;
        }

        output_stream() << "  --- Num Scalar: " << num_scalar << "\n  --- Num Vector: " << num_vector
                        << "\n  --- Num Waitcnts: " << num_waitcnts
                        << "\n  --- Other instructions: " << num_other
                        << "\nKernel has source references to: " << std::endl;
        for (auto& ref : references)
            output_stream() << '\t' << ref << std::endl;
    }

    (void)callback_data;
}

int tool_init(rocprofiler_client_finalize_t, void* user_data)
{
    std::clog << "[rocprofiler-compute] In tool init\n";
    g_sdk_wrapper->create_context(&get_client_ctx());

    g_sdk_wrapper->configure_callback_dispatch_counting_service(get_client_ctx(),
                                                                dispatch_callback,
                                                                user_data,
                                                                record_callback,
                                                                user_data);
    g_sdk_wrapper->configure_callback_tracing_service(get_client_ctx(),
                                                      ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                      nullptr,
                                                      0,
                                                      code_object_tracing_callback,
                                                      user_data);
    g_sdk_wrapper->start_context(get_client_ctx());

    return 0;
}

void generate_output(tool_data_t& tool_data)
{
    // Dispatches before the kernel to be filtered was registered may have been
    // profiled. Remove any records whose kernel id does not match the
    // target_kernel_ids
    if (!tool_data.counter_records.empty() && !tool_data.counters_output_filename.empty())
    {
        if (!tool_data.target_kernel_ids.empty())
        {
            tool_data.counter_records.erase(
                std::remove_if(tool_data.counter_records.begin(),
                               tool_data.counter_records.end(),
                               [&tool_data](const counter_info_record_t& record)
                               {
                                   return tool_data.target_kernel_ids.find(record.kernel_id) ==
                                          tool_data.target_kernel_ids.end();
                               }),
                tool_data.counter_records.end());
        }
        g_counters_writer->write_counters(tool_data.counters_output_filename, tool_data.counter_records);
    }

    if (tool_data.pc_sampling_mode != PcSamplingMode::Disabled)
    {
        code_object_writer_json_t obj_writer;
        tool_data.pc_sampling_collector.rlock([&](const pc_sampling_collector_t::ptr& ptr)
                                              { ptr->write(obj_writer); });
        obj_writer.flush(tool_data.code_obj_output_filename);
    }
}

void tool_fini(void* user_data)
{
    Expects(user_data) std::clog << "[rocprofiler-compute] In tool fini\n";
    rocprofiler_stop_context(get_client_ctx());

    auto* tool_data_ptr = static_cast<tool_data_t*>(user_data);
    generate_output(*tool_data_ptr);

    delete tool_data_ptr;
}
}  // namespace rocm_compute

static std::string generate_output_filename(const std::string& output_path, const std::string& suffix)
{
    Expects(!output_path.empty());
    std::string result_output_path = output_path;
    if (result_output_path.back() != '/')
        result_output_path += '/';

    const std::string filename = std::to_string(getpid()) + suffix;
    return result_output_path + filename;
}

std::unique_ptr<tool_data_t> create_tool_data(rocprofiler_client_id_t* /*id*/)
{
    auto tool_data = std::make_unique<tool_data_t>();

    tool_data->sdk_callbacks = std::make_shared<sdk_callbacks_impl_t>(g_sdk_wrapper);
    tool_data->pc_sampling_collector.wlock([](auto& ptr) { ptr = pc_sampling_collector_t::create(); });
    tool_data->pc_sampling_mode = pc_sampling_mode(g_input_parameters->get_pc_sampling_mode());

    tool_data->counters_output_filename =
        generate_output_filename(g_input_parameters->get_output_path(), "_native_counter_collection.csv");
    tool_data->code_obj_output_filename = generate_output_filename(g_input_parameters->get_output_path(),
                                                                   "_code_obj_info.csv");

    // ROCPROF_COUNTERS env. var. is a string like "pmc: counter1 counter2 ..."
    tool_data->requested_counters = g_input_parameters->get_requested_counters();

    tool_data->iteration_multiplexing_mode = iteration_multiplexing_mode(
        g_input_parameters->get_iteration_multiplexing_mode());

    // ROCPROF_KERNEL_FILTER_INCLUDE_REGEX env. var. is a regex string like
    // kernel_name_1|kernel_name_2|... Used to collect counters only for kernels
    // with names matching the regex
    tool_data->kernel_filter_include_regex = g_input_parameters->get_kernel_filter_include_regex();

    // ROCPROF_KERNEL_FILTER_RANGE env. var. is a string like "[4,7-9,...]"
    if (!g_input_parameters->get_kernel_filter_range().empty())
    {
        // Remove square brackets at the ends if present
        std::string v_str = g_input_parameters->get_kernel_filter_range();
        if (!v_str.empty() && v_str.front() == '[')
            v_str.erase(0, 1);
        if (!v_str.empty() && v_str.back() == ']')
            v_str.pop_back();
        // Parse the range string into vector of pairs
        std::istringstream ss(v_str.c_str());
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
    // set the client kernel_name
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
        g_cfg = std::make_shared<rocprofiler_tool_configure_result_t>(
            rocprofiler_tool_configure_result_t{sizeof(rocprofiler_tool_configure_result_t),
                                                &tool_init,
                                                &tool_fini,
                                                tool_data.release()});

    return g_cfg.get();
}
