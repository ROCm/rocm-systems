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

#include "helper.hpp"
#include "rocprofiler-sdk/context.h"
#include "trace_buffer.hpp"

#include "lib/common/environment.hpp"
#include "lib/common/filesystem.hpp"

#include <fmt/core.h>
#include <unistd.h>

#include <fstream>
#include <iomanip>
#include <mutex>
#include <shared_mutex>
#include <unordered_map>
#include <vector>

namespace common = ::rocprofiler::common;
namespace fs     = common::filesystem;

TRACE_BUFFER_INSTANTIATE();

// static const uint32_t lds_block_size = 128 * 4;

namespace
{
auto tool_buffer = rocprofiler_buffer_id_t{};
auto context_id  = rocprofiler_context_id_t{};
auto output_path =
    fs::path{common::get_env<std::string>("ROCPROFILER_OUTPUT_PATH", fs::current_path().string())};
auto output_file_name =
    common::get_env<std::string>("ROCPROFILER_OUTPUT_FILE_NAME", std::to_string(getpid()) + "-");

std::pair<std::ostream*, void (*)(std::ostream*&)>
get_output_stream(const std::string& fname, const std::string& ext = ".csv")
{
    if(output_path.string().empty()) return {&std::clog, [](auto*&) {}};

    if(fs::exists(output_path) && !fs::is_directory(fs::status(output_path)))
        throw std::runtime_error{
            fmt::format("ROCPROFILER_OUTPUT_PATH ({}) already exists and is not a directory",
                        output_path.string())};
    if(!fs::exists(output_path)) fs::create_directories(output_path);

    auto  output_file = output_path / (output_file_name + fname + ext);
    auto* _ofs        = new std::ofstream{output_file};
    if(!_ofs && !*_ofs)
        throw std::runtime_error{fmt::format("Failed to open {} for output", output_file.string())};
    std::cout << "Results File: " << output_file << std::endl;
    return {_ofs, [](std::ostream*& v) {
                if(v) dynamic_cast<std::ofstream*>(v)->close();
                delete v;
                v = nullptr;
            }};
}

template <typename Tp>
std::string
as_hex(Tp _v, size_t _width = 16)
{
    auto _ss = std::stringstream{};
    _ss.fill('0');
    _ss << "0x" << std::hex << std::setw(_width) << _v;
    return _ss.str();
}
}  // namespace

struct output_file
{
    output_file(std::string name, std::vector<std::string>&& header)
    : m_name{std::move(name)}
    {
        std::tie(m_stream, m_dtor) = get_output_stream(m_name);
        auto ss                    = std::stringstream{};
        for(auto&& itr : header)
        {
            ss << "," << itr;
        }

        // write the csv header
        if(!ss.str().empty()) *m_stream << ss.str().substr(1) << '\n';
    }

    ~output_file() { m_dtor(m_stream); }

    output_file(const output_file&) = delete;
    output_file& operator=(const output_file&) = delete;

    std::string name() const { return m_name; }

    template <typename T>
    std::ostream& operator<<(T&& value)
    {
        return (*m_stream) << std::forward<T>(value);
    }

    std::ostream& operator<<(std::ostream& (*func)(std::ostream&) ) { return (*m_stream) << func; }

    operator bool() const { return m_stream != nullptr; }

private:
    using stream_dtor_t = void (*)(std::ostream*&);

    const std::string m_name   = {};
    std::ostream*     m_stream = nullptr;
    stream_dtor_t     m_dtor   = [](std::ostream*&) {};
};

auto&
get_hsa_api_file()
{
    static auto _v =
        output_file{"hsa_api_trace", {"KERNEL_NAME", "BEGIN_TS", "END_TS", "CORRELATION_ID"}};
    return _v;
}

auto&
get_kernel_trace_file()
{
    static auto _v = output_file{"kernel_trace",
                                 {"AGENT_ID",
                                  "QUEUE_ID",
                                  "KERNEL_ID",
                                  "KERNEL_NAME",
                                  "CONTEXT_ID",
                                  "BUFFER_ID",
                                  "CORRELATION_ID",
                                  "KIND",
                                  "START_TS",
                                  "END_TS",
                                  "PRIVATE_SEGMENT_SIZE",
                                  "GROUP_SEGMENT_SIZE",
                                  "WORKGROUP_SIZE_X",
                                  "WORKGROUP_SIZE_Y",
                                  "WORKGROUP_SIZE_Z",
                                  "GRID_SIZE_X",
                                  "GRID_SIZE_Y",
                                  "GRID_SIZE_Z"}};
    return _v;
}

std::shared_mutex                                        kernel_data_mutex;
std::unordered_map<rocprofiler_kernel_id_t, std::string> kernel_data;

struct hsa_api_trace_entry_t
{
    std::atomic<uint32_t>                 valid;
    rocprofiler_callback_tracing_record_t record;
    rocprofiler_timestamp_t               begin_timestamp;
    rocprofiler_timestamp_t               end_timestamp;
    std::string_view                      api_name;

    hsa_api_trace_entry_t(rocprofiler_timestamp_t               begin,
                          rocprofiler_timestamp_t               end,
                          rocprofiler_callback_tracing_record_t tracer_record,
                          std::string_view                      name

                          )
    : valid(TRACE_ENTRY_INIT)
    , record(tracer_record)
    , begin_timestamp(begin)
    , end_timestamp(end)
    , api_name(name)
    {}
};

TraceBuffer<hsa_api_trace_entry_t> hsa_api_buffer("HSA API",
                                                  0x200000,
                                                  [](hsa_api_trace_entry_t* entry) {
                                                      TracerFlushRecord(
                                                          entry,
                                                          ROCPROFILER_CALLBACK_TRACING_HSA_API);
                                                  });

rocprofiler_tool_callback_name_info_t name_info;

void
TracerFlushRecord(void* data, rocprofiler_callback_tracing_kind_t kind)
{
    if(kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
    {
        auto* entry = reinterpret_cast<hsa_api_trace_entry_t*>(data);
        get_hsa_api_file() << "\"" << entry->api_name << "\""
                           << "," << entry->begin_timestamp << ":" << entry->end_timestamp << " "
                           << entry->record.correlation_id.internal << '\n';
    }
}
void
rocprofiler_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t*              user_data,
                             void*                                 data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_HIP_API)
    {
        // To be implemented
        throw std::runtime_error{"not implemented"};
    }

    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
    {
        auto timestamp = rocprofiler_timestamp_t{};
        ROCPROFILER_CALL(rocprofiler_get_timestamp(&timestamp), "timestamp failed");

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_ENTER)
        {
            user_data->value = timestamp;
        }
        else if(record.phase == ROCPROFILER_CALLBACK_PHASE_EXIT)
        {
            const auto* info_name_str = name_info.operation_names[record.kind][record.operation];
            hsa_api_trace_entry_t& entry =
                hsa_api_buffer.Emplace(user_data->value, timestamp, record, info_name_str);
            entry.valid.store(TRACE_ENTRY_COMPLETE, std::memory_order_release);
        }
    }
    else if(record.kind == ROCPROFILER_CALLBACK_TRACING_MARKER_API)
    {
        // To be implemented
        throw std::runtime_error{"not implemented"};
    }
    (void) (data);
}

void
code_object_tracing_callback(rocprofiler_callback_tracing_record_t record,
                             rocprofiler_user_data_t*              user_data,
                             void*                                 data)
{
    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_LOAD)
    {
        if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        {
            // flush the buffer to ensure that any lookups for the client kernel names for the code
            // object are completed
            auto flush_status = rocprofiler_flush_buffer(tool_buffer);
            if(flush_status != ROCPROFILER_STATUS_ERROR_BUFFER_BUSY)
                ROCPROFILER_CALL(flush_status, "buffer flush");
        }
    }

    if(record.kind == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT &&
       record.operation == ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER)
    {
        auto* sym_data =
            static_cast<rocprofiler_callback_tracing_code_object_kernel_symbol_register_data_t*>(
                record.payload);

        if(record.phase == ROCPROFILER_CALLBACK_PHASE_LOAD)
        {
            auto kernel_name =
                std::regex_replace(sym_data->kernel_name, std::regex{"(\\.kd)$"}, "");
            int demangle_status = 0;
            kernel_name         = cxa_demangle(kernel_name, &demangle_status);
            std::unique_lock<std::shared_mutex> lock(kernel_data_mutex);
            kernel_data.emplace(sym_data->kernel_id, kernel_name);
        }
        // The map entry cannot be erased here
        // since we are tracing the kernel symbols here not the kernel dispatch
        // else if(record.phase == ROCPROFILER_CALLBACK_PHASE_UNLOAD)
        //{

        //  kernel_data.erase(data->kernel_id);

        //}
    }

    (void) user_data;
    (void) data;
}

void
kernel_tracing_callback(rocprofiler_context_id_t      context,
                        rocprofiler_buffer_id_t       buffer_id,
                        rocprofiler_record_header_t** headers,
                        size_t                        num_headers,
                        void*                         user_data,
                        uint64_t /*drop_count*/)
{
    if(num_headers == 0)
        throw std::runtime_error{
            "rocprofiler invoked a buffer callback with no headers. this should never happen"};
    else if(headers == nullptr)
        throw std::runtime_error{"rocprofiler invoked a buffer callback with a null pointer to the "
                                 "array of headers. this should never happen"};

    auto kernel_trace_ss = std::stringstream{};
    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header == nullptr)
        {
            throw std::runtime_error{
                "rocprofiler provided a null pointer to header. this should never happen"};
        }
        else if(header->hash !=
                rocprofiler_record_header_compute_hash(header->category, header->kind))
        {
            throw std::runtime_error{"rocprofiler_record_header_t (category | kind) != hash"};
        }
        else if(header->category == ROCPROFILER_BUFFER_CATEGORY_TRACING &&
                header->kind == ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH)
        {
            auto* record =
                static_cast<rocprofiler_buffer_tracing_kernel_dispatch_record_t*>(header->payload);
            std::string kernel_name;
            {
                std::shared_lock<std::shared_mutex> lock(kernel_data_mutex);
                kernel_name = kernel_data.at(record->kernel_id);
            }

            kernel_trace_ss << record->agent_id.handle << "," << record->queue_id.handle << ","
                            << record->kernel_id << ",\"" << kernel_name << "\"," << context.handle
                            << "," << buffer_id.handle << "," << record->correlation_id.internal
                            << "," << record->kind << "," << record->start_timestamp << ","
                            << record->end_timestamp << "," << record->private_segment_size << ","
                            << record->group_segment_size << "," << record->workgroup_size.x << ","
                            << record->workgroup_size.y << "," << record->workgroup_size.z << ","
                            << record->grid_size.x << "," << record->grid_size.y << ","
                            << record->grid_size.z << '\n';
        }
    }

    static auto _sync = std::mutex{};
    auto        _lk   = std::unique_lock<std::mutex>{_sync};
    if(get_kernel_trace_file())
        get_kernel_trace_file() << kernel_trace_ss.str();
    else
        std::cerr << "kernel trace file already closed: " << kernel_trace_ss.str();

    (void) (user_data);
}

rocprofiler_tool_callback_name_info_t
get_callback_id_names()
{
    auto cb_name_info = rocprofiler_tool_callback_name_info_t{};
    //
    // callback for each kind operation
    //
    static auto tracing_kind_operation_cb =
        [](rocprofiler_callback_tracing_kind_t kindv, uint32_t operation, void* data_v) {
            auto* name_info_v = static_cast<rocprofiler_tool_callback_name_info_t*>(data_v);

            if(kindv == ROCPROFILER_CALLBACK_TRACING_HSA_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback failed");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }

            if(kindv == ROCPROFILER_CALLBACK_TRACING_HIP_API)
            {
                const char* name = nullptr;
                ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_operation_name(
                                     kindv, operation, &name, nullptr),
                                 "query callback failed");
                if(name) name_info_v->operation_names[kindv][operation] = name;
            }
            return 0;
        };

    //
    //  callback for each callback kind (i.e. domain)
    //
    static auto tracing_kind_cb = [](rocprofiler_callback_tracing_kind_t kind, void* data) {
        //  store the callback kind name
        auto*       name_info_v = static_cast<rocprofiler_tool_callback_name_info_t*>(data);
        const char* name        = nullptr;
        ROCPROFILER_CALL(rocprofiler_query_callback_tracing_kind_name(kind, &name, nullptr),
                         "query callback failed");

        if(name) name_info_v->kind_names[kind] = name;

        if(kind == ROCPROFILER_CALLBACK_TRACING_HSA_API)
        {
            ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kind_operations(
                                 kind, tracing_kind_operation_cb, static_cast<void*>(data)),
                             "query callback failed");
        }
        return 0;
    };

    ROCPROFILER_CALL(rocprofiler_iterate_callback_tracing_kinds(tracing_kind_cb,
                                                                static_cast<void*>(&cb_name_info)),
                     "iterate_callback failed");

    return cb_name_info;
}

int
tool_init(rocprofiler_client_finalize_t /*fini_func*/, void* tool_data)
{
    name_info = get_callback_id_names();

    ROCPROFILER_CALL(rocprofiler_create_context(&context_id), "create context failed");

    if(common::get_env("ROCPROFILER_KERNEL_TRACE", false))
    {
        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(context_id,
                                                           ROCPROFILER_CALLBACK_TRACING_CODE_OBJECT,
                                                           nullptr,
                                                           0,
                                                           code_object_tracing_callback,
                                                           nullptr),
            "tracing configure failed");

        ROCPROFILER_CALL(rocprofiler_create_buffer(context_id,
                                                   4096,
                                                   2048,
                                                   ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                                   kernel_tracing_callback,
                                                   tool_data,
                                                   &tool_buffer),
                         "buffer creation");

        ROCPROFILER_CALL(
            rocprofiler_configure_buffer_tracing_service(
                context_id, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, tool_buffer),
            "buffer tracing service for kernel dispatch configure");
    }

    if(common::get_env("ROCPROFILER_HSA_API_TRACE", false))
    {
        // Requesting all operations
        ROCPROFILER_CALL(
            rocprofiler_configure_callback_tracing_service(context_id,
                                                           ROCPROFILER_CALLBACK_TRACING_HSA_API,
                                                           nullptr,
                                                           0,
                                                           rocprofiler_tracing_callback,
                                                           nullptr),
            "tracing configure failed");
    }

    ROCPROFILER_CALL(rocprofiler_start_context(context_id), "start context failed");

    return 0;
}

void
tool_fini(void* tool_data)
{
    rocprofiler_flush_buffer(tool_buffer);
    rocprofiler_stop_context(context_id);
    (void) (tool_data);
}

extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t /*version*/,
                      const char* /*runtime_version*/,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    // only activate if main tool
    if(priority > 0) return nullptr;

    // set the client name
    id->name = "rocporfiler-tool";

    // store client info
    // client::client_id = id;

    // create configure data
    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    // return pointer to configure data
    return &cfg;
    // data passed around all the callbacks
}
