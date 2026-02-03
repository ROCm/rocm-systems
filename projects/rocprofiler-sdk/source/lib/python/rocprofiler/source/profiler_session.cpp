// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "profiler_session.hpp"

#include <rocprofiler-sdk/agent.h>
#include <rocprofiler-sdk/buffer.h>
#include <rocprofiler-sdk/context.h>
#include <rocprofiler-sdk/counters.h>
#include <rocprofiler-sdk/dispatch_counting_service.h>
#include <rocprofiler-sdk/registration.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <cstring>
#include <iostream>
#include <mutex>
#include <stdexcept>

namespace rocprofiler
{
namespace python
{
// Static members
ProfilerSession* ProfilerSession::s_active_session_ = nullptr;
std::mutex       ProfilerSession::s_session_mutex_;

// Global initialization state
static rocprofiler_client_id_t* g_client_id      = nullptr;
static bool                     g_tool_init_done = false;
static std::once_flag           g_init_flag;

// Kernel symbol resolver implementation
void
KernelSymbolResolver::register_kernel(uint64_t kernel_id, const std::string& name)
{
    std::unique_lock lock(mutex_);
    kernel_names_[kernel_id] = name;
}

std::string
KernelSymbolResolver::resolve(uint64_t kernel_id) const
{
    std::shared_lock lock(mutex_);
    auto             it = kernel_names_.find(kernel_id);
    return (it != kernel_names_.end()) ? it->second : "<unknown>";
}

// Dispatch callback - called when a kernel is enqueued
void
dispatch_callback(rocprofiler_dispatch_counting_service_data_t dispatch_data,
                  rocprofiler_counter_config_id_t*             config,
                  rocprofiler_user_data_t*                     user_data,
                  void*                                        callback_data_args)
{
    auto* session = static_cast<ProfilerSession*>(callback_data_args);
    if(!session) return;

    try
    {
        // Get cached config for this agent
        *config =
            session->config_manager().get_config_for_agent(dispatch_data.dispatch_info.agent_id);

        // Store dispatch ID for correlation
        user_data->value = dispatch_data.dispatch_info.dispatch_id;
    } catch(const std::exception& e)
    {
        std::cerr << "[rocprofiler-python] dispatch_callback error: " << e.what() << std::endl;
    }
}

// Buffered callback - called when buffer is full or flushed
void
buffered_callback(rocprofiler_context_id_t      context,
                  rocprofiler_buffer_id_t       buffer_id,
                  rocprofiler_record_header_t** headers,
                  size_t                        num_headers,
                  void*                         user_data,
                  uint64_t                      drop_count)
{
    (void) context;
    (void) buffer_id;
    (void) drop_count;

    auto* session = static_cast<ProfilerSession*>(user_data);
    if(!session) return;

    for(size_t i = 0; i < num_headers; ++i)
    {
        auto* header = headers[i];

        if(header->category == ROCPROFILER_BUFFER_CATEGORY_COUNTERS)
        {
            if(header->kind == ROCPROFILER_COUNTER_RECORD_PROFILE_COUNTING_DISPATCH_HEADER)
            {
                auto* record =
                    static_cast<rocprofiler_dispatch_counting_service_record_t*>(header->payload);

                // Resolve kernel name
                std::string kernel_name =
                    session->kernel_resolver().resolve(record->dispatch_info.kernel_id);

                session->record_collector().add_dispatch_header(*record, kernel_name);
            }
            else if(header->kind == ROCPROFILER_COUNTER_RECORD_VALUE)
            {
                auto* record = static_cast<rocprofiler_counter_record_t*>(header->payload);
                session->record_collector().add_counter_record(*record, session->config_manager());
            }
        }
    }

    // Invoke Python callback if set
    if(session->python_callback())
    {
        py::gil_scoped_acquire acquire;
        try
        {
            auto records = session->record_collector().get_records();
            (*session->python_callback())(records);
        } catch(const py::error_already_set& e)
        {
            // Log error but don't propagate to rocprofiler
            std::cerr << "[rocprofiler-python] Python callback error: " << e.what() << std::endl;
            PyErr_Clear();
        }
    }
}

// Note: Code object tracking for kernel name resolution would be done via
// callback tracing service if needed. For now, kernel names are resolved
// from the dispatch info directly.

// Tool initialization callback
int
tool_init(rocprofiler_client_finalize_t fini_func, void* tool_data)
{
    (void) fini_func;
    (void) tool_data;

    g_tool_init_done = true;
    return 0;
}

// Tool finalization callback
void
tool_fini(void* tool_data)
{
    (void) tool_data;
}

// rocprofiler_configure implementation for force_configure
extern "C" rocprofiler_tool_configure_result_t*
rocprofiler_configure(uint32_t                 version,
                      const char*              runtime_version,
                      uint32_t                 priority,
                      rocprofiler_client_id_t* id)
{
    (void) version;
    (void) runtime_version;
    (void) priority;

    id->name    = "rocprofiler-python";
    g_client_id = id;

    static auto cfg = rocprofiler_tool_configure_result_t{
        sizeof(rocprofiler_tool_configure_result_t), &tool_init, &tool_fini, nullptr};

    return &cfg;
}

bool
ensure_rocprofiler_initialized()
{
    static bool init_success = false;

    std::call_once(g_init_flag, []() {
        int status = 0;
        rocprofiler_is_initialized(&status);

        if(status == 0)
        {
            // Not yet initialized - we can register via force_configure
            auto result  = rocprofiler_force_configure(&rocprofiler_configure);
            init_success = (result == ROCPROFILER_STATUS_SUCCESS);
        }
        else if(status == 1)
        {
            // Already initialized - check if our tool is registered
            init_success = (g_client_id != nullptr);
        }
    });

    return init_success;
}

ProfilerSession::ProfilerSession(const std::vector<std::string>& metric_names,
                                 bool                            per_kernel,
                                 std::optional<py::function>     callback)
: metrics_(metric_names)
, per_kernel_(per_kernel)
, python_callback_(std::move(callback))
{
    if(metrics_.empty())
    {
        throw std::invalid_argument("At least one metric name must be specified");
    }

    if(!ensure_rocprofiler_initialized())
    {
        throw std::runtime_error("Failed to initialize rocprofiler");
    }

    config_manager_   = std::make_unique<CounterConfigManager>(metrics_);
    record_collector_ = std::make_unique<RecordCollector>();
}

ProfilerSession::~ProfilerSession()
{
    if(started_)
    {
        try
        {
            stop();
        } catch(...)
        {
            // Swallow exceptions in destructor
        }
    }
}

void
ProfilerSession::start()
{
    if(started_)
    {
        throw std::runtime_error("Profiling session already started");
    }

    std::lock_guard lock(s_session_mutex_);

    if(s_active_session_ != nullptr)
    {
        throw std::runtime_error("Another profiling session is already active");
    }

    // Create context
    auto status = rocprofiler_create_context(&ctx_);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create context: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Create buffer
    status = rocprofiler_create_buffer(ctx_,
                                       4096,  // buffer size
                                       2048,  // watermark
                                       ROCPROFILER_BUFFER_POLICY_LOSSLESS,
                                       buffered_callback,
                                       this,
                                       &buffer_);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create buffer: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Create callback thread
    rocprofiler_callback_thread_t client_thread = {};
    status = rocprofiler_create_callback_thread(&client_thread);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to create callback thread: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Assign callback thread to buffer
    status = rocprofiler_assign_callback_thread(buffer_, client_thread);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to assign callback thread: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Pre-build counter configs for all GPU agents
    auto agents = get_gpu_agents();
    for(const auto& agent : agents)
    {
        try
        {
            rocprofiler_agent_id_t agent_id = {.handle = agent.id};
            config_manager_->get_config_for_agent(agent_id);
        } catch(const std::exception& e)
        {
            std::cerr << "[rocprofiler-python] Warning: Failed to build config for agent "
                      << agent.name << ": " << e.what() << std::endl;
        }
    }

    // Configure dispatch counting service
    status = rocprofiler_configure_buffer_dispatch_counting_service(
        ctx_, buffer_, dispatch_callback, this);

    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to configure dispatch counting service: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    // Start the context
    status = rocprofiler_start_context(ctx_);
    if(status != ROCPROFILER_STATUS_SUCCESS)
    {
        throw std::runtime_error("Failed to start context: " +
                                 std::string(rocprofiler_get_status_string(status)));
    }

    s_active_session_ = this;
    started_          = true;
}

void
ProfilerSession::stop()
{
    if(!started_)
    {
        return;
    }

    std::lock_guard lock(s_session_mutex_);

    // Stop the context
    rocprofiler_stop_context(ctx_);

    // Flush the buffer to get remaining records
    rocprofiler_flush_buffer(buffer_);

    // Destroy the buffer
    if(buffer_.handle != 0)
    {
        rocprofiler_destroy_buffer(buffer_);
        buffer_ = {};
    }

    // Destroy counter configs
    config_manager_->destroy_all_configs();

    s_active_session_ = nullptr;
    started_          = false;
}

bool
ProfilerSession::is_active() const
{
    return started_;
}

std::vector<CounterRecord>
ProfilerSession::get_records() const
{
    return record_collector_->get_records();
}

void
ProfilerSession::clear_records()
{
    record_collector_->clear();
}

std::vector<CounterInfo>
ProfilerSession::get_available_counters(std::optional<int> device_id)
{
    std::vector<CounterInfo> result;

    auto agents = get_gpu_agents();

    for(const auto& agent : agents)
    {
        if(device_id.has_value() && agent.device_index != device_id.value())
        {
            continue;
        }

        rocprofiler_agent_id_t agent_id = {.handle = agent.id};

        std::vector<rocprofiler_counter_id_t> counters;

        auto status = rocprofiler_iterate_agent_supported_counters(
            agent_id,
            [](rocprofiler_agent_id_t,
               rocprofiler_counter_id_t* counter_arr,
               size_t                    num_counters,
               void*                     user_data) {
                auto* vec = static_cast<std::vector<rocprofiler_counter_id_t>*>(user_data);
                for(size_t i = 0; i < num_counters; i++)
                {
                    vec->push_back(counter_arr[i]);
                }
                return ROCPROFILER_STATUS_SUCCESS;
            },
            static_cast<void*>(&counters));

        if(status != ROCPROFILER_STATUS_SUCCESS)
        {
            continue;
        }

        for(const auto& counter : counters)
        {
            try
            {
                result.push_back(CounterConfigManager::get_counter_info(counter));
            } catch(...)
            {
                // Skip counters we can't query
            }
        }

        // If device_id was specified, we only wanted that device
        if(device_id.has_value())
        {
            break;
        }
    }

    return result;
}

std::vector<AgentInfo>
ProfilerSession::get_gpu_agents()
{
    std::vector<AgentInfo> result;

    auto iterate_cb = [](rocprofiler_agent_version_t agents_ver,
                         const void**                agents_arr,
                         size_t                      num_agents,
                         void*                       user_data) {
        if(agents_ver != ROCPROFILER_AGENT_INFO_VERSION_0) return ROCPROFILER_STATUS_ERROR;

        auto* agents = static_cast<std::vector<AgentInfo>*>(user_data);

        for(size_t i = 0; i < num_agents; ++i)
        {
            const auto* agent = static_cast<const rocprofiler_agent_v0_t*>(agents_arr[i]);
            if(agent->type == ROCPROFILER_AGENT_TYPE_GPU)
            {
                AgentInfo info;
                info.id           = agent->id.handle;
                info.name         = agent->name ? agent->name : "";
                info.product_name = agent->product_name ? agent->product_name : "";
                info.device_index = static_cast<int>(agent->logical_node_id);
                info.gfx_version  = agent->gfx_target_version;
                agents->push_back(info);
            }
        }

        return ROCPROFILER_STATUS_SUCCESS;
    };

    rocprofiler_query_available_agents(ROCPROFILER_AGENT_INFO_VERSION_0,
                                       iterate_cb,
                                       sizeof(rocprofiler_agent_v0_t),
                                       static_cast<void*>(&result));

    return result;
}

}  // namespace python
}  // namespace rocprofiler
