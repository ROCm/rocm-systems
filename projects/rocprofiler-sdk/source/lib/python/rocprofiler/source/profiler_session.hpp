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

#pragma once

#include "counter_config_manager.hpp"
#include "record_collector.hpp"
#include "types.hpp"

#include <rocprofiler-sdk/fwd.h>
#include <rocprofiler-sdk/rocprofiler.h>

#include <pybind11/functional.h>
#include <pybind11/pybind11.h>

#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace py = pybind11;

namespace rocprofiler
{
namespace python
{
/**
 * @brief Kernel symbol resolver for mapping kernel IDs to names
 */
class __attribute__((visibility("default"))) KernelSymbolResolver
{
public:
    void register_kernel(uint64_t kernel_id, const std::string& name);

    std::string resolve(uint64_t kernel_id) const;

private:
    mutable std::shared_mutex                 mutex_;
    std::unordered_map<uint64_t, std::string> kernel_names_;
};

/**
 * @brief Main profiling session class
 *
 * This class manages a hardware counter profiling session. It handles
 * context creation, counter configuration, and record collection.
 */
class __attribute__((visibility("default"))) ProfilerSession
{
public:
    /**
     * @brief Construct a new profiler session
     * @param metric_names List of counter names to collect
     * @param per_kernel If true, collect counters per kernel dispatch
     * @param callback Optional Python callback for streaming results
     */
    ProfilerSession(const std::vector<std::string>& metric_names,
                    bool                            per_kernel,
                    std::optional<py::function>     callback);

    ~ProfilerSession();

    // Non-copyable, non-movable
    ProfilerSession(const ProfilerSession&) = delete;
    ProfilerSession& operator=(const ProfilerSession&) = delete;
    ProfilerSession(ProfilerSession&&)                 = delete;
    ProfilerSession& operator=(ProfilerSession&&) = delete;

    /**
     * @brief Start the profiling session
     */
    void start();

    /**
     * @brief Stop the profiling session
     */
    void stop();

    /**
     * @brief Check if profiling is active
     */
    bool is_active() const;

    /**
     * @brief Get all collected counter records
     */
    std::vector<CounterRecord> get_records() const;

    /**
     * @brief Clear all collected records
     */
    void clear_records();

    /**
     * @brief Get list of available counters for a device
     * @param device_id Optional device ID (nullopt for all devices)
     */
    static std::vector<CounterInfo> get_available_counters(std::optional<int> device_id);

    /**
     * @brief Get list of GPU agents
     */
    static std::vector<AgentInfo> get_gpu_agents();

    // Accessors for callback functions
    CounterConfigManager& config_manager() { return *config_manager_; }
    RecordCollector&      record_collector() { return *record_collector_; }
    KernelSymbolResolver& kernel_resolver() { return kernel_resolver_; }

    const std::optional<py::function>& python_callback() const { return python_callback_; }

    // Static session pointer for callbacks (public for access from free functions)
    static ProfilerSession* s_active_session_;
    static std::mutex       s_session_mutex_;

private:
    std::unique_ptr<CounterConfigManager> config_manager_;
    std::unique_ptr<RecordCollector>      record_collector_;
    KernelSymbolResolver                  kernel_resolver_;

    std::vector<std::string>    metrics_;
    bool                        per_kernel_ = true;
    std::optional<py::function> python_callback_;
    bool                        started_ = false;
};

/**
 * @brief Ensure rocprofiler is initialized
 * @return true if initialization succeeded
 */
__attribute__((visibility("default"))) bool
ensure_rocprofiler_initialized();

}  // namespace python
}  // namespace rocprofiler
