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

#include "libpyrocprofiler.hpp"
#include "source/profiler_session.hpp"
#include "source/types.hpp"

namespace py = pybind11;

using namespace rocprofiler::python;

PYBIND11_MODULE(libpyrocprofiler, m)
{
    py::options options;
    options.disable_function_signatures();

    m.doc() = "ROCProfiler SDK Python bindings for hardware counter collection";

    // Register exception type
    static py::exception<std::runtime_error> RocprofilerError(m, "RocprofilerError");

    // CounterInfo data class
    py::class_<CounterInfo>(m, "CounterInfo", "Information about a hardware counter")
        .def_readonly("id", &CounterInfo::id, "Counter ID")
        .def_readonly("name", &CounterInfo::name, "Counter name")
        .def_readonly("description", &CounterInfo::description, "Counter description")
        .def_readonly("block", &CounterInfo::block, "Hardware block")
        .def_readonly("expression", &CounterInfo::expression, "Counter expression")
        .def_readonly("is_constant", &CounterInfo::is_constant, "Whether counter is constant")
        .def_readonly("is_derived", &CounterInfo::is_derived, "Whether counter is derived")
        .def("__repr__",
             [](const CounterInfo& c) { return "<CounterInfo name='" + c.name + "'>"; });

    // AgentInfo data class
    py::class_<AgentInfo>(m, "AgentInfo", "Information about a GPU agent (device)")
        .def_readonly("id", &AgentInfo::id, "Agent ID")
        .def_readonly("name", &AgentInfo::name, "Agent name")
        .def_readonly("product_name", &AgentInfo::product_name, "Product name")
        .def_readonly("device_index", &AgentInfo::device_index, "Device index")
        .def_readonly("gfx_version", &AgentInfo::gfx_version, "GFX version")
        .def("__repr__", [](const AgentInfo& a) {
            return "<AgentInfo name='" + a.name +
                   "' device_index=" + std::to_string(a.device_index) + ">";
        });

    // CounterRecord data class
    py::class_<CounterRecord>(m, "CounterRecord", "A single hardware counter measurement")
        .def_readonly("dispatch_id", &CounterRecord::dispatch_id, "Dispatch ID")
        .def_readonly("counter_id", &CounterRecord::counter_id, "Counter ID")
        .def_readonly("counter_name", &CounterRecord::counter_name, "Counter name")
        .def_readonly("kernel_name", &CounterRecord::kernel_name, "Kernel name")
        .def_readonly("value", &CounterRecord::value, "Counter value")
        .def_readonly("agent_id", &CounterRecord::agent_id, "Agent ID")
        .def_readonly("dimensions", &CounterRecord::dimensions, "Dimension positions")
        .def("__repr__", [](const CounterRecord& r) {
            return "<CounterRecord kernel='" + r.kernel_name + "' counter='" + r.counter_name +
                   "' value=" + std::to_string(r.value) + ">";
        });

    // ProfilerSession - main profiling class
    py::class_<ProfilerSession>(m, "ProfilerSession", R"doc(
Hardware counter profiling session.

This class manages a hardware counter collection session. It handles
context creation, counter configuration, and record collection.

Parameters
----------
metrics : list of str
    List of counter names to collect (e.g., ["SQ_WAVES", "TCC_HIT"])
per_kernel : bool, optional
    If True, collect counters per kernel dispatch (default True)
callback : callable, optional
    Optional callback invoked when records are ready

Example
-------
>>> session = ProfilerSession(["SQ_WAVES"])
>>> session.start()
>>> # Run GPU kernels
>>> session.stop()
>>> records = session.get_records()
)doc")
        .def(py::init<const std::vector<std::string>&, bool, std::optional<py::function>>(),
             py::arg("metrics"),
             py::arg("per_kernel") = true,
             py::arg("callback")   = std::nullopt)
        .def("start", &ProfilerSession::start, "Start the profiling session")
        .def("stop", &ProfilerSession::stop, "Stop the profiling session")
        .def("is_active", &ProfilerSession::is_active, "Check if profiling is active")
        .def("get_records", &ProfilerSession::get_records, "Get collected counter records")
        .def("clear_records", &ProfilerSession::clear_records, "Clear collected records")
        .def(
            "__enter__",
            [](ProfilerSession& self) -> ProfilerSession& {
                self.start();
                return self;
            },
            "Context manager entry")
        .def(
            "__exit__",
            [](ProfilerSession& self, py::object exc_type, py::object exc_val, py::object exc_tb) {
                self.stop();
                return false;  // Don't suppress exceptions
            },
            "Context manager exit");

    // Module-level functions
    m.def("get_available_counters",
          &ProfilerSession::get_available_counters,
          py::arg("device_id") = std::nullopt,
          R"doc(
Get list of available hardware counters.

Parameters
----------
device_id : int, optional
    Device ID to query. If None, returns counters for all devices.

Returns
-------
list of CounterInfo
    List of available hardware counters
)doc");

    m.def("get_gpu_agents",
          &ProfilerSession::get_gpu_agents,
          R"doc(
Get list of available GPU agents (devices).

Returns
-------
list of AgentInfo
    List of available GPU devices
)doc");

    m.def(
        "is_available",
        []() { return ensure_rocprofiler_initialized(); },
        R"doc(
Check if rocprofiler-sdk is available and initialized.

Returns
-------
bool
    True if rocprofiler is available
)doc");
}
