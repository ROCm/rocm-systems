// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "libpyroctx.hpp"

#include <rocprofiler-sdk-roctx/roctx.h>
#include <rocprofiler-sdk-roctx/types.h>

namespace py = ::pybind11;

PYBIND11_MODULE(libpyroctx, pyroctx)
{
    py::doc("Rocprofiler-SDK ROCTx Python bindings");

    pyroctx.def(
        "roctxMark",
        [](const std::string& _msg) { roctxMarkA(_msg.c_str()); },
        "Mark an event in any attached profiler");

    pyroctx.def(
        "roctxProfilerPause",
        [](roctx_thread_id_t tid) { return roctxProfilerPause(tid); },
        "Pause data collection in any attached profiler");

    pyroctx.def(
        "roctxProfilerResume",
        [](roctx_thread_id_t tid) { return roctxProfilerResume(tid); },
        "Resume data collection in any attached profiler");

    pyroctx.def(
        "roctxGetThreadId",
        []() {
            auto _tid = roctx_thread_id_t{0};
            roctxGetThreadId(&_tid);
            return _tid;
        },
        "Get the current thread ID");

    pyroctx.def(
        "roctxRangePush",
        [](const std::string& _msg) { return roctxRangePushA(_msg.c_str()); },
        "Start a new nested range");

    pyroctx.def(
        "roctxRangePop", []() { return roctxRangePop(); }, "Stop the current nested range");

    pyroctx.def(
        "roctxRangeStart",
        [](const std::string& _msg) { return roctxRangeStartA(_msg.c_str()); },
        "Start a process range");

    pyroctx.def(
        "roctxRangeStop", [](roctx_range_id_t id) { roctxRangeStop(id); }, "Stop a process range");

    pyroctx.def(
        "roctxNameOsThread",
        [](const std::string& name) { return roctxNameOsThread(name.c_str()); },
        "Label the current CPU OS thread with the provided name");

    // pyroctx.def(
    //     "roctxNameHsaAgent",
    //     [](const std::string& name, const struct hsa_agent_s* agent) { return
    //     roctxNameHsaAgent(name.c_str(), agent); }, "Label the given HSA agent with the provided
    //     name");

    pyroctx.def(
        "roctxNameHipDevice",
        [](const std::string& name, int device_id) {
            return roctxNameHipDevice(name.c_str(), device_id);
        },
        "Label the given HIP device id with the provided name");

    // pyroctx.def(
    //     "roctxNameHipStream",
    //     [](const std::string& name, const struct ihipStream_t* stream) { return
    //     roctxNameHipStream(name.c_str(), stream); }, "Label the given HIP stream with the
    //     provided name");
}
