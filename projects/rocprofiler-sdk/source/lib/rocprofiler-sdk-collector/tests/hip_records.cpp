// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler-sdk-collector/hip_records.hpp"

#include "format_args.hpp"

#include <rocprofiler-sdk/buffer_tracing.h>
#include <rocprofiler-sdk/hip/compiler_api_id.h>
#include <rocprofiler-sdk/hip/runtime_api_id.h>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <tuple>
#include <vector>

namespace
{
namespace collector = rocprofiler::collector;
namespace ctf       = rocprofiler::collector::ctf;

using rocprofiler::collector::test::arg_row;
using rocprofiler::collector::test::format_args;

ctf::field_value
unsigned_value(const std::string& name, uint64_t value)
{
    auto result = ctf::field_value{};
    result.name = name;
    result.kind = ctf::value_kind::unsigned_integer;
    result.u64  = value;
    result.i64  = static_cast<int64_t>(value);
    return result;
}

ctf::field_value
signed_value(const std::string& name, int64_t value)
{
    auto result = unsigned_value(name, static_cast<uint64_t>(value));
    result.kind = ctf::value_kind::signed_integer;
    return result;
}

ctf::field_value
string_value(const std::string& name, const std::string& value)
{
    auto result = ctf::field_value{};
    result.name = name;
    result.kind = ctf::value_kind::string;
    result.str  = value;
    return result;
}

// A schema event: (field, param, member path) of each argument field.
collector::schema_event
schema_event(
    const std::string&                                                                 api,
    const std::vector<std::tuple<std::string, std::string, std::vector<std::string>>>& fields)
{
    auto event = collector::schema_event{};
    event.api  = api;
    for(const auto& [name, param, path] : fields)
    {
        event.fields.push_back({name, "argument", param, path, "scalars"});
        if(event.params.empty() || event.params.back() != param) event.params.push_back(param);
    }
    return event;
}

collector::api_call
call_of(const std::string& api, std::vector<ctf::field_value> arguments)
{
    auto call           = collector::api_call{};
    call.provider       = "rocm_hip";
    call.name           = api;
    call.pid            = 10;
    call.tid            = 11;
    call.correlation_id = 5;
    call.parent_id      = 2;
    call.start          = 1000;
    call.end            = 3000;
    call.arguments      = std::move(arguments);
    call.results        = {signed_value("result", 0), unsigned_value("exception", 0)};
    return call;
}

rocprofiler_buffer_tracing_hip_api_ext_record_t
make_record(const collector::api_call&     call,
            const collector::schema_event& enter,
            collector::string_store*       strings,
            uint32_t                       expected_missing = 0)
{
    auto record   = rocprofiler_buffer_tracing_hip_api_ext_record_t{};
    auto clocks   = collector::clock_conversion{};
    clocks.offset = 500;
    clocks.stable = true;
    auto status   = collector::make_hip_api_record(call, enter, nullptr, clocks, strings, &record);
    EXPECT_TRUE(status.known_api);
    EXPECT_EQ(status.missing_args, expected_missing);
    return record;
}

// The record rocprofiler-sdk would write in process, with only the args to fill.
rocprofiler_buffer_tracing_hip_api_ext_record_t
reference_record(rocprofiler_buffer_tracing_kind_t kind, rocprofiler_tracing_operation_t op)
{
    auto record = rocprofiler_buffer_tracing_hip_api_ext_record_t{};
    std::memset(static_cast<void*>(&record), 0, sizeof(record));
    record.size      = sizeof(record);
    record.kind      = kind;
    record.operation = op;
    return record;
}
}  // namespace

TEST(hip_records, finds_rocprofiler_sdk_operations)
{
    auto kind      = rocprofiler_buffer_tracing_kind_t{};
    auto operation = rocprofiler_tracing_operation_t{};
    ASSERT_TRUE(collector::find_hip_operation("hipMemcpyAsync", &kind, &operation));
    EXPECT_EQ(kind, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT);
    EXPECT_EQ(operation, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync);
    ASSERT_TRUE(collector::find_hip_operation("__hipRegisterFunction", &kind, &operation));
    EXPECT_EQ(kind, ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT);
    EXPECT_EQ(operation, ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFunction);
    EXPECT_FALSE(collector::find_hip_operation("hipNotAnApi", &kind, &operation));
}

TEST(hip_records, rebuilds_the_record_rocprofv3_writes)
{
    auto strings = collector::string_store{};
    auto enter   = schema_event("hipMemcpyAsync",
                              {{"dst", "dst", {}},
                               {"src", "src", {}},
                               {"sizeBytes", "sizeBytes", {}},
                               {"kind", "kind", {}},
                               {"stream", "stream", {}}});
    auto call    = call_of("hipMemcpyAsync",
                        {unsigned_value("dst", 0x7f3235600000),
                         unsigned_value("src", 0x7f3355418010),
                         unsigned_value("sizeBytes", 4194304),
                         signed_value("kind", hipMemcpyHostToDevice),
                         unsigned_value("stream", 0)});
    auto record  = make_record(call, enter, &strings);

    EXPECT_EQ(record.size, sizeof(record));
    EXPECT_EQ(record.kind, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT);
    EXPECT_EQ(record.operation, ROCPROFILER_HIP_RUNTIME_API_ID_hipMemcpyAsync);
    EXPECT_EQ(record.correlation_id.internal, 5u);
    EXPECT_EQ(record.correlation_id.ancestor, 2u);
    EXPECT_EQ(record.thread_id, 11u);
    EXPECT_EQ(record.start_timestamp, 1500u);  // CLOCK_MONOTONIC + offset
    EXPECT_EQ(record.end_timestamp, 3500u);
    EXPECT_EQ(record.retval.hipError_t_retval, hipSuccess);

    // The rows of rocprofv3's rocpd_arg table for this call.
    EXPECT_EQ(format_args(record),
              (std::vector<arg_row>{{0, "void*", "dst", "0x7f3235600000"},
                                    {1, "void const*", "src", "0x7f3355418010"},
                                    {2, "unsigned long", "sizeBytes", "4194304"},
                                    {3, "hipMemcpyKind", "kind", "HostToDevice"},
                                    {4, "ihipStream_t*", "stream", "0"}}));
}

TEST(hip_records, formats_like_an_in_process_record)
{
    auto strings = collector::string_store{};

    // dim3 arguments: the probe flattens them, rocprofiler-sdk stores rocprofiler_dim3_t.
    auto  launch          = schema_event("hipLaunchKernel",
                               {{"function_address", "function_address", {}},
                                {"numBlocks_x", "numBlocks", {"x"}},
                                {"numBlocks_y", "numBlocks", {"y"}},
                                {"numBlocks_z", "numBlocks", {"z"}},
                                {"dimBlocks_x", "dimBlocks", {"x"}},
                                {"dimBlocks_y", "dimBlocks", {"y"}},
                                {"dimBlocks_z", "dimBlocks", {"z"}},
                                {"args", "args", {}},
                                {"sharedMemBytes", "sharedMemBytes", {}},
                                {"stream", "stream", {}}});
    auto  record          = make_record(call_of("hipLaunchKernel",
                                      {unsigned_value("function_address", 0x55a44b400000),
                                       unsigned_value("numBlocks_x", 256),
                                       unsigned_value("numBlocks_y", 1),
                                       unsigned_value("numBlocks_z", 1),
                                       unsigned_value("dimBlocks_x", 64),
                                       unsigned_value("dimBlocks_y", 2),
                                       unsigned_value("dimBlocks_z", 1),
                                       unsigned_value("args", 0x7fff65470460),
                                       unsigned_value("sharedMemBytes", 0),
                                       unsigned_value("stream", 0x55a44b4332b0)}),
                              launch,
                              &strings);
    auto  reference       = reference_record(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                                      ROCPROFILER_HIP_RUNTIME_API_ID_hipLaunchKernel);
    auto& args            = reference.args.hipLaunchKernel;
    args.function_address = reinterpret_cast<const void*>(0x55a44b400000);
    args.numBlocks        = rocprofiler_dim3_t{256, 1, 1};
    args.dimBlocks        = rocprofiler_dim3_t{64, 2, 1};
    args.args             = reinterpret_cast<void**>(0x7fff65470460);
    args.sharedMemBytes   = 0;
    args.stream           = reinterpret_cast<hipStream_t>(0x55a44b4332b0);
    auto rows             = format_args(record);
    EXPECT_EQ(rows, format_args(reference));
    ASSERT_EQ(rows.size(), 6u);
    EXPECT_EQ(std::get<3>(rows[1]), "{z=1, y=1, x=256}");

    // By-value structs other than dim3.
    auto memset3d = schema_event("hipMemset3D",
                                 {{"pitchedDevPtr_ptr", "pitchedDevPtr", {"ptr"}},
                                  {"pitchedDevPtr_pitch", "pitchedDevPtr", {"pitch"}},
                                  {"pitchedDevPtr_xsize", "pitchedDevPtr", {"xsize"}},
                                  {"pitchedDevPtr_ysize", "pitchedDevPtr", {"ysize"}},
                                  {"value", "value", {}},
                                  {"extent_width", "extent", {"width"}},
                                  {"extent_height", "extent", {"height"}},
                                  {"extent_depth", "extent", {"depth"}}});
    record        = make_record(call_of("hipMemset3D",
                                 {unsigned_value("pitchedDevPtr_ptr", 0x7f0000001000),
                                  unsigned_value("pitchedDevPtr_pitch", 512),
                                  unsigned_value("pitchedDevPtr_xsize", 400),
                                  unsigned_value("pitchedDevPtr_ysize", 300),
                                  signed_value("value", -1),
                                  unsigned_value("extent_width", 400),
                                  unsigned_value("extent_height", 300),
                                  unsigned_value("extent_depth", 2)}),
                         memset3d,
                         &strings);
    reference     = reference_record(ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API_EXT,
                                 ROCPROFILER_HIP_RUNTIME_API_ID_hipMemset3D);
    reference.args.hipMemset3D.pitchedDevPtr =
        hipPitchedPtr{reinterpret_cast<void*>(0x7f0000001000), 512, 400, 300};
    reference.args.hipMemset3D.value  = -1;
    reference.args.hipMemset3D.extent = hipExtent{400, 300, 2};
    EXPECT_EQ(format_args(record), format_args(reference));

    // Strings point to the collector's copies; the probe names tid "arg_tid".
    auto reg                = schema_event("__hipRegisterFunction",
                            {{"modules", "modules", {}},
                             {"hostFunction", "hostFunction", {}},
                             {"deviceFunction", "deviceFunction", {}},
                             {"deviceName", "deviceName", {}},
                             {"threadLimit", "threadLimit", {}},
                             {"arg_tid", "tid", {}},
                             {"bid", "bid", {}},
                             {"blockDim", "blockDim", {}},
                             {"gridDim", "gridDim", {}},
                             {"wSize", "wSize", {}}});
    record                  = make_record(call_of("__hipRegisterFunction",
                                 {unsigned_value("modules", 0x5643412c1f60),
                                  unsigned_value("hostFunction", 0x559ca3232ed8),
                                  string_value("deviceFunction", "_Z4vaddPKfS0_Pfi"),
                                  string_value("deviceName", "_Z4vaddPKfS0_Pfi"),
                                  unsigned_value("threadLimit", 4294967295),
                                  unsigned_value("arg_tid", 0),
                                  unsigned_value("bid", 0),
                                  unsigned_value("blockDim", 0),
                                  unsigned_value("gridDim", 0x7fff65470460),
                                  unsigned_value("wSize", 0)}),
                         reg,
                         &strings);
    reference               = reference_record(ROCPROFILER_BUFFER_TRACING_HIP_COMPILER_API_EXT,
                                 ROCPROFILER_HIP_COMPILER_API_ID___hipRegisterFunction);
    auto& reg_args          = reference.args.__hipRegisterFunction;
    auto  name              = std::string{"_Z4vaddPKfS0_Pfi"};
    reg_args.modules        = reinterpret_cast<void**>(0x5643412c1f60);
    reg_args.hostFunction   = reinterpret_cast<const void*>(0x559ca3232ed8);
    reg_args.deviceFunction = name.data();
    reg_args.deviceName     = name.c_str();
    reg_args.threadLimit    = 4294967295;
    reg_args.gridDim        = reinterpret_cast<dim3*>(0x7fff65470460);
    rows                    = format_args(record);
    EXPECT_EQ(rows, format_args(reference));
    EXPECT_EQ(std::get<3>(rows[3]), "_Z4vaddPKfS0_Pfi");
    EXPECT_EQ(std::get<3>(rows[5]), "(null)");  // a null pointer to a complete type
    EXPECT_NE(record.args.__hipRegisterFunction.deviceName, name.c_str());  // a copy
}

TEST(hip_records, matches_renamed_parameters_by_position)
{
    // The probe's header names the parameter "error"; rocprofiler-sdk names it "hipError".
    auto strings = collector::string_store{};
    auto enter   = schema_event("hipGetErrorString", {{"error", "error", {}}});
    auto call    = call_of("hipGetErrorString", {signed_value("error", hipErrorInvalidValue)});
    call.results = {string_value("result", "invalid argument"), unsigned_value("exception", 0)};
    auto record  = make_record(call, enter, &strings);
    EXPECT_EQ(record.args.hipGetErrorString.hipError, hipErrorInvalidValue);
    ASSERT_NE(record.retval.const_charp_retval, nullptr);
    EXPECT_STREQ(record.retval.const_charp_retval, "invalid argument");
}

TEST(hip_records, reports_what_it_cannot_fill)
{
    auto strings = collector::string_store{};
    auto enter   = schema_event("hipMemcpyAsync",
                              {{"dst", "dst", {}},
                               {"src", "src", {}},
                               {"sizeBytes", "sizeBytes", {}},
                               {"kind", "kind", {}},
                               {"stream", "stream", {}}});

    // A field the trace does not have is left zero and counted.
    auto call   = call_of("hipMemcpyAsync",
                        {unsigned_value("dst", 1),
                         unsigned_value("src", 2),
                         unsigned_value("sizeBytes", 3),
                         unsigned_value("stream", 4)});
    auto record = make_record(call, enter, &strings, 1);
    EXPECT_EQ(record.args.hipMemcpyAsync.kind, hipMemcpyHostToHost);

    // No exit: the end is the start.
    call.end = collector::no_end;
    call.results.clear();
    call.arguments.push_back(signed_value("kind", hipMemcpyDeviceToHost));
    record = make_record(call, enter, &strings);
    EXPECT_EQ(record.end_timestamp, record.start_timestamp);

    // An API rocprofiler-sdk does not trace.
    auto unknown = rocprofiler_buffer_tracing_hip_api_ext_record_t{};
    auto status  = collector::make_hip_api_record(call_of("hipNotAnApi", {}),
                                                 schema_event("hipNotAnApi", {}),
                                                 nullptr,
                                                 collector::clock_conversion{},
                                                 &strings,
                                                 &unknown);
    EXPECT_FALSE(status.known_api);
}

TEST(hip_records, keeps_one_copy_of_each_string)
{
    auto        strings = collector::string_store{};
    const auto* first   = collector::intern_string(&strings, "kernel");
    const auto* second  = collector::intern_string(&strings, std::string{"kern"} + "el");
    EXPECT_EQ(first, second);
    EXPECT_STREQ(first, "kernel");
    EXPECT_EQ(strings.values.size(), 1u);
}
