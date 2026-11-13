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

#include "lib/rocprofiler-sdk-collector/schema.hpp"

#include "ctf_writer.hpp"
#include "scratch_dir.hpp"

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace
{
namespace collector = rocprofiler::collector;
namespace test      = rocprofiler::collector::test;
}  // namespace

TEST(schema, loads_a_probe_schema)
{
    auto dir = test::scratch_dir{};
    // As the generator writes it: flattened structs (with array indices in the path),
    // outputs, results, and keys the collector does not use.
    test::write_file(dir.path / "rocm_hip.json", R"({
  "schema": 1, "version": "7.14", "provider": "rocm_hip", "string_cap": 256,
  "events": [
    {"name": "hipLaunchKernel_enter", "api": "hipLaunchKernel", "phase": "enter",
     "fields": [
       {"name": "function_address", "role": "argument", "param": "function_address",
        "type": "const void *", "encoding": "Pointer", "capture": "scalars"},
       {"name": "numBlocks_x", "role": "argument", "param": "numBlocks", "path": ["x"],
        "type": "uint32_t", "param_type": "dim3", "capture": "scalars"},
       {"name": "numBlocks_y", "role": "argument", "param": "numBlocks", "path": ["y"],
        "type": "uint32_t", "param_type": "dim3", "capture": "scalars"},
       {"name": "pTexDesc_borderColor_0", "role": "argument", "param": "pTexDesc",
        "path": ["borderColor", 0], "type": "float", "capture": "full"},
       {"name": "stream", "role": "argument", "param": "stream", "type": "hipStream_t",
        "capture": "scalars"}]},
    {"name": "hipLaunchKernel_exit", "api": "hipLaunchKernel", "phase": "exit",
     "fields": [
       {"name": "result", "role": "result", "param": "", "type": "hipError_t",
        "capture": "none", "capped": true},
       {"name": "array", "role": "output", "param": "array", "type": "void *",
        "capture": "full", "on_success": true, "count": 16}]}
  ]
})");

    auto schema = collector::probe_schema{};
    auto error  = std::string{};
    ASSERT_TRUE(
        collector::load_probe_schema((dir.path / "rocm_hip.json").string(), &schema, &error))
        << error;
    EXPECT_EQ(schema.provider, "rocm_hip");
    EXPECT_EQ(schema.version, "7.14");
    ASSERT_EQ(schema.events.size(), 2u);

    const auto& enter = schema.events.at("hipLaunchKernel_enter");
    EXPECT_EQ(enter.api, "hipLaunchKernel");
    ASSERT_EQ(enter.fields.size(), 5u);
    EXPECT_EQ(enter.fields[1].param, "numBlocks");
    EXPECT_EQ(enter.fields[1].path, (std::vector<std::string>{"x"}));
    EXPECT_EQ(enter.fields[3].path, (std::vector<std::string>{"borderColor", "0"}));
    EXPECT_EQ(enter.fields[3].capture, "full");
    // Each argument parameter once, in order.
    EXPECT_EQ(enter.params,
              (std::vector<std::string>{"function_address", "numBlocks", "pTexDesc", "stream"}));

    const auto& exit = schema.events.at("hipLaunchKernel_exit");
    EXPECT_TRUE(exit.params.empty());
    ASSERT_NE(collector::find_schema_field(exit, "result"), nullptr);
    EXPECT_EQ(collector::find_schema_field(exit, "result")->role, "result");
    EXPECT_EQ(collector::find_schema_field(exit, "missing"), nullptr);
}

TEST(schema, reports_errors)
{
    auto dir    = test::scratch_dir{};
    auto schema = collector::probe_schema{};
    auto error  = std::string{};
    EXPECT_FALSE(collector::load_probe_schema((dir.path / "none.json").string(), &schema, &error));
    EXPECT_NE(error.find("cannot open"), std::string::npos) << error;

    test::write_file(dir.path / "bad.json", R"({"provider": "rocm_hip", "events": [{"name": 1}]})");
    EXPECT_FALSE(collector::load_probe_schema((dir.path / "bad.json").string(), &schema, &error));
    EXPECT_NE(error.find("bad.json"), std::string::npos) << error;

    test::write_file(dir.path / "truncated.json", R"({"provider": )");
    EXPECT_FALSE(
        collector::load_probe_schema((dir.path / "truncated.json").string(), &schema, &error));
}
