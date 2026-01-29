// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "json_serializers.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <string>

namespace
{

using namespace rocstorage;
using namespace rocstorage::data_types;
using json = nlohmann::json;

// --------------------- Call Stack Tests ---------------------

TEST(JsonSerializerTest, EmptyCallStack)
{
    call_stack_t empty_stack;
    std::string  result = json_serializers::serialize_call_stack(empty_stack);
    EXPECT_EQ(result, "{}");
}

TEST(JsonSerializerTest, CallStackWithSingleFrame)
{
    call_stack_t stack;

    program_counter_info_t pc_info;
    pc_info.function    = "main";
    pc_info.filename    = "/path/to/main.cpp";
    pc_info.line_number = 42;
    pc_info.extdata     = "";

    address_range_info_t addr_range;
    addr_range.address_base = 0x1000;
    addr_range.address_low  = 0x1000;
    addr_range.address_high = 0x2000;
    addr_range.extdata      = "";

    stack_frame_t frame;
    frame.program_counter = pc_info;
    frame.address_range   = addr_range;
    frame.extdata         = "";

    stack.push_back(frame);

    std::string result = json_serializers::serialize_call_stack(stack);

    // Parse JSON to verify structure
    json j = json::parse(result);
    ASSERT_TRUE(j.contains("frames"));
    ASSERT_TRUE(j["frames"].is_array());
    ASSERT_EQ(j["frames"].size(), 1);

    auto& frame_json = j["frames"][0];
    ASSERT_TRUE(frame_json.contains("program_counter"));
    EXPECT_EQ(frame_json["program_counter"]["function"], "main");
    EXPECT_EQ(frame_json["program_counter"]["filename"], "/path/to/main.cpp");
    EXPECT_EQ(frame_json["program_counter"]["line_number"], 42);

    ASSERT_TRUE(frame_json.contains("address_range"));
    EXPECT_EQ(frame_json["address_range"]["address_base"], 0x1000);
    EXPECT_EQ(frame_json["address_range"]["address_low"], 0x1000);
    EXPECT_EQ(frame_json["address_range"]["address_high"], 0x2000);
}

TEST(JsonSerializerTest, CallStackWithMultipleFrames)
{
    call_stack_t stack;

    // Frame 1
    {
        program_counter_info_t pc_info;
        pc_info.function    = "foo";
        pc_info.filename    = "/path/to/foo.cpp";
        pc_info.line_number = 10;
        pc_info.extdata     = "";

        stack_frame_t frame;
        frame.program_counter = pc_info;
        frame.extdata         = "";

        stack.push_back(frame);
    }

    // Frame 2
    {
        program_counter_info_t pc_info;
        pc_info.function    = "bar";
        pc_info.filename    = "/path/to/bar.cpp";
        pc_info.line_number = 20;
        pc_info.extdata     = "";

        stack_frame_t frame;
        frame.program_counter = pc_info;
        frame.extdata         = "";

        stack.push_back(frame);
    }

    std::string result = json_serializers::serialize_call_stack(stack);

    json j = json::parse(result);
    ASSERT_EQ(j["frames"].size(), 2);
    EXPECT_EQ(j["frames"][0]["program_counter"]["function"], "foo");
    EXPECT_EQ(j["frames"][1]["program_counter"]["function"], "bar");
}

TEST(JsonSerializerTest, CallStackWithOptionalFields)
{
    call_stack_t stack;

    // Frame with only program counter (no address range)
    {
        program_counter_info_t pc_info;
        pc_info.function    = "test_func";
        pc_info.filename    = nullptr;       // No filename
        pc_info.line_number = std::nullopt;  // No line number
        pc_info.extdata     = "";

        stack_frame_t frame;
        frame.program_counter = pc_info;
        frame.address_range   = std::nullopt;  // No address range
        frame.extdata         = "";

        stack.push_back(frame);
    }

    std::string result = json_serializers::serialize_call_stack(stack);

    json j = json::parse(result);
    ASSERT_EQ(j["frames"].size(), 1);

    auto& frame_json = j["frames"][0];
    ASSERT_TRUE(frame_json.contains("program_counter"));
    EXPECT_EQ(frame_json["program_counter"]["function"], "test_func");
    EXPECT_FALSE(frame_json["program_counter"].contains("filename"));
    EXPECT_FALSE(frame_json["program_counter"].contains("line_number"));
    EXPECT_FALSE(frame_json.contains("address_range"));
}

TEST(JsonSerializerTest, CallStackWithExtdata)
{
    call_stack_t stack;

    program_counter_info_t pc_info;
    pc_info.function    = "main";
    pc_info.filename    = "/main.cpp";
    pc_info.line_number = 1;
    pc_info.extdata     = "{\"custom\":\"data\"}";

    stack_frame_t frame;
    frame.program_counter = pc_info;
    frame.extdata         = "{\"frame\":\"info\"}";

    stack.push_back(frame);

    std::string result = json_serializers::serialize_call_stack(stack);

    json j = json::parse(result);
    EXPECT_EQ(j["frames"][0]["program_counter"]["extdata"], "{\"custom\":\"data\"}");
    EXPECT_EQ(j["frames"][0]["extdata"], "{\"frame\":\"info\"}");
}

// --------------------- Source Context Tests ---------------------

TEST(JsonSerializerTest, EmptySourceContext)
{
    source_context_list_t empty_list;
    std::string           result = json_serializers::serialize_source_context(empty_list);
    EXPECT_EQ(result, "{}");
}

TEST(JsonSerializerTest, SourceContextWithProgramCounter)
{
    source_context_list_t list;

    program_counter_info_t pc_info;
    pc_info.function    = "kernel_func";
    pc_info.filename    = "/path/to/kernel.cpp";
    pc_info.line_number = 100;
    pc_info.extdata     = "";

    line_info_entry_t entry;
    entry.program_counter = pc_info;
    entry.source_code     = std::nullopt;
    entry.address_range   = std::nullopt;

    list.push_back(entry);

    std::string result = json_serializers::serialize_source_context(list);

    json j = json::parse(result);
    ASSERT_TRUE(j.contains("entries"));
    ASSERT_EQ(j["entries"].size(), 1);

    auto& entry_json = j["entries"][0];
    ASSERT_TRUE(entry_json.contains("program_counter"));
    EXPECT_EQ(entry_json["program_counter"]["function"], "kernel_func");
    EXPECT_EQ(entry_json["program_counter"]["filename"], "/path/to/kernel.cpp");
    EXPECT_EQ(entry_json["program_counter"]["line_number"], 100);
}

TEST(JsonSerializerTest, SourceContextWithSourceCode)
{
    source_context_list_t list;

    source_code_info_t source_code;
    source_code.filename                   = "/path/to/source.cpp";
    source_code.starting_line_number       = 10;
    source_code.source_code_lines          = { "line 10 content",
                                               "line 11 content",
                                               "line 12 content" };
    source_code.assembly_instruction_lines = { "mov rax, rbx", "add rax, 1", "ret" };
    source_code.extdata                    = "";

    line_info_entry_t entry;
    entry.source_code = source_code;

    list.push_back(entry);

    std::string result = json_serializers::serialize_source_context(list);

    json j = json::parse(result);
    ASSERT_EQ(j["entries"].size(), 1);

    auto& entry_json = j["entries"][0];
    ASSERT_TRUE(entry_json.contains("source_code"));

    auto& sc_json = entry_json["source_code"];
    EXPECT_EQ(sc_json["filename"], "/path/to/source.cpp");
    EXPECT_EQ(sc_json["starting_line_number"], 10);

    ASSERT_TRUE(sc_json.contains("source_code_lines"));
    ASSERT_EQ(sc_json["source_code_lines"].size(), 3);
    EXPECT_EQ(sc_json["source_code_lines"][0], "line 10 content");
    EXPECT_EQ(sc_json["source_code_lines"][1], "line 11 content");
    EXPECT_EQ(sc_json["source_code_lines"][2], "line 12 content");

    ASSERT_TRUE(sc_json.contains("assembly_instruction_lines"));
    ASSERT_EQ(sc_json["assembly_instruction_lines"].size(), 3);
    EXPECT_EQ(sc_json["assembly_instruction_lines"][0], "mov rax, rbx");
    EXPECT_EQ(sc_json["assembly_instruction_lines"][1], "add rax, 1");
    EXPECT_EQ(sc_json["assembly_instruction_lines"][2], "ret");
}

TEST(JsonSerializerTest, SourceContextWithMultipleEntries)
{
    source_context_list_t list;

    // Entry 1
    {
        program_counter_info_t pc_info;
        pc_info.function    = "func1";
        pc_info.filename    = "/file1.cpp";
        pc_info.line_number = 50;
        pc_info.extdata     = "";

        line_info_entry_t entry;
        entry.program_counter = pc_info;

        list.push_back(entry);
    }

    // Entry 2
    {
        program_counter_info_t pc_info;
        pc_info.function    = "func2";
        pc_info.filename    = "/file2.cpp";
        pc_info.line_number = 60;
        pc_info.extdata     = "";

        line_info_entry_t entry;
        entry.program_counter = pc_info;

        list.push_back(entry);
    }

    std::string result = json_serializers::serialize_source_context(list);

    json j = json::parse(result);
    ASSERT_EQ(j["entries"].size(), 2);
    EXPECT_EQ(j["entries"][0]["program_counter"]["function"], "func1");
    EXPECT_EQ(j["entries"][1]["program_counter"]["function"], "func2");
}

TEST(JsonSerializerTest, SourceContextWithCompleteEntry)
{
    source_context_list_t list;

    // Create complete entry with all fields
    source_code_info_t source_code;
    source_code.filename                   = "/complete.cpp";
    source_code.starting_line_number       = 1;
    source_code.source_code_lines          = { "int main() {" };
    source_code.assembly_instruction_lines = { "push rbp" };
    source_code.extdata                    = "";

    program_counter_info_t pc_info;
    pc_info.function    = "main";
    pc_info.filename    = "/complete.cpp";
    pc_info.line_number = 1;
    pc_info.extdata     = "";

    address_range_info_t addr_range;
    addr_range.address_base = 0x4000;
    addr_range.address_low  = 0x4000;
    addr_range.address_high = 0x5000;
    addr_range.extdata      = "";

    line_info_entry_t entry;
    entry.source_code     = source_code;
    entry.program_counter = pc_info;
    entry.address_range   = addr_range;

    list.push_back(entry);

    std::string result = json_serializers::serialize_source_context(list);

    json j = json::parse(result);
    ASSERT_EQ(j["entries"].size(), 1);

    auto& entry_json = j["entries"][0];
    EXPECT_TRUE(entry_json.contains("source_code"));
    EXPECT_TRUE(entry_json.contains("program_counter"));
    EXPECT_TRUE(entry_json.contains("address_range"));
}

TEST(JsonSerializerTest, SourceContextWithNullPointers)
{
    source_context_list_t list;

    source_code_info_t source_code;
    source_code.filename                   = std::nullopt;  // No filename
    source_code.starting_line_number       = std::nullopt;  // No line number
    source_code.source_code_lines          = { nullptr,
                                               "valid line",
                                               nullptr };  // Mixed null/valid
    source_code.assembly_instruction_lines = {};           // Empty
    source_code.extdata                    = "";

    line_info_entry_t entry;
    entry.source_code = source_code;

    list.push_back(entry);

    std::string result = json_serializers::serialize_source_context(list);

    json j = json::parse(result);
    ASSERT_EQ(j["entries"].size(), 1);

    auto& entry_json = j["entries"][0];
    ASSERT_TRUE(entry_json.contains("source_code"));

    auto& sc_json = entry_json["source_code"];
    EXPECT_FALSE(sc_json.contains("filename"));
    EXPECT_FALSE(sc_json.contains("starting_line_number"));

    // Should only have one valid line (null pointers filtered out)
    if(sc_json.contains("source_code_lines"))
    {
        EXPECT_EQ(sc_json["source_code_lines"].size(), 1);
        EXPECT_EQ(sc_json["source_code_lines"][0], "valid line");
    }

    EXPECT_FALSE(sc_json.contains("assembly_instruction_lines"));
}

}  // namespace
