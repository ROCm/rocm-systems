// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "json_serializers.hpp"

#include <gtest/gtest.h>
#include <nlohmann/json.hpp>

#include <array>
#include <string>

namespace
{

using namespace profiler_hub;
using namespace profiler_hub::shared_types;
using json = nlohmann::json;

// --------------------- Call Stack Tests ---------------------

TEST(json_serializer_test, empty_call_stack)
{
    call_stack_t empty_stack;
    std::string  result = json_serializers::serialize_call_stack(empty_stack);
    EXPECT_EQ(result, "{}");
}

TEST(json_serializer_test, call_stack_with_single_frame)
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

TEST(json_serializer_test, call_stack_with_multiple_frames)
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

TEST(json_serializer_test, call_stack_with_optional_fields)
{
    call_stack_t stack;

    // Frame with only program counter (no address range)
    {
        program_counter_info_t pc_info;
        pc_info.function    = "test_func";
        pc_info.filename    = std::nullopt;  // No filename
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

TEST(json_serializer_test, call_stack_with_extdata)
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

TEST(json_serializer_test, empty_source_context)
{
    source_context_list_t empty_list;
    std::string           result = json_serializers::serialize_source_context(empty_list);
    EXPECT_EQ(result, "{}");
}

TEST(json_serializer_test, source_context_with_program_counter)
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

TEST(json_serializer_test, source_context_with_source_code)
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

TEST(json_serializer_test, source_context_with_multiple_entries)
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

TEST(json_serializer_test, source_context_with_complete_entry)
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

TEST(json_serializer_test, source_context_with_null_pointers)
{
    source_context_list_t list;

    source_code_info_t source_code;
    source_code.filename             = std::nullopt;              // No filename
    source_code.starting_line_number = std::nullopt;              // No line number
    source_code.source_code_lines    = { "", "valid line", "" };  // Mixed empty/valid
    source_code.assembly_instruction_lines = {};                  // Empty
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

// ============================================================================
// Deserialization tests
// ============================================================================

TEST(json_serializer_test, deserialize_call_stack_empty_json)
{
    auto result = json_serializers::deserialize_call_stack("{}");
    ASSERT_TRUE(result.empty());
}

TEST(json_serializer_test, deserialize_call_stack_empty_string)
{
    auto result = json_serializers::deserialize_call_stack("");
    ASSERT_TRUE(result.empty());
}

TEST(json_serializer_test, deserialize_call_stack_malformed_json)
{
    auto result = json_serializers::deserialize_call_stack("not json");
    ASSERT_TRUE(result.empty());
}

TEST(json_serializer_test, deserialize_call_stack_single_frame)
{
    // Build a call stack, serialize, then deserialize and compare
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

    std::string serialized   = json_serializers::serialize_call_stack(stack);
    auto        deserialized = json_serializers::deserialize_call_stack(serialized);

    ASSERT_EQ(deserialized.size(), 1);
    ASSERT_TRUE(deserialized[0].program_counter.has_value());
    EXPECT_EQ(deserialized[0].program_counter->function, "main");
    EXPECT_EQ(deserialized[0].program_counter->filename, "/path/to/main.cpp");
    ASSERT_TRUE(deserialized[0].program_counter->line_number.has_value());
    EXPECT_EQ(deserialized[0].program_counter->line_number.value(), 42);

    ASSERT_TRUE(deserialized[0].address_range.has_value());
    EXPECT_EQ(deserialized[0].address_range->address_base, 0x1000);
    EXPECT_EQ(deserialized[0].address_range->address_low, 0x1000);
    EXPECT_EQ(deserialized[0].address_range->address_high, 0x2000);
}

TEST(json_serializer_test, deserialize_call_stack_multiple_frames)
{
    // Strings must outlive the call_stack since stack_frame_t uses const char*
    std::array<std::string, 3> func_names = { "func_0", "func_1", "func_2" };

    call_stack_t stack;
    for(int i = 0; i < 3; ++i)
    {
        program_counter_info_t pc_info;
        pc_info.function    = func_names[i].c_str();
        pc_info.filename    = "";
        pc_info.line_number = static_cast<size_t>(i * 10);
        pc_info.extdata     = "";

        stack_frame_t frame;
        frame.program_counter = pc_info;
        frame.extdata         = "";
        stack.push_back(frame);
    }

    std::string serialized   = json_serializers::serialize_call_stack(stack);
    auto        deserialized = json_serializers::deserialize_call_stack(serialized);

    ASSERT_EQ(deserialized.size(), 3);
    EXPECT_EQ(deserialized[0].program_counter->function, "func_0");
    EXPECT_EQ(deserialized[1].program_counter->function, "func_1");
    EXPECT_EQ(deserialized[2].program_counter->function, "func_2");
}

TEST(json_serializer_test, deserialize_source_context_empty_json)
{
    auto result = json_serializers::deserialize_source_context("{}");
    ASSERT_TRUE(result.empty());
}

TEST(json_serializer_test, deserialize_source_context_empty_string)
{
    auto result = json_serializers::deserialize_source_context("");
    ASSERT_TRUE(result.empty());
}

TEST(json_serializer_test, deserialize_source_context_with_source_code)
{
    source_context_list_t list;

    source_code_info_t source_code;
    source_code.filename                   = "/path/to/source.cpp";
    source_code.starting_line_number       = 10;
    source_code.source_code_lines          = { "line 10 content", "line 11 content" };
    source_code.assembly_instruction_lines = { "mov rax, rbx", "ret" };
    source_code.extdata                    = "";

    program_counter_info_t pc_info;
    pc_info.function    = "test_func";
    pc_info.filename    = "/path/to/source.cpp";
    pc_info.line_number = 10;
    pc_info.extdata     = "";

    line_info_entry_t entry;
    entry.source_code     = source_code;
    entry.program_counter = pc_info;

    list.push_back(entry);

    std::string serialized   = json_serializers::serialize_source_context(list);
    auto        deserialized = json_serializers::deserialize_source_context(serialized);

    ASSERT_EQ(deserialized.size(), 1);

    ASSERT_TRUE(deserialized[0].source_code.has_value());
    ASSERT_TRUE(deserialized[0].source_code->filename.has_value());
    EXPECT_EQ(deserialized[0].source_code->filename.value(), "/path/to/source.cpp");
    ASSERT_TRUE(deserialized[0].source_code->starting_line_number.has_value());
    EXPECT_EQ(deserialized[0].source_code->starting_line_number.value(), 10);
    ASSERT_EQ(deserialized[0].source_code->source_code_lines.size(), 2);
    EXPECT_EQ(deserialized[0].source_code->source_code_lines[0], "line 10 content");
    ASSERT_EQ(deserialized[0].source_code->assembly_instruction_lines.size(), 2);
    EXPECT_EQ(deserialized[0].source_code->assembly_instruction_lines[0], "mov rax, rbx");

    ASSERT_TRUE(deserialized[0].program_counter.has_value());
    EXPECT_EQ(deserialized[0].program_counter->function, "test_func");
}

}  // namespace
