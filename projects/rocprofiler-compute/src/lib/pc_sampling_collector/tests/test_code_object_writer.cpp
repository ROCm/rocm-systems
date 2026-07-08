// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_code_object_writer.h"

#include "nlohmann/json.hpp"

#include <string>
#include <string_view>

namespace
{
template<typename Operation>
void expect_runtime_throw(Operation operation)
{
    EXPECT_THROW(operation(), std::runtime_error);
}

void expect_kernel_json(const nlohmann::json& kernel, uint64_t id, std::string_view name)
{
    EXPECT_EQ(kernel["kernel_id"].get<uint64_t>(), id);
    EXPECT_EQ(kernel["name"].get<std::string>(), std::string{name});
}

void expect_instruction_json(const nlohmann::json&                          instruction,
                             const rocprofiler_compute_tool::instruction_t& expected)
{
    EXPECT_EQ(instruction["name"], expected.name);
    EXPECT_EQ(instruction["comment"], expected.comment);
    EXPECT_EQ(instruction["virtual_address"], expected.virtual_address);
    EXPECT_EQ(instruction["code_obj_offset"], expected.code_obj_offset);
    EXPECT_EQ(instruction["size"], expected.size);
}
}  // namespace

TEST_F(test_code_object_writer_t, ProvidedNoData_ReturnsMinimalJson)
{
    const auto& result = m_writer.get_result();
    EXPECT_FALSE(result.empty());
    EXPECT_TRUE(nlohmann::json::accept(result));
}

TEST_F(test_code_object_writer_t, ProvidedNoCodeObjects_IsEmpty)
{
    EXPECT_TRUE(m_writer.empty());
}

TEST_F(test_code_object_writer_t, ProvidedCodeObject_IsNotEmpty)
{
    m_writer.start_code_obj(0);
    m_writer.end_code_obj();
    EXPECT_FALSE(m_writer.empty());
}

TEST_F(test_code_object_writer_t, ProvidedStartCodeObjWithoutEnd_Throws)
{
    m_writer.start_code_obj(0);
    expect_runtime_throw([&]() { m_writer.get_result(); });
}

TEST_F(test_code_object_writer_t, ProvidedEndCodeObjWithoutStart_Throws)
{
    expect_runtime_throw([&]() { m_writer.end_code_obj(); });
}

TEST_F(test_code_object_writer_t, ProvidedTwoStartCodeObjCalls_Throws)
{
    m_writer.start_code_obj(0);
    expect_runtime_throw([&]() { m_writer.start_code_obj(1); });
}

TEST_F(test_code_object_writer_t, ProvidedTwoEndCodeObjCalls_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.end_code_obj();
    expect_runtime_throw([&]() { m_writer.end_code_obj(); });
}

TEST_F(test_code_object_writer_t, ProvidedStartSymbolWithoutEnd_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{});
    expect_runtime_throw([&]() { m_writer.get_result(); });
}

TEST_F(test_code_object_writer_t, ProvidedEndSymbolWithoutStart_Throws)
{
    m_writer.start_code_obj(0);
    expect_runtime_throw([&]() { m_writer.end_symbol(); });
}

TEST_F(test_code_object_writer_t, ProvidedTwoStartSymbolCalls_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{});
    expect_runtime_throw([&]() { m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedTwoEndSymbolCalls_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{});
    m_writer.end_symbol();
    expect_runtime_throw([&]() { m_writer.end_symbol(); });
}

TEST_F(test_code_object_writer_t, ProvidedSymbolCallWithoutStartCodeObj_Throws)
{
    expect_runtime_throw([&]() { m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteInstructionWithoutAnyScope_Throws)
{
    expect_runtime_throw([&]()
                         { m_writer.write_instruction(rocprofiler_compute_tool::instruction_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteInstructionInsideCodeObjWithoutSymbol_Throws)
{
    m_writer.start_code_obj(0);
    expect_runtime_throw([&]()
                         { m_writer.write_instruction(rocprofiler_compute_tool::instruction_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteInstructionAfterEndSymbol_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{});
    m_writer.end_symbol();
    expect_runtime_throw([&]()
                         { m_writer.write_instruction(rocprofiler_compute_tool::instruction_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteInstructionAfterEndCodeObj_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.start_symbol(rocprofiler_compute_tool::symbol_t{});
    m_writer.end_symbol();
    m_writer.end_code_obj();
    expect_runtime_throw([&]()
                         { m_writer.write_instruction(rocprofiler_compute_tool::instruction_t{}); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteKernelWithoutAnyScope_Throws)
{
    expect_runtime_throw([&]() { m_writer.write_kernel(1, "kernel0"); });
}

TEST_F(test_code_object_writer_t, ProvidedWriteKernelAfterEndCodeObj_Throws)
{
    m_writer.start_code_obj(0);
    m_writer.end_code_obj();
    expect_runtime_throw([&]() { m_writer.write_kernel(1, "kernel0"); });
}

TEST_F(test_code_object_writer_t, ProvidedCodeObjDesc_SerializesIt)
{
    constexpr uint32_t id0 = 10;
    constexpr uint32_t id1 = 20;
    m_writer.start_code_obj(id0);
    m_writer.end_code_obj();
    m_writer.start_code_obj(id1);
    m_writer.end_code_obj();
    const auto& result = m_writer.get_result();
    const auto  json   = nlohmann::json::parse(result);
    EXPECT_EQ(json["code_objects"][0]["id"], id0);
    EXPECT_EQ(json["code_objects"][1]["id"], id1);
}

TEST_F(test_code_object_writer_t, ProvidedNoSymbols_SerializesEmptySymbolsArray)
{
    m_writer.start_code_obj(1);
    m_writer.end_code_obj();

    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_TRUE(json["code_objects"][0].contains("symbols"));
    EXPECT_TRUE(json["code_objects"][0]["symbols"].is_array());
    EXPECT_EQ(json["code_objects"][0]["symbols"].size(), 0);
}

TEST_F(test_code_object_writer_t, ProvidedNoKernels_SerializesEmptyKernelsArray)
{
    m_writer.start_code_obj(1);
    m_writer.end_code_obj();

    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_TRUE(json["code_objects"][0].contains("kernels"));
    EXPECT_TRUE(json["code_objects"][0]["kernels"].is_array());
    EXPECT_EQ(json["code_objects"][0]["kernels"].size(), 0);
}

TEST_F(test_code_object_writer_t, ProvidedKernel_SerializesItInsideCodeObj)
{
    constexpr uint64_t kernel_id = 42;
    const std::string  name      = "kernel0";

    m_writer.start_code_obj(1);
    m_writer.write_kernel(kernel_id, name);
    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& kernels = json["code_objects"][0]["kernels"];
    ASSERT_EQ(kernels.size(), 1);
    expect_kernel_json(kernels[0], kernel_id, name);
}

TEST_F(test_code_object_writer_t, ProvidedMultipleKernels_SerializesAllInOrder)
{
    m_writer.start_code_obj(1);
    m_writer.write_kernel(42, "kernel0");
    m_writer.write_kernel(43, "kernel1");
    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& kernels = json["code_objects"][0]["kernels"];
    ASSERT_EQ(kernels.size(), 2);
    expect_kernel_json(kernels[0], 42, "kernel0");
    expect_kernel_json(kernels[1], 43, "kernel1");
}

TEST_F(test_code_object_writer_t, ProvidedKernelsInDifferentCodeObjs_SerializesUnderRespectiveOwners)
{
    m_writer.start_code_obj(100);
    m_writer.write_kernel(42, "kernel0");
    m_writer.write_kernel(43, "kernel1");
    m_writer.end_code_obj();

    m_writer.start_code_obj(200);
    m_writer.write_kernel(44, "kernel2");
    m_writer.end_code_obj();

    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_EQ(json["code_objects"].size(), 2u);

    EXPECT_EQ(json["code_objects"][0]["id"], 100);
    ASSERT_EQ(json["code_objects"][0]["kernels"].size(), 2);
    expect_kernel_json(json["code_objects"][0]["kernels"][0], 42, "kernel0");
    expect_kernel_json(json["code_objects"][0]["kernels"][1], 43, "kernel1");

    EXPECT_EQ(json["code_objects"][1]["id"], 200);
    ASSERT_EQ(json["code_objects"][1]["kernels"].size(), 1);
    expect_kernel_json(json["code_objects"][1]["kernels"][0], 44, "kernel2");
}

TEST_F(test_code_object_writer_t, ProvidedSymbol_SerializesItInsideCodeObj)
{
    const rocprofiler_compute_tool::symbol_t symbol{"sym0", 0x10, 0x1000, 0x20};

    m_writer.start_code_obj(1);
    m_writer.start_symbol(symbol);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& symbols = json["code_objects"][0]["symbols"];
    ASSERT_EQ(symbols.size(), 1);
    EXPECT_EQ(symbols[0]["name"], symbol.name);
    EXPECT_EQ(symbols[0]["code_object_offset"], symbol.code_object_offset);
    EXPECT_EQ(symbols[0]["virtual_address"], symbol.virtual_address);
    EXPECT_EQ(symbols[0]["size"], symbol.size);
}

TEST_F(test_code_object_writer_t, ProvidedMultipleSymbols_SerializesAllInOrder)
{
    m_writer.start_code_obj(1);
    m_writer.start_symbol(m_symbol0);
    m_writer.end_symbol();
    m_writer.start_symbol(m_symbol1);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& symbols = json["code_objects"][0]["symbols"];
    ASSERT_EQ(symbols.size(), 2);
    EXPECT_EQ(symbols[0]["name"], m_symbol0.name);
    EXPECT_EQ(symbols[1]["name"], m_symbol1.name);
}

TEST_F(test_code_object_writer_t, ProvidedSymbolsInDifferentCodeObjs_SerializesUnderRespectiveOwners)
{
    m_writer.start_code_obj(100);
    m_writer.start_symbol(m_symbol0);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    m_writer.start_code_obj(200);
    m_writer.start_symbol(m_symbol1);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_EQ(json["code_objects"].size(), 2u);

    EXPECT_EQ(json["code_objects"][0]["id"], 100);
    ASSERT_EQ(json["code_objects"][0]["symbols"].size(), 1);
    EXPECT_EQ(json["code_objects"][0]["symbols"][0]["name"], m_symbol0.name);

    EXPECT_EQ(json["code_objects"][1]["id"], 200);
    ASSERT_EQ(json["code_objects"][1]["symbols"].size(), 1);
    EXPECT_EQ(json["code_objects"][1]["symbols"][0]["name"], m_symbol1.name);
}

TEST_F(test_code_object_writer_t, ProvidedNoInstructions_SerializesEmptyInstructionsArray)
{
    m_writer.start_code_obj(1);
    m_writer.start_symbol(m_symbol0);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json   = nlohmann::json::parse(m_writer.get_result());
    const auto& symbol = json["code_objects"][0]["symbols"][0];
    ASSERT_TRUE(symbol.contains("instructions"));
    EXPECT_TRUE(symbol["instructions"].is_array());
    EXPECT_EQ(symbol["instructions"].size(), 0);
}

TEST_F(test_code_object_writer_t, ProvidedInstruction_SerializesItInsideSymbol)
{
    m_writer.start_code_obj(1);
    m_writer.start_symbol(m_symbol0);
    m_writer.write_instruction(m_inst0);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json         = nlohmann::json::parse(m_writer.get_result());
    const auto& instructions = json["code_objects"][0]["symbols"][0]["instructions"];
    ASSERT_EQ(instructions.size(), 1);
    expect_instruction_json(instructions[0], m_inst0);
}

TEST_F(test_code_object_writer_t, ProvidedKernelAndInstruction_PreservesInstructionOutput)
{
    m_writer.start_code_obj(1);
    m_writer.write_kernel(42, "kernel0");
    m_writer.start_symbol(m_symbol0);
    m_writer.write_instruction(m_inst0);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& obj     = json["code_objects"][0];
    const auto& symbols = obj["symbols"];
    ASSERT_EQ(obj["kernels"].size(), 1);
    ASSERT_EQ(symbols.size(), 1);
    EXPECT_EQ(symbols[0]["name"], m_symbol0.name);
    EXPECT_EQ(symbols[0]["code_object_offset"], m_symbol0.code_object_offset);
    EXPECT_EQ(symbols[0]["virtual_address"], m_symbol0.virtual_address);
    EXPECT_EQ(symbols[0]["size"], m_symbol0.size);

    const auto& instructions = symbols[0]["instructions"];
    ASSERT_EQ(instructions.size(), 1);
    expect_instruction_json(instructions[0], m_inst0);
}

TEST_F(test_code_object_writer_t, ProvidedMultipleInstructions_SerializesAllInOrder)
{
    m_writer.start_code_obj(1);
    m_writer.start_symbol(m_symbol0);
    m_writer.write_instruction(m_inst0);
    m_writer.write_instruction(m_inst1);
    m_writer.end_symbol();
    m_writer.end_code_obj();

    const auto  json         = nlohmann::json::parse(m_writer.get_result());
    const auto& instructions = json["code_objects"][0]["symbols"][0]["instructions"];
    ASSERT_EQ(instructions.size(), 2);
    expect_instruction_json(instructions[0], m_inst0);
    expect_instruction_json(instructions[1], m_inst1);
}

TEST_F(test_code_object_writer_t, ProvidedInstructionsInDifferentSymbols_SerializesUnderRespectiveOwners)
{
    m_writer.start_code_obj(1);

    m_writer.start_symbol(m_symbol0);
    m_writer.write_instruction(m_inst0);
    m_writer.end_symbol();

    m_writer.start_symbol(m_symbol1);
    m_writer.write_instruction(m_inst1);
    m_writer.end_symbol();

    m_writer.end_code_obj();

    const auto  json    = nlohmann::json::parse(m_writer.get_result());
    const auto& symbols = json["code_objects"][0]["symbols"];
    ASSERT_EQ(symbols.size(), 2);

    ASSERT_EQ(symbols[0]["instructions"].size(), 1);
    EXPECT_EQ(symbols[0]["instructions"][0]["name"], m_inst0.name);

    ASSERT_EQ(symbols[1]["instructions"].size(), 1);
    EXPECT_EQ(symbols[1]["instructions"][0]["name"], m_inst1.name);
}

TEST_F(test_code_object_writer_t, ProvidedEmptyOutputFilePath_Throws)
{
    expect_runtime_throw([&]() { m_writer.flush(""); });
}
