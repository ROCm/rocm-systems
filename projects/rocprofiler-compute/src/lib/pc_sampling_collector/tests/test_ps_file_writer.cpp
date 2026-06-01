// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_ps_file_writer.h"

#include "nlohmann/json.hpp"

#include <cstdio>
#include <filesystem>
#include <fstream>

namespace
{
nlohmann::json tool_node(const std::string& serialized)
{
    const auto json = nlohmann::json::parse(serialized);
    return json["rocprofiler-sdk-tool"][0];
}
}  // namespace

TEST_F(test_ps_file_writer_t, PsFileWriterEmitsStochasticSamplesInOrder)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec0{};
    rec0.code_object_id     = 1;
    rec0.code_object_offset = 0x10;
    rec0.dispatch_id        = 5;
    rec0.wave_issued        = true;
    rec0.is_stochastic      = true;
    rec0.stall_reason       = std::nullopt;

    rocprofiler_compute_tool::pc_sampling_record_t rec1{};
    rec1.code_object_id     = 2;
    rec1.code_object_offset = 0x20;
    rec1.dispatch_id        = 6;
    rec1.wave_issued        = false;
    rec1.is_stochastic      = true;
    rec1.stall_reason = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY";

    m_writer.add_stochastic_sample(0, rec0);
    m_writer.add_stochastic_sample(1, rec1);

    const auto  node       = tool_node(m_writer.get_result());
    const auto& stochastic = node["buffer_records"]["pc_sample_stochastic"];
    ASSERT_EQ(stochastic.size(), 2u);

    EXPECT_EQ(stochastic[0]["inst_index"], 0);
    EXPECT_EQ(stochastic[0]["record"]["pc"]["code_object_id"], 1);
    EXPECT_EQ(stochastic[0]["record"]["pc"]["code_object_offset"], 0x10);
    EXPECT_EQ(stochastic[0]["record"]["dispatch_id"], 5);
    EXPECT_EQ(stochastic[0]["record"]["wave_issued"], true);
    EXPECT_FALSE(stochastic[0]["record"]["snapshot"].contains("stall_reason"));

    EXPECT_EQ(stochastic[1]["inst_index"], 1);
    EXPECT_EQ(stochastic[1]["record"]["wave_issued"], false);
    EXPECT_EQ(stochastic[1]["record"]["snapshot"]["stall_reason"],
              "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_ALU_DEPENDENCY");
}

TEST_F(test_ps_file_writer_t, PsFileWriterEmitsHostTrapSamplesSeparately)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id     = 3;
    rec.code_object_offset = 0x30;
    rec.dispatch_id        = 7;
    rec.wave_issued        = true;
    rec.is_stochastic      = false;

    m_writer.add_host_trap_sample(0, rec);

    const auto  node    = tool_node(m_writer.get_result());
    const auto& records = node["buffer_records"];
    ASSERT_EQ(records["pc_sample_host_trap"].size(), 1u);
    EXPECT_EQ(records["pc_sample_stochastic"].size(), 0u);
    EXPECT_EQ(records["pc_sample_host_trap"][0]["inst_index"], 0);
    EXPECT_EQ(records["pc_sample_host_trap"][0]["record"]["pc"]["code_object_id"], 3);
}

TEST_F(test_ps_file_writer_t, PsFileWriterOmitsWaveIssuedForHostTrap)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id     = 4;
    rec.code_object_offset = 0x40;
    rec.dispatch_id        = 9;
    rec.is_stochastic      = false;
    // wave_issued left UNSET (nullopt) for host-trap records.

    m_writer.add_host_trap_sample(0, rec);

    const auto  node   = tool_node(m_writer.get_result());
    const auto& record = node["buffer_records"]["pc_sample_host_trap"][0]["record"];
    EXPECT_FALSE(record.contains("wave_issued"));
}

TEST_F(test_ps_file_writer_t, PsFileWriterEmitsWaveIssuedForStochastic)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id     = 5;
    rec.code_object_offset = 0x50;
    rec.dispatch_id        = 10;
    rec.wave_issued        = true;
    rec.is_stochastic      = true;

    m_writer.add_stochastic_sample(0, rec);

    const auto  node   = tool_node(m_writer.get_result());
    const auto& record = node["buffer_records"]["pc_sample_stochastic"][0]["record"];
    ASSERT_TRUE(record.contains("wave_issued"));
    EXPECT_EQ(record["wave_issued"], true);
}

TEST_F(test_ps_file_writer_t, PsFileWriterOmitsStallReasonWhenIssued)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id = 1;
    rec.dispatch_id    = 1;
    rec.wave_issued    = true;
    rec.is_stochastic  = true;
    rec.stall_reason   = std::nullopt;

    m_writer.add_stochastic_sample(0, rec);

    const auto  node     = tool_node(m_writer.get_result());
    const auto& snapshot = node["buffer_records"]["pc_sample_stochastic"][0]["record"]["snapshot"];
    EXPECT_FALSE(snapshot.contains("stall_reason"));
}

TEST_F(test_ps_file_writer_t, PsFileWriterEmitsPrefixedStallReasonWhenStalled)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id = 1;
    rec.dispatch_id    = 1;
    rec.wave_issued    = false;
    rec.is_stochastic  = true;
    rec.stall_reason   = "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT";

    m_writer.add_stochastic_sample(0, rec);

    const auto  node     = tool_node(m_writer.get_result());
    const auto& snapshot = node["buffer_records"]["pc_sample_stochastic"][0]["record"]["snapshot"];
    EXPECT_EQ(snapshot["stall_reason"], "ROCPROFILER_PC_SAMPLING_INSTRUCTION_NOT_ISSUED_REASON_WAITCNT");
}

TEST_F(test_ps_file_writer_t, PsFileWriterStringsAndKernelSymbolsSerialized)
{
    m_writer.set_instruction_strings({"s_load", "v_add"}, {"line1", "line2"});
    m_writer.add_kernel_symbol(7, "vcopy(int*)");

    const auto  node    = tool_node(m_writer.get_result());
    const auto& strings = node["strings"];
    ASSERT_EQ(strings["pc_sample_instructions"].size(), 2u);
    EXPECT_EQ(strings["pc_sample_instructions"][0], "s_load");
    EXPECT_EQ(strings["pc_sample_instructions"][1], "v_add");
    ASSERT_EQ(strings["pc_sample_comments"].size(), 2u);
    EXPECT_EQ(strings["pc_sample_comments"][0], "line1");
    EXPECT_EQ(strings["pc_sample_comments"][1], "line2");

    const auto& kernel_symbols = node["kernel_symbols"];
    ASSERT_EQ(kernel_symbols.size(), 1u);
    EXPECT_EQ(kernel_symbols[0]["code_object_id"], 7);
    EXPECT_EQ(kernel_symbols[0]["formatted_kernel_name"], "vcopy(int*)");
}

TEST_F(test_ps_file_writer_t, PsFileWriterEmptyHasFullTopLevelShape)
{
    const auto json = nlohmann::json::parse(m_writer.get_result());
    ASSERT_TRUE(json.contains("rocprofiler-sdk-tool"));
    ASSERT_TRUE(json["rocprofiler-sdk-tool"].is_array());
    ASSERT_EQ(json["rocprofiler-sdk-tool"].size(), 1u);

    const auto& node = json["rocprofiler-sdk-tool"][0];
    ASSERT_TRUE(node.contains("buffer_records"));
    EXPECT_TRUE(node["buffer_records"]["pc_sample_stochastic"].is_array());
    EXPECT_EQ(node["buffer_records"]["pc_sample_stochastic"].size(), 0u);
    EXPECT_TRUE(node["buffer_records"]["pc_sample_host_trap"].is_array());
    EXPECT_EQ(node["buffer_records"]["pc_sample_host_trap"].size(), 0u);

    ASSERT_TRUE(node.contains("strings"));
    EXPECT_TRUE(node["strings"]["pc_sample_instructions"].is_array());
    EXPECT_EQ(node["strings"]["pc_sample_instructions"].size(), 0u);
    EXPECT_TRUE(node["strings"]["pc_sample_comments"].is_array());
    EXPECT_EQ(node["strings"]["pc_sample_comments"].size(), 0u);

    ASSERT_TRUE(node.contains("kernel_symbols"));
    EXPECT_TRUE(node["kernel_symbols"].is_array());
    EXPECT_EQ(node["kernel_symbols"].size(), 0u);
}

TEST_F(test_ps_file_writer_t, PsFileWriterFlushWritesFile)
{
    rocprofiler_compute_tool::pc_sampling_record_t rec{};
    rec.code_object_id = 1;
    rec.dispatch_id    = 1;
    rec.wave_issued    = true;
    rec.is_stochastic  = true;
    m_writer.add_stochastic_sample(0, rec);

    const auto path = std::filesystem::temp_directory_path() /
                      "test_ps_file_writer_flush_output.json";
    std::filesystem::remove(path);

    m_writer.flush(path);
    ASSERT_TRUE(std::filesystem::exists(path));

    std::ifstream in(path);
    const std::string contents((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    in.close();

    EXPECT_EQ(nlohmann::json::parse(contents), nlohmann::json::parse(m_writer.get_result()));

    std::filesystem::remove(path);
}
