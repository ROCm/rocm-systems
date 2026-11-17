// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_collector.h"

#include <filesystem>
#include <set>
#include <string>
#include <utility>

using namespace rocprofiler_compute_tool;

namespace
{
std::set<std::filesystem::path> collect_source_paths_from_comment(std::string comment,
                                                                  std::string separator = " -> ")
{
    auto translator  = std::make_shared<mock_code_object_translator_t>();
    auto sdk_wrapper = std::make_shared<mock_sdk_wrapper_t>();
    sdk_wrapper->set_source_frame_separator(std::move(separator));
    auto collector = std::make_shared<pc_sampling_collector_impl_t>(translator, sdk_wrapper);
    auto writer    = std::make_shared<mock_code_object_writer_t>();

    translator->add_code_object("test_code_object", 1, 0x1000, 0x100);
    translator->add_symbols(1, {{"name0", 0x10, 0x1000, 1}});
    translator->add_instruction({"inst0", std::move(comment), 0x1000, 0x10, 1});

    collector->finalize(*writer);
    return collector->get_source_paths();
}
}  // namespace

TEST_F(test_pc_sampling_collector_t, ProvidedFileCodeObject_PassesItToDecode)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    const auto file_info = m_translator->get_file_code_object_info();
    const auto mem_info  = m_translator->get_mem_code_object_info();
    EXPECT_EQ(file_info.size(), 1);
    EXPECT_EQ(file_info[0].filepath, m_file_info.uri);
    EXPECT_EQ(file_info[0].id, m_file_info.code_object_id);
    EXPECT_EQ(file_info[0].load_base, m_file_info.load_base);
    EXPECT_EQ(file_info[0].load_size, m_file_info.load_size);
    EXPECT_TRUE(mem_info.empty());
}

TEST_F(test_pc_sampling_collector_t, ProvidedMemoryCodeObject_PassesItToDecode)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    const auto file_info = m_translator->get_file_code_object_info();
    const auto mem_info  = m_translator->get_mem_code_object_info();
    EXPECT_EQ(mem_info.size(), 1);
    EXPECT_EQ(mem_info[0].memory_base, m_mem_info.memory_base);
    EXPECT_EQ(mem_info[0].memory_size, m_mem_info.memory_size);
    EXPECT_EQ(mem_info[0].id, m_mem_info.code_object_id);
    EXPECT_EQ(mem_info[0].load_base, m_mem_info.load_base);
    EXPECT_EQ(mem_info[0].load_size, m_mem_info.load_size);
    EXPECT_TRUE(file_info.empty());
}

TEST_F(test_pc_sampling_collector_t, ProvidedCodeObjects_WritesTheirIds)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    m_pc_sampling_collector->finalize(*m_writer);
    EXPECT_EQ(m_writer->get_start_code_obj_ids().size(), 2);
    EXPECT_EQ(m_writer->get_end_code_obj_count(), 2);
    EXPECT_EQ(m_writer->get_start_code_obj_ids()[0], m_file_info.code_object_id);
    EXPECT_EQ(m_writer->get_start_code_obj_ids()[1], m_mem_info.code_object_id);
}

TEST_F(test_pc_sampling_collector_t, ProvidedCodeObjectSymbols_WritesThem)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    const std::vector<symbol_t> symbols0 = {{"name0", 0x10, 0x1000, 1}, {"name1", 0x20, 0x2000, 0x60}};
    const std::vector<symbol_t> symbols1 = {{"name2", 0x11, 0x1001, 1}, {"name3", 0x21, 0x2001, 0x61}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols0);
    m_translator->add_symbols(m_mem_info.code_object_id, symbols1);
    m_pc_sampling_collector->finalize(*m_writer);
    EXPECT_EQ(m_writer->get_symbol_descriptions().size(), 4);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[0].name, symbols0[0].name);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[1].name, symbols0[1].name);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[2].name, symbols1[0].name);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[3].name, symbols1[1].name);
}

TEST_F(test_pc_sampling_collector_t, ProvidedSymbolInstructions_WritesThem)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    const std::vector<symbol_t> symbols = {{"name0", 0x10, 0x1000, 4}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols);
    m_translator->set_instruction(m_file_info.code_object_id,
                                  0x1000,
                                  {"inst0", "comment0", 0x1000, 0x10, 1});
    m_translator->set_instruction(m_file_info.code_object_id,
                                  0x1001,
                                  {"inst1", "comment1", 0x1001, 0x11, 2});
    m_translator->set_instruction(m_file_info.code_object_id,
                                  0x1003,
                                  {"inst2", "comment2", 0x1003, 0x13, 1});

    m_pc_sampling_collector->finalize(*m_writer);

    const std::vector<std::pair<size_t, uint64_t>> expected_requests = {
        {m_file_info.code_object_id, 0x1000},
        {m_file_info.code_object_id, 0x1001},
        {m_file_info.code_object_id, 0x1003},
    };
    EXPECT_EQ(m_translator->get_instruction_requests(), expected_requests);

    const auto& instructions = m_writer->get_instruction_descriptions();
    ASSERT_EQ(instructions.size(), 3);
    EXPECT_EQ(instructions[0].name, "inst0");
    EXPECT_EQ(instructions[1].name, "inst1");
    EXPECT_EQ(instructions[2].name, "inst2");
}

TEST_F(test_pc_sampling_collector_t, ProvidedSymbolInstructionSizeZero_Throws)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    const std::vector<symbol_t> symbols = {{"name0", 0x10, 0x1000, 2}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols);
    m_translator->add_symbols(m_mem_info.code_object_id, symbols);
    const instruction_t instruction = {"inst0", "comment0", 0x1000, 0x10, 0};
    m_translator->add_instruction(instruction);

    EXPECT_THROW(m_pc_sampling_collector->finalize(*m_writer), std::runtime_error);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedEmptyComment_ReturnsNoPaths)
{
    const auto paths = collect_source_paths_from_comment("");
    EXPECT_TRUE(paths.empty());
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedSingleFileLine_ReturnsFilePath)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:42");
    const std::set<std::filesystem::path> expected = {"kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedMultipleFrames_ReturnsAllFilePaths)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:42 -> header.h:8");
    const std::set<std::filesystem::path> expected = {"header.h", "kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedInjectedFrameSeparator_ReturnsAllFilePaths)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:42 | header.h:8", " | ");
    const std::set<std::filesystem::path> expected = {"header.h", "kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedUnknownLineToken_ReturnsFilePath)
{
    const auto                            paths = collect_source_paths_from_comment("kernel.cpp:?");
    const std::set<std::filesystem::path> expected = {"kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedNoColonFrame_ReturnsWholeFrame)
{
    const auto                            paths = collect_source_paths_from_comment("kernel.cpp");
    const std::set<std::filesystem::path> expected = {"kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedTrailingColon_ReturnsWholeFrame)
{
    const auto                            paths = collect_source_paths_from_comment("kernel.cpp:");
    const std::set<std::filesystem::path> expected = {"kernel.cpp:"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedDuplicateFrames_ReturnsDeduplicatedPaths)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:42 -> kernel.cpp:42");
    const std::set<std::filesystem::path> expected = {"kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedPathWithDirectories_ReturnsDirectoryPath)
{
    const auto paths = collect_source_paths_from_comment("/tmp/project/include/header.h:8");
    const std::set<std::filesystem::path> expected = {"/tmp/project/include/header.h"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedTrailingSeparator_IgnoresEmptyFrame)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:42 -> ");
    const std::set<std::filesystem::path> expected = {"kernel.cpp"};
    EXPECT_EQ(paths, expected);
}

TEST(test_pc_sampling_collector_source_paths_t, ProvidedNonNumericColonSuffix_ReturnsWholeFrame)
{
    const auto paths = collect_source_paths_from_comment("kernel.cpp:label");
    const std::set<std::filesystem::path> expected = {"kernel.cpp:label"};
    EXPECT_EQ(paths, expected);
}

TEST_F(test_pc_sampling_collector_t, ProvidedSymbolsAndInstructions_CollectsDeduplicatedSourcePaths)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    m_translator->add_symbols(m_file_info.code_object_id,
                              {{"name0", 0x10, 0x1000, 2}, {"name1", 0x20, 0x2000, 1}});
    m_translator->add_symbols(m_mem_info.code_object_id, {{"name2", 0x30, 0x3000, 1}});
    m_translator->set_instruction(m_file_info.code_object_id,
                                  0x1000,
                                  {"inst0", "kernel.cpp:42 -> header.h:8", 0x1000, 0x10, 1});
    m_translator->set_instruction(m_file_info.code_object_id, 0x1001, {"inst1", "", 0x1001, 0x11, 1});
    m_translator->set_instruction(m_file_info.code_object_id,
                                  0x2000,
                                  {"inst2", "kernel.cpp:42", 0x2000, 0x20, 1});
    m_translator->set_instruction(m_mem_info.code_object_id,
                                  0x3000,
                                  {"inst3", "other.cpp:?", 0x3000, 0x30, 1});

    m_pc_sampling_collector->finalize(*m_writer);

    const std::set<std::filesystem::path> expected_paths = {"header.h", "kernel.cpp", "other.cpp"};
    EXPECT_EQ(m_pc_sampling_collector->get_source_paths(), expected_paths);
}

void test_pc_sampling_collector_t::SetUp()
{
    m_translator  = std::make_shared<mock_code_object_translator_t>();
    m_sdk_wrapper = std::make_shared<mock_sdk_wrapper_t>();
    m_pc_sampling_collector = std::make_shared<pc_sampling_collector_impl_t>(m_translator, m_sdk_wrapper);
    m_writer = std::make_shared<mock_code_object_writer_t>();

    m_mem_info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
    m_mem_info.memory_base    = 0x1000;
    m_mem_info.memory_size    = 0x2000;
    m_mem_info.code_object_id = 111;
    m_mem_info.load_base      = 0x1000;
    m_mem_info.load_size      = 0x2000;

    m_file_info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE;
    m_file_info.uri            = "test_code_object.co";
    m_file_info.code_object_id = 222;
    m_file_info.load_base      = 0x1000;
    m_file_info.load_size      = 0x2000;
}
