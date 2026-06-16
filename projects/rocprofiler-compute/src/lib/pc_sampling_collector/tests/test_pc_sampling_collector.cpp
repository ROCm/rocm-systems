// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_collector.h"

#include "nlohmann/json.hpp"

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>

using namespace rocprofiler_compute_tool;

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
    m_pc_sampling_collector->write(*m_writer);
    EXPECT_EQ(m_writer->get_start_code_obj_ids().size(), 2);
    EXPECT_EQ(m_writer->get_end_code_obj_count(), 2);
    EXPECT_EQ(m_writer->get_start_code_obj_ids()[0], m_file_info.code_object_id);
    EXPECT_EQ(m_writer->get_start_code_obj_ids()[1], m_mem_info.code_object_id);
}

TEST_F(test_pc_sampling_collector_t, WritesOneSpanSymbolPerCodeObject)
{
    // write() now disassembles the whole loaded range under a single span symbol
    // per code object, independent of named ELF symbols.
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    const instruction_t instruction = {"inst0", "comment0", 0x1000, 0x10, 4};
    m_translator->add_instruction(instruction);

    m_pc_sampling_collector->write(*m_writer);

    // One span symbol per code object, each covering the full load_size.
    ASSERT_EQ(m_writer->get_symbol_descriptions().size(), 2u);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[0].size, m_file_info.load_size);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[1].size, m_mem_info.load_size);
    EXPECT_EQ(m_writer->get_symbol_descriptions()[0].code_object_offset, 0u);
}

TEST_F(test_pc_sampling_collector_t, WritesInstructionsAcrossTheWholeLoadedRange)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    // Fixed-size instruction; the walk covers [load_base, load_base + load_size).
    const instruction_t instruction = {"inst0", "comment0", 0x1000, 0x10, 4};
    m_translator->add_instruction(instruction);

    m_pc_sampling_collector->write(*m_writer);

    // load_size / inst.size instructions, regardless of named symbols.
    const size_t expected = m_mem_info.load_size / instruction.size;
    EXPECT_EQ(m_writer->get_instruction_descriptions().size(), expected);
    // The walk queries the decoder at load_base + offset across the range.
    const auto& queries = m_translator->get_instruction_queries();
    ASSERT_FALSE(queries.empty());
    EXPECT_EQ(queries.front().second, m_mem_info.load_base);
}

TEST_F(test_pc_sampling_collector_t, UndecodableRegionDoesNotAbortWalk)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    // size 0 models an undecodable word; the walk must skip it and keep going,
    // not loop forever or throw.
    const instruction_t instruction = {"", "", 0x1000, 0x0, 0};
    m_translator->add_instruction(instruction);

    EXPECT_NO_THROW(m_pc_sampling_collector->write(*m_writer));
    EXPECT_EQ(m_writer->get_instruction_descriptions().size(), 0u);
    EXPECT_EQ(m_writer->get_end_code_obj_count(), 1u);
}

TEST_F(test_pc_sampling_collector_t, WriteSamples_DedupsAndRoutesByKind)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    // The mock translator returns this instruction for any (id, vaddr) lookup,
    // so both samples resolve to the same (name, comment) index 0.
    m_translator->add_instruction({"v_add", "kernel.cpp:7", 0x1000, 0x10, 4});

    pc_sample_record_t stochastic{};
    stochastic.kind                  = pc_sample_kind_t::Stochastic;
    stochastic.pc.code_object_id     = m_mem_info.code_object_id;
    stochastic.pc.code_object_offset = 0x10;

    pc_sample_record_t host_trap{};
    host_trap.kind                  = pc_sample_kind_t::HostTrap;
    host_trap.pc.code_object_id     = m_mem_info.code_object_id;
    host_trap.pc.code_object_offset = 0x20;

    m_pc_sampling_collector->append_sample(stochastic);
    m_pc_sampling_collector->append_sample(host_trap);

    pc_sample_writer_json_t writer;
    m_pc_sampling_collector->write_samples(writer);

    const auto  json = nlohmann::json::parse(writer.get_result());
    const auto& root = json["rocprofiler-sdk-tool"][0];

    ASSERT_EQ(root["buffer_records"]["pc_sample_stochastic"].size(), 1u);
    ASSERT_EQ(root["buffer_records"]["pc_sample_host_trap"].size(), 1u);

    EXPECT_EQ(root["buffer_records"]["pc_sample_stochastic"][0]["inst_index"], 0);
    EXPECT_EQ(root["buffer_records"]["pc_sample_host_trap"][0]["inst_index"], 0);
    ASSERT_EQ(root["strings"]["pc_sample_instructions"].size(), 1u);
    EXPECT_EQ(root["strings"]["pc_sample_instructions"][0], "v_add");
    EXPECT_EQ(root["strings"]["pc_sample_comments"][0], "kernel.cpp:7");
}

TEST_F(test_pc_sampling_collector_t, WriteSamples_TranslatesOffsetByLoadBase)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    m_translator->add_instruction({"v_add", "kernel.cpp:7", 0x1000, 0x10, 4});

    pc_sample_record_t sample{};
    sample.kind                  = pc_sample_kind_t::HostTrap;
    sample.pc.code_object_id     = m_mem_info.code_object_id;
    sample.pc.code_object_offset = 0x40;
    m_pc_sampling_collector->append_sample(sample);

    pc_sample_writer_json_t writer;
    m_pc_sampling_collector->write_samples(writer);

    // The PC offset (0x40) must be translated to a virtual address by adding the
    // code object's load base (0x1000) before instruction lookup.
    const auto& queries = m_translator->get_instruction_queries();
    ASSERT_EQ(queries.size(), 1u);
    EXPECT_EQ(queries[0].first, m_mem_info.code_object_id);
    EXPECT_EQ(queries[0].second, 0x40u + m_mem_info.load_base);
}

TEST_F(test_pc_sampling_collector_t, WriteSamples_UnresolvedPcInsertsEmptyInstruction)
{
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    m_translator->add_instruction({"v_add", "kernel.cpp:7", 0x1000, 0x10, 4});

    // A PC whose translated virtual address has no decoded instruction: the
    // translator throws std::out_of_range and the sample degrades to an empty
    // (instruction, comment) entry instead of aborting serialization.
    const uint64_t offset = 0x40;
    m_translator->throw_for_virtual_address(offset + m_mem_info.load_base);

    pc_sample_record_t sample{};
    sample.kind                  = pc_sample_kind_t::HostTrap;
    sample.pc.code_object_id     = m_mem_info.code_object_id;
    sample.pc.code_object_offset = offset;
    m_pc_sampling_collector->append_sample(sample);

    pc_sample_writer_json_t writer;
    EXPECT_NO_THROW(m_pc_sampling_collector->write_samples(writer));

    const auto  json = nlohmann::json::parse(writer.get_result());
    const auto& root = json["rocprofiler-sdk-tool"][0];

    ASSERT_EQ(root["buffer_records"]["pc_sample_host_trap"].size(), 1u);
    EXPECT_EQ(root["buffer_records"]["pc_sample_host_trap"][0]["inst_index"], 0);
    ASSERT_EQ(root["strings"]["pc_sample_instructions"].size(), 1u);
    EXPECT_EQ(root["strings"]["pc_sample_instructions"][0], "");
    EXPECT_EQ(root["strings"]["pc_sample_comments"][0], "");
}

TEST_F(test_pc_sampling_collector_t, SnapshotSources_CopiesSourcesParsedFromInstructionComments)
{
    namespace fs = std::filesystem;

    // snapshot_sources resolves refs against the current working directory, so
    // the source file must live under CWD to be inside the allowed root.
    const fs::path rel_dir = fs::path{"rpc_snapshot_sources_test_" + std::to_string(::getpid())};
    const fs::path rel_src = rel_dir / "kernel.cpp";
    fs::create_directories(rel_dir);
    {
        std::ofstream ofs(rel_src);
        ofs << "source contents\n";
    }
    const fs::path out_root = fs::temp_directory_path() /
                              ("rpc_snapshot_out_" + std::to_string(::getpid()));

    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    // Comment parses to rel_src. The full-range walk surfaces it without needing
    // a named symbol.
    m_translator->add_instruction({"v_add", rel_src.string() + ":7", 0x1000, 0x10, 1});

    const size_t copied = m_pc_sampling_collector->snapshot_sources(out_root);

    EXPECT_EQ(copied, 1u);
    EXPECT_TRUE(fs::exists(out_root / "code_obj_sources" / rel_dir / "kernel.cpp"));

    std::error_code ec;
    fs::remove_all(rel_dir, ec);
    fs::remove_all(out_root, ec);
}

void test_pc_sampling_collector_t::SetUp()
{
    m_translator            = std::make_shared<mock_code_object_translator_t>();
    m_pc_sampling_collector = std::make_shared<pc_sampling_collector_impl_t>(m_translator);
    m_writer                = std::make_shared<mock_code_object_writer_t>();

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
