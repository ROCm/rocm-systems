// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sample_record_collector.h"

#include "code_object_translator.h"
#include "nlohmann/json.hpp"
#include "ps_file_writer.h"

#include <stdexcept>
#include <thread>
#include <vector>

using namespace rocprofiler_compute_tool;

namespace
{
pc_sampling_record_t make_record(uint64_t code_object_id,
                                 uint64_t offset,
                                 bool     is_stochastic,
                                 uint64_t dispatch_id = 0)
{
    pc_sampling_record_t rec;
    rec.code_object_id     = code_object_id;
    rec.code_object_offset = offset;
    rec.dispatch_id        = dispatch_id;
    rec.wave_issued        = true;
    rec.is_stochastic      = is_stochastic;
    return rec;
}
}  // namespace

TEST_F(test_pc_sample_record_collector_t, CollectorAssignsStableInstIndexPerPc)
{
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->write(m_writer);

    const auto& host_trap = m_writer.get_host_trap_samples();
    ASSERT_EQ(host_trap.size(), 2);
    EXPECT_EQ(host_trap[0].first, host_trap[1].first);

    // Only one unique PC -> a single instruction/comment entry.
    EXPECT_EQ(m_writer.get_instruction_strings().size(), 1);
    EXPECT_EQ(m_writer.get_comment_strings().size(), 1);

    const int idx = host_trap[0].first;
    ASSERT_GE(idx, 0);
    ASSERT_LT(static_cast<size_t>(idx), m_writer.get_instruction_strings().size());
    EXPECT_EQ(m_writer.get_instruction_strings()[idx], "inst0");
    EXPECT_EQ(m_writer.get_comment_strings()[idx], "comment0");
}

TEST_F(test_pc_sample_record_collector_t, CollectorRoutesStochasticAndHostTrapSeparately)
{
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, true));
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->write(m_writer);

    EXPECT_EQ(m_writer.get_stochastic_samples().size(), 1);
    EXPECT_EQ(m_writer.get_host_trap_samples().size(), 1);
}

TEST_F(test_pc_sample_record_collector_t, CollectorBuildsKernelSymbolsFromTranslator)
{
    const std::vector<symbol_t> symbols = {{"kern(int*)", k_offset_a, k_load_base + k_offset_a, 4}};
    m_translator->add_symbols(k_code_object_id, symbols);

    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->write(m_writer);

    const auto& kernel_symbols = m_writer.get_kernel_symbols();
    bool        found          = false;
    for (const auto& [id, name] : kernel_symbols)
    {
        if (id == k_code_object_id && name == "kern(int*)")
        {
            found = true;
        }
    }
    EXPECT_TRUE(found);
}

TEST_F(test_pc_sample_record_collector_t, CollectorBuildsCoIndexedInstructionAndCommentStrings)
{
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->add_record(make_record(k_code_object_id, k_offset_b, false));
    m_collector->write(m_writer);

    EXPECT_EQ(m_writer.get_instruction_strings().size(), m_writer.get_comment_strings().size());
    EXPECT_EQ(m_writer.get_instruction_strings().size(), 2);
}

TEST_F(test_pc_sample_record_collector_t, CollectorEmittedInstIndexInRange)
{
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, true));
    m_collector->add_record(make_record(k_code_object_id, k_offset_b, false));
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->write(m_writer);

    const size_t inst_count = m_writer.get_instruction_strings().size();
    ASSERT_GT(inst_count, 0u);

    for (const auto& [idx, rec] : m_writer.get_stochastic_samples())
    {
        EXPECT_GE(idx, 0);
        EXPECT_LT(static_cast<size_t>(idx), inst_count);
    }
    for (const auto& [idx, rec] : m_writer.get_host_trap_samples())
    {
        EXPECT_GE(idx, 0);
        EXPECT_LT(static_cast<size_t>(idx), inst_count);
    }
}

TEST_F(test_pc_sample_record_collector_t, CollectorConcurrentAddRecordIsThreadSafe)
{
    constexpr int            num_threads = 4;
    constexpr int            per_thread  = 250;
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for (int t = 0; t < num_threads; ++t)
    {
        threads.emplace_back(
            [this]()
            {
                for (int i = 0; i < per_thread; ++i)
                {
                    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
                }
            });
    }
    for (auto& thread : threads)
    {
        thread.join();
    }

    m_collector->write(m_writer);

    const size_t total = m_writer.get_stochastic_samples().size() +
                         m_writer.get_host_trap_samples().size();
    EXPECT_EQ(total, static_cast<size_t>(num_threads * per_thread));
}

TEST_F(test_pc_sample_record_collector_t, CollectorRoundTripsThroughJsonWriter)
{
    constexpr int         num_records = 5;
    ps_file_writer_json_t json_writer;
    for (int i = 0; i < num_records; ++i)
    {
        m_collector->add_record(make_record(k_code_object_id, k_offset_a, false, i));
    }
    m_collector->write(json_writer);

    const auto  parsed         = nlohmann::json::parse(json_writer.get_result());
    const auto& buffer_records = parsed.at("rocprofiler-sdk-tool").at(0).at("buffer_records");

    const size_t stochastic = buffer_records.at("pc_sample_stochastic").size();
    const size_t host_trap  = buffer_records.at("pc_sample_host_trap").size();
    EXPECT_EQ(stochastic + host_trap, static_cast<size_t>(num_records));

    for (const auto& sample : buffer_records.at("pc_sample_host_trap"))
    {
        const auto& pc = sample.at("record").at("pc");
        EXPECT_EQ(pc.at("code_object_id").get<uint64_t>(), k_code_object_id);
        EXPECT_EQ(pc.at("code_object_offset").get<uint64_t>(), k_offset_a);
    }
}

TEST_F(test_pc_sample_record_collector_t, CollectorSkipsRecordsForUnknownCodeObject)
{
    constexpr uint64_t unknown_id = 999;
    m_collector->add_record(make_record(unknown_id, k_offset_a, false));
    EXPECT_NO_THROW(m_collector->write(m_writer));

    EXPECT_TRUE(m_writer.get_stochastic_samples().empty());
    EXPECT_TRUE(m_writer.get_host_trap_samples().empty());
}

TEST_F(test_pc_sample_record_collector_t, CollectorEmitsEmptyInstructionForUndecodablePc)
{
    // An undecodable PC: the translator returns an empty instruction (no name/comment).
    const instruction_t empty_instruction = {"", "", k_load_base + k_offset_a, k_offset_a, 4};
    m_translator->add_instruction(empty_instruction);

    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->write(m_writer);

    const auto& host_trap = m_writer.get_host_trap_samples();
    ASSERT_EQ(host_trap.size(), 1);

    const int idx = host_trap[0].first;
    ASSERT_GE(idx, 0);
    ASSERT_LT(static_cast<size_t>(idx), m_writer.get_instruction_strings().size());
    EXPECT_EQ(m_writer.get_instruction_strings()[idx], "");
}

TEST_F(test_pc_sample_record_collector_t, CollectorAssignsDistinctInstIndexForDistinctPcs)
{
    m_collector->add_record(make_record(k_code_object_id, k_offset_a, false));
    m_collector->add_record(make_record(k_code_object_id, k_offset_b, false));
    m_collector->write(m_writer);

    const auto& host_trap = m_writer.get_host_trap_samples();
    ASSERT_EQ(host_trap.size(), 2);
    EXPECT_NE(host_trap[0].first, host_trap[1].first);

    const size_t inst_count = m_writer.get_instruction_strings().size();
    for (const auto& [idx, rec] : host_trap)
    {
        EXPECT_GE(idx, 0);
        EXPECT_LT(static_cast<size_t>(idx), inst_count);
    }
}

TEST_F(test_pc_sample_record_collector_t, CollectorForwardsMemoryStorageCodeObject)
{
    rocprofiler_callback_tracing_code_object_load_data_t mem_info = {};
    mem_info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
    mem_info.memory_base    = 0x5000;
    mem_info.memory_size    = 0x800;
    mem_info.code_object_id = 444;
    mem_info.load_base      = 0x6000;
    mem_info.load_size      = 0x900;

    m_collector->on_code_object_load(mem_info);

    const auto& mem_info_captured = m_translator->get_mem_code_object_info();
    ASSERT_FALSE(mem_info_captured.empty());
    const auto& entry = mem_info_captured.back();
    EXPECT_EQ(entry.memory_base, mem_info.memory_base);
    EXPECT_EQ(entry.memory_size, mem_info.memory_size);
    EXPECT_EQ(entry.id, mem_info.code_object_id);
    EXPECT_EQ(entry.load_base, mem_info.load_base);
    EXPECT_EQ(entry.load_size, mem_info.load_size);
}

TEST_F(test_pc_sample_record_collector_t, TranslatorGetLoadBaseThrowsForUnknownId)
{
    code_object_translator_impl_t real_translator;
    EXPECT_THROW(real_translator.get_load_base(12345), std::out_of_range);
}

void test_pc_sample_record_collector_t::SetUp()
{
    m_translator = std::make_shared<mock_code_object_translator_t>();
    m_collector  = pc_sample_record_collector_t::create(m_translator);

    m_file_info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE;
    m_file_info.uri            = "test_code_object.co";
    m_file_info.code_object_id = k_code_object_id;
    m_file_info.load_base      = k_load_base;
    m_file_info.load_size      = 0x2000;

    // Registers the code object id with the mock translator (via add_code_object).
    m_collector->on_code_object_load(m_file_info);

    m_translator->set_load_base(k_code_object_id, k_load_base);
    const std::vector<symbol_t> symbols = {{"name0", k_offset_a, k_load_base + k_offset_a, 4}};
    m_translator->add_symbols(k_code_object_id, symbols);
    const instruction_t instruction = {"inst0", "comment0", k_load_base + k_offset_a, k_offset_a, 4};
    m_translator->add_instruction(instruction);
}
