// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_collector.h"

using namespace rocm_compute;

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

TEST_F(test_pc_sampling_collector_t, ProvidedCodeObjectId_IsStored)
{
    m_pc_sampling_collector->on_code_object_load(m_file_info);
    m_pc_sampling_collector->on_code_object_load(m_mem_info);
    m_pc_sampling_collector->write(*m_writer);
    EXPECT_EQ(m_writer->get_started_code_obj_ids().size(), 2);
    EXPECT_EQ(m_writer->get_ended_code_obj_desc_ids().size(), 2);
}

TEST_F(test_pc_sampling_collector_t, TODO)
{
    const std::vector<obj_symbol_t> symbols = {{"name0", 0x10, 0x1000, 0x50}, {"name1", 0x20, 0x2000, 0x60}};
    m_translator->add_symbols(1, symbols);
    m_pc_sampling_collector->write(*m_writer);
}

void test_pc_sampling_collector_t::SetUp()
{
    m_translator            = std::make_shared<mock_code_object_translator_t>();
    m_pc_sampling_collector = std::make_shared<pc_sampling_collector_impl_t>(m_translator);
    m_writer                = std::make_shared<mock_pc_samples_writer_t>();

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