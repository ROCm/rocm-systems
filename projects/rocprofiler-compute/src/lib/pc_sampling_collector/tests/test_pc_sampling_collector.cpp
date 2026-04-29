// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_pc_sampling_collector.h"

using namespace rocm_compute;

TEST_F(test_pc_sampling_collector_t, ProvidedFileCodeObject_PassesItToDecode)
{
    rocprofiler_callback_tracing_code_object_load_data_t info;
    info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE;
    info.uri            = "test_code_object.co";
    info.code_object_id = 123;
    info.load_base      = 0x1000;
    info.load_size      = 0x2000;
    m_pc_sampling_collector->on_code_object_load(info);
    const auto file_info = m_translator->get_file_code_object_info();
    const auto mem_info  = m_translator->get_mem_code_object_info();
    EXPECT_EQ(file_info.size(), 1);
    EXPECT_EQ(file_info[0].filepath, info.uri);
    EXPECT_EQ(file_info[0].id, info.code_object_id);
    EXPECT_EQ(file_info[0].load_base, info.load_base);
    EXPECT_EQ(file_info[0].load_size, info.load_size);
    EXPECT_TRUE(mem_info.empty());
}

TEST_F(test_pc_sampling_collector_t, ProvidedMemoryCodeObject_PassesItToDecode)
{
    rocprofiler_callback_tracing_code_object_load_data_t info;
    info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_MEMORY;
    info.memory_base    = 0x1000;
    info.memory_size    = 0x2000;
    info.code_object_id = 123;
    info.load_base      = 0x1000;
    info.load_size      = 0x2000;
    m_pc_sampling_collector->on_code_object_load(info);
    const auto file_info = m_translator->get_file_code_object_info();
    const auto mem_info  = m_translator->get_mem_code_object_info();
    EXPECT_EQ(mem_info.size(), 1);
    EXPECT_EQ(mem_info[0].memory_base, info.memory_base);
    EXPECT_EQ(mem_info[0].memory_size, info.memory_size);
    EXPECT_EQ(mem_info[0].id, info.code_object_id);
    EXPECT_EQ(mem_info[0].load_base, info.load_base);
    EXPECT_EQ(mem_info[0].load_size, info.load_size);
    EXPECT_TRUE(file_info.empty());
}

TEST_F(test_pc_sampling_collector_t, ProvidedCodeObjectId_IsWritten)
{
    rocprofiler_callback_tracing_code_object_load_data_t info;
    info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE;
    info.uri            = "test_code_object.co";
    info.code_object_id = 123;
    info.load_base      = 0x1000;
    info.load_size      = 0x2000;
    m_pc_sampling_collector->on_code_object_load(info);
    const auto code_object_ids = m_translator->get_code_object_ids();
    EXPECT_EQ(code_object_ids.size(), 1);
    EXPECT_EQ(code_object_ids[0], info.code_object_id);
}

void test_pc_sampling_collector_t::SetUp()
{
    m_translator            = std::make_shared<mock_code_object_translator_t>();
    m_pc_sampling_collector = std::make_shared<pc_sampling_collector_impl_t>(m_translator);
}