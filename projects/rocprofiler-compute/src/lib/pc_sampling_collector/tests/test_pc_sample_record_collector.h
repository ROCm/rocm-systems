// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "gtest/gtest.h"
#include "mocks.h"
#include "pc_sample_record_collector.h"
#include "pc_sampling_record.h"

#include <memory>

class test_pc_sample_record_collector_t : public ::testing::Test
{
protected:
    void SetUp() override;

    static constexpr uint64_t k_code_object_id = 222;
    static constexpr uint64_t k_load_base      = 0x1000;
    static constexpr uint64_t k_offset_a       = 0x10;
    static constexpr uint64_t k_offset_b       = 0x20;

    std::shared_ptr<mock_code_object_translator_t>              m_translator;
    mock_ps_file_writer_t                                       m_writer;
    rocprofiler_compute_tool::pc_sample_record_collector_t::ptr m_collector;
    rocprofiler_callback_tracing_code_object_load_data_t        m_file_info = {};
};
