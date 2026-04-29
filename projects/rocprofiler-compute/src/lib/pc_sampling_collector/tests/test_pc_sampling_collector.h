// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sampling_collector.h"
#include "mocks.h"
#include "gtest/gtest.h"

class test_pc_sampling_collector_t : public ::testing::Test
{
protected:
	void SetUp() override;

	std::shared_ptr<mock_code_object_translator_t>       m_translator;
	rocm_compute::pc_sampling_collector_impl_t::ptr      m_pc_sampling_collector;
    rocprofiler_callback_tracing_code_object_load_data_t m_mem_info = {};
    rocprofiler_callback_tracing_code_object_load_data_t m_file_info = {};
};
