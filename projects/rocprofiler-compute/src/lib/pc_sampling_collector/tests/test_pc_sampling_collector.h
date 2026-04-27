// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once
#include "pc_sampling_collector.h"
#include "gtest/gtest.h"

class test_pc_sampling_collector_t : public ::testing::Test
{
protected:
	rocprofiler_compute_tool::pc_sampling_collector_impl_t::Ptr m_pc_sampling_collector;
};
