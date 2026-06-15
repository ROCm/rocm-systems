// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "pc_sample_writer.h"

class test_pc_string_table_t : public ::testing::Test
{
protected:
    rocprofiler_compute_tool::pc_string_table_t m_string_table;
};
