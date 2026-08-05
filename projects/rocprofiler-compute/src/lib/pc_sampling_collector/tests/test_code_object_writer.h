// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "code_object_writer.h"
#include "gtest/gtest.h"

class test_code_object_writer_t : public ::testing::Test
{
protected:
    rocprofiler_compute_tool::code_object_writer_json_t m_writer;
    const rocprofiler_compute_tool::symbol_t            m_symbol0{"sym0", 0x10, 0x1000, 0x20};
    const rocprofiler_compute_tool::symbol_t            m_symbol1{"sym1", 0x30, 0x1100, 0x40};
    const rocprofiler_compute_tool::instruction_t m_inst0{"s_mov_b32", "// mov", 0x1000, 0x10, 4};
    const rocprofiler_compute_tool::instruction_t m_inst1{"s_add_i32", "// add", 0x1004, 0x14, 4};
};
