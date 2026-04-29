// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "code_object_writer.h"
#include "gtest/gtest.h"

class test_code_object_writer_t : public ::testing::Test
{
protected:
    void SetUp() override;

    rocm_compute::code_object_writer_json_t m_writer;
    const rocm_compute::symbol_t m_symbol0{"sym0", 0x10, 0x1000, 0x20};
    const rocm_compute::symbol_t m_symbol1{"sym1", 0x30, 0x1100, 0x40};
};
