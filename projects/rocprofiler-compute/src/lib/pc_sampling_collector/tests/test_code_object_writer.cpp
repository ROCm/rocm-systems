// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_code_object_writer.h"
#include "nlohmann/json.hpp"

TEST_F(test_code_object_writer_t, ProvidedNoData_ReturnsMinimalJson)
{
    const auto& result = m_writer.get_result();
    EXPECT_TRUE(!result.empty());
}

void test_code_object_writer_t::SetUp()
{
}