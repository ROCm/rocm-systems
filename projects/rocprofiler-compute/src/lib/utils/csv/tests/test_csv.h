// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <gtest/gtest.h>

#include <string>
#include <vector>

class TestCsv : public ::testing::Test
{
protected:
    /// Formats rows of "id,name" and records how many batches the sink saw.
    std::string format(const std::vector<std::pair<int, std::string>>& rows);

    int m_batches = 0;
};
