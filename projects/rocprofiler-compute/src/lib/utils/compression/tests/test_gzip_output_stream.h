// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

// Decompress with zlib directly, independent of the code under test.
std::string gunzip(const std::string& compressed);

class TestGzipOutputStream : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

    std::string compress(const std::string& data);
    std::string raw_bytes() const;

    std::filesystem::path m_path;
};
