// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "mocks.h"
#include "source_snapshotter.h"

#include <filesystem>
#include <map>
#include <memory>
#include <string>

class test_source_snapshotter_t : public ::testing::Test
{
protected:
    void SetUp() override;

    using source_path_map_t = std::map<std::filesystem::path, std::filesystem::path>;

    std::filesystem::path destination_path(const std::filesystem::path& source_path) const;
    void                  set_regular_source(const std::filesystem::path& source_path);
    void                  expect_no_copy() const;

    // The map this process would have written, and what the snapshotter wrote.
    static std::string source_path_map_json(const source_path_map_t& source_path_map);
    std::string        written_source_path_map() const;
    void               expect_no_source_path_map() const;

    std::shared_ptr<mock_filesystem_wrapper_t>                           m_filesystem;
    std::shared_ptr<rocprofiler_compute_tool::source_snapshotter_impl_t> m_snapshotter;
    const std::filesystem::path m_destination_root = "/snapshot";
};
