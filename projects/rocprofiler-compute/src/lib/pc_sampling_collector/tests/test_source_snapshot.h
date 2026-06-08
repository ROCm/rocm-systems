// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "mocks.h"
#include "pc_sampling_collector.h"

#include <filesystem>
#include <memory>

// Fixture providing a unique, self-cleaning temporary directory rooted under the
// system temp area for copy_source_files tests.
class test_source_snapshot_t : public ::testing::Test
{
protected:
    void SetUp() override;
    void TearDown() override;

    // Creates a file at the given absolute path (creating parent dirs) with content.
    static void write_file(const std::filesystem::path& path, const std::string& content);

    std::filesystem::path m_tmp;        // sandbox unique to this test
    std::filesystem::path m_src_root;   // where source files are created
    std::filesystem::path m_dest_root;  // copy destination root
};

// Fixture exercising pc_sampling_collector_impl_t::collect_source_paths through the
// existing mock translator.
class test_collect_source_paths_t : public ::testing::Test
{
protected:
    void SetUp() override;

    std::shared_ptr<mock_code_object_translator_t>              m_translator;
    rocprofiler_compute_tool::pc_sampling_collector_impl_t::ptr m_collector;
    rocprofiler_callback_tracing_code_object_load_data_t        m_file_info = {};
};
