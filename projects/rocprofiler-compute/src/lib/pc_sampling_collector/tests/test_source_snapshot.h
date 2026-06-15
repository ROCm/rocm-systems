// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#pragma once

#include "gtest/gtest.h"
#include "source_snapshot.h"

#include <filesystem>
#include <fstream>
#include <string>

class test_source_snapshot_t : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_tmp_root    = std::filesystem::temp_directory_path() /
                        ("rpc_source_snapshot_" + std::to_string(::getpid()) + "_" +
                         std::to_string(reinterpret_cast<uintptr_t>(this)));
        m_output_root = m_tmp_root / "out";

        m_file_a = m_tmp_root / "proj" / "a.cpp";
        m_file_b = m_tmp_root / "proj" / "sub" / "b.cpp";

        std::filesystem::create_directories(m_file_a.parent_path());
        std::filesystem::create_directories(m_file_b.parent_path());
        std::filesystem::create_directories(m_output_root);

        write_file(m_file_a, m_contents_a);
        write_file(m_file_b, m_contents_b);
    }

    void TearDown() override
    {
        std::error_code ec;
        std::filesystem::remove_all(m_tmp_root, ec);
    }

    static void write_file(const std::filesystem::path& path, const std::string& contents)
    {
        std::ofstream ofs(path);
        ofs << contents;
    }

    static std::string read_file(const std::filesystem::path& path)
    {
        std::ifstream ifs(path);
        return std::string(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
    }

    rocprofiler_compute_tool::source_snapshot_impl_t m_snapshotter;

    std::filesystem::path m_tmp_root;
    std::filesystem::path m_output_root;
    std::filesystem::path m_file_a;
    std::filesystem::path m_file_b;
    const std::string     m_contents_a{"contents of a.cpp\n"};
    const std::string     m_contents_b{"contents of sub/b.cpp\n"};
};
