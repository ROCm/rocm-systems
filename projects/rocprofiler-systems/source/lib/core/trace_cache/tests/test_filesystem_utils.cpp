// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "common/filesystem.hpp"

#include <algorithm>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <vector>

namespace fs = rocprofsys::common::fs;

class FilesystemUtilsTest : public ::testing::Test
{
protected:
    void SetUp() override { m_test_dir = create_temp_dir(); }

    void TearDown() override { cleanup_temp_dir(m_test_dir); }

    std::string create_temp_dir()
    {
        std::error_code ec;
        auto            temp_base = fs::temp_directory_path(ec);
        if(ec)
        {
            throw std::runtime_error("Failed to get temp directory path: " +
                                     ec.message());
        }

        auto        temp_path = temp_base / "rocprofsys_fs_test_XXXXXX";
        std::string path_str  = temp_path.string();

        std::vector<char> path_buf(path_str.begin(), path_str.end());
        path_buf.push_back('\0');

        char* dir = mkdtemp(path_buf.data());
        if(!dir)
        {
            throw std::runtime_error("Failed to create temp directory");
        }
        return std::string{ dir };
    }

    void cleanup_temp_dir(const std::string& dir)
    {
        if(dir.empty()) return;
        std::error_code ec;
        fs::remove_all(dir, ec);
    }

    void create_file(const std::string& name)
    {
        std::ofstream ofs(m_test_dir + "/" + name);
        ofs << "test content";
    }

    void create_subdir(const std::string& name)
    {
        std::error_code ec;
        fs::create_directory(m_test_dir + "/" + name, ec);
    }

    std::string m_test_dir;
};

TEST_F(FilesystemUtilsTest, DirectoryIterator_EmptyDirectory)
{
    std::vector<std::string> files;
    std::error_code          ec;

    for(const auto& entry : fs::directory_iterator(m_test_dir, ec))
    {
        files.emplace_back(entry.path().filename().string());
    }

    EXPECT_TRUE(files.empty());
    EXPECT_FALSE(ec);
}

TEST_F(FilesystemUtilsTest, DirectoryIterator_WithFiles)
{
    create_file("file1.txt");
    create_file("file2.txt");
    create_file("file3.bin");

    std::vector<std::string> files;
    std::error_code          ec;

    for(const auto& entry : fs::directory_iterator(m_test_dir, ec))
    {
        files.emplace_back(entry.path().filename().string());
    }

    EXPECT_FALSE(ec);
    EXPECT_EQ(files.size(), 3u);
    EXPECT_TRUE(std::find(files.begin(), files.end(), "file1.txt") != files.end());
    EXPECT_TRUE(std::find(files.begin(), files.end(), "file2.txt") != files.end());
    EXPECT_TRUE(std::find(files.begin(), files.end(), "file3.bin") != files.end());
}

TEST_F(FilesystemUtilsTest, DirectoryIterator_WithSubdirectories)
{
    create_file("file.txt");
    create_subdir("subdir1");
    create_subdir("subdir2");

    std::vector<std::string> entries;
    std::error_code          ec;

    for(const auto& entry : fs::directory_iterator(m_test_dir, ec))
    {
        entries.emplace_back(entry.path().filename().string());
    }

    EXPECT_FALSE(ec);
    EXPECT_EQ(entries.size(), 3u);
}

TEST_F(FilesystemUtilsTest, DirectoryIterator_NonexistentDirectory)
{
    std::error_code ec;
    auto            iter = fs::directory_iterator("/nonexistent/path/xyz", ec);

    EXPECT_TRUE(ec);
}

TEST_F(FilesystemUtilsTest, Exists_ExistingDirectory)
{
    std::error_code ec;
    EXPECT_TRUE(fs::exists(m_test_dir, ec));
    EXPECT_FALSE(ec);
}

TEST_F(FilesystemUtilsTest, Exists_NonexistentPath)
{
    std::error_code ec;
    EXPECT_FALSE(fs::exists("/nonexistent/path/xyz", ec));
}

TEST_F(FilesystemUtilsTest, IsDirectory_Directory)
{
    std::error_code ec;
    EXPECT_TRUE(fs::is_directory(m_test_dir, ec));
    EXPECT_FALSE(ec);
}

TEST_F(FilesystemUtilsTest, IsDirectory_File)
{
    create_file("test.txt");
    std::error_code ec;
    EXPECT_FALSE(fs::is_directory(m_test_dir + "/test.txt", ec));
    EXPECT_FALSE(ec);
}

TEST_F(FilesystemUtilsTest, RemoveAll_DirectoryWithContents)
{
    create_file("file1.txt");
    create_subdir("subdir");
    std::ofstream ofs(m_test_dir + "/subdir/nested.txt");
    ofs << "nested content";
    ofs.close();

    std::error_code ec;
    EXPECT_TRUE(fs::exists(m_test_dir, ec));

    fs::remove_all(m_test_dir, ec);
    EXPECT_FALSE(ec);
    EXPECT_FALSE(fs::exists(m_test_dir, ec));

    m_test_dir.clear();
}

TEST_F(FilesystemUtilsTest, CacheFilePattern_BufferedStorage)
{
    create_file("buffered_storage_1234_5678.bin");
    create_file("metadata_1234_5678.json");
    create_file("other_file.txt");

    std::vector<std::string> cache_files;
    std::error_code          ec;

    for(const auto& entry : fs::directory_iterator(m_test_dir, ec))
    {
        std::string filename = entry.path().filename().string();
        if(filename.find("buffered_storage_") == 0 || filename.find("metadata_") == 0)
        {
            cache_files.emplace_back(filename);
        }
    }

    EXPECT_FALSE(ec);
    EXPECT_EQ(cache_files.size(), 2u);
}
