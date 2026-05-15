// MIT License
//
// Copyright (c) 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common/environment.hpp"
#include "filesystem.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace rocprofsys::common;

class DiscoverLlvmLibdirTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        m_test_dir = create_temp_dir();
        save_env_vars();
    }

    void TearDown() override
    {
        restore_env_vars();
        cleanup_temp_dir(m_test_dir);
    }

    std::string create_temp_dir()
    {
        char  tmpl[] = "/tmp/rocprofsys_llvm_test_XXXXXX";
        char* dir    = mkdtemp(tmpl);
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
        test_common::fs::remove_all(dir, ec);
    }

    void create_directory(const std::string& path)
    {
        std::error_code ec;
        test_common::fs::create_directories(path, ec);
    }

    void create_libomptarget(const std::string& dir)
    {
        create_directory(dir);
        std::string   lib_path = dir + "/libomptarget.so";
        std::ofstream ofs(lib_path);
        ofs << "fake library content";
    }

    void save_env_vars()
    {
        char* rocm_path = getenv("ROCM_PATH");
        char* rocmv_dir = getenv("ROCmVersion_DIR");
        m_saved_rocm_path =
            rocm_path ? std::optional<std::string>{ rocm_path } : std::nullopt;
        m_saved_rocmv_dir =
            rocmv_dir ? std::optional<std::string>{ rocmv_dir } : std::nullopt;
    }

    void restore_env_vars()
    {
        if(m_saved_rocm_path.has_value())
            setenv("ROCM_PATH", m_saved_rocm_path->c_str(), 1);
        else
            unsetenv("ROCM_PATH");

        if(m_saved_rocmv_dir.has_value())
            setenv("ROCmVersion_DIR", m_saved_rocmv_dir->c_str(), 1);
        else
            unsetenv("ROCmVersion_DIR");
    }

    void set_rocm_path(const std::string& path) { setenv("ROCM_PATH", path.c_str(), 1); }

    void set_rocmv_dir(const std::string& path)
    {
        setenv("ROCmVersion_DIR", path.c_str(), 1);
    }

    void clear_rocm_path() { unsetenv("ROCM_PATH"); }

    void clear_rocmv_dir() { unsetenv("ROCmVersion_DIR"); }

    std::string                m_test_dir;
    std::optional<std::string> m_saved_rocm_path;
    std::optional<std::string> m_saved_rocmv_dir;
};

TEST_F(DiscoverLlvmLibdirTest, FindsLibInRocmPathLlvmLib)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, FindsLibInRocmPathLibLlvmLib)
{
    std::string rocm_path    = m_test_dir + "/rocm";
    std::string lib_llvm_lib = rocm_path + "/lib/llvm/lib";
    create_libomptarget(lib_llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, lib_llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, PrefersLlvmLibOverLibLlvmLib)
{
    std::string rocm_path    = m_test_dir + "/rocm";
    std::string llvm_lib     = rocm_path + "/llvm/lib";
    std::string lib_llvm_lib = rocm_path + "/lib/llvm/lib";
    create_libomptarget(llvm_lib);
    create_libomptarget(lib_llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLlvmLib)
{
    std::string rocmv_dir = m_test_dir + "/rocm-version";
    std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    create_libomptarget(llvm_lib);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLib)
{
    std::string rocmv_dir = m_test_dir + "/rocm-version";
    std::string lib_dir   = rocmv_dir + "/lib";
    create_libomptarget(lib_dir);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, lib_dir);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirTakesPrecedence)
{
    std::string rocm_path     = m_test_dir + "/rocm";
    std::string rocmv_dir     = m_test_dir + "/rocm-version";
    std::string rocm_llvm_lib = rocm_path + "/llvm/lib";
    std::string rocmv_lib     = rocmv_dir + "/llvm/lib";
    create_libomptarget(rocm_llvm_lib);
    create_libomptarget(rocmv_lib);

    set_rocm_path(rocm_path);
    set_rocmv_dir(rocmv_dir);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, rocmv_lib);
}

TEST_F(DiscoverLlvmLibdirTest, NoLibomptargetFound_InCustomPath)
{
    std::string rocm_path = m_test_dir + "/rocm";
    create_directory(rocm_path + "/llvm/lib");

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    // Result should either be empty (no ROCm installed) or one of the default
    // fallback paths if ROCm is installed on the system
    if(!result.empty())
    {
        bool is_default_path =
            (result == "/opt/rocm/llvm/lib" || result == "/opt/rocm/lib/llvm/lib");
        EXPECT_TRUE(is_default_path)
            << "Expected empty or default ROCm path, got: " << result;
    }
}

TEST_F(DiscoverLlvmLibdirTest, TrailingSlashInRocmPath)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path + "/");
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, TrailingSlashInRocmVersionDir)
{
    std::string rocmv_dir = m_test_dir + "/rocm-version";
    std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    create_libomptarget(llvm_lib);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir + "/");

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, EmptyRocmVersionDir)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    setenv("ROCmVersion_DIR", "", 1);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, VerboseModeDoesNotCrash)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    EXPECT_NO_THROW({ discover_llvm_libdir_for_ompt(true); });
}

TEST_F(DiscoverLlvmLibdirTest, VerboseModeNoLibFound)
{
    std::string rocm_path = m_test_dir + "/rocm";
    create_directory(rocm_path);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    EXPECT_NO_THROW({ discover_llvm_libdir_for_ompt(true); });
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLlvmLibPreferredOverLib)
{
    std::string rocmv_dir = m_test_dir + "/rocm-version";
    std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    std::string lib_dir   = rocmv_dir + "/lib";
    create_libomptarget(llvm_lib);
    create_libomptarget(lib_dir);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, FallsBackToRocmPathWhenRocmVersionDirHasNoLib)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string rocmv_dir = m_test_dir + "/rocm-version";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_directory(rocmv_dir + "/llvm/lib");
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    set_rocmv_dir(rocmv_dir);

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, MultipleTrailingSlashes)
{
    std::string rocm_path = m_test_dir + "/rocm";
    std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path + "/");
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("llvm/lib"), std::string::npos);
}

TEST_F(DiscoverLlvmLibdirTest, PathWithSpaces)
{
    std::string rocm_path = m_test_dir + "/rocm with spaces";
    std::string llvm_lib  = rocm_path + "/llvm/lib";

    std::error_code ec;
    test_common::fs::create_directories(llvm_lib, ec);

    std::string   lib_path = llvm_lib + "/libomptarget.so";
    std::ofstream ofs(lib_path);
    ofs << "fake library content";
    ofs.close();

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, BothEnvVarsUnset_UsesDefaultRocmPath)
{
    clear_rocm_path();
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    if(!result.empty())
    {
        EXPECT_TRUE(result.find("/opt/rocm") != std::string::npos ||
                    result.find("llvm/lib") != std::string::npos);
    }
}

TEST_F(DiscoverLlvmLibdirTest, SymlinkToLibomptarget)
{
    std::string rocm_path   = m_test_dir + "/rocm";
    std::string llvm_lib    = rocm_path + "/llvm/lib";
    std::string actual_dir  = m_test_dir + "/actual";
    std::string actual_file = actual_dir + "/libomptarget.so";

    create_directory(actual_dir);
    std::ofstream ofs(actual_file);
    ofs << "fake library content";
    ofs.close();

    create_directory(llvm_lib);
    std::string link_path = llvm_lib + "/libomptarget.so";
    symlink(actual_file.c_str(), link_path.c_str());

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}
