// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

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
        const std::string lib_path = dir + "/libomptarget.so";
        std::ofstream     ofs(lib_path);
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
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, FindsLibInRocmPathLibLlvmLib)
{
    const std::string rocm_path    = m_test_dir + "/rocm";
    const std::string lib_llvm_lib = rocm_path + "/lib/llvm/lib";
    create_libomptarget(lib_llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, lib_llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, PrefersLlvmLibOverLibLlvmLib)
{
    const std::string rocm_path    = m_test_dir + "/rocm";
    const std::string llvm_lib     = rocm_path + "/llvm/lib";
    const std::string lib_llvm_lib = rocm_path + "/lib/llvm/lib";
    create_libomptarget(llvm_lib);
    create_libomptarget(lib_llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLlvmLib)
{
    const std::string rocmv_dir = m_test_dir + "/rocm-version";
    const std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    create_libomptarget(llvm_lib);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLib)
{
    const std::string rocmv_dir = m_test_dir + "/rocm-version";
    const std::string lib_dir   = rocmv_dir + "/lib";
    create_libomptarget(lib_dir);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, lib_dir);
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirTakesPrecedence)
{
    const std::string rocm_path     = m_test_dir + "/rocm";
    const std::string rocmv_dir     = m_test_dir + "/rocm-version";
    const std::string rocm_llvm_lib = rocm_path + "/llvm/lib";
    const std::string rocmv_lib     = rocmv_dir + "/llvm/lib";
    create_libomptarget(rocm_llvm_lib);
    create_libomptarget(rocmv_lib);

    set_rocm_path(rocm_path);
    set_rocmv_dir(rocmv_dir);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, rocmv_lib);
}

TEST_F(DiscoverLlvmLibdirTest, NoLibomptargetFound_InCustomPath)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    create_directory(rocm_path + "/llvm/lib");

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    // Result should either be empty (no ROCm installed) or one of the default
    // fallback paths if ROCm is installed on the system
    if(!result.empty())
    {
        const bool is_default_path =
            (result == "/opt/rocm/llvm/lib" || result == "/opt/rocm/lib/llvm/lib");
        EXPECT_TRUE(is_default_path)
            << "Expected empty or default ROCm path, got: " << result;
    }
}

TEST_F(DiscoverLlvmLibdirTest, TrailingSlashInRocmPath)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path + "/");
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, TrailingSlashInRocmVersionDir)
{
    const std::string rocmv_dir = m_test_dir + "/rocm-version";
    const std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    create_libomptarget(llvm_lib);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir + "/");

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, EmptyRocmVersionDir)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    setenv("ROCmVersion_DIR", "", 1);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, FoundLibDoesNotThrow)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    EXPECT_NO_THROW({ discover_llvm_libdir_for_ompt(); });
}

TEST_F(DiscoverLlvmLibdirTest, NoLibFoundDoesNotThrow)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    create_directory(rocm_path);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    EXPECT_NO_THROW({ discover_llvm_libdir_for_ompt(); });
}

TEST_F(DiscoverLlvmLibdirTest, RocmVersionDirLlvmLibPreferredOverLib)
{
    const std::string rocmv_dir = m_test_dir + "/rocm-version";
    const std::string llvm_lib  = rocmv_dir + "/llvm/lib";
    const std::string lib_dir   = rocmv_dir + "/lib";
    create_libomptarget(llvm_lib);
    create_libomptarget(lib_dir);

    clear_rocm_path();
    set_rocmv_dir(rocmv_dir);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, FallsBackToRocmPathWhenRocmVersionDirHasNoLib)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string rocmv_dir = m_test_dir + "/rocm-version";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_directory(rocmv_dir + "/llvm/lib");
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path);
    set_rocmv_dir(rocmv_dir);

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, MultipleTrailingSlashes)
{
    const std::string rocm_path = m_test_dir + "/rocm";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";
    create_libomptarget(llvm_lib);

    set_rocm_path(rocm_path + "/");
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_FALSE(result.empty());
    EXPECT_NE(result.find("llvm/lib"), std::string::npos);
}

TEST_F(DiscoverLlvmLibdirTest, PathWithSpaces)
{
    const std::string rocm_path = m_test_dir + "/rocm with spaces";
    const std::string llvm_lib  = rocm_path + "/llvm/lib";

    std::error_code ec;
    test_common::fs::create_directories(llvm_lib, ec);

    const std::string lib_path = llvm_lib + "/libomptarget.so";
    std::ofstream     ofs(lib_path);
    ofs << "fake library content";
    ofs.close();

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}

TEST_F(DiscoverLlvmLibdirTest, BothEnvVarsUnset_UsesDefaultRocmPath)
{
    clear_rocm_path();
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    if(!result.empty())
    {
        EXPECT_TRUE(result.find("/opt/rocm") != std::string::npos ||
                    result.find("llvm/lib") != std::string::npos);
    }
}

TEST_F(DiscoverLlvmLibdirTest, SymlinkToLibomptarget)
{
    const std::string rocm_path   = m_test_dir + "/rocm";
    const std::string llvm_lib    = rocm_path + "/llvm/lib";
    const std::string actual_dir  = m_test_dir + "/actual";
    const std::string actual_file = actual_dir + "/libomptarget.so";

    create_directory(actual_dir);
    std::ofstream ofs(actual_file);
    ofs << "fake library content";
    ofs.close();

    create_directory(llvm_lib);
    const std::string link_path = llvm_lib + "/libomptarget.so";
    EXPECT_EQ(symlink(actual_file.c_str(), link_path.c_str()), 0);

    set_rocm_path(rocm_path);
    clear_rocmv_dir();

    const std::string result = discover_llvm_libdir_for_ompt();

    EXPECT_EQ(result, llvm_lib);
}
