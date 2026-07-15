// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/environment.hpp"
#include "common/path.hpp"
#include "filesystem.hpp"

#include <cstdio>
#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

using namespace rocprofsys::common;
using namespace rocprofsys::common::path;

class PathTest : public ::testing::Test
{
protected:
    void SetUp() override { m_test_dir = create_temp_dir(); }

    void TearDown() override { cleanup_temp_dir(m_test_dir); }

    std::string create_temp_dir()
    {
        char  tmpl[] = "/tmp/rocprofsys_path_test_XXXXXX";
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

    std::string create_file(const std::string& name, const std::string& content = "test")
    {
        std::string   path = m_test_dir + "/" + name;
        std::ofstream ofs(path);
        ofs << content;
        return path;
    }

    std::string create_symlink(const std::string& target, const std::string& link_name)
    {
        std::string link_path = m_test_dir + "/" + link_name;
        symlink(target.c_str(), link_path.c_str());
        return link_path;
    }

    std::string create_subdir(const std::string& name)
    {
        std::string path = m_test_dir + "/" + name;
        mkdir(path.c_str(), 0755);
        return path;
    }

    std::string m_test_dir;
};

TEST_F(PathTest, Dirname_StandardPath)
{
    EXPECT_EQ(dirname("/usr/local/bin/program"), "/usr/local/bin");
}

TEST_F(PathTest, Dirname_SingleLevel) { EXPECT_EQ(dirname("/usr/file"), "/usr"); }

TEST_F(PathTest, Dirname_RootFile) { EXPECT_EQ(dirname("/file"), ""); }

TEST_F(PathTest, Dirname_NoSlash) { EXPECT_EQ(dirname("filename"), ""); }

TEST_F(PathTest, Dirname_EmptyString) { EXPECT_EQ(dirname(""), ""); }

TEST_F(PathTest, Dirname_TrailingSlash)
{
    EXPECT_EQ(dirname("/usr/local/"), "/usr/local");
}

TEST_F(PathTest, Dirname_MultipleSlashes)
{
    EXPECT_EQ(dirname("/a/b/c/d/e"), "/a/b/c/d");
}

TEST_F(PathTest, Exists_ExistingFile)
{
    std::string file_path = create_file("existing_file.txt");
    EXPECT_TRUE(exists(file_path));
}

TEST_F(PathTest, Exists_NonexistentFile)
{
    EXPECT_FALSE(exists(m_test_dir + "/nonexistent_file.txt"));
}

TEST_F(PathTest, Exists_ExistingDirectory) { EXPECT_TRUE(exists(m_test_dir)); }

TEST_F(PathTest, Exists_NonexistentDirectory)
{
    EXPECT_FALSE(exists("/nonexistent/path/to/dir"));
}

TEST_F(PathTest, Exists_SymbolicLink)
{
    std::string target    = create_file("target.txt");
    std::string link_path = create_symlink(target, "link_to_target");
    EXPECT_TRUE(exists(link_path));
}

TEST_F(PathTest, Exists_BrokenSymlink)
{
    std::string link_path = create_symlink("/nonexistent/target", "broken_link");
    EXPECT_TRUE(exists(link_path));
}

TEST_F(PathTest, Exists_EmptyPath) { EXPECT_FALSE(exists("")); }

TEST_F(PathTest, IsLink_RegularFile)
{
    std::string file_path = create_file("regular.txt");
    EXPECT_FALSE(is_link(file_path));
}

TEST_F(PathTest, IsLink_Directory) { EXPECT_FALSE(is_link(m_test_dir)); }

TEST_F(PathTest, IsLink_SymbolicLink)
{
    std::string target    = create_file("target.txt");
    std::string link_path = create_symlink(target, "symbolic_link");
    EXPECT_TRUE(is_link(link_path));
}

TEST_F(PathTest, IsLink_NonexistentPath) { EXPECT_FALSE(is_link("/nonexistent/path")); }

TEST_F(PathTest, Readlink_SymbolicLink)
{
    std::string target    = create_file("readlink_target.txt");
    std::string link_path = create_symlink(target, "readlink_link");
    EXPECT_EQ(readlink(link_path), target);
}

TEST_F(PathTest, Readlink_NotALink)
{
    std::string file_path = create_file("not_a_link.txt");
    EXPECT_EQ(readlink(file_path), file_path);
}

TEST_F(PathTest, Readlink_NonexistentPath)
{
    std::string path = "/nonexistent/path";
    EXPECT_EQ(readlink(path), path);
}

TEST_F(PathTest, Realpath_RelativePath)
{
    std::string file_path = create_file("realpath_test.txt");

    char  cwd[PATH_MAX];
    char* cwd_result = getcwd(cwd, PATH_MAX);
    ASSERT_NE(cwd_result, nullptr);

    if(chdir(m_test_dir.c_str()) == 0)
    {
        std::string resolved = realpath("realpath_test.txt");
        EXPECT_EQ(resolved, file_path);
        ASSERT_EQ(chdir(cwd), 0);
    }
}

TEST_F(PathTest, Realpath_AbsolutePath)
{
    std::string file_path = create_file("absolute_test.txt");
    std::string resolved  = realpath(file_path);
    EXPECT_EQ(resolved, file_path);
}

TEST_F(PathTest, Realpath_WithSymlink)
{
    std::string target    = create_file("realpath_target.txt");
    std::string link_path = create_symlink(target, "realpath_link");
    std::string resolved  = realpath(link_path);
    EXPECT_EQ(resolved, target);
}

TEST_F(PathTest, Realpath_NonexistentPath)
{
    std::string nonexistent = "/nonexistent/path/to/file";
    std::string resolved    = realpath(nonexistent);
    EXPECT_EQ(resolved, nonexistent);
}

TEST_F(PathTest, Realpath_WithResolvedOutput)
{
    std::string file_path = create_file("resolved_output_test.txt");
    std::string resolved_output;
    std::string result = realpath(file_path, &resolved_output);
    EXPECT_EQ(result, file_path);
    EXPECT_EQ(resolved_output, file_path);
}

TEST_F(PathTest, IsTextFile_TextFile)
{
    std::string text_content = "This is a text file\nwith multiple lines\n";
    std::string file_path    = create_file("text_file.txt", text_content);
    EXPECT_TRUE(is_text_file(file_path));
}

TEST_F(PathTest, IsTextFile_BinaryFile)
{
    std::string   file_path = m_test_dir + "/binary_file.bin";
    std::ofstream ofs(file_path, std::ios::binary);
    char binary_data[] = { 'H', 'e', 'l', 'l', 'o', '\0', 'W', 'o', 'r', 'l', 'd' };
    ofs.write(binary_data, sizeof(binary_data));
    ofs.close();
    EXPECT_FALSE(is_text_file(file_path));
}

TEST_F(PathTest, IsTextFile_EmptyFile)
{
    std::string file_path = create_file("empty_file.txt", "");
    EXPECT_TRUE(is_text_file(file_path));
}

TEST_F(PathTest, PathType_Directory)
{
    path_type pt(m_test_dir);
    EXPECT_TRUE(pt.exists());
    EXPECT_TRUE(static_cast<bool>(pt));
}

TEST_F(PathTest, PathType_RegularFile)
{
    std::string file_path = create_file("pathtype_file.txt");
    path_type   pt(file_path);
    EXPECT_TRUE(pt.exists());
}

TEST_F(PathTest, PathType_SymbolicLink)
{
    std::string target    = create_file("pathtype_target.txt");
    std::string link_path = create_symlink(target, "pathtype_link");
    path_type   pt(link_path);
    EXPECT_TRUE(pt.exists());
}

TEST_F(PathTest, PathType_Nonexistent)
{
    path_type pt("/nonexistent/path");
    EXPECT_FALSE(pt.exists());
    EXPECT_FALSE(static_cast<bool>(pt));
}

TEST_F(PathTest, GetCwd_ReturnsNonEmpty) { EXPECT_FALSE(get_cwd().empty()); }

TEST_F(PathTest, GetCwd_MatchesGetcwd)
{
    char  buffer[PATH_MAX];
    char* result = ::getcwd(buffer, PATH_MAX);
    ASSERT_NE(result, nullptr);
    EXPECT_EQ(get_cwd(), std::string{ buffer });
}

// NOTE: fully qualify path::basename so overload resolution does not prefer glibc's
// global ::basename(const char*) for string-literal arguments.
TEST_F(PathTest, Basename_StandardPath) { EXPECT_EQ(path::basename("/a/b.so"), "b.so"); }

TEST_F(PathTest, Basename_NoSlash) { EXPECT_EQ(path::basename("a"), "a"); }

TEST_F(PathTest, Basename_RootLevel) { EXPECT_EQ(path::basename("/a"), "a"); }

TEST_F(PathTest, Basename_EmptyString) { EXPECT_EQ(path::basename(""), ""); }

TEST_F(PathTest, Basename_OwningLifetime)
{
    // result must stay valid after the source string is destroyed.
    std::string result;
    {
        std::string source = "/tmp/some/dir/libexample.so";
        result             = path::basename(source);
    }
    EXPECT_EQ(result, "libexample.so");
}

TEST_F(PathTest, IsDirectory_Directory) { EXPECT_TRUE(is_directory(m_test_dir)); }

TEST_F(PathTest, IsDirectory_RegularFile)
{
    std::string file_path = create_file("is_directory_file.txt");
    EXPECT_FALSE(is_directory(file_path));
}

TEST_F(PathTest, IsDirectory_SymlinkToDirectory)
{
    std::string subdir    = create_subdir("is_directory_subdir");
    std::string link_path = create_symlink(subdir, "is_directory_link");
    EXPECT_TRUE(is_directory(link_path));
}

TEST_F(PathTest, IsDirectory_Missing)
{
    EXPECT_FALSE(is_directory(m_test_dir + "/nonexistent_dir"));
}

TEST_F(PathTest, Makedir_SingleLevel)
{
    std::string dir = m_test_dir + "/single";
    EXPECT_EQ(makedir(dir), 0);
    EXPECT_TRUE(is_directory(dir));
}

TEST_F(PathTest, Makedir_Nested)
{
    std::string dir = m_test_dir + "/a/b/c/d";
    EXPECT_EQ(makedir(dir), 0);
    EXPECT_TRUE(is_directory(dir));
}

TEST_F(PathTest, Makedir_IdempotentExisting)
{
    std::string dir = m_test_dir + "/idem";
    EXPECT_EQ(makedir(dir), 0);
    EXPECT_TRUE(is_directory(dir));
    // second call on an already-existing directory must succeed (EEXIST tolerated)
    EXPECT_EQ(makedir(dir), 0);
    EXPECT_TRUE(is_directory(dir));
}

TEST_F(PathTest, Makedir_RelativePathResolvedViaCwd)
{
    char  cwd[PATH_MAX];
    char* cwd_result = getcwd(cwd, PATH_MAX);
    ASSERT_NE(cwd_result, nullptr);

    ASSERT_EQ(chdir(m_test_dir.c_str()), 0);
    EXPECT_EQ(makedir("rel/sub"), 0);
    EXPECT_TRUE(is_directory(m_test_dir + "/rel/sub"));
    ASSERT_EQ(chdir(cwd), 0);
}

TEST_F(PathTest, Makedir_DotAndDotDotComponents)
{
    // '.' is skipped and '..' pops a component: x/./y/../z resolves to x/z
    std::string dir = m_test_dir + "/x/./y/../z";
    EXPECT_EQ(makedir(dir), 0);
    EXPECT_TRUE(is_directory(m_test_dir + "/x/z"));
}

TEST_F(PathTest, Makedir_FailureReturnsNonZero)
{
    // a regular file as a parent component makes mkdir fail with ENOTDIR (not EEXIST),
    // so makedir must return -1 — the load-bearing failure contract for the open/fopen
    // shim
    std::string blocker = create_file("blocker");
    EXPECT_EQ(makedir(blocker + "/child"), -1);
}

TEST_F(PathTest, OpenOfstream_CreatesParentTreeAndOpens)
{
    std::string   file_path = m_test_dir + "/new/nested/tree/out.txt";
    std::ofstream ofs;
    EXPECT_TRUE(open(ofs, file_path));
    EXPECT_TRUE(ofs.is_open());
    ofs << "hello";
    ofs.close();
    EXPECT_TRUE(is_directory(m_test_dir + "/new/nested/tree"));
    EXPECT_TRUE(exists(file_path));
}

TEST_F(PathTest, OpenOfstream_ForwardsBinaryFlag)
{
    std::string   file_path = m_test_dir + "/bin/data.bin";
    std::ofstream ofs;
    EXPECT_TRUE(open(ofs, file_path, std::ios::out | std::ios::binary));
    EXPECT_TRUE(ofs.is_open());
    char payload[] = { 'A', '\0', 'B' };
    ofs.write(payload, sizeof(payload));
    ofs.close();
    // a binary write of 3 bytes (incl. an embedded NUL) must land verbatim
    std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
    ASSERT_TRUE(ifs.is_open());
    char readback[sizeof(payload)] = {};
    ifs.read(readback, sizeof(readback));
    EXPECT_EQ(ifs.gcount(), static_cast<std::streamsize>(sizeof(payload)));
    EXPECT_EQ(std::string(readback, sizeof(readback)),
              std::string(payload, sizeof(payload)));
}

TEST_F(PathTest, OpenOfstream_FallsBackWhenParentUncreatable)
{
    // a regular file as an intermediate parent component makes makedir fail with ENOTDIR
    // (a deeper level than the file, so it is not tolerated as EEXIST); the ofstream shim
    // must fall back to ./<base> in the current directory and still open successfully
    std::string blocker = create_file("open_blocker");

    char  cwd[PATH_MAX];
    char* cwd_result = getcwd(cwd, PATH_MAX);
    ASSERT_NE(cwd_result, nullptr);
    ASSERT_EQ(chdir(m_test_dir.c_str()), 0);

    std::ofstream ofs;
    EXPECT_TRUE(open(ofs, blocker + "/sub/fallback.txt"));
    EXPECT_TRUE(ofs.is_open());
    ofs.close();
    // the fallback file is ./fallback.txt (i.e. <cwd>/fallback.txt)
    EXPECT_TRUE(exists(m_test_dir + "/fallback.txt"));

    ASSERT_EQ(chdir(cwd), 0);
}

TEST_F(PathTest, OpenIfstream_OpensExistingNoDirCreation)
{
    std::string   file_path = create_file("readme.txt", "content");
    std::ifstream ifs;
    EXPECT_TRUE(open(ifs, file_path));
    EXPECT_TRUE(ifs.is_open());
    ifs.close();
}

TEST_F(PathTest, OpenIfstream_MissingFileReturnsFalse)
{
    std::string   file_path = m_test_dir + "/does/not/exist.txt";
    std::ifstream ifs;
    EXPECT_FALSE(open(ifs, file_path));
    EXPECT_FALSE(ifs.is_open());
    // must NOT have created the directory tree
    EXPECT_FALSE(is_directory(m_test_dir + "/does"));
}

TEST_F(PathTest, Fopen_CreatesParentTreeAndReturnsHandle)
{
    std::string file_path = m_test_dir + "/fnew/nested/data.txt";
    std::FILE*  fp        = fopen(file_path, "w");
    ASSERT_NE(fp, nullptr);
    std::fputs("hello", fp);
    std::fclose(fp);
    EXPECT_TRUE(is_directory(m_test_dir + "/fnew/nested"));
    EXPECT_TRUE(exists(file_path));
}

TEST_F(PathTest, GetRocprofsysRoot_ReturnsNonEmpty)
{
    std::string root = get_rocprofsys_root();
    EXPECT_FALSE(root.empty());
}

TEST_F(PathTest, GetRocprofsysRoot_EndsWithParentDir)
{
    std::string root = get_rocprofsys_root();
    EXPECT_TRUE(root.length() >= 2);
    EXPECT_EQ(root.substr(root.length() - 2), "..");
}

TEST_F(PathTest, GetInternalLibdir_ContainsLib)
{
    std::string libdir = get_internal_libdir();
    EXPECT_NE(libdir.find("lib"), std::string::npos);
}

TEST_F(PathTest, GetInternalScriptPath_ContainsLibexec)
{
    std::string script_path = get_internal_script_path();
    EXPECT_NE(script_path.find("libexec"), std::string::npos);
    EXPECT_NE(script_path.find("rocprofiler-systems"), std::string::npos);
}

TEST_F(PathTest, GetInternalLibpath_ContainsLibName)
{
    std::string libpath = get_internal_libpath("librocprof-sys.so");
    EXPECT_NE(libpath.find("librocprof-sys.so"), std::string::npos);
}

TEST_F(PathTest, GetInternalLibpath_ContainsLib)
{
    std::string libpath = get_internal_libpath("test.so");
    EXPECT_NE(libpath.find("lib"), std::string::npos);
}

TEST_F(PathTest, GetDefaultLibSearchPaths_ReturnsNonEmpty)
{
    auto paths = get_default_lib_search_paths<std::string>();
    EXPECT_FALSE(paths.empty());
}

TEST_F(PathTest, GetDefaultLibSearchPaths_AsVector)
{
    auto paths = get_default_lib_search_paths<std::vector<std::string>>();
    EXPECT_FALSE(paths.empty());
}

TEST_F(PathTest, FindPath_AbsoluteExisting)
{
    std::string file_path = create_file("findpath_test.txt");
    std::string result    = find_path(file_path, 0);
    EXPECT_EQ(result, file_path);
}

TEST_F(PathTest, FindPath_NonexistentReturnsOriginal)
{
    std::string nonexistent = "nonexistent_file_xyz.txt";
    std::string result      = find_path(nonexistent, 0);
    EXPECT_EQ(result, nonexistent);
}

TEST_F(PathTest, FindPath_InSearchPath)
{
    std::string file_path = create_file("searchable.txt");
    std::string result    = find_path("searchable.txt", 0, m_test_dir);
    EXPECT_EQ(result, file_path);
}

TEST_F(PathTest, Dirname_ComplexPath)
{
    EXPECT_EQ(dirname("/opt/rocm/lib/rocprofiler-systems/librocprof-sys.so"),
              "/opt/rocm/lib/rocprofiler-systems");
}

TEST_F(PathTest, ChainedSymlinks)
{
    std::string target     = create_file("chain_target.txt");
    std::string link1      = create_symlink(target, "chain_link1");
    std::string link2_path = m_test_dir + "/chain_link2";
    symlink("chain_link1", link2_path.c_str());

    EXPECT_TRUE(is_link(link1));
    EXPECT_TRUE(is_link(link2_path));

    std::string resolved = realpath(link2_path);
    EXPECT_EQ(resolved, target);
}

TEST_F(PathTest, Exists_SpecialCharactersInPath)
{
    std::string file_path = create_file("file with spaces.txt");
    EXPECT_TRUE(exists(file_path));
}

TEST_F(PathTest, Dirname_RocprofsysTypicalPath)
{
    std::string path   = "/opt/rocm-6.0.0/lib/rocprofiler-systems/librocprof-sys-dl.so";
    std::string result = dirname(path);
    EXPECT_EQ(result, "/opt/rocm-6.0.0/lib/rocprofiler-systems");
}

TEST_F(PathTest, NestedDirectories)
{
    std::string subdir1 = create_subdir("level1");
    std::string subdir2 = subdir1 + "/level2";
    mkdir(subdir2.c_str(), 0755);
    std::string subdir3 = subdir2 + "/level3";
    mkdir(subdir3.c_str(), 0755);

    EXPECT_TRUE(exists(subdir1));
    EXPECT_TRUE(exists(subdir2));
    EXPECT_TRUE(exists(subdir3));

    EXPECT_EQ(dirname(subdir3), subdir2);
    EXPECT_EQ(dirname(subdir2), subdir1);
}
