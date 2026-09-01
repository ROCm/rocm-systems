// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/path.hpp"
#include "filesystem.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <sys/stat.h>
#include <system_error>
#include <unistd.h>

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
        EXPECT_EQ(symlink(target.c_str(), link_path.c_str()), 0);
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

TEST_F(PathTest, ParentPath_StandardPath)
{
    EXPECT_EQ(parent_path("/usr/local/bin/program"), "/usr/local/bin");
}

TEST_F(PathTest, ParentPath_SingleLevel) { EXPECT_EQ(parent_path("/usr/file"), "/usr"); }

TEST_F(PathTest, ParentPath_RootFile) { EXPECT_EQ(parent_path("/file"), "/"); }

TEST_F(PathTest, ParentPath_NoSlash) { EXPECT_EQ(parent_path("filename"), ""); }

TEST_F(PathTest, ParentPath_EmptyString) { EXPECT_EQ(parent_path(""), ""); }

TEST_F(PathTest, ParentPath_TrailingSlash)
{
    EXPECT_EQ(parent_path("/usr/local/"), "/usr/local");
}

TEST_F(PathTest, ParentPath_MultipleSlashes)
{
    EXPECT_EQ(parent_path("/a/b/c/d/e"), "/a/b/c/d");
}

TEST_F(PathTest, ReadSymlink_SymbolicLink)
{
    const std::string target    = create_file("read_symlink_target.txt");
    const std::string link_path = create_symlink(target, "read_symlink_link");
    EXPECT_EQ(read_symlink(link_path), target);
}

TEST_F(PathTest, ReadSymlink_NotALink)
{
    const std::string file_path = create_file("not_a_link.txt");
    EXPECT_EQ(read_symlink(file_path), file_path);
}

TEST_F(PathTest, ReadSymlink_NonexistentPath)
{
    const std::string path = "/nonexistent/path";
    EXPECT_EQ(read_symlink(path), path);
}

TEST_F(PathTest, ReadSymlink_BrokenSymlink)
{
    const std::string link_path =
        create_symlink("/nonexistent/target", "broken_read_symlink");
    EXPECT_EQ(read_symlink(link_path), "/nonexistent/target");
}

TEST_F(PathTest, Realpath_RelativePath)
{
    const std::string file_path = create_file("realpath_test.txt");

    char  cwd[PATH_MAX];
    char* cwd_result = getcwd(cwd, PATH_MAX);
    ASSERT_NE(cwd_result, nullptr);

    if(chdir(m_test_dir.c_str()) == 0)
    {
        const std::string resolved = realpath("realpath_test.txt");
        EXPECT_EQ(resolved, file_path);
        ASSERT_EQ(chdir(cwd), 0);
    }
}

TEST_F(PathTest, Realpath_AbsolutePath)
{
    const std::string file_path = create_file("absolute_test.txt");
    const std::string resolved  = realpath(file_path);
    EXPECT_EQ(resolved, file_path);
}

TEST_F(PathTest, Realpath_WithSymlink)
{
    const std::string target    = create_file("realpath_target.txt");
    const std::string link_path = create_symlink(target, "realpath_link");
    const std::string resolved  = realpath(link_path);
    EXPECT_EQ(resolved, target);
}

TEST_F(PathTest, Realpath_NonexistentPath)
{
    const std::string nonexistent = "/nonexistent/path/to/file";
    const std::string resolved    = realpath(nonexistent);
    EXPECT_EQ(resolved, nonexistent);
}

TEST_F(PathTest, IsTextFile_TextFile)
{
    const std::string text_content = "This is a text file\nwith multiple lines\n";
    const std::string file_path    = create_file("text_file.txt", text_content);
    EXPECT_TRUE(is_text_file(file_path));
}

TEST_F(PathTest, IsTextFile_BinaryFile)
{
    const std::string file_path = m_test_dir + "/binary_file.bin";
    std::ofstream     ofs(file_path, std::ios::binary);
    char binary_data[] = { 'H', 'e', 'l', 'l', 'o', '\0', 'W', 'o', 'r', 'l', 'd' };
    ofs.write(binary_data, sizeof(binary_data));
    ofs.close();
    EXPECT_FALSE(is_text_file(file_path));
}

TEST_F(PathTest, IsTextFile_EmptyFile)
{
    const std::string file_path = create_file("empty_file.txt", "");
    EXPECT_TRUE(is_text_file(file_path));
}

TEST_F(PathTest, PathType_Directory)
{
    const path_type pt(m_test_dir);
    EXPECT_TRUE(pt.exists());
    EXPECT_TRUE(static_cast<bool>(pt));
}

TEST_F(PathTest, PathType_RegularFile)
{
    const std::string file_path = create_file("pathtype_file.txt");
    const path_type   pt(file_path);
    EXPECT_TRUE(pt.exists());
}

TEST_F(PathTest, PathType_SymbolicLink)
{
    const std::string target    = create_file("pathtype_target.txt");
    const std::string link_path = create_symlink(target, "pathtype_link");
    const path_type   pt(link_path);
    EXPECT_TRUE(pt.exists());
}

TEST_F(PathTest, PathType_Nonexistent)
{
    const path_type pt("/nonexistent/path");
    EXPECT_FALSE(pt.exists());
    EXPECT_FALSE(static_cast<bool>(pt));
}

TEST_F(PathTest, IsDirectory_ExistingDirectory) { EXPECT_TRUE(is_directory(m_test_dir)); }

TEST_F(PathTest, IsDirectory_RegularFile)
{
    const std::string file_path = create_file("isdir_file.txt");
    EXPECT_FALSE(is_directory(file_path));
}

TEST_F(PathTest, IsDirectory_SymlinkToDirectory)
{
    const std::string subdir    = create_subdir("isdir_target_dir");
    const std::string link_path = create_symlink(subdir, "isdir_link_to_dir");
    EXPECT_TRUE(is_directory(link_path));
}

TEST_F(PathTest, IsDirectory_SymlinkToFile)
{
    const std::string file_path = create_file("isdir_link_target.txt");
    const std::string link_path = create_symlink(file_path, "isdir_link_to_file");
    EXPECT_FALSE(is_directory(link_path));
}

TEST_F(PathTest, IsDirectory_BrokenSymlink)
{
    const std::string link_path =
        create_symlink("/nonexistent/target", "isdir_broken_link");
    EXPECT_FALSE(is_directory(link_path));
}

TEST_F(PathTest, IsDirectory_NonexistentPath)
{
    EXPECT_FALSE(is_directory("/nonexistent/path"));
}

TEST_F(PathTest, IsDirectory_EmptyPath) { EXPECT_FALSE(is_directory("")); }

TEST_F(PathTest, IsDirectory_AcceptsTrailingSlash)
{
    EXPECT_TRUE(is_directory(m_test_dir + "/"));
}

TEST_F(PathTest, IsDirectory_RelativePath)
{
    const std::string subdir_name = "isdir_relative_dir";
    create_subdir(subdir_name);  // Creates in m_test_dir.

    char saved_cwd[PATH_MAX];
    ASSERT_NE(getcwd(saved_cwd, sizeof(saved_cwd)), nullptr);
    ASSERT_EQ(chdir(m_test_dir.c_str()), 0);

    EXPECT_TRUE(is_directory(subdir_name));

    ASSERT_EQ(chdir(saved_cwd), 0);
}

TEST_F(PathTest, IsRegularFile_ExistingFile)
{
    std::string file_path = create_file("isregular_file.txt");
    EXPECT_TRUE(is_regular_file(file_path));
}

TEST_F(PathTest, IsRegularFile_NonexistentFile)
{
    EXPECT_FALSE(is_regular_file(m_test_dir + "/nonexistent_file.txt"));
}

TEST_F(PathTest, IsRegularFile_Directory) { EXPECT_FALSE(is_regular_file(m_test_dir)); }

TEST_F(PathTest, IsRegularFile_NonexistentPath)
{
    EXPECT_FALSE(is_regular_file("/nonexistent/path"));
}

TEST_F(PathTest, IsRegularFile_SymlinkToFile)
{
    std::string target    = create_file("isregular_target.txt");
    std::string link_path = create_symlink(target, "isregular_link_to_file");
    EXPECT_TRUE(is_regular_file(link_path));
}

TEST_F(PathTest, IsRegularFile_SymlinkToDirectory)
{
    std::string subdir    = create_subdir("isregular_target_dir");
    std::string link_path = create_symlink(subdir, "isregular_link_to_dir");
    EXPECT_FALSE(is_regular_file(link_path));
}

TEST_F(PathTest, IsRegularFile_BrokenSymlink)
{
    std::string link_path =
        create_symlink("/nonexistent/target", "isregular_broken_link");
    EXPECT_FALSE(is_regular_file(link_path));
}

TEST_F(PathTest, IsRegularFile_EmptyPath) { EXPECT_FALSE(is_regular_file("")); }

TEST_F(PathTest, IsRegularFile_SpecialCharactersInPath)
{
    std::string file_path = create_file("isregular file with spaces.txt");
    EXPECT_TRUE(is_regular_file(file_path));
}

TEST_F(PathTest, IsRegularFile_Fifo)
{
    std::string fifo_path = m_test_dir + "/isregular_fifo";
    ASSERT_EQ(mkfifo(fifo_path.c_str(), 0644), 0);
    EXPECT_FALSE(is_regular_file(fifo_path));
}

TEST_F(PathTest, IsRegularFile_RelativePath)
{
    const std::string file_name = "isregular_relative_file.txt";
    create_file(file_name);

    char saved_cwd[PATH_MAX];
    ASSERT_NE(getcwd(saved_cwd, sizeof(saved_cwd)), nullptr);
    ASSERT_EQ(chdir(m_test_dir.c_str()), 0);

    EXPECT_TRUE(is_regular_file(file_name));

    ASSERT_EQ(chdir(saved_cwd), 0);
}

TEST_F(PathTest, GetRocprofsysRoot_ReturnsNonEmptyAbsolute)
{
    std::string root = get_rocprofsys_root();
    EXPECT_FALSE(root.empty());
    EXPECT_EQ(root.front(), '/');
}

TEST_F(PathTest, GetInternalLibdir_ContainsLib)
{
    const std::string libdir = get_internal_libdir();
    EXPECT_NE(libdir.find("lib"), std::string::npos);
}

TEST_F(PathTest, GetInternalScriptPath_ContainsLibexec)
{
    const std::string script_path = get_internal_script_path();
    EXPECT_NE(script_path.find("libexec"), std::string::npos);
    EXPECT_NE(script_path.find("rocprofiler-systems"), std::string::npos);
}

TEST_F(PathTest, GetInternalLibpath_ContainsLibName)
{
    const std::string libpath = get_internal_libpath("librocprof-sys.so");
    EXPECT_NE(libpath.find("librocprof-sys.so"), std::string::npos);
}

TEST_F(PathTest, GetInternalLibpath_ContainsLib)
{
    const std::string libpath = get_internal_libpath("test.so");
    EXPECT_NE(libpath.find("lib"), std::string::npos);
}

TEST_F(PathTest, FindLibrary_AbsoluteExisting)
{
    const std::string file_path = create_file("findpath_test.txt");
    const std::string result    = find_library(file_path, 0, "");
    EXPECT_EQ(result, file_path);
}

TEST_F(PathTest, FindLibrary_NonexistentReturnsOriginal)
{
    const std::string nonexistent = "nonexistent_file_xyz.txt";
    const std::string result      = find_library(nonexistent, 0, m_test_dir);
    EXPECT_EQ(result, nonexistent);
}

TEST_F(PathTest, FindLibrary_InSearchPath)
{
    const std::string file_path = create_file("searchable.txt");
    const std::string result    = find_library("searchable.txt", 0, m_test_dir);
    EXPECT_EQ(result, file_path);
}

TEST_F(PathTest, ParentPath_ComplexPath)
{
    EXPECT_EQ(parent_path("/opt/rocm/lib/rocprofiler-systems/librocprof-sys.so"),
              "/opt/rocm/lib/rocprofiler-systems");
}

TEST_F(PathTest, ChainedSymlinks)
{
    const std::string target     = create_file("chain_target.txt");
    const std::string link1      = create_symlink(target, "chain_link1");
    const std::string link2_path = m_test_dir + "/chain_link2";
    EXPECT_EQ(symlink("chain_link1", link2_path.c_str()), 0);

    // Verify that link1 and link2_path are actual links:
    // for a non-link read_symlink() would return unchanged input
    EXPECT_NE(read_symlink(link1), link1);
    EXPECT_NE(read_symlink(link2_path), link2_path);

    const std::string resolved = realpath(link2_path);
    EXPECT_EQ(resolved, target);
}

TEST_F(PathTest, ParentPath_RocprofsysTypicalPath)
{
    const std::string path =
        "/opt/rocm-6.0.0/lib/rocprofiler-systems/librocprof-sys-dl.so";
    const std::string result = parent_path(path);
    EXPECT_EQ(result, "/opt/rocm-6.0.0/lib/rocprofiler-systems");
}

TEST_F(PathTest, NestedDirectories)
{
    const std::string subdir1 = create_subdir("level1");
    const std::string subdir2 = subdir1 + "/level2";
    mkdir(subdir2.c_str(), 0755);
    const std::string subdir3 = subdir2 + "/level3";
    mkdir(subdir3.c_str(), 0755);

    EXPECT_TRUE(is_directory(subdir1));
    EXPECT_TRUE(is_directory(subdir2));
    EXPECT_TRUE(is_directory(subdir3));

    EXPECT_EQ(parent_path(subdir3), subdir2);
    EXPECT_EQ(parent_path(subdir2), subdir1);
}

TEST_F(PathTest, ParentPath_TwoLevels) { EXPECT_EQ(parent_path("/a/b/c", 2), "/a"); }

TEST_F(PathTest, ParentPath_ThreeLevels) { EXPECT_EQ(parent_path("/a/b/c", 3), "/"); }

TEST_F(PathTest, ParentPath_RootClamp_OneComponent) { EXPECT_EQ(parent_path("/a"), "/"); }

TEST_F(PathTest, ParentPath_RootClamp_Overwalk) { EXPECT_EQ(parent_path("/a", 5), "/"); }

TEST_F(PathTest, ParentPath_RootClamp_Root) { EXPECT_EQ(parent_path("/", 1), "/"); }

TEST_F(PathTest, ParentPath_Relative_OneLevel) { EXPECT_EQ(parent_path("a/b", 1), "a"); }

TEST_F(PathTest, ParentPath_Relative_Overwalk) { EXPECT_EQ(parent_path("a/b", 5), ""); }

TEST_F(PathTest, ParentPath_Identity) { EXPECT_EQ(parent_path("/a/b/c", 0), "/a/b/c"); }

TEST_F(PathTest, ParentPath_Identity_RedundantSlashesVerbatim)
{
    EXPECT_EQ(parent_path("/a//b", 0), "/a//b");
    EXPECT_EQ(parent_path("/a/b//", 0), "/a/b//");
}

TEST_F(PathTest, ParentPath_NegativeArgWrapsAndClamps)
{
    // -1 converts to uint16 max (65535): should clamp at the root / relative bottom
    EXPECT_EQ(parent_path("/a/b/c", -1), "/");
    EXPECT_EQ(parent_path("a/b/c", -1), "");
}

TEST_F(PathTest, ParentPath_TrailingSlash_Redundant)
{
    EXPECT_EQ(parent_path("/a/b//"), "/a/b");
}

TEST_F(PathTest, ParentPath_InteriorRedundantSlash)
{
    EXPECT_EQ(parent_path("/a//b"), "/a");
}

TEST_F(PathTest, ParentPath_RealExe_TwoLevels)
{
    std::string result = parent_path(realpath("/proc/self/exe"), 2);
    EXPECT_FALSE(result.empty());
    EXPECT_EQ(result.front(), '/');
}

TEST_F(PathTest, Filename_StandardPath) { EXPECT_EQ(filename("/a/b.so"), "b.so"); }

TEST_F(PathTest, Filename_NoSlash) { EXPECT_EQ(filename("filename"), "filename"); }

TEST_F(PathTest, Filename_TrailingSlash) { EXPECT_EQ(filename("/a/b/"), ""); }

TEST_F(PathTest, Filename_TrailingDoubleSlash) { EXPECT_EQ(filename("/a/b//"), ""); }

TEST_F(PathTest, Filename_Root) { EXPECT_EQ(filename("/"), ""); }

TEST_F(PathTest, Filename_EmptyString) { EXPECT_EQ(filename(""), ""); }

TEST_F(PathTest, Filename_Dot) { EXPECT_EQ(filename("."), "."); }

TEST_F(PathTest, Filename_DotDot) { EXPECT_EQ(filename(".."), ".."); }

TEST_F(PathTest, Filename_DoubleSlash) { EXPECT_EQ(filename("/a//b"), "b"); }

TEST_F(PathTest, Filename_RedundantInteriorSlash) { EXPECT_EQ(filename("/a///b"), "b"); }

TEST_F(PathTest, Filename_MultipleExtensions)
{
    EXPECT_EQ(filename("/a/b.tar.gz"), "b.tar.gz");
}

TEST_F(PathTest, Filename_Dotfile) { EXPECT_EQ(filename("/a/.bashrc"), ".bashrc"); }

TEST_F(PathTest, Filename_RelativePath) { EXPECT_EQ(filename("a/b/c"), "c"); }

TEST_F(PathTest, Filename_AcceptsTemporary)
{
    EXPECT_EQ(filename(std::string("/a/b/c.so")), "c.so");
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_CreatesParentTree)
{
    const std::string file_path = m_test_dir + "/new/nested/tree/out.txt";

    std::ofstream out_fstream;
    EXPECT_TRUE(create_parent_dirs_and_open_ofstream(out_fstream, file_path));
    EXPECT_TRUE(out_fstream.is_open());
    out_fstream << "hello";
    out_fstream.close();

    EXPECT_TRUE(is_directory(m_test_dir + "/new/nested/tree"));
    EXPECT_TRUE(is_regular_file(file_path));
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_ExistingDirectoryIsNotAnError)
{
    const std::string existing_dir = create_subdir("already_here");

    std::ofstream out_fstream;
    EXPECT_TRUE(
        create_parent_dirs_and_open_ofstream(out_fstream, existing_dir + "/out.txt"));
    EXPECT_TRUE(out_fstream.is_open());
    out_fstream.close();

    EXPECT_TRUE(is_regular_file(existing_dir + "/out.txt"));
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_BareFilenameCreatesNoDirectory)
{
    char saved_cwd[PATH_MAX];
    ASSERT_NE(getcwd(saved_cwd, sizeof(saved_cwd)), nullptr);
    ASSERT_EQ(chdir(m_test_dir.c_str()), 0);

    std::ofstream out_fstream;
    EXPECT_TRUE(create_parent_dirs_and_open_ofstream(out_fstream, "bare.txt"));
    EXPECT_TRUE(out_fstream.is_open());
    out_fstream.close();

    EXPECT_TRUE(is_regular_file(m_test_dir + "/bare.txt"));

    ASSERT_EQ(chdir(saved_cwd), 0);
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_UncreatableParentReturnsFalse)
{
    // a regular file as an intermediate component makes create_directories fail
    const std::string blocker = create_file("blocker");

    std::ofstream out_fstream;
    EXPECT_FALSE(
        create_parent_dirs_and_open_ofstream(out_fstream, blocker + "/sub/out.txt"));
    EXPECT_FALSE(out_fstream.is_open());
    EXPECT_FALSE(is_regular_file(blocker + "/sub/out.txt"));
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_TargetIsDirectoryReturnsFalse)
{
    const std::string existing_dir = create_subdir("a_directory");

    std::ofstream out_fstream;
    EXPECT_FALSE(create_parent_dirs_and_open_ofstream(out_fstream, existing_dir));
    EXPECT_FALSE(out_fstream.is_open());
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_EmptyPathReturnsFalse)
{
    std::ofstream out_fstream;
    EXPECT_FALSE(create_parent_dirs_and_open_ofstream(out_fstream, ""));
    EXPECT_FALSE(out_fstream.is_open());
}

TEST_F(PathTest, CreateParentDirsAndOpenOfstream_TruncatesExistingFile)
{
    const std::string file_path = create_file("truncate_me.txt", "0123456789");

    std::ofstream out_fstream;
    EXPECT_TRUE(create_parent_dirs_and_open_ofstream(out_fstream, file_path));
    out_fstream << "ab";
    out_fstream.close();

    std::ifstream in_fstream{ file_path };
    std::string   content;
    std::getline(in_fstream, content);
    EXPECT_EQ(content, "ab");
}
