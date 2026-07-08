// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "common/path.hpp"
#include "filesystem.hpp"

#include <fstream>
#include <gtest/gtest.h>
#include <string>
#include <sys/stat.h>
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
    // Unified exists() now uses std::filesystem::exists semantics (follows symlinks),
    // so a broken symlink no longer "exists" (design §5.4 / intended change).
    std::string link_path = create_symlink("/nonexistent/target", "broken_link");
    EXPECT_FALSE(exists(link_path));
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

//======================================================================================//
//
//  New unified filesystem API (design §5) — additive tests
//
//======================================================================================//

// ---- 5.1 parent_path: full edge table (design §5.1) ----

TEST_F(PathTest, ParentPath_StripOne) { EXPECT_EQ(parent_path("/a/b/c"), "/a/b"); }

TEST_F(PathTest, ParentPath_StripTwo) { EXPECT_EQ(parent_path("/a/b/c", 2), "/a"); }

TEST_F(PathTest, ParentPath_StripToRoot) { EXPECT_EQ(parent_path("/a/b/c", 3), "/"); }

TEST_F(PathTest, ParentPath_AbsoluteClampsAtRoot) { EXPECT_EQ(parent_path("/a", 1), "/"); }

TEST_F(PathTest, ParentPath_AbsoluteOverWalk) { EXPECT_EQ(parent_path("/a", 5), "/"); }

TEST_F(PathTest, ParentPath_RootParentIsRoot) { EXPECT_EQ(parent_path("/", 3), "/"); }

TEST_F(PathTest, ParentPath_Relative) { EXPECT_EQ(parent_path("a/b", 1), "a"); }

TEST_F(PathTest, ParentPath_RelativeBottomsOut) { EXPECT_EQ(parent_path("a/b", 5), ""); }

TEST_F(PathTest, ParentPath_NoSeparator) { EXPECT_EQ(parent_path("file", 1), ""); }

TEST_F(PathTest, ParentPath_NoSeparatorOverWalk) { EXPECT_EQ(parent_path("file", 3), ""); }

TEST_F(PathTest, ParentPath_ZeroLevelsIsIdentity)
{
    EXPECT_EQ(parent_path("/a/b/c", 0), "/a/b/c");
}

TEST_F(PathTest, ParentPath_TrailingSlash) { EXPECT_EQ(parent_path("/a/b/"), "/a/b"); }

// ---- 5.1 filename / stem / extension ----

TEST_F(PathTest, Filename_Standard) { EXPECT_EQ(filename("/a/b.so"), "b.so"); }

TEST_F(PathTest, Filename_NoDir) { EXPECT_EQ(filename("b.so"), "b.so"); }

TEST_F(PathTest, Stem_MultiExtension) { EXPECT_EQ(stem("/a/b.tar.gz"), "b.tar"); }

TEST_F(PathTest, Stem_NoExtension) { EXPECT_EQ(stem("/a/name"), "name"); }

TEST_F(PathTest, Extension_WithDot) { EXPECT_EQ(extension("/a/b.so"), ".so"); }

TEST_F(PathTest, Extension_None) { EXPECT_EQ(extension("a"), ""); }

TEST_F(PathTest, Normalize_CollapsesDotDot) { EXPECT_EQ(normalize("a/./b/../c"), "a/c"); }

TEST_F(PathTest, IsAbsolute_True) { EXPECT_TRUE(is_absolute("/a/b")); }

TEST_F(PathTest, IsAbsolute_False) { EXPECT_FALSE(is_absolute("a/b")); }

TEST_F(PathTest, IsRelative_True) { EXPECT_TRUE(is_relative("a/b")); }

// ---- 5.2 classification ----

TEST_F(PathTest, HasExtension_WithAndWithoutDot)
{
    EXPECT_TRUE(has_extension("x.so", ".so"));
    EXPECT_TRUE(has_extension("x.so", "so"));
    EXPECT_FALSE(has_extension("x.so", "o"));
    EXPECT_FALSE(has_extension("libfoo.so.1", ".so"));
}

TEST_F(PathTest, HasAnyExtension)
{
    EXPECT_TRUE(has_any_extension("libx.a", { ".so", ".a" }));
    EXPECT_FALSE(has_any_extension("libx.dylib", { ".so", ".a" }));
}

TEST_F(PathTest, StripKnownExtension)
{
    EXPECT_EQ(strip_known_extension("f.json", { ".txt", ".json" }), "f");
    EXPECT_EQ(strip_known_extension("f.cfg", { ".txt", ".json" }), "f.cfg");
}

// ---- 5.3 read_symlink ----

TEST_F(PathTest, ReadSymlink_Link)
{
    std::string target    = create_file("rs_target.txt");
    std::string link_path = create_symlink(target, "rs_link");
    EXPECT_EQ(read_symlink(link_path), target);
}

TEST_F(PathTest, ReadSymlink_NotALink)
{
    std::string file_path = create_file("rs_notalink.txt");
    EXPECT_EQ(read_symlink(file_path), file_path);
}

// ---- 5.4 type predicates ----

TEST_F(PathTest, IsDirectory_Dir) { EXPECT_TRUE(is_directory(m_test_dir)); }

TEST_F(PathTest, IsDirectory_File)
{
    EXPECT_FALSE(is_directory(create_file("id_file.txt")));
}

TEST_F(PathTest, IsDirectory_Missing) { EXPECT_FALSE(is_directory("/no/such/dir")); }

TEST_F(PathTest, IsRegularFile_File)
{
    EXPECT_TRUE(is_regular_file(create_file("irf_file.txt")));
}

TEST_F(PathTest, IsRegularFile_Dir) { EXPECT_FALSE(is_regular_file(m_test_dir)); }

TEST_F(PathTest, IsSymlink_Link)
{
    std::string target    = create_file("issl_target.txt");
    std::string link_path = create_symlink(target, "issl_link");
    EXPECT_TRUE(is_symlink(link_path));
}

TEST_F(PathTest, IsSymlink_RegularFile)
{
    EXPECT_FALSE(is_symlink(create_file("issl_reg.txt")));
}

// ---- 5.4 is_elf ----

TEST_F(PathTest, IsElf_ElfFile)
{
    std::string   file_path = m_test_dir + "/elf_file";
    std::ofstream ofs(file_path, std::ios::binary);
    char          magic[] = { (char) 0x7F, 'E', 'L', 'F', 0x02, 0x01 };
    ofs.write(magic, sizeof(magic));
    ofs.close();
    EXPECT_TRUE(is_elf(file_path));
}

TEST_F(PathTest, IsElf_TextFile)
{
    EXPECT_FALSE(is_elf(create_file("elf_text.txt", "not an elf file")));
}

TEST_F(PathTest, IsElf_EmptyFile)
{
    EXPECT_FALSE(is_elf(create_file("elf_empty.txt", "")));
}

TEST_F(PathTest, IsElf_Unopenable) { EXPECT_FALSE(is_elf("/no/such/elf")); }

// ---- 5.4 file_size_or_zero ----

TEST_F(PathTest, FileSizeOrZero_KnownSize)
{
    EXPECT_EQ(file_size_or_zero(create_file("fsz.txt", "12345")), 5u);
}

TEST_F(PathTest, FileSizeOrZero_Missing)
{
    EXPECT_EQ(file_size_or_zero("/no/such/file"), 0u);
}

// ---- 5.5 make_dirs / make_parent_dirs ----

TEST_F(PathTest, MakeDirs_CreatesTree)
{
    std::string nested = m_test_dir + "/x/y/z";
    EXPECT_TRUE(make_dirs(nested));
    EXPECT_TRUE(is_directory(nested));
}

TEST_F(PathTest, MakeDirs_Idempotent)
{
    std::string nested = m_test_dir + "/a/b";
    EXPECT_TRUE(make_dirs(nested));
    EXPECT_TRUE(make_dirs(nested));  // second call must still succeed
    EXPECT_TRUE(is_directory(nested));
}

TEST_F(PathTest, MakeParentDirs_CreatesParent)
{
    std::string file_path = m_test_dir + "/p/q/file.txt";
    EXPECT_TRUE(make_parent_dirs(file_path));
    EXPECT_TRUE(is_directory(m_test_dir + "/p/q"));
}

// ---- 5.5 remove / remove_all / list_directory ----

TEST_F(PathTest, Remove_File)
{
    std::string file_path = create_file("to_remove.txt");
    EXPECT_TRUE(rocprofsys::common::path::remove(file_path));
    EXPECT_FALSE(exists(file_path));
}

TEST_F(PathTest, Remove_MissingTolerated)
{
    EXPECT_TRUE(rocprofsys::common::path::remove(m_test_dir + "/never_existed"));
}

TEST_F(PathTest, RemoveAll_Tree)
{
    std::string nested = m_test_dir + "/ra/sub";
    ASSERT_TRUE(make_dirs(nested));
    EXPECT_GT(remove_all(m_test_dir + "/ra"), 0u);
    EXPECT_FALSE(exists(m_test_dir + "/ra"));
}

TEST_F(PathTest, ListDirectory_Names)
{
    create_file("ld_a.txt");
    create_file("ld_b.txt");
    auto names = list_directory(m_test_dir);
    EXPECT_EQ(names.size(), 2u);
}

TEST_F(PathTest, ListDirectory_Filtered)
{
    create_file("keep.json");
    create_file("skip.txt");
    auto names = list_directory(
        m_test_dir, [](const std::string& n) { return has_extension(n, ".json"); });
    ASSERT_EQ(names.size(), 1u);
    EXPECT_EQ(names.front(), "keep.json");
}

TEST_F(PathTest, ListDirectory_MissingIsEmpty)
{
    EXPECT_TRUE(list_directory("/no/such/dir").empty());
}

// ---- 5.6 open shim: auto-mkdir + ./base fallback ----

TEST_F(PathTest, Open_OutputAutoCreatesParentDir)
{
    std::string   file_path = m_test_dir + "/auto/made/out.txt";
    std::ofstream ofs{};
    EXPECT_TRUE(open(ofs, file_path));
    ofs << "hello";
    ofs.close();
    EXPECT_TRUE(is_regular_file(file_path));
}

TEST_F(PathTest, Open_OutputWithFlags)
{
    std::string   file_path = m_test_dir + "/bin/out.bin";
    std::ofstream ofs{};
    EXPECT_TRUE(open(ofs, file_path, std::ios::out | std::ios::binary));
    ofs.close();
    EXPECT_TRUE(is_regular_file(file_path));
}

TEST_F(PathTest, Open_InputExisting)
{
    std::string   file_path = create_file("open_in.txt", "content");
    std::ifstream ifs{};
    EXPECT_TRUE(open(ifs, file_path));
}

TEST_F(PathTest, Open_InputMissingFails)
{
    std::ifstream ifs{};
    EXPECT_FALSE(open(ifs, m_test_dir + "/missing_in.txt"));
}

TEST_F(PathTest, Fopen_AutoCreatesParentDir)
{
    std::string file_path = m_test_dir + "/cauto/made/out.dat";
    std::FILE*  f         = rocprofsys::common::path::fopen(file_path, "w");
    ASSERT_NE(f, nullptr);
    std::fclose(f);
    EXPECT_TRUE(is_regular_file(file_path));
}

// ---- 5.7 process / environment paths ----

TEST_F(PathTest, TempDir_NonEmpty) { EXPECT_FALSE(temp_dir().empty()); }

TEST_F(PathTest, ExecutablePath_Absolute)
{
    auto exe = executable_path();
    EXPECT_FALSE(exe.empty());
    EXPECT_TRUE(is_absolute(exe));
}

TEST_F(PathTest, FindInDirs_Found)
{
    create_file("fid.txt");
    EXPECT_EQ(find_in_dirs("fid.txt", { m_test_dir }), m_test_dir + "/fid.txt");
}

TEST_F(PathTest, FindInDirs_NotFoundReturnsInput)
{
    EXPECT_EQ(find_in_dirs("nope.txt", { m_test_dir }), "nope.txt");
}

// ---- realpath verbatim-fallback pinned invariant (design §7, issue #4) ----

TEST_F(PathTest, Realpath_VerbatimFallbackNonexistent)
{
    EXPECT_EQ(realpath("/does/not/exist"), "/does/not/exist");
}

TEST_F(PathTest, Realpath_VerbatimFallbackRelativeNonexistent)
{
    EXPECT_EQ(realpath("./a/../b"), "./a/../b");
}
