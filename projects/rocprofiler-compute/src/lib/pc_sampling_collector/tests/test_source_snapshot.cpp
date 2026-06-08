// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_source_snapshot.h"

#include "source_snapshot.h"

#include <fstream>
#include <system_error>

using namespace rocprofiler_compute_tool;
namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// parse_source_path
// ---------------------------------------------------------------------------

TEST(parse_source_path_t, SplitsOnLastColon)
{
    const auto result = parse_source_path("/a/b.cpp:42");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "/a/b.cpp");
}

TEST(parse_source_path_t, MultipleColonsSplitsOnLast)
{
    const auto result = parse_source_path("/a/b.cpp:10:42");
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(*result, "/a/b.cpp:10");
}

TEST(parse_source_path_t, EmptyCommentReturnsNullopt)
{
    EXPECT_FALSE(parse_source_path("").has_value());
}

TEST(parse_source_path_t, NoColonReturnsNullopt)
{
    EXPECT_FALSE(parse_source_path("noColonHere").has_value());
}

TEST(parse_source_path_t, EmptyPathBeforeColonReturnsNullopt)
{
    EXPECT_FALSE(parse_source_path(":42").has_value());
}

// ---------------------------------------------------------------------------
// copy_source_files
// ---------------------------------------------------------------------------

void test_source_snapshot_t::write_file(const fs::path& path, const std::string& content)
{
    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    out << content;
    out.close();
}

void test_source_snapshot_t::SetUp()
{
    std::error_code ec;
    const auto      base = fs::temp_directory_path(ec);
    ASSERT_FALSE(ec);
    // Unique sandbox per test using the gtest test name.
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    m_tmp = base / (std::string("rpc_src_snapshot_") + info->test_case_name() + "_" + info->name());
    fs::remove_all(m_tmp, ec);
    ASSERT_TRUE(fs::create_directories(m_tmp, ec));
    m_src_root  = m_tmp / "src";
    m_dest_root = m_tmp / "dest";
    fs::create_directories(m_src_root);
    fs::create_directories(m_dest_root);
}

void test_source_snapshot_t::TearDown()
{
    std::error_code ec;
    fs::remove_all(m_tmp, ec);
}

TEST_F(test_source_snapshot_t, AbsolutePathReproducedUnderDestRoot)
{
    // Create a real absolute source file under the sandbox.
    const fs::path src = m_src_root / "foo" / "bar.cpp";
    write_file(src, "int main() { return 0; }\n");
    ASSERT_TRUE(src.is_absolute());

    copy_source_files({src.string()}, m_dest_root);

    // dest = dest_root / src.relative_path() (leading '/' stripped).
    const fs::path expected = m_dest_root / src.relative_path();

    EXPECT_TRUE(fs::exists(expected)) << "expected copy at " << expected;

    // The destination is distinct from and below dest_root; original untouched.
    EXPECT_NE(fs::weakly_canonical(expected), fs::weakly_canonical(src));
    const auto rel = fs::relative(expected, m_dest_root);
    EXPECT_FALSE(rel.empty());
    EXPECT_NE(*rel.begin(), "..") << "dest escaped dest_root";
    EXPECT_TRUE(fs::exists(src)) << "original source must be left in place";
}

TEST_F(test_source_snapshot_t, DuplicatePathsAreIdempotent)
{
    const fs::path src = m_src_root / "dup.cpp";
    write_file(src, "// dup\n");

    EXPECT_NO_THROW(copy_source_files({src.string(), src.string()}, m_dest_root));

    const fs::path expected = m_dest_root / src.relative_path();
    EXPECT_TRUE(fs::exists(expected));
}

TEST_F(test_source_snapshot_t, MissingSourceSkippedValidStillCopied)
{
    const fs::path good    = m_src_root / "good.cpp";
    const fs::path missing = m_src_root / "does_not_exist.cpp";
    write_file(good, "// good\n");
    ASSERT_FALSE(fs::exists(missing));

    EXPECT_NO_THROW(copy_source_files({missing.string(), good.string()}, m_dest_root));

    const fs::path good_dest    = m_dest_root / good.relative_path();
    const fs::path missing_dest = m_dest_root / missing.relative_path();
    EXPECT_TRUE(fs::exists(good_dest));
    EXPECT_FALSE(fs::exists(missing_dest));
}

TEST_F(test_source_snapshot_t, CopyFailureOnOneEntryDoesNotAbortOthers)
{
    // Force a failure for one entry by making a needed parent path component a
    // regular file, so create_directories() for that entry fails.
    const fs::path blocker_dir = m_dest_root / m_src_root.relative_path() / "blocked";
    fs::create_directories(blocker_dir.parent_path());
    // Create a regular file where a directory would be required.
    write_file(blocker_dir, "i am a file, not a dir\n");

    // This source's destination requires blocker_dir/ to be a directory -> fails.
    const fs::path failing = m_src_root / "blocked" / "inner.cpp";
    write_file(failing, "// blocked\n");

    // A valid source that should still be copied.
    const fs::path good = m_src_root / "ok.cpp";
    write_file(good, "// ok\n");

    EXPECT_NO_THROW(copy_source_files({failing.string(), good.string()}, m_dest_root));

    const fs::path good_dest = m_dest_root / good.relative_path();
    EXPECT_TRUE(fs::exists(good_dest)) << "valid entry must still be copied";
}

TEST_F(test_source_snapshot_t, TraversalPathNotCopiedValidStillCopied)
{
    // A crafted comment path with leading ".." segments must never produce a
    // copy outside dest_root; the valid entry alongside it is still copied.
    const fs::path good = m_src_root / "ok.cpp";
    write_file(good, "// ok\n");

    // Sibling of dest_root that an unguarded join could escape to. It must be
    // left untouched (neither created nor overwritten) by the traversal entry.
    const fs::path sibling = m_tmp / "escaped.cpp";
    ASSERT_FALSE(fs::exists(sibling));

    const std::string traversal = "../escaped.cpp";
    EXPECT_NO_THROW(copy_source_files({traversal, good.string()}, m_dest_root));

    EXPECT_FALSE(fs::exists(sibling)) << "traversal entry escaped dest_root";
    EXPECT_TRUE(fs::exists(m_dest_root / good.relative_path()));
}

TEST_F(test_source_snapshot_t, SymlinkSkippedValidStillCopied)
{
    const fs::path good = m_src_root / "real.cpp";
    write_file(good, "// real\n");

    // A symlink pointing at a real regular file must be skipped (symlink_status
    // does not follow links) rather than chased into its target.
    const fs::path target = m_src_root / "target.cpp";
    write_file(target, "// target\n");
    const fs::path link = m_src_root / "link.cpp";
    std::error_code ec;
    fs::create_symlink(target, link, ec);
    ASSERT_FALSE(ec) << "failed to create symlink: " << ec.message();

    EXPECT_NO_THROW(copy_source_files({link.string(), good.string()}, m_dest_root));

    EXPECT_FALSE(fs::exists(m_dest_root / link.relative_path()))
        << "symlink must not be reproduced under dest_root";
    EXPECT_TRUE(fs::exists(m_dest_root / good.relative_path()))
        << "regular file must still be copied";
}

// ---------------------------------------------------------------------------
// collect_source_paths (through the mock translator)
// ---------------------------------------------------------------------------

void test_collect_source_paths_t::SetUp()
{
    m_translator = std::make_shared<mock_code_object_translator_t>();
    m_collector  = std::make_shared<pc_sampling_collector_impl_t>(m_translator);

    m_file_info.storage_type   = ROCPROFILER_CODE_OBJECT_STORAGE_TYPE_FILE;
    m_file_info.uri            = "test_code_object.co";
    m_file_info.code_object_id = 222;
    m_file_info.load_base      = 0x1000;
    m_file_info.load_size      = 0x2000;
}

TEST_F(test_collect_source_paths_t, DedupsRepeatedSourcePath)
{
    // A symbol spanning several instruction slots; the mock returns the same
    // instruction (and comment) for every address, so the parsed source path
    // repeats and must collapse to a single distinct entry.
    m_collector->on_code_object_load(m_file_info);
    const std::vector<symbol_t> symbols = {{"name0", 0x10, 0x1000, 4}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols);
    m_translator->add_instruction({"inst0", "/path/to/source.cpp:42", 0x1000, 0x10, 1});

    const auto paths = m_collector->collect_source_paths();
    ASSERT_EQ(paths.size(), 1u);
    EXPECT_EQ(paths[0], "/path/to/source.cpp");
}

TEST_F(test_collect_source_paths_t, SkipsInstructionsWithoutParsableComment)
{
    m_collector->on_code_object_load(m_file_info);
    const std::vector<symbol_t> symbols = {{"name0", 0x10, 0x1000, 4}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols);
    // No ':' -> parse returns nullopt -> nothing collected.
    m_translator->add_instruction({"inst0", "no_colon_comment", 0x1000, 0x10, 1});

    const auto paths = m_collector->collect_source_paths();
    EXPECT_TRUE(paths.empty());
}

TEST_F(test_collect_source_paths_t, ZeroSizeInstructionDoesNotHang)
{
    // A zero-size instruction must be skipped (the impl breaks out of the loop)
    // rather than looping forever.
    m_collector->on_code_object_load(m_file_info);
    const std::vector<symbol_t> symbols = {{"name0", 0x10, 0x1000, 4}};
    m_translator->add_symbols(m_file_info.code_object_id, symbols);
    m_translator->add_instruction({"inst0", "/path/to/source.cpp:42", 0x1000, 0x10, 0});

    std::vector<std::string> paths;
    EXPECT_NO_THROW(paths = m_collector->collect_source_paths());
    EXPECT_TRUE(paths.empty());
}

TEST_F(test_collect_source_paths_t, NoCodeObjectsReturnsEmpty)
{
    const auto paths = m_collector->collect_source_paths();
    EXPECT_TRUE(paths.empty());
}
