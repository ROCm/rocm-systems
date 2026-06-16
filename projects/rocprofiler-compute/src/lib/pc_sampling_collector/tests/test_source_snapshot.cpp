// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_source_snapshot.h"

#include <optional>
#include <string>
#include <vector>

namespace
{
bool copied_somewhere_with_tail(const std::filesystem::path& code_obj_sources,
                                const std::filesystem::path& original,
                                const std::string&           expected_contents)
{
    if (!std::filesystem::exists(code_obj_sources))
        return false;

    const std::string tail = original.filename().string();
    for (const auto& entry : std::filesystem::recursive_directory_iterator(code_obj_sources))
    {
        if (!entry.is_regular_file())
            continue;
        if (entry.path().filename().string() != tail)
            continue;

        std::ifstream     ifs(entry.path());
        const std::string contents{std::istreambuf_iterator<char>(ifs),
                                   std::istreambuf_iterator<char>{}};
        if (contents == expected_contents)
            return true;
    }
    return false;
}
}  // namespace

TEST_F(test_source_snapshot_t, ParseSourceRef_PathBeforeLastColon)
{
    EXPECT_EQ(m_snapshotter.parse_ref("foo/bar.cpp:42"), std::optional<std::string>{"foo/bar.cpp"});
    EXPECT_EQ(m_snapshotter.parse_ref("a:b:10"), std::optional<std::string>{"a:b"});
}

TEST_F(test_source_snapshot_t, ParseSourceRef_NoColon_ReturnsNullopt)
{
    EXPECT_EQ(m_snapshotter.parse_ref("no colon here"), std::nullopt);
}

TEST_F(test_source_snapshot_t, ParseSourceRef_LeadingColonOrEmpty_ReturnsNullopt)
{
    EXPECT_EQ(m_snapshotter.parse_ref(":30"), std::nullopt);
    EXPECT_EQ(m_snapshotter.parse_ref(""), std::nullopt);
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_CopiesExistingFilesPreservingTail)
{
    const std::vector<std::string> refs{m_file_a.string(), m_file_b.string()};

    const size_t copied = m_snapshotter.snapshot(refs, m_output_root, m_tmp_root);

    const auto code_obj_sources = m_output_root / "code_obj_sources";
    EXPECT_EQ(copied, 2u);
    EXPECT_TRUE(copied_somewhere_with_tail(code_obj_sources, m_file_a, m_contents_a));
    EXPECT_TRUE(copied_somewhere_with_tail(code_obj_sources, m_file_b, m_contents_b));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_SkipsMissingRefs)
{
    const auto                     missing = (m_tmp_root / "proj" / "does_not_exist.cpp").string();
    const std::vector<std::string> refs{m_file_a.string(), missing, m_file_b.string()};

    size_t copied = 0;
    EXPECT_NO_THROW(copied = m_snapshotter.snapshot(refs, m_output_root, m_tmp_root));

    const auto code_obj_sources = m_output_root / "code_obj_sources";
    EXPECT_EQ(copied, 2u);
    EXPECT_TRUE(copied_somewhere_with_tail(code_obj_sources, m_file_a, m_contents_a));
    EXPECT_TRUE(copied_somewhere_with_tail(code_obj_sources, m_file_b, m_contents_b));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_DedupsDuplicateRefs)
{
    const std::vector<std::string> refs{m_file_a.string(), m_file_a.string()};

    const size_t copied = m_snapshotter.snapshot(refs, m_output_root, m_tmp_root);

    const auto code_obj_sources = m_output_root / "code_obj_sources";
    EXPECT_EQ(copied, 1u);
    EXPECT_TRUE(copied_somewhere_with_tail(code_obj_sources, m_file_a, m_contents_a));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_RefEscapingAllowedRootViaDotDot_IsRejected)
{
    // A pre-existing file OUTSIDE the allowed root that a traversal ref targets.
    const auto outside = m_tmp_root / "victim.txt";
    write_file(outside, "original\n");

    // allowed_root is the project subtree (m_tmp_root/proj). A ref that uses ".."
    // to climb out of it to the victim must be refused: the resolved source lies
    // outside allowed_root.
    const auto allowed_root = m_tmp_root / "proj";
    const auto traversal    = (allowed_root / ".." / "victim.txt").string();

    size_t copied = 0;
    EXPECT_NO_THROW(copied = m_snapshotter.snapshot({traversal}, m_output_root, allowed_root));

    EXPECT_EQ(copied, 0u);
    EXPECT_EQ(read_file(outside), "original\n");  // unchanged: not read/copied
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_RefOutsideAllowedRoot_IsRejected)
{
    // A real, existing file that lives OUTSIDE the allowed root. A hostile ISA
    // comment naming an absolute path (e.g. /etc/passwd) must not be read/copied.
    const auto outside = m_tmp_root / "secret.txt";
    write_file(outside, "secret\n");

    const auto allowed_root = m_tmp_root / "proj";

    size_t copied = 0;
    EXPECT_NO_THROW(copied = m_snapshotter.snapshot({outside.string()}, m_output_root, allowed_root));

    EXPECT_EQ(copied, 0u);
    EXPECT_FALSE(copied_somewhere_with_tail(m_output_root / "code_obj_sources", outside, "secret\n"));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_RefInsideAllowedRoot_IsCopied)
{
    // Sanity: a file inside the allowed root IS copied (m_file_a lives under proj/).
    const auto allowed_root = m_tmp_root / "proj";

    const size_t copied = m_snapshotter.snapshot({m_file_a.string()}, m_output_root, allowed_root);

    EXPECT_EQ(copied, 1u);
    EXPECT_TRUE(copied_somewhere_with_tail(m_output_root / "code_obj_sources", m_file_a, m_contents_a));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_SymlinkedSourceInsideAllowedRoot_IsRejected)
{
    // A symlink that lives INSIDE allowed_root but points OUTSIDE it must not be
    // followed: copy_file would otherwise read the out-of-root target. Rejecting
    // symlinked sources closes that TOCTOU window.
    const auto allowed_root = m_tmp_root / "proj";
    const auto outside      = m_tmp_root / "secret.txt";
    write_file(outside, "secret\n");

    const auto      link = allowed_root / "link.cpp";
    std::error_code ec;
    std::filesystem::create_symlink(outside, link, ec);
    ASSERT_FALSE(ec) << ec.message();

    size_t copied = 0;
    EXPECT_NO_THROW(copied = m_snapshotter.snapshot({link.string()}, m_output_root, allowed_root));

    EXPECT_EQ(copied, 0u);
    EXPECT_FALSE(copied_somewhere_with_tail(m_output_root / "code_obj_sources", outside, "secret\n"));
}

TEST_F(test_source_snapshot_t, SnapshotSourceFiles_DestinationParentSymlinkEscapingSourcesRoot_IsRejected)
{
    // Pre-plant a symlink at the first relative-path component inside
    // code_obj_sources, pointing outside it. The destination for m_file_a resolves
    // THROUGH that symlink, so the write must be refused: canon_dst_parent lands
    // outside sources_root.
    const auto allowed_root     = m_tmp_root / "proj";
    const auto code_obj_sources = m_output_root / "code_obj_sources";
    const auto escape_target    = m_tmp_root / "escape_target";
    std::filesystem::create_directories(code_obj_sources);
    std::filesystem::create_directories(escape_target);

    // The impl strips leading '/' from the ref; the first remaining component is
    // where we plant the redirecting symlink.
    std::string relative = m_file_a.string();
    while (!relative.empty() && relative.front() == '/')
        relative.erase(relative.begin());
    const auto first_component = std::filesystem::path{relative}.begin()->string();

    std::error_code ec;
    std::filesystem::create_directory_symlink(escape_target, code_obj_sources / first_component, ec);
    ASSERT_FALSE(ec) << ec.message();

    size_t copied = 0;
    EXPECT_NO_THROW(copied = m_snapshotter.snapshot({m_file_a.string()}, m_output_root, allowed_root));

    EXPECT_EQ(copied, 0u);
    // Nothing was written THROUGH the symlink into escape_target.
    bool escaped = false;
    for (const auto& entry : std::filesystem::recursive_directory_iterator(escape_target))
    {
        if (entry.is_regular_file())
            escaped = true;
    }
    EXPECT_FALSE(escaped);
}
