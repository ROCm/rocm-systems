// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "gtest/gtest.h"

#include "common/size_format.hpp"
#include "core/output/helper_collapser.hpp"
#include "core/output/perfetto_log_filter.hpp"
#include "core/output/process_tree_builder.hpp"
#include "core/output/role_classifier.hpp"
#include "core/output/summary_renderer.hpp"
#include "core/output_file_registry.hpp"

#include <perfetto.h>

#include <unistd.h>

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

TEST(summary_renderer, empty_registry_emits_nothing)
{
    rocprofsys::output_file_registry registry;
    std::ostringstream               oss;
    rocprofsys::output::print_summary(oss, registry);
    EXPECT_TRUE(oss.str().empty());
}

TEST(summary_renderer, single_row_renders_header_basename_and_viewer_footer)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);

    const std::string out = oss.str();
    // R1 header substrings appear exactly once.
    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("Run: "), std::string::npos);
    EXPECT_NE(out.find("Duration: "), std::string::npos);
    EXPECT_NE(out.find("Processes: "), std::string::npos);
    EXPECT_NE(out.find("Output dir: "), std::string::npos);
    // R6: basename in the body, no '/' on the body row.
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    // R7: per-format viewer footer line.
    EXPECT_NE(out.find("Open Perfetto trace: Open in https://ui.perfetto.dev"),
              std::string::npos);
}

TEST(summary_renderer, multi_row_uses_branch_glyphs_and_dedups_viewer_footer)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    registry.register_file("/tmp/rocprofsys-test/wall_clock.txt",
                           rocprofsys::output_format::text, "wall_clock");

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);

    const std::string out = oss.str();
    EXPECT_NE(out.find("Perfetto trace"), std::string::npos);
    EXPECT_NE(out.find("Profile (wall_clock)"), std::string::npos);
    EXPECT_NE(out.find("perfetto-trace.proto"), std::string::npos);
    EXPECT_NE(out.find("wall_clock.txt"), std::string::npos);
    // Per-format footer present.
    EXPECT_NE(out.find("Open Perfetto trace:"), std::string::npos);
    EXPECT_NE(out.find("Open Profile (wall_clock):"), std::string::npos);
}

TEST(summary_renderer, registry_member_forwards_to_renderer)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    EXPECT_EQ(registry.rows().size(), 1u);
}

namespace
{
struct size_format_case
{
    std::optional<std::uintmax_t> input;
    std::string                   expected;
};

class format_size_human_test : public ::testing::TestWithParam<size_format_case>
{};
}  // namespace

TEST_P(format_size_human_test, formats_at_threshold_boundaries)
{
    EXPECT_EQ(rocprofsys::common::format_size_human(GetParam().input),
              GetParam().expected);
}

INSTANTIATE_TEST_SUITE_P(
    boundaries, format_size_human_test,
    ::testing::Values(size_format_case{ std::nullopt, "?" },
                      size_format_case{ 0u, "0.00 KB" },
                      size_format_case{ 1023u, "1.00 KB" },
                      size_format_case{ 1024u, "1.00 KB" },
                      size_format_case{ 1024ULL * 1024 - 1, "1024.00 KB" },
                      size_format_case{ 1024ULL * 1024, "1.00 MB" },
                      size_format_case{ 1024ULL * 1024 * 1024 - 1, "1024.00 MB" },
                      size_format_case{ 1024ULL * 1024 * 1024, "1.00 GB" },
                      size_format_case{ 5ULL * 1024 * 1024 * 1024, "5.00 GB" }));

TEST(output_file_registry, default_pid_resolves_to_getpid)
{
    rocprofsys::output_file_registry registry;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, getpid());
    EXPECT_EQ(rows.front().session_id, 1u);  // initial session id
}

TEST(output_file_registry, bump_session_filters_prior_rows_from_view)
{
    rocprofsys::output_file_registry registry;

    // Session 1
    registry.register_file("/tmp/rocprofsys-test/session1-a.proto",
                           rocprofsys::output_format::perfetto);
    EXPECT_EQ(registry.rows().size(), 1u);

    const auto session_two = registry.bump_session();
    EXPECT_EQ(session_two, 2u);

    // After bump, the prior-session row is filtered out of rows().
    EXPECT_TRUE(registry.rows().empty());

    // New rows in session 2.
    registry.register_file("/tmp/rocprofsys-test/session2-a.proto",
                           rocprofsys::output_format::perfetto);
    registry.register_file("/tmp/rocprofsys-test/session2-b.proto",
                           rocprofsys::output_format::perfetto);
    const auto rows_v2 = registry.rows();
    EXPECT_EQ(rows_v2.size(), 2u);
    for(const auto& r : rows_v2)
        EXPECT_EQ(r.session_id, 2u);
}

TEST(output_file_registry, bump_session_is_race_safe_with_concurrent_register)
{
    // Stress test: a writer thread registers files while another
    // thread bumps the session id. Each row must consistently land in
    // either the prior or the new session — never torn, never seen as
    // the wrong session by rows().
    rocprofsys::output_file_registry registry;
    constexpr int                    WRITES_PER_ROUND = 50;

    std::atomic<bool> stop{ false };
    std::thread       writer([&]() {
        int i = 0;
        while(!stop.load(std::memory_order_relaxed))
        {
            registry.register_file("/tmp/rocprofsys-test/stress-" + std::to_string(i++) +
                                             ".proto",
                                         rocprofsys::output_format::perfetto);
        }
    });

    // Bump several times while writer is active.
    for(int round = 0; round < 5; ++round)
    {
        for(int i = 0; i < WRITES_PER_ROUND; ++i)
            std::this_thread::yield();
        const auto sid = registry.bump_session();
        EXPECT_GE(sid, 2u);
    }
    stop.store(true, std::memory_order_relaxed);
    writer.join();

    // After all bumps, rows() returns only the final-session rows;
    // every returned row carries the current session_id (no torn
    // entries with stale ids).
    const auto rows = registry.rows();
    for(const auto& r : rows)
        EXPECT_EQ(r.session_id, 6u);
}

TEST(output_file_registry_singleton, registry_accessor_returns_same_instance)
{
    auto& a = rocprofsys::registry();
    auto& b = rocprofsys::registry();
    EXPECT_EQ(&a, &b);
}

namespace
{
rocprofsys::output_file
make_row(const std::string& path, pid_t pid,
         std::optional<std::uintmax_t> size_bytes = std::nullopt,
         rocprofsys::output_format     format     = rocprofsys::output_format::perfetto)
{
    rocprofsys::output_file f{};
    f.label      = "row";
    f.path       = path;
    f.viewer     = "viewer";
    f.pid        = pid;
    f.size_bytes = size_bytes;
    (void) format;
    return f;
}

rocprofsys::output::process_metadata
make_meta(pid_t pid, pid_t ppid, std::string command = "", std::vector<int> gpu_ids = {})
{
    rocprofsys::output::process_metadata m{};
    m.pid     = pid;
    m.ppid    = ppid;
    m.command = std::move(command);
    m.gpu_ids = std::move(gpu_ids);
    return m;
}
}  // namespace

TEST(process_tree_builder, single_pid_becomes_single_root)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("a", 100) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 100);
    EXPECT_EQ(built.roots.front().rows.size(), 1u);
    EXPECT_TRUE(built.roots.front().children.empty());
    EXPECT_TRUE(built.diagnostics.missing_metadata_pids.empty());
}

TEST(process_tree_builder, parent_with_two_children_nests_under_parent)
{
    std::vector<rocprofsys::output_file> rows{ make_row("p", 100), make_row("c1", 200),
                                               make_row("c2", 201) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1),
                                                                 make_meta(200, 100),
                                                                 make_meta(201, 100) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 100);
    ASSERT_EQ(built.roots.front().children.size(), 2u);
    EXPECT_EQ(built.roots.front().children[0].meta.pid, 200);
    EXPECT_EQ(built.roots.front().children[1].meta.pid, 201);
}

TEST(process_tree_builder, orphan_with_missing_ppid_attaches_at_root)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("p", 100),
                                               make_row("orphan", 999) };
    std::vector<rocprofsys::output::process_metadata> processes{
        make_meta(100, -1), make_meta(999, 12345 /* unknown ppid */)
    };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 2u);
    EXPECT_EQ(built.roots[0].meta.pid, 100);
    EXPECT_EQ(built.roots[1].meta.pid, 999);
}

TEST(process_tree_builder, missing_metadata_pid_is_diagnosed)
{
    std::vector<rocprofsys::output_file>              rows{ make_row("p", 100),
                                               make_row("ghost", 555) };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    EXPECT_EQ(built.diagnostics.missing_metadata_pids, (std::vector<pid_t>{ 555 }));
    // Ghost still renders at root depth (RF1 graceful degradation).
    ASSERT_EQ(built.roots.size(), 2u);
}

TEST(process_tree_builder, deep_parent_chain_does_not_overflow_stack)
{
    // 1000-deep parent chain: pid 1000 → 999 → ... → 1 → orphan.
    // Recursive implementations blow the default 8 MiB stack here;
    // the iterative two-pass walk must succeed.
    constexpr int                                     CHAIN_DEPTH = 1000;
    std::vector<rocprofsys::output::process_metadata> processes;
    processes.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        processes.push_back(make_meta(pid, pid == 1 ? -1 : pid - 1));

    std::vector<rocprofsys::output_file> rows;
    rows.reserve(CHAIN_DEPTH);
    for(pid_t pid = 1; pid <= CHAIN_DEPTH; ++pid)
        rows.push_back(make_row(std::to_string(pid), pid));

    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    EXPECT_EQ(built.roots.front().meta.pid, 1);

    // Walk the chain iteratively in the test too — descend
    // CHAIN_DEPTH-1 children layers deep.
    const rocprofsys::output::process_node* cur   = &built.roots.front();
    int                                     depth = 1;
    while(!cur->children.empty())
    {
        ASSERT_EQ(cur->children.size(), 1u);
        cur = &cur->children.front();
        ++depth;
    }
    EXPECT_EQ(depth, CHAIN_DEPTH);
}

TEST(process_tree_builder, rows_sorted_descending_by_size)
{
    std::vector<rocprofsys::output_file> rows{
        make_row("small", 100, std::optional<std::uintmax_t>{ 1024 }),
        make_row("large", 100, std::optional<std::uintmax_t>{ 1024ULL * 1024 }),
        make_row("medium", 100, std::optional<std::uintmax_t>{ 4096 })
    };
    std::vector<rocprofsys::output::process_metadata> processes{ make_meta(100, -1) };
    auto built = rocprofsys::output::build_tree(rows, processes);
    ASSERT_EQ(built.roots.size(), 1u);
    const auto& sorted_rows = built.roots.front().rows;
    ASSERT_EQ(sorted_rows.size(), 3u);
    EXPECT_EQ(sorted_rows[0].path, "large");
    EXPECT_EQ(sorted_rows[1].path, "medium");
    EXPECT_EQ(sorted_rows[2].path, "small");
}

TEST(role_classifier, main_pid_gets_main_role)
{
    rocprofsys::output::process_node node{};
    node.meta = make_meta(100, -1);
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::main);
}

TEST(role_classifier, gpu_agent_gets_gpu_role)
{
    rocprofsys::output::process_node node{};
    node.meta = make_meta(200, 100, "", { 0 });
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::gpu);
}

TEST(role_classifier, parent_of_gpu_child_gets_engine_role)
{
    rocprofsys::output::process_node child{};
    child.meta = make_meta(200, 100, "", { 0 });
    rocprofsys::output::process_node parent{};
    parent.meta = make_meta(100, -1);
    parent.children.push_back(child);
    std::vector<rocprofsys::output::process_node> roots{ parent };
    rocprofsys::output::classify(roots, 999 /* main is elsewhere */);
    ASSERT_TRUE(roots.front().role.has_value());
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::engine);
    ASSERT_TRUE(roots.front().children.front().role.has_value());
    EXPECT_EQ(*roots.front().children.front().role, rocprofsys::output::role_hint::gpu);
}

TEST(role_classifier, main_takes_precedence_over_gpu)
{
    // Main PID with GPU agents must still classify as *main*.
    rocprofsys::output::process_node node{};
    node.meta = make_meta(100, -1, "", { 0 });
    std::vector<rocprofsys::output::process_node> roots{ node };
    rocprofsys::output::classify(roots, 100);
    EXPECT_EQ(*roots.front().role, rocprofsys::output::role_hint::main);
}

TEST(helper_collapser, collapses_helper_siblings)
{
    auto helper = [](pid_t pid) {
        rocprofsys::output::process_node n{};
        n.meta = make_meta(pid, 100);
        n.rows.push_back(make_row("tiny", pid, std::optional<std::uintmax_t>{ 4096 }));
        return n;
    };
    rocprofsys::output::process_node parent{};
    parent.meta     = make_meta(100, -1);
    parent.children = { helper(200), helper(201), helper(202) };
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_helpers(std::move(roots));
    ASSERT_EQ(collapsed.size(), 1u);
    ASSERT_EQ(collapsed.front().children.size(), 1u);
    const auto& rangenode = collapsed.front().children.front();
    ASSERT_TRUE(rangenode.collapsed.has_value());
    EXPECT_EQ(rangenode.collapsed->min_pid, 200);
    EXPECT_EQ(rangenode.collapsed->max_pid, 202);
    EXPECT_EQ(rangenode.collapsed->count, 3u);
}

TEST(helper_collapser, single_helper_is_not_collapsed)
{
    rocprofsys::output::process_node parent{};
    parent.meta = make_meta(100, -1);
    rocprofsys::output::process_node child{};
    child.meta = make_meta(200, 100);
    child.rows.push_back(make_row("tiny", 200, std::optional<std::uintmax_t>{ 4096 }));
    parent.children.push_back(child);
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_helpers(std::move(roots));
    ASSERT_EQ(collapsed.front().children.size(), 1u);
    EXPECT_FALSE(collapsed.front().children.front().collapsed.has_value());
}

TEST(helper_collapser, gpu_sibling_never_collapsed)
{
    auto helper = [](pid_t pid) {
        rocprofsys::output::process_node n{};
        n.meta = make_meta(pid, 100);
        n.rows.push_back(make_row("tiny", pid, std::optional<std::uintmax_t>{ 4096 }));
        return n;
    };
    rocprofsys::output::process_node gpu_node{};
    gpu_node.meta = make_meta(300, 100, "", { 0 });
    gpu_node.rows.push_back(
        make_row("big", 300, std::optional<std::uintmax_t>{ 100ULL * 1024 * 1024 }));
    rocprofsys::output::process_node parent{};
    parent.meta     = make_meta(100, -1);
    parent.children = { helper(200), helper(201), gpu_node };
    std::vector<rocprofsys::output::process_node> roots{ parent };
    auto collapsed = rocprofsys::output::collapse_helpers(std::move(roots));
    ASSERT_EQ(collapsed.front().children.size(), 2u);
    // First child: the GPU node (kept as-is, original order preserved
    // for non-helpers).
    EXPECT_EQ(collapsed.front().children[0].meta.pid, 300);
    EXPECT_FALSE(collapsed.front().children[0].collapsed.has_value());
    // Second child: the collapsed helper range.
    ASSERT_TRUE(collapsed.front().children[1].collapsed.has_value());
}

TEST(wrap_to_width, breaks_on_word_boundary_when_available)
{
    const auto chunks = rocprofsys::output::wrap_to_width(
        "the quick brown fox jumps over the lazy dog", 16);
    ASSERT_FALSE(chunks.empty());
    // Each chunk must end on a word (no trailing space; no leading space
    // beyond the first chunk) and be <= width.
    for(const auto& c : chunks)
    {
        EXPECT_LE(c.size(), 16u);
        EXPECT_FALSE(c.empty());
    }
    // Joining with " " round-trips the original string (since the
    // wrapper consumes the breaking space).
    std::string joined;
    for(std::size_t i = 0; i < chunks.size(); ++i)
    {
        if(i > 0) joined += ' ';
        joined += chunks[i];
    }
    EXPECT_EQ(joined, "the quick brown fox jumps over the lazy dog");
}

TEST(wrap_to_width, falls_back_to_byte_chunking_for_path_without_spaces)
{
    const std::string path   = "/very/long/path/with/many/components/and/no/spaces.txt";
    const auto        chunks = rocprofsys::output::wrap_to_width(path, 20);
    ASSERT_FALSE(chunks.empty());
    for(const auto& c : chunks)
        EXPECT_LE(c.size(), 20u);
    // Concatenation reconstructs the path exactly (no space dropped).
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, path);
}

TEST(wrap_to_width, never_splits_a_utf8_codepoint)
{
    // 6 box-drawing chars (3 bytes each) = 18 bytes. Width 10 forces
    // a split mid-string; the split must land between code points,
    // i.e. no chunk after the first may START with a UTF-8
    // continuation byte (10xxxxxx).
    const std::string content = "──────";  // 6 × U+2500
    const auto        chunks  = rocprofsys::output::wrap_to_width(content, 10);
    ASSERT_FALSE(chunks.empty());
    for(std::size_t i = 1; i < chunks.size(); ++i)
    {
        ASSERT_FALSE(chunks[i].empty());
        const auto first_byte = static_cast<unsigned char>(chunks[i].front());
        EXPECT_NE(first_byte & 0xC0, 0x80);
    }
    // Concatenation must reconstruct the input byte-for-byte.
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, content);
}

TEST(wrap_to_width, never_splits_a_utf8_codepoint_in_mixed_ascii_input)
{
    // Mixed ASCII + 3-byte UTF-8 (a─b─c─d─...): exercises the
    // backoff over an ASCII byte adjacent to a multi-byte code point.
    const std::string content = "a─b─c─d─e─f─g─h─i─j";
    const auto        chunks  = rocprofsys::output::wrap_to_width(content, 7);
    ASSERT_FALSE(chunks.empty());
    for(std::size_t i = 1; i < chunks.size(); ++i)
    {
        ASSERT_FALSE(chunks[i].empty());
        const auto first_byte = static_cast<unsigned char>(chunks[i].front());
        EXPECT_NE(first_byte & 0xC0, 0x80);
    }
    std::string joined;
    for(const auto& c : chunks)
        joined += c;
    EXPECT_EQ(joined, content);
}

TEST(summary_renderer_pipeline, fork_layout_emits_tree_glyphs_and_range_line)
{
    namespace fs = std::filesystem;

    // Build real files on disk so size_bytes is populated. Required
    // because is_helper is conservative on unknown sizes (rejects
    // collapse if size_bytes is nullopt).
    const auto base = fs::temp_directory_path() / "rocprofsys-pipeline-test";
    fs::create_directories(base);
    auto write_file = [&](const std::string& name, std::size_t bytes) {
        const auto        p = base / name;
        std::ofstream     o(p, std::ios::binary);
        const std::string buf(bytes, 'x');
        o.write(buf.data(), static_cast<std::streamsize>(buf.size()));
        return p.string();
    };

    // Synthetic: parent + 3 small-trace helper children + 1 GPU child.
    rocprofsys::output_file_registry registry;
    registry.register_file(write_file("main-trace.proto", 4096),
                           rocprofsys::output_format::perfetto, getpid());
    registry.register_file(write_file("helper-1.proto", 1024),
                           rocprofsys::output_format::perfetto, 1001);
    registry.register_file(write_file("helper-2.proto", 1024),
                           rocprofsys::output_format::perfetto, 1002);
    registry.register_file(write_file("helper-3.proto", 1024),
                           rocprofsys::output_format::perfetto, 1003);
    registry.register_file(write_file("worker-tp0.proto",
                                      100ULL * 1024 * 1024),  // 100 MiB
                           rocprofsys::output_format::perfetto, 1100);

    registry.record_process(make_meta(getpid(), -1, "main"));
    registry.record_process(make_meta(1001, getpid()));
    registry.record_process(make_meta(1002, getpid()));
    registry.record_process(make_meta(1003, getpid()));
    registry.record_process(make_meta(1100, getpid(), "worker", { 0 }));

    std::ostringstream oss;
    rocprofsys::output::print_summary(oss, registry);
    const auto out = oss.str();

    EXPECT_NE(out.find("Output Summary"), std::string::npos);
    EXPECT_NE(out.find("├─"), std::string::npos);
    EXPECT_NE(out.find("└─"), std::string::npos);
    // Helpers under 16 KiB AND no GPU agents → collapse into one line.
    EXPECT_NE(out.find("[1001..1003] (3 short-lived helpers)"), std::string::npos);
    // Worker has GPU agent → not collapsed; renders as its own row.
    EXPECT_NE(out.find("worker-tp0.proto"), std::string::npos);

    fs::remove_all(base);
}

TEST(output_file_registry, explicit_pid_is_preserved)
{
    rocprofsys::output_file_registry registry;
    constexpr pid_t                  CHILD_PID = 4242;
    registry.register_file("/tmp/rocprofsys-test/perfetto-trace.proto",
                           rocprofsys::output_format::perfetto, CHILD_PID);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_EQ(rows.front().pid, CHILD_PID);
}

TEST(output_file_registry, missing_file_yields_nullopt_size)
{
    namespace fs = std::filesystem;
    const auto missing =
        fs::temp_directory_path() / "rocprofsys-no-such-file-9b7c2.proto";
    fs::remove(missing);  // ensure absence

    rocprofsys::output_file_registry registry;
    registry.register_file(missing.string(), rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    EXPECT_FALSE(rows.front().size_bytes.has_value());
}

TEST(output_file_registry, existing_file_size_is_captured)
{
    namespace fs        = std::filesystem;
    const auto base_dir = fs::temp_directory_path() / "rocprofsys-registry-test";
    fs::create_directories(base_dir);
    const auto path = base_dir / "sized.bin";
    {
        std::ofstream     out(path, std::ios::binary);
        const std::string payload(2048, 'x');  // 2 KiB
        out.write(payload.data(), static_cast<std::streamsize>(payload.size()));
    }

    rocprofsys::output_file_registry registry;
    registry.register_file(path.string(), rocprofsys::output_format::perfetto);
    const auto rows = registry.rows();
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_TRUE(rows.front().size_bytes.has_value());
    EXPECT_EQ(*rows.front().size_bytes, 2048u);

    fs::remove(path);
}

TEST(perfetto_log_filter, drops_debug_and_info_levels)
{
    namespace pblog = ::perfetto::base;

    // Sanity check: filter_fn for each severity must not throw and
    // must return void. Drops happen silently (no observable side-
    // effect available without intercepting the rocprof-sys logger),
    // so we exercise each branch and rely on the ASAN/UBSAN runner
    // to catch any UB.
    pblog::LogMessageCallbackArgs args{};
    args.filename = "perfetto.cc";
    args.line     = 1;
    args.message  = "smoke";

    args.level = pblog::kLogDebug;
    rocprofsys::output::perfetto_log_filter::filter_fn(args);
    args.level = pblog::kLogInfo;
    rocprofsys::output::perfetto_log_filter::filter_fn(args);
    args.level = pblog::kLogImportant;
    rocprofsys::output::perfetto_log_filter::filter_fn(args);
    args.level = pblog::kLogError;
    rocprofsys::output::perfetto_log_filter::filter_fn(args);

    SUCCEED();
}

TEST(perfetto_log_filter, install_is_idempotent)
{
    // Double-install should not crash; std::call_once guards the
    // underlying SetLogMessageCallback so the second call is a no-op.
    rocprofsys::output::perfetto_log_filter::install();
    rocprofsys::output::perfetto_log_filter::install();
    SUCCEED();
}

TEST(output_file_registry, concurrent_register_is_thread_safe)
{
    rocprofsys::output_file_registry registry;
    constexpr int                    THREAD_COUNT = 4;
    constexpr int                    PER_THREAD   = 25;

    std::vector<std::thread> threads;
    threads.reserve(THREAD_COUNT);
    for(int t = 0; t < THREAD_COUNT; ++t)
    {
        threads.emplace_back([&registry, t]() {
            for(int i = 0; i < PER_THREAD; ++i)
            {
                registry.register_file("/tmp/rocprofsys-test/concurrent-" +
                                           std::to_string(t) + "-" + std::to_string(i) +
                                           ".proto",
                                       rocprofsys::output_format::perfetto);
            }
        });
    }
    for(auto& th : threads)
        th.join();

    EXPECT_EQ(registry.rows().size(),
              static_cast<std::size_t>(THREAD_COUNT * PER_THREAD));
}
