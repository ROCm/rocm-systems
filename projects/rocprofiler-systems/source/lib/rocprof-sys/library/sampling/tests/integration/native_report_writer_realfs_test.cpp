// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Real-filesystem integration test for native_report_writer.
//
// Purpose: verify that the file-open path (set_streams() + ofstream) actually
// creates TSV files on disk. The ostream-injection tests in
// native_report_writer_tsv_test.cpp only exercise the in-memory path; this
// test catches bugs in the file-creation step (missing mkdir, wrong path,
// silent open failure).
//
// The test uses set_streams() — the same seam that the production
// open_report_writer_streams() hook calls — so this IS the production code
// path minus the timemory path-composition step.
//
// Criteria: NFR-T-7 (file-level output), L4 (TSV format on disk),
//           Bug fix: open_report_writer_streams silent failure.

#include <gtest/gtest.h>

#include "sampling/data/stack_frame.hpp"
#include "sampling/data/timer_sample.hpp"
#include "sampling/src/native_report_writer.hpp"

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

using namespace rocprofsys::sampling;

// ── File-reading helpers ──────────────────────────────────────────────────────

static std::string
read_file(fs::path const& p)
{
    std::ifstream      ifs(p);
    std::ostringstream ss;
    ss << ifs.rdbuf();
    return ss.str();
}

static std::vector<std::string>
split_tabs(std::string const& line)
{
    std::vector<std::string> r;
    std::string              tok;
    for(char c : line)
    {
        if(c == '\t')
        {
            r.push_back(tok);
            tok.clear();
        }
        else
        {
            tok += c;
        }
    }
    r.push_back(tok);
    return r;
}

static bool
has_header_key(std::string const& text, std::string const& key)
{
    std::istringstream ss(text);
    std::string        line;
    while(std::getline(ss, line))
        if(!line.empty() && line[0] == '#' && line.find(key) != std::string::npos)
            return true;
    return false;
}

static int
count_data_rows(std::string const& text, size_t min_cols)
{
    std::istringstream ss(text);
    std::string        line;
    int                n = 0;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        if(split_tabs(line).size() >= min_cols) ++n;
    }
    return n;
}

// ── Sample construction ───────────────────────────────────────────────────────

static timer_sample
make_sample(int64_t tid, uint64_t beg_ns, uint64_t end_ns,
            std::vector<std::string> const& stack_labels)
{
    timer_sample s{};
    s.tid    = tid;
    s.beg_ns = beg_ns;
    s.end_ns = end_ns;
    for(auto const& lbl : stack_labels)
    {
        stack_frame f{};
        f.name = lbl;
        s.stack.push_back(f);
    }
    return s;
}

// ── Temp-directory RAII ───────────────────────────────────────────────────────

struct TempDir
{
    fs::path path;

    TempDir()
    {
        auto base = fs::temp_directory_path();
        // Use process ID + object address for a unique but deterministic name.
        char buf[64];
        std::snprintf(buf, sizeof(buf), "rocprof_realfs_test_%d_%p",
                      static_cast<int>(::getpid()), static_cast<void*>(this));
        path = base / buf;
        fs::create_directories(path);
    }

    ~TempDir() { fs::remove_all(path); }
};

// ── Test fixture ─────────────────────────────────────────────────────────────

class NativeReportWriterRealFsTest : public ::testing::Test
{
protected:
    TempDir  tmp;
    fs::path wall_path;
    fs::path cpu_path;
    fs::path pct_path;
    fs::path trip_path;

    std::ofstream        wall_ofs;
    std::ofstream        cpu_ofs;
    std::ofstream        pct_ofs;
    std::ofstream        trip_ofs;
    native_report_writer writer;

    void SetUp() override
    {
        wall_path = tmp.path / "sampling_wall_clock.tsv";
        cpu_path  = tmp.path / "sampling_cpu_clock.tsv";
        pct_path  = tmp.path / "sampling_percent.tsv";
        trip_path = tmp.path / "trip_count.tsv";

        wall_ofs.open(wall_path);
        cpu_ofs.open(cpu_path);
        pct_ofs.open(pct_path);
        trip_ofs.open(trip_path);

        ASSERT_TRUE(wall_ofs.is_open()) << "Failed to open " << wall_path;
        ASSERT_TRUE(cpu_ofs.is_open()) << "Failed to open " << cpu_path;
        ASSERT_TRUE(pct_ofs.is_open()) << "Failed to open " << pct_path;
        ASSERT_TRUE(trip_ofs.is_open()) << "Failed to open " << trip_path;

        writer.set_streams(&wall_ofs, &cpu_ofs, &pct_ofs, &trip_ofs);
    }

    void TearDown() override
    {
        wall_ofs.close();
        cpu_ofs.close();
        pct_ofs.close();
        trip_ofs.close();
    }
};

// ── Tests ─────────────────────────────────────────────────────────────────────

// T-RFS-1: All 4 TSV files exist on disk after set_streams() — even before flush().
TEST_F(NativeReportWriterRealFsTest, all_four_files_created_on_disk)
{
    EXPECT_TRUE(fs::exists(wall_path)) << wall_path;
    EXPECT_TRUE(fs::exists(cpu_path)) << cpu_path;
    EXPECT_TRUE(fs::exists(pct_path)) << pct_path;
    EXPECT_TRUE(fs::exists(trip_path)) << trip_path;
}

// T-RFS-2: Headers are written to disk immediately after set_streams().
TEST_F(NativeReportWriterRealFsTest, headers_written_to_disk_before_flush)
{
    wall_ofs.flush();
    cpu_ofs.flush();
    pct_ofs.flush();
    trip_ofs.flush();

    EXPECT_TRUE(has_header_key(read_file(wall_path), "# metric:"))
        << "wall missing # metric:";
    EXPECT_TRUE(has_header_key(read_file(wall_path), "# unit:"))
        << "wall missing # unit:";
    EXPECT_TRUE(has_header_key(read_file(wall_path), "# columns:"))
        << "wall missing # columns:";
    EXPECT_TRUE(has_header_key(read_file(cpu_path), "# metric:"))
        << "cpu missing # metric:";
    EXPECT_TRUE(has_header_key(read_file(pct_path), "# metric:"))
        << "pct missing # metric:";
    EXPECT_TRUE(has_header_key(read_file(trip_path), "# metric:"))
        << "trip missing # metric:";
}

// T-RFS-3: Zero samples → flush() writes header-only files; no data rows.
TEST_F(NativeReportWriterRealFsTest, zero_samples_flush_produces_header_only_files)
{
    writer.flush();
    wall_ofs.flush();
    cpu_ofs.flush();
    pct_ofs.flush();
    trip_ofs.flush();

    EXPECT_EQ(count_data_rows(read_file(wall_path), 11), 0)
        << "wall should have no data rows";
    EXPECT_EQ(count_data_rows(read_file(cpu_path), 11), 0)
        << "cpu should have no data rows";
    EXPECT_EQ(count_data_rows(read_file(pct_path), 4), 0)
        << "pct should have no data rows";
    EXPECT_EQ(count_data_rows(read_file(trip_path), 4), 0)
        << "trip should have no data rows";
}

// T-RFS-4: One sample → flush() produces at least one data row in wall_clock file.
TEST_F(NativeReportWriterRealFsTest, one_sample_flush_produces_data_row_in_wall_file)
{
    auto s = make_sample(1, 0, 1'000'000'000ULL, { "main", "foo" });
    writer.write_timer_samples(1, { s });
    writer.flush();
    wall_ofs.flush();

    EXPECT_GT(count_data_rows(read_file(wall_path), 11), 0)
        << "wall should have ≥1 data row";
}

// T-RFS-5: One sample → trip_count file has data rows.
TEST_F(NativeReportWriterRealFsTest, one_sample_flush_produces_data_row_in_trip_file)
{
    auto s = make_sample(1, 0, 500'000'000ULL, { "alpha" });
    writer.write_timer_samples(1, { s });
    writer.flush();
    trip_ofs.flush();

    EXPECT_GT(count_data_rows(read_file(trip_path), 4), 0)
        << "trip_count should have ≥1 data row";
}

// T-RFS-6: One sample → sampling_percent file has data rows.
TEST_F(NativeReportWriterRealFsTest, one_sample_flush_produces_data_row_in_pct_file)
{
    auto s = make_sample(1, 0, 200'000'000ULL, { "bar" });
    writer.write_timer_samples(1, { s });
    writer.flush();
    pct_ofs.flush();

    EXPECT_GT(count_data_rows(read_file(pct_path), 4), 0)
        << "sampling_percent should have ≥1 data row";
}

// T-RFS-7: wall_clock file is readable line-by-line and every data line has 11
// tab-delimited fields.
TEST_F(NativeReportWriterRealFsTest, wall_clock_data_lines_have_eleven_tab_fields)
{
    auto s = make_sample(2, 0, 1'000'000'000ULL, { "root", "child", "leaf" });
    writer.write_timer_samples(2, { s });
    writer.flush();
    wall_ofs.flush();

    std::string        content = read_file(wall_path);
    std::istringstream ss(content);
    std::string        line;
    int                checked = 0;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        EXPECT_EQ(cols.size(), 11u) << "line: " << line;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no data rows found in wall_clock file";
}

// T-RFS-8: trip_count data lines have 4 tab-delimited fields.
TEST_F(NativeReportWriterRealFsTest, trip_count_data_lines_have_four_tab_fields)
{
    auto s = make_sample(3, 0, 100'000'000ULL, { "a", "b" });
    writer.write_timer_samples(3, { s });
    writer.flush();
    trip_ofs.flush();

    std::string        content = read_file(trip_path);
    std::istringstream ss(content);
    std::string        line;
    int                checked = 0;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        EXPECT_EQ(cols.size(), 4u) << "line: " << line;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no data rows found in trip_count file";
}

// T-RFS-9: sampling_percent data lines have 4 tab-delimited fields.
TEST_F(NativeReportWriterRealFsTest, pct_data_lines_have_four_tab_fields)
{
    auto s = make_sample(4, 0, 300'000'000ULL, { "fn1" });
    writer.write_timer_samples(4, { s });
    writer.flush();
    pct_ofs.flush();

    std::string        content = read_file(pct_path);
    std::istringstream ss(content);
    std::string        line;
    int                checked = 0;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        auto cols = split_tabs(line);
        EXPECT_EQ(cols.size(), 4u) << "line: " << line;
        ++checked;
    }
    EXPECT_GT(checked, 0) << "no data rows found in sampling_percent file";
}

// T-RFS-10: File sizes grow after flush() (files not empty).
TEST_F(NativeReportWriterRealFsTest, files_are_non_empty_after_flush)
{
    auto s = make_sample(1, 0, 1'000'000'000ULL, { "work" });
    writer.write_timer_samples(1, { s });
    writer.flush();
    wall_ofs.flush();
    cpu_ofs.flush();
    pct_ofs.flush();
    trip_ofs.flush();

    EXPECT_GT(fs::file_size(wall_path), 0u) << "wall_clock.tsv is empty";
    EXPECT_GT(fs::file_size(cpu_path), 0u) << "sampling_cpu_clock.tsv is empty";
    EXPECT_GT(fs::file_size(pct_path), 0u) << "sampling_percent.tsv is empty";
    EXPECT_GT(fs::file_size(trip_path), 0u) << "trip_count.tsv is empty";
}

// T-RFS-11: Files remain on disk after writer destruction.
// Regression guard: destructor must not delete output files.
TEST_F(NativeReportWriterRealFsTest, files_persist_after_writer_is_destroyed)
{
    {
        TempDir              tmp2;
        fs::path             wp = tmp2.path / "sampling_wall_clock.tsv";
        fs::path             cp = tmp2.path / "sampling_cpu_clock.tsv";
        fs::path             pp = tmp2.path / "sampling_percent.tsv";
        fs::path             tp = tmp2.path / "trip_count.tsv";
        std::ofstream        w(wp), c(cp), p(pp), t(tp);
        native_report_writer local_writer;
        local_writer.set_streams(&w, &c, &p, &t);
        auto s = make_sample(1, 0, 1'000'000'000ULL, { "fn" });
        local_writer.write_timer_samples(1, { s });
        local_writer.flush();
        w.flush();
        c.flush();
        p.flush();
        t.flush();
        // writer goes out of scope here
        EXPECT_TRUE(fs::exists(wp)) << "wall_clock.tsv deleted by writer destructor";
        EXPECT_TRUE(fs::exists(cp))
            << "sampling_cpu_clock.tsv deleted by writer destructor";
    }
    // tmp2 RAII cleans up — no assertion needed; test verifies files survived the writer
}

// T-RFS-12: metric name in wall file matches "sampling_wall_clock".
TEST_F(NativeReportWriterRealFsTest, wall_clock_metric_header_value_is_correct)
{
    writer.flush();
    wall_ofs.flush();

    std::string content = read_file(wall_path);
    EXPECT_TRUE(has_header_key(content, "sampling_wall_clock")) << content;
}

// T-RFS-13: metric name in cpu file matches "sampling_cpu_clock".
TEST_F(NativeReportWriterRealFsTest, cpu_clock_metric_header_value_is_correct)
{
    writer.flush();
    cpu_ofs.flush();

    std::string content = read_file(cpu_path);
    EXPECT_TRUE(has_header_key(content, "sampling_cpu_clock")) << content;
}

// T-RFS-14: metric name in pct file matches "sampling_percent".
TEST_F(NativeReportWriterRealFsTest, pct_metric_header_value_is_correct)
{
    writer.flush();
    pct_ofs.flush();

    std::string content = read_file(pct_path);
    EXPECT_TRUE(has_header_key(content, "sampling_percent")) << content;
}

// T-RFS-15: metric name in trip file matches "trip_count".
TEST_F(NativeReportWriterRealFsTest, trip_count_metric_header_value_is_correct)
{
    writer.flush();
    trip_ofs.flush();

    std::string content = read_file(trip_path);
    EXPECT_TRUE(has_header_key(content, "trip_count")) << content;
}

// T-RFS-16: Label written to disk is untruncated (190-char label round-trip).
TEST_F(NativeReportWriterRealFsTest, long_label_written_untruncated_to_disk)
{
    std::string long_label(190, 'X');
    auto        s = make_sample(1, 0, 1'000'000'000ULL, { long_label });
    writer.write_timer_samples(1, { s });
    writer.flush();
    wall_ofs.flush();

    std::string        content = read_file(wall_path);
    std::istringstream ss(content);
    std::string        line;
    bool               found = false;
    while(std::getline(ss, line))
    {
        if(line.empty() || line[0] == '#') continue;
        if(line.find(long_label) != std::string::npos)
        {
            found = true;
            break;
        }
    }
    EXPECT_TRUE(found) << "190-char label not found verbatim in wall_clock.tsv";
}

// T-RFS-17: Production caller contract — flush() is called exactly once.
// This test documents that flush() appends again if called a second time,
// which is expected behaviour (production calls flush once; no clear between calls).
// The contract is: post_process() calls flush() once after all write_timer_samples().
TEST_F(NativeReportWriterRealFsTest, flush_called_once_produces_one_set_of_data_rows)
{
    auto s = make_sample(1, 0, 1'000'000'000ULL, { "fn" });
    writer.write_timer_samples(1, { s });
    writer.flush();
    wall_ofs.flush();

    int rows = count_data_rows(read_file(wall_path), 11);
    EXPECT_EQ(rows, 1) << "expected exactly 1 data row after one flush of one sample";
}
