// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_csv.h"

#include "csv/csv.h"

#include <algorithm>
#include <string_view>

namespace
{
constexpr const char* kHeader = "id,name\n";

void write_row(std::ostream& out, const std::pair<int, std::string>& row)
{
    out << row.first << ',' << rocprofiler_compute_tool::csv::quote(row.second);
}
}  // namespace

std::string TestCsv::format(const std::vector<std::pair<int, std::string>>& rows)
{
    std::string text;
    m_batches = 0;

    EXPECT_TRUE(rocprofiler_compute_tool::csv::format(kHeader,
                                                      rows,
                                                      write_row,
                                                      [this, &text](std::string_view batch)
                                                      {
                                                          ++m_batches;
                                                          text.append(batch);
                                                          return true;
                                                      }));

    return text;
}

TEST_F(TestCsv, NoRows_WritesOnlyTheHeader)
{
    EXPECT_EQ(format({}), kHeader);
}

TEST_F(TestCsv, Rows_AreWrittenOnePerLineInOrder)
{
    EXPECT_EQ(format({{1, "first"}, {2, "second"}}),
              std::string{kHeader} + "1,\"first\"\n" + "2,\"second\"\n");
}

TEST_F(TestCsv, ManyRows_AreAllWrittenAcrossBatches)
{
    std::vector<std::pair<int, std::string>> rows;
    for (int i = 0; i < 50000; ++i)
        rows.emplace_back(i, "a kernel name long enough to fill the batch");

    const auto text = format(rows);

    EXPECT_EQ(std::count(text.begin(), text.end(), '\n'), rows.size() + 1);
    EXPECT_GT(m_batches, 1);
}

TEST_F(TestCsv, SinkFailureMidStream_StopsAndIsReported)
{
    std::vector<std::pair<int, std::string>> rows;
    for (int i = 0; i < 50000; ++i)
        rows.emplace_back(i, "a kernel name long enough to fill the batch");

    int calls = 0;

    EXPECT_FALSE(rocprofiler_compute_tool::csv::format(kHeader,
                                                       rows,
                                                       write_row,
                                                       [&calls](std::string_view)
                                                       { return ++calls < 2; }));
    EXPECT_EQ(calls, 2);
}

TEST_F(TestCsv, SinkFailureOnTheFinalBatch_IsReported)
{
    const std::vector<std::pair<int, std::string>> rows = {{1, "first"}};

    EXPECT_FALSE(rocprofiler_compute_tool::csv::format(kHeader,
                                                       rows,
                                                       write_row,
                                                       [](std::string_view) { return false; }));
}

TEST_F(TestCsv, FieldWithACommaOrQuote_IsQuoted)
{
    EXPECT_EQ(rocprofiler_compute_tool::csv::quote("a,b"), "\"a,b\"");
    EXPECT_EQ(rocprofiler_compute_tool::csv::quote("say \"hi\""), "\"say \"\"hi\"\"\"");
    EXPECT_EQ(rocprofiler_compute_tool::csv::quote(""), "\"\"");
}
