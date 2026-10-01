// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_counters_writer.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr const char* kHeader =
    "dispatch_id,gpu_id,kernel_id,lds_per_workgroup,counter_id,counter_name,counter_value\n";

std::filesystem::path test_directory()
{
    return std::filesystem::temp_directory_path() /
           ("counters_writer_test_" + std::to_string(::getpid()));
}
}  // namespace

std::string TestCountersWriter::format()
{
    std::string csv;
    m_batches = 0;

    EXPECT_TRUE(format_counters_csv(m_tool_data,
                                    [this, &csv](std::string_view batch)
                                    {
                                        ++m_batches;
                                        csv.append(batch);
                                        return true;
                                    }));

    return csv;
}

counter_info_record_t TestCountersWriter::make_record(uint64_t    dispatch_id,
                                                      std::string counter_name,
                                                      double      counter_value)
{
    counter_info_record_t record{};
    record.dispatch_id     = dispatch_id;
    record.agent_id        = 2;
    record.kernel_id       = 3;
    record.LDS_memory_size = 4;
    record.counter_id      = 5;
    record.counter_name    = std::move(counter_name);
    record.counter_value   = counter_value;
    return record;
}

TEST_F(TestCountersWriter, NoRecords_WritesOnlyTheHeader)
{
    EXPECT_EQ(format(), kHeader);
}

TEST_F(TestCountersWriter, Records_AreWrittenOnePerLineInOrder)
{
    m_tool_data.counter_records = {make_record(0, "SQ_WAVES", 10), make_record(1, "SQ_WAVES", 20)};

    EXPECT_EQ(format(), std::string{kHeader} + "0,2,3,4,5,SQ_WAVES,10\n" + "1,2,3,4,5,SQ_WAVES,20\n");
}

TEST_F(TestCountersWriter, CounterValue_KeepsDefaultOstreamFormatting)
{
    // ostream formatting is part of the analyze contract.
    m_tool_data.counter_records = {make_record(0, "SQ_WAVES", 0.5),
                                   make_record(1, "SQ_WAVES", 1234567890.0),
                                   make_record(2, "SQ_WAVES", 1e-7)};

    std::ostringstream expected;
    expected << kHeader;
    expected << "0,2,3,4,5,SQ_WAVES," << 0.5 << '\n';
    expected << "1,2,3,4,5,SQ_WAVES," << 1234567890.0 << '\n';
    expected << "2,2,3,4,5,SQ_WAVES," << 1e-7 << '\n';

    EXPECT_EQ(format(), expected.str());
}

TEST_F(TestCountersWriter, ManyRecords_AreAllWrittenAcrossBatches)
{
    // More rows than fit in one batch.
    constexpr int kRecords = 50000;
    for (int i = 0; i < kRecords; ++i)
        m_tool_data.counter_records.push_back(make_record(static_cast<uint64_t>(i), "SQ_WAVES", i));

    const auto csv = format();

    EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), kRecords + 1);
    EXPECT_EQ(csv.compare(0, std::string{kHeader}.size(), kHeader), 0);
    EXPECT_NE(csv.find("\n49999,2,3,4,5,SQ_WAVES,49999\n"), std::string::npos);
    EXPECT_GT(m_batches, 1);
}

TEST_F(TestCountersWriter, SinkFailureMidStream_StopsAndIsReported)
{
    constexpr int kRecords = 50000;
    for (int i = 0; i < kRecords; ++i)
        m_tool_data.counter_records.push_back(make_record(static_cast<uint64_t>(i), "SQ_WAVES", i));

    int calls = 0;

    EXPECT_FALSE(format_counters_csv(m_tool_data, [&calls](std::string_view) { return ++calls < 2; }));
    EXPECT_EQ(calls, 2);
}

TEST_F(TestCountersWriter, SinkFailureOnTheFinalBatch_IsReported)
{
    m_tool_data.counter_records = {make_record(0, "SQ_WAVES", 10)};

    EXPECT_FALSE(format_counters_csv(m_tool_data, [](std::string_view) { return false; }));
}

TEST_F(TestCountersWriter, OutputFilename_IsTheOnlyFileWritten)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    const auto path = directory / "1234_native_counter_collection.csv.gz";

    m_tool_data.output_filename = path.string();
    m_tool_data.counter_records = {make_record(0, "SQ_WAVES", 10)};

    CsvCountersWriter writer;
    writer.write_counters(&m_tool_data);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              1);

    std::filesystem::remove_all(directory);
}

TEST_F(TestCountersWriter, UnopenableOutput_DoesNotCrash)
{
    const std::filesystem::path path = "/nonexistent-directory/counters.csv.gz";
    m_tool_data.output_filename      = path.string();
    m_tool_data.counter_records      = {make_record(0, "SQ_WAVES", 10)};

    CsvCountersWriter writer;
    EXPECT_NO_THROW(writer.write_counters(&m_tool_data));
    EXPECT_FALSE(std::filesystem::exists(path));
}

TEST_F(TestCountersWriter, Writer_WritesGzip)
{
    const auto path = std::filesystem::temp_directory_path() /
                      ("counters_writer_test_" + std::to_string(::getpid()) + ".csv.gz");
    m_tool_data.output_filename = path.string();
    m_tool_data.counter_records = {make_record(0, "SQ_WAVES", 10)};

    CsvCountersWriter writer;
    writer.write_counters(&m_tool_data);

    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.is_open());
    char magic[2] = {};
    file.read(magic, sizeof(magic));
    file.close();
    std::filesystem::remove(path);

    EXPECT_EQ(static_cast<unsigned char>(magic[0]), 0x1f);
    EXPECT_EQ(static_cast<unsigned char>(magic[1]), 0x8b);
}
