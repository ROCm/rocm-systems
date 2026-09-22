// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_dispatch_writer.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <iterator>
#include <string>
#include <string_view>

using namespace rocprofiler_compute_tool;

namespace
{
constexpr const char* kHeader = "dispatch_id,gpu_id,kernel_id,grid_size,workgroup_size,"
                                "lds_per_workgroup,scratch_per_workitem,start_timestamp,"
                                "end_timestamp,correlation_id\n";

std::filesystem::path test_directory()
{
    return std::filesystem::temp_directory_path() /
           ("dispatch_writer_test_" + std::to_string(::getpid()));
}
}  // namespace

std::string TestDispatchWriter::format()
{
    std::string csv;
    m_batches = 0;

    EXPECT_TRUE(format_dispatch_csv(m_tool_data,
                                    [this, &csv](std::string_view batch)
                                    {
                                        ++m_batches;
                                        csv.append(batch);
                                        return true;
                                    }));

    return csv;
}

dispatch_record_t TestDispatchWriter::make_record(uint64_t dispatch_id, uint64_t kernel_id)
{
    dispatch_record_t record{};
    record.dispatch_id          = dispatch_id;
    record.agent_id             = 2;
    record.kernel_id            = kernel_id;
    record.grid_size            = 1048576;
    record.workgroup_size       = 256;
    record.lds_per_workgroup    = 16384;
    record.scratch_per_workitem = 0;
    record.start_timestamp      = 1000;
    record.end_timestamp        = 2000;
    record.correlation_id       = dispatch_id + 1;
    return record;
}

TEST_F(TestDispatchWriter, NoRecords_WritesOnlyTheHeader)
{
    EXPECT_EQ(format(), kHeader);
}

TEST_F(TestDispatchWriter, Records_AreWrittenOnePerLineInOrder)
{
    m_tool_data.dispatch_records = {make_record(0, 7), make_record(1, 8)};

    EXPECT_EQ(format(),
              std::string{kHeader} + "0,2,7,1048576,256,16384,0,1000,2000,1\n" +
                  "1,2,8,1048576,256,16384,0,1000,2000,2\n");
}

TEST_F(TestDispatchWriter, ManyRecords_AreAllWrittenAcrossBatches)
{
    // More rows than fit in one batch.
    constexpr int kRecords = 50000;
    for (int i = 0; i < kRecords; ++i)
        m_tool_data.dispatch_records.push_back(make_record(static_cast<uint64_t>(i), 7));

    const auto csv = format();

    EXPECT_EQ(std::count(csv.begin(), csv.end(), '\n'), kRecords + 1);
    EXPECT_EQ(csv.compare(0, std::string{kHeader}.size(), kHeader), 0);
    EXPECT_GT(m_batches, 1);
}

TEST_F(TestDispatchWriter, SinkFailureMidStream_StopsAndIsReported)
{
    constexpr int kRecords = 50000;
    for (int i = 0; i < kRecords; ++i)
        m_tool_data.dispatch_records.push_back(make_record(static_cast<uint64_t>(i), 7));

    int calls = 0;

    EXPECT_FALSE(format_dispatch_csv(m_tool_data, [&calls](std::string_view) { return ++calls < 2; }));
    EXPECT_EQ(calls, 2);
}

TEST_F(TestDispatchWriter, NoRecords_WritesNoFile)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    m_tool_data.dispatch_filename = (directory / "1234_dispatch.csv.gz").string();

    DispatchWriter writer;
    writer.write(m_tool_data);

    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              0);

    std::filesystem::remove_all(directory);
}

TEST_F(TestDispatchWriter, DispatchFilename_IsTheOnlyFileWritten)
{
    const auto directory = test_directory();
    std::filesystem::remove_all(directory);
    ASSERT_TRUE(std::filesystem::create_directories(directory));
    const auto path = directory / "1234_dispatch.csv.gz";

    m_tool_data.dispatch_filename = path.string();
    m_tool_data.dispatch_records  = {make_record(0, 7)};

    DispatchWriter writer;
    writer.write(m_tool_data);

    EXPECT_TRUE(std::filesystem::exists(path));
    EXPECT_FALSE(std::filesystem::exists(path.string() + ".tmp"));
    EXPECT_EQ(std::distance(std::filesystem::directory_iterator{directory},
                            std::filesystem::directory_iterator{}),
              1);

    std::filesystem::remove_all(directory);
}

TEST_F(TestDispatchWriter, UnopenableOutput_DoesNotCrash)
{
    const std::filesystem::path path = "/nonexistent-directory/dispatch.csv.gz";
    m_tool_data.dispatch_filename    = path.string();
    m_tool_data.dispatch_records     = {make_record(0, 7)};

    DispatchWriter writer;
    EXPECT_NO_THROW(writer.write(m_tool_data));
    EXPECT_FALSE(std::filesystem::exists(path));
}
