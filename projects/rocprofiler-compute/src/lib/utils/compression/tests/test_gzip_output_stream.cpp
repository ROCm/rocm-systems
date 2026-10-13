// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT
#include "test_gzip_output_stream.h"

#include "compression/gzip_output_stream.h"

#include <unistd.h>
#include <zlib.h>

#include <fstream>
#include <string>
#include <vector>

using namespace rocprofiler_compute_tool::compression;

std::string gunzip(const std::string& compressed)
{
    z_stream zs{};
    // 15 + 16 selects gzip framing, matching what the writer produces.
    EXPECT_EQ(inflateInit2(&zs, 15 + 16), Z_OK);

    zs.next_in  = reinterpret_cast<Bytef*>(const_cast<char*>(compressed.data()));
    zs.avail_in = static_cast<uInt>(compressed.size());

    std::string       out;
    std::vector<char> buffer(64 * 1024);
    int               status = Z_OK;
    do
    {
        zs.next_out  = reinterpret_cast<Bytef*>(buffer.data());
        zs.avail_out = static_cast<uInt>(buffer.size());

        status = inflate(&zs, Z_NO_FLUSH);
        if (status != Z_OK && status != Z_STREAM_END)
        {
            inflateEnd(&zs);
            ADD_FAILURE() << "inflate failed with " << status;
            return {};
        }
        out.append(buffer.data(), buffer.size() - zs.avail_out);
    }
    while (status != Z_STREAM_END);

    inflateEnd(&zs);
    return out;
}

void TestGzipOutputStream::SetUp()
{
    m_path = std::filesystem::temp_directory_path() /
             ("gzip_stream_test_" + std::to_string(::getpid()) + ".csv.gz");
    std::filesystem::remove(m_path);
}

void TestGzipOutputStream::TearDown()
{
    std::filesystem::remove(m_path);
}

std::string TestGzipOutputStream::raw_bytes() const
{
    std::ifstream file(m_path, std::ios::binary);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

std::string TestGzipOutputStream::compress(const std::string& data)
{
    GzipFileOutputStream stream(m_path.string());
    EXPECT_TRUE(stream.write(data));
    EXPECT_TRUE(stream.close());
    return raw_bytes();
}

TEST_F(TestGzipOutputStream, WrittenData_RoundTrips)
{
    const std::string data = "dispatch_id,gpu_id\n0,1\n2,3\n";
    EXPECT_EQ(gunzip(compress(data)), data);
}

TEST_F(TestGzipOutputStream, NoData_ProducesAnEmptyButValidMember)
{
    // Empty member must still be a valid gzip file, not zero bytes.
    const auto compressed = compress("");
    EXPECT_FALSE(compressed.empty());
    EXPECT_EQ(gunzip(compressed), "");
}

TEST_F(TestGzipOutputStream, ManyWrites_RoundTripAsOneStream)
{
    std::string expected;
    {
        GzipFileOutputStream stream(m_path.string());
        for (int i = 0; i < 10000; ++i)
        {
            const auto row = "0,1,2,3,4,SQ_WAVES," + std::to_string(i) + "\n";
            ASSERT_TRUE(stream.write(row));
            expected += row;
        }
        ASSERT_TRUE(stream.close());
    }

    EXPECT_EQ(gunzip(raw_bytes()), expected);
}

TEST_F(TestGzipOutputStream, OutOfRangeLevel_StillCompresses)
{
    const std::string data(64 * 1024, 'x');
    {
        GzipFileOutputStream stream(m_path.string(), 10);
        ASSERT_TRUE(stream.write(data));
        ASSERT_TRUE(stream.close());
    }

    const auto compressed = raw_bytes();
    EXPECT_EQ(gunzip(compressed), data);
    EXPECT_LT(compressed.size(), data.size());
}

TEST_F(TestGzipOutputStream, IncompressibleDataLargerThanTheInternalBuffer_RoundTrips)
{
    // Incompressible input forces deflate to spill its output buffer repeatedly.
    std::string data;
    data.reserve(1 << 20);
    for (std::size_t i = 0; i < (1 << 20); ++i)
        data.push_back(static_cast<char>((i * 2654435761u) >> 24));

    EXPECT_EQ(gunzip(compress(data)), data);
}

TEST_F(TestGzipOutputStream, RepetitiveData_IsSmallerThanTheInput)
{
    std::string data;
    for (int i = 0; i < 5000; ++i)
        data += "0,1,2,3,4,SQ_WAVES,100\n";

    EXPECT_LT(compress(data).size(), data.size());
}

TEST_F(TestGzipOutputStream, StartsAsOneGzipMember)
{
    const auto compressed = compress("some counter rows\n");
    ASSERT_GE(compressed.size(), 3u);
    EXPECT_EQ(static_cast<unsigned char>(compressed[0]), 0x1f);
    EXPECT_EQ(static_cast<unsigned char>(compressed[1]), 0x8b);
    EXPECT_EQ(static_cast<unsigned char>(compressed[2]), 0x08);
}

TEST_F(TestGzipOutputStream, WriteAfterClose_LeavesTheStreamFailed)
{
    GzipFileOutputStream stream(m_path.string());

    ASSERT_TRUE(stream.write("rows\n"));
    ASSERT_TRUE(stream.close());
    ASSERT_FALSE(stream.write("late"));

    EXPECT_FALSE(stream.ok());
    EXPECT_FALSE(stream.close());
}

TEST_F(TestGzipOutputStream, CloseIsIdempotent)
{
    GzipFileOutputStream stream(m_path.string());

    ASSERT_TRUE(stream.write("rows\n"));
    ASSERT_TRUE(stream.close());
    const auto after_first_close = raw_bytes();

    EXPECT_TRUE(stream.close());
    EXPECT_EQ(raw_bytes(), after_first_close);
}

TEST_F(TestGzipOutputStream, DestructorClosesTheStream)
{
    // Writer relies on the destructor to emit the gzip trailer.
    {
        GzipFileOutputStream stream(m_path.string());
        ASSERT_TRUE(stream.write("rows\n"));
    }
    EXPECT_EQ(gunzip(raw_bytes()), "rows\n");
}

TEST_F(TestGzipOutputStream, UnwritablePath_FailsRatherThanCrashing)
{
    GzipFileOutputStream stream("/nonexistent-directory/counters.csv.gz");

    EXPECT_FALSE(stream.is_open());
    EXPECT_FALSE(stream.ok());
    EXPECT_FALSE(stream.write("data"));
    EXPECT_FALSE(stream.close());
}
