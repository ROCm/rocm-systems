// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <vector>

#include "mi400/mi400token.h"
#include "rocprof_trace_decoder/rocprof_trace_decoder.h"

namespace
{
class BitStreamBuilder
{
public:
    void writeBits(uint64_t value, size_t count)
    {
        for (size_t i = 0; i < count; ++i)
        {
            const size_t byte_idx = bit_pos_ / 8;
            const size_t bit_idx = bit_pos_ % 8;
            if (byte_idx >= data_.size()) data_.resize(byte_idx + 1, 0);
            if ((value >> i) & 1u) data_[byte_idx] |= static_cast<uint8_t>(1u << bit_idx);
            ++bit_pos_;
        }
    }

    std::vector<uint8_t> finish(size_t trailing_zero_bytes)
    {
        while ((bit_pos_ % 8) != 0) writeBits(0, 1);
        data_.resize(data_.size() + trailing_zero_bytes, 0);
        return data_;
    }

private:
    std::vector<uint8_t> data_{};
    size_t bit_pos_ = 0;
};

struct HandleGuard
{
    rocprof_trace_decoder_handle_t value{};
    ~HandleGuard()
    {
        if (value.handle != 0) rocprof_trace_decoder_destroy_handle(value);
    }
};

rocprofiler_thread_trace_decoder_status_t
collect_records(rocprofiler_thread_trace_decoder_record_type_t, void*, uint64_t, void*)
{
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

rocprofiler_thread_trace_decoder_status_t
nop_isa(char*, uint64_t* memory_size, uint64_t* size, rocprofiler_thread_trace_decoder_pc_t, void*)
{
    *memory_size = 4;
    *size = 0;
    return ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS;
}

std::vector<uint8_t> make_mi400_logging_trace()
{
    BitStreamBuilder builder;

    mi400::header_type header{};
    header.header = 0b0010001;
    header.version = 5;
    header.DWGP = 0;
    header.DSIMD = 0;
    header.DSA = 0;
    builder.writeBits(header.raw, 64);

    mi400::misc_type misc{};
    misc.header = 0b1010001;
    misc.tm = 1;
    misc.CLF = 1;
    misc.CLID = 6;
    builder.writeBits(misc.raw, 24);

    mi400::immed_one_type immed{};
    immed.header = 0b1101;
    immed.wid = 3;
    builder.writeBits(immed.raw, 16);

    return builder.finish(16);
}
} // namespace

TEST(RdnaLoggingTest, SqttLoggingEnablesMi400DiagnosticPrints)
{
    const char* logging = std::getenv("SQTT_LOGGING");
    if (!logging || *logging == '0') GTEST_SKIP() << "CTest must set SQTT_LOGGING=1 for this executable";

    const auto trace = make_mi400_logging_trace();

    HandleGuard handle;
    ASSERT_EQ(rocprof_trace_decoder_create_handle(&handle.value), ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    ASSERT_EQ(
        rocprof_trace_decoder_set_isa_callback(handle.value, nop_isa, nullptr),
        ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS
    );

    testing::internal::CaptureStdout();
    const auto status = rocprof_trace_decoder_parse(handle.value, trace.data(), trace.size(), collect_records, nullptr);
    const std::string output = testing::internal::GetCapturedStdout();

    ASSERT_EQ(status, ROCPROFILER_THREAD_TRACE_DECODER_STATUS_SUCCESS);
    EXPECT_NE(output.find("HEADER - TT Version:5"), std::string::npos);
    EXPECT_NE(output.find("MISC - raw:0x"), std::string::npos);
    EXPECT_NE(output.find("IMMEDONE - wid:3"), std::string::npos);
}
