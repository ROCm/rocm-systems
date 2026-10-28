#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "cpu_snap/sampling/triggers/perf/perf_sample_decoder.hpp"

namespace
{

using cpu_snap::sampling::triggers::perf::perf_sample_decoder;
using cpu_snap::sampling::unwind::machine_snapshot;

struct copy_unwinder
{
    static constexpr cpu_snap::sampling::unwind::capture_kind required_capture() noexcept
    {
        return cpu_snap::sampling::unwind::capture_kind::kernel_callchain;
    }

    std::size_t unwind(const machine_snapshot& snap, std::span<std::uint64_t> out) const noexcept
    {
        const std::size_t n = std::min(snap.ready_frames.size(), out.size());
        std::copy_n(snap.ready_frames.begin(), n, out.begin());
        return n;
    }
};

template <typename T>
void
put(std::vector<std::uint8_t>& buf, T value)
{
    const auto* p = reinterpret_cast<const std::uint8_t*>(&value);
    buf.insert(buf.end(), p, p + sizeof(T));
}

std::vector<std::uint8_t>
prefix(std::uint32_t tid, std::uint64_t time)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 0xdead);
    put<std::uint32_t>(buf, 1);
    put<std::uint32_t>(buf, tid);
    put<std::uint64_t>(buf, time);
    return buf;
}

}  // namespace

TEST(PerfSampleDecoder, PrefixYieldsTidAndTimeAndAdvancesCursor)
{
    const auto          buf = prefix(77, 123456);
    const std::uint8_t* cur = buf.data();

    const auto pfx = perf_sample_decoder::decode_sample_prefix(cur);

    EXPECT_EQ(pfx.tid, 77U);
    EXPECT_EQ(pfx.time, 123456U);
    EXPECT_EQ(cur, buf.data() + perf_sample_decoder::k_sample_prefix_bytes);
}

TEST(PerfSampleDecoder, CallchainKeepsUserIpsAndDropsKernelSentinels)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 4);
    put<std::uint64_t>(buf, 0x1000);
    put<std::uint64_t>(buf, ~std::uint64_t{ 127 });
    put<std::uint64_t>(buf, 0x2000);
    put<std::uint64_t>(buf, perf_sample_decoder::k_perf_user_ip_ceiling);

    perf_sample_decoder::decode_scratch scratch{ 8 };
    copy_unwinder                       unw;
    const auto n = perf_sample_decoder::decode_frame_pointer_frames(
        buf.data(), buf.data() + buf.size(), scratch, unw);

    ASSERT_EQ(n, 2U);
    EXPECT_EQ(scratch.frames[0], 0x1000U);
    EXPECT_EQ(scratch.frames[1], 0x2000U);
}

TEST(PerfSampleDecoder, CallchainCountBeyondRecordIsClamped)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 1000);
    put<std::uint64_t>(buf, 0x1000);
    put<std::uint64_t>(buf, 0x2000);

    perf_sample_decoder::decode_scratch scratch{ 8 };
    copy_unwinder                       unw;
    EXPECT_EQ(perf_sample_decoder::decode_frame_pointer_frames(
                  buf.data(), buf.data() + buf.size(), scratch, unw),
              2U);
}

TEST(PerfSampleDecoder, CallchainIsCappedAtScratchDepth)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 5);
    for(std::uint64_t i = 1; i <= 5; ++i)
    {
        put<std::uint64_t>(buf, 0x1000 * i);
    }

    perf_sample_decoder::decode_scratch scratch{ 3 };
    copy_unwinder                       unw;
    EXPECT_EQ(perf_sample_decoder::decode_frame_pointer_frames(
                  buf.data(), buf.data() + buf.size(), scratch, unw),
              3U);
}

TEST(PerfSampleDecoder, CallchainWithoutCountFieldYieldsNothing)
{
    const std::vector<std::uint8_t> buf(4, 0);

    perf_sample_decoder::decode_scratch scratch{ 8 };
    copy_unwinder                       unw;
    EXPECT_EQ(perf_sample_decoder::decode_frame_pointer_frames(
                  buf.data(), buf.data() + buf.size(), scratch, unw),
              0U);
}

TEST(PerfSampleDecoder, RegsAndStackWithTruncatedRegsYieldsNothing)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 2);
    put<std::uint64_t>(buf, 0);

    perf_sample_decoder::decode_scratch scratch{ 8 };
    copy_unwinder                       unw;
    EXPECT_EQ(perf_sample_decoder::decode_regs_and_stack_frames(
                  buf.data(), buf.data() + buf.size(), scratch, unw),
              0U);
}

TEST(PerfSampleDecoder, RegsAndStackWithEmptyStackYieldsNothing)
{
    std::vector<std::uint8_t> buf;
    put<std::uint64_t>(buf, 2);
    for(std::size_t i = 0; i < cpu_snap::sampling::unwind::k_perf_dwarf_n_regs_x86_64; ++i)
    {
        put<std::uint64_t>(buf, 0);
    }
    put<std::uint64_t>(buf, 0);

    perf_sample_decoder::decode_scratch scratch{ 8 };
    copy_unwinder                       unw;
    EXPECT_EQ(perf_sample_decoder::decode_regs_and_stack_frames(
                  buf.data(), buf.data() + buf.size(), scratch, unw),
              0U);
}
