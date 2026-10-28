#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <span>

#include "cpu_snap/sampling/unwind/kernel_callchain_unwinder.hpp"

using cpu_snap::sampling::unwind::kernel_callchain_unwinder;
using cpu_snap::sampling::unwind::machine_snapshot;

TEST(KernelCallchainUnwinderTest, NonLeafFramesBecomeCallSiteAddresses)
{
    const std::array<std::uint64_t, 4> chain{ 0x1000, 0x2000, 0x3000, 0x4000 };
    const machine_snapshot             snap{ .ready_frames = chain, .regs = nullptr, .stack_addr = 0, .stack = {} };

    std::array<std::uint64_t, 8> out{};
    const kernel_callchain_unwinder unw;
    const std::size_t               n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 4U);
    EXPECT_EQ(out[0], 0x1000U) << "the leaf is the interrupted instruction and stays exact";
    EXPECT_EQ(out[1], 0x1fffU);
    EXPECT_EQ(out[2], 0x2fffU);
    EXPECT_EQ(out[3], 0x3fffU);
}

TEST(KernelCallchainUnwinderTest, NullReturnAddressIsLeftAsZero)
{
    const std::array<std::uint64_t, 3> chain{ 0x1000, 0, 0x3000 };
    const machine_snapshot             snap{ .ready_frames = chain, .regs = nullptr, .stack_addr = 0, .stack = {} };

    std::array<std::uint64_t, 8> out{};
    const kernel_callchain_unwinder unw;
    ASSERT_EQ(unw.unwind(snap, std::span<std::uint64_t>{ out }), 3U);
    EXPECT_EQ(out[1], 0U);
    EXPECT_EQ(out[2], 0x2fffU);
}
