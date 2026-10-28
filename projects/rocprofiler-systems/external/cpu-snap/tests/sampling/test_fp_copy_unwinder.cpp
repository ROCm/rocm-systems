// Unit tests for fp_copy_unwinder on hand-crafted synthetic stack buffers,
// one per stopping condition: null return address, out-of-bounds rbp, full
// output buffer.

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "cpu_snap/sampling/unwind/fp_copy_unwinder.hpp"
#include "cpu_snap/sampling/unwind/perf_regs.hpp"
#include "cpu_snap/core/frame_limits.hpp"

namespace
{

using cpu_snap::sampling::unwind::capture_kind;
using cpu_snap::sampling::unwind::fp_copy_unwinder;
using cpu_snap::sampling::unwind::machine_snapshot;

using cpu_snap::sampling::unwind::k_perf_slot_rbp;
using cpu_snap::sampling::unwind::k_perf_slot_rip;
void
write_u64(std::vector<std::byte>& buf, std::size_t offset, std::uint64_t value)
{
    std::memcpy(buf.data() + offset, &value, sizeof(value));
}

}  // namespace

TEST(FpCopyUnwinderTest, RequiredCaptureIsRegsAndStack)
{
    EXPECT_EQ(fp_copy_unwinder::required_capture(), capture_kind::regs_and_stack);
}

TEST(FpCopyUnwinderTest, EmptyOutputBufferReturnsZero)
{
    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = 0x1000;

    const machine_snapshot snap{ .ready_frames = {}, .regs = regs.data(), .stack_addr = 0x2000, .stack = {} };

    fp_copy_unwinder unw;
    EXPECT_EQ(unw.unwind(snap, std::span<std::uint64_t>{}), 0U);
}

TEST(FpCopyUnwinderTest, NullRegsReturnsZero)
{
    std::array<std::uint64_t, 4> out{};
    const machine_snapshot       snap{ .ready_frames = {}, .regs = nullptr, .stack_addr = 0x2000, .stack = {} };

    fp_copy_unwinder unw;
    EXPECT_EQ(unw.unwind(snap, std::span<std::uint64_t>{ out }), 0U);
}

// 3 caller frames chained via rbp, terminated by a null return address.
// unwind() must recover exactly [leaf_ip, 0x3001, 0x3002, 0x3003] and stop -
// the null-terminated frame's return address is never appended.
TEST(FpCopyUnwinderTest, NullReturnAddressEndsChain)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;

    std::vector<std::byte> stack(64, std::byte{ 0 });
    // Frame at offset 0 (initial rbp): saved_rbp -> offset 16, return -> 0x3001.
    write_u64(stack, 0, stack_addr + 16);
    write_u64(stack, 8, 0x3001);
    // Frame at offset 16: saved_rbp -> offset 32, return -> 0x3002.
    write_u64(stack, 16, stack_addr + 32);
    write_u64(stack, 24, 0x3002);
    // Frame at offset 32: saved_rbp -> offset 48, return -> 0x3003.
    write_u64(stack, 32, stack_addr + 48);
    write_u64(stack, 40, 0x3003);
    // Frame at offset 48: saved_rbp -> offset 64 (in bounds, unused), return -> 0 (end of chain).
    write_u64(stack, 48, stack_addr + 64);
    write_u64(stack, 56, 0);

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::array<std::uint64_t, 8> out{};
    fp_copy_unwinder              unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 4U);
    EXPECT_EQ(out[0], leaf_ip);
    EXPECT_EQ(out[1], 0x3000U);
    EXPECT_EQ(out[2], 0x3001U);
    EXPECT_EQ(out[3], 0x3002U);
}

// The first caller frame's saved rbp points OUTSIDE the copied stack window
// (past its end). The walk must recover the one in-bounds caller frame, then
// stop safely without reading past the buffer.
TEST(FpCopyUnwinderTest, OutOfBoundsRbpStopsWalkSafely)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;

    std::vector<std::byte> stack(16, std::byte{ 0 });  // only one frame's worth of bytes
    // Frame at offset 0: saved_rbp points far outside the 16-byte window; return -> 0x4001.
    write_u64(stack, 0, stack_addr + 0x10000);
    write_u64(stack, 8, 0x4001);

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::array<std::uint64_t, 8> out{};
    fp_copy_unwinder              unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 2U);
    EXPECT_EQ(out[0], leaf_ip);
    EXPECT_EQ(out[1], 0x4000U);
}

// A corrupt rbp pointing BELOW stack_addr must also stop the walk (not
// underflow the offset subtraction).
TEST(FpCopyUnwinderTest, RbpBelowStackAddrStopsWalkSafely)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;

    std::vector<std::byte> stack(16, std::byte{ 0 });
    write_u64(stack, 0, stack_addr - 8);  // saved_rbp below stack_addr: corrupt
    write_u64(stack, 8, 0x4001);

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::array<std::uint64_t, 8> out{};
    fp_copy_unwinder              unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 2U);
    EXPECT_EQ(out[0], leaf_ip);
    EXPECT_EQ(out[1], 0x4000U);
}

// A long, well-formed chain must be cut at the output capacity, which is not
// limited to any compile-time depth.
TEST(FpCopyUnwinderTest, CapsWalkAtOutputCapacity)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;
    constexpr std::uint64_t return_ip  = 0x5000;
    constexpr std::size_t   k_capacity = 300;
    constexpr std::size_t   k_frames   = k_capacity + 1;

    std::vector<std::byte> stack((k_frames + 1) * 16, std::byte{ 0 });
    for(std::size_t i = 0; i < k_frames; ++i)
    {
        write_u64(stack, i * 16, stack_addr + (i + 1) * 16);
        write_u64(stack, i * 16 + 8, return_ip);
    }

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::vector<std::uint64_t> out(k_capacity, 0);
    fp_copy_unwinder            unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, k_capacity);
    EXPECT_EQ(out[0], leaf_ip);
    for(std::size_t i = 1; i < n; ++i)
    {
        EXPECT_EQ(out[i], return_ip - 1);
    }
}

TEST(FpCopyUnwinderTest, SelfLinkedFrameStopsAfterOneFrame)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;
    constexpr std::uint64_t return_ip  = 0x5000;

    std::vector<std::byte> stack(16, std::byte{ 0 });
    write_u64(stack, 0, stack_addr);
    write_u64(stack, 8, return_ip);

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::array<std::uint64_t, 8> out{};
    fp_copy_unwinder              unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 2U);
    EXPECT_EQ(out[0], leaf_ip);
    EXPECT_EQ(out[1], return_ip - 1);
}

TEST(FpCopyUnwinderTest, BackwardsLinkedFrameStopsAfterOneFrame)
{
    constexpr std::uint64_t stack_addr = 0x2000;
    constexpr std::uint64_t leaf_ip    = 0x1000;
    constexpr std::uint64_t return_ip  = 0x5000;

    std::vector<std::byte> stack(64, std::byte{ 0 });
    write_u64(stack, 32, stack_addr + 16);
    write_u64(stack, 40, return_ip);
    write_u64(stack, 16, stack_addr + 32);
    write_u64(stack, 24, 0x6000);

    std::array<std::uint64_t, 9> regs{};
    regs[k_perf_slot_rip] = leaf_ip;
    regs[k_perf_slot_rbp] = stack_addr + 32;

    const machine_snapshot snap{
        .ready_frames = {},
        .regs       = regs.data(),
        .stack_addr = stack_addr,
        .stack      = stack,
    };

    std::array<std::uint64_t, 8> out{};
    fp_copy_unwinder              unw;
    const std::size_t n = unw.unwind(snap, std::span<std::uint64_t>{ out });

    ASSERT_EQ(n, 2U);
    EXPECT_EQ(out[0], leaf_ip);
    EXPECT_EQ(out[1], return_ip - 1);
}
