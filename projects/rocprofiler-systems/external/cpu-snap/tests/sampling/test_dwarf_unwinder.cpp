// Unit tests for cpu_snap::sampling::unwind::dwarf_unwinder (libdwfl-backed).
//
// Two scenarios:
//   1. smoke - construct + unwind on an all-zero snapshot, expect 0
//      frames and no crash.
//   2. real_walk - capture live register + stack state at a known
//      leaf via getcontext(), copy a slab of the current stack, feed
//      the snapshot to the unwinder, expect >= 2 frames back and the
//      innermost frame within +/- 256 bytes of the capture site.

#include <bits/types/struct_iovec.h>
#include <gtest/gtest.h>
#include <sys/types.h>
#include <ios>

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif
#include <sys/ucontext.h>
#include <sys/uio.h>
#include <ucontext.h>
#include <dlfcn.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <string_view>
#include <vector>

#include "cpu_snap/sampling/unwind/dwarf_unwinder.hpp"

namespace
{

constexpr std::uint32_t k_perf_n_regs = cpu_snap::sampling::unwind::k_perf_dwarf_n_regs_x86_64;

// Pack a glibc ucontext_t into the perf_event REGS_USER slot ordering
// the snapshot consumer expects.
void
pack_perf_regs(const ucontext_t& uc, std::array<std::uint64_t, k_perf_n_regs>& out)
{
    const auto& g = uc.uc_mcontext.gregs;
    out[0]        = static_cast<std::uint64_t>(g[REG_RAX]);
    out[1]        = static_cast<std::uint64_t>(g[REG_RBX]);
    out[2]        = static_cast<std::uint64_t>(g[REG_RCX]);
    out[3]        = static_cast<std::uint64_t>(g[REG_RDX]);
    out[4]        = static_cast<std::uint64_t>(g[REG_RSI]);
    out[5]        = static_cast<std::uint64_t>(g[REG_RDI]);
    out[6]        = static_cast<std::uint64_t>(g[REG_RBP]);
    out[7]        = static_cast<std::uint64_t>(g[REG_RSP]);
    out[8]        = static_cast<std::uint64_t>(g[REG_RIP]);
    out[9]        = static_cast<std::uint64_t>(g[REG_EFL]);
    out[10]       = static_cast<std::uint64_t>(g[REG_R8]);
    out[11]       = static_cast<std::uint64_t>(g[REG_R9]);
    out[12]       = static_cast<std::uint64_t>(g[REG_R10]);
    out[13]       = static_cast<std::uint64_t>(g[REG_R11]);
    out[14]       = static_cast<std::uint64_t>(g[REG_R12]);
    out[15]       = static_cast<std::uint64_t>(g[REG_R13]);
    out[16]       = static_cast<std::uint64_t>(g[REG_R14]);
    out[17]       = static_cast<std::uint64_t>(g[REG_R15]);
}

// A 3-deep call chain. The leaf grabs its own register state and a
// stack slab and stores them in *out. Compiled with -fomit-frame-pointer
// (see CMakeLists.txt) so DWARF .eh_frame is the only thing libdwfl
// has to work with.
struct capture
{
    std::array<std::uint64_t, k_perf_n_regs> regs{};
    std::vector<std::uint8_t>                stack;
    std::uint64_t                            stack_addr = 0;
    std::uint64_t                            leaf_rip   = 0;
};

[[gnu::noinline]] void
capture_leaf(capture* out) noexcept
{
    ucontext_t uc;
    if(::getcontext(&uc) != 0) return;

    pack_perf_regs(uc, out->regs);
    out->leaf_rip = out->regs[8];  // RIP slot

    // Copy a slab of the current stack starting from RSP upward
    // (toward higher addresses, i.e. the caller frames).
    // process_vm_readv returns a short read on EFAULT rather than
    // SEGV'ing if the slab extends past the stack mapping; we resize
    // the buffer to the actual bytes obtained.
    constexpr std::uint64_t k_slab_bytes = 8 * 1024;
    const std::uint64_t     rsp          = out->regs[7];
    out->stack.resize(k_slab_bytes);
    struct iovec const local
    {
        out->stack.data(), k_slab_bytes
    };
    struct iovec const remote
    {
        reinterpret_cast<void*>(rsp), k_slab_bytes
    };
    const ssize_t got = ::process_vm_readv(::getpid(), &local, 1, &remote, 1, 0);
    if(got <= 0)
    {
        out->stack.clear();
        return;
    }
    out->stack.resize(static_cast<std::size_t>(got));
    out->stack_addr = rsp;
}

[[gnu::noinline]] void
capture_mid(capture* out) noexcept
{
    capture_leaf(out);
    // Defeat tail-call optimization so this frame stays on the stack.
    asm volatile("" ::: "memory");
}

[[gnu::noinline]] void
capture_outer(capture* out) noexcept
{
    capture_mid(out);
    asm volatile("" ::: "memory");
}

}  // namespace

// Hand-written frame-pointer functions with the CFI shape clang emits for an
// epilogue: after `pop %rbp` the CFA switches back to rsp+8, but the
// "rbp saved at CFA-16" rule is never cancelled. A sample that lands on the
// `ret` therefore sees a stale rule for a slot that is already popped.
//
//   cs_test_ret_chain    push rbp; mov rsp,rbp; call cs_test_ret_capture; pop rbp; ret
//   cs_test_ret_capture  records rsp/rbp as they are on entry (RA on top of the
//                        stack), pretends rip is cs_test_ret_leaf_ret, and
//                        copies the stack from rsp upward.
//   cs_test_ret_leaf     only supplies the `ret` (and its CFI) rip points at.
asm(R"(
    .pushsection .text
    .p2align 4
    .globl  cs_test_ret_leaf
    .type   cs_test_ret_leaf, @function
cs_test_ret_leaf:
    .cfi_startproc
    push    %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov     %rsp, %rbp
    .cfi_def_cfa_register %rbp
    pop     %rbp
    .cfi_def_cfa %rsp, 8
    .globl  cs_test_ret_leaf_ret
cs_test_ret_leaf_ret:
    ret
    .cfi_endproc
    .size   cs_test_ret_leaf, .-cs_test_ret_leaf

    .p2align 4
    .globl  cs_test_ret_capture
    .type   cs_test_ret_capture, @function
cs_test_ret_capture:
    mov     %rsp, %rax
    mov     %rax, 56(%rdi)
    mov     %rbp, 48(%rdi)
    lea     cs_test_ret_leaf_ret(%rip), %rcx
    mov     %rcx, 64(%rdi)
    mov     %rsi, %rdi
    mov     %rax, %rsi
    mov     %rdx, %rcx
    cld
    rep movsb
    ret
    .size   cs_test_ret_capture, .-cs_test_ret_capture

    .p2align 4
    .globl  cs_test_ret_chain
    .type   cs_test_ret_chain, @function
cs_test_ret_chain:
    .cfi_startproc
    push    %rbp
    .cfi_def_cfa_offset 16
    .cfi_offset %rbp, -16
    mov     %rsp, %rbp
    .cfi_def_cfa_register %rbp
    call    cs_test_ret_capture
    .globl  cs_test_ret_chain_site
cs_test_ret_chain_site:
    pop     %rbp
    .cfi_def_cfa %rsp, 8
    ret
    .cfi_endproc
    .size   cs_test_ret_chain, .-cs_test_ret_chain

    # Code without any CFI, like a linker-generated PLT stub: no FDE covers it.
    .p2align 4
    .globl  cs_test_nocfi_stub
    .type   cs_test_nocfi_stub, @function
cs_test_nocfi_stub:
    nop
    jmp     *%r11
    .size   cs_test_nocfi_stub, .-cs_test_nocfi_stub

    .p2align 4
    .globl  cs_test_capture_at
    .type   cs_test_capture_at, @function
cs_test_capture_at:
    mov     %rsp, %rax
    mov     %rax, 56(%rdi)
    movq    $1, 48(%rdi)
    mov     %rcx, 64(%rdi)
    mov     %rsi, %rdi
    mov     %rax, %rsi
    mov     %rdx, %rcx
    cld
    rep movsb
    ret
    .size   cs_test_capture_at, .-cs_test_capture_at

    .p2align 4
    .globl  cs_test_chain_at
    .type   cs_test_chain_at, @function
cs_test_chain_at:
    .cfi_startproc
    sub     $8, %rsp
    .cfi_def_cfa_offset 16
    call    cs_test_capture_at
    .globl  cs_test_chain_at_site
cs_test_chain_at_site:
    add     $8, %rsp
    .cfi_def_cfa_offset 8
    ret
    .cfi_endproc
    .size   cs_test_chain_at, .-cs_test_chain_at
    .popsection
)");

extern "C" {
void cs_test_ret_chain(std::uint64_t* regs, std::uint8_t* stack, std::uint64_t nbytes);
void cs_test_chain_at(std::uint64_t* regs, std::uint8_t* stack, std::uint64_t nbytes,
                      const void* rip);
extern const char cs_test_nocfi_stub[];
extern const char cs_test_chain_at_site[];
extern const char cs_test_ret_leaf_ret[];
extern const char cs_test_ret_chain_site[];
extern const char __executable_start[];
extern const char etext[];
}

TEST(DwarfUnwinderTest, SmokeZeroSnapshotDoesNotCrash)
{
    // libdwfl always emits at least the initial PC frame (from
    // set_initial_registers); with RIP=0 it cannot walk further. The
    // contract this test enforces is "no crash, no run-away", not a
    // specific frame count.
    cpu_snap::sampling::unwind::dwarf_unwinder dw;

    std::array<std::uint64_t, k_perf_n_regs> zero_regs{};
    cpu_snap::sampling::unwind::dwarf_snapshot                 snap{};
    snap.regs       = zero_regs.data();
    snap.stack_addr = 0;
    snap.stack_data = nullptr;
    snap.stack_size = 0;

    std::array<std::uint64_t, 32> out{};
    const std::uint32_t           n = dw.unwind(snap, out.data(), out.size());
    EXPECT_LE(n, 1U);
}

TEST(DwarfUnwinderTest, SmokeNullOutputReturnsZero)
{
    cpu_snap::sampling::unwind::dwarf_unwinder                 dw;
    std::array<std::uint64_t, k_perf_n_regs> zero_regs{};
    cpu_snap::sampling::unwind::dwarf_snapshot                 const snap{ zero_regs.data(), 0, nullptr, 0 };
    EXPECT_EQ(dw.unwind(snap, nullptr, 16), 0U);
}

TEST(DwarfUnwinderTest, RealWalkRecoversMultipleFrames)
{
    capture cap;
    capture_outer(&cap);

    ASSERT_FALSE(cap.stack.empty()) << "getcontext failed";

    cpu_snap::sampling::unwind::dwarf_snapshot const snap{ cap.regs.data(), cap.stack_addr, cap.stack.data(),
                                   cap.stack.size() };

    cpu_snap::sampling::unwind::dwarf_unwinder      dw;
    std::array<std::uint64_t, 64> frames{};
    const std::uint32_t           n = dw.unwind(snap, frames.data(), frames.size());

    if(n == 0)
    {
        GTEST_SKIP() << "libdwfl produced 0 frames in this build / env; "
                        ".eh_frame may be missing or stripped";
    }

    EXPECT_GE(n, 2U) << "expected leaf + at least one caller via DWARF";

    // A non-leaf frame is the call site (return address - 1), i.e. the last
    // byte of the direct `call rel32` (E8) that entered the callee.
    if(n >= 2)
    {
        const auto call_opcode = *reinterpret_cast<const std::uint8_t*>(frames[1] - 4);
        EXPECT_EQ(call_opcode, 0xE8U) << "frame 1 " << std::hex << frames[1] << " is not a call site";
    }

    // The innermost frame should land inside (or right after, due to
    // the RIP captured right after the syscall returns) the capture
    // site. Allow generous tolerance (4 KB) since libdwfl may adjust
    // the PC for non-activation frames.
    const std::uint64_t delta = (frames[0] > cap.leaf_rip) ? (frames[0] - cap.leaf_rip)
                                                           : (cap.leaf_rip - frames[0]);
    EXPECT_LE(delta, 4096U) << "innermost frame " << std::hex << frames[0]
                            << " not near capture site " << cap.leaf_rip;
}

TEST(DwarfUnwinderTest, SampleOnRetInstructionKeepsLiveRbpAndWalksToCallers)
{
    std::array<std::uint64_t, k_perf_n_regs> regs{};
    std::vector<std::uint8_t>                stack(4096);
    cs_test_ret_chain(regs.data(), stack.data(), stack.size());

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ regs.data(), regs[7], stack.data(),
                                                              stack.size() };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};
    const std::uint32_t                                 n = dw.unwind(snap, frames.data(), frames.size());

    ASSERT_GE(n, 3U) << "walk stopped after the `ret` frame and its direct caller";
    EXPECT_EQ(frames[0], reinterpret_cast<std::uint64_t>(cs_test_ret_leaf_ret));
    EXPECT_EQ(frames[1], reinterpret_cast<std::uint64_t>(cs_test_ret_chain_site) - 1);

    const auto text_begin = reinterpret_cast<std::uint64_t>(__executable_start);
    const auto text_end   = reinterpret_cast<std::uint64_t>(etext);
    EXPECT_GE(frames[2], text_begin);
    EXPECT_LT(frames[2], text_end) << "third frame is not code of this binary: rbp was restored "
                                      "from the already-popped slot at CFA-16";
}

TEST(DwarfUnwinderTest, SnapshotWordOffsetIsOverflowSafe)
{
    using cpu_snap::sampling::unwind::dwarf_snapshot;
    using cpu_snap::sampling::unwind::snapshot_word_offset;

    const dwarf_snapshot snap{ nullptr, 0x1000, nullptr, 32 };
    EXPECT_EQ(snapshot_word_offset(snap, 0x1000), 0U);
    EXPECT_EQ(snapshot_word_offset(snap, 0x1018), 24U) << "the last whole word";
    EXPECT_FALSE(snapshot_word_offset(snap, 0x1019).has_value()) << "word runs past the end";
    EXPECT_FALSE(snapshot_word_offset(snap, 0x0FF8).has_value()) << "below the base";
    EXPECT_FALSE(snapshot_word_offset(snap, 0xFFFFFFFFFFFFFFF8ULL).has_value())
        << "addr + 8 wraps to 0 and used to pass the bounds check";
    EXPECT_FALSE(snapshot_word_offset(snap, 0xFFFFFFFFFFFFFFFFULL).has_value());

    const dwarf_snapshot too_small{ nullptr, 0x1000, nullptr, 7 };
    EXPECT_FALSE(snapshot_word_offset(too_small, 0x1000).has_value());

    const dwarf_snapshot at_top{ nullptr, 0xFFFFFFFFFFFFFFF0ULL, nullptr, 16 };
    EXPECT_EQ(snapshot_word_offset(at_top, 0xFFFFFFFFFFFFFFF8ULL), 8U);
    EXPECT_FALSE(snapshot_word_offset(at_top, 0xFFFFFFFFFFFFFFFCULL).has_value());
}

// A library loaded after the unwinder was created is unknown to libdwfl; the
// walk must pick it up instead of stopping at its frames.
TEST(DwarfUnwinderTest, FramesOfALibraryLoadedAfterConstructionAreUnwound)
{
    cpu_snap::sampling::unwind::dwarf_unwinder dw;

    void* fixture = ::dlopen(DWARF_REFRESH_FIXTURE_PATH, RTLD_NOW | RTLD_LOCAL);
    ASSERT_NE(fixture, nullptr) << ::dlerror();
    using call_fn = void (*)(void (*)(void*), void*);
    const auto fixture_call = reinterpret_cast<call_fn>(::dlsym(fixture, "fixture_call"));
    ASSERT_NE(fixture_call, nullptr);

    capture cap;
    fixture_call([](void* arg) { capture_outer(static_cast<capture*>(arg)); }, &cap);
    ASSERT_FALSE(cap.stack.empty());

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ cap.regs.data(), cap.stack_addr,
                                                              cap.stack.data(), cap.stack.size() };
    std::array<std::uint64_t, 64> frames{};
    const std::uint32_t           n = dw.unwind(snap, frames.data(), frames.size());

    Dl_info      info{};
    std::int32_t fixture_index = -1;
    for(std::uint32_t i = 0; i < n; ++i)
    {
        if(::dladdr(reinterpret_cast<void*>(frames[i]), &info) != 0 && info.dli_fname != nullptr &&
           std::string_view(info.dli_fname).find("dwarf_refresh_fixture") != std::string_view::npos)
        {
            fixture_index = static_cast<std::int32_t>(i);
            break;
        }
    }
    ASSERT_GE(fixture_index, 0) << "no frame inside the fixture library";
    EXPECT_GT(n, static_cast<std::uint32_t>(fixture_index) + 1)
        << "the walk stopped at the frame of the library loaded after construction";

    ::dlclose(fixture);
}

// Linker-generated stubs (PLT) have no CFI. A sample inside one is a frame that
// has just been entered: the return address is on top of the stack, so the walk
// can continue with the caller instead of ending after one frame. The caller
// uses no frame pointer and rbp holds a non-pointer, as in a library built
// without frame pointers, so libdwfl's own frame-pointer fallback cannot help.
TEST(DwarfUnwinderTest, FrameWithoutCfiContinuesAtTheReturnAddressOnTheStack)
{
    std::array<std::uint64_t, k_perf_n_regs> regs{};
    std::vector<std::uint8_t>                stack(4096);
    cs_test_chain_at(regs.data(), stack.data(), stack.size(), cs_test_nocfi_stub);

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ regs.data(), regs[7], stack.data(),
                                                              stack.size() };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};
    const std::uint32_t                                 n = dw.unwind(snap, frames.data(), frames.size());

    ASSERT_GE(n, 3U) << "the walk ended at the frame without CFI";
    EXPECT_EQ(frames[0], reinterpret_cast<std::uint64_t>(cs_test_nocfi_stub));
    EXPECT_EQ(frames[1], reinterpret_cast<std::uint64_t>(cs_test_chain_at_site) - 1);
}

// Same stub, but rbp happens to point into the stack copy, so libdwfl's own
// frame-pointer fallback "succeeds" with a garbage return address instead of
// failing. The walk must not trust it for code that has no CFI.
TEST(DwarfUnwinderTest, FrameWithoutCfiIgnoresFramePointerFallbackGarbage)
{
    constexpr std::size_t   k_fake_frame_offset = 64;
    constexpr std::uint64_t k_garbage_return    = 0x6d656d5f006362;

    std::array<std::uint64_t, k_perf_n_regs> regs{};
    std::vector<std::uint8_t>                stack(4096);
    cs_test_chain_at(regs.data(), stack.data(), stack.size(), cs_test_nocfi_stub);
    regs[6] = regs[7] + k_fake_frame_offset;
    std::memset(stack.data() + k_fake_frame_offset, 0, sizeof(std::uint64_t));
    std::memcpy(stack.data() + k_fake_frame_offset + sizeof(std::uint64_t), &k_garbage_return,
                sizeof(k_garbage_return));

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ regs.data(), regs[7], stack.data(),
                                                              stack.size() };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};
    const std::uint32_t                                 n = dw.unwind(snap, frames.data(), frames.size());

    ASSERT_GE(n, 3U);
    EXPECT_EQ(frames[1], reinterpret_cast<std::uint64_t>(cs_test_chain_at_site) - 1);
}

// A value on top of the stack that is not a return address must not be followed.
TEST(DwarfUnwinderTest, FrameWithoutCfiIsNotFollowedWhenTopOfStackIsNotAReturnAddress)
{
    std::array<std::uint64_t, k_perf_n_regs> regs{};
    std::vector<std::uint8_t>                stack(4096);
    cs_test_chain_at(regs.data(), stack.data(), stack.size(), cs_test_nocfi_stub);
    const std::uint64_t not_a_return_address = 0x1234;
    std::memcpy(stack.data(), &not_a_return_address, sizeof(not_a_return_address));

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ regs.data(), regs[7], stack.data(),
                                                              stack.size() };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};
    const std::uint32_t                                 n = dw.unwind(snap, frames.data(), frames.size());

    EXPECT_EQ(n, 1U);
}

// Libraries loaded one right after the other (a runtime loading its plugins)
// must each be picked up; the refresh must not be rate limited by time.
TEST(DwarfUnwinderTest, LibrariesLoadedInQuickSuccessionAreAllUnwound)
{
    cpu_snap::sampling::unwind::dwarf_unwinder dw;

    // Keep every library loaded: a freed address could be reused by the next one
    // and would then resolve through the stale module of the first.
    std::vector<void*> loaded;
    const auto unwound_through = [&](const char* path) {
        void* lib = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
        EXPECT_NE(lib, nullptr) << ::dlerror();
        if(lib == nullptr) return false;
        loaded.push_back(lib);
        using call_fn = void (*)(void (*)(void*), void*);
        const auto call = reinterpret_cast<call_fn>(::dlsym(lib, "fixture_call"));
        capture cap;
        call([](void* arg) { capture_outer(static_cast<capture*>(arg)); }, &cap);
        const cpu_snap::sampling::unwind::dwarf_snapshot snap{ cap.regs.data(), cap.stack_addr,
                                                                  cap.stack.data(), cap.stack.size() };
        std::array<std::uint64_t, 64> frames{};
        const std::uint32_t           n = dw.unwind(snap, frames.data(), frames.size());
        bool                          in_lib = false;
        std::uint32_t                 lib_index = 0;
        for(std::uint32_t i = 0; i < n; ++i)
        {
            Dl_info info{};
            if(::dladdr(reinterpret_cast<void*>(frames[i]), &info) != 0 && info.dli_fname != nullptr &&
               std::string_view(info.dli_fname) == path)
            {
                in_lib    = true;
                lib_index = i;
                break;
            }
        }
        return in_lib && n > lib_index + 1;
    };

    EXPECT_TRUE(unwound_through(DWARF_REFRESH_FIXTURE_PATH));
    EXPECT_TRUE(unwound_through(DWARF_REFRESH_FIXTURE_B_PATH));
    for(void* lib : loaded)
    {
        ::dlclose(lib);
    }
}

TEST(DwarfUnwinderTest, LiveStackReadCounterStaysZeroWhenWindowCoversTheWalk)
{
    capture cap;
    capture_outer(&cap);
    ASSERT_FALSE(cap.stack.empty()) << "getcontext failed";

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ cap.regs.data(), cap.stack_addr,
                                                              cap.stack.data(),
                                                              cap.stack.size() };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};
    if(dw.unwind(snap, frames.data(), frames.size()) < 2)
    {
        GTEST_SKIP() << "libdwfl could not walk the captured stack in this env";
    }
    EXPECT_EQ(dw.live_stack_read_samples(), 0U);
}

TEST(DwarfUnwinderTest, LiveStackReadCounterCountsEveryUnwindThatLeavesTheWindow)
{
    capture cap;
    capture_outer(&cap);
    ASSERT_FALSE(cap.stack.empty()) << "getcontext failed";
    constexpr std::size_t k_tiny_window = 16;
    ASSERT_GT(cap.stack.size(), k_tiny_window);

    const cpu_snap::sampling::unwind::dwarf_snapshot snap{ cap.regs.data(), cap.stack_addr,
                                                              cap.stack.data(), k_tiny_window };
    cpu_snap::sampling::unwind::dwarf_unwinder       dw;
    std::array<std::uint64_t, 64>                       frames{};

    constexpr std::uint64_t k_calls = 3;
    for(std::uint64_t i = 0; i < k_calls; ++i)
    {
        (void) dw.unwind(snap, frames.data(), frames.size());
    }
    EXPECT_EQ(dw.live_stack_read_samples(), k_calls);
}

TEST(DwarfUnwinderTest, LiveStackReadCounterCountsMachineSnapshotOverload)
{
    capture cap;
    capture_outer(&cap);
    ASSERT_FALSE(cap.stack.empty()) << "getcontext failed";

    const auto* const                                    bytes = cap.stack.data();
    cpu_snap::sampling::unwind::machine_snapshot      small{};
    small.regs       = cap.regs.data();
    small.stack_addr = cap.stack_addr;
    small.stack      = { reinterpret_cast<const std::byte*>(bytes), 16 };
    cpu_snap::sampling::unwind::machine_snapshot      full = small;
    full.stack = { reinterpret_cast<const std::byte*>(bytes), cap.stack.size() };

    cpu_snap::sampling::unwind::dwarf_unwinder dw;
    std::array<std::uint64_t, 64>                 frames{};
    if(dw.unwind(full, frames) < 2)
    {
        GTEST_SKIP() << "libdwfl could not walk the captured stack in this env";
    }
    EXPECT_EQ(dw.live_stack_read_samples(), 0U);
    (void) dw.unwind(small, frames);
    EXPECT_EQ(dw.live_stack_read_samples(), 1U);
}
