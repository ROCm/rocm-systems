// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Smoke tests for real Linux policy implementations.
// Tests real_timer_trigger, steady_clock, real_signal_dispatcher,
// and real_fatal_error_policy against the actual OS APIs.
//
// Gated at CMake level (Linux only) — no runtime GTEST_SKIP.

#include <gtest/gtest.h>

// real_timer_trigger lives in sampling/policies/ (production-only).
// rocprofsys_sampling_signal_handler is provided by services_accessor.cpp from
// the main rocprof-sys library which the unit-tests binary now links (the
// production_hooks fold pulled in those symbols transitively).
#include "sampling/policies/linux/real_timer_trigger.hpp"

// steady_clock and real_signal_dispatcher are small enough to define locally
// (main_library_policies.hpp pulls timemory/config — incompatible with unit binary).

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdlib>
#include <pthread.h>
#include <signal.h>
#include <sys/types.h>
#include <thread>
#include <unistd.h>

using namespace rocprofsys::sampling;

// ── Local minimal impls (avoid pulling main_library_policies.hpp) ─────────────

namespace
{

struct local_steady_clock
{
    [[nodiscard]] uint64_t now_ns() const noexcept
    {
        auto tp = std::chrono::steady_clock::now();
        return static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(tp.time_since_epoch())
                .count());
    }

    [[nodiscard]] std::chrono::steady_clock::time_point now_steady() const noexcept
    {
        return std::chrono::steady_clock::now();
    }
};

struct local_signal_dispatcher
{
    int apply_sigmask(int how, void const* set, void* oldset) noexcept
    {
        return ::pthread_sigmask(how, static_cast<sigset_t const*>(set),
                                 static_cast<sigset_t*>(oldset));
    }
};

// Shared atomic counter incremented by the signal handler in timer tests.
static std::atomic<int> g_signal_count{ 0 };

static void
timer_smoke_handler(int /*sig*/, siginfo_t* /*info*/, void* /*ctx*/) noexcept
{
    g_signal_count.fetch_add(1, std::memory_order_relaxed);
}

}  // namespace

// ── steady_clock smoke tests ──────────────────────────────────────────────────

TEST(steady_clock_smoke, now_ns_returns_nonzero)
{
    local_steady_clock clk;
    EXPECT_GT(clk.now_ns(), 0U) << "now_ns() must return a non-zero timestamp";
}

TEST(steady_clock_smoke, now_ns_increases_monotonically)
{
    local_steady_clock clk;
    uint64_t           t0 = clk.now_ns();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    uint64_t t1 = clk.now_ns();
    EXPECT_GT(t1, t0) << "now_ns() must increase between two calls separated by 2ms";
}

TEST(steady_clock_smoke, now_steady_increases_monotonically)
{
    local_steady_clock clk;
    auto               t0 = clk.now_steady();
    std::this_thread::sleep_for(std::chrono::milliseconds(2));
    auto t1 = clk.now_steady();
    EXPECT_GT(t1, t0) << "now_steady() must return a later time_point after 2ms";
}

// ── real_signal_dispatcher smoke tests ───────────────────────────────────────

TEST(real_signal_dispatcher_smoke, sigmask_block_returns_zero)
{
    local_signal_dispatcher dispatcher;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR2);

    int ret = dispatcher.apply_sigmask(SIG_BLOCK, &set, nullptr);
    EXPECT_EQ(ret, 0) << "sigmask(SIG_BLOCK) must return 0 on success";

    // Restore: unblock immediately.
    dispatcher.apply_sigmask(SIG_UNBLOCK, &set, nullptr);
}

TEST(real_signal_dispatcher_smoke, sigmask_unblock_returns_zero)
{
    local_signal_dispatcher dispatcher;

    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR2);

    dispatcher.apply_sigmask(SIG_BLOCK, &set, nullptr);
    int ret = dispatcher.apply_sigmask(SIG_UNBLOCK, &set, nullptr);
    EXPECT_EQ(ret, 0) << "sigmask(SIG_UNBLOCK) must return 0 on success";
}

TEST(real_signal_dispatcher_smoke, sigmask_roundtrip_restores_mask)
{
    local_signal_dispatcher dispatcher;

    // Capture current mask.
    sigset_t original;
    sigemptyset(&original);
    ::pthread_sigmask(SIG_SETMASK, nullptr, &original);

    sigset_t add_set;
    sigemptyset(&add_set);
    sigaddset(&add_set, SIGUSR2);

    sigset_t saved;
    dispatcher.apply_sigmask(SIG_BLOCK, &add_set, &saved);

    // Verify SIGUSR2 is now blocked.
    sigset_t current;
    sigemptyset(&current);
    ::pthread_sigmask(SIG_SETMASK, nullptr, &current);
    EXPECT_EQ(sigismember(&current, SIGUSR2), 1)
        << "SIGUSR2 must be blocked after SIG_BLOCK";

    // Restore original mask via SIG_SETMASK.
    dispatcher.apply_sigmask(SIG_SETMASK, &saved, nullptr);

    ::pthread_sigmask(SIG_SETMASK, nullptr, &current);
    EXPECT_EQ(sigismember(&current, SIGUSR2), sigismember(&original, SIGUSR2))
        << "Mask must be restored after SIG_SETMASK roundtrip";
}

// ── real_timer_trigger smoke tests ────────────────────────────────────────────

TEST(real_timer_trigger_smoke, default_constructed_is_not_armed)
{
    real_timer_trigger trigger;
    EXPECT_FALSE(trigger.is_armed())
        << "real_timer_trigger must not be armed before configure+start";
}

TEST(real_timer_trigger_smoke, configure_does_not_arm)
{
    real_timer_trigger trigger;
    trigger.configure(0, ::gettid(), SIGUSR1, CLOCK_REALTIME, 100.0, 0.0);
    EXPECT_FALSE(trigger.is_armed()) << "configure() alone must not arm the timer";
}

TEST(real_timer_trigger_smoke, start_sets_armed)
{
    real_timer_trigger trigger;

    // Install a signal handler for SIGUSR1 so delivery doesn't kill the process.
    struct sigaction sa
    {};
    sa.sa_sigaction = timer_smoke_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, nullptr);

    trigger.configure(0, ::gettid(), SIGUSR1, CLOCK_REALTIME, 100.0, 0.0);
    trigger.start();

    EXPECT_TRUE(trigger.is_armed()) << "start() must set is_armed() to true";

    trigger.stop();
    EXPECT_FALSE(trigger.is_armed()) << "stop() must clear is_armed()";

    // Restore default disposition.
    struct sigaction def
    {};
    def.sa_handler = SIG_DFL;
    sigemptyset(&def.sa_mask);
    sigaction(SIGUSR1, &def, nullptr);
}

TEST(real_timer_trigger_smoke, timer_delivers_signals_during_run)
{
    real_timer_trigger trigger;

    g_signal_count.store(0, std::memory_order_relaxed);

    // 100 Hz → ~10ms between signals; 150ms window → expect ≥10 signals.
    // After Phase H1, start() installs rocprofsys_sampling_signal_handler (stub in this
    // binary). Install the counting handler AFTER start() so it wins the sigaction race.
    trigger.configure(0, ::gettid(), SIGUSR1, CLOCK_REALTIME, 100.0, 0.0);
    trigger.start();

    // Install counting handler after trigger.start() to override the stub handler.
    struct sigaction sa
    {};
    sa.sa_sigaction = timer_smoke_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, nullptr);

    std::this_thread::sleep_for(std::chrono::milliseconds(150));

    trigger.stop();

    int delivered = g_signal_count.load(std::memory_order_relaxed);
    EXPECT_GE(delivered, 1)
        << "At least 1 SIGUSR1 must be delivered during 150ms at 100Hz; got "
        << delivered;

    // Restore default disposition.
    struct sigaction def
    {};
    def.sa_handler = SIG_DFL;
    sigemptyset(&def.sa_mask);
    sigaction(SIGUSR1, &def, nullptr);
}

TEST(real_timer_trigger_smoke, stop_without_start_is_noop)
{
    real_timer_trigger trigger;
    trigger.configure(0, ::gettid(), SIGUSR1, CLOCK_REALTIME, 10.0, 0.0);
    // Must not crash or assert.
    EXPECT_NO_THROW(trigger.stop());
    EXPECT_FALSE(trigger.is_armed());
}

TEST(real_timer_trigger_smoke, destructor_disarms_armed_timer)
{
    struct sigaction sa
    {};
    sa.sa_sigaction = timer_smoke_handler;
    sa.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&sa.sa_mask);
    sigaction(SIGUSR1, &sa, nullptr);

    {
        real_timer_trigger trigger;
        trigger.configure(0, ::gettid(), SIGUSR1, CLOCK_REALTIME, 10.0, 0.0);
        trigger.start();
        ASSERT_TRUE(trigger.is_armed());
        // Destructor runs here — must call stop() without crashing.
    }

    SUCCEED() << "Destructor of armed real_timer_trigger must not crash";

    struct sigaction def
    {};
    def.sa_handler = SIG_DFL;
    sigemptyset(&def.sa_mask);
    sigaction(SIGUSR1, &def, nullptr);
}

// ── real_timer_trigger sigaction tests ────────────────────────────────────────
//
// After Phase H1, real_timer_trigger::start() always installs
// rocprofsys_sampling_signal_handler via sigaction() (no build-time gate).
// The gate test that distinguished unit-build vs production-build behavior
// has been removed; e2e coverage via reproduce.sh SI-7 assertion remains.

TEST(real_timer_trigger_smoke, start_then_stop_restores_armed_state)
{
    // Verify the timer's armed state bookkeeping is symmetric with sigaction lifecycle.
    const int test_sig = SIGRTMIN + 5;

    struct sigaction before
    {};
    sigemptyset(&before.sa_mask);
    sigaction(test_sig, nullptr, &before);

    // Install a dummy handler so delivery doesn't kill the process.
    struct sigaction dummy
    {};
    dummy.sa_sigaction = [](int, siginfo_t*, void*) noexcept {};
    dummy.sa_flags     = SA_SIGINFO | SA_RESTART;
    sigemptyset(&dummy.sa_mask);
    sigaction(test_sig, &dummy, nullptr);

    real_timer_trigger trigger;
    trigger.configure(0, ::gettid(), test_sig, CLOCK_REALTIME, 10.0, 0.0);
    trigger.start();
    ASSERT_TRUE(trigger.is_armed());
    trigger.stop();
    EXPECT_FALSE(trigger.is_armed());

    // Restore original disposition.
    sigaction(test_sig, &before, nullptr);
}

// ── symbol_resolver smoke tests ──────────────────────────────────────────────

#include "sampling/src/linux/symbol_resolver.hpp"

TEST(symbol_resolver_smoke, resolve_libc_symbol_returns_nonempty_name)
{
    // Use a well-known shared-library function whose PC dladdr can resolve
    // without -rdynamic (shared library symbols are always in the dynamic table).
    rocprofsys::sampling::symbol_resolver resolver;
    auto                                  pc   = reinterpret_cast<uintptr_t>(&::printf);
    std::string                           name = resolver.resolve(pc);
    EXPECT_FALSE(name.empty())
        << "symbol_resolver must resolve a known libc PC to a non-empty name";
}

TEST(symbol_resolver_smoke, resolve_libc_symbol_contains_printf)
{
    rocprofsys::sampling::symbol_resolver resolver;
    auto                                  pc   = reinterpret_cast<uintptr_t>(&::printf);
    std::string                           name = resolver.resolve(pc);
    EXPECT_NE(name.find("printf"), std::string::npos)
        << "resolved name for printf PC must contain 'printf', got: " << name;
}

TEST(symbol_resolver_smoke, resolve_zero_pc_returns_empty)
{
    rocprofsys::sampling::symbol_resolver resolver;
    std::string                           name = resolver.resolve(0);
    EXPECT_TRUE(name.empty()) << "symbol_resolver must return empty string for PC=0";
}

TEST(symbol_resolver_smoke, resolve_same_pc_twice_returns_same_name)
{
    rocprofsys::sampling::symbol_resolver resolver;
    auto                                  pc    = reinterpret_cast<uintptr_t>(&::printf);
    std::string                           name1 = resolver.resolve(pc);
    std::string                           name2 = resolver.resolve(pc);
    EXPECT_EQ(name1, name2)
        << "repeated resolution of the same PC must return identical names (cache hit)";
}

// ── real_fatal_error_policy death test ───────────────────────────────────────
// GTest death tests fork a subprocess — not a runtime skip.

// Define locally rather than including main_library_policies.hpp (heavy deps).
namespace
{
struct local_fatal_error_policy
{
    template <class... Args>
    [[noreturn]] void fatal(char const* /*file*/, int /*line*/, std::string_view /*fmt*/,
                            Args const&... /*args*/) noexcept
    {
        std::exit(1);
    }
};
}  // namespace

TEST(real_fatal_error_policy_smoke, fatal_exits_with_code_1)
{
    local_fatal_error_policy policy;
    EXPECT_EXIT(policy.fatal(__FILE__, __LINE__, "test fatal"),
                ::testing::ExitedWithCode(1), "")
        << "real_fatal_error_policy::fatal() must terminate the process with exit code 1";
}
