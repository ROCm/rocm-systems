// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "rocprof-sys/library/components/sampling_gotcha.hpp"

#include <cerrno>
#include <csignal>
#include <cstddef>
#include <functional>
#include <gtest/gtest.h>
#include <pthread.h>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

// Everything below has internal linkage on purpose: the component tests are compiled
// into a single rocprof-sys-unit-tests binary and the sibling suites declare their own
// test_globals with names such as g_callee_calls, so anything with external linkage
// here collides at link time.
namespace
{
// Real-time signals well clear of the ones glibc reserves for itself, so that
// blocking them in a test cannot perturb anything else in the process.
int
signal_a()
{
    return SIGRTMIN + 4;
}

int
signal_b()
{
    return SIGRTMIN + 5;
}

std::set<int>
to_set(const sigset_t* _mask)
{
    auto _signals = std::set<int>{};
    if(!_mask) return _signals;
    for(int i = 1; i <= SIGRTMAX; ++i)
        if(sigismember(_mask, i) == 1) _signals.emplace(i);
    return _signals;
}

sigset_t
make_mask(const std::set<int>& _signals)
{
    auto _mask = sigset_t{};
    sigemptyset(&_mask);
    for(auto itr : _signals)
        sigaddset(&_mask, itr);
    return _mask;
}

bool
is_blocked(int _signal)
{
    auto _current = sigset_t{};
    sigemptyset(&_current);
    pthread_sigmask(SIG_BLOCK, nullptr, &_current);
    return sigismember(&_current, _signal) == 1;
}

struct MockGotchaData
{
    int verbose = 0;
};

namespace test_globals
{
// signal set the policy reports for the current thread
std::set<int> g_signals;
bool          g_signals_available = true;

// recorded sigmask traffic
struct sigmask_call
{
    std::string   op;
    std::set<int> signals;
};
std::vector<sigmask_call> g_calls;

// what the recording policy hands back as the caller's previous mask
std::set<int> g_previous_mask;
int           g_block_result = 0;

// mocked gotcha state
bool                        g_suppress_warnings = true;
bool                        g_is_running        = false;
int                         g_start_calls       = 0;
int                         g_stop_calls        = 0;
int                         g_disable_calls     = 0;
std::function<void()>       g_initializer;
std::vector<std::string>    g_configured;
std::vector<MockGotchaData> g_data;

// observations made from inside a wrapped call
bool g_blocked_during_call = false;
int  g_callee_calls        = 0;

void
reset()
{
    g_signals           = { signal_a() };
    g_signals_available = true;
    g_calls.clear();
    g_previous_mask     = { signal_b() };
    g_block_result      = 0;
    g_suppress_warnings = true;
    g_is_running        = false;
    g_start_calls       = 0;
    g_stop_calls        = 0;
    g_disable_calls     = 0;
    g_initializer       = nullptr;
    g_configured.clear();
    g_blocked_during_call = false;
    g_callee_calls        = 0;
}
}  // namespace test_globals

// Stands in for the timemory gotcha type: the static half is the binding interface
// used by configure()/shutdown(), the instance half is what the bundle hands out.
struct MockGotcha
{
    template <size_t N, typename Ret, typename... Args>
    static void configure(std::string _name)
    {
        test_globals::g_configured.emplace_back(std::move(_name));
    }

    static size_t capacity() { return test_globals::g_data.size(); }

    static MockGotchaData* at(size_t _idx)
    {
        return (_idx < test_globals::g_data.size()) ? &test_globals::g_data.at(_idx)
                                                    : nullptr;
    }

    static std::function<void()>& get_initializer()
    {
        return test_globals::g_initializer;
    }

    static void disable() { ++test_globals::g_disable_calls; }

    bool get_is_running() const { return test_globals::g_is_running; }

    void start()
    {
        test_globals::g_is_running = true;
        ++test_globals::g_start_calls;
    }

    void stop()
    {
        test_globals::g_is_running = false;
        ++test_globals::g_stop_calls;
    }
};

struct MockGotchaBundle
{
    MockGotcha instance;

    template <typename>
    MockGotcha* get()
    {
        return &instance;
    }
};

// Records the sigmask traffic instead of performing it, so the exact sequence of
// operations and the contents of each mask can be asserted.
struct RecordingPolicy
{
    using gotcha_data_t   = MockGotchaData;
    using gotcha_t        = MockGotcha;
    using gotcha_bundle_t = MockGotchaBundle;

    static bool suppress_binding_warnings() { return test_globals::g_suppress_warnings; }

    static const std::set<int>* get_sampling_signals()
    {
        return (test_globals::g_signals_available) ? &test_globals::g_signals : nullptr;
    }

    static int block_signals(const sigset_t* _blocked, sigset_t* _prev)
    {
        test_globals::g_calls.push_back({ "block", to_set(_blocked) });
        if(_prev) *_prev = make_mask(test_globals::g_previous_mask);
        return test_globals::g_block_result;
    }

    static int restore_signals(const sigset_t* _prev)
    {
        test_globals::g_calls.push_back({ "restore", to_set(_prev) });
        return 0;
    }
};

// Drives the real pthread_sigmask so that the mask the thread actually ends up with
// can be observed.
struct RealSigmaskPolicy
{
    using gotcha_data_t   = MockGotchaData;
    using gotcha_t        = MockGotcha;
    using gotcha_bundle_t = MockGotchaBundle;

    static bool suppress_binding_warnings() { return true; }

    static const std::set<int>* get_sampling_signals()
    {
        return (test_globals::g_signals_available) ? &test_globals::g_signals : nullptr;
    }

    static int block_signals(const sigset_t* _blocked, sigset_t* _prev)
    {
        return pthread_sigmask(SIG_BLOCK, _blocked, _prev);
    }

    static int restore_signals(const sigset_t* _prev)
    {
        return pthread_sigmask(SIG_SETMASK, _prev, nullptr);
    }
};

using recording_block_t = rocprofsys::component::scoped_sampling_block<RecordingPolicy>;
using real_block_t      = rocprofsys::component::scoped_sampling_block<RealSigmaskPolicy>;
using recording_gotcha_t = rocprofsys::component::sampling_gotcha<RecordingPolicy>;
using real_gotcha_t      = rocprofsys::component::sampling_gotcha<RealSigmaskPolicy>;

int
probe_callee(int _value)
{
    ++test_globals::g_callee_calls;
    test_globals::g_blocked_during_call = is_blocked(signal_a());
    return _value * 2;
}

void
void_callee()
{
    ++test_globals::g_callee_calls;
    test_globals::g_blocked_during_call = is_blocked(signal_a());
}

int
throwing_callee(int)
{
    throw std::runtime_error{ "wrapped call failed" };
}
}  // namespace

class sampling_gotcha_test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        test_globals::reset();
        // the mock stands in for the real gotcha's slot array, so it has to be sized
        // from the component rather than a literal that can drift from it
        test_globals::g_data.assign(recording_gotcha_t::gotcha_capacity,
                                    MockGotchaData{});

        // start every test from a known, empty mask
        auto _empty = sigset_t{};
        sigemptyset(&_empty);
        pthread_sigmask(SIG_SETMASK, &_empty, &m_entry_mask);
    }

    void TearDown() override { pthread_sigmask(SIG_SETMASK, &m_entry_mask, nullptr); }

private:
    sigset_t m_entry_mask{};
};

// --- mask neutrality -------------------------------------------------------------

// A caller that deliberately holds the sampling signals blocked must still have them
// blocked once the wrapper returns. Unblocking here would resume sampling on a thread
// that had switched it off.
TEST_F(sampling_gotcha_test, LeavesSignalsBlockedWhenCallerHadThemBlocked)
{
    auto _blocked = make_mask({ signal_a() });
    pthread_sigmask(SIG_BLOCK, &_blocked, nullptr);
    ASSERT_TRUE(is_blocked(signal_a()));

    {
        real_block_t _scope{};
        EXPECT_TRUE(is_blocked(signal_a()));
    }

    EXPECT_TRUE(is_blocked(signal_a()));
}

TEST_F(sampling_gotcha_test, BlocksDuringScopeAndUnblocksWhenCallerHadThemUnblocked)
{
    ASSERT_FALSE(is_blocked(signal_a()));

    {
        real_block_t _scope{};
        EXPECT_TRUE(is_blocked(signal_a()));
    }

    EXPECT_FALSE(is_blocked(signal_a()));
}

TEST_F(sampling_gotcha_test, LeavesUnrelatedSignalsInTheMaskUntouched)
{
    auto _blocked = make_mask({ signal_b() });
    pthread_sigmask(SIG_BLOCK, &_blocked, nullptr);

    {
        real_block_t _scope{};
        EXPECT_TRUE(is_blocked(signal_b()));
    }

    EXPECT_TRUE(is_blocked(signal_b()));
    EXPECT_FALSE(is_blocked(signal_a()));
}

TEST_F(sampling_gotcha_test, RestoresTheMaskThatBlockSaved)
{
    {
        recording_block_t _scope{};
    }

    ASSERT_EQ(test_globals::g_calls.size(), 2U);
    EXPECT_EQ(test_globals::g_calls[0].op, "block");
    EXPECT_EQ(test_globals::g_calls[0].signals, std::set<int>({ signal_a() }));
    EXPECT_EQ(test_globals::g_calls[1].op, "restore");
    EXPECT_EQ(test_globals::g_calls[1].signals, std::set<int>({ signal_b() }));
}

TEST_F(sampling_gotcha_test, BlocksExactlyTheRegisteredSamplingSignals)
{
    test_globals::g_signals = { signal_a(), signal_b() };

    {
        recording_block_t _scope{};
    }

    ASSERT_FALSE(test_globals::g_calls.empty());
    EXPECT_EQ(test_globals::g_calls[0].signals,
              std::set<int>({ signal_a(), signal_b() }));
}

// --- nesting ---------------------------------------------------------------------

TEST_F(sampling_gotcha_test, NestedScopesTouchTheMaskOnlyOnTheOuterBoundary)
{
    {
        recording_block_t _outer{};
        {
            recording_block_t _inner{};
            EXPECT_EQ(test_globals::g_calls.size(), 1U);
        }
        EXPECT_EQ(test_globals::g_calls.size(), 1U);
    }

    ASSERT_EQ(test_globals::g_calls.size(), 2U);
    EXPECT_EQ(test_globals::g_calls[0].op, "block");
    EXPECT_EQ(test_globals::g_calls[1].op, "restore");
}

TEST_F(sampling_gotcha_test, NestingDepthUnwindsSoALaterScopeBlocksAgain)
{
    {
        recording_block_t _outer{};
        recording_block_t _inner{};
    }
    test_globals::g_calls.clear();

    {
        recording_block_t _scope{};
    }

    ASSERT_EQ(test_globals::g_calls.size(), 2U);
    EXPECT_EQ(test_globals::g_calls[0].op, "block");
}

TEST_F(sampling_gotcha_test, NestedRealScopesRestoreTheOutermostCallersMask)
{
    auto _blocked = make_mask({ signal_a() });
    pthread_sigmask(SIG_BLOCK, &_blocked, nullptr);

    {
        real_block_t _outer{};
        {
            real_block_t _inner{};
            EXPECT_TRUE(is_blocked(signal_a()));
        }
        EXPECT_TRUE(is_blocked(signal_a()));
    }

    EXPECT_TRUE(is_blocked(signal_a()));
}

// --- degenerate signal sets ------------------------------------------------------

TEST_F(sampling_gotcha_test, DoesNotTouchTheMaskWhenNoSignalsAreRegistered)
{
    test_globals::g_signals.clear();

    {
        recording_block_t _scope{};
    }

    EXPECT_TRUE(test_globals::g_calls.empty());
}

TEST_F(sampling_gotcha_test, DoesNotTouchTheMaskWhenTheSignalSetIsUnavailable)
{
    test_globals::g_signals_available = false;

    {
        recording_block_t _scope{};
    }

    EXPECT_TRUE(test_globals::g_calls.empty());
}

TEST_F(sampling_gotcha_test, DoesNotRestoreWhenBlockingFails)
{
    test_globals::g_block_result = EINVAL;

    {
        recording_block_t _scope{};
    }

    ASSERT_EQ(test_globals::g_calls.size(), 1U);
    EXPECT_EQ(test_globals::g_calls[0].op, "block");
}

// --- exception safety ------------------------------------------------------------

TEST_F(sampling_gotcha_test, RestoresTheMaskWhenTheWrappedCallThrows)
{
    ASSERT_FALSE(is_blocked(signal_a()));

    EXPECT_THROW(
        {
            real_gotcha_t _gotcha{};
            _gotcha(MockGotchaData{}, &throwing_callee, 1);
        },
        std::runtime_error);

    EXPECT_FALSE(is_blocked(signal_a()));
}

// --- per-thread independence -----------------------------------------------------

TEST_F(sampling_gotcha_test, NestingDepthIsTrackedPerThread)
{
    recording_block_t _outer_on_main{};
    ASSERT_EQ(test_globals::g_calls.size(), 1U);
    test_globals::g_calls.clear();

    // a fresh thread starts at depth zero, so it masks the signals itself instead of
    // assuming the depth held by the thread that created it
    std::thread{ []() { recording_block_t _scope{}; } }.join();

    ASSERT_EQ(test_globals::g_calls.size(), 2U);
    EXPECT_EQ(test_globals::g_calls[0].op, "block");
    EXPECT_EQ(test_globals::g_calls[1].op, "restore");
}

// A new thread inherits its creator's mask, so a thread spawned while the signals are
// blocked has to leave them blocked -- the same reason the wrapper restores rather
// than unblocks.
TEST_F(sampling_gotcha_test, PreservesAnInheritedBlockedMaskInANewThread)
{
    real_block_t _outer_on_main{};
    ASSERT_TRUE(is_blocked(signal_a()));

    auto _blocked_inside_thread = false;
    auto _blocked_after_thread  = false;

    std::thread{ [&]() {
        {
            real_block_t _scope{};
            _blocked_inside_thread = is_blocked(signal_a());
        }
        _blocked_after_thread = is_blocked(signal_a());
    } }.join();

    EXPECT_TRUE(_blocked_inside_thread);
    EXPECT_TRUE(_blocked_after_thread);
    EXPECT_TRUE(is_blocked(signal_a()));
}

// --- the wrapper itself ----------------------------------------------------------

TEST_F(sampling_gotcha_test, ForwardsArgumentsAndReturnValueAndBlocksDuringTheCall)
{
    real_gotcha_t _gotcha{};

    EXPECT_EQ(_gotcha(MockGotchaData{}, &probe_callee, 21), 42);
    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_TRUE(test_globals::g_blocked_during_call);
    EXPECT_FALSE(is_blocked(signal_a()));
}

TEST_F(sampling_gotcha_test, WrapsCallsThatReturnVoid)
{
    real_gotcha_t _gotcha{};

    _gotcha(MockGotchaData{}, &void_callee);

    EXPECT_EQ(test_globals::g_callee_calls, 1);
    EXPECT_TRUE(test_globals::g_blocked_during_call);
    EXPECT_FALSE(is_blocked(signal_a()));
}

// --- bindings --------------------------------------------------------------------

TEST_F(sampling_gotcha_test, ConfigureRegistersEveryWrappedSymbol)
{
    recording_gotcha_t::configure();
    ASSERT_TRUE(static_cast<bool>(test_globals::g_initializer));
    test_globals::g_initializer();

    EXPECT_EQ(test_globals::g_configured, std::vector<std::string>({ "hsa_init" }));
}

// the number of bindings must not outgrow the capacity reserved for them
TEST_F(sampling_gotcha_test, ConfigureFillsTheDeclaredCapacityExactly)
{
    recording_gotcha_t::configure();
    ASSERT_TRUE(static_cast<bool>(test_globals::g_initializer));
    test_globals::g_initializer();

    EXPECT_EQ(test_globals::g_configured.size(), recording_gotcha_t::gotcha_capacity);
}

TEST_F(sampling_gotcha_test, ConfigureSilencesBindingWarningsWhenRequested)
{
    test_globals::g_suppress_warnings = true;

    recording_gotcha_t::configure();

    for(const auto& itr : test_globals::g_data)
        EXPECT_EQ(itr.verbose, -1);
}

TEST_F(sampling_gotcha_test, ConfigureKeepsBindingWarningsWhenVerbose)
{
    test_globals::g_suppress_warnings = false;

    recording_gotcha_t::configure();

    for(const auto& itr : test_globals::g_data)
        EXPECT_EQ(itr.verbose, 0);
}

TEST_F(sampling_gotcha_test, ShutdownDisablesTheBindings)
{
    recording_gotcha_t::shutdown();

    EXPECT_EQ(test_globals::g_disable_calls, 1);
}

TEST_F(sampling_gotcha_test, StartIsIdempotentWhileTheGotchaIsRunning)
{
    recording_gotcha_t::start();
    EXPECT_EQ(test_globals::g_start_calls, 1);

    recording_gotcha_t::start();
    EXPECT_EQ(test_globals::g_start_calls, 1);
}

// stop() runs on the finalization path, so a subsequent start() has to be able to
// bring the bindings back up rather than find them still marked as running
TEST_F(sampling_gotcha_test, StopDeactivatesTheBindingsSoStartCanRunAgain)
{
    recording_gotcha_t::start();
    ASSERT_TRUE(test_globals::g_is_running);

    recording_gotcha_t::stop();
    EXPECT_EQ(test_globals::g_stop_calls, 1);
    EXPECT_FALSE(test_globals::g_is_running);

    recording_gotcha_t::start();
    EXPECT_EQ(test_globals::g_start_calls, 2);
}
