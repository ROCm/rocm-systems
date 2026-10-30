// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include "lib/common/fork_lifecycle.hpp"
#include "finalization_gate.hpp"
#include "lib/common/fork_control.hpp"
#include "lib/common/session_lifecycle.hpp"
#include "lib/common/static_object.hpp"
#include "lib/common/static_tl_object.hpp"

#include <gtest/gtest.h>

#include <pthread.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

#include <atomic>
#include <cstdlib>
#include <future>
#include <mutex>
#include <thread>
#include <utility>

namespace
{
using rocprofiler::common::fork_lifecycle;
using rocprofiler::common::fork_sequence;
namespace common = rocprofiler::common;

struct fixture
{
    fork_lifecycle                       sdk{};
    fork_lifecycle                       tool{};
    std::mutex                           registry{};
    rocprofiler::tool::finalization_gate completion{};
    bool                                 nested_prepare = false;
    bool                                 sdk_admitted   = false;
    bool                                 poison_cleanup = false;
};

fixture*                   current = nullptr;
thread_local fork_sequence sequence{};
thread_local bool          nesting = false;

void
poison_destructor()
{
    // The vanished parent finalizer owns this mutex in the abandonment cases.
    if(current->nested_prepare && current->tool.abandoned()) std::_Exit(22);
    auto lock = std::lock_guard<std::mutex>{current->registry};
}

void
exit_hook()
{
    if(!current) return;
    if(!current->tool.abandoned() && current->poison_cleanup)
        current->completion.wait_until_complete_or_reentered();
    common::destroy_static_tl_objects();
    common::destroy_static_objects();
}

void
application_prepare()
{
    if(!current || !current->nested_prepare || nesting) return;
    nesting          = true;
    const auto child = fork();
    if(child == 0) std::_Exit(current->tool.abandoned() ? 0 : 19);
    if(child < 0) std::_Exit(10);
    int status = 0;
    if(waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        std::_Exit(11);
    nesting = false;
}

void
tool_prepare()
{
    if(!current || !sequence.begin()) return;
    current->sdk_admitted = current->sdk.try_prepare();
    sequence.admitted     = current->tool.try_prepare();
    if(sequence.admitted)
    {
        // Match production: finish the parent snapshot before later prepare handlers run.
        auto lock = std::lock_guard<std::mutex>{current->registry};
    }
}

void
tool_parent()
{
    if(!current || !sequence.finish()) return;
    if(current->sdk_admitted) current->sdk.parent_complete();
    if(sequence.admitted) current->tool.parent_complete();
}

void
tool_child()
{
    if(!current) return;
    current->sdk.child_complete(!current->sdk_admitted || sequence.nested());
    current->tool.child_complete(!sequence.admitted || sequence.nested() ||
                                 current->sdk.abandoned());
    if(current->tool.abandoned())
    {
        common::abandon_static_objects_after_fork();
        return;
    }
    common::adopt_static_objects_after_fork();
    sequence.finish();
}

enum class scenario
{
    idle,
    nested,
    tool_finalization,
    sdk_finalization,
    concurrent_fork,
};

void
run(scenario test)
{
    // An alarm is a failure bound, not the mechanism establishing the fork/finalize ordering.
    alarm(10);
    fixture state{};
    current              = &state;
    state.nested_prepare = test == scenario::nested;
    state.poison_cleanup =
        test == scenario::tool_finalization || test == scenario::sdk_finalization;
    if(pthread_atfork(application_prepare, nullptr, nullptr) != 0 ||
       pthread_atfork(tool_prepare, tool_parent, tool_child) != 0 || std::atexit(exit_hook) != 0)
        std::_Exit(12);

    if(state.poison_cleanup || state.nested_prepare)
    {
        common::register_static_dtor(poison_destructor);
        common::register_static_tl_dtor(poison_destructor);
    }

    std::promise<void> started;
    std::promise<void> release;
    auto               released = release.get_future().share();
    std::thread        owner;
    if(state.poison_cleanup)
    {
        owner = std::thread{[&]() {
            auto& lifecycle = test == scenario::sdk_finalization ? state.sdk : state.tool;
            auto  activity  = fork_lifecycle::activity{lifecycle};
            auto  lock      = std::unique_lock<std::mutex>{state.registry, std::defer_lock};
            if(test != scenario::sdk_finalization) lock.lock();
            started.set_value();
            released.wait();
        }};
        started.get_future().wait();
    }
    else if(test == scenario::concurrent_fork)
    {
        owner = std::thread{[&]() {
            if(!state.tool.try_prepare()) std::_Exit(13);
            auto lock = std::lock_guard<std::mutex>{state.registry};
            started.set_value();
            released.wait();
            state.tool.parent_complete();
        }};
        started.get_future().wait();
    }

    const auto child = fork();
    if(child == 0)
    {
        alarm(5);
        const bool expected_abandon = state.poison_cleanup || test == scenario::concurrent_fork;
        if(state.tool.abandoned() != expected_abandon) std::_Exit(14);
        // Exercise real atexit and automatic TLS teardown, including poisoned inherited cleanup.
        std::exit(0);
    }
    if(child < 0) std::_Exit(15);
    int status = 0;
    if(waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        std::_Exit(16);
    if(owner.joinable())
    {
        release.set_value();
        owner.join();
    }
    if(state.sdk.abandoned() || state.tool.abandoned()) std::_Exit(17);
    if(!state.tool.try_prepare()) std::_Exit(18);
    state.tool.parent_complete();
    current = nullptr;
    std::_Exit(0);
}
}  // namespace

TEST(fork_lifecycle, idle_child_keeps_profiling)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(run(scenario::idle), ::testing::ExitedWithCode(0), "");
}

TEST(fork_lifecycle, nested_prepare_does_not_reacquire_registry)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(run(scenario::nested), ::testing::ExitedWithCode(0), "");
}

TEST(fork_lifecycle, child_does_not_wait_for_vanished_tool_finalizer_or_cleanup)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(run(scenario::tool_finalization), ::testing::ExitedWithCode(0), "");
}

TEST(fork_lifecycle, child_abandons_sdk_teardown_before_tool_callback)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(run(scenario::sdk_finalization), ::testing::ExitedWithCode(0), "");
}

TEST(fork_lifecycle, concurrent_fork_cannot_inherit_another_threads_registry_lock)
{
    ::testing::FLAGS_gtest_death_test_style = "threadsafe";
    ASSERT_EXIT(run(scenario::concurrent_fork), ::testing::ExitedWithCode(0), "");
}

TEST(fork_lifecycle, prepare_reentry_proceeds_and_invalidates_snapshot)
{
    fork_lifecycle state{};
    ASSERT_TRUE(state.try_prepare());
    {
        auto activity = fork_lifecycle::activity{state};
        EXPECT_FALSE(state.try_prepare());
        EXPECT_FALSE(state.child_eligible());
        state.parent_complete();
        EXPECT_FALSE(state.try_prepare());  // parent completion must preserve the activity count
    }
    EXPECT_TRUE(state.try_prepare());
    {
        auto completed_activity = fork_lifecycle::activity{state};
    }
    EXPECT_FALSE(state.child_eligible());  // completed lifecycle work is still a new epoch
    state.child_complete();
    EXPECT_TRUE(state.abandoned());
}

TEST(session_lifecycle, detach_publication_precedes_finalization)
{
    common::session_lifecycle state{};
    bool                      published = false;
    std::promise<void>        started;
    std::promise<void>        release;
    auto                      released = release.get_future();
    auto                      detach   = std::thread{[&]() {
        auto operation = state.update();
        EXPECT_TRUE(operation);
        started.set_value();
        released.wait();
        published = true;
    }};
    started.get_future().wait();
    auto finalizer = std::thread{[&]() {
        auto operation = state.finalize();
        EXPECT_TRUE(published);
    }};
    release.set_value();
    detach.join();
    finalizer.join();
    EXPECT_FALSE(state.update());
}

TEST(session_lifecycle, synchronous_output_is_owned_until_complete)
{
    common::session_lifecycle state{};
    std::promise<void>        started;
    std::promise<void>        release;
    auto                      released        = release.get_future();
    bool                      output_complete = false;
    auto                      detach          = std::thread{[&]() {
        auto operation = state.update();
        started.set_value();
        released.wait();
        output_complete = true;
    }};
    started.get_future().wait();
    auto finalizer = std::thread{[&]() {
        auto operation = state.finalize();
        EXPECT_TRUE(output_complete);
        // SDK finalization dispatches its client finalizer while retaining exclusive ownership.
        auto client = state.dispatch_finalizer();
        EXPECT_TRUE(client);
    }};
    release.set_value();
    detach.join();
    finalizer.join();
}

namespace
{
int rebuild_step   = 0;
int core_step      = 0;
int auxiliary_step = 0;
void
rebuild_core()
{
    core_step = ++rebuild_step;
}
void
rebuild_auxiliary()
{
    auxiliary_step = ++rebuild_step;
}
}  // namespace

TEST(session_lifecycle, prepare_admission_does_not_wait_for_another_owner)
{
    common::session_lifecycle state{};
    std::promise<void>        started;
    std::promise<void>        release;
    auto                      released = release.get_future();
    auto                      owner    = std::thread{[&]() {
        auto operation = state.update();
        started.set_value();
        released.wait();
    }};
    started.get_future().wait();
    EXPECT_FALSE(state.try_update());
    release.set_value();
    owner.join();
    EXPECT_TRUE(state.try_update());
}

TEST(fork_dispatcher, core_rebuild_precedes_later_registered_auxiliary_workers)
{
    common::fork_dispatcher  dispatcher{};
    common::fork_participant core{[](bool) {},
                                  []() noexcept {},
                                  []() noexcept { return true; },
                                  []() noexcept {},
                                  rebuild_core,
                                  nullptr,
                                  nullptr,
                                  common::fork_rebuild_phase::core};
    common::fork_participant auxiliary{[](bool) {},
                                       []() noexcept {},
                                       []() noexcept { return true; },
                                       []() noexcept {},
                                       rebuild_auxiliary};

    // add() prepends, matching production where auxiliary services register after the tool.
    dispatcher.add(core);
    dispatcher.add(auxiliary);
    rebuild_step = core_step = auxiliary_step = 0;
    dispatcher.rebuild();
    EXPECT_EQ(core_step, 1);
    EXPECT_EQ(auxiliary_step, 2);
}
