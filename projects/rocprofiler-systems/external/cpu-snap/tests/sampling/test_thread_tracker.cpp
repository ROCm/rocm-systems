#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/thread_tracker.hpp"

namespace
{

using namespace cpu_snap::sampling;
using internal::thread_entry;
using internal::thread_tracker;

class fake_trigger final : public triggers::i_trigger
{
public:
    std::vector<pid_t>      attached;
    std::vector<pid_t>      detached;
    std::unordered_set<pid_t> failing;
    std::unordered_set<pid_t> tracked;

    void attach_tid(pid_t tid) override
    {
        attached.push_back(tid);
        if(failing.contains(tid))
        {
            throw cpu_snap::environment_error("boom");
        }
        tracked.insert(tid);
    }
    void detach_tid(pid_t tid) noexcept override
    {
        detached.push_back(tid);
        tracked.erase(tid);
    }
    bool is_tracked(pid_t tid) const noexcept override { return tracked.contains(tid); }
    void collect_tids(std::vector<pid_t>& out) const override
    {
        out.assign(tracked.begin(), tracked.end());
    }

    bool can_provide(unwind::capture_kind) const noexcept override { return true; }
    void close_all() noexcept override {}
    void enable_all() noexcept override {}
    void disable_all() noexcept override {}
    std::size_t tracked_tids() const noexcept override { return tracked.size(); }
    int wait(int) noexcept override { return 0; }
    void drain_into(internal::streaming_pipeline&, unwind::unwinder&) noexcept override {}
};

class ThreadTrackerTest : public ::testing::Test
{
protected:
    thread_tracker make(bool include_main = true)
    {
        return thread_tracker(
            [this] { return threads; }, [this](const log_record& r) { logs.emplace_back(r.message); },
            clock_kind::monotonic, include_main);
    }

    std::size_t count_logs(const std::string& needle) const
    {
        return static_cast<std::size_t>(std::ranges::count_if(
            logs, [&](const std::string& m) { return m.find(needle) != std::string::npos; }));
    }

    internal::rescan_result rescan(thread_tracker& t)
    {
        auto current = t.poll();
        return t.rescan(trig, nullptr, -1, *current);
    }

    std::optional<std::vector<thread_entry>> threads;
    std::vector<std::string>                 logs;
    fake_trigger                             trig;
};

TEST_F(ThreadTrackerTest, InitialAttachSkipsExcludedTid)
{
    threads = std::vector<thread_entry>{ { 10, 1 }, { 11, 2 }, { 12, 3 } };
    auto t  = make();
    t.attach_initial(trig, 11);
    EXPECT_EQ(trig.attached, (std::vector<pid_t>{ 10, 12 }));
}

TEST_F(ThreadTrackerTest, InitialAttachThrowsWhenListingFails)
{
    threads.reset();
    auto t = make();
    EXPECT_THROW(t.attach_initial(trig, -1), cpu_snap::environment_error);
}

TEST_F(ThreadTrackerTest, RescanAttachesNewAndDetachesExited)
{
    threads = std::vector<thread_entry>{ { 10, 1 }, { 11, 2 } };
    auto t  = make();
    t.attach_initial(trig, -1);

    threads = std::vector<thread_entry>{ { 11, 2 }, { 12, 3 } };
    const auto res = rescan(t);
    EXPECT_EQ(res.failed_attaches, 0U);
    EXPECT_EQ(trig.detached, (std::vector<pid_t>{ 10 }));
    EXPECT_TRUE(trig.tracked.contains(12));
    EXPECT_FALSE(trig.tracked.contains(10));
}

TEST_F(ThreadTrackerTest, RescanWithPipelineNull)
{
    threads = std::vector<thread_entry>{ { 10, 1 } };
    auto t  = make();
    t.attach_initial(trig, -1);
    threads = std::vector<thread_entry>{};
    auto current = t.poll();
    EXPECT_NO_THROW(t.rescan(trig, nullptr, -1, *current));
    EXPECT_EQ(trig.detached, (std::vector<pid_t>{ 10 }));
}

TEST_F(ThreadTrackerTest, RecycledTidIsDetachedAndReattached)
{
    threads = std::vector<thread_entry>{ { 10, 100 } };
    auto t  = make();
    t.attach_initial(trig, -1);

    threads = std::vector<thread_entry>{ { 10, 200 } };
    (void) rescan(t);
    EXPECT_EQ(trig.detached, (std::vector<pid_t>{ 10 }));
    EXPECT_EQ(trig.attached, (std::vector<pid_t>{ 10, 10 }));
    EXPECT_TRUE(trig.tracked.contains(10));
}

TEST_F(ThreadTrackerTest, UnchangedStartTicksAreNotRecycled)
{
    threads = std::vector<thread_entry>{ { 10, 100 } };
    auto t  = make();
    t.attach_initial(trig, -1);
    (void) rescan(t);
    EXPECT_TRUE(trig.detached.empty());
}

TEST_F(ThreadTrackerTest, AttachFailureCountedAndLoggedOncePerTid)
{
    auto t = make();
    threads = std::vector<thread_entry>{ { 10, 1 } };
    trig.failing = { 10 };

    for(int i = 0; i < 3; ++i)
    {
        EXPECT_EQ(rescan(t).failed_attaches, 1U);
    }
    EXPECT_EQ(count_logs("cannot attach to tid 10: boom"), 1U);

    threads = std::vector<thread_entry>{};
    (void) rescan(t);
    threads = std::vector<thread_entry>{ { 10, 5 } };
    (void) rescan(t);
    EXPECT_EQ(count_logs("cannot attach to tid 10: boom"), 2U);
}

TEST_F(ThreadTrackerTest, ListingFailureLoggedOnceUntilSuccess)
{
    auto t = make();
    threads.reset();
    EXPECT_FALSE(t.poll());
    EXPECT_FALSE(t.poll());
    EXPECT_EQ(count_logs("cannot enumerate"), 1U);

    threads = std::vector<thread_entry>{};
    EXPECT_TRUE(t.poll());
    threads.reset();
    EXPECT_FALSE(t.poll());
    EXPECT_EQ(count_logs("cannot enumerate"), 2U);
}

TEST_F(ThreadTrackerTest, ResetForgetsLoggedFailuresAndStartTicks)
{
    auto t = make();
    threads = std::vector<thread_entry>{ { 10, 1 } };
    trig.failing = { 10 };
    (void) rescan(t);
    EXPECT_EQ(count_logs("cannot attach"), 1U);

    t.reset();
    (void) rescan(t);
    EXPECT_EQ(count_logs("cannot attach"), 2U);

    trig.failing.clear();
    t.attach_initial(trig, -1);
    t.reset();
    threads = std::vector<thread_entry>{ { 10, 999 } };
    (void) rescan(t);
    EXPECT_TRUE(trig.detached.empty());
}

}  // namespace
