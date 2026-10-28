// Lifecycle tests for cpu_snap::sampling::sampler driven by a fake trigger,
// so failures in attach/listing can be injected deterministically.

#include <gtest/gtest.h>

#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/triggers/i_trigger.hpp"

namespace
{

using namespace std::chrono_literals;
using cpu_snap::sampling::sampler;

class fake_trigger final : public cpu_snap::sampling::triggers::i_trigger
{
public:
    [[nodiscard]] bool can_provide(
        cpu_snap::sampling::unwind::capture_kind) const noexcept override
    {
        return true;
    }

    void attach_tid(pid_t tid) override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        if(m_tids.contains(tid)) return;
        ++m_attach_calls;
        if(m_fail_attach.load() ||
           (m_throw_on_call != 0 && m_attach_calls == m_throw_on_call))
        {
            throw cpu_snap::environment_error("attach failed");
        }
        m_tids.insert(tid);
    }

    void detach_tid(pid_t tid) noexcept override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        m_tids.erase(tid);
        ++m_detach_calls;
    }

    [[nodiscard]] int wait(int) noexcept override
    {
        std::this_thread::sleep_for(1ms);
        return 0;
    }

    void drain_into(cpu_snap::sampling::internal::streaming_pipeline&,
                    cpu_snap::sampling::unwind::unwinder&) noexcept override
    {
    }

    // Deliberately unsynchronised, like the real backends: the sampler must
    // serialise these against attach/detach.
    void disable_all() noexcept override { visit_tids(); }
    void enable_all() noexcept override { visit_tids(); }

    void close_all() noexcept override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        m_tids.clear();
    }

    [[nodiscard]] std::size_t tracked_tids() const noexcept override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        return m_tids.size();
    }

    [[nodiscard]] bool is_tracked(pid_t tid) const noexcept override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        return m_tids.contains(tid);
    }

    void collect_tids(std::vector<pid_t>& out) const override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        out.assign(m_tids.begin(), m_tids.end());
    }

    [[nodiscard]] std::size_t detach_calls() const
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        return m_detach_calls;
    }

    void visit_tids() noexcept
    {
        for(const pid_t tid : m_tids) m_visited += static_cast<std::size_t>(tid > 0);
    }

    mutable std::mutex m_mutex;
    std::set<pid_t>    m_tids;
    std::size_t        m_visited       = 0;
    std::size_t        m_attach_calls  = 0;
    std::size_t        m_detach_calls  = 0;
    std::size_t        m_throw_on_call = 0;
    std::atomic<bool>  m_fail_attach{ false };
};

struct counting_sink
{
    std::shared_ptr<std::atomic<std::size_t>> bytes =
        std::make_shared<std::atomic<std::size_t>>(0);

    [[nodiscard]] cpu_snap::core::sink_fn fn() const
    {
        return [b = bytes](std::span<const std::byte> data) { b->fetch_add(data.size()); };
    }
};

sampler::config
fake_config(cpu_snap::core::sink_fn sink)
{
    sampler::config cfg;
    cfg.backend                        = cpu_snap::sampling::backend_kind::signal;
    cfg.streaming.sink                 = std::move(sink);
    cfg.threads.rescan_period          = 10ms;
    cfg.threads.exclude_sampler_worker = false;
    return cfg;
}

pid_t
gettid_now()
{
    return static_cast<pid_t>(::syscall(SYS_gettid));
}

class parked_thread
{
public:
    parked_thread()
    : m_thread([this] {
        m_tid.store(gettid_now());
        while(!m_release.load()) std::this_thread::sleep_for(1ms);
    })
    {
        while(m_tid.load() == 0) std::this_thread::sleep_for(1ms);
    }

    ~parked_thread()
    {
        m_release.store(true);
        m_thread.join();
    }

    parked_thread(const parked_thread&)            = delete;
    parked_thread& operator=(const parked_thread&) = delete;

    [[nodiscard]] pid_t tid() const { return m_tid.load(); }

private:
    std::atomic<pid_t> m_tid{ 0 };
    std::atomic<bool>  m_release{ false };
    std::thread        m_thread;
};

template <typename Pred>
bool
wait_for(Pred pred, std::chrono::milliseconds budget = 3000ms)
{
    const auto deadline = std::chrono::steady_clock::now() + budget;
    while(std::chrono::steady_clock::now() < deadline)
    {
        if(pred()) return true;
        std::this_thread::sleep_for(2ms);
    }
    return pred();
}

}  // namespace

TEST(SamplerForkTest, ChildDoesNotWriteIntoParentSink)
{
    int fds[2];
    ASSERT_EQ(::pipe2(fds, O_NONBLOCK | O_CLOEXEC), 0);

    sampler smp(fake_config([wfd = fds[1]](std::span<const std::byte>) {
        const pid_t me = ::getpid();
        [[maybe_unused]] const auto written = ::write(wfd, &me, sizeof(me));
    }));
    smp.start();
    ASSERT_TRUE(smp.is_running());

    const pid_t child = ::fork();
    ASSERT_NE(child, -1);
    if(child == 0)
    {
        ::_exit(0);
    }
    int status = 0;
    ASSERT_EQ(::waitpid(child, &status, 0), child);

    bool  child_wrote = false;
    pid_t writer      = 0;
    while(::read(fds[0], &writer, sizeof(writer)) == static_cast<ssize_t>(sizeof(writer)))
    {
        child_wrote = child_wrote || writer == child;
    }
    smp.stop();
    ::close(fds[0]);
    ::close(fds[1]);
    EXPECT_FALSE(child_wrote);
}

TEST(SamplerStartTest, FailedStartRollsBackAttachedTidsAndStreaming)
{
    parked_thread t1;
    parked_thread t2;
    parked_thread t3;
    counting_sink sink;
    auto          trigger = std::make_unique<fake_trigger>();
    auto*         raw     = trigger.get();
    raw->m_throw_on_call  = 2;

    std::size_t bytes_after_throw = 0;
    {
        sampler smp(fake_config(sink.fn()), std::move(trigger));
        EXPECT_THROW(smp.start(), cpu_snap::environment_error);
        EXPECT_FALSE(smp.is_running());
        EXPECT_EQ(raw->tracked_tids(), 0u);
        bytes_after_throw = sink.bytes->load();
    }
    EXPECT_EQ(sink.bytes->load(), bytes_after_throw);
}

TEST(SamplerRescanTest, AttachFailureDoesNotKillWorker)
{
    counting_sink sink;
    auto          trigger = std::make_unique<fake_trigger>();
    auto*         raw     = trigger.get();
    sampler       smp(fake_config(sink.fn()), std::move(trigger));
    smp.start();

    raw->m_fail_attach.store(true);
    {
        parked_thread failing;
        std::this_thread::sleep_for(100ms);
        raw->m_fail_attach.store(false);
        parked_thread healthy;
        EXPECT_TRUE(wait_for([&] { return raw->is_tracked(healthy.tid()); }));
        EXPECT_TRUE(raw->is_tracked(failing.tid()));
    }
    EXPECT_GT(smp.rescan_failures(), 0u);
    EXPECT_FALSE(smp.worker_failed());
    smp.stop();
}

TEST(SamplerRescanTest, ListingFailureDoesNotDetachTrackedThreads)
{
    counting_sink sink;
    auto          trigger = std::make_unique<fake_trigger>();
    auto*         raw     = trigger.get();
    sampler       smp(fake_config(sink.fn()), std::move(trigger));
    smp.start();
    ASSERT_TRUE(wait_for([&] { return raw->tracked_tids() >= 1; }));

    std::vector<int> hogged;
    for(;;)
    {
        const int fd = ::dup(0);
        if(fd < 0) break;
        hogged.push_back(fd);
    }
    std::this_thread::sleep_for(150ms);
    const auto detached = raw->detach_calls();
    const auto failures = smp.rescan_failures();
    for(const int fd : hogged) ::close(fd);
    smp.stop();
    EXPECT_EQ(detached, 0u);
    EXPECT_GT(failures, 0u);
}

TEST(SamplerPauseResumeTest, ConcurrentPauseResumeWithThreadChurnAndStats)
{
    counting_sink sink;
    auto          trigger = std::make_unique<fake_trigger>();
    sampler       smp(fake_config(sink.fn()), std::move(trigger));
    smp.start();

    std::atomic<bool> done{ false };
    std::thread       toggler([&] {
        while(!done.load())
        {
            smp.pause();
            smp.resume();
        }
    });
    std::thread       poller([&] {
        while(!done.load())
        {
            (void) smp.tracked_threads();
            (void) smp.streaming_internal_errors();
        }
    });
    for(int i = 0; i < 20; ++i)
    {
        std::thread churn([] { std::this_thread::sleep_for(15ms); });
        churn.join();
    }
    done.store(true);
    toggler.join();
    poller.join();
    smp.stop();
    EXPECT_FALSE(smp.worker_failed());
}

TEST(SamplerStartTest, RestartAfterPausedStopStartsUnpaused)
{
    counting_sink sink;
    sampler       smp(fake_config(sink.fn()), std::make_unique<fake_trigger>());

    smp.start();
    smp.pause();
    ASSERT_TRUE(smp.is_paused());
    smp.stop();

    smp.start();
    EXPECT_FALSE(smp.is_paused()) << "a paused sampler stopped and started again must sample";
    smp.stop();
}

TEST(SamplerBackendSelectionTest, InjectedTriggerWithAutomaticIsTreatedAsPerfWithoutReason)
{
    counting_sink sink;
    auto          cfg = fake_config(sink.fn());
    cfg.backend       = cpu_snap::sampling::backend_kind::automatic;
    sampler smp(std::move(cfg), std::make_unique<fake_trigger>());

    EXPECT_EQ(smp.selected_backend().kind, cpu_snap::sampling::backend_kind::perf);
    EXPECT_TRUE(smp.selected_backend().reason.empty());
}
