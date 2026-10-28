// Tests for the injected logger (sampler::config::logger).

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <sys/syscall.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include "cpu_snap/error.hpp"
#include "cpu_snap/sampling/backend_selection.hpp"
#include "cpu_snap/sampling/clock_kind.hpp"
#include "cpu_snap/sampling/logger.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/sampling/triggers/i_trigger.hpp"

namespace
{

using namespace std::chrono_literals;
using cpu_snap::sampling::backend_choice;
using cpu_snap::sampling::backend_kind;
using cpu_snap::sampling::log_level;
using cpu_snap::sampling::log_record;
using cpu_snap::sampling::sampler;

struct owned_record
{
    log_level     level;
    std::uint64_t timestamp_ns;
    std::string   message;
    std::thread::id thread;
};

class recording_sink
{
public:
    [[nodiscard]] cpu_snap::sampling::log_sink fn()
    {
        return [this](const log_record& r) {
            const std::lock_guard<std::mutex> lk(m_mutex);
            m_records.push_back(
                { r.level, r.timestamp_ns, std::string(r.message), std::this_thread::get_id() });
        };
    }

    [[nodiscard]] std::vector<owned_record> records() const
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        return m_records;
    }

private:
    mutable std::mutex       m_mutex;
    std::vector<owned_record> m_records;
};

std::uint64_t
mono_now()
{
    timespec ts{};
    ::clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

class stub_trigger final : public cpu_snap::sampling::triggers::i_trigger
{
public:
    void fail_tid(pid_t tid) { m_failing_tid.store(tid); }

    [[nodiscard]] bool can_provide(cpu_snap::sampling::unwind::capture_kind) const noexcept override
    {
        return true;
    }

    void attach_tid(pid_t tid) override
    {
        if(tid == m_failing_tid.load()) throw cpu_snap::environment_error("boom attach refused");
        const std::lock_guard<std::mutex> lk(m_mutex);
        m_tids.insert(tid);
    }

    void detach_tid(pid_t tid) noexcept override
    {
        const std::lock_guard<std::mutex> lk(m_mutex);
        m_tids.erase(tid);
    }

    [[nodiscard]] int wait(int) noexcept override
    {
        std::this_thread::sleep_for(1ms);
        return 0;
    }

    void drain_into(cpu_snap::sampling::internal::streaming_pipeline&,
                    cpu_snap::sampling::unwind::unwinder&) noexcept override
    {}
    void disable_all() noexcept override {}
    void enable_all() noexcept override {}

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

private:
    std::atomic<pid_t> m_failing_tid{ -1 };
    mutable std::mutex m_mutex;
    std::set<pid_t>    m_tids;
};

sampler::config
make_config(cpu_snap::sampling::log_sink logger)
{
    sampler::config cfg;
    cfg.backend                        = backend_kind::signal;
    cfg.streaming.sink                 = [](std::span<const std::byte>) {};
    cfg.threads.rescan_period          = 10ms;
    cfg.threads.exclude_sampler_worker = false;
    cfg.logger                         = std::move(logger);
    return cfg;
}

TEST(logger, no_logger_set_lifecycle_works)
{
    sampler s(make_config({}), std::make_unique<stub_trigger>());
    s.start();
    s.stop();
    SUCCEED();
}

TEST(logger, sink_receives_ctor_start_stop_in_order_with_monotonic_timestamps)
{
    recording_sink rec;

    const auto t0 = mono_now();
    sampler     s(make_config(rec.fn()), std::make_unique<stub_trigger>());
    const auto t1 = mono_now();
    const auto ctor_records = rec.records();
    ASSERT_FALSE(ctor_records.empty());
    EXPECT_EQ(ctor_records.front().level, log_level::info);
    EXPECT_NE(ctor_records.front().message.find("signal backend"), std::string::npos);
    for(const auto& r : ctor_records)
    {
        EXPECT_GE(r.timestamp_ns, t0);
        EXPECT_LE(r.timestamp_ns, t1);
    }

    const auto t2 = mono_now();
    s.start();
    const auto t3 = mono_now();
    const auto after_start = rec.records();
    ASSERT_EQ(after_start.size(), ctor_records.size() + 1);
    const auto& start_rec = after_start.back();
    EXPECT_EQ(start_rec.level, log_level::info);
    EXPECT_NE(start_rec.message.find("sampler started"), std::string::npos);
    EXPECT_NE(start_rec.message.find("backend=signal"), std::string::npos);
    EXPECT_NE(start_rec.message.find("freq="), std::string::npos);
    EXPECT_NE(start_rec.message.find("unwind="), std::string::npos);
    EXPECT_GE(start_rec.timestamp_ns, t2);
    EXPECT_LE(start_rec.timestamp_ns, t3);

    const auto t4 = mono_now();
    s.stop();
    const auto t5 = mono_now();
    const auto all = rec.records();
    ASSERT_EQ(all.size(), after_start.size() + 1);
    const auto& stop_rec = all.back();
    EXPECT_EQ(stop_rec.level, log_level::info);
    EXPECT_NE(stop_rec.message.find("sampler stopped"), std::string::npos);
    EXPECT_NE(stop_rec.message.find("samples="), std::string::npos);
    EXPECT_NE(stop_rec.message.find("dropped="), std::string::npos);
    EXPECT_NE(stop_rec.message.find("internal_errors="), std::string::npos);
    EXPECT_NE(stop_rec.message.find("live_stack_samples="), std::string::npos);
    EXPECT_GE(stop_rec.timestamp_ns, t4);
    EXPECT_LE(stop_rec.timestamp_ns, t5);
}

std::vector<std::string>
ctor_messages(sampler::config cfg)
{
    recording_sink rec;
    cfg.logger = rec.fn();
    sampler s(std::move(cfg), std::make_unique<stub_trigger>());
    std::vector<std::string> out;
    for(const auto& r : rec.records()) out.push_back(r.message);
    return out;
}

bool
contains_line(const std::vector<std::string>& lines, const std::string& needle)
{
    return std::ranges::any_of(
        lines, [&](const std::string& l) { return l.find(needle) != std::string::npos; });
}

TEST(logger, configuration_line_reports_explicit_unwind_without_auto_suffix)
{
    auto cfg                  = make_config({});
    cfg.clock                 = cpu_snap::sampling::clock_kind::boottime;
    cfg.perf.unwind           = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.signal.sample_freq_hz = 500;
    const auto lines          = ctor_messages(std::move(cfg));
    ASSERT_GE(lines.size(), 2U);
    EXPECT_NE(lines.front().find("signal backend"), std::string::npos);
    EXPECT_EQ(lines[1],
              "configuration: backend=signal clock=boottime unwind=frame_pointer freq=500Hz");
}

TEST(logger, configuration_line_marks_auto_best)
{
    const auto lines = ctor_messages(make_config({}));
    EXPECT_TRUE(contains_line(
        lines, "configuration: backend=signal clock=monotonic unwind=dwarf (auto_best) freq=200Hz"));
}

TEST(logger, frame_pointer_hint_logged_for_dwarf_at_10khz)
{
    auto cfg                  = make_config({});
    cfg.signal.sample_freq_hz = 10000;
    EXPECT_TRUE(contains_line(ctor_messages(std::move(cfg)), "hint: at >= 10 kHz"));
}

TEST(logger, frame_pointer_hint_not_logged_at_default_rate_or_for_frame_pointer)
{
    EXPECT_FALSE(contains_line(ctor_messages(make_config({})), "hint:"));

    auto cfg                  = make_config({});
    cfg.perf.unwind           = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.signal.sample_freq_hz = 10000;
    EXPECT_FALSE(contains_line(ctor_messages(std::move(cfg)), "hint:"));
}

TEST(logger, throwing_sink_never_propagates)
{
    auto thrower = [](const log_record&) { throw std::runtime_error("sink failure"); };
    EXPECT_NO_THROW({
        sampler s(make_config(thrower), std::make_unique<stub_trigger>());
        s.start();
        s.stop();
    });
}

TEST(logger, failed_attach_is_logged_once_per_tid)
{
    recording_sink rec;
    auto           trigger = std::make_unique<stub_trigger>();
    auto*          stub    = trigger.get();
    sampler        s(make_config(rec.fn()), std::move(trigger));
    s.start();

    std::atomic<pid_t> failing{ 0 };
    std::atomic<bool>  release{ false };
    std::thread        victim([&] {
        failing.store(static_cast<pid_t>(::syscall(SYS_gettid)));
        while(!release.load()) std::this_thread::sleep_for(1ms);
    });
    while(failing.load() == 0) std::this_thread::sleep_for(1ms);
    stub->fail_tid(failing.load());
    const auto deadline = std::chrono::steady_clock::now() + 5s;
    while(s.rescan_failures() < 4 && std::chrono::steady_clock::now() < deadline)
    {
        std::this_thread::sleep_for(5ms);
    }
    s.stop();
    release.store(true);
    victim.join();

    EXPECT_GE(s.rescan_failures(), 3U);
    const auto records = rec.records();
    const auto warns   = std::count_if(records.begin(), records.end(), [&](const owned_record& r) {
        return r.level == log_level::warn &&
               r.message.find("boom attach refused") != std::string::npos &&
               r.message.find(std::to_string(failing.load())) != std::string::npos;
    });
    EXPECT_EQ(warns, 1);
}

TEST(logger, describe_backend_choice_texts)
{
    using cpu_snap::sampling::describe_backend_choice;
    EXPECT_EQ(describe_backend_choice({ backend_kind::perf, {} }), "using the perf backend");
    EXPECT_EQ(describe_backend_choice({ backend_kind::signal, {} }), "using the signal backend");
    const backend_choice fallback{ backend_kind::signal,
                                   "perf_event_open failed (Permission denied); using the "
                                   "signal backend" };
    EXPECT_EQ(describe_backend_choice(fallback), fallback.reason);
}

TEST(logger, real_signal_sampler_logs_only_from_constructing_or_worker_thread)
{
    recording_sink rec;
    sampler::config cfg;
    cfg.backend            = backend_kind::signal;
    cfg.streaming.sink     = [](std::span<const std::byte>) {};
    cfg.logger             = rec.fn();
    cfg.threads.rescan_period = 10ms;

    const auto main_id = std::this_thread::get_id();
    {
        sampler s(std::move(cfg));
        s.start();
        std::this_thread::sleep_for(300ms);
        s.stop();
    }
    const auto records = rec.records();
    ASSERT_GE(records.size(), 3U);
    std::set<std::thread::id> others;
    for(const auto& r : records)
    {
        if(r.thread != main_id) others.insert(r.thread);
    }
    EXPECT_LE(others.size(), 1U);
}

}  // namespace
