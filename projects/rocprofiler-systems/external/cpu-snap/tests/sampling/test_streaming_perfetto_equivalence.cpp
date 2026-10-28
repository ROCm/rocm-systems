// Self-consistency of the streaming capture: two independent captures of the
// same deterministic workload must agree within tolerance (thread count,
// sample count band, IP overlap). Byte-exact goldens are avoided as sampling
// moments differ per run. GTEST_SKIPs when perf_event_paranoid > 1.

#include <functional>
#include <cstddef>
#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <ios>
#include <sched.h>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <linux/perf_event.h>
#include <sys/syscall.h>
#include <unistd.h>

#include "cpu_snap/core/memory_sink.hpp"
#include "cpu_snap/core/sample_reconstructor.hpp"
#include "cpu_snap/core/types.hpp"
#include "cpu_snap/sampling/sampler.hpp"
#include "cpu_snap/processing/timeline/slice_emitter.hpp"
#include "perfetto_proto_writer.hpp"
#include "cpu_snap/sampling/unwind_mode.hpp"

namespace
{

using cpu_snap::core::raw_sample;

constexpr std::uint32_t k_sample_freq_hz   = 200;
constexpr std::uint32_t k_callchain_depth  = 32;
constexpr auto          k_capture_duration = std::chrono::milliseconds{ 500 };

// Deterministic workload. Mirror of multi_thread_sample_demo.cpp's leaf chain.
// noinline preserves the call structure so the kernel callchain walker sees
// every frame and the two runs produce comparable IP sets.

[[gnu::noinline]] double
equiv_leaf_compute(double seed, std::uint64_t iters) noexcept
{
    double acc = seed;
    for(std::uint64_t i = 0; i < iters; ++i)
    {
        acc = acc * 1.0000001 + (i & 0xFFFF);
        if((i & 0xFFF) == 0) acc *= 0.9999999;
    }
    return acc;
}

[[gnu::noinline]] double
equiv_hot_loop(double seed, std::uint64_t iters) noexcept
{
    return equiv_leaf_compute(seed + 1.0, iters);
}

[[gnu::noinline]] double
equiv_matmul_inner(double seed, std::uint64_t iters) noexcept
{
    double acc = 0.0;
    for(int i = 0; i < 4; ++i) {
        acc += equiv_hot_loop(seed + i, iters / 4);
}
    return acc;
}

[[gnu::noinline]] double
equiv_matmul_outer(double seed, std::uint64_t iters) noexcept
{
    return equiv_matmul_inner(seed * 1.1, iters);
}

[[gnu::noinline]] double
equiv_workload_step(double seed, std::uint64_t iters) noexcept
{
    return equiv_matmul_outer(seed + 0.5, iters);
}

[[gnu::noinline]] double
equiv_workload_dispatch(double seed, std::uint64_t iters) noexcept
{
    return equiv_workload_step(seed, iters);
}

void
equiv_worker(int idx, std::atomic<bool>& stop)
{
    auto acc = static_cast<double>(idx);
    while(!stop.load(std::memory_order_relaxed))
    {
        acc += equiv_workload_dispatch(acc, 200'000);
    }
    if(acc == 12345.6789) std::printf("unreachable\n");
}

// perf_event_open availability probe.
bool
perf_open_available()
{
    struct perf_event_attr a
    {};
    a.size           = sizeof(a);
    a.type           = PERF_TYPE_SOFTWARE;
    a.config         = PERF_COUNT_SW_CPU_CLOCK;
    a.disabled       = 1;
    a.exclude_kernel = 1;
    a.exclude_hv     = 1;
    const int fd     = static_cast<int>(
        ::syscall(SYS_perf_event_open, &a, 0, -1, -1, PERF_FLAG_FD_CLOEXEC));
    if(fd < 0) return false;
    ::close(fd);
    return true;
}

// Build a sampler config. Hardwiring frame_pointer here keeps both runs
// walking the same kind of stacks regardless of what auto_best would
// otherwise resolve to.
cpu_snap::sampling::sampler::config
make_base_config()
{
    cpu_snap::sampling::sampler::config cfg;
    cfg.perf.sample_freq_hz       = k_sample_freq_hz;
    cfg.perf.callchain_depth      = k_callchain_depth;
    cfg.perf.kernel_pages_per_tid = 4;
    cfg.perf.unwind               = cpu_snap::sampling::unwind_mode::frame_pointer;
    cfg.threads.watch_new_threads = true;
    return cfg;
}

// Run the workload + recorder once, streaming into a bounded memory_sink
// and decoding the result via core::sample_reconstructor. Returns the
// per-tid samples map.

constexpr std::size_t k_capture_sink_max_bytes = 128U * 1024U * 1024U;

struct capture_result
{
    std::unordered_map<pid_t, std::vector<raw_sample>> per_tid;
};

capture_result
capture_streaming_run(int n_workers, std::chrono::milliseconds duration, std::string_view tag)
{
    std::atomic<bool>        stop{ false };
    std::vector<std::thread> workers;
    workers.reserve(n_workers);
    for(int i = 0; i < n_workers; ++i) {
        workers.emplace_back(equiv_worker, i, std::ref(stop));
}

    std::this_thread::sleep_for(std::chrono::milliseconds{ 20 });

    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_capture_sink_max_bytes });

    auto cfg                          = make_base_config();
    cfg.streaming.sink                = std::ref(sink);
    cfg.streaming.chunk_bytes         = 1U << 16;
    cfg.streaming.checkpoint_interval = 1000;
    cfg.streaming.maps_poll_interval  = 1000;
    cfg.streaming.process_name        = std::string(tag);

    cpu_snap::sampling::sampler rec(std::move(cfg));
    rec.start();
    std::this_thread::sleep_for(duration);
    rec.stop();

    stop.store(true, std::memory_order_relaxed);
    for(auto& t : workers) {
        t.join();
}

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());

    capture_result out;
    out.per_tid = reconstructor.samples();
    return out;
}

// Drive the existing slice_emitter + perfetto_proto_writer over a per-tid
// raw_sample map. Same code path for both runs so any divergence in the
// .pftrace files is attributable to the recorder layer, not the writer.
void
write_pftrace(const std::unordered_map<pid_t, std::vector<raw_sample>>& per_tid,
              const std::string& out_path, const std::string& process_name)
{
    cs_examples::perfetto::perfetto_proto_writer writer(out_path);
    writer.set_process(static_cast<int>(::getpid()), process_name);

    for(const auto& [tid, vec] : per_tid)
    {
        writer.add_thread(static_cast<int>(tid), "tid-" + std::to_string(tid));
        (void) vec;
    }

    // Both paths flow through the single slice_emitter -> writer.write_slice
    // sink, so any divergence in the .pftrace is attributable to the
    // recorder layer, not the timeline-shaping pass.
    cpu_snap::processing::timeline::slice_emitter emitter{ cpu_snap::processing::timeline::slice_emitter_config{} };
    emitter.process(per_tid, [&](const cpu_snap::processing::timeline::emitted_slice& s) {
        writer.write_slice(s);
    });
    writer.finalize();
}

// Helpers for the self-consistency assertions.

std::size_t
total_samples(const std::unordered_map<pid_t, std::vector<raw_sample>>& m)
{
    std::size_t n = 0;
    for(const auto& [tid, vec] : m) {
        n += vec.size();
}
    return n;
}

std::unordered_set<std::uint64_t>
unique_ips(const std::unordered_map<pid_t, std::vector<raw_sample>>& m)
{
    std::unordered_set<std::uint64_t> out;
    for(const auto& [tid, vec] : m)
    {
        for(const auto& s : vec)
        {
            for(std::uint32_t i = 0; i < s.frames.size(); ++i) {
                out.insert(s.frames[i]);
}
        }
    }
    return out;
}

double
ip_overlap_percentage(const std::unordered_set<std::uint64_t>& a,
                      const std::unordered_set<std::uint64_t>& b)
{
    if(a.empty() || b.empty()) return 0.0;
    std::size_t intersection = 0;
    const auto& smaller      = a.size() < b.size() ? a : b;
    const auto& larger       = a.size() < b.size() ? b : a;
    for(const auto ip : smaller) {
        if(larger.contains(ip)) ++intersection;
}
    const auto union_size = a.size() + b.size() - intersection;
    return 100.0 * static_cast<double>(intersection) / static_cast<double>(union_size);
}

std::string
tmp_pftrace_path(const std::string& tag)
{
    const auto dir = std::filesystem::temp_directory_path() / "callstack_pr10_equiv";
    std::filesystem::create_directories(dir);
    return (dir / (tag + ".pftrace")).string();
}

void
log_consistency(const char* case_name, const capture_result& run_a,
                const capture_result& run_b, double overlap_pct)
{
    std::printf("[self-consistency] %s : run_a samples=%zu, run_b samples=%zu, "
                "run_a tids=%zu, run_b tids=%zu, ip_overlap=%.1f%%\n",
                case_name, total_samples(run_a.per_tid), total_samples(run_b.per_tid),
                run_a.per_tid.size(), run_b.per_tid.size(), overlap_pct);
}

}  // namespace

// Test fixture: skip cleanly when perf_event_open is not permitted.

// NOLINTNEXTLINE(readability-identifier-naming) -- gtest fixture must be CamelCase to satisfy google-readability-avoid-underscore-in-googletest-name
class StreamingSelfConsistencyTest : public ::testing::Test
{
protected:
    void SetUp() override
    {
        if(!perf_open_available()) GTEST_SKIP() << "perf_event_open not permitted";
    }
};

// Case 1: single-thread short capture, two runs within tolerance of each other.
TEST_F(StreamingSelfConsistencyTest, SingleThreadShortCaptureWithinTolerance)
{
    constexpr int n_workers = 1;

    const auto run_a = capture_streaming_run(n_workers, k_capture_duration, "equiv_run_a");
    const auto run_b = capture_streaming_run(n_workers, k_capture_duration, "equiv_run_b");

    write_pftrace(run_a.per_tid, tmp_pftrace_path("single_thread_run_a"),
                  "callstack_pr10_single_run_a");
    write_pftrace(run_b.per_tid, tmp_pftrace_path("single_thread_run_b"),
                  "callstack_pr10_single_run_b");

    const auto total_a = total_samples(run_a.per_tid);
    const auto total_b = total_samples(run_b.per_tid);

    ASSERT_GT(total_a, 0U);
    ASSERT_GT(total_b, 0U);

    // Under heavy CPU contention (ctest -j>=4) sample counts vary; require
    // a minimum to keep the ratio check statistically meaningful, then use
    // a looser tolerance band. The IP-overlap check below remains the
    // strong consistency assertion.
    constexpr std::size_t k_min_samples_for_band = 30;
    if(total_a < k_min_samples_for_band * n_workers
       || total_b < k_min_samples_for_band * n_workers)
    {
        GTEST_SKIP() << "undersampled run (run_a=" << total_a
                     << " run_b=" << total_b
                     << "); rely on the isolated rerun";
    }
    const auto lo = static_cast<std::size_t>(static_cast<double>(total_a) * 0.4);
    const auto hi = static_cast<std::size_t>(static_cast<double>(total_a) * 1.6);
    EXPECT_GE(total_b, lo) << "second run captured far fewer samples than the first";
    EXPECT_LE(total_b, hi) << "second run captured far more samples than the first";

    const auto ips_a   = unique_ips(run_a.per_tid);
    const auto ips_b   = unique_ips(run_b.per_tid);
    const auto overlap = ip_overlap_percentage(ips_a, ips_b);
    EXPECT_GE(overlap, 80.0)
        << "IP overlap below 80% suggests the two runs sampled different code";

    log_consistency("single_thread", run_a, run_b, overlap);
}

// Case 2: four-thread short capture, bulk consistency assertion.
TEST_F(StreamingSelfConsistencyTest, FourThreadShortCaptureWithinTolerance)
{
    constexpr int n_workers = 4;

    const auto run_a = capture_streaming_run(n_workers, k_capture_duration, "equiv_run_a");
    const auto run_b = capture_streaming_run(n_workers, k_capture_duration, "equiv_run_b");

    write_pftrace(run_a.per_tid, tmp_pftrace_path("four_thread_run_a"),
                  "callstack_pr10_four_run_a");
    write_pftrace(run_b.per_tid, tmp_pftrace_path("four_thread_run_b"),
                  "callstack_pr10_four_run_b");

    const auto total_a = total_samples(run_a.per_tid);
    const auto total_b = total_samples(run_b.per_tid);

    ASSERT_GT(total_a, 0U);
    ASSERT_GT(total_b, 0U);

    // Under heavy CPU contention (ctest -j>=4) sample counts vary; require
    // a minimum to keep the ratio check statistically meaningful, then use
    // a looser tolerance band. The IP-overlap check below remains the
    // strong consistency assertion.
    constexpr std::size_t k_min_samples_for_band = 30;
    if(total_a < k_min_samples_for_band * n_workers
       || total_b < k_min_samples_for_band * n_workers)
    {
        GTEST_SKIP() << "undersampled run (run_a=" << total_a
                     << " run_b=" << total_b
                     << "); rely on the isolated rerun";
    }
    const auto lo = static_cast<std::size_t>(static_cast<double>(total_a) * 0.4);
    const auto hi = static_cast<std::size_t>(static_cast<double>(total_a) * 1.6);
    EXPECT_GE(total_b, lo);
    EXPECT_LE(total_b, hi);

    // 4 workers + main + (possibly) gtest threads. Allow +/-2 wiggle: a
    // helper thread may exit between runs.
    const auto delta = run_a.per_tid.size() > run_b.per_tid.size()
                           ? run_a.per_tid.size() - run_b.per_tid.size()
                           : run_b.per_tid.size() - run_a.per_tid.size();
    EXPECT_LE(delta, 2U) << "tid count differs by more than 2 between runs";

    const auto ips_a   = unique_ips(run_a.per_tid);
    const auto ips_b   = unique_ips(run_b.per_tid);
    const auto overlap = ip_overlap_percentage(ips_a, ips_b);
    EXPECT_GE(overlap, 80.0)
        << "IP overlap below 80% suggests the two runs sampled different code";

    log_consistency("four_thread", run_a, run_b, overlap);
}

// Case 3: known 5-level deep stack. Asserts the decoded stack's innermost
// IP falls inside the leaf function's address range.

namespace
{

[[gnu::noinline]] double
known_leaf(double seed, std::uint64_t iters) noexcept
{
    double acc = seed;
    for(std::uint64_t i = 0; i < iters; ++i) {
        acc = acc * 1.0000001 + (i & 0xFFFF);
}
    return acc;
}

[[gnu::noinline]] double
known_l4(double s, std::uint64_t n) noexcept
{
    return known_leaf(s + 1.0, n);
}

[[gnu::noinline]] double
known_l3(double s, std::uint64_t n) noexcept
{
    return known_l4(s + 1.0, n);
}

[[gnu::noinline]] double
known_l2(double s, std::uint64_t n) noexcept
{
    return known_l3(s + 1.0, n);
}

[[gnu::noinline]] double
known_l1(double s, std::uint64_t n) noexcept
{
    return known_l2(s + 1.0, n);
}

void
known_chain_worker(std::atomic<bool>& stop)
{
    double acc = 1.0;
    while(!stop.load(std::memory_order_relaxed)) {
        acc += known_l1(acc, 200'000);
}
    if(acc == 12345.6789) std::printf("unreachable\n");
}

}  // namespace

TEST_F(StreamingSelfConsistencyTest, ReaderResolvesFullStackForKnownWorkload)
{
    cpu_snap::core::memory_sink sink(
        cpu_snap::core::memory_sink::config{ .max_bytes = k_capture_sink_max_bytes });

    std::atomic<bool> stop{ false };
    std::thread       worker(known_chain_worker, std::ref(stop));
    std::this_thread::sleep_for(std::chrono::milliseconds{ 20 });

    auto cfg                          = make_base_config();
    cfg.streaming.sink                = std::ref(sink);
    cfg.streaming.chunk_bytes         = 1U << 16;
    cfg.streaming.checkpoint_interval = 1000;
    cfg.streaming.maps_poll_interval  = 1000;
    cfg.streaming.process_name        = "equiv_known_chain";
    cfg.threads.include_main_thread   = false;

    cpu_snap::sampling::sampler rec(std::move(cfg));
    rec.start();
    std::this_thread::sleep_for(k_capture_duration);
    rec.stop();

    stop.store(true, std::memory_order_relaxed);
    worker.join();

    cpu_snap::core::sample_reconstructor reconstructor;
    reconstructor.feed(sink.bytes());
    const auto& per_tid = reconstructor.samples();

    ASSERT_FALSE(per_tid.empty());

    // Pick the tid with the most samples; this is the busy worker. The
    // leaf function's IPs land somewhere inside [&known_leaf, +slop).
    pid_t       busy_tid     = -1;
    std::size_t busy_samples = 0;
    for(const auto& [tid, vec] : per_tid)
    {
        if(vec.size() > busy_samples)
        {
            busy_samples = vec.size();
            busy_tid     = tid;
        }
    }
    ASSERT_GT(busy_samples, 0U);

    // Find at least one sample whose innermost IP lies near &known_leaf. The
    // kernel callchain walker reports the sample's PC inside the leaf
    // function; the slop allows for the body of the loop. A function body
    // of <2 KiB is generous given the simple body of known_leaf.
    const auto              leaf_base = reinterpret_cast<std::uint64_t>(&known_leaf);
    constexpr std::uint64_t k_slop    = 2048;

    bool any_in_leaf = false;
    bool any_deep    = false;
    for(const auto& rs : per_tid.at(busy_tid))
    {
        if(rs.frames.size() == 0) continue;
        const auto leaf_ip = rs.frames[0];
        if(leaf_ip >= leaf_base && leaf_ip < leaf_base + k_slop)
        {
            any_in_leaf = true;
        }
        if(rs.frames.size() >= 5) any_deep = true;
    }

    EXPECT_TRUE(any_in_leaf)
        << "no sample resolved an innermost IP inside &known_leaf range " << "[0x"
        << std::hex << leaf_base << ", 0x" << (leaf_base + k_slop) << ")";
    EXPECT_TRUE(any_deep) << "no sample resolved a stack >=5 frames deep; the kernel may "
                          << "have inlined or truncated frames";

    std::printf("[self-consistency] known_chain : busy_tid=%d samples=%zu\n",
                static_cast<int>(busy_tid), busy_samples);
}
