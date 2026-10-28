// Unit tests for cpu_snap::processing::symbols::sample_postprocessor.
// Pure dependency-injection tests: no kernel syscalls, no perf perms.

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <sched.h>
#include <initializer_list>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <unordered_map>
#include <vector>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/raw_sample.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"
#include "cpu_snap/processing/symbols/sample_postprocessor.hpp"

namespace
{

cpu_snap::core::raw_sample
make_raw(pid_t tid, std::uint64_t ts, std::initializer_list<std::uint64_t> frames)
{
    cpu_snap::core::raw_sample rs{};
    rs.tid          = tid;
    rs.timestamp_ns = ts;
    rs.ip           = frames.size() != 0 ? *frames.begin() : 0;
    rs.frames.assign(frames.begin(), frames.end());
    return rs;
}

}  // namespace

TEST(SamplePostprocessorTest, EmptyInputYieldsNoOutput)
{
    cpu_snap::processing::symbols::sample_postprocessor proc({});
    std::size_t                          n = 0;
    proc.process(std::span<const cpu_snap::core::raw_sample>{},
                 [&n](const cpu_snap::processing::symbols::processed_sample&) { ++n; });
    EXPECT_EQ(n, 0U);
    EXPECT_EQ(proc.delivered_samples(), 0U);
}

TEST(SamplePostprocessorTest, PassesThroughWithoutPredicates)
{
    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(100, 1, { 0x1000, 0x2000 }));
    raws.push_back(make_raw(100, 2, { 0x3000 }));

    cpu_snap::processing::symbols::sample_postprocessor proc({});
    std::vector<std::size_t>             sizes;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&sizes](const cpu_snap::processing::symbols::processed_sample& s) {
                     sizes.push_back(s.frames.size());
                 });
    EXPECT_EQ(proc.delivered_samples(), 2U);
    ASSERT_EQ(sizes.size(), 2U);
    EXPECT_EQ(sizes[0], 2U);
    EXPECT_EQ(sizes[1], 1U);
}

TEST(SamplePostprocessorTest, ResolverCalledOncePerUniqueIp)
{
    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(1, 1, { 0xAA, 0xBB }));
    raws.push_back(make_raw(1, 2, { 0xBB, 0xCC }));
    raws.push_back(make_raw(1, 3, { 0xAA, 0xCC, 0xDD }));

    std::size_t                                 n_calls = 0;
    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.resolver = [&n_calls](std::uint64_t ip) {
        ++n_calls;
        return "fn_" + std::to_string(ip);
    };

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [](const cpu_snap::processing::symbols::processed_sample&) {});
    EXPECT_EQ(n_calls, 4U);  // AA, BB, CC, DD - one each
    EXPECT_EQ(proc.unique_symbols(), 4U);
}

TEST(SamplePostprocessorTest, FramePredicateDropsMatchingFrames)
{
    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(7, 1, { 0x10, 0x20, 0x30 }));

    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.resolver = [](std::uint64_t ip) {
        return ip == 0x20 ? std::string{} : "fn_" + std::to_string(ip);
    };
    // Drop frames with empty resolved names.
    cfg.frame_predicate = [](std::uint64_t, std::string_view name) {
        return !name.empty();
    };

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));
    std::size_t                          kept_frames = 0;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&kept_frames](const cpu_snap::processing::symbols::processed_sample& s) {
                     kept_frames = s.frames.size();
                 });
    EXPECT_EQ(proc.delivered_samples(), 1U);
    EXPECT_EQ(proc.dropped_frames(), 1U);
    EXPECT_EQ(kept_frames, 2U);
}

TEST(SamplePostprocessorTest, FramePredicateDropsHexAndMangledNames)
{
    // Simulate the example's name_looks_resolved policy: drop empty,
    // "0x...", and "_Z..." names.
    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(7, 1, { 0x10, 0x20, 0x30, 0x40, 0x50 }));

    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.resolver = [](std::uint64_t ip) -> std::string {
        switch(ip)
        {
            case 0x10: return "good_fn";
            case 0x20: return "0x1234abcd";  // hex fallback
            case 0x30: return "";            // unresolved
            case 0x40: return "_Z7mangled";  // failed demangle
            case 0x50: return "another_good_fn";
            default: return "";
        }
    };
    cfg.frame_predicate = [](std::uint64_t, std::string_view name) {
        if(name.empty()) return false;
        if(name.starts_with("0x")) return false;
        if(name.starts_with("_Z")) return false;
        return true;
    };

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));
    std::vector<std::string>             kept_names;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&kept_names](const cpu_snap::processing::symbols::processed_sample& s) {
                     for(const auto& f : s.frames) {
                         kept_names.emplace_back(f.resolved);
}
                 });
    EXPECT_EQ(proc.dropped_frames(), 3U);
    ASSERT_EQ(kept_names.size(), 2U);
    EXPECT_EQ(kept_names[0], "good_fn");
    EXPECT_EQ(kept_names[1], "another_good_fn");
}

TEST(SamplePostprocessorTest, ClearCacheForcesReresolve)
{
    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(1, 1, { 0xAB, 0xCD }));

    std::size_t                                 n_calls = 0;
    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.resolver = [&n_calls](std::uint64_t) {
        ++n_calls;
        return std::string{ "x" };
    };

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [](const cpu_snap::processing::symbols::processed_sample&) {});
    EXPECT_EQ(n_calls, 2U);

    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [](const cpu_snap::processing::symbols::processed_sample&) {});
    EXPECT_EQ(n_calls, 2U);  // both cached

    proc.clear_cache();
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [](const cpu_snap::processing::symbols::processed_sample&) {});
    EXPECT_EQ(n_calls, 4U);  // re-resolved
}

TEST(SamplePostprocessorTest, ProcessAllIteratesEveryTid)
{
    std::unordered_map<pid_t, std::vector<cpu_snap::core::raw_sample>> by_tid;
    by_tid[10].push_back(make_raw(10, 1, { 0x1 }));
    by_tid[10].push_back(make_raw(10, 2, { 0x2 }));
    by_tid[20].push_back(make_raw(20, 1, { 0x3 }));

    cpu_snap::processing::symbols::sample_postprocessor proc({});
    std::size_t                          n = 0;
    proc.process_all(by_tid, [&n](const cpu_snap::processing::symbols::processed_sample&) { ++n; });
    EXPECT_EQ(n, 3U);
    EXPECT_EQ(proc.delivered_samples(), 3U);
}

namespace
{

[[gnu::noinline]] int
test_local_anchor(int xx) noexcept
{
    return xx + 7;
}

std::uint64_t
ip_of_local()
{
    auto          fp  = &test_local_anchor;
    std::uint64_t out = 0;
    std::memcpy(&out, &fp, sizeof(out));
    return out;
}

std::uint64_t
ip_of_malloc()
{
    auto          fp  = &std::malloc;
    std::uint64_t out = 0;
    std::memcpy(&out, &fp, sizeof(out));
    return out;
}

}  // namespace

TEST(SamplePostprocessorTest, DropLibraryFramesStripsLibcKeepsMainExe)
{
    const std::uint64_t local_ip = ip_of_local();
    const std::uint64_t libc_ip  = ip_of_malloc();

    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(42, 1, { local_ip, libc_ip, local_ip, libc_ip }));

    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.drop_library_frames = true;

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));

    std::size_t                delivered_frames = 0;
    std::vector<std::uint64_t> kept_ips;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&](const cpu_snap::processing::symbols::processed_sample& sm) {
                     ++delivered_frames;
                     for(const auto& fr : sm.frames)
                     {
                         kept_ips.push_back(fr.ip);
                     }
                 });

    EXPECT_EQ(delivered_frames, 1U);
    EXPECT_EQ(kept_ips.size(), 2U);
    EXPECT_EQ(proc.dropped_frames(), 2U);
    for(const std::uint64_t ip : kept_ips)
    {
        EXPECT_EQ(ip, local_ip);
    }
}

TEST(SamplePostprocessorTest, DropLibraryFramesComposesWithUserPredicate)
{
    const std::uint64_t local_ip = ip_of_local();
    const std::uint64_t libc_ip  = ip_of_malloc();

    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(42, 1, { local_ip, libc_ip }));

    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.drop_library_frames = true;
    cfg.frame_predicate     = [](std::uint64_t, std::string_view) { return false; };

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));
    std::size_t                    delivered_frames = 0;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&](const cpu_snap::processing::symbols::processed_sample& sm) {
                     delivered_frames += sm.frames.size();
                 });
    EXPECT_EQ(delivered_frames, 0U);
    EXPECT_EQ(proc.dropped_frames(), 2U);
}

// Offline variant of DropLibraryFramesStripsLibcKeepsMainExe: the library
// filter resolves through a synthetic core::module_map instead of a live
// /proc/self/maps read.
TEST(SamplePostprocessorTest, DropLibraryFramesViaModuleMapStripsLibcKeepsMainExe)
{
    const std::uint64_t local_ip = ip_of_local();
    const std::uint64_t libc_ip  = ip_of_malloc();

    cpu_snap::processing::symbols::proc_maps live_maps;
    const auto* local_mapping = live_maps.find(local_ip);
    const auto* libc_mapping  = live_maps.find(libc_ip);
    ASSERT_NE(local_mapping, nullptr);
    ASSERT_NE(libc_mapping, nullptr);

    const auto to_module_info = [](const cpu_snap::processing::symbols::vm_mapping& m) {
        cpu_snap::core::module_info info;
        info.base_va     = m.start;
        info.size        = m.end - m.start;
        info.file_offset = m.file_offset;
        info.path        = m.path;
        return info;
    };

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules = { to_module_info(*local_mapping), to_module_info(*libc_mapping) };
    cpu_snap::core::module_map modules;
    modules.apply(init);

    std::vector<cpu_snap::core::raw_sample> raws;
    raws.push_back(make_raw(42, 1, { local_ip, libc_ip, local_ip, libc_ip }));

    cpu_snap::processing::symbols::sample_postprocessor_config cfg;
    cfg.drop_library_frames          = true;
    cfg.library_filter_modules       = &modules;
    cfg.library_filter_timestamp_ns  = 150;
    cfg.main_exe_path                = local_mapping->path;

    cpu_snap::processing::symbols::sample_postprocessor proc(std::move(cfg));

    std::size_t                delivered_frames = 0;
    std::vector<std::uint64_t> kept_ips;
    proc.process(std::span<const cpu_snap::core::raw_sample>(raws),
                 [&](const cpu_snap::processing::symbols::processed_sample& sm) {
                     ++delivered_frames;
                     for(const auto& fr : sm.frames)
                     {
                         kept_ips.push_back(fr.ip);
                     }
                 });

    EXPECT_EQ(delivered_frames, 1U);
    EXPECT_EQ(kept_ips.size(), 2U);
    EXPECT_EQ(proc.dropped_frames(), 2U);
    for(const std::uint64_t ip : kept_ips)
    {
        EXPECT_EQ(ip, local_ip);
    }
}
