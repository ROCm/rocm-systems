// Unit tests for cpu_snap::sampling::dl_module_watcher.

#include <algorithm>
#include <dlfcn.h>
#include <time.h>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <gtest/gtest.h>

#include <sstream>
#include <string>
#include <unistd.h>
#include <utility>
#include <vector>

#include "cpu_snap/sampling/dl_module_watcher.hpp"
#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"
#include "cpu_snap/core/records.hpp"

namespace
{

using cpu_snap::core::maps_delta_record;
using cpu_snap::core::maps_init_record;
using cpu_snap::core::module_info;
using cpu_snap::sampling::dl_module_watcher;

dl_module_watcher::iterator_fn
make_static_iterator(std::vector<dl_module_watcher::injected_module> modules)
{
    return [mods = std::move(modules)](auto emit) {
        for(const auto& m : mods) {
            emit(m);
}
    };
}

[[nodiscard]] const module_info*
find_by_path(const std::vector<module_info>& v, const std::string& path)
{
    for(const auto& m : v) {
        if(m.path == path) return &m;
}
    return nullptr;
}

[[nodiscard]] const maps_delta_record*
find_delta_by_path(const std::vector<maps_delta_record>& v, const std::string& path)
{
    for(const auto& d : v) {
        if(d.module.path == path) return &d;
}
    return nullptr;
}

}  // namespace

TEST(MapsWatcher, RealWalkListsLoadedModules)
{
    dl_module_watcher           w;
    const maps_init_record snap = w.snapshot();

    ASSERT_FALSE(snap.modules.empty())
        << "dl_iterate_phdr should expose at least the main exe";

    bool any_with_build_id = false;
    bool any_with_size     = false;
    bool saw_libc          = false;
    bool saw_libstdcpp     = false;
    for(const auto& m : snap.modules)
    {
        EXPECT_FALSE(m.path.empty()) << "every module entry must carry a path";
        if(m.size != 0) any_with_size = true;
        if(!m.build_id.empty()) any_with_build_id = true;
        if(m.path.find("libc.so") != std::string::npos) saw_libc = true;
        if(m.path.find("libstdc++") != std::string::npos) saw_libstdcpp = true;
    }
    EXPECT_TRUE(any_with_size)
        << "at least one PT_LOAD segment should report a non-zero size";
    EXPECT_TRUE(any_with_build_id) << "libc on Linux is always built with --build-id";
    EXPECT_TRUE(saw_libc) << "libc should be among the loaded modules";
    EXPECT_TRUE(saw_libstdcpp) << "libstdc++ should be among the loaded modules";

    EXPECT_EQ(w.known_module_count(), snap.modules.size());
    EXPECT_GT(snap.wall_ns, 0U);
}

TEST(MapsWatcher, InjectedSnapshotThenUnchangedPollIsEmpty)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({
        { 0x1000, 0x100, 0, "/lib/a.so" },
        { 0x2000, 0x200, 0, "/lib/b.so" },
    }));

    const maps_init_record snap = w.snapshot();
    EXPECT_EQ(snap.modules.size(), 2U);
    EXPECT_EQ(w.known_module_count(), 2U);

    const auto deltas = w.poll();
    EXPECT_TRUE(deltas.empty()) << "second poll with same set must emit no deltas";
}

TEST(MapsWatcher, PollEmitsAddOnNewModule)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({
        { 0x1000, 0x100, 0, "/lib/a.so" },
        { 0x2000, 0x200, 0, "/lib/b.so" },
    }));
    (void) w.snapshot();

    w.set_iterator_for_tests(make_static_iterator({
        { 0x1000, 0x100, 0, "/lib/a.so" },
        { 0x2000, 0x200, 0, "/lib/b.so" },
        { 0x3000, 0x300, 0, "/lib/c.so" },
    }));

    const auto deltas = w.poll();
    ASSERT_EQ(deltas.size(), 1U);
    EXPECT_EQ(deltas[0].operation, maps_delta_record::op::add);
    EXPECT_EQ(deltas[0].module.path, "/lib/c.so");
    EXPECT_EQ(deltas[0].module.base_va, 0x3000U);
    EXPECT_EQ(w.known_module_count(), 3U);
}

TEST(MapsWatcher, PollEmitsRemoveOnUnloadedModule)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({
        { 0x1000, 0x100, 0, "/lib/a.so" },
        { 0x2000, 0x200, 0, "/lib/b.so" },
        { 0x3000, 0x300, 0, "/lib/c.so" },
    }));
    (void) w.snapshot();

    w.set_iterator_for_tests(make_static_iterator({
        { 0x2000, 0x200, 0, "/lib/b.so" },
        { 0x3000, 0x300, 0, "/lib/c.so" },
    }));

    const auto deltas = w.poll();
    ASSERT_EQ(deltas.size(), 1U);
    EXPECT_EQ(deltas[0].operation, maps_delta_record::op::remove);
    EXPECT_EQ(deltas[0].module.path, "/lib/a.so");
    EXPECT_EQ(w.known_module_count(), 2U);
}

TEST(MapsWatcher, PollReportsSimultaneousAddAndRemove)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({
        { 0x1000, 0x100, 0, "/lib/a.so" },
        { 0x2000, 0x200, 0, "/lib/b.so" },
    }));
    (void) w.snapshot();

    w.set_iterator_for_tests(make_static_iterator({
        { 0x2000, 0x200, 0, "/lib/b.so" },
        { 0x3000, 0x300, 0, "/lib/c.so" },
    }));

    const auto deltas = w.poll();
    ASSERT_EQ(deltas.size(), 2U);

    const auto* removed = find_delta_by_path(deltas, "/lib/a.so");
    const auto* added   = find_delta_by_path(deltas, "/lib/c.so");
    ASSERT_NE(removed, nullptr);
    ASSERT_NE(added, nullptr);
    EXPECT_EQ(removed->operation, maps_delta_record::op::remove);
    EXPECT_EQ(added->operation, maps_delta_record::op::add);
    EXPECT_EQ(w.known_module_count(), 2U);
}

TEST(MapsWatcher, BuildIdForTestBinaryIsSha1SizedWhenPresent)
{
    dl_module_watcher           w;
    const maps_init_record snap = w.snapshot();

    char       exe_buf[4096] = { 0 };
    const auto n             = ::readlink("/proc/self/exe", exe_buf, sizeof(exe_buf) - 1);
    ASSERT_GT(n, 0);
    const std::string exe(exe_buf, static_cast<std::size_t>(n));

    const module_info* self = find_by_path(snap.modules, exe);
    ASSERT_NE(self, nullptr) << "main executable should appear in the module list";

    if(self->build_id.empty())
    {
        GTEST_SKIP() << "test binary built without --build-id";
    }
    EXPECT_EQ(self->build_id.size(), 20U)
        << "default GNU build-id is a 20-byte SHA1 digest";
}

namespace
{

struct exec_mapping
{
    std::uint64_t start  = 0;
    std::uint64_t end    = 0;
    std::uint64_t offset = 0;
    std::string   path;
};

[[nodiscard]] std::vector<exec_mapping>
read_exec_mappings()
{
    std::vector<exec_mapping> out;
    std::ifstream             maps("/proc/self/maps");
    std::string               line;
    while(std::getline(maps, line))
    {
        std::istringstream in(line);
        std::string        range;
        std::string        perms;
        std::string        dev;
        std::string        path;
        std::uint64_t      offset = 0;
        std::uint64_t      inode  = 0;
        in >> range >> perms >> std::hex >> offset >> dev >> std::dec >> inode >> path;
        if(perms.size() < 3 || perms[2] != 'x' || path.empty() || path[0] != '/') continue;
        exec_mapping m;
        m.start  = std::stoull(range.substr(0, range.find('-')), nullptr, 16);
        m.end    = std::stoull(range.substr(range.find('-') + 1), nullptr, 16);
        m.offset = offset;
        m.path   = path;
        out.push_back(std::move(m));
    }
    return out;
}

}  // namespace

TEST(MapsWatcher, RealModuleRangesMatchExecutableMappings)
{
    dl_module_watcher      w;
    const maps_init_record snap     = w.snapshot();
    const auto             mappings = read_exec_mappings();

    std::size_t checked = 0;
    for(const auto& m : snap.modules)
    {
        if(m.path.empty() || m.path[0] != '/' || m.size == 0) continue;
        const std::string real_path = std::filesystem::canonical(m.path).string();
        const auto it = std::find_if(mappings.begin(), mappings.end(),
                                     [&](const exec_mapping& em) {
                                         return em.path == real_path && m.base_va >= em.start &&
                                                m.base_va < em.end;
                                     });
        ASSERT_NE(it, mappings.end()) << m.path << " base_va=0x" << std::hex << m.base_va;
        EXPECT_LE(m.base_va + m.size, it->end) << m.path;
        EXPECT_EQ(m.file_offset, it->offset + (m.base_va - it->start)) << m.path;
        ++checked;
    }
    EXPECT_GT(checked, 2U);
}

namespace
{

[[nodiscard]] std::uint64_t
monotonic_ns()
{
    timespec ts{};
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<std::uint64_t>(ts.tv_sec) * 1'000'000'000ULL +
           static_cast<std::uint64_t>(ts.tv_nsec);
}

}  // namespace

TEST(MapsWatcher, SnapshotTimestampIsInSampleClock)
{
    dl_module_watcher w;
    const auto        before = monotonic_ns();
    const auto        snap   = w.snapshot();
    const auto        after  = monotonic_ns();
    EXPECT_GE(snap.wall_ns, before);
    EXPECT_LE(snap.wall_ns, after);
}

TEST(MapsWatcher, PollWithoutLoaderActivitySkipsFullWalk)
{
    dl_module_watcher w;
    (void) w.snapshot();
    ASSERT_EQ(w.full_walk_count(), 1U);

    for(int i = 0; i < 3; ++i)
    {
        EXPECT_TRUE(w.poll().empty());
    }
    EXPECT_EQ(w.full_walk_count(), 1U);
}

TEST(MapsWatcher, PollReportsRealDlopenAndDlclose)
{
    dl_module_watcher w;
    const auto        snap = w.snapshot();

    const char* const candidates[] = { "libBrokenLocale.so.1", "libnsl.so.1", "libanl.so.1" };
    void*             handle       = nullptr;
    std::string       loaded_name;
    for(const char* name : candidates)
    {
        const bool already_loaded = std::any_of(
            snap.modules.begin(), snap.modules.end(), [&](const module_info& m) {
                return m.path.find(name) != std::string::npos;
            });
        if(already_loaded) continue;
        handle = dlopen(name, RTLD_NOW);
        if(handle != nullptr)
        {
            loaded_name = name;
            break;
        }
    }
    if(handle == nullptr) GTEST_SKIP() << "no unloaded candidate library available";

    const auto added = w.poll();
    const auto add_it = std::find_if(added.begin(), added.end(), [&](const maps_delta_record& d) {
        return d.operation == maps_delta_record::op::add &&
               d.module.path.find(loaded_name) != std::string::npos;
    });
    EXPECT_NE(add_it, added.end());
    if(add_it != added.end())
    {
        EXPECT_EQ(add_it->wall_ns, snap.wall_ns);
    }

    ASSERT_EQ(dlclose(handle), 0);
    const auto removed = w.poll();
    EXPECT_TRUE(std::any_of(removed.begin(), removed.end(), [&](const maps_delta_record& d) {
        return d.operation == maps_delta_record::op::remove &&
               d.module.path.find(loaded_name) != std::string::npos;
    }));
}

TEST(MapsWatcher, AddedModuleIsAttributableFromPreviousObservation)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({ { 0x1000, 0x100, 0, "/lib/a.so" } }));
    const auto snap = w.snapshot();

    w.set_iterator_for_tests(make_static_iterator({ { 0x1000, 0x100, 0, "/lib/a.so" },
                                                    { 0x2000, 0x100, 0, "/lib/b.so" } }));
    const auto deltas = w.poll();
    ASSERT_EQ(deltas.size(), 1U);
    EXPECT_EQ(deltas[0].wall_ns, snap.wall_ns);
}

TEST(MapsWatcher, ReusedBaseAddressAddIsStampedAfterRemove)
{
    dl_module_watcher w;
    w.set_iterator_for_tests(make_static_iterator({ { 0x1000, 0x100, 0, "/lib/a.so" } }));
    (void) w.snapshot();

    w.set_iterator_for_tests(make_static_iterator({ { 0x1000, 0x100, 0, "/lib/b.so" } }));
    const auto deltas = w.poll();
    ASSERT_EQ(deltas.size(), 2U);
    EXPECT_EQ(deltas[0].operation, maps_delta_record::op::remove);
    EXPECT_EQ(deltas[1].operation, maps_delta_record::op::add);
    EXPECT_GE(deltas[1].wall_ns, deltas[0].wall_ns);
}

[[gnu::noinline]] int
watcher_anchor_fn(int value) noexcept
{
    return value * 7;
}

TEST(MapsWatcher, SnapshotModulesResolveOwnFunctionThroughModuleMap)
{
    dl_module_watcher      w;
    const maps_init_record snap = w.snapshot();

    cpu_snap::core::module_map modules;
    modules.apply(snap);

    auto          fp = &watcher_anchor_fn;
    std::uint64_t ip = 0;
    std::memcpy(&ip, &fp, sizeof(ip));

    const auto ref = modules.resolve(ip, snap.wall_ns + 1);
    ASSERT_TRUE(ref.has_value());

    cpu_snap::processing::symbols::symbolizer sym(modules);
    sym.preload_at({ { ip, snap.wall_ns + 1 } });
    EXPECT_NE(std::string{ sym.resolve(ip) }.find("watcher_anchor_fn"), std::string::npos);
}

TEST(MapsWatcher, ResolvedModuleMatchesProcMapsForLibcAndTestBinary)
{
    dl_module_watcher      w;
    const maps_init_record snap = w.snapshot();

    cpu_snap::core::module_map modules;
    modules.apply(snap);
    const auto mappings = read_exec_mappings();

    const auto libc_fn = &::getpid;
    const auto own_fn  = &watcher_anchor_fn;
    const std::uint64_t addrs[] = { reinterpret_cast<std::uint64_t>(libc_fn),
                                    reinterpret_cast<std::uint64_t>(own_fn) };

    for(const auto addr : addrs)
    {
        const auto ref = modules.resolve(addr, snap.wall_ns + 1);
        ASSERT_TRUE(ref.has_value()) << std::hex << addr;

        const auto it = std::find_if(mappings.begin(), mappings.end(),
                                     [&](const exec_mapping& em) {
                                         return addr >= em.start && addr < em.end;
                                     });
        ASSERT_NE(it, mappings.end()) << std::hex << addr;

        EXPECT_EQ(std::filesystem::canonical(ref->path), std::filesystem::canonical(it->path));
        EXPECT_EQ(ref->file_offset + (addr - ref->base_va), it->offset + (addr - it->start))
            << ref->path;
    }
}
