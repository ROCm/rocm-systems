// Proves cpu_snap::processing::symbols::symbolizer can resolve addresses using ONLY a
// core::module_map -- the capability offline / cross-process
// symbolization needs (measured broken before this change: 0/18 frames
// resolved when symbolizing a capture from a different process, because
// the old resolution path always consulted the symbolizing process's OWN
// /proc/self/maps).

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"
#include "cpu_snap/processing/symbols/symbolizer.hpp"

namespace
{

using cpu_snap::processing::symbols::symbolizer;

[[gnu::noinline]] int
offline_anchor_fn(int value) noexcept
{
    return value + 3;
}

std::uint64_t
ip_of_offline_anchor()
{
    auto          fp  = &offline_anchor_fn;
    std::uint64_t out = 0;
    std::memcpy(&out, &fp, sizeof(out));
    return out;
}

/**
 * Ground-truth base/size/file_offset/path of the mapping covering @p ip,
 * read via the live-process path purely to seed realistic fixture data --
 * exactly what a real MAPS_INIT record would have captured. The
 * resolution path under test (symbolizer constructed from a module_map)
 * never touches this live proc_maps instance.
 */
cpu_snap::core::module_info
module_info_for(std::uint64_t ip)
{
    cpu_snap::processing::symbols::proc_maps live_maps;
    const auto*                  mapping = live_maps.find(ip);
    if(mapping == nullptr) return {};

    cpu_snap::core::module_info info;
    info.base_va     = mapping->start;
    info.size        = mapping->end - mapping->start;
    info.file_offset = mapping->file_offset;
    info.path        = mapping->path;
    return info;
}

}  // namespace

TEST(ModuleMapSymbolization, ResolvesKnownAddressFromModuleMapOnly)
{
    const std::uint64_t ip   = ip_of_offline_anchor();
    const auto           info = module_info_for(ip);
    ASSERT_FALSE(info.path.empty());

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules.push_back(info);

    cpu_snap::core::module_map modules;
    modules.apply(init);

    symbolizer sym(modules);
    sym.preload_at({ { ip, 150 } });

    const std::string resolved{ sym.resolve(ip) };
    EXPECT_NE(resolved.find("offline_anchor_fn"), std::string::npos) << resolved;
}

// Simulates the realistic cross-process failure mode: the recording
// process's ASLR slide differs from the symbolizing process's, so the ip
// recorded on the wire does not appear anywhere in the symbolizing
// process's OWN /proc/self/maps. Only a module_map carrying the module's
// actual load address from the recording session can resolve it -- this
// is the exact "0/18 resolved" scenario from the plan doc, reproduced
// with a synthetic address-space shift instead of a second process.
TEST(ModuleMapSymbolization, ResolvesAddressAtDifferentBaseThanLiveProcessMaps)
{
    constexpr std::uint64_t k_simulated_aslr_shift = 0x1000000;

    const std::uint64_t real_ip   = ip_of_offline_anchor();
    const auto           real_info = module_info_for(real_ip);
    ASSERT_FALSE(real_info.path.empty());

    cpu_snap::core::module_info shifted = real_info;
    shifted.base_va += k_simulated_aslr_shift;
    const std::uint64_t simulated_ip = real_ip + k_simulated_aslr_shift;

    // Sanity: the simulated ip must NOT appear in this process's own live
    // maps -- otherwise this test would pass even without module_map.
    cpu_snap::processing::symbols::proc_maps live_maps;
    ASSERT_EQ(live_maps.find(simulated_ip), nullptr);

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules.push_back(shifted);

    cpu_snap::core::module_map modules;
    modules.apply(init);

    symbolizer sym(modules);
    sym.preload_at({ { simulated_ip, 150 } });

    const std::string resolved{ sym.resolve(simulated_ip) };
    EXPECT_NE(resolved.find("offline_anchor_fn"), std::string::npos) << resolved;
}

// Proves resolution is genuinely time-gated via module_map -- a
// capability /proc/self/maps has no equivalent for, since it only ever
// reflects "now". A symbolizer that fell back to live /proc/self/maps
// would resolve this ip regardless of the timestamp argument.
TEST(ModuleMapSymbolization, TimestampBeforeLoadFailsToResolve)
{
    const std::uint64_t ip   = ip_of_offline_anchor();
    const auto           info = module_info_for(ip);
    ASSERT_FALSE(info.path.empty());

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules.push_back(info);

    cpu_snap::core::module_map modules;
    modules.apply(init);

    symbolizer sym(modules);
    sym.preload_at({ { ip, 50 } });  // before the module's load_ts=100

    const std::string resolved{ sym.resolve(ip) };
    EXPECT_EQ(resolved.find("offline_anchor_fn"), std::string::npos) << resolved;
}

class ModuleMapVdsoSymbolization : public ::testing::TestWithParam<const char*>
{
};

TEST_P(ModuleMapVdsoSymbolization, VdsoIpResolvesToModulePlusOffset)
{
    constexpr std::uint64_t k_base   = 0x7f0000a11000;
    constexpr std::uint64_t k_offset = 0xc4b;

    cpu_snap::core::module_info info;
    info.base_va = k_base;
    info.size    = 0x2000;
    info.path    = GetParam();

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules.push_back(info);

    cpu_snap::core::module_map modules;
    modules.apply(init);

    symbolizer sym(modules);
    sym.preload_at({ { k_base + k_offset, 150 } });

    EXPECT_EQ(std::string{ sym.resolve(k_base + k_offset) },
              std::string{ GetParam() } + "+0xc4b");
    EXPECT_FALSE(sym.symbol_identity(k_base + k_offset, 150).has_value());
}

INSTANTIATE_TEST_SUITE_P(VdsoPaths, ModuleMapVdsoSymbolization,
                         ::testing::Values("linux-vdso.so.1", "[vdso]"));
