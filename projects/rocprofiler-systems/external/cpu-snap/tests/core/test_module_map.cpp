// Unit tests for cpu_snap::core::module_map.

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <utility>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/records.hpp"

namespace
{

using cpu_snap::core::maps_delta_record;
using cpu_snap::core::maps_init_record;
using cpu_snap::core::module_info;
using cpu_snap::core::module_map;

module_info
make_module(std::uint64_t base_va, std::uint64_t size, std::string path)
{
    module_info mod;
    mod.base_va = base_va;
    mod.size    = size;
    mod.path    = std::move(path);
    return mod;
}

/**
 * Builds a map with:
 *   - module A [0x1000, 0x1100) and module B [0x2000, 0x2100), both
 *     loaded at t=100 via MAPS_INIT
 *   - A removed at t=200 (a MAPS_DELTA)
 *   - module C [0x3000, 0x3100) added at t=300 (a MAPS_DELTA)
 */
module_map
build_sample_map()
{
    module_map map;

    maps_init_record init;
    init.wall_ns = 100;
    init.modules = { make_module(0x1000, 0x100, "/lib/a.so"),
                     make_module(0x2000, 0x100, "/lib/b.so") };
    map.apply(init);

    maps_delta_record remove_a;
    remove_a.wall_ns   = 200;
    remove_a.operation = maps_delta_record::op::remove;
    remove_a.module    = make_module(0x1000, 0x100, "/lib/a.so");
    map.apply(remove_a);

    maps_delta_record add_c;
    add_c.wall_ns   = 300;
    add_c.operation = maps_delta_record::op::add;
    add_c.module    = make_module(0x3000, 0x100, "/lib/c.so");
    map.apply(add_c);

    return map;
}

}  // namespace

TEST(ModuleMap, ResolvesModuleLoadedAtInit)
{
    const module_map map = build_sample_map();
    const auto result = map.resolve(0x1050, 150);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->path, "/lib/a.so");
}

TEST(ModuleMap, NulloptBeforeInitTimestamp)
{
    const module_map map = build_sample_map();
    EXPECT_FALSE(map.resolve(0x1050, 50).has_value());
}

TEST(ModuleMap, UnloadedModuleUnavailableAfterUnload)
{
    const module_map map = build_sample_map();
    EXPECT_FALSE(map.resolve(0x1050, 250).has_value())
        << "module A was removed at t=200 and never reloaded";
}

TEST(ModuleMap, UnrelatedModuleUnaffectedByRemoval)
{
    const module_map map = build_sample_map();
    const auto result = map.resolve(0x2050, 250);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->path, "/lib/b.so");
}

TEST(ModuleMap, NulloptBeforeDeltaAddLoadTime)
{
    const module_map map = build_sample_map();
    EXPECT_FALSE(map.resolve(0x3050, 250).has_value())
        << "module C is not loaded until t=300";
}

TEST(ModuleMap, ResolvesModuleAddedPartwayThrough)
{
    const module_map map = build_sample_map();
    const auto result = map.resolve(0x3050, 350);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->path, "/lib/c.so");
}

TEST(ModuleMap, NulloptForIpInNoLoadedModule)
{
    const module_map map = build_sample_map();
    EXPECT_FALSE(map.resolve(0x9000, 150).has_value());
}

TEST(ModuleMap, ResolvedModuleCarriesBaseAndSize)
{
    const module_map map = build_sample_map();
    const auto result = map.resolve(0x2050, 150);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->base_va, 0x2000U);
    EXPECT_EQ(result->size, 0x100U);
}

TEST(ModuleMap, ResolvesIpInModuleEndingAtAddressSpaceTop)
{
    module_map       map;
    maps_init_record init;
    init.wall_ns = 100;
    init.modules = { make_module(UINT64_MAX - 0xF, 0x100, "/lib/top.so") };
    map.apply(init);

    EXPECT_TRUE(map.resolve(UINT64_MAX - 1, 150).has_value());
    EXPECT_FALSE(map.resolve(0x5, 150).has_value());
}

TEST(ModuleMap, RemoveMatchesPathAsWellAsBase)
{
    module_map       map;
    maps_init_record init;
    init.wall_ns = 100;
    init.modules = { make_module(0x1000, 0x100, "/lib/a.so") };
    map.apply(init);

    maps_delta_record remove_other;
    remove_other.wall_ns   = 200;
    remove_other.operation = maps_delta_record::op::remove;
    remove_other.module    = make_module(0x1000, 0x100, "/lib/other.so");
    map.apply(remove_other);

    const auto result = map.resolve(0x1050, 300);
    ASSERT_TRUE(result.has_value());
    EXPECT_EQ(result->path, "/lib/a.so");
}

TEST(ModuleMap, ResolvesEachOfManyModulesToItsOwn)
{
    module_map       map;
    maps_init_record init;
    init.wall_ns = 100;
    for(std::uint64_t i = 0; i < 1000; ++i)
    {
        init.modules.push_back(make_module(0x100000 + i * 0x2000, 0x1000,
                                           "/lib/m" + std::to_string(i) + ".so"));
    }
    map.apply(init);

    for(std::uint64_t i = 0; i < 1000; ++i)
    {
        const auto result = map.resolve(0x100000 + i * 0x2000 + 0xFFF, 150);
        ASSERT_TRUE(result.has_value());
        EXPECT_EQ(result->path, "/lib/m" + std::to_string(i) + ".so");
        EXPECT_FALSE(map.resolve(0x100000 + i * 0x2000 + 0x1000, 150).has_value());
    }
}
