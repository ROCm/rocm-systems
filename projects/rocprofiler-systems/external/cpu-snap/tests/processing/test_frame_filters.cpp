// Unit tests for cpu_snap::processing::symbols::frame_filters and
// cpu_snap::processing::symbols::main_executable_path. Combines synthetic-proc_maps
// cases (via a faked maps loaded from a temp /proc/self/maps-style
// file would require re-architecting proc_maps; here we use a real
// proc_maps and pick IPs known to live in libc + the test exe).

#include <cstring>
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <string>
#include <string_view>
#include <vector>

#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/core/records.hpp"
#include "cpu_snap/processing/symbols/frame_filters.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"

namespace
{

[[gnu::noinline]] int
local_anchor_fn(int xx) noexcept
{
    return xx + 1;
}

std::uint64_t
ip_of_local_anchor()
{
    auto fp = &local_anchor_fn;
    std::uint64_t out = 0;
    std::memcpy(&out, static_cast<const void*>(&fp), sizeof(out));
    return out;
}

std::uint64_t
ip_of_libc_symbol()
{
    auto fp = &std::malloc;
    std::uint64_t out = 0;
    std::memcpy(&out, static_cast<const void*>(&fp), sizeof(out));
    return out;
}

/** Ground-truth module_info for the mapping covering @p ip, built from
 *  the live process purely to seed a realistic synthetic module_map. */
cpu_snap::core::module_info
module_info_for(std::uint64_t ip)
{
    cpu_snap::processing::symbols::proc_maps live_maps;
    const auto*                  mapping = live_maps.find(ip);
    cpu_snap::core::module_info  info;
    if(mapping == nullptr) return info;
    info.base_va     = mapping->start;
    info.size        = mapping->end - mapping->start;
    info.file_offset = mapping->file_offset;
    info.path        = mapping->path;
    return info;
}

}  // namespace

TEST(MakeMainExeOnlyFilterTest, KeepsMainExeFramesDropsLibc)
{
    cpu_snap::processing::symbols::proc_maps maps;
    auto                       pred = cpu_snap::processing::symbols::make_main_exe_only_filter(maps);

    const std::uint64_t local_ip = ip_of_local_anchor();
    const std::uint64_t libc_ip  = ip_of_libc_symbol();

    const cpu_snap::processing::symbols::vm_mapping* local_map = maps.find(local_ip);
    const cpu_snap::processing::symbols::vm_mapping* libc_map  = maps.find(libc_ip);

    ASSERT_NE(local_map, nullptr);
    ASSERT_NE(libc_map, nullptr);
    EXPECT_TRUE(local_map->is_main_exe);
    EXPECT_FALSE(libc_map->is_main_exe);

    EXPECT_TRUE(pred(local_ip, std::string_view{}));
    EXPECT_FALSE(pred(libc_ip, std::string_view{}));
}

TEST(MakeMainExeOnlyFilterTest, KeepsUnknownIpAsAnonymous)
{
    cpu_snap::processing::symbols::proc_maps maps;
    auto                       pred = cpu_snap::processing::symbols::make_main_exe_only_filter(maps);

    const std::uint64_t unknown_ip = 0xDEADBEEF1234ULL;
    EXPECT_EQ(maps.find(unknown_ip), nullptr);
    EXPECT_TRUE(pred(unknown_ip, std::string_view{}));
}

TEST(MakeLibraryExcludeFilterTest, DropsMatchingSubstring)
{
    cpu_snap::processing::symbols::proc_maps     maps;
    const std::vector<std::string> excluded{ "libc" };
    auto pred = cpu_snap::processing::symbols::make_library_exclude_filter(maps, excluded);

    const std::uint64_t local_ip = ip_of_local_anchor();
    const std::uint64_t libc_ip  = ip_of_libc_symbol();

    EXPECT_TRUE(pred(local_ip, std::string_view{}));
    EXPECT_FALSE(pred(libc_ip, std::string_view{}));
}

TEST(MakeLibraryExcludeFilterTest, KeepsWhenNoPatternMatches)
{
    cpu_snap::processing::symbols::proc_maps     maps;
    const std::vector<std::string> excluded{ "libthis_lib_does_not_exist" };
    auto pred = cpu_snap::processing::symbols::make_library_exclude_filter(maps, excluded);

    EXPECT_TRUE(pred(ip_of_local_anchor(), std::string_view{}));
    EXPECT_TRUE(pred(ip_of_libc_symbol(), std::string_view{}));
}

TEST(MakeLibraryExcludeFilterTest, EmptyPatternListKeepsAll)
{
    cpu_snap::processing::symbols::proc_maps maps;
    auto                       pred =
        cpu_snap::processing::symbols::make_library_exclude_filter(maps, {});

    EXPECT_TRUE(pred(ip_of_local_anchor(), std::string_view{}));
    EXPECT_TRUE(pred(ip_of_libc_symbol(), std::string_view{}));
}

TEST(MakeMainExeOnlyFilterTest, ModuleMapVariantKeepsMainExeDropsLibc)
{
    const std::uint64_t local_ip  = ip_of_local_anchor();
    const std::uint64_t libc_ip   = ip_of_libc_symbol();
    const auto           main_info = module_info_for(local_ip);
    const auto           libc_info = module_info_for(libc_ip);
    ASSERT_FALSE(main_info.path.empty());
    ASSERT_FALSE(libc_info.path.empty());

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules = { main_info, libc_info };
    cpu_snap::core::module_map modules;
    modules.apply(init);

    auto pred = cpu_snap::processing::symbols::make_main_exe_only_filter(modules, 150, main_info.path);

    EXPECT_TRUE(pred(local_ip, std::string_view{}));
    EXPECT_FALSE(pred(libc_ip, std::string_view{}));
}

TEST(MakeLibraryExcludeFilterTest, ModuleMapVariantDropsMatchingSubstring)
{
    const std::uint64_t local_ip  = ip_of_local_anchor();
    const std::uint64_t libc_ip   = ip_of_libc_symbol();
    const auto           main_info = module_info_for(local_ip);
    const auto           libc_info = module_info_for(libc_ip);
    ASSERT_FALSE(main_info.path.empty());
    ASSERT_FALSE(libc_info.path.empty());

    cpu_snap::core::maps_init_record init;
    init.wall_ns = 100;
    init.modules = { main_info, libc_info };
    cpu_snap::core::module_map modules;
    modules.apply(init);

    auto pred = cpu_snap::processing::symbols::make_library_exclude_filter(modules, 150, { "libc" });

    EXPECT_TRUE(pred(local_ip, std::string_view{}));
    EXPECT_FALSE(pred(libc_ip, std::string_view{}));
}

TEST(StdlibFilterTest, IsStdlibLabelMatchesOnlyTheFunctionsOwnName)
{
    using cpu_snap::processing::symbols::is_stdlib_label;
    struct label_case
    {
        std::string_view label;
        bool             stdlib;
    };
    const std::vector<label_case> cases{
        // the function itself is in the standard library
        { "std::vector<int, std::allocator<int> >::begin() [libhip.so]", true },
        { "std::function<void ()>::operator()() const [libhsa.so]", true },
        { "void* std::__invoke_impl<void*, int&>(int&) [libhip.so]", true },
        { "std::__atomic_base<unsigned int>::operator unsigned int() const [libhsa.so]", true },
        { "std::function<int ()>::operator bool() const [libhsa.so]", true },
        { "__gnu_cxx::__normal_iterator<int*, std::vector<int> >::operator*() [libhip.so]", true },
        // any frame of libstdc++, whatever its name
        { "operator new(unsigned long) [libstdc++.so.6.0.33]", true },
        { "0x7f00 [libstdc++.so.6.0.33]", true },
        // std:: only in parameters, template arguments or the return type
        { "rocr::Runtime::Copy(void*, std::vector<int>&) [libhsa.so]", false },
        { "hip::Foo<std::vector<int> >::bar() [libhip.so]", false },
        { "std::error_code llvm::sys::fs::remove(llvm::Twine const&) [libLLVM.so]", false },
        { "main [demo]", false },
        { "0x7f1234", false },
    };
    for(const auto& c : cases)
    {
        EXPECT_EQ(is_stdlib_label(c.label), c.stdlib) << c.label;
    }
}
