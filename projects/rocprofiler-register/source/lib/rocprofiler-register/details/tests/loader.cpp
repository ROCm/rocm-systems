// Copyright (c) 2024 Advanced Micro Devices, Inc.
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

#include "details/platform/loader.hpp"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

namespace
{
#if defined(_WIN32)
constexpr const char* kSelfLibName        = "kernel32.dll";
constexpr const char* kSymbolInSelf       = "GetCurrentProcessId";
constexpr const char* kBogusLibName       = "definitely-not-a-real-library-xyz";
#else
constexpr const char* kSelfLibName        = "libc.so.6";
constexpr const char* kSymbolInSelf       = "getpid";
constexpr const char* kBogusLibName       = "libdefinitely-not-real-xyz.so";
#endif
}  // namespace

TEST(loader, module_open_returns_nonnull_for_known_lib)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open(kSelfLibName);
    ASSERT_NE(handle, nullptr);
    module_close(handle);
}

TEST(loader, module_open_returns_null_for_bogus_lib)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open(kBogusLibName);
    EXPECT_EQ(handle, nullptr);
}

TEST(loader, module_open_already_loaded_finds_loaded_lib)
{
    using namespace rocprofiler_register::platform;
    // Make sure it's loaded.
    auto* loaded = module_open(kSelfLibName);
    ASSERT_NE(loaded, nullptr);

    auto* probe = module_open_already_loaded(kSelfLibName);
    EXPECT_NE(probe, nullptr);

    module_close(loaded);
}

TEST(loader, module_sym_resolves_known_symbol)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open(kSelfLibName);
    ASSERT_NE(handle, nullptr);

    auto* sym = module_sym(handle, kSymbolInSelf);
    EXPECT_NE(sym, nullptr);

    module_close(handle);
}

TEST(loader, module_sym_returns_null_for_unknown_symbol)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open(kSelfLibName);
    ASSERT_NE(handle, nullptr);

    auto* sym = module_sym(handle, "definitely_not_a_real_symbol_xyz_12345");
    EXPECT_EQ(sym, nullptr);

    module_close(handle);
}

TEST(loader, module_sym_default_resolves_known_symbol)
{
    using namespace rocprofiler_register::platform;
    auto* sym = module_sym_default(kSymbolInSelf);
    EXPECT_NE(sym, nullptr);
}

TEST(loader, module_path_returns_nonempty_for_loaded_lib)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open(kSelfLibName);
    ASSERT_NE(handle, nullptr);

    auto path = module_path(handle);
    EXPECT_FALSE(path.empty());

    module_close(handle);
}

TEST(loader, get_segment_addresses_returns_nonempty_for_self)
{
    using namespace rocprofiler_register::platform;
    auto modules = get_segment_addresses();
    EXPECT_GT(modules.size(), 0u);
    for(const auto& mod : modules)
    {
        EXPECT_FALSE(mod.filepath.empty());
        EXPECT_GT(mod.ranges.size(), 0u);
        for(const auto& range : mod.ranges)
        {
            EXPECT_LE(range.start, range.last);
        }
    }
}

TEST(loader, module_open_with_fallback_resolves_known_lib)
{
    using namespace rocprofiler_register::platform;
    auto* handle = module_open_with_fallback(kSelfLibName);
    EXPECT_NE(handle, nullptr);
    if(handle != nullptr) module_close(handle);
}
