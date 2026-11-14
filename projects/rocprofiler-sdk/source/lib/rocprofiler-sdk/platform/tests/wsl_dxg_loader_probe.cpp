// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
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
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// The child half of wsl_dxg_loader.cpp: runs the SDK's librocdxg lookup in a
// process whose environment and mapped objects the parent chose, and prints
// what it found, one "key value" line per fact.
//
//   wsl-dxg-loader-probe CALLER [--preload PATH]... [--then-open DSO NAME]
//
// CALLER is a fixture exporting fixture_dlopen(). Every dlopen the lookup makes
// is issued from it, so a bare name is searched with CALLER's dynamic tags
// rather than this executable's. --preload maps PATH before the lookup runs: a
// libhsa-runtime64, or a thunk the HSA runtime already loaded. --then-open
// dlopens NAME from DSO once the lookup has closed its handle, as the HSA
// runtime would.

#include "lib/rocprofiler-sdk/platform/wsl/dxg_thunk.hpp"

#include <dlfcn.h>
#include <link.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

namespace
{
using fixture_dlopen_t = void* (*) (const char*, int);

[[noreturn]] void
fail(const char* what, const char* detail)
{
    std::fprintf(stderr, "wsl-dxg-loader-probe: %s: %s\n", what, detail != nullptr ? detail : "");
    std::exit(2);
}

fixture_dlopen_t
fixture_opener(const char* path)
{
    void* dso = ::dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if(dso == nullptr) fail("dlopen", ::dlerror());
    auto* opener = reinterpret_cast<fixture_dlopen_t>(::dlsym(dso, "fixture_dlopen"));
    if(opener == nullptr) fail("fixture_dlopen", path);
    return opener;
}

std::string
real_path(const char* path)
{
    char* real = ::realpath(path, nullptr);
    if(real == nullptr) return path;
    auto out = std::string{real};
    ::free(real);
    return out;
}

// Which of DT_RUNPATH and DT_RPATH the already mapped `path` carries.
const char*
search_tag(const char* path)
{
    void*     dso = ::dlopen(path, RTLD_NOW | RTLD_LOCAL | RTLD_NOLOAD);
    link_map* map = nullptr;
    if(dso == nullptr || ::dlinfo(dso, RTLD_DI_LINKMAP, &map) != 0) fail("dlinfo", path);

    auto runpath = false;
    auto rpath   = false;
    for(const ElfW(Dyn)* dyn = map->l_ld; dyn->d_tag != DT_NULL; ++dyn)
    {
        runpath = runpath || dyn->d_tag == DT_RUNPATH;
        rpath   = rpath || dyn->d_tag == DT_RPATH;
    }
    ::dlclose(dso);
    return runpath ? "RUNPATH" : (rpath ? "RPATH" : "NONE");
}

// One `key path` line per mapped object whose file name starts with `prefix`,
// or `key none`.
void
print_mapped(const char* key, std::string_view prefix, bool canonical)
{
    struct Search
    {
        std::string_view         prefix;
        std::vector<std::string> paths;
    } search{prefix, {}};

    ::dl_iterate_phdr(
        [](dl_phdr_info* info, size_t, void* data) -> int {
            auto*      out   = static_cast<Search*>(data);
            const auto path  = std::string_view{info->dlpi_name != nullptr ? info->dlpi_name : ""};
            const auto slash = path.find_last_of('/');
            const auto name  = slash == std::string_view::npos ? path : path.substr(slash + 1);
            if(name.rfind(out->prefix, 0) == 0) out->paths.emplace_back(path);
            return 0;
        },
        &search);

    if(search.paths.empty()) std::printf("%s none\n", key);
    for(const auto& path : search.paths)
        std::printf("%s %s\n", key, canonical ? real_path(path.c_str()).c_str() : path.c_str());
}
}  // namespace

int
main(int argc, char** argv)
{
    std::printf("probe started\n");
    std::fflush(stdout);
    if(argc < 2) fail("usage", "CALLER [--preload PATH]... [--then-open DSO NAME]");

    const char* caller    = argv[1];
    const char* then_dso  = nullptr;
    const char* then_name = nullptr;
    for(int i = 2; i < argc; ++i)
    {
        if(std::strcmp(argv[i], "--preload") == 0 && i + 1 < argc)
        {
            if(::dlopen(argv[++i], RTLD_NOW | RTLD_LOCAL) == nullptr) fail("preload", ::dlerror());
        }
        else if(std::strcmp(argv[i], "--then-open") == 0 && i + 2 < argc)
        {
            then_dso  = argv[++i];
            then_name = argv[++i];
        }
        else
        {
            fail("unexpected argument", argv[i]);
        }
    }

    const auto opener = fixture_opener(caller);
    std::printf("caller-tag %s\n", search_tag(caller));
    print_mapped("hsa-runtime", "libhsa-runtime64.so", /*canonical=*/false);

    auto ops = rocprofiler::platform::wsl::default_loader_ops();
    ops.open = [opener](const char* name, int flags) { return opener(name, flags); };
    std::printf("nodes %zu\n", rocprofiler::platform::wsl::read_dxg_gpu_topology(ops).size());
    print_mapped("after-read", "librocdxg", /*canonical=*/true);

    if(then_dso != nullptr)
    {
        void*     handle = fixture_opener(then_dso)(then_name, RTLD_NOW | RTLD_LOCAL);
        link_map* map    = nullptr;
        if(handle != nullptr && ::dlinfo(handle, RTLD_DI_LINKMAP, &map) == 0)
            std::printf("then-open %s\n", real_path(map->l_name).c_str());
        else
            std::printf("then-open none\n");
        print_mapped("after-then-open", "librocdxg", /*canonical=*/true);
    }

    std::fflush(stdout);
    return 0;
}
