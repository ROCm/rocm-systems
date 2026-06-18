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

#include "lib/common/filesystem.hpp"
#include "lib/common/scope_destructor.hpp"
#include "lib/rocprofiler-sdk/code_object/hip/code_object.hpp"

#include <gtest/gtest.h>

#if ROCPROFILER_TEST_HAS_ROCM_KPACK
#    include <rocm_kpack/kpack.h>
#endif

#include <unistd.h>

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

#ifndef CODEOBJ_BINARY_DIR
static_assert(false && "Please define CODEOBJ_BINARY_DIR to codeobj tests binary, "
                       "e.g. ../source/lib/tests/codeobj/");
#endif

#ifndef CODEOBJ_INSTALL_DIR
static_assert(false && "Please define CODEOBJ_INSTALL_DIR to the installed tests bin directory "
                       "(e.g. <prefix>/share/rocprofiler-sdk/tests/unit-tests/bin/)");
#endif

namespace hip = rocprofiler::code_object::hip;
namespace fs  = rocprofiler::common::filesystem;

namespace
{
constexpr auto kKpackBinaryName = std::string_view{"bin/rocprofiler-codeobj-kpack-test"};
constexpr auto kKpackTocKey     = std::string_view{"bin/rocprofiler-codeobj-kpack-test#0"};
constexpr auto kKpackArch       = std::string_view{"gfx90a"};

std::string
get_data_file_path(const char* name)
{
    const auto try_path = [&](const fs::path& base) -> std::string {
        std::error_code ec = {};
        fs::path        p  = base / name;
        if(fs::exists(p, ec) && fs::is_regular_file(p, ec)) return p.string();
        return {};
    };

    for(const char* base : {CODEOBJ_BINARY_DIR, CODEOBJ_INSTALL_DIR})
    {
        if(auto found = try_path(fs::path(base)); !found.empty()) return found;
    }

    std::error_code ec      = {};
    fs::path        exe_dir = fs::read_symlink("/proc/self/exe", ec).parent_path();
    if(!ec)
    {
        if(auto found = try_path(exe_dir); !found.empty()) return found;
    }

    return {};
}

std::vector<char>
read_file(const std::string& path)
{
    auto file = std::ifstream{path, std::ios::binary};
    return {std::istreambuf_iterator<char>{file}, std::istreambuf_iterator<char>{}};
}

void
append_be(std::vector<char>& data, uint64_t value, size_t bytes)
{
    for(size_t i = 0; i < bytes; ++i)
    {
        const auto shift = 8 * (bytes - i - 1);
        data.emplace_back(static_cast<char>((value >> shift) & 0xff));
    }
}

void
append_le(std::ofstream& file, uint64_t value, size_t bytes)
{
    for(size_t i = 0; i < bytes; ++i)
        file.put(static_cast<char>((value >> (8 * i)) & 0xff));
}

void
append_msgpack_uint(std::vector<char>& data, uint64_t value)
{
    if(value <= 0x7f)
    {
        data.emplace_back(static_cast<char>(value));
    }
    else if(value <= 0xff)
    {
        data.emplace_back(static_cast<char>(0xcc));
        append_be(data, value, 1);
    }
    else if(value <= 0xffff)
    {
        data.emplace_back(static_cast<char>(0xcd));
        append_be(data, value, 2);
    }
    else if(value <= 0xffffffff)
    {
        data.emplace_back(static_cast<char>(0xce));
        append_be(data, value, 4);
    }
    else
    {
        data.emplace_back(static_cast<char>(0xcf));
        append_be(data, value, 8);
    }
}

void
append_msgpack_str(std::vector<char>& data, std::string_view value)
{
    if(value.size() <= 31)
    {
        data.emplace_back(static_cast<char>(0xa0 | value.size()));
    }
    else if(value.size() <= 0xff)
    {
        data.emplace_back(static_cast<char>(0xd9));
        append_be(data, value.size(), 1);
    }
    else
    {
        ASSERT_LE(value.size(), 0xffff);
        data.emplace_back(static_cast<char>(0xda));
        append_be(data, value.size(), 2);
    }
    data.insert(data.end(), value.begin(), value.end());
}

void
append_msgpack_array(std::vector<char>& data, size_t size)
{
    ASSERT_LE(size, 15);
    data.emplace_back(static_cast<char>(0x90 | size));
}

void
append_msgpack_map(std::vector<char>& data, size_t size)
{
    ASSERT_LE(size, 15);
    data.emplace_back(static_cast<char>(0x80 | size));
}

std::vector<char>
make_hipk_metadata(const std::string& kpack_path)
{
    // Matches librocm_kpack loader metadata:
    // {"kernel_name": "...", "kpack_search_paths": ["..."]}
    auto data = std::vector<char>{};
    append_msgpack_map(data, 2);
    append_msgpack_str(data, "kernel_name");
    append_msgpack_str(data, kKpackBinaryName);
    append_msgpack_str(data, "kpack_search_paths");
    append_msgpack_array(data, 1);
    append_msgpack_str(data, kpack_path);

    data.resize(64 * 1024, 0);
    return data;
}

std::vector<char>
make_kpack_toc(size_t code_object_size)
{
    constexpr uint64_t blob_offset = 64;

    auto data = std::vector<char>{};
    append_msgpack_map(data, 7);

    append_msgpack_str(data, "format_version");
    append_msgpack_uint(data, 1);

    append_msgpack_str(data, "group_name");
    append_msgpack_str(data, "codeobj");

    append_msgpack_str(data, "gfx_arch_family");
    append_msgpack_str(data, kKpackArch);

    append_msgpack_str(data, "gfx_arches");
    append_msgpack_array(data, 1);
    append_msgpack_str(data, kKpackArch);

    append_msgpack_str(data, "toc");
    append_msgpack_map(data, 1);
    append_msgpack_str(data, kKpackTocKey);
    append_msgpack_map(data, 1);
    append_msgpack_str(data, kKpackArch);
    append_msgpack_map(data, 3);
    append_msgpack_str(data, "type");
    append_msgpack_str(data, "hsaco");
    append_msgpack_str(data, "ordinal");
    append_msgpack_uint(data, 0);
    append_msgpack_str(data, "original_size");
    append_msgpack_uint(data, code_object_size);

    append_msgpack_str(data, "compression_scheme");
    append_msgpack_str(data, "none");

    append_msgpack_str(data, "blobs");
    append_msgpack_array(data, 1);
    append_msgpack_map(data, 2);
    append_msgpack_str(data, "offset");
    append_msgpack_uint(data, blob_offset);
    append_msgpack_str(data, "size");
    append_msgpack_uint(data, code_object_size);

    return data;
}

void
write_kpack_archive(const std::string& path, const std::vector<char>& code_object)
{
    constexpr uint64_t header_size = 16;
    constexpr uint64_t blob_offset = 64;

    auto toc        = make_kpack_toc(code_object.size());
    auto toc_offset = blob_offset + code_object.size();

    auto file = std::ofstream{path, std::ios::binary};
    ASSERT_TRUE(file) << path;

    file.write("KPAK", 4);
    append_le(file, 1, 4);
    append_le(file, toc_offset, 8);

    for(uint64_t i = header_size; i < blob_offset; ++i)
        file.put('\0');

    file.write(code_object.data(), static_cast<std::streamsize>(code_object.size()));
    file.write(toc.data(), static_cast<std::streamsize>(toc.size()));
    ASSERT_TRUE(file) << path;
}
}  // namespace

TEST(KpackCodeObject, LoadedHipkCodeObjectCanBeParsedForKernelSymbols)
{
#if !ROCPROFILER_TEST_HAS_ROCM_KPACK
    GTEST_SKIP() << "rocm-kpack development package was not found";
#else
    auto code_object_path = get_data_file_path("syncthreads_kernel.bin");
    ASSERT_FALSE(code_object_path.empty());

    auto code_object = read_file(code_object_path);
    ASSERT_FALSE(code_object.empty()) << code_object_path;

    auto kpack_path =
        std::string{"/tmp/rocprofiler-sdk-codeobj-kpack-"} + std::to_string(::getpid()) + ".kpack";
    auto cleanup_kpack =
        rocprofiler::common::scope_destructor{[&]() { std::remove(kpack_path.c_str()); }};
    write_kpack_archive(kpack_path, code_object);

    auto hipk_metadata = make_hipk_metadata(kpack_path);

    kpack_cache_t cache = nullptr;
    ASSERT_EQ(kpack_cache_create(&cache), KPACK_SUCCESS);
    ASSERT_NE(cache, nullptr);
    auto cleanup_cache =
        rocprofiler::common::scope_destructor{[&]() { kpack_cache_destroy(cache); }};

    const char* arch_list[]             = {kKpackArch.data()};
    void*       loaded_code_object      = nullptr;
    size_t      loaded_code_object_size = 0;

    ASSERT_EQ(kpack_load_code_object(cache,
                                     hipk_metadata.data(),
                                     "/tmp/rocprofiler-sdk-codeobj-kpack-host",
                                     0,
                                     arch_list,
                                     1,
                                     &loaded_code_object,
                                     &loaded_code_object_size),
              KPACK_SUCCESS);
    ASSERT_NE(loaded_code_object, nullptr);
    auto cleanup_code_object = rocprofiler::common::scope_destructor{
        [&]() { kpack_free_code_object(loaded_code_object); }};

    EXPECT_EQ(loaded_code_object_size, code_object.size());

    auto symbol_map = hip::get_kernel_symbol_device_name_map_from_executable(
        loaded_code_object, loaded_code_object_size);

    ASSERT_FALSE(symbol_map.empty());
    auto symbol = symbol_map.find("_Z18syncthreads_kernelPi.kd");
    ASSERT_NE(symbol, symbol_map.end());
    EXPECT_EQ(symbol->second, "_Z18syncthreads_kernelPi");
#endif
}
