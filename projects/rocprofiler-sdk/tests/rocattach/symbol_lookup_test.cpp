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

#include "symbol_lookup.hpp"

#include "lib/common/scope_destructor.hpp"

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <array>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

namespace
{
constexpr auto REGISTER_LIBRARY_NAME = "librocprofiler-register.so";
constexpr auto ATTACH_SYMBOL_NAME    = "rocprofiler_register_attach";

struct loaded_library
{
    std::string path              = {};
    void*       handle            = nullptr;
    bool        remove_on_cleanup = false;
};

void
cleanup_loaded_library(loaded_library& library)
{
    if(library.handle != nullptr)
    {
        dlclose(library.handle);
        library.handle = nullptr;
    }

    if(library.remove_on_cleanup && !library.path.empty())
    {
        std::error_code ec;
        std::filesystem::remove(library.path, ec);
    }
}

loaded_library
load_library(const char* path)
{
    auto* handle = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if(!handle)
    {
        std::cerr << "dlopen failed for " << path << ": " << dlerror() << '\n';
        std::exit(1);
    }
    return loaded_library{path, handle};
}

uintptr_t
symbol_offset(const loaded_library& library)
{
    auto* expected = dlsym(library.handle, ATTACH_SYMBOL_NAME);
    if(!expected)
    {
        std::cerr << "dlsym failed for " << library.path << "::" << ATTACH_SYMBOL_NAME << ": "
                  << dlerror() << '\n';
        std::exit(1);
    }

    auto info = Dl_info{};
    if(dladdr(expected, &info) == 0 || !info.dli_fbase)
    {
        std::cerr << "dladdr failed for " << library.path << "::" << ATTACH_SYMBOL_NAME << '\n';
        std::exit(1);
    }

    return reinterpret_cast<uintptr_t>(expected) - reinterpret_cast<uintptr_t>(info.dli_fbase);
}

void
expect_resolves_to_dlsym(const loaded_library& library)
{
    auto* expected = dlsym(library.handle, ATTACH_SYMBOL_NAME);
    if(!expected)
    {
        std::cerr << "dlsym failed for " << library.path << "::" << ATTACH_SYMBOL_NAME << ": "
                  << dlerror() << '\n';
        std::exit(1);
    }

    void* resolved = nullptr;
    if(!rocprofiler::rocattach::find_symbol(getpid(), resolved, library.path, ATTACH_SYMBOL_NAME))
    {
        std::cerr << "find_symbol failed for exact mapped path " << library.path << '\n';
        std::exit(1);
    }

    if(resolved != expected)
    {
        std::cerr << "find_symbol returned " << resolved << " for " << library.path
                  << ", expected dlsym address " << expected << '\n';
        std::exit(1);
    }
}

void
expect_different_symbol_offsets(const loaded_library& first, const loaded_library& second)
{
    auto first_offset  = symbol_offset(first);
    auto second_offset = symbol_offset(second);
    if(first_offset == second_offset)
    {
        std::cerr << "Expected different symbol offsets for " << first.path << " and "
                  << second.path << ", but both were 0x" << std::hex << first_offset << std::dec
                  << '\n';
        std::exit(1);
    }
}

void
expect_ambiguous_basename_fails()
{
    void* resolved = nullptr;
    if(rocprofiler::rocattach::find_symbol(
           getpid(), resolved, REGISTER_LIBRARY_NAME, ATTACH_SYMBOL_NAME))
    {
        std::cerr << "find_symbol unexpectedly resolved ambiguous " << REGISTER_LIBRARY_NAME
                  << " to " << resolved << '\n';
        std::exit(1);
    }
}

loaded_library
create_and_load_sectionless_copy(const loaded_library& source, std::string_view label)
{
    auto path =
        std::filesystem::temp_directory_path() /
        ("librocprofiler-register.so.rocattach-sectionless-" + std::string{label} + "-XXXXXX");
    auto path_buffer = path.string();
    auto fd          = mkstemp(path_buffer.data());
    if(fd < 0)
    {
        std::cerr << "mkstemp failed for sectionless ELF fixture\n";
        std::exit(1);
    }
    close(fd);

    std::ifstream input{source.path, std::ios::binary};
    std::ofstream output{path_buffer, std::ios::binary | std::ios::trunc};
    output << input.rdbuf();
    input.close();
    output.close();

    std::fstream elf{path_buffer, std::ios::binary | std::ios::in | std::ios::out};
    if(!elf)
    {
        std::cerr << "failed to open sectionless ELF copy " << path_buffer << '\n';
        std::exit(1);
    }

    const auto zero64 = uint64_t{0};
    const auto zero16 = uint16_t{0};
    // Make section-header lookup impossible so the resolver must use PT_DYNAMIC.
    elf.seekp(offsetof(Elf64_Ehdr, e_shoff));
    elf.write(reinterpret_cast<const char*>(&zero64), sizeof(zero64));
    elf.seekp(offsetof(Elf64_Ehdr, e_shnum));
    elf.write(reinterpret_cast<const char*>(&zero16), sizeof(zero16));
    elf.seekp(offsetof(Elf64_Ehdr, e_shstrndx));
    elf.write(reinterpret_cast<const char*>(&zero16), sizeof(zero16));
    if(!elf)
    {
        std::cerr << "failed to patch section header fields in " << path_buffer << '\n';
        std::exit(1);
    }

    auto library              = load_library(path_buffer.c_str());
    library.remove_on_cleanup = true;
    return library;
}

void
expect_malformed_mapped_elf_fails()
{
    auto path = std::filesystem::temp_directory_path() /
                "librocprofiler-register.so.rocattach-malformed-XXXXXX";
    auto path_buffer = path.string();
    auto fd          = mkstemp(path_buffer.data());
    if(fd < 0)
    {
        std::cerr << "mkstemp failed for malformed ELF fixture\n";
        std::exit(1);
    }

    constexpr auto contents = std::string_view{"not-an-elf"};
    if(write(fd, contents.data(), contents.size()) != static_cast<ssize_t>(contents.size()))
    {
        std::cerr << "write failed for malformed ELF fixture\n";
        close(fd);
        std::exit(1);
    }

    auto* mapping = mmap(nullptr, contents.size(), PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if(mapping == MAP_FAILED)
    {
        std::cerr << "mmap failed for malformed ELF fixture\n";
        std::exit(1);
    }

    // The resolver should inspect mapped files defensively and fail before
    // treating arbitrary mapped bytes as an ELF shared object.
    void* resolved = nullptr;
    if(rocprofiler::rocattach::find_symbol(getpid(), resolved, path_buffer, ATTACH_SYMBOL_NAME))
    {
        std::cerr << "find_symbol unexpectedly resolved malformed mapped ELF " << path_buffer
                  << " to " << resolved << '\n';
        munmap(mapping, contents.size());
        std::filesystem::remove(path_buffer);
        std::exit(1);
    }

    munmap(mapping, contents.size());
    std::filesystem::remove(path_buffer);
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 7)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <normal-lib> <gnu-hash-lib> <sysv-hash-lib> <shifted-lib> "
                     "<ambiguous-a> <ambiguous-b>\n";
        return 1;
    }

    auto libraries = std::vector<loaded_library>{};
    libraries.reserve(6);
    for(auto* path :
        std::array<const char*, 6>{argv[1], argv[2], argv[3], argv[4], argv[5], argv[6]})
    {
        libraries.emplace_back(load_library(path));
    }

    expect_resolves_to_dlsym(libraries.at(0));
    expect_resolves_to_dlsym(libraries.at(1));
    expect_resolves_to_dlsym(libraries.at(2));
    expect_resolves_to_dlsym(libraries.at(3));
    expect_different_symbol_offsets(libraries.at(0), libraries.at(3));
    {
        auto sectionless_normal  = create_and_load_sectionless_copy(libraries.at(0), "normal");
        auto sectionless_gnu     = create_and_load_sectionless_copy(libraries.at(1), "gnu");
        auto sectionless_sysv    = create_and_load_sectionless_copy(libraries.at(2), "sysv");
        auto cleanup_sectionless = rocprofiler::common::scope_destructor{[&]() {
            cleanup_loaded_library(sectionless_sysv);
            cleanup_loaded_library(sectionless_gnu);
            cleanup_loaded_library(sectionless_normal);
        }};
        expect_resolves_to_dlsym(sectionless_normal);
        expect_resolves_to_dlsym(sectionless_gnu);
        expect_resolves_to_dlsym(sectionless_sysv);
    }
    expect_ambiguous_basename_fails();
    expect_malformed_mapped_elf_fails();

    std::cout << "Test PASSED: target ELF resolver resolved exact mapped libraries and rejected "
                 "ambiguous and malformed mappings\n";
    return 0;
}
