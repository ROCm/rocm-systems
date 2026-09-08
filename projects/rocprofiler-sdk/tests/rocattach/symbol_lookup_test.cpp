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

#include "common/filesystem.hpp"

#include "lib/common/scope_destructor.hpp"

#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <optional>
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
        common::fs::remove(library.path, ec);
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
        common::fs::temp_directory_path() /
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
    auto path =
        common::fs::temp_directory_path() / "librocprofiler-register.so.rocattach-malformed-XXXXXX";
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
        common::fs::remove(path_buffer);
        std::exit(1);
    }

    munmap(mapping, contents.size());
    common::fs::remove(path_buffer);
}

std::optional<size_t>
find_build_id_descriptor(const std::vector<uint8_t>& bytes)
{
    auto header = Elf64_Ehdr{};
    if(bytes.size() < sizeof(header)) return std::nullopt;
    std::memcpy(&header, bytes.data(), sizeof(header));

    for(auto idx = uint16_t{0}; idx < header.e_phnum; ++idx)
    {
        auto segment = Elf64_Phdr{};
        auto offset  = header.e_phoff + (static_cast<size_t>(idx) * header.e_phentsize);
        if(offset + sizeof(segment) > bytes.size()) return std::nullopt;
        std::memcpy(&segment, bytes.data() + offset, sizeof(segment));
        if(segment.p_type != PT_NOTE) continue;

        auto cursor = static_cast<size_t>(segment.p_offset);
        auto end    = cursor + static_cast<size_t>(segment.p_filesz);
        if(end > bytes.size()) return std::nullopt;

        while(cursor + sizeof(Elf64_Nhdr) <= end)
        {
            auto note = Elf64_Nhdr{};
            std::memcpy(&note, bytes.data() + cursor, sizeof(note));

            auto name_offset = cursor + sizeof(note);
            auto descriptor  = name_offset + ((note.n_namesz + 3U) & ~3U);
            auto next        = descriptor + ((note.n_descsz + 3U) & ~3U);
            if(descriptor > end || next > end) break;

            if(note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 && note.n_descsz > 0 &&
               std::memcmp(bytes.data() + name_offset, "GNU", 4) == 0)
            {
                return descriptor;
            }
            cursor = next;
        }
    }
    return std::nullopt;
}

// Copies source and flips one byte inside its NT_GNU_BUILD_ID descriptor. The
// result is byte-identical to the source everywhere else, so a rejection can
// only come from the Build ID comparison and never from a layout difference.
void
copy_with_flipped_build_id(const std::string& source, const std::string& destination)
{
    auto input = std::ifstream{source, std::ios::binary};
    auto bytes = std::vector<uint8_t>{std::istreambuf_iterator<char>{input},
                                      std::istreambuf_iterator<char>{}};
    input.close();

    auto descriptor = find_build_id_descriptor(bytes);
    if(!descriptor)
    {
        std::cerr << "could not locate a GNU Build ID note in " << source << '\n';
        std::exit(1);
    }
    bytes.at(*descriptor) ^= 0xffU;

    auto output = std::ofstream{destination, std::ios::binary | std::ios::trunc};
    output.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    output.close();
    if(!output)
    {
        std::cerr << "failed to write Build ID variant " << destination << '\n';
        std::exit(1);
    }
}

bool
expect_pathname_lookup_validates_build_id(const loaded_library& source,
                                          const loaded_library& no_build_id)
{
    auto path =
        common::fs::temp_directory_path() / "librocprofiler-register.so.rocattach-replaced-XXXXXX";
    auto path_buffer = path.string();
    auto fd          = mkstemp(path_buffer.data());
    if(fd < 0)
    {
        std::cerr << "mkstemp failed for replaced ELF fixture\n";
        return false;
    }
    close(fd);

    common::fs::copy_file(source.path, path_buffer, common::fs::copy_options::overwrite_existing);

    auto mapped = load_library(path_buffer.c_str());
    auto unload_mapped =
        rocprofiler::common::scope_destructor{[&]() { cleanup_loaded_library(mapped); }};

    auto* symbol = dlsym(mapped.handle, ATTACH_SYMBOL_NAME);
    if(symbol == nullptr)
    {
        std::cerr << "dlsym failed for " << path_buffer << "::" << ATTACH_SYMBOL_NAME << '\n';
        return false;
    }
    auto expected = reinterpret_cast<uintptr_t>(symbol);

    struct stat mapped_identity
    {};
    if(stat(path_buffer.c_str(), &mapped_identity) != 0)
    {
        std::cerr << "stat failed for " << path_buffer << '\n';
        return false;
    }

    auto done_pipe = std::array<int, 2>{};
    if(pipe(done_pipe.data()) != 0)
    {
        std::cerr << "pipe failed for cross-process Build ID test\n";
        return false;
    }

    auto child = fork();
    if(child < 0)
    {
        std::cerr << "fork failed for cross-process Build ID test\n";
        return false;
    }
    if(child == 0)
    {
        close(done_pipe[1]);
        auto done = char{};
        while(read(done_pipe[0], &done, sizeof(done)) < 0 && errno == EINTR)
        {}
        _exit(0);
    }

    close(done_pipe[0]);
    auto release_child = rocprofiler::common::scope_destructor{[&]() {
        close(done_pipe[1]);
        auto status = int{};
        if(waitpid(child, &status, 0) != child || !WIFEXITED(status) || WEXITSTATUS(status) != 0)
        {
            std::cerr << "cross-process Build ID child did not exit cleanly\n";
        }
        std::error_code ec;
        common::fs::remove(path_buffer, ec);
        common::fs::remove(path_buffer + ".replacement", ec);
        common::fs::remove(path_buffer + ".flipped", ec);
    }};

    // Repoint the pathname at a different inode while the child keeps the
    // original one mapped. Confirm the inode actually changed,
    // otherwise the Build ID comparison would never be reached.
    auto install_at_pathname = [&](const std::string& file) {
        auto replacement = path_buffer + ".replacement";
        common::fs::copy_file(file, replacement, common::fs::copy_options::overwrite_existing);
        common::fs::rename(replacement, path_buffer);

        struct stat installed
        {};
        if(stat(path_buffer.c_str(), &installed) != 0 ||
           (installed.st_dev == mapped_identity.st_dev &&
            installed.st_ino == mapped_identity.st_ino))
        {
            std::cerr << "replacing " << path_buffer << " did not change its inode\n";
            return false;
        }
        return true;
    };

    // Forces the /proc/<pid>/root path instead of /proc/<pid>/map_files.
    constexpr auto pathname_only = true;

    auto expect_lookup = [&](bool should_resolve, const char* description) {
        void* resolved    = nullptr;
        auto  did_resolve = rocprofiler::rocattach::find_symbol(
            child, resolved, path_buffer, ATTACH_SYMBOL_NAME, pathname_only);
        if(did_resolve != should_resolve)
        {
            std::cerr << description << '\n';
            return false;
        }
        if(should_resolve && reinterpret_cast<uintptr_t>(resolved) != expected)
        {
            std::cerr << "find_symbol resolved the wrong address: " << description << '\n';
            return false;
        }
        return true;
    };

    if(!install_at_pathname(source.path)) return false;
    if(!expect_lookup(true, "find_symbol rejected a pathname replacement with a matching Build ID"))
    {
        return false;
    }

    auto flipped = path_buffer + ".flipped";
    copy_with_flipped_build_id(source.path, flipped);
    if(!install_at_pathname(flipped)) return false;
    if(!expect_lookup(false,
                      "find_symbol accepted a replacement whose only difference is its Build ID"))
    {
        return false;
    }

    if(!install_at_pathname(no_build_id.path)) return false;
    if(!expect_lookup(false, "find_symbol accepted a replacement without a Build ID"))
    {
        return false;
    }
    return true;
}
}  // namespace

int
main(int argc, char** argv)
{
    if(argc != 8)
    {
        std::cerr << "Usage: " << argv[0]
                  << " <normal-lib> <gnu-hash-lib> <sysv-hash-lib> <shifted-lib> "
                     "<no-build-id-lib> <ambiguous-a> <ambiguous-b>\n";
        return 1;
    }

    auto libraries = std::vector<loaded_library>{};
    libraries.reserve(7);
    for(const auto* path :
        std::array<const char*, 7>{argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7]})
    {
        libraries.emplace_back(load_library(path));
    }

    expect_resolves_to_dlsym(libraries.at(0));
    expect_resolves_to_dlsym(libraries.at(1));
    expect_resolves_to_dlsym(libraries.at(2));
    expect_resolves_to_dlsym(libraries.at(3));
    expect_resolves_to_dlsym(libraries.at(4));
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
    if(!expect_pathname_lookup_validates_build_id(libraries.at(0), libraries.at(4))) return 1;

    std::cout << "Test PASSED: target ELF resolver resolved exact mapped libraries and rejected "
                 "ambiguous and malformed mappings\n";
    return 0;
}
