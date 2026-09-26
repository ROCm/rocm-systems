// MIT License
//
// Copyright (c) 2025-2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "symbol_lookup.hpp"

#include "lib/common/filesystem.hpp"
#include "lib/common/hasher.hpp"
#include "lib/common/logging.hpp"
#include "lib/common/scope_destructor.hpp"

#include <fmt/format.h>
#include <elfio/elfio.hpp>
#include <elfio/elfio_symbols.hpp>

#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <sys/uio.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

namespace rocprofiler
{
namespace rocattach
{
namespace
{
namespace fs = common::filesystem;

// Keep limits generous enough for debug builds, but bounded so malformed target
// ELFs cannot force unbounded memory use or symbol/hash traversal.
constexpr auto MAX_TARGET_ELF_SIZE      = uint64_t{512} * 1024 * 1024;
constexpr auto MAX_ELF_NOTE_SIZE        = uint64_t{1024} * 1024;
constexpr auto MAX_TARGET_READ_SIZE     = uint64_t{1024} * 1024;
constexpr auto MAX_BUILD_ID_SIZE        = uint64_t{4096};
constexpr auto MAX_DYNAMIC_SYMBOLS      = size_t{1} << 24;
constexpr auto MAX_GNU_HASH_CHAIN_STEPS = size_t{1} << 24;

// Note names and descriptors are padded to 4 bytes regardless of the containing
// segment's p_align. GNU emits .note.gnu.property in a PT_NOTE declaring
// p_align 8, yet its name is still padded to 4.
constexpr auto ELF_NOTE_ALIGNMENT = uint64_t{4};

struct memory_mapping
{
    uintptr_t   start        = 0;
    uintptr_t   end          = 0;
    uint64_t    file_offset  = 0;
    std::string permissions  = {};
    uint32_t    device_major = 0;
    uint32_t    device_minor = 0;
    uint64_t    inode        = 0;
    std::string path         = {};
};

struct mapped_object
{
    uint32_t                    device_major = 0;
    uint32_t                    device_minor = 0;
    uint64_t                    inode        = 0;
    std::string                 path         = {};
    std::vector<memory_mapping> mappings     = {};
};

struct mapped_object_key
{
    uint32_t device_major = 0;
    uint32_t device_minor = 0;
    uint64_t inode        = 0;

    bool operator==(const mapped_object_key& rhs) const
    {
        return device_major == rhs.device_major && device_minor == rhs.device_minor &&
               inode == rhs.inode;
    }
};

struct mapped_object_key_hash
{
    size_t operator()(const mapped_object_key& key) const
    {
        return common::fnv1a_hasher::combine(key.device_major, key.device_minor, key.inode);
    }
};

struct target_elf
{
    std::string             path                     = {};
    std::vector<uint8_t>    data                     = {};
    ELFIO::elfio            reader                   = {};
    std::vector<Elf64_Phdr> load_segments            = {};
    bool                    identity_matches_mapping = false;
};

std::optional<uint64_t>
checked_add(uint64_t lhs, uint64_t rhs)
{
    if(lhs > std::numeric_limits<uint64_t>::max() - rhs) return std::nullopt;
    return lhs + rhs;
}

std::optional<uint64_t>
checked_sub(uint64_t lhs, uint64_t rhs)
{
    if(lhs < rhs) return std::nullopt;
    return lhs - rhs;
}

std::optional<uint64_t>
checked_mul(uint64_t lhs, uint64_t rhs)
{
    if(lhs != 0 && rhs > std::numeric_limits<uint64_t>::max() / lhs) return std::nullopt;
    return lhs * rhs;
}

std::optional<uint64_t>
align_down(uint64_t value, uint64_t alignment)
{
    if(alignment == 0) return std::nullopt;
    return value - (value % alignment);
}

std::optional<uint64_t>
align_up(uint64_t value, uint64_t alignment)
{
    if(alignment == 0) return std::nullopt;
    auto remainder = value % alignment;
    if(remainder == 0) return value;
    return checked_add(value, alignment - remainder);
}

bool
has_range(size_t file_size, uint64_t offset, uint64_t size)
{
    return offset <= file_size && size <= file_size - offset;
}

template <typename Tp>
std::optional<Tp>
read_as(const std::vector<uint8_t>& data, uint64_t offset)
{
    if(!has_range(data.size(), offset, sizeof(Tp))) return std::nullopt;

    auto value = Tp{};
    std::memcpy(&value, data.data() + offset, sizeof(Tp));
    return value;
}

Elf64_Phdr
to_phdr(const ELFIO::segment& segment)
{
    auto phdr     = Elf64_Phdr{};
    phdr.p_type   = segment.get_type();
    phdr.p_flags  = segment.get_flags();
    phdr.p_offset = segment.get_offset();
    phdr.p_vaddr  = segment.get_virtual_address();
    phdr.p_paddr  = segment.get_physical_address();
    phdr.p_filesz = segment.get_file_size();
    phdr.p_memsz  = segment.get_memory_size();
    phdr.p_align  = segment.get_align();
    return phdr;
}

std::string
trim_left(std::string value)
{
    value.erase(value.begin(), std::find_if(value.begin(), value.end(), [](unsigned char ch) {
                    return !std::isspace(ch);
                }));
    return value;
}

std::string
strip_deleted_suffix(std::string value)
{
    constexpr auto deleted_suffix = std::string_view{" (deleted)"};
    if(value.size() >= deleted_suffix.size() &&
       value.compare(value.size() - deleted_suffix.size(), deleted_suffix.size(), deleted_suffix) ==
           0)
    {
        value.resize(value.size() - deleted_suffix.size());
    }
    return value;
}

std::string
basename(std::string path)
{
    path     = strip_deleted_suffix(std::move(path));
    auto pos = path.find_last_of('/');
    return (pos == std::string::npos) ? path : path.substr(pos + 1);
}

bool
library_name_matches(const memory_mapping& mapping, const std::string& library)
{
    if(mapping.path.empty()) return false;

    auto clean_path = strip_deleted_suffix(mapping.path);
    if(clean_path == library) return true;

    auto map_base = basename(clean_path);
    auto lib_base = basename(library);
    return map_base == lib_base || map_base.rfind(fmt::format("{}.", lib_base), 0) == 0;
}

bool
library_path_matches_exactly(const mapped_object& object, const std::string& library)
{
    if(library.find('/') == std::string::npos) return false;

    auto clean_library = strip_deleted_suffix(library);
    for(const auto& mapping : object.mappings)
    {
        if(strip_deleted_suffix(mapping.path) == clean_library) return true;
    }
    return false;
}

std::optional<memory_mapping>
parse_maps_line(const std::string& line)
{
    auto        iss = std::istringstream{line};
    std::string range;
    std::string offset;
    std::string device;
    auto        mapping = memory_mapping{};

    if(!(iss >> range >> mapping.permissions >> offset >> device >> mapping.inode))
    {
        return std::nullopt;
    }

    auto dash_pos = range.find('-');
    auto dev_pos  = device.find(':');
    if(dash_pos == std::string::npos || dev_pos == std::string::npos) return std::nullopt;

    try
    {
        mapping.start = static_cast<uintptr_t>(std::stoull(range.substr(0, dash_pos), nullptr, 16));
        mapping.end = static_cast<uintptr_t>(std::stoull(range.substr(dash_pos + 1), nullptr, 16));
        mapping.file_offset = std::stoull(offset, nullptr, 16);
        mapping.device_major =
            static_cast<uint32_t>(std::stoul(device.substr(0, dev_pos), nullptr, 16));
        mapping.device_minor =
            static_cast<uint32_t>(std::stoul(device.substr(dev_pos + 1), nullptr, 16));

        auto path = std::string{};
        std::getline(iss, path);
        mapping.path = trim_left(std::move(path));
    } catch(const std::exception& e)
    {
        ROCP_TRACE << "[rocprofiler-sdk-rocattach] Failed to parse maps line: " << line
                   << ". error: " << e.what();
        return std::nullopt;
    }

    return mapping;
}

std::vector<memory_mapping>
parse_maps(pid_t pid)
{
    auto filename = fmt::format("/proc/{}/maps", pid);
    errno         = 0;
    auto maps     = std::ifstream{filename};
    auto open_err = errno;
    auto result   = std::vector<memory_mapping>{};

    if(!maps)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't open " << filename << ": "
                   << ((open_err != 0) ? std::strerror(open_err) : "unknown error");
        return result;
    }

    auto line = std::string{};
    while(std::getline(maps, line))
    {
        if(auto mapping = parse_maps_line(line); mapping) result.emplace_back(*mapping);
    }
    return result;
}

std::vector<mapped_object>
find_mapped_objects(pid_t pid, const std::string& library)
{
    auto objects = std::unordered_map<mapped_object_key, mapped_object, mapped_object_key_hash>{};

    for(const auto& mapping : parse_maps(pid))
    {
        if(mapping.inode == 0 || !library_name_matches(mapping, library)) continue;

        // Multiple maps entries usually belong to the same ELF: readonly headers,
        // executable text, writable data, and so on. Device/inode is the stable key
        // for grouping those segments even when paths differ through namespaces.
        auto  key    = mapped_object_key{mapping.device_major, mapping.device_minor, mapping.inode};
        auto& object = objects[key];
        object.device_major = mapping.device_major;
        object.device_minor = mapping.device_minor;
        object.inode        = mapping.inode;
        if(object.path.empty()) object.path = mapping.path;
        object.mappings.emplace_back(mapping);
    }

    auto result = std::vector<mapped_object>{};
    result.reserve(objects.size());
    for(auto& itr : objects)
    {
        std::sort(itr.second.mappings.begin(),
                  itr.second.mappings.end(),
                  [](const auto& lhs, const auto& rhs) { return lhs.start < rhs.start; });
        result.emplace_back(std::move(itr.second));
    }

    std::sort(result.begin(), result.end(), [](const auto& lhs, const auto& rhs) {
        return lhs.mappings.front().start < rhs.mappings.front().start;
    });
    return result;
}

void
log_mapped_object_candidates(const std::vector<mapped_object>& objects)
{
    for(const auto& object : objects)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Candidate mapped object: " << object.path
                   << " dev " << object.device_major << ":" << object.device_minor << " inode "
                   << object.inode << " load address 0x" << std::hex
                   << object.mappings.front().start << std::dec;
    }
}

std::optional<mapped_object>
select_unique_mapped_object(pid_t pid, const std::string& library)
{
    auto objects = find_mapped_objects(pid, library);
    if(library.find('/') != std::string::npos)
    {
        objects.erase(std::remove_if(objects.begin(),
                                     objects.end(),
                                     [&](const auto& object) {
                                         return !library_path_matches_exactly(object, library);
                                     }),
                      objects.end());
    }

    if(objects.empty())
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't find mapped target library " << library
                   << " in /proc/" << pid << "/maps";
        return std::nullopt;
    }

    if(objects.size() > 1)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Found multiple mapped target libraries matching "
                   << library << " in /proc/" << pid << "/maps; refusing to choose one implicitly";
        log_mapped_object_candidates(objects);
        return std::nullopt;
    }

    return objects.front();
}

std::optional<std::vector<uint8_t>>
read_file_from_fd(int fd, const std::string& path, uint64_t size)
{
    if(size == 0 || size > MAX_TARGET_ELF_SIZE)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Refusing to read unusually large target ELF "
                   << path << " (" << size << " bytes)";
        return std::nullopt;
    }

    auto data       = std::vector<uint8_t>(static_cast<size_t>(size));
    auto bytes_read = size_t{0};
    while(bytes_read < data.size())
    {
        auto ret = ::read(fd, data.data() + bytes_read, data.size() - bytes_read);
        if(ret < 0)
        {
            if(errno == EINTR) continue;
            ROCP_WARNING << "[rocprofiler-sdk-rocattach] Failed reading target ELF " << path << ": "
                         << std::strerror(errno);
            return std::nullopt;
        }
        if(ret == 0) break;
        bytes_read += static_cast<size_t>(ret);
    }

    if(bytes_read != data.size())
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Short read while reading target ELF " << path
                     << ": expected " << data.size() << " bytes, read " << bytes_read;
        return std::nullopt;
    }

    return data;
}

// When require_identity_match is set, the file is rejected unless its device
// and inode match what /proc/<pid>/maps reports for the mapping. The outcome is
// recorded either way in target_elf::identity_matches_mapping; a file admitted
// without that equality must later pass resolve_target_symbol's Build ID check.
std::optional<target_elf>
read_mapped_file(const std::string& path, const mapped_object& object, bool require_identity_match)
{
    struct stat st
    {};

    auto fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if(fd < 0)
    {
        ROCP_TRACE << "[rocprofiler-sdk-rocattach] Could not open target ELF " << path << ": "
                   << std::strerror(errno);
        return std::nullopt;
    }
    auto close_fd = common::scope_destructor{[fd]() { ::close(fd); }};

    if(::fstat(fd, &st) != 0)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Could not stat opened target ELF " << path
                     << ": " << std::strerror(errno);
        return std::nullopt;
    }

    auto identity_matches = major(st.st_dev) == object.device_major &&
                            minor(st.st_dev) == object.device_minor &&
                            static_cast<uint64_t>(st.st_ino) == object.inode;
    if(require_identity_match && !identity_matches)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Refusing to use " << path
                     << " because its opened device/inode does not match the mapped object";
        return std::nullopt;
    }

    if(!S_ISREG(st.st_mode))
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Refusing to use " << path
                     << " because the opened target ELF is not a regular file";
        return std::nullopt;
    }

    if(st.st_size <= 0)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Refusing to use " << path
                     << " because the opened target ELF has invalid size " << st.st_size;
        return std::nullopt;
    }

    auto data = read_file_from_fd(fd, path, static_cast<uint64_t>(st.st_size));
    if(!data || data->empty()) return std::nullopt;

    auto elf                     = target_elf{};
    elf.path                     = path;
    elf.data                     = std::move(*data);
    elf.identity_matches_mapping = identity_matches;
    return elf;
}

std::optional<target_elf>
open_target_elf(pid_t pid, const mapped_object& object, bool pathname_only)
{
    // map_files is preferred because it names the exact file backing the
    // mapping, so its device/inode must match. The target-root pathname is only
    // a best guess at the same file, so it settles for a Build ID match.
    if(!pathname_only)
    {
        // Any mapping of this object will do: mapped_object groups mappings by
        // device/inode, so all of their map_files links resolve to one file.
        const auto& mapping        = object.mappings.front();
        auto        map_files_path = fmt::format("/proc/{}/map_files/{:x}-{:x}",
                                          pid,
                                          static_cast<uint64_t>(mapping.start),
                                          static_cast<uint64_t>(mapping.end));
        if(auto elf = read_mapped_file(map_files_path, object, /*require_identity_match=*/true))
        {
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] Opened target ELF via " << map_files_path;
            return elf;
        }
    }

    auto target_path = strip_deleted_suffix(object.path);
    if(!target_path.empty() && target_path.front() == '/')
    {
        auto root_path =
            (fs::path{fmt::format("/proc/{}/root", pid)} / fs::path{target_path}.relative_path())
                .string();
        if(auto elf = read_mapped_file(root_path, object, /*require_identity_match=*/false))
        {
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] Opened target ELF via " << root_path;
            return elf;
        }
    }

    ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not open mapped target ELF for " << object.path
               << " via /proc/" << pid << "/map_files or /proc/" << pid << "/root";
    return std::nullopt;
}

bool
parse_target_elf(target_elf& elf)
{
    auto elf_bytes = std::string{reinterpret_cast<const char*>(elf.data.data()), elf.data.size()};
    auto stream    = std::istringstream{elf_bytes, std::ios::binary};
    if(!elf.reader.load(stream))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to parse target ELF: " << elf.path;
        return false;
    }

    if(elf.reader.get_class() != ELFCLASS64 || elf.reader.get_encoding() != ELFDATA2LSB ||
       elf.reader.get_machine() != EM_X86_64 || elf.reader.get_type() != ET_DYN)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Unsupported target ELF format: " << elf.path;
        return false;
    }

    // PT_LOAD segments are used later to translate ELF virtual addresses into
    // target-process addresses.
    elf.load_segments.clear();
    for(const auto& segment : elf.reader.segments)
    {
        if(segment == nullptr || segment->get_type() != PT_LOAD) continue;

        elf.load_segments.emplace_back(to_phdr(*segment));
    }

    if(elf.load_segments.empty())
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Target ELF has no PT_LOAD segments: "
                   << elf.path;
        return false;
    }

    return true;
}

std::optional<uint64_t>
calculate_load_bias(const target_elf& elf, const mapped_object& object)
{
    auto page_size_value = ::sysconf(_SC_PAGESIZE);
    auto page_size =
        (page_size_value > 0) ? static_cast<uint64_t>(page_size_value) : uint64_t{4096};

    // Derive a bias from every (mapping, PT_LOAD segment) pair instead of
    // assuming mappings.front() corresponds to the lowest-p_vaddr segment, and
    // keep a candidate only if it accounts for every file-backed PT_LOAD
    // segment rather than just the one it came from. Each such segment is still
    // required to have its own maps entry whose start and file offset match the
    // candidate exactly; segments sharing one merged VMA are not handled.
    auto segment_is_mapped_at_bias = [&](const Elf64_Phdr& segment, uint64_t bias) {
        if(segment.p_filesz == 0) return true;

        auto segment_file_page    = align_down(segment.p_offset, page_size);
        auto segment_virtual_page = align_down(segment.p_vaddr, page_size);
        if(!segment_file_page || !segment_virtual_page) return false;

        auto expected_start = checked_add(bias, *segment_virtual_page);
        if(!expected_start) return false;

        return std::any_of(
            object.mappings.begin(), object.mappings.end(), [&](const auto& mapping) {
                return mapping.start == *expected_start &&
                       mapping.file_offset == *segment_file_page;
            });
    };

    auto candidates = std::vector<uint64_t>{};
    for(const auto& mapping : object.mappings)
    {
        for(const auto& segment : elf.load_segments)
        {
            if(segment.p_filesz == 0) continue;

            auto segment_file_page = align_down(segment.p_offset, page_size);
            if(!segment_file_page || mapping.file_offset != *segment_file_page) continue;

            auto segment_virtual_page = align_down(segment.p_vaddr, page_size);
            if(!segment_virtual_page) continue;

            auto candidate =
                checked_sub(static_cast<uint64_t>(mapping.start), *segment_virtual_page);
            if(!candidate) continue;

            auto all_segments_mapped = std::all_of(
                elf.load_segments.begin(), elf.load_segments.end(), [&](const auto& other) {
                    return segment_is_mapped_at_bias(other, *candidate);
                });
            if(all_segments_mapped &&
               std::find(candidates.begin(), candidates.end(), *candidate) == candidates.end())
            {
                candidates.emplace_back(*candidate);
            }
        }
    }

    if(candidates.size() != 1)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not calculate a unique load bias for "
                   << object.path << " from its PT_LOAD segments and mappings";
        return std::nullopt;
    }

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Calculated target load bias for " << object.path
               << " as 0x" << std::hex << candidates.front() << std::dec;
    return candidates.front();
}

std::optional<std::vector<uint8_t>>
parse_gnu_build_id(const uint8_t* data, size_t size)
{
    auto result = std::vector<uint8_t>{};
    auto found  = false;
    auto cursor = uint64_t{0};

    while(cursor < size)
    {
        if(sizeof(Elf64_Nhdr) > size - cursor) return std::nullopt;

        auto header = Elf64_Nhdr{};
        std::memcpy(&header, data + cursor, sizeof(header));
        cursor += sizeof(header);

        auto name_size = align_up(header.n_namesz, ELF_NOTE_ALIGNMENT);
        auto desc_size = align_up(header.n_descsz, ELF_NOTE_ALIGNMENT);
        if(!name_size || !desc_size || *name_size > size - cursor) return std::nullopt;

        auto name_offset = cursor;
        cursor += *name_size;
        if(*desc_size > size - cursor) return std::nullopt;

        auto is_gnu_build_id = header.n_type == NT_GNU_BUILD_ID && header.n_namesz == 4 &&
                               header.n_descsz > 0 && header.n_descsz <= MAX_BUILD_ID_SIZE &&
                               std::memcmp(data + name_offset, "GNU", 4) == 0;
        if(is_gnu_build_id)
        {
            auto build_id = std::vector<uint8_t>(data + cursor, data + cursor + header.n_descsz);
            // Repeated notes carrying the same Build ID are consistent. Notes
            // that disagree are ambiguous and must not relax identity checks.
            if(found && result != build_id) return std::nullopt;
            result = std::move(build_id);
            found  = true;
        }
        cursor += *desc_size;
    }

    if(!found) return std::nullopt;
    return result;
}

bool
note_is_file_backed_load(const target_elf& elf, const Elf64_Phdr& note)
{
    if(note.p_filesz == 0) return false;

    return std::any_of(elf.load_segments.begin(), elf.load_segments.end(), [&](const auto& load) {
        if(note.p_vaddr < load.p_vaddr || note.p_offset < load.p_offset) return false;

        auto virtual_delta = note.p_vaddr - load.p_vaddr;
        auto file_delta    = note.p_offset - load.p_offset;

        // The note bytes read from the file must be the same bytes represented
        // at note.p_vaddr in the file-backed PT_LOAD segment.
        return virtual_delta == file_delta && virtual_delta <= load.p_filesz &&
               note.p_filesz <= load.p_filesz - virtual_delta;
    });
}

// Walks the usable PT_NOTE segments and returns the Build ID they agree on.
// read_note supplies the bytes of one note segment and returns nullopt when
// they cannot be obtained from that source.
template <typename ReadNoteT>
std::optional<std::vector<uint8_t>>
unique_gnu_build_id(const target_elf& elf, ReadNoteT&& read_note)
{
    auto result = std::optional<std::vector<uint8_t>>{};

    for(const auto& segment : elf.reader.segments)
    {
        if(segment == nullptr || segment->get_type() != PT_NOTE) continue;

        auto note = to_phdr(*segment);
        // Bounds every read of this note, including the pointer arithmetic in
        // the file-backed read_note, and keeps both sources describing the same
        // bytes so their Build IDs are comparable.
        if(note.p_filesz == 0 || note.p_filesz > MAX_ELF_NOTE_SIZE ||
           note.p_memsz < note.p_filesz ||
           !has_range(elf.data.size(), note.p_offset, note.p_filesz) ||
           !note_is_file_backed_load(elf, note))
        {
            continue;
        }

        auto note_data = read_note(note);
        if(!note_data) continue;

        auto build_id = parse_gnu_build_id(note_data->data(), note_data->size());
        if(!build_id) continue;

        if(result && *result != *build_id) return std::nullopt;
        result = std::move(*build_id);
    }

    return result;
}

std::optional<std::vector<uint8_t>>
build_id_from_file(const target_elf& elf)
{
    return unique_gnu_build_id(elf, [&elf](const Elf64_Phdr& note) {
        const auto* begin = elf.data.data() + note.p_offset;
        return std::optional<std::vector<uint8_t>>{std::in_place, begin, begin + note.p_filesz};
    });
}

bool
range_is_readable_mapping(const mapped_object& object, uint64_t address, uint64_t size)
{
    auto end = checked_add(address, size);
    if(!end) return false;

    return std::any_of(object.mappings.begin(), object.mappings.end(), [&](const auto& mapping) {
        return mapping.permissions.find('r') != std::string::npos && address >= mapping.start &&
               *end <= mapping.end;
    });
}

std::optional<std::vector<uint8_t>>
read_target_memory(pid_t pid, uint64_t address, size_t size)
{
    if(size == 0 || size > MAX_TARGET_READ_SIZE || !checked_add(address, size))
    {
        return std::nullopt;
    }

    auto data = std::vector<uint8_t>(size);
    auto read = size_t{0};
    while(read < size)
    {
        auto remote_address = checked_add(address, read);
        if(!remote_address) return std::nullopt;

        auto local = iovec{data.data() + read, size - read};
        auto remote =
            // process_vm_readv represents the remote process address as a pointer.
            // NOLINTNEXTLINE(performance-no-int-to-ptr)
            iovec{reinterpret_cast<void*>(static_cast<uintptr_t>(*remote_address)), size - read};
        auto ret = ::process_vm_readv(pid, &local, 1, &remote, 1, 0);
        if(ret < 0)
        {
            if(errno == EINTR) continue;
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] Could not read target memory at 0x"
                       << std::hex << *remote_address << std::dec << ": " << std::strerror(errno);
            return std::nullopt;
        }
        if(ret == 0) return std::nullopt;
        read += static_cast<size_t>(ret);
    }
    return data;
}

std::optional<std::vector<uint8_t>>
build_id_from_target_memory(pid_t                pid,
                            const target_elf&    elf,
                            const mapped_object& object,
                            uint64_t             load_bias)
{
    return unique_gnu_build_id(elf, [&](const Elf64_Phdr& note) {
        auto runtime_address = checked_add(load_bias, note.p_vaddr);
        if(!runtime_address || !range_is_readable_mapping(object, *runtime_address, note.p_filesz))
        {
            return std::optional<std::vector<uint8_t>>{};
        }
        return read_target_memory(pid, *runtime_address, static_cast<size_t>(note.p_filesz));
    });
}

// Runs only when the file's device/inode did not match the mapping, which is
// the case for a file found by pathname under /proc/<pid>/root after an overlay
// or bind mount gave the same content a different inode. Matching Build IDs
// then say the file and the mapped image are the same build. That is an
// identity check, not an integrity check: the linker writes the Build ID once
// and nothing recomputes it, so a library whose bytes are edited after linking
// keeps its original note and is still accepted here.
bool
build_id_matches_target(pid_t                pid,
                        const target_elf&    elf,
                        const mapped_object& object,
                        uint64_t             load_bias)
{
    auto file_build_id = build_id_from_file(elf);
    if(!file_build_id)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Cannot validate " << elf.path
                     << " against the mapped object because it has no unique mapped GNU Build ID";
        return false;
    }

    auto target_build_id = build_id_from_target_memory(pid, elf, object, load_bias);
    if(!target_build_id)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Cannot validate " << elf.path
                     << " because no unique GNU Build ID could be read from the target mapping";
        return false;
    }

    if(*target_build_id != *file_build_id)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Refusing to use " << elf.path
                     << " because its GNU Build ID does not match the target's mapped image";
        return false;
    }
    return true;
}

std::optional<uint64_t>
vaddr_to_offset(const target_elf& elf, uint64_t vaddr, uint64_t size)
{
    for(const auto& segment : elf.load_segments)
    {
        if(vaddr < segment.p_vaddr) continue;
        auto delta = vaddr - segment.p_vaddr;
        if(delta > segment.p_filesz || size > segment.p_filesz - delta) continue;
        return checked_add(segment.p_offset, delta);
    }
    return std::nullopt;
}

bool
symbol_is_supported_function(unsigned char bind,
                             unsigned char type,
                             Elf64_Section section_index,
                             unsigned char other)
{
    auto visibility = ELF64_ST_VISIBILITY(other);

    return section_index != SHN_UNDEF && section_index != SHN_ABS && type == STT_FUNC &&
           (bind == STB_GLOBAL || bind == STB_WEAK) &&
           (visibility == STV_DEFAULT || visibility == STV_PROTECTED);
}

bool
symbol_is_supported_function(const Elf64_Sym& sym)
{
    return symbol_is_supported_function(
        ELF64_ST_BIND(sym.st_info), ELF64_ST_TYPE(sym.st_info), sym.st_shndx, sym.st_other);
}

std::optional<uint64_t>
lookup_symbol_from_sections(const target_elf& elf, std::string_view symbol_name)
{
    // Prefer section-backed .dynsym lookup when section headers are present;
    // sectionless files fall back to PT_DYNAMIC.
    for(const auto& section : elf.reader.sections)
    {
        if(section == nullptr || section->get_type() != SHT_DYNSYM) continue;

        ELFIO::const_symbol_section_accessor symbols{elf.reader, section.get()};
        auto                                 symbol_count = symbols.get_symbols_num();
        if(symbol_count > MAX_DYNAMIC_SYMBOLS) return std::nullopt;

        for(ELFIO::Elf_Xword i = 0; i < symbol_count; ++i)
        {
            auto name  = std::string{};
            auto value = ELFIO::Elf64_Addr{0};
            auto size  = ELFIO::Elf_Xword{0};
            auto bind  = static_cast<unsigned char>(0);
            auto type  = static_cast<unsigned char>(0);
            auto index = ELFIO::Elf_Half{0};
            auto other = static_cast<unsigned char>(0);

            if(!symbols.get_symbol(i, name, value, size, bind, type, index, other))
            {
                continue;
            }

            if(name == symbol_name && symbol_is_supported_function(bind, type, index, other))
            {
                return value;
            }
        }
    }

    return std::nullopt;
}

std::optional<size_t>
symbol_count_from_sysv_hash(const target_elf& elf, uint64_t hash_vaddr)
{
    auto hash_offset = vaddr_to_offset(elf, hash_vaddr, 2 * sizeof(uint32_t));
    if(!hash_offset) return std::nullopt;

    auto nchain_offset = checked_add(*hash_offset, sizeof(uint32_t));
    if(!nchain_offset) return std::nullopt;
    auto nchain = read_as<uint32_t>(elf.data, *nchain_offset);
    if(!nchain) return std::nullopt;
    if(*nchain > MAX_DYNAMIC_SYMBOLS) return std::nullopt;
    return static_cast<size_t>(*nchain);
}

std::optional<size_t>
symbol_count_from_gnu_hash(const target_elf& elf, uint64_t hash_vaddr)
{
    auto hash_offset = vaddr_to_offset(elf, hash_vaddr, 4 * sizeof(uint32_t));
    if(!hash_offset) return std::nullopt;

    auto nbuckets      = read_as<uint32_t>(elf.data, *hash_offset);
    auto symndx_offset = checked_add(*hash_offset, sizeof(uint32_t));
    auto mask_offset   = checked_add(*hash_offset, 2 * sizeof(uint32_t));
    if(!symndx_offset || !mask_offset) return std::nullopt;
    auto symndx    = read_as<uint32_t>(elf.data, *symndx_offset);
    auto maskwords = read_as<uint32_t>(elf.data, *mask_offset);
    if(!nbuckets || !symndx || !maskwords) return std::nullopt;

    auto bloom_size = checked_mul(*maskwords, sizeof(uint64_t));
    auto bloom_base = checked_add(*hash_offset, 4 * sizeof(uint32_t));
    auto buckets_offset =
        (bloom_base && bloom_size) ? checked_add(*bloom_base, *bloom_size) : std::nullopt;
    auto buckets_size = checked_mul(*nbuckets, sizeof(uint32_t));
    if(!buckets_offset || !buckets_size ||
       !has_range(elf.data.size(), *buckets_offset, *buckets_size))
    {
        return std::nullopt;
    }

    auto max_bucket = uint32_t{0};
    for(uint32_t i = 0; i < *nbuckets; ++i)
    {
        auto bucket_delta = checked_mul(i, sizeof(uint32_t));
        auto bucket_offset =
            bucket_delta ? checked_add(*buckets_offset, *bucket_delta) : std::nullopt;
        if(!bucket_offset) return std::nullopt;
        auto bucket = read_as<uint32_t>(elf.data, *bucket_offset);
        if(bucket) max_bucket = std::max(max_bucket, *bucket);
    }
    if(max_bucket < *symndx) return *symndx;

    auto chains_offset = checked_add(*buckets_offset, *buckets_size);
    if(!chains_offset) return std::nullopt;
    auto chain_index = static_cast<uint64_t>(max_bucket - *symndx);
    for(size_t steps = 0; steps < MAX_GNU_HASH_CHAIN_STEPS; ++steps)
    {
        auto chain_delta  = checked_mul(chain_index, sizeof(uint32_t));
        auto chain_offset = chain_delta ? checked_add(*chains_offset, *chain_delta) : std::nullopt;
        if(!chain_offset || !has_range(elf.data.size(), *chain_offset, sizeof(uint32_t)))
        {
            return std::nullopt;
        }
        auto chain = read_as<uint32_t>(elf.data, *chain_offset);
        if(!chain) return std::nullopt;
        if((*chain & 1U) != 0) return static_cast<size_t>(*symndx + chain_index + 1);
        ++chain_index;
    }
    return std::nullopt;
}

std::optional<uint64_t>
lookup_symbol_from_pt_dynamic(const target_elf& elf, std::string_view symbol_name)
{
    auto dynamic_segment = std::optional<Elf64_Phdr>{};

    for(const auto& segment : elf.reader.segments)
    {
        if(segment != nullptr && segment->get_type() == PT_DYNAMIC)
        {
            dynamic_segment = to_phdr(*segment);
            break;
        }
    }
    if(!dynamic_segment ||
       !has_range(elf.data.size(), dynamic_segment->p_offset, dynamic_segment->p_filesz))
    {
        return std::nullopt;
    }

    // Sectionless ELF files still carry dynamic-loader metadata in PT_DYNAMIC.
    // The hash tables bound how many dynamic symbols we can scan safely.
    auto symtab_vaddr   = std::optional<uint64_t>{};
    auto strtab_vaddr   = std::optional<uint64_t>{};
    auto strsz          = std::optional<uint64_t>{};
    auto syment         = uint64_t{sizeof(Elf64_Sym)};
    auto hash_vaddr     = std::optional<uint64_t>{};
    auto gnu_hash_vaddr = std::optional<uint64_t>{};

    auto entries = dynamic_segment->p_filesz / sizeof(Elf64_Dyn);
    for(size_t i = 0; i < entries; ++i)
    {
        auto dyn_delta = checked_mul(i, sizeof(Elf64_Dyn));
        auto dyn_offset =
            dyn_delta ? checked_add(dynamic_segment->p_offset, *dyn_delta) : std::nullopt;
        if(!dyn_offset) return std::nullopt;
        auto dyn = read_as<Elf64_Dyn>(elf.data, *dyn_offset);
        if(!dyn) return std::nullopt;
        if(dyn->d_tag == DT_NULL) break;
        if(dyn->d_tag == DT_SYMTAB) symtab_vaddr = dyn->d_un.d_ptr;
        if(dyn->d_tag == DT_STRTAB) strtab_vaddr = dyn->d_un.d_ptr;
        if(dyn->d_tag == DT_STRSZ) strsz = dyn->d_un.d_val;
        if(dyn->d_tag == DT_SYMENT) syment = dyn->d_un.d_val;
        if(dyn->d_tag == DT_HASH) hash_vaddr = dyn->d_un.d_ptr;
        if(dyn->d_tag == DT_GNU_HASH) gnu_hash_vaddr = dyn->d_un.d_ptr;
    }

    if(!symtab_vaddr || !strtab_vaddr || !strsz || syment < sizeof(Elf64_Sym)) return std::nullopt;

    auto symbol_count = std::optional<size_t>{};
    if(hash_vaddr) symbol_count = symbol_count_from_sysv_hash(elf, *hash_vaddr);
    if(!symbol_count && gnu_hash_vaddr)
    {
        symbol_count = symbol_count_from_gnu_hash(elf, *gnu_hash_vaddr);
    }
    if(!symbol_count) return std::nullopt;
    if(*symbol_count > MAX_DYNAMIC_SYMBOLS) return std::nullopt;

    auto symtab_size = checked_mul(*symbol_count, syment);
    if(!symtab_size) return std::nullopt;
    auto symtab_offset = vaddr_to_offset(elf, *symtab_vaddr, *symtab_size);
    auto strtab_offset = vaddr_to_offset(elf, *strtab_vaddr, *strsz);
    if(!symtab_offset || !strtab_offset) return std::nullopt;

    for(size_t idx = 0; idx < *symbol_count; ++idx)
    {
        auto sym_delta  = checked_mul(idx, syment);
        auto sym_offset = sym_delta ? checked_add(*symtab_offset, *sym_delta) : std::nullopt;
        if(!sym_offset) return std::nullopt;
        auto sym = read_as<Elf64_Sym>(elf.data, *sym_offset);
        if(!sym || sym->st_name >= *strsz) continue;
        const auto* name =
            reinterpret_cast<const char*>(elf.data.data() + *strtab_offset + sym->st_name);
        auto max_name_len = *strsz - sym->st_name;
        if(std::string_view{name, strnlen(name, max_name_len)} == symbol_name &&
           symbol_is_supported_function(*sym))
        {
            return sym->st_value;
        }
    }

    return std::nullopt;
}

std::optional<uint64_t>
lookup_dynamic_symbol(const target_elf& elf, std::string_view symbol_name)
{
    if(auto sym = lookup_symbol_from_sections(elf, symbol_name)) return sym;
    return lookup_symbol_from_pt_dynamic(elf, symbol_name);
}

bool
address_in_executable_mapping(const mapped_object& object, uint64_t address)
{
    for(const auto& mapping : object.mappings)
    {
        if(mapping.permissions.find('x') == std::string::npos) continue;
        if(address >= mapping.start && address < mapping.end) return true;
    }
    return false;
}

std::optional<uint64_t>
resolve_target_symbol(pid_t                pid,
                      const mapped_object& object,
                      std::string_view     symbol,
                      bool                 pathname_only)
{
    auto elf = open_target_elf(pid, object, pathname_only);
    if(!elf || !parse_target_elf(*elf)) return std::nullopt;

    auto load_bias = calculate_load_bias(*elf, object);
    if(!load_bias) return std::nullopt;
    if(!elf->identity_matches_mapping && !build_id_matches_target(pid, *elf, object, *load_bias))
    {
        return std::nullopt;
    }

    auto sym = lookup_dynamic_symbol(*elf, symbol);
    if(!sym)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not find dynamic symbol " << symbol
                   << " in target ELF " << elf->path;
        return std::nullopt;
    }

    // Dynamic symbol values are ELF virtual addresses; adding load bias yields
    // the target-process address.
    auto address = checked_add(*load_bias, *sym);
    if(!address)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Resolved address for " << symbol << " in "
                   << elf->path << " overflowed";
        return std::nullopt;
    }
    if(!address_in_executable_mapping(object, *address))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Resolved address 0x" << std::hex << *address
                   << std::dec << " for " << symbol << " is not inside an executable mapping for "
                   << object.path;
        return std::nullopt;
    }

    ROCP_TRACE << "[rocprofiler-sdk-rocattach] Resolved " << symbol << " in target pid " << pid
               << " from " << elf->path << " (mapped as " << object.path << ", dev "
               << object.device_major << ":" << object.device_minor << ", inode " << object.inode
               << ") at 0x" << std::hex << *address << std::dec;
    return address;
}
}  // namespace

bool
find_symbol(int                target_pid,
            void*&             addr,
            const std::string& library,
            const std::string& symbol,
            bool               pathname_only)
{
    addr = nullptr;

    auto object = select_unique_mapped_object(target_pid, library);
    if(!object) return false;

    if(auto resolved = resolve_target_symbol(target_pid, *object, symbol, pathname_only))
    {
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        addr = reinterpret_cast<void*>(*resolved);
        return true;
    }

    ROCP_ERROR << "[rocprofiler-sdk-rocattach] Failed to resolve " << library << "::" << symbol
               << " from target pid " << target_pid << " ELF mappings";
    return false;
}
}  // namespace rocattach
}  // namespace rocprofiler
