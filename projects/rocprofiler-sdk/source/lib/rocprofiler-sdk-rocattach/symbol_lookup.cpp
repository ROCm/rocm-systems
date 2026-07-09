// MIT License
//
// Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/common/logging.hpp"

#include <fmt/format.h>

#include <elf.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/sysmacros.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
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
// Keep limits generous enough for debug builds, but bounded so malformed target
// ELFs cannot force unbounded memory use or symbol/hash traversal.
constexpr auto MAX_TARGET_ELF_SIZE      = uint64_t{512} * 1024 * 1024;
constexpr auto MAX_DYNAMIC_SYMBOLS      = size_t{1} << 24;
constexpr auto MAX_GNU_HASH_CHAIN_STEPS = size_t{1} << 24;

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

struct target_elf
{
    std::string             path          = {};
    std::vector<uint8_t>    data          = {};
    std::vector<Elf64_Phdr> load_segments = {};
};

class unique_fd
{
public:
    explicit unique_fd(int fd)
    : m_fd{fd}
    {}

    unique_fd(const unique_fd&) = delete;
    unique_fd& operator=(const unique_fd&) = delete;

    unique_fd(unique_fd&& other) noexcept
    : m_fd{std::exchange(other.m_fd, -1)}
    {}

    unique_fd& operator=(unique_fd&& other) noexcept
    {
        if(this != &other)
        {
            reset();
            m_fd = std::exchange(other.m_fd, -1);
        }
        return *this;
    }

    ~unique_fd() { reset(); }

    int get() const { return m_fd; }

    void reset()
    {
        if(m_fd >= 0)
        {
            ::close(m_fd);
            m_fd = -1;
        }
    }

private:
    int m_fd = -1;
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

uint64_t
align_down(uint64_t value, uint64_t alignment)
{
    return value - (value % alignment);
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
    auto maps     = std::ifstream{filename};
    auto result   = std::vector<memory_mapping>{};

    if(!maps)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't open " << filename;
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
    auto objects = std::unordered_map<std::string, mapped_object>{};

    for(const auto& mapping : parse_maps(pid))
    {
        if(mapping.inode == 0 || !library_name_matches(mapping, library)) continue;

        auto key =
            fmt::format("{}:{}:{}", mapping.device_major, mapping.device_minor, mapping.inode);
        auto& object        = objects[key];
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

std::optional<target_elf>
read_file_if_matches_mapping(const std::string& path, const mapped_object& object)
{
    struct stat st
    {};

    auto raw_fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if(raw_fd < 0) return std::nullopt;
    auto fd = unique_fd{raw_fd};

    if(::fstat(fd.get(), &st) != 0)
    {
        ROCP_WARNING << "[rocprofiler-sdk-rocattach] Could not stat opened target ELF " << path
                     << ": " << std::strerror(errno);
        return std::nullopt;
    }

    if(major(st.st_dev) != object.device_major || minor(st.st_dev) != object.device_minor ||
       static_cast<uint64_t>(st.st_ino) != object.inode)
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

    auto data = read_file_from_fd(fd.get(), path, static_cast<uint64_t>(st.st_size));
    if(!data || data->empty()) return std::nullopt;

    return target_elf{path, std::move(*data), {}};
}

std::optional<target_elf>
open_target_elf(pid_t pid, const mapped_object& object)
{
    for(const auto& mapping : object.mappings)
    {
        auto map_files_path = fmt::format("/proc/{}/map_files/{:x}-{:x}",
                                          pid,
                                          static_cast<uint64_t>(mapping.start),
                                          static_cast<uint64_t>(mapping.end));
        if(auto elf = read_file_if_matches_mapping(map_files_path, object))
        {
            ROCP_TRACE << "[rocprofiler-sdk-rocattach] Opened target ELF via " << map_files_path;
            return elf;
        }
    }

    auto target_path = strip_deleted_suffix(object.path);
    if(!target_path.empty() && target_path.front() == '/')
    {
        auto root_path = (std::filesystem::path{fmt::format("/proc/{}/root", pid)} /
                          std::filesystem::path{target_path}.relative_path())
                             .string();
        if(auto elf = read_file_if_matches_mapping(root_path, object))
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
parse_elf_headers(target_elf& elf)
{
    auto ehdr = read_as<Elf64_Ehdr>(elf.data, 0);
    if(!ehdr)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Target ELF is too small: " << elf.path;
        return false;
    }

    if(std::memcmp(ehdr->e_ident, ELFMAG, SELFMAG) != 0 || ehdr->e_ident[EI_CLASS] != ELFCLASS64 ||
       ehdr->e_ident[EI_DATA] != ELFDATA2LSB || ehdr->e_machine != EM_X86_64 ||
       ehdr->e_phentsize != sizeof(Elf64_Phdr))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Unsupported target ELF format: " << elf.path;
        return false;
    }

    auto phdr_table_size = checked_mul(ehdr->e_phnum, sizeof(Elf64_Phdr));
    if(ehdr->e_phnum == 0 || !phdr_table_size ||
       !has_range(elf.data.size(), ehdr->e_phoff, *phdr_table_size))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Invalid program header table in " << elf.path;
        return false;
    }

    elf.load_segments.clear();
    for(size_t i = 0; i < ehdr->e_phnum; ++i)
    {
        auto phdr = read_as<Elf64_Phdr>(elf.data, ehdr->e_phoff + i * sizeof(Elf64_Phdr));
        if(!phdr) return false;
        if(phdr->p_type == PT_LOAD) elf.load_segments.emplace_back(*phdr);
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

    auto candidates = std::vector<uint64_t>{};

    for(const auto& mapping : object.mappings)
    {
        for(const auto& segment : elf.load_segments)
        {
            auto segment_file_page    = align_down(segment.p_offset, page_size);
            auto segment_virtual_page = align_down(segment.p_vaddr, page_size);
            if(mapping.file_offset != segment_file_page) continue;

            auto candidate =
                checked_sub(static_cast<uint64_t>(mapping.start), segment_virtual_page);
            if(!candidate)
            {
                ROCP_ERROR << "[rocprofiler-sdk-rocattach] Invalid target mapping/segment pair for "
                           << object.path << ": mapping start is below segment virtual page";
                return std::nullopt;
            }

            if(std::find(candidates.begin(), candidates.end(), *candidate) == candidates.end())
            {
                candidates.emplace_back(*candidate);
            }
        }
    }

    auto best_bias  = std::optional<uint64_t>{};
    auto best_score = size_t{0};
    auto is_tied    = false;
    for(auto candidate : candidates)
    {
        auto score = size_t{0};
        for(const auto& segment : elf.load_segments)
        {
            auto segment_file_page    = align_down(segment.p_offset, page_size);
            auto segment_virtual_page = align_down(segment.p_vaddr, page_size);
            auto expected_start       = checked_add(candidate, segment_virtual_page);
            if(!expected_start) continue;

            auto mapping_itr = std::find_if(
                object.mappings.begin(), object.mappings.end(), [&](const auto& mapping) {
                    return mapping.file_offset == segment_file_page &&
                           static_cast<uint64_t>(mapping.start) == *expected_start;
                });
            if(mapping_itr != object.mappings.end()) ++score;
        }

        if(score > best_score)
        {
            best_bias  = candidate;
            best_score = score;
            is_tied    = false;
        }
        else if(score == best_score && score > 0)
        {
            is_tied = true;
        }
    }

    if(best_bias && !is_tied)
    {
        ROCP_TRACE << "[rocprofiler-sdk-rocattach] Calculated target load bias for " << object.path
                   << " as 0x" << std::hex << *best_bias << std::dec << " using " << best_score
                   << " PT_LOAD mapping matches";
        return best_bias;
    }

    if(is_tied)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Ambiguous target load-bias candidates for "
                   << object.path;
        return std::nullopt;
    }

    ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not match target mappings for " << object.path
               << " to ELF PT_LOAD segments";
    return std::nullopt;
}

std::optional<uint64_t>
vaddr_to_offset(const target_elf& elf, uint64_t vaddr, uint64_t size)
{
    for(const auto& segment : elf.load_segments)
    {
        if(segment.p_type != PT_LOAD) continue;
        if(vaddr < segment.p_vaddr) continue;
        auto delta = vaddr - segment.p_vaddr;
        if(delta > segment.p_filesz || size > segment.p_filesz - delta) continue;
        return checked_add(segment.p_offset, delta);
    }
    return std::nullopt;
}

bool
symbol_is_supported_function(const Elf64_Sym& sym)
{
    auto type       = ELF64_ST_TYPE(sym.st_info);
    auto bind       = ELF64_ST_BIND(sym.st_info);
    auto visibility = ELF64_ST_VISIBILITY(sym.st_other);

    return sym.st_shndx != SHN_UNDEF && type == STT_FUNC &&
           (bind == STB_GLOBAL || bind == STB_WEAK) &&
           (visibility == STV_DEFAULT || visibility == STV_PROTECTED);
}

std::optional<Elf64_Sym>
lookup_symbol_from_sections(const target_elf& elf, std::string_view symbol_name)
{
    auto ehdr            = read_as<Elf64_Ehdr>(elf.data, 0);
    auto shdr_table_size = ehdr ? checked_mul(ehdr->e_shnum, sizeof(Elf64_Shdr)) : std::nullopt;
    if(!ehdr || ehdr->e_shoff == 0 || ehdr->e_shnum == 0 ||
       ehdr->e_shentsize != sizeof(Elf64_Shdr) || !shdr_table_size ||
       !has_range(elf.data.size(), ehdr->e_shoff, *shdr_table_size))
    {
        return std::nullopt;
    }

    for(size_t i = 0; i < ehdr->e_shnum; ++i)
    {
        auto shdr = read_as<Elf64_Shdr>(elf.data, ehdr->e_shoff + i * sizeof(Elf64_Shdr));
        if(!shdr || shdr->sh_type != SHT_DYNSYM || shdr->sh_entsize < sizeof(Elf64_Sym))
        {
            continue;
        }
        if(shdr->sh_link >= ehdr->e_shnum) continue;

        auto strtab =
            read_as<Elf64_Shdr>(elf.data, ehdr->e_shoff + shdr->sh_link * sizeof(Elf64_Shdr));
        if(!strtab || !has_range(elf.data.size(), strtab->sh_offset, strtab->sh_size) ||
           !has_range(elf.data.size(), shdr->sh_offset, shdr->sh_size))
        {
            continue;
        }

        auto symbol_count = shdr->sh_size / shdr->sh_entsize;
        if(symbol_count > MAX_DYNAMIC_SYMBOLS) return std::nullopt;
        for(size_t idx = 0; idx < symbol_count; ++idx)
        {
            auto sym_delta  = checked_mul(idx, shdr->sh_entsize);
            auto sym_offset = sym_delta ? checked_add(shdr->sh_offset, *sym_delta) : std::nullopt;
            if(!sym_offset) return std::nullopt;
            auto sym = read_as<Elf64_Sym>(elf.data, *sym_offset);
            if(!sym || sym->st_name >= strtab->sh_size) continue;
            const auto* name =
                reinterpret_cast<const char*>(elf.data.data() + strtab->sh_offset + sym->st_name);
            auto max_name_len = strtab->sh_size - sym->st_name;
            if(std::string_view{name, strnlen(name, max_name_len)} == symbol_name &&
               symbol_is_supported_function(*sym))
            {
                return sym;
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

std::optional<Elf64_Sym>
lookup_symbol_from_dynamic(const target_elf& elf, std::string_view symbol_name)
{
    auto dynamic_segment = std::optional<Elf64_Phdr>{};
    auto ehdr            = read_as<Elf64_Ehdr>(elf.data, 0);
    if(!ehdr) return std::nullopt;

    for(size_t i = 0; i < ehdr->e_phnum; ++i)
    {
        auto phdr = read_as<Elf64_Phdr>(elf.data, ehdr->e_phoff + i * sizeof(Elf64_Phdr));
        if(phdr && phdr->p_type == PT_DYNAMIC)
        {
            dynamic_segment = *phdr;
            break;
        }
    }
    if(!dynamic_segment ||
       !has_range(elf.data.size(), dynamic_segment->p_offset, dynamic_segment->p_filesz))
    {
        return std::nullopt;
    }

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
            return sym;
        }
    }

    return std::nullopt;
}

std::optional<Elf64_Sym>
lookup_dynamic_symbol(const target_elf& elf, std::string_view symbol_name)
{
    if(auto sym = lookup_symbol_from_sections(elf, symbol_name)) return sym;
    return lookup_symbol_from_dynamic(elf, symbol_name);
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
resolve_target_symbol(pid_t pid, const mapped_object& object, std::string_view symbol)
{
    auto elf = open_target_elf(pid, object);
    if(!elf || !parse_elf_headers(*elf)) return std::nullopt;

    auto load_bias = calculate_load_bias(*elf, object);
    if(!load_bias) return std::nullopt;

    auto sym = lookup_dynamic_symbol(*elf, symbol);
    if(!sym)
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Could not find dynamic symbol " << symbol
                   << " in target ELF " << elf->path;
        return std::nullopt;
    }

    if(!symbol_is_supported_function(*sym))
    {
        ROCP_ERROR << "[rocprofiler-sdk-rocattach] Symbol " << symbol
                   << " is not a supported exported function in " << elf->path;
        return std::nullopt;
    }

    auto address = checked_add(*load_bias, sym->st_value);
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
find_library(void*& addr, int inpid, const std::string& library)
{
    auto object = select_unique_mapped_object(inpid, library);
    if(!object) return false;

    for(const auto& mapping : object->mappings)
    {
        if(mapping.file_offset != 0) continue;
        // NOLINTNEXTLINE(performance-no-int-to-ptr)
        addr = reinterpret_cast<void*>(mapping.start);
        return true;
    }

    ROCP_ERROR << "[rocprofiler-sdk-rocattach] Couldn't find library " << library
               << " (with file offset 0) in /proc/" << inpid << "/maps";
    return false;
}

bool
find_symbol(int target_pid, void*& addr, const std::string& library, const std::string& symbol)
{
    addr = nullptr;

    auto object = select_unique_mapped_object(target_pid, library);
    if(!object) return false;

    if(auto resolved = resolve_target_symbol(target_pid, *object, symbol))
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
