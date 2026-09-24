// Copyright (c) Advanced Micro Devices, Inc.
// SPDX-License-Identifier:  MIT

#ifndef _GNU_SOURCE
#    define _GNU_SOURCE
#endif

#include "runtime.h"

#include "torch_abi.h"

#include <dlfcn.h>
#include <elf.h>
#include <link.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <optional>

namespace torch_abi
{
namespace
{

constexpr const char* kRecordFunctionNameSymbol = "_ZNK2at14RecordFunction4nameEv";
constexpr const char* kThreadLocalDebugInfoGetSymbol =
    "_ZN3c1020ThreadLocalDebugInfo3getENS_13DebugInfoKindE";
constexpr std::size_t kElfNoteAlignment = 4;

struct BuildIdLookup
{
    std::uintptr_t            symbol_address;
    std::optional<GnuBuildId> build_id;
};

struct AddressRange
{
    std::uintptr_t address;
    std::size_t    size;
};

constexpr std::optional<std::size_t> aligned_note_size(std::uint32_t size) noexcept
{
    const std::size_t converted = size;
    if (converted > std::numeric_limits<std::size_t>::max() - (kElfNoteAlignment - 1))
    {
        return std::nullopt;
    }
    return (converted + kElfNoteAlignment - 1) & ~(kElfNoteAlignment - 1);
}

[[nodiscard]] bool consume_segment_bytes(std::size_t amount, std::size_t segment_size, std::size_t& offset) noexcept
{
    if (offset > segment_size || amount > segment_size - offset)
    {
        return false;
    }
    offset += amount;
    return true;
}

std::optional<GnuBuildId> build_id_from_note_segment(const std::byte* notes, std::size_t segment_size) noexcept
{
    std::size_t offset = 0;
    while (segment_size - offset >= sizeof(ElfW(Nhdr)))
    {
        ElfW(Nhdr) header{};
        std::memcpy(&header, notes + offset, sizeof(header));
        offset += sizeof(header);

        const auto name_size = aligned_note_size(header.n_namesz);
        const auto desc_size = aligned_note_size(header.n_descsz);
        if (!name_size.has_value() || !desc_size.has_value() || *name_size > segment_size - offset)
        {
            return std::nullopt;
        }
        const std::byte* name = notes + offset;
        if (!consume_segment_bytes(*name_size, segment_size, offset) || *desc_size > segment_size - offset)
        {
            return std::nullopt;
        }
        const std::byte* description = notes + offset;
        if (!consume_segment_bytes(*desc_size, segment_size, offset))
        {
            return std::nullopt;
        }

        constexpr char gnu_note_name[] = "GNU";
        if (header.n_type != NT_GNU_BUILD_ID || header.n_namesz != sizeof(gnu_note_name) ||
            header.n_descsz != kGnuBuildIdSize ||
            std::memcmp(name, gnu_note_name, sizeof(gnu_note_name)) != 0)
        {
            continue;
        }

        GnuBuildId build_id{};
        std::memcpy(build_id.data(), description, build_id.size());
        return build_id;
    }
    return std::nullopt;
}

bool load_segment_contains(const dl_phdr_info& object,
                           const ElfW(Phdr) & segment,
                           const AddressRange& range) noexcept
{
    if (segment.p_type != PT_LOAD ||
        segment.p_vaddr > std::numeric_limits<std::uintptr_t>::max() - object.dlpi_addr)
    {
        return false;
    }
    const std::uintptr_t start = object.dlpi_addr + segment.p_vaddr;
    if (range.address < start)
    {
        return false;
    }
    const std::uintptr_t offset = range.address - start;
    return offset <= segment.p_memsz && range.size <= segment.p_memsz - offset;
}

bool object_contains_address(const dl_phdr_info& object, std::uintptr_t address) noexcept
{
    const auto* headers = object.dlpi_phdr;
    if (headers == nullptr)
    {
        return false;
    }

    return std::any_of(headers,
                       headers + object.dlpi_phnum,
                       [&object, address](const ElfW(Phdr) & segment)
                       { return load_segment_contains(object, segment, AddressRange{address, 1}); });
}

std::optional<GnuBuildId> mapped_build_id(const dl_phdr_info& object) noexcept
{
    const auto* headers = object.dlpi_phdr;
    if (headers == nullptr)
    {
        return std::nullopt;
    }

    for (std::size_t index = 0; index < object.dlpi_phnum; ++index)
    {
        const auto& segment      = headers[index];
        const auto  segment_size = std::min<std::size_t>(segment.p_filesz, segment.p_memsz);
        if (segment.p_type != PT_NOTE ||
            segment.p_vaddr > std::numeric_limits<std::uintptr_t>::max() - object.dlpi_addr ||
            segment_size == 0)
        {
            continue;
        }
        const std::uintptr_t note_address = object.dlpi_addr + segment.p_vaddr;
        const auto contains_note = [&object, note_address, segment_size](const ElfW(Phdr) & load_segment)
        {
            return load_segment_contains(object, load_segment, AddressRange{note_address, segment_size});
        };
        const bool note_is_mapped = std::any_of(headers, headers + object.dlpi_phnum, contains_note);
        if (!note_is_mapped)
        {
            continue;
        }
        const auto* notes = reinterpret_cast<const std::byte*>(note_address);  // NOLINT(performance-no-int-to-ptr)
        if (auto build_id = build_id_from_note_segment(notes, segment_size); build_id.has_value())
        {
            return build_id;
        }
    }
    return std::nullopt;
}

int visit_object_for_provider_build_id(dl_phdr_info* object, std::size_t, void* state_pointer) noexcept
{
    if (object == nullptr || state_pointer == nullptr)
    {
        return 0;
    }
    auto& state = *static_cast<BuildIdLookup*>(state_pointer);
    if (!object_contains_address(*object, state.symbol_address))
    {
        return 0;
    }

    state.build_id = mapped_build_id(*object);
    return 1;
}

std::optional<GnuBuildId> provider_build_id(const char* symbol_name) noexcept
{
    void* symbol = dlsym(RTLD_DEFAULT, symbol_name);
    if (symbol == nullptr)
    {
        return std::nullopt;
    }

    BuildIdLookup lookup{reinterpret_cast<std::uintptr_t>(symbol), std::nullopt};
    dl_iterate_phdr(visit_object_for_provider_build_id, &lookup);
    return lookup.build_id;
}

}  // namespace

bool runtime_pair_is_supported(const GnuBuildId& torch_cpu_build_id, const GnuBuildId& c10_build_id) noexcept
{
    return std::any_of(kSupportedRuntimes.begin(),
                       kSupportedRuntimes.end(),
                       [&torch_cpu_build_id, &c10_build_id](const SupportedRuntime& runtime)
                       {
                           return runtime.torch_cpu_build_id == torch_cpu_build_id &&
                                  runtime.c10_build_id == c10_build_id;
                       });
}

bool runtime_is_supported() noexcept
{
    const auto torch_cpu_build_id = provider_build_id(kRecordFunctionNameSymbol);
    const auto c10_build_id       = provider_build_id(kThreadLocalDebugInfoGetSymbol);
    if (!torch_cpu_build_id.has_value() || !c10_build_id.has_value())
    {
        return false;
    }

    return runtime_pair_is_supported(*torch_cpu_build_id, *c10_build_id);
}

}  // namespace torch_abi
