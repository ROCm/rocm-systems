#pragma once

/**
 * @file proc_maps.hpp
 * @brief Read /proc/self/maps and look up the ELF mapping that covers
 *        a runtime instruction pointer. Used by symbolizer.hpp to
 *        translate IPs into "binary path + file offset" pairs suitable
 *        for addr2line.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include <elf.h>

#include "cpu_snap/core/self_exe.hpp"
#include "cpu_snap/processing/symbols/elf_symbols.hpp"

namespace cpu_snap::processing::symbols
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants; the .clang-tidy GlobalConstantCase
//   rule defaults to UPPER_CASE but the entire project ships with k_*.
namespace detail
{
/** /proc/self/maps names of the kernel-provided mappings that have no backing file. */
inline constexpr std::string_view k_vdso_path     = "[vdso]";
inline constexpr std::string_view k_vsyscall_path = "[vsyscall]";
/** Line buffer size when reading /proc/self/maps. */
inline constexpr std::size_t k_maps_line_max = 4096;
/** Backing-file path buffer size in /proc/self/maps line parsing. */
inline constexpr std::size_t k_maps_path_max = 2048;
/** rwxp permissions field buffer size (e.g. "r-xp\0"). */
inline constexpr std::size_t k_perms_buf = 8;
/** Device field buffer size (e.g. "fd:01\0"). */
inline constexpr std::size_t k_dev_buf = 16;
/** Minimum tokens sscanf must extract before the path (start,end,perms,off,dev,inode). */
inline constexpr int k_min_scan_tokens = 6;
/** Total tokens including the optional path. */
inline constexpr int k_full_scan_tokens = 7;

/**
 * @brief One PT_LOAD program header, reduced to the fields needed to
 *        translate a file offset into the ELF's linked virtual address.
 */
struct elf_load_segment
{
    std::uint64_t p_offset = 0;
    std::uint64_t p_vaddr  = 0;
    std::uint64_t p_filesz = 0;
};

/**
 * @brief Find the PT_LOAD segment covering @p file_off and translate it
 *        into an ELF virtual address.
 *
 * @param file_off ELF file offset (e.g. from a /proc/self/maps mapping).
 * @param segments PT_LOAD segments of the ELF, in any order.
 * @return ELF virtual address, or @c std::nullopt if no segment covers
 *         @p file_off.
 */
[[nodiscard]] constexpr std::optional<std::uint64_t> find_covering_vaddr(
    std::uint64_t file_off, std::span<const elf_load_segment> segments) noexcept
{
    const auto covers = [file_off](const elf_load_segment& seg)
    {
        return file_off >= seg.p_offset && file_off < seg.p_offset + seg.p_filesz;
    };
    const auto seg = std::ranges::find_if(segments, covers);
    if(seg == segments.end()) return std::nullopt;
    return file_off - seg->p_offset + seg->p_vaddr;
}

/**
 * @brief Translate a file offset into an ELF virtual address using the
 *        PT_LOAD segment that covers it.
 *
 * @param file_off ELF file offset (e.g. from a /proc/self/maps mapping).
 * @param segments PT_LOAD segments of the ELF, in any order.
 * @return ELF virtual address, or @p file_off unchanged if no segment
 *         covers it (e.g. @p segments is empty because the header could
 *         not be parsed).
 */
[[nodiscard]] constexpr std::uint64_t file_offset_to_vaddr(
    std::uint64_t file_off, std::span<const elf_load_segment> segments) noexcept
{
    return find_covering_vaddr(file_off, segments).value_or(file_off);
}

/**
 * @brief Read the PT_LOAD program headers of an ELF file.
 *
 * x86_64 only: rejects anything that is not a 64-bit little-endian ELF.
 * @param path Path to the ELF file.
 * @return PT_LOAD segments in program-header order; empty if @p path
 *         cannot be opened, is not a 64-bit LE ELF, or has no PT_LOAD
 *         headers.
 */
[[nodiscard]] inline std::vector<elf_load_segment> read_elf_load_segments(
    const std::string& path)
{
    std::vector<elf_load_segment> segments;

    const auto elf = open_elf64(path);
    if(!elf) return segments;

    FILE*             file = elf->file.get();
    const Elf64_Ehdr& ehdr = elf->ehdr;
    for(Elf64_Half i = 0; i < ehdr.e_phnum; ++i)
    {
        const auto phdr_pos = static_cast<std::int64_t>(
            ehdr.e_phoff + (static_cast<std::uint64_t>(i) * ehdr.e_phentsize));
        Elf64_Phdr phdr{};
        if(std::fseek(file, phdr_pos, SEEK_SET) != 0
           || std::fread(&phdr, sizeof(phdr), 1, file) != 1)
        {
            break;
        }
        if(phdr.p_type != PT_LOAD) continue;
        segments.push_back(elf_load_segment{ .p_offset = phdr.p_offset,
                                              .p_vaddr  = phdr.p_vaddr,
                                              .p_filesz = phdr.p_filesz, });
    }

    return segments;
}
}  // namespace detail
// NOLINTEND(readability-identifier-naming)

/**
 * @brief One executable region from /proc/self/maps.
 */
struct vm_mapping
{
    /** Start address of the mapping (inclusive). */
    std::uint64_t start       = 0;
    /** End address of the mapping (exclusive). */
    std::uint64_t end         = 0;
    /** ELF file offset of the byte at @c start. */
    std::uint64_t file_offset = 0;
    /** True if @c path equals /proc/self/exe. */
    bool          is_main_exe = false;
    /** Backing file path (e.g. "/usr/lib/libc.so.6"). */
    std::string   path;
};

/**
 * @brief Snapshot of executable mappings of the current process.
 */
class proc_maps
{
public:
    /** @brief Construct and read /proc/self/maps. */
    proc_maps() { reload(); }

    /** @brief Re-read /proc/self/maps. Old data is replaced. */
    void reload()
    {
        m_maps.clear();
        m_pseudo_maps.clear();
        m_main_exe.clear();
        m_segments_by_path.clear();
        m_symbols_by_path.clear();

        m_main_exe = core::self_exe_path();

        FILE* file = std::fopen("/proc/self/maps", "r");
        if(file == nullptr)
        {
            return;
        }

        std::array<char, detail::k_maps_line_max> line{};
        while(std::fgets(line.data(), static_cast<int>(line.size()), file) != nullptr)
        {
            std::uint64_t                              start_addr = 0;
            std::uint64_t                              end_addr   = 0;
            std::uint64_t                              off        = 0;
            std::array<char, detail::k_perms_buf>      perms{};
            std::array<char, detail::k_dev_buf>        dev{};
            std::array<char, detail::k_maps_path_max>  path{};
            std::uint64_t                              inode      = 0;
            // NOLINTNEXTLINE(cert-err34-c): /proc/self/maps format is kernel-defined, no error path needed
            const int matched = std::sscanf(line.data(),
                                            "%lx-%lx %7s %lx %15s %lu %2047[^\n]",
                                            &start_addr, &end_addr, perms.data(), &off,
                                            dev.data(), &inode, path.data());
            if(matched < detail::k_min_scan_tokens)
            {
                continue;
            }
            if(std::strchr(perms.data(), 'x') == nullptr)
            {
                continue;
            }

            std::string parsed_path =
                (matched == detail::k_full_scan_tokens) ? path.data() : "";
            while(!parsed_path.empty() && parsed_path[0] == ' ')
            {
                parsed_path.erase(parsed_path.begin());
            }
            if(parsed_path.empty())
            {
                continue;
            }
            const bool is_pseudo = parsed_path[0] == '[';
            if(is_pseudo && parsed_path != detail::k_vdso_path
               && parsed_path != detail::k_vsyscall_path)
            {
                continue;
            }

            vm_mapping mapping;
            mapping.start       = start_addr;
            mapping.end         = end_addr;
            mapping.file_offset = off;
            mapping.path        = std::move(parsed_path);
            mapping.is_main_exe = (mapping.path == m_main_exe);
            (is_pseudo ? m_pseudo_maps : m_maps).push_back(std::move(mapping));
        }
        (void) std::fclose(file);
    }

    /**
     * @brief Find the executable mapping that covers @p ip.
     * @param ip Runtime instruction pointer.
     * @return   Pointer to the owning mapping, or @c nullptr if none
     *           covers @p ip (vDSO / unmapped / kernel).
     */
    [[nodiscard]] const vm_mapping* find(std::uint64_t ip) const noexcept
    {
        for(const auto& mapping : m_maps)
        {
            if(ip >= mapping.start && ip < mapping.end)
            {
                return &mapping;
            }
        }
        return nullptr;
    }

    /**
     * @brief Find the file-less [vdso] / [vsyscall] mapping that covers @p ip.
     *
     * Kept apart from find() so that frame filters keep treating these
     * IPs as outside every module.
     * @param ip Runtime instruction pointer.
     * @return   Pointer to the pseudo mapping, or @c nullptr if none covers @p ip.
     */
    [[nodiscard]] const vm_mapping* find_pseudo(std::uint64_t ip) const noexcept
    {
        for(const auto& mapping : m_pseudo_maps)
        {
            if(ip >= mapping.start && ip < mapping.end)
            {
                return &mapping;
            }
        }
        return nullptr;
    }

    /**
     * @brief Translate a runtime IP into an ELF virtual address suitable
     *        for addr2line.
     *
     * Program headers are read once per binary path and cached; a
     * binary that cannot be parsed (missing file, non-64-bit ELF) falls
     * back to returning the file offset unchanged.
     *
     * @param ip      Runtime instruction pointer inside @p mapping.
     * @param mapping Owning mapping (from find()).
     * @return        ELF virtual address for @p ip.
     */
    [[nodiscard]] std::uint64_t ip_to_elf_vaddr(std::uint64_t ip,
                                                const vm_mapping& mapping) const
    {
        const std::uint64_t file_off = ip - mapping.start + mapping.file_offset;
        return detail::file_offset_to_vaddr(file_off, load_segments_for(mapping.path));
    }

    /**
     * @brief All cached executable mappings in /proc/self/maps order.
     * @return Reference valid until the next reload() call.
     */
    [[nodiscard]] const std::vector<vm_mapping>& mappings() const noexcept
    {
        return m_maps;
    }

    /**
     * @brief Absolute path of the main executable.
     * @return Reference valid until the next reload() call; empty on
     *         readlink failure (e.g. inside a chroot that strips /proc).
     */
    [[nodiscard]] const std::string& main_exe() const noexcept { return m_main_exe; }

    /**
     * @brief FUNC-like ELF symbols for @p path, reading and caching on
     *        first use.
     * @param path Binary path (from a vm_mapping).
     * @return Reference valid until the next reload() call; has an
     *         empty symbol list if the ELF could not be parsed.
     */
    [[nodiscard]] const detail::elf_symbol_table& symbols_for(const std::string& path) const
    {
        const auto found_it = m_symbols_by_path.find(path);
        if(found_it != m_symbols_by_path.end())
        {
            return found_it->second;
        }
        [[maybe_unused]] const auto [inserted_it, did_insert] =
            m_symbols_by_path.emplace(path, detail::read_elf_symbol_table(path));
        return inserted_it->second;
    }

private:
    /**
     * @brief PT_LOAD segments for @p path, reading and caching on first use.
     * @param path Binary path (from a vm_mapping).
     * @return Reference valid until the next reload() call; empty if the
     *         ELF header could not be parsed.
     */
    [[nodiscard]] const std::vector<detail::elf_load_segment>& load_segments_for(
        const std::string& path) const
    {
        const auto found_it = m_segments_by_path.find(path);
        if(found_it != m_segments_by_path.end())
        {
            return found_it->second;
        }
        [[maybe_unused]] const auto [inserted_it, did_insert] =
            m_segments_by_path.emplace(path, detail::read_elf_load_segments(path));
        return inserted_it->second;
    }

    std::vector<vm_mapping> m_maps;
    std::vector<vm_mapping> m_pseudo_maps;
    std::string             m_main_exe;
    mutable std::unordered_map<std::string, std::vector<detail::elf_load_segment>>
        m_segments_by_path;
    mutable std::unordered_map<std::string, detail::elf_symbol_table> m_symbols_by_path;
};

}  // namespace cpu_snap::processing::symbols
