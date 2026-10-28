#pragma once

/**
 * @file symbolizer.hpp
 * @brief Batched IP -> "function [binary]" string resolver. Bundles
 *        proc_maps with addr2line. Call preload() once with all unique
 *        IPs (one popen per binary instead of per IP), then look up
 *        via resolve() or operator().
 *
 * Default output format: `"function(args) [binary]"`, e.g.
 *   `"__nanosleep [libc.so.6]"` or `"main [myapp]"`.
 * The single knob is tunable via symbolizer_config:
 *   - include_binary_name=false drops the suffix: `"function(args)"`.
 *
 * Unresolved IPs follow the same toggle:
 *   - include_binary_name=true  -> `"0x<hex> [binary]"`
 *   - include_binary_name=false -> `"0x<hex>"`
 * When no /proc/self/maps mapping is found the output is always
 * `"0x<hex>"` (no binary to attach).
 *
 * Inline-frame expansion: each cache entry stores the full inline
 * call-chain returned by `addr2line -i` (innermost first). Use
 * resolve_inline() to surface the chain; resolve() keeps its
 * outer-name-only contract (returns the back() of the chain).
 */

#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "cpu_snap/core/format_hex.hpp"
#include "cpu_snap/core/module_map.hpp"
#include "cpu_snap/processing/symbols/addr2line_client.hpp"
#include "cpu_snap/processing/symbols/elf_symbols.hpp"
#include "cpu_snap/processing/symbols/proc_maps.hpp"
#include "cpu_snap/processing/symbols/symbol_name.hpp"

namespace cpu_snap::processing::symbols
{

/**
 * @brief Symbolizer formatting knob.
 *
 * Applies to the final per-IP string stored in the cache and returned
 * by resolve()/operator().
 */
struct symbolizer_config
{
    /** When true (default), append " [binary]" to resolved names, e.g.
     *  "function(args) [binary]". When false, the output is just
     *  "function(args)" with no suffix. */
    bool include_binary_name = true;

    /**
     * @brief Expand inline-call chains via addr2line -i.
     *
     * When true, resolve_inline(ip) returns the full innermost-first
     * chain of inlined frames the optimizer folded into the IP's
     * physical frame. When false (default), resolve_inline(ip) returns
     * a single-element vector with just the outer non-inlined function
     * name (matching resolve(ip)'s legacy behavior).
     *
     * Cost when true: trace size grows ~2-3x per sample (more frames
     * per IP). Symbolizer preload time unchanged (addr2line -ifCe is
     * invoked either way; only the parse + cache shape changes).
     *
     * Default false preserves the pre-feature behavior for callers
     * that don't opt in explicitly.
     */
    bool inline_frames = false;
};

/**
 * @brief Batched IP -> "function[binary]" resolver wrapping proc_maps
 *        and @c addr2line.
 *
 * See the file-level comment for the format toggles.
 */
class symbolizer
{
public:
    /** @brief Construct with default formatting (include_binary_name=true). */
    symbolizer() = default;

    /**
     * @brief Construct with explicit formatting knobs.
     * @param cfg Format policy applied to every cached string.
     */
    explicit symbolizer(symbolizer_config cfg) noexcept
    : m_cfg{ cfg }
    {}

    /**
     * @brief Construct in module_map mode: resolve IP -> module via a
     *        @c core::module_map instead of /proc/self/maps.
     *
     * This is the offline / cross-process path: @p modules typically
     * comes from decoding a capture's MAPS_INIT/MAPS_DELTA records
     * (core::sample_reconstructor), so it works in a process other
     * than the one that recorded the samples, and after that process
     * has exited.
     *
     * @param modules Time-aware module map. Must outlive this
     *                symbolizer.
     * @param cfg     Format policy applied to every cached string.
     */
    explicit symbolizer(const cpu_snap::core::module_map& modules,
                        symbolizer_config                 cfg = {}) noexcept
    : m_cfg{ cfg }
    , m_module_map{ &modules }
    {}

    /**
     * @brief Re-read @c /proc/self/maps.
     *
     * No-op in module_map mode (there is no live process to rescan).
     * Existing cache entries are preserved so previously resolved IPs
     * keep returning the same string.
     */
    void rescan_maps()
    {
        if(m_module_map == nullptr) m_maps.reload();
    }

    /**
     * @brief Resolve a batch of IPs in one pass.
     *
     * IPs are grouped by owning binary so @c addr2line is invoked at
     * most once per binary, not once per IP. Results land in the
     * internal cache.
     *
     * @param ips Instruction pointers to resolve. Already-cached IPs
     *            are skipped (idempotent).
     */
    void preload(const std::vector<std::uint64_t>& ips)
    {
        preload_impl(ips, [](std::uint64_t /*ip*/) { return std::uint64_t{ 0 }; });
    }

    /**
     * @brief Resolve a batch of (ip, sample timestamp_ns) pairs in one
     *        pass, using the module_map supplied at construction.
     *
     * Each ip is resolved against the module loaded at ITS OWN
     * timestamp (core::module_map::resolve), so a capture whose module
     * layout changed mid-stream (dlopen/dlclose) still resolves every
     * sample against the layout that was actually in effect when it was
     * taken. Only meaningful in module_map mode; in live-process mode
     * every timestamp is ignored (m_maps.find() has no time axis).
     *
     * @param ip_timestamps (ip, timestamp_ns) pairs. Already-cached ips
     *                      are skipped, matching preload(ips)'s
     *                      idempotent contract.
     *
     * @note Named preload_at (not an overload of preload()): a
     *       single-element braced-init-list argument such as
     *       `preload({ip})` is ambiguous between `vector<uint64_t>`
     *       and `vector<pair<uint64_t, uint64_t>>` under overload
     *       resolution, so the two are kept as distinctly-named
     *       methods instead.
     */
    void preload_at(const std::vector<std::pair<std::uint64_t, std::uint64_t>>& ip_timestamps)
    {
        std::unordered_map<std::uint64_t, std::uint64_t> timestamp_by_ip;
        std::vector<std::uint64_t>                        ips;
        ips.reserve(ip_timestamps.size());
        for(const auto& [ip, ts] : ip_timestamps)
        {
            timestamp_by_ip.emplace(ip, ts);
            ips.push_back(ip);
        }
        preload_impl(ips, [&timestamp_by_ip](std::uint64_t ip) {
            return timestamp_by_ip.at(ip);
        });
    }

    /**
     * @brief Cache lookup with no side effects.
     * @param ip Instruction pointer.
     * @return   Cached resolved string (OUTERMOST frame when an inline
     *           chain is present), or empty view when @p ip has not
     *           been through preload() / operator().
     */
    std::string_view resolve(std::uint64_t ip) const noexcept
    {
        if(auto it = m_cache.find(ip); it != m_cache.end() && !it->second.empty())
        {
            return std::string_view(it->second.back());
        }
        return {};
    }

    /**
     * @brief Return the inline-call chain for an IP.
     *
     * Returned vector is innermost-first: index 0 = the function the IP
     * lives in physically (often the only inlined leaf the optimizer
     * left); subsequent entries are each enclosing inlined frame; last
     * entry is the outer non-inlined function.
     *
     * When symbolizer_config::inline_frames is true, returns the full
     * chain. When false (default), returns a single-element vector with
     * just the outer non-inlined frame so callers see resolve()-equivalent
     * output without changing call sites.
     *
     * Empty vector if the IP has not been preloaded. Each string is
     * formatted per include_binary_name (matching resolve()).
     *
     * @param ip Instruction pointer (raw kernel sample value).
     * @return   Innermost-first vector of formatted frame strings.
     */
    [[nodiscard]] std::vector<std::string> resolve_inline(std::uint64_t ip) const
    {
        auto it = m_cache.find(ip);
        if(it == m_cache.end())
        {
            return {};
        }
        if(m_cfg.inline_frames)
        {
            return it->second;
        }
        if(it->second.empty())
        {
            return {};
        }
        return { it->second.back() };
    }

    /**
     * @brief Source location captured by addr2line for an IP.
     * @param ip Instruction pointer.
     * @return "file:line" string captured by addr2line for this IP, or
     *         empty view if unavailable.
     */
    std::string_view file_line(std::uint64_t ip) const noexcept
    {
        if(auto it = m_file_line.find(ip); it != m_file_line.end())
        {
            return std::string_view(it->second);
        }
        return {};
    }

    /**
     * @brief std::function-compatible single-IP resolver.
     *
     * Suitable for plugging directly into
     * sample_postprocessor_config::resolver and friends.
     *
     * @param ip Instruction pointer.
     * @return   The cached string, or an owned copy of the freshly
     *           resolved string. Falls back to a single-IP addr2line
     *           invocation if @p ip was not in the preload batch; in
     *           the worst case returns @c "0x<hex>".
     *
     * @note Each fallback path costs one @c popen; prefer preload() for
     *       bulk symbolisation.
     */
    std::string operator()(std::uint64_t ip)
    {
        if(auto it = m_cache.find(ip); it != m_cache.end() && !it->second.empty())
        {
            return it->second.back();
        }
        preload({ ip });
        if(auto it = m_cache.find(ip); it != m_cache.end() && !it->second.empty())
        {
            return it->second.back();
        }
        return core::format_hex(ip);
    }

    /**
     * @brief Symbol-identity key for @p ip, suitable for
     *        slice_emitter_config::symbol_identity.
     *
     * Two IPs share a key iff they fall in the same mapped module and
     * resolve to the same ELF symbol's [st_value, st_value + st_size)
     * range, so callers can treat different call sites inside one
     * function as the same stack frame without comparing display
     * strings. Reuses the same proc_maps mapping lookup and cached
     * per-module elf_symbol_table that correct_symbol_name() uses; no
     * extra ELF parsing.
     *
     * @param ip Instruction pointer.
     * @return   Opaque identity key, or @c std::nullopt if @p ip is
     *           outside every mapping or every known symbol's range.
     */
    [[nodiscard]] std::optional<std::uint64_t> symbol_identity(std::uint64_t ip) const
    {
        return symbol_identity(ip, 0);
    }

    /**
     * @brief Timestamp-aware overload of symbol_identity() for
     *        module_map mode.
     * @param ip           Instruction pointer.
     * @param timestamp_ns Sample timestamp; ignored in live-process mode.
     */
    [[nodiscard]] std::optional<std::uint64_t> symbol_identity(std::uint64_t ip,
                                                                std::uint64_t timestamp_ns) const
    {
        const auto mapping = resolve_mapping(ip, timestamp_ns);
        if(!mapping || is_pseudo_module(mapping->path))
        {
            return std::nullopt;
        }
        const std::uint64_t vaddr = m_maps.ip_to_elf_vaddr(ip, *mapping);
        const auto&         table = m_maps.symbols_for(mapping->path);
        const auto found = cpu_snap::processing::symbols::detail::find_symbol(vaddr, table.symbols);
        if(!found)
        {
            return std::nullopt;
        }

        std::size_t key = std::hash<std::string>{}(mapping->path);
        key ^= std::hash<std::uint64_t>{}(found->st_value) + 0x9e3779b97f4a7c15ULL
             + (key << 6) + (key >> 2);
        return static_cast<std::uint64_t>(key);
    }

    /**
     * @brief Apply the configured include_binary_name toggle to a
     *        `name`/`binary` pair.
     *
     * Exposed for unit testing the format policy independently of
     * addr2line/proc_maps.
     * @param name Function name.
     * @param base Binary basename.
     * @return Formatted "name [base]" or just "name" depending on the toggle.
     */
    [[nodiscard]] std::string format_with_binary(std::string_view name,
                                                 std::string_view base) const
    {
        if(!m_cfg.include_binary_name)
        {
            return std::string{ name };
        }
        std::string out;
        out.reserve(name.size() + base.size() + 3);
        out.assign(name);
        out += ' ';
        out += '[';
        out.append(base);
        out += ']';
        return out;
    }

private:
    /**
     * @brief Resolve which module covers @p ip, dispatching on
     *        construction mode.
     *
     * Module_map mode never touches /proc/self/maps: the module set
     * and its validity windows come entirely from @p timestamp_ns
     * against the module_map supplied at construction, which is
     * exactly the data a decoded MAPS_INIT/MAPS_DELTA stream carries.
     * That is what makes symbolizing a capture from another (possibly
     * no-longer-running) process possible. Live-process mode ignores
     * @p timestamp_ns; m_maps.find() reflects whatever the last
     * reload() saw.
     *
     * Either way the returned vm_mapping only feeds m_maps'
     * per-path ELF caches (ip_to_elf_vaddr / symbols_for), which are
     * pure functions of a binary path read from disk -- so the same
     * proc_maps instance backs both modes without ever needing a live
     * mapping list in module_map mode.
     */
    [[nodiscard]] std::optional<cpu_snap::processing::symbols::vm_mapping>
    resolve_mapping(std::uint64_t ip, std::uint64_t timestamp_ns) const
    {
        if(m_module_map != nullptr)
        {
            const auto mod = m_module_map->resolve(ip, timestamp_ns);
            if(!mod)
            {
                return std::nullopt;
            }
            return cpu_snap::processing::symbols::vm_mapping{ .start       = mod->base_va,
                                                  .end         = mod->base_va + mod->size,
                                                  .file_offset = mod->file_offset,
                                                  .is_main_exe = false,
                                                  .path        = mod->path };
        }
        const cpu_snap::processing::symbols::vm_mapping* mapping = m_maps.find(ip);
        if(mapping == nullptr)
        {
            mapping = m_maps.find_pseudo(ip);
        }
        if(mapping == nullptr)
        {
            return std::nullopt;
        }
        return *mapping;
    }

    /** Shared preload body; @p timestamp_of supplies the per-ip
     *  timestamp used only in module_map mode. */
    template <typename TimestampOf>
    void preload_impl(const std::vector<std::uint64_t>& ips, TimestampOf&& timestamp_of)
    {
        std::unordered_map<std::string,
                           std::vector<std::pair<std::uint64_t, std::uint64_t>>>
            by_path;

        for(const std::uint64_t ip : ips)
        {
            if(m_cache.contains(ip))
            {
                continue;
            }
            const auto mapping = resolve_mapping(ip, timestamp_of(ip));
            if(!mapping)
            {
                std::vector<std::string> chain;
                chain.push_back(core::format_hex(ip));
                m_cache.emplace(ip, std::move(chain));
                continue;
            }
            if(is_pseudo_module(mapping->path))
            {
                m_cache.emplace(ip, std::vector<std::string>{ fallback_hex_for(
                                        ip - mapping->start, basename_of(mapping->path)) });
                continue;
            }
            by_path[mapping->path].emplace_back(
                ip, m_maps.ip_to_elf_vaddr(ip, *mapping));
        }

        for(auto& [path, pairs] : by_path)
        {
            resolve_batch(path, pairs);
        }
    }

    [[nodiscard]] static std::string basename_of(std::string_view path)
    {
        const auto slash = path.find_last_of('/');
        return std::string{ slash == std::string_view::npos ? path
                                                             : path.substr(slash + 1) };
    }

    /** True for modules with no backing file to symbolize: the vDSO
     *  (as "[vdso]" from /proc/self/maps or "linux-vdso.so.1" from
     *  dl_iterate_phdr) and [vsyscall]. */
    [[nodiscard]] static bool is_pseudo_module(std::string_view path)
    {
        return path == detail::k_vdso_path || path == detail::k_vsyscall_path
               || basename_of(path) == "linux-vdso.so.1";
    }

    /** "<basename>+0x<addr>" fallback for the popen-failure path and for
     *  addresses outside every ELF symbol. */
    [[nodiscard]] static std::string
    fallback_hex_for(std::uint64_t addr, std::string_view base)
    {
        std::string out{ base };
        out += '+';
        out += core::format_hex(addr);
        return out;
    }

    /** Without .debug_info addr2line names an address after the nearest
     *  preceding symbol; use the sized ELF symbol containing @p vaddr,
     *  or <base>+0x<vaddr> if none. With .debug_info, trust @p addr2line_name. */
    [[nodiscard]] std::string correct_symbol_name(std::string_view path,
                                                  std::string_view base,
                                                  std::uint64_t     vaddr,
                                                  std::string_view  addr2line_name) const
    {
        const auto& table = m_maps.symbols_for(std::string{ path });
        if(table.has_debug_info)
        {
            return format_with_binary(addr2line_name, base);
        }

        const auto found = cpu_snap::processing::symbols::detail::find_symbol(vaddr, table.symbols);
        if(!found)
        {
            return fallback_hex_for(vaddr, base);
        }

        std::string elf_name = symbol_name::demangle(found->name);
        symbol_name::clean(elf_name);
        return format_with_binary(elf_name, base);
    }

    void resolve_batch(std::string_view                                            path,
                       const std::vector<std::pair<std::uint64_t, std::uint64_t>>& pairs)
    {
        const std::string base    = basename_of(path);
        auto              results = addr2line_client::resolve(path, pairs);
        for(std::size_t i = 0; i < pairs.size(); ++i)
        {
            const auto& [ip, vaddr] = pairs[i];
            if(!results[i])
            {
                m_cache.emplace(ip, std::vector<std::string>{ fallback_hex_for(ip, base) });
                continue;
            }
            store_chain(ip, vaddr, path, base, std::move(*results[i]));
        }
    }

    void store_chain(std::uint64_t     ip,
                     std::uint64_t     vaddr,
                     std::string_view  path,
                     std::string_view  base,
                     addr2line_chain&& raw)
    {
        std::vector<std::string> chain;
        chain.reserve(raw.functions.size());
        for(auto& func_name : raw.functions)
        {
            symbol_name::clean(func_name);
            chain.push_back(correct_symbol_name(path, base, vaddr, func_name));
        }
        if(chain.empty())
        {
            chain.push_back(format_with_binary(core::format_hex(ip), base));
        }
        if(!raw.file_line.empty())
        {
            m_file_line.emplace(ip, std::move(raw.file_line));
        }
        m_cache.emplace(ip, std::move(chain));
    }

    symbolizer_config                                           m_cfg{};
    cpu_snap::processing::symbols::proc_maps                                m_maps;
    const cpu_snap::core::module_map*                           m_module_map = nullptr;
    std::unordered_map<std::uint64_t, std::vector<std::string>> m_cache;
    std::unordered_map<std::uint64_t, std::string>              m_file_line;
};

}  // namespace cpu_snap::processing::symbols
