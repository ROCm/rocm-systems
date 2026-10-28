#pragma once

/**
 * @file addr2line_client.hpp
 * @brief Runs `addr2line -ifCe` for a batch of ELF addresses and parses its
 *        output into per-address inline chains. Stateless: no caching and no
 *        knowledge of module maps.
 */

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <stdio.h>  // NOLINT(misc-include-cleaner,modernize-deprecated-headers): POSIX popen/pclose live in <stdio.h>, not <cstdio>

#include "cpu_snap/core/format_hex.hpp"

namespace cpu_snap::processing::symbols
{

// NOLINTBEGIN(readability-identifier-naming): codebase uses k_*-prefixed
//   lower_case for global constants; the .clang-tidy GlobalConstantCase
//   rule defaults to UPPER_CASE but the entire project ships with k_*.
namespace detail
{
/** Buffer size for one addr2line output line ("function at file:line"). */
inline constexpr std::size_t k_addr2line_line_buf = 2048;
/** Max IPs per addr2line invocation, keeping the popen command line well
 *  under ARG_MAX even for large batches. */
inline constexpr std::size_t k_max_ips_per_invocation = 1000;
}  // namespace detail
// NOLINTEND(readability-identifier-naming)

/**
 * @brief Raw addr2line answer for one address.
 */
struct addr2line_chain
{
    /** Function names, innermost inline frame first; empty when addr2line
     *  produced nothing usable for the address. Names are as emitted. */
    std::vector<std::string> functions;
    /** "file:line" of the first frame; empty when unavailable. */
    std::string file_line;
};

class addr2line_client
{
public:
    /** (ip, ELF address) pair; only the ELF address is sent to addr2line. */
    using address_pair = std::pair<std::uint64_t, std::uint64_t>;

    /**
     * @brief Resolve every pair's ELF address in @p path.
     * @return One entry per pair, in order. @c std::nullopt marks pairs whose
     *         addr2line invocation could not be started (popen failure).
     */
    [[nodiscard]] static std::vector<std::optional<addr2line_chain>>
    resolve(std::string_view path, std::span<const address_pair> pairs)
    {
        std::vector<std::optional<addr2line_chain>> results;
        results.reserve(pairs.size());
        for(std::size_t offset = 0; offset < pairs.size();
            offset += detail::k_max_ips_per_invocation)
        {
            const std::size_t count =
                std::min(detail::k_max_ips_per_invocation, pairs.size() - offset);
            run_invocation(path, pairs.subspan(offset, count), results);
        }
        return results;
    }

    [[nodiscard]] static std::string shell_single_quote(std::string_view value)
    {
        std::string out;
        out.reserve(value.size() + 2);
        out += '\'';
        for(const char ch : value)
        {
            if(ch == '\'')
            {
                out += "'\\''";
            }
            else
            {
                out += ch;
            }
        }
        out += '\'';
        return out;
    }

    [[nodiscard]] static std::string build_command(std::string_view path,
                                                   std::span<const address_pair> pairs)
    {
        // -i: surface the full inline-call chain per IP (innermost first).
        // -f: function name. -C: demangle. NO -p: each entry is two lines
        //     (function, file:line), so the inline chain reads as N pairs.
        // A sentinel address (0xffffffffffffffff) is inserted between IPs
        // so the parser can find chain boundaries reliably; addr2line
        // emits "??\n??:0" for it.
        std::string cmd = "addr2line -ifCe ";
        cmd += shell_single_quote(path);
        cmd += " 2>/dev/null";
        for(std::size_t i = 0; i < pairs.size(); ++i)
        {
            cmd += ' ';
            cmd += core::format_hex(pairs[i].second);
            if(i + 1 < pairs.size())
            {
                cmd += " 0xffffffffffffffff";
            }
        }
        return cmd;
    }

    /** Sentinel addr2line emits for an unmapped IP. */
    [[nodiscard]] static bool is_sentinel_function(std::string_view fn) noexcept
    {
        return fn == "??";
    }

    /** Returns empty for the ??:0 / ??:? / empty sentinels. */
    [[nodiscard]] static std::string normalize_file_line(std::string raw)
    {
        if(raw == "??:0" || raw == "??:?" || raw.empty())
        {
            raw.clear();
        }
        return raw;
    }

    /**
     * @brief Parse one inline chain (sequence of {function, file:line}
     *        pairs), stopping at the next sentinel pair or end of input.
     *        The sentinel pair is consumed and never recorded.
     * @param next_line Callable returning the next line without its newline,
     *                  or @c std::nullopt at end of input.
     */
    template <typename LineSource>
    [[nodiscard]] static addr2line_chain read_chain(LineSource&& next_line)
    {
        addr2line_chain chain;
        bool            first_pair = true;
        while(true)
        {
            auto func = next_line();
            if(!func) break;
            auto        file      = next_line();
            std::string file_line = normalize_file_line(file.value_or(std::string{}));

            if(is_sentinel_function(*func))
            {
                if(first_pair)
                {
                    // unresolved IP: consume trailing separator pair
                    [[maybe_unused]] const auto sep_func = next_line();
                    [[maybe_unused]] const auto sep_file = next_line();
                }
                break;
            }
            first_pair = false;

            if(chain.functions.empty() && !file_line.empty())
            {
                chain.file_line = file_line;
            }
            chain.functions.push_back(std::move(*func));
        }
        return chain;
    }

private:
    static void run_invocation(std::string_view                               path,
                               std::span<const address_pair>                  pairs,
                               std::vector<std::optional<addr2line_chain>>&   results)
    {
        if(pairs.empty())
        {
            return;
        }

        const std::string cmd = build_command(path, pairs);
        // NOLINTNEXTLINE(cert-env33-c,concurrency-mt-unsafe): addr2line requires popen for the batched per-binary invocation
        FILE* pipe = ::popen(cmd.c_str(), "r");
        if(pipe == nullptr)
        {
            results.insert(results.end(), pairs.size(), std::nullopt);
            return;
        }

        std::array<char, detail::k_addr2line_line_buf> buf{};
        const auto next_line = [&pipe, &buf]() { return read_line(pipe, buf); };
        for(std::size_t i = 0; i < pairs.size(); ++i)
        {
            results.emplace_back(read_chain(next_line));
        }
        [[maybe_unused]] const auto pclose_rc = ::pclose(pipe);
    }

    /** Read one line from the pipe, stripping trailing newline. Returns
     *  std::nullopt on EOF. */
    static std::optional<std::string>
    read_line(FILE* pipe, std::array<char, detail::k_addr2line_line_buf>& buf)
    {
        if(std::fgets(buf.data(), static_cast<int>(buf.size()), pipe) == nullptr)
        {
            return std::nullopt;
        }
        std::string raw = buf.data();
        if(!raw.empty() && raw.back() == '\n')
        {
            raw.pop_back();
        }
        return raw;
    }
};

}  // namespace cpu_snap::processing::symbols
