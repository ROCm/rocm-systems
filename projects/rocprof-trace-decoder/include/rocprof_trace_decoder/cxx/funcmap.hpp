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

#pragma once

#include <elf.h>

#include <cctype>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

// Decoder-side reader for the `.sqtt_funcmap` ELF section emitted by the
// sqtt_instrumentation LLVM pass. The pass writes one ASCII row per
// instrumented function/marker; at runtime the matching ID surfaces in the
// trace as `rocprofiler_thread_trace_decoder_shaderdata_t::value`
// (see trace_decoder_types.h: bit 0 = exit_prev, bit 1 = is_enter,
//  bits [31:2] = ID — or [7:2] when emitted via s_ttracedata_imm).

namespace rocprof_trace_decoder
{
namespace codeobj
{
// ─── Declarations ───────────────────────────────────────────────────────────

enum class FuncmapEntryKind
{
    Function,  // F:ID:name[@source_loc] — instrumented device function (entry/exit scope)
    Kernel,    // K:name[@source_loc]    — kernel name (no ID — for vaddr lookup only)
    UserScope, // U:ID:name              — named user scope marker
    Point      // P:ID:name[@source_loc] — point marker (barrier, memory op, addr trace, …)
};

struct FuncmapEntry
{
    FuncmapEntryKind kind{};
    uint32_t id{0}; // 0 for Kernel rows (no ID)
    std::string name{};
    std::string source_loc{}; // empty if absent
    uint64_t vaddr{0};        // resolved by CodeobjDecoderComponent; 0 if unresolved
};

struct FuncmapDiagnostic
{
    enum class Severity
    {
        Warning,
        Error
    };
    Severity severity{};
    std::string message{};
    size_t line_no{0}; // 1-based line into the funcmap blob; 0 if N/A
};

struct Funcmap
{
    using EntryPtr = std::shared_ptr<const FuncmapEntry>;

    std::vector<EntryPtr> entries{};                // owns rows, stable insertion order
    std::unordered_map<uint32_t, EntryPtr> by_id{}; // ID -> entry; last-writer-wins on dup
    uint32_t wave_size{0};                          // 0 if no `W:` row
    std::vector<FuncmapDiagnostic> diagnostics{};   // collected during parse / extraction

    // Returns the entry for `marker_id` (refcount bump, no string copy), or
    // nullptr if absent.
    EntryPtr find(uint32_t marker_id) const;
};

struct MarkerValue
{
    uint32_t id;
    bool is_enter;
    bool exit_prev;
};

// Parse the `.sqtt_funcmap` ASCII blob. Diagnostics go into
// `Funcmap::diagnostics` AND are echoed to std::cerr (matching the existing
// THROW_COMGR diagnostic pattern in disassembly.hpp). Pass `silent=true` to
// suppress the std::cerr echo while still populating diagnostics.
// Parsing is best-effort: malformed rows produce a Warning and parsing
// continues with the next row.
inline Funcmap parse_funcmap_section(std::string_view blob, bool silent = false);

// Extract a section's bytes from an in-memory ELF64 image. Returns nullopt
// when the section is absent (common case — non-instrumented binaries) OR
// when the ELF header is rejected as malformed (see Error diagnostics).
// `diagnostics` is filled with Warning/Error rows.
inline std::optional<std::string_view> extract_elf_section(
    const char* elf_data, size_t elf_size, std::string_view section_name, std::vector<FuncmapDiagnostic>& diagnostics
);

// Decode a marker value emitted by an `s_ttracedata`/`s_ttracedata_imm`
// instruction (see trace_decoder_types.h:210).
constexpr MarkerValue decode_marker_value(uint32_t v) noexcept;

// ─── Inline definitions ─────────────────────────────────────────────────────

inline Funcmap::EntryPtr Funcmap::find(uint32_t marker_id) const
{
    auto it = by_id.find(marker_id);
    return (it == by_id.end()) ? nullptr : it->second;
}

constexpr MarkerValue decode_marker_value(uint32_t v) noexcept
{
    return MarkerValue{v >> 2, bool((v >> 1) & 1u), bool(v & 1u)};
}

namespace detail
{
inline void emit_diag(
    std::vector<FuncmapDiagnostic>& diagnostics,
    FuncmapDiagnostic::Severity sev,
    std::string msg,
    size_t line_no,
    bool silent
)
{
    if (!silent)
    {
        std::cerr << "rocprof-trace-decoder: .sqtt_funcmap "
                  << (sev == FuncmapDiagnostic::Severity::Error ? "error" : "warning");
        if (line_no != 0) std::cerr << " (line " << line_no << ')';
        std::cerr << ": " << msg << '\n';
    }
    diagnostics.push_back(FuncmapDiagnostic{sev, std::move(msg), line_no});
}

inline std::string_view rstrip_ws(std::string_view s) noexcept
{
    while (!s.empty() && (s.back() == '\r' || s.back() == ' ' || s.back() == '\t' || s.back() == '\0'))
        s.remove_suffix(1);
    return s;
}

// Split ID prefix for F/U/P rows. Returns {id, name+optional@loc} or nullopt
// if the ID failed to parse.
inline std::optional<std::pair<uint32_t, std::string_view>> split_id_payload(std::string_view payload)
{
    size_t colon = payload.find(':');
    if (colon == std::string_view::npos) return std::nullopt;

    std::string_view id_str = payload.substr(0, colon);
    std::string_view name_loc = payload.substr(colon + 1);
    if (id_str.empty()) return std::nullopt;

    uint64_t id = 0;
    for (char c : id_str)
    {
        if (c < '0' || c > '9') return std::nullopt;
        id = id * 10 + uint64_t(c - '0');
        if (id > 0xFFFFFFFFull) return std::nullopt;
    }
    return std::make_pair(uint32_t(id), name_loc);
}

inline std::pair<std::string, std::string> split_name_loc(std::string_view name_loc)
{
    size_t at = name_loc.find('@');
    if (at == std::string_view::npos) return {std::string(name_loc), std::string{}};
    return {std::string(name_loc.substr(0, at)), std::string(name_loc.substr(at + 1))};
}
} // namespace detail

inline Funcmap parse_funcmap_section(std::string_view blob, bool silent)
{
    Funcmap out;
    size_t line_no = 0;
    size_t pos = 0;

    while (pos <= blob.size())
    {
        // Find next newline OR end of blob OR embedded NUL.
        size_t end = pos;
        while (end < blob.size() && blob[end] != '\n' && blob[end] != '\0') ++end;
        std::string_view line = detail::rstrip_ws(blob.substr(pos, end - pos));
        ++line_no;

        // Advance past the terminator (or stop if we ran off the end).
        if (end >= blob.size())
        {
            pos = end + 1;
            if (line.empty()) break;
        }
        else { pos = end + 1; }

        if (line.empty()) continue;

        // Need at minimum a one-char prefix and a ':'.
        if (line.size() < 2 || line[1] != ':')
        {
            detail::emit_diag(
                out.diagnostics,
                FuncmapDiagnostic::Severity::Warning,
                "malformed row (no prefix:): \"" + std::string(line) + '"',
                line_no,
                silent
            );
            continue;
        }

        char prefix = line[0];
        std::string_view payload = line.substr(2);

        auto record = [&](FuncmapEntryKind kind, uint32_t id, std::string name, std::string source_loc)
        {
            auto entry =
                std::make_shared<FuncmapEntry>(FuncmapEntry{kind, id, std::move(name), std::move(source_loc), 0});
            out.entries.push_back(entry);

            if (kind == FuncmapEntryKind::Kernel) return; // K rows have no ID

            auto inserted = out.by_id.emplace(id, entry);
            if (!inserted.second)
            {
                const auto& prev = inserted.first->second;
                std::string msg = "duplicate marker ID " + std::to_string(id) + " — previous \"" + prev->name +
                                  "\" replaced by \"" + entry->name + "\"";
                detail::emit_diag(
                    out.diagnostics, FuncmapDiagnostic::Severity::Warning, std::move(msg), line_no, silent
                );
                inserted.first->second = entry;
            }
        };

        switch (prefix)
        {
            case 'W':
            {
                uint64_t w = 0;
                bool ok = !payload.empty();
                for (char c : payload)
                {
                    if (c < '0' || c > '9')
                    {
                        ok = false;
                        break;
                    }
                    w = w * 10 + uint64_t(c - '0');
                    if (w > 0xFFFFFFFFull)
                    {
                        ok = false;
                        break;
                    }
                }
                if (!ok)
                {
                    detail::emit_diag(
                        out.diagnostics,
                        FuncmapDiagnostic::Severity::Warning,
                        "malformed W: row: \"" + std::string(line) + '"',
                        line_no,
                        silent
                    );
                }
                else { out.wave_size = uint32_t(w); }
                break;
            }
            case 'K':
            {
                auto [name, loc] = detail::split_name_loc(payload);
                if (name.empty())
                {
                    detail::emit_diag(
                        out.diagnostics,
                        FuncmapDiagnostic::Severity::Warning,
                        "K: row missing name: \"" + std::string(line) + '"',
                        line_no,
                        silent
                    );
                    break;
                }
                record(FuncmapEntryKind::Kernel, 0, std::move(name), std::move(loc));
                break;
            }
            case 'F':
            case 'U':
            case 'P':
            {
                auto split = detail::split_id_payload(payload);
                if (!split)
                {
                    detail::emit_diag(
                        out.diagnostics,
                        FuncmapDiagnostic::Severity::Warning,
                        std::string("malformed ") + prefix + ": row (bad ID): \"" + std::string(line) + '"',
                        line_no,
                        silent
                    );
                    break;
                }
                auto [name, loc] = detail::split_name_loc(split->second);
                if (name.empty())
                {
                    detail::emit_diag(
                        out.diagnostics,
                        FuncmapDiagnostic::Severity::Warning,
                        std::string("malformed ") + prefix + ": row (empty name): \"" + std::string(line) + '"',
                        line_no,
                        silent
                    );
                    break;
                }
                FuncmapEntryKind kind = (prefix == 'F') ? FuncmapEntryKind::Function
                                      : (prefix == 'U') ? FuncmapEntryKind::UserScope
                                                        : FuncmapEntryKind::Point;
                record(kind, split->first, std::move(name), std::move(loc));
                break;
            }
            default:
            {
                detail::emit_diag(
                    out.diagnostics,
                    FuncmapDiagnostic::Severity::Warning,
                    std::string("unknown row prefix '") + prefix + "': \"" + std::string(line) + '"',
                    line_no,
                    silent
                );
                break;
            }
        }
    }

    return out;
}

inline std::optional<std::string_view> extract_elf_section(
    const char* elf_data, size_t elf_size, std::string_view section_name, std::vector<FuncmapDiagnostic>& diagnostics
)
{
    auto reject = [&](std::string msg) -> std::optional<std::string_view>
    {
        diagnostics.push_back(FuncmapDiagnostic{FuncmapDiagnostic::Severity::Error, std::move(msg), 0});
        return std::nullopt;
    };

    if (elf_data == nullptr || elf_size < sizeof(Elf64_Ehdr)) return reject("ELF buffer too small for an Elf64_Ehdr");

    if (std::memcmp(elf_data, ELFMAG, SELFMAG) != 0) return reject("not an ELF image (bad ELFMAG)");

    Elf64_Ehdr ehdr;
    std::memcpy(&ehdr, elf_data, sizeof(ehdr));

    if (ehdr.e_ident[EI_CLASS] != ELFCLASS64) return reject("not ELF64 (EI_CLASS)");
    if (ehdr.e_shentsize != sizeof(Elf64_Shdr))
        return reject("unexpected e_shentsize (" + std::to_string(ehdr.e_shentsize) + ')');
    if (ehdr.e_shoff == 0 || ehdr.e_shoff > elf_size) return reject("e_shoff out of range");
    if (ehdr.e_shstrndx == SHN_UNDEF) return reject("no section name string table (SHN_UNDEF)");

    uint64_t shdr_table_bytes = uint64_t(ehdr.e_shnum) * sizeof(Elf64_Shdr);
    if (shdr_table_bytes / sizeof(Elf64_Shdr) != uint64_t(ehdr.e_shnum))
        return reject("section header table size overflow");
    // Use subtraction to avoid wrap; e_shoff <= elf_size from the prior check.
    if (shdr_table_bytes > uint64_t(elf_size) - ehdr.e_shoff)
        return reject("section header table extends past end of buffer");
    if (ehdr.e_shstrndx >= ehdr.e_shnum)
        return reject("e_shstrndx (" + std::to_string(ehdr.e_shstrndx) + ") >= e_shnum");

    auto read_shdr = [&](unsigned idx)
    {
        Elf64_Shdr s;
        std::memcpy(&s, elf_data + ehdr.e_shoff + idx * sizeof(Elf64_Shdr), sizeof(Elf64_Shdr));
        return s;
    };

    Elf64_Shdr shstr = read_shdr(ehdr.e_shstrndx);
    if (shstr.sh_offset > elf_size || shstr.sh_size > uint64_t(elf_size) - shstr.sh_offset)
        return reject("section name string table out of range");

    const char* str_base = elf_data + shstr.sh_offset;
    size_t str_len = shstr.sh_size;

    auto name_of = [&](uint32_t name_off) -> std::string_view
    {
        if (name_off >= str_len) return std::string_view{};
        size_t end = name_off;
        while (end < str_len && str_base[end] != '\0') ++end;
        return std::string_view(str_base + name_off, end - name_off);
    };

    for (unsigned i = 0; i < ehdr.e_shnum; ++i)
    {
        Elf64_Shdr s = read_shdr(i);
        if (name_of(s.sh_name) != section_name) continue;

        if (s.sh_size == 0)
        {
            diagnostics.push_back(FuncmapDiagnostic{
                FuncmapDiagnostic::Severity::Warning, std::string(section_name) + " section present but empty", 0});
            return std::nullopt;
        }
        if (s.sh_offset > elf_size || s.sh_size > uint64_t(elf_size) - s.sh_offset)
        {
            return reject(std::string(section_name) + " section data out of range");
        }
        return std::string_view(elf_data + s.sh_offset, s.sh_size);
    }

    return std::nullopt; // section absent — common case, no diagnostic
}

} // namespace codeobj
} // namespace rocprof_trace_decoder
