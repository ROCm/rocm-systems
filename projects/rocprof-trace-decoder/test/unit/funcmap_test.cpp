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

#include "rocprof_trace_decoder/cxx/funcmap.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace rocprof_trace_decoder::codeobj;

namespace
{
// RAII redirect of std::cerr → an internal ostringstream.
class CerrCapture
{
public:
    CerrCapture() : prev(std::cerr.rdbuf(buf.rdbuf())) {}
    ~CerrCapture() { std::cerr.rdbuf(prev); }
    std::string str() const { return buf.str(); }

private:
    std::ostringstream buf;
    std::streambuf* prev;
};

bool has_warning(const std::vector<FuncmapDiagnostic>& diags, const std::string& needle)
{
    for (const auto& d : diags)
    {
        if (d.severity == FuncmapDiagnostic::Severity::Warning && d.message.find(needle) != std::string::npos)
            return true;
    }
    return false;
}

bool has_error(const std::vector<FuncmapDiagnostic>& diags)
{
    for (const auto& d : diags)
        if (d.severity == FuncmapDiagnostic::Severity::Error) return true;
    return false;
}
} // namespace

// ─── decode_marker_value ────────────────────────────────────────────────────

TEST(DecodeMarkerValue, EnterClearsExitPrev)
{
    // bit 0 = exit_prev, bit 1 = is_enter, bits[31:2] = ID
    auto m = decode_marker_value((42u << 2) | 0b10u);
    EXPECT_EQ(m.id, 42u);
    EXPECT_TRUE(m.is_enter);
    EXPECT_FALSE(m.exit_prev);
}

TEST(DecodeMarkerValue, ExitOnlySetsExitPrev)
{
    // s_ttracedata_imm 1 -> raw value 1
    auto m = decode_marker_value(1u);
    EXPECT_EQ(m.id, 0u);
    EXPECT_FALSE(m.is_enter);
    EXPECT_TRUE(m.exit_prev);
}

TEST(DecodeMarkerValue, PointMarkerNoFlags)
{
    auto m = decode_marker_value(7u << 2);
    EXPECT_EQ(m.id, 7u);
    EXPECT_FALSE(m.is_enter);
    EXPECT_FALSE(m.exit_prev);
}

TEST(DecodeMarkerValue, EnterAfterExitTransition)
{
    auto m = decode_marker_value((9u << 2) | 0b11u);
    EXPECT_EQ(m.id, 9u);
    EXPECT_TRUE(m.is_enter);
    EXPECT_TRUE(m.exit_prev);
}

TEST(DecodeMarkerValue, ShaderClockEncoding)
{
    MarkerEncoding enc{12, 4};
    uint32_t raw = (0xABCu << 20) | (42u << 2) | 0b10u;
    auto m = decode_marker_value(raw, enc);
    EXPECT_EQ(m.id, 42u);
    EXPECT_EQ(enc.decode_shader_clock(raw), 0xABCu);
    EXPECT_TRUE(m.is_enter);
    EXPECT_FALSE(m.exit_prev);

    Funcmap fm;
    fm.marker_encoding = enc;
    EXPECT_EQ(decode_marker_value(raw, fm).id, raw >> 2);
}

TEST(MarkerEncoding, ExposesRawAndSourceClockMasks)
{
    MarkerEncoding enc{12, 4};

    // The raw shaderdata word reserves bits [31:20] for the packed clock,
    // bits [19:2] for the ID, and preserves the two low flags.
    EXPECT_TRUE(enc.is_valid());
    EXPECT_TRUE(enc.has_shader_clock());
    EXPECT_EQ(enc.marker_id_mask(), 0x000FFFFCu);
    EXPECT_EQ(enc.packed_shader_clock_mask(), 0xFFF00000u);

    // The sampled source-clock low word uses bits [15:4].
    EXPECT_EQ(enc.shader_clock_source_mask(), 0x0000FFF0u);

    uint32_t raw = (0xABCu << 20) | (42u << 2) | 0b10u;
    EXPECT_EQ((raw & enc.marker_id_mask()) >> 2, 42u);
    EXPECT_EQ(raw & enc.packed_shader_clock_mask(), 0xABC00000u);
}

TEST(MarkerEncoding, LegacyEncodingRemainsFullWidthIdOnly)
{
    MarkerEncoding enc{};

    EXPECT_TRUE(enc.is_valid());
    EXPECT_FALSE(enc.has_shader_clock());
    EXPECT_EQ(enc.marker_id_mask(), 0xFFFFFFFCu);
    EXPECT_EQ(enc.packed_shader_clock_mask(), 0u);
    EXPECT_EQ(enc.shader_clock_source_mask(), 0u);

    uint32_t raw = (0x3FFFFFFFu << 2) | 0b01u;
    EXPECT_EQ(decode_marker_value(raw, enc).id, 0x3FFFFFFFu);
    EXPECT_EQ(enc.decode_shader_clock(raw), 0u);

    MarkerEncoding invalid{30, 0};
    EXPECT_FALSE(invalid.is_valid());
    EXPECT_EQ(decode_marker_value(raw, invalid).id, 0x3FFFFFFFu);
}

TEST(MarkerEncoding, SupportsBoundaryClockWindows)
{
    MarkerEncoding high_bit{1, 31};
    EXPECT_TRUE(high_bit.is_valid());
    EXPECT_EQ(high_bit.marker_id_mask(), 0x7FFFFFFCu);
    EXPECT_EQ(high_bit.packed_shader_clock_mask(), 0x80000000u);
    EXPECT_EQ(high_bit.shader_clock_source_mask(), 0x80000000u);

    MarkerEncoding wide{29, 3};
    EXPECT_TRUE(wide.is_valid());
    EXPECT_EQ(wide.marker_id_mask(), 0x00000004u);
    EXPECT_EQ(wide.packed_shader_clock_mask(), 0xFFFFFFF8u);
    EXPECT_EQ(wide.shader_clock_source_mask(), 0xFFFFFFF8u);
}

TEST(DecodeMarkerValue, FuncmapLeavesUnregisteredValuesLegacy)
{
    Funcmap fm;
    fm.marker_encoding = {12, 4};
    auto entry = std::make_shared<FuncmapEntry>();
    entry->kind = FuncmapEntryKind::Point;
    entry->id = 42;
    fm.entries.push_back(entry);
    fm.by_id.emplace(entry->id, entry);

    uint32_t known = (0xABCu << 20) | (42u << 2);
    EXPECT_EQ(decode_marker_value(known, fm).id, 42u);
    EXPECT_EQ(fm.marker_encoding.decode_shader_clock(known), 0xABCu);

    uint32_t numeric = (0xABCu << 20) | (123u << 2);
    EXPECT_EQ(decode_marker_value(numeric, fm).id, numeric >> 2);

    uint32_t exit = (0xABCu << 20) | 1u;
    EXPECT_EQ(decode_marker_value(exit, fm).id, 0u);
    EXPECT_EQ(fm.marker_encoding.decode_shader_clock(exit), 0xABCu);
}

TEST(FuncmapCompatibility, ExistingAggregateInitializersStillWork)
{
    // Keep source compatibility for consumers that aggregate-initialize the
    // public structs from the pre-packed-marker layout.
    std::vector<FuncmapDiagnostic> diagnostics;
    Funcmap map{
        std::vector<Funcmap::EntryPtr>{},
        std::unordered_map<uint32_t, Funcmap::EntryPtr>{},
        0,
        diagnostics,
    };
    FuncmapEntry entry{FuncmapEntryKind::Point, 7, "point", "source.cpp:1", 0x100000000ull};
    MarkerValue marker{7, true, false};

    EXPECT_TRUE(map.diagnostics.empty());
    EXPECT_FALSE(map.marker_encoding.has_shader_clock());
    EXPECT_EQ(entry.vaddr, 0x100000000ull);
    EXPECT_EQ(entry.extra_payload_count, 0u);
    EXPECT_EQ(marker.id, 7u);
    EXPECT_TRUE(marker.is_enter);
    EXPECT_FALSE(marker.exit_prev);
}

// ─── parse_funcmap_section ──────────────────────────────────────────────────

TEST(ParseFuncmap, EachRowKind)
{
    std::string blob = "F:1:my_device_fn@/p/foo.cpp:42\n"
                       "K:my_kernel\n"
                       "U:2:my_scope\n"
                       "P:3:vmem_load@a.cpp:7\n"
                       "W:64\n";

    CerrCapture cap;
    Funcmap m = parse_funcmap_section(blob);

    EXPECT_EQ(m.entries.size(), 4u);
    EXPECT_EQ(m.wave_size, 64u);
    EXPECT_TRUE(m.diagnostics.empty()) << "unexpected diagnostics emitted";

    auto f = m.find(1);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->kind, FuncmapEntryKind::Function);
    EXPECT_EQ(f->name, "my_device_fn");
    EXPECT_EQ(f->source_loc, "/p/foo.cpp:42");

    // K: row carries no ID, so by_id should not contain it; entries[1] should be it.
    EXPECT_EQ(m.entries[1]->kind, FuncmapEntryKind::Kernel);
    EXPECT_EQ(m.entries[1]->name, "my_kernel");

    auto u = m.find(2);
    ASSERT_TRUE(u);
    EXPECT_EQ(u->kind, FuncmapEntryKind::UserScope);
    EXPECT_EQ(u->name, "my_scope");
    EXPECT_TRUE(u->source_loc.empty());

    auto p = m.find(3);
    ASSERT_TRUE(p);
    EXPECT_EQ(p->kind, FuncmapEntryKind::Point);
    EXPECT_EQ(p->name, "vmem_load");
    EXPECT_EQ(p->source_loc, "a.cpp:7");
}

TEST(ParseFuncmap, ParsesExtraPayloadAndMarkerEncoding)
{
    std::string blob = "M:shader_clock_bits=12;shader_clock_shift=4\n"
                       "P:7:payload_point\n"
                       "R:7:extra_payload_count=1\n";

    Funcmap m = parse_funcmap_section(blob, /*silent=*/true);
    auto p = m.find(7);
    ASSERT_TRUE(p);
    EXPECT_EQ(p->extra_payload_count, 1u);
    EXPECT_EQ(m.marker_encoding.shader_clock_bits, 12u);
    EXPECT_EQ(m.marker_encoding.shader_clock_shift, 4u);
    EXPECT_EQ(m.marker_encoding.marker_id_mask(), 0x000FFFFCu);
    EXPECT_EQ(m.marker_encoding.packed_shader_clock_mask(), 0xFFF00000u);
    EXPECT_EQ(m.marker_encoding.shader_clock_source_mask(), 0x0000FFF0u);
}

TEST(ParseFuncmap, PackedMarkerEncodingRequiresClockShift)
{
    Funcmap m = parse_funcmap_section("M:shader_clock_bits=12\n", /*silent=*/true);

    EXPECT_FALSE(m.marker_encoding.has_shader_clock());
    EXPECT_TRUE(has_warning(m.diagnostics, "malformed M:"));
}

TEST(ParseFuncmap, ToleratesBlankLinesCRLFAndTrailingNUL)
{
    std::string blob = "F:5:foo\r\n\r\nK:k1\r\n";
    blob.push_back('\0'); // common: ConstantDataArray::getString(AddNull=true)

    CerrCapture cap;
    Funcmap m = parse_funcmap_section(blob);
    ASSERT_EQ(m.entries.size(), 2u);
    EXPECT_TRUE(m.diagnostics.empty()) << cap.str();
    auto f = m.find(5);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->name, "foo");
}

TEST(ParseFuncmap, SourceLocPreservesColonsAndAtSigns)
{
    // First `@` splits name from source_loc; subsequent `@` and `:` survive in source_loc.
    std::string blob = "F:1:my_fn@/p/file.cpp:10:5\n";
    Funcmap m = parse_funcmap_section(blob, /*silent=*/true);
    ASSERT_EQ(m.entries.size(), 1u);
    auto f = m.find(1);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->name, "my_fn");
    EXPECT_EQ(f->source_loc, "/p/file.cpp:10:5");
}

TEST(ParseFuncmap, DuplicateIdsLastWriterWinsAndWarns)
{
    std::string blob = "F:7:first\n"
                       "U:7:second\n";
    CerrCapture cap;
    Funcmap m = parse_funcmap_section(blob);

    EXPECT_EQ(m.entries.size(), 2u); // both rows retained in entries
    auto e = m.find(7);
    ASSERT_TRUE(e);
    EXPECT_EQ(e->name, "second"); // last-writer-wins in by_id
    EXPECT_TRUE(has_warning(m.diagnostics, "duplicate marker ID 7"));
    EXPECT_NE(cap.str().find("duplicate marker ID 7"), std::string::npos);
}

TEST(ParseFuncmap, MalformedRowWarnsAndContinues)
{
    std::string blob = "F:1:good\n"
                       "garbage line\n"
                       "F:notanumber:bad\n"
                       "X:9:unknown\n"
                       "F:2:also_good\n";

    CerrCapture cap;
    Funcmap m = parse_funcmap_section(blob);

    EXPECT_TRUE(m.find(1));
    EXPECT_TRUE(m.find(2));
    EXPECT_FALSE(m.find(9));
    EXPECT_GE(m.diagnostics.size(), 3u);
    EXPECT_TRUE(has_warning(m.diagnostics, "malformed row"));
    EXPECT_TRUE(has_warning(m.diagnostics, "malformed F:"));
    EXPECT_TRUE(has_warning(m.diagnostics, "unknown row prefix"));

    // Each diagnostic must carry the offending line content + 1-based line no.
    bool found_line_2 = false;
    for (const auto& d : m.diagnostics)
    {
        if (d.line_no == 2 && d.message.find("garbage line") != std::string::npos) found_line_2 = true;
    }
    EXPECT_TRUE(found_line_2);

    EXPECT_FALSE(cap.str().empty()); // echoed to cerr by default
}

TEST(ParseFuncmap, SilentSuppressesCerrButPopulatesDiagnostics)
{
    std::string blob = "garbage\n"
                       "F:7:first\n"
                       "F:7:second\n";

    CerrCapture cap;
    Funcmap m = parse_funcmap_section(blob, /*silent=*/true);

    EXPECT_TRUE(cap.str().empty()) << "expected silent mode to leave cerr untouched";
    EXPECT_GE(m.diagnostics.size(), 2u); // garbage + duplicate
}

TEST(ParseFuncmap, FindReturnsSameInstanceAcrossLookups)
{
    std::string blob = "F:1:foo\n";
    Funcmap m = parse_funcmap_section(blob, /*silent=*/true);
    auto a = m.find(1);
    auto b = m.find(1);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_EQ(a.get(), b.get()) << "Funcmap::find should not allocate per call";
}

TEST(ParseFuncmap, EmptyBlobYieldsEmptyFuncmap)
{
    CerrCapture cap;
    Funcmap m = parse_funcmap_section("");
    EXPECT_TRUE(m.entries.empty());
    EXPECT_TRUE(m.diagnostics.empty());
    EXPECT_TRUE(cap.str().empty());
    EXPECT_EQ(m.wave_size, 0u);
}

TEST(ParseFuncmap, MalformedWaveSizeWarns)
{
    std::string blob = "W:notanumber\n";
    Funcmap m = parse_funcmap_section(blob, /*silent=*/true);
    EXPECT_EQ(m.wave_size, 0u);
    EXPECT_TRUE(has_warning(m.diagnostics, "malformed W:"));
}

// ─── extract_elf_section: build a synthetic ELF64 in memory ─────────────────

namespace
{
// Lay out: ehdr | sections (data) | shstrtab data | shdr table.
// Keeps offsets simple to compute deterministically.
struct ElfBuildOptions
{
    uint16_t shentsize = static_cast<uint16_t>(sizeof(detail::Elf64Shdr));
    bool zero_shstrndx = false;
    bool shoff_oob = false;
};

struct ElfBuilder
{
    std::vector<uint8_t> elf;

    struct SectionDesc
    {
        std::string name;
        std::vector<uint8_t> data;
    };

    // Builds an ELF with `sections` plus an implicit `.shstrtab`. Returns
    // the byte buffer. The first synthetic section is reserved as the SHT_NULL
    // entry (index 0), as per ELF convention.
    static std::vector<uint8_t> build(const std::vector<SectionDesc>& sections, ElfBuildOptions options = {})
    {
        // Layout:
        // [Elf64_Ehdr][section data, packed]  [shstrtab data]  [shdr table (NULL + sections + shstrtab)]
        std::vector<uint8_t> shstrtab;
        shstrtab.push_back(0); // index 0 must be NUL

        std::vector<uint32_t> name_offsets(sections.size(), 0);
        for (size_t i = 0; i < sections.size(); ++i)
        {
            name_offsets[i] = uint32_t(shstrtab.size());
            for (char c : sections[i].name) shstrtab.push_back(uint8_t(c));
            shstrtab.push_back(0);
        }
        uint32_t shstrtab_name_off = uint32_t(shstrtab.size());
        const std::string shstrtab_name = ".shstrtab";
        for (char c : shstrtab_name) shstrtab.push_back(uint8_t(c));
        shstrtab.push_back(0);

        // Compute offsets.
        size_t cursor = sizeof(detail::Elf64Ehdr);
        std::vector<uint64_t> data_offsets(sections.size(), 0);
        for (size_t i = 0; i < sections.size(); ++i)
        {
            data_offsets[i] = cursor;
            cursor += sections[i].data.size();
        }
        uint64_t shstrtab_off = cursor;
        cursor += shstrtab.size();

        uint64_t shoff = cursor;
        // 1 SHT_NULL entry at index 0, then `sections.size()` user sections,
        // then 1 .shstrtab section. So total = sections.size() + 2.
        uint16_t shnum = uint16_t(sections.size() + 2);
        uint16_t shstrndx = uint16_t(sections.size() + 1);
        if (options.zero_shstrndx) shstrndx = detail::ShnUndef;

        size_t shdr_size = sizeof(detail::Elf64Shdr) * shnum;

        std::vector<uint8_t> out(shoff + shdr_size, 0);

        detail::Elf64Ehdr ehdr{};
        std::memcpy(ehdr.e_ident, detail::ElfMagic, detail::ElfMagicSize);
        ehdr.e_ident[detail::EiClass] = detail::ElfClass64;
        ehdr.e_shoff = options.shoff_oob ? out.size() + sizeof(detail::Elf64Ehdr) : shoff;
        ehdr.e_ehsize = sizeof(detail::Elf64Ehdr);
        ehdr.e_shentsize = options.shentsize;
        ehdr.e_shnum = shnum;
        ehdr.e_shstrndx = shstrndx;
        std::memcpy(out.data(), &ehdr, sizeof(ehdr));

        // Pack section data.
        for (size_t i = 0; i < sections.size(); ++i)
        {
            if (!sections[i].data.empty())
                std::memcpy(out.data() + data_offsets[i], sections[i].data.data(), sections[i].data.size());
        }
        std::memcpy(out.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

        // SHT_NULL entry at index 0 (already zero).
        // User sections at indices [1..sections.size()].
        for (size_t i = 0; i < sections.size(); ++i)
        {
            detail::Elf64Shdr s{};
            s.sh_name = name_offsets[i];
            s.sh_flags = 0;
            s.sh_offset = data_offsets[i];
            s.sh_size = sections[i].data.size();
            std::memcpy(out.data() + shoff + (i + 1) * sizeof(detail::Elf64Shdr), &s, sizeof(s));
        }
        // .shstrtab at index sections.size() + 1.
        detail::Elf64Shdr s_str{};
        s_str.sh_name = shstrtab_name_off;
        s_str.sh_offset = shstrtab_off;
        s_str.sh_size = shstrtab.size();
        std::memcpy(out.data() + shoff + (sections.size() + 1) * sizeof(detail::Elf64Shdr), &s_str, sizeof(s_str));

        return out;
    }
};

std::vector<uint8_t> bytes_of(const std::string& s) { return std::vector<uint8_t>(s.begin(), s.end()); }
} // namespace

TEST(ExtractElfSection, FindsSection)
{
    std::string body = "F:1:foo\n";
    auto elf = ElfBuilder::build({
        {".sqtt_funcmap", bytes_of(body)},
    });

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    ASSERT_TRUE(sec);
    EXPECT_EQ(*sec, body);
    EXPECT_TRUE(diags.empty());
}

TEST(ExtractElfSection, AbsentSectionReturnsNulloptNoDiag)
{
    auto elf = ElfBuilder::build({
        {".text", {0x90, 0x90}}
    });

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(diags.empty()); // common case — non-instrumented binary
}

TEST(ExtractElfSection, EmptySectionReturnsNulloptWithWarning)
{
    auto elf = ElfBuilder::build({
        {".sqtt_funcmap", {}}
    });

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    ASSERT_EQ(diags.size(), 1u);
    EXPECT_EQ(diags[0].severity, FuncmapDiagnostic::Severity::Warning);
}

TEST(ExtractElfSection, MultipleProgbitsSelectByName)
{
    auto elf = ElfBuilder::build({
        {".text",         {0x90, 0x90, 0x90}   },
        {".rodata",       bytes_of("hello")    },
        {".sqtt_funcmap", bytes_of("F:1:hit\n")},
    });

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    ASSERT_TRUE(sec);
    EXPECT_EQ(*sec, "F:1:hit\n");
    EXPECT_TRUE(diags.empty());
}

TEST(ExtractElfSection, RejectsNonElf64Class)
{
    auto elf = ElfBuilder::build({
        {".sqtt_funcmap", bytes_of("x")}
    });
    elf[detail::EiClass] = static_cast<uint8_t>(~elf[detail::EiClass]);

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

TEST(ExtractElfSection, RejectsBadShentsize)
{
    ElfBuildOptions options;
    options.shentsize = static_cast<uint16_t>(sizeof(detail::Elf64Shdr) - 1);
    auto elf = ElfBuilder::build(
        {
            {".sqtt_funcmap", bytes_of("x")}
    },
        options
    );

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

TEST(ExtractElfSection, RejectsShoffOutOfRange)
{
    ElfBuildOptions options;
    options.shoff_oob = true;
    auto elf = ElfBuilder::build(
        {
            {".sqtt_funcmap", bytes_of("x")}
    },
        options
    );

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

TEST(ExtractElfSection, RejectsNoShstrndx)
{
    ElfBuildOptions options;
    options.zero_shstrndx = true;
    auto elf = ElfBuilder::build(
        {
            {".sqtt_funcmap", bytes_of("x")}
    },
        options
    );

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

TEST(ExtractElfSection, RejectsTinyBuffer)
{
    char tiny[4] = {0};
    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(tiny, sizeof(tiny), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

TEST(ExtractElfSection, RejectsNullBuffer)
{
    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(nullptr, 0, ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}

// Adversarial sh_size near 2^64 must not wrap and pass the bound check.
TEST(ExtractElfSection, RejectsAdversarialShSizeWrap)
{
    auto elf = ElfBuilder::build({
        {".sqtt_funcmap", bytes_of("F:1:x\n")}
    });

    // Find the .sqtt_funcmap shdr (index 1: NULL=0, our section=1) and clobber sh_size.
    detail::Elf64Ehdr ehdr;
    std::memcpy(&ehdr, elf.data(), sizeof(ehdr));
    detail::Elf64Shdr s;
    size_t shdr_off = ehdr.e_shoff + 1 * sizeof(detail::Elf64Shdr);
    std::memcpy(&s, elf.data() + shdr_off, sizeof(s));
    // sh_offset is small and valid; sh_size huge so sh_offset + sh_size wraps.
    s.sh_size = ~uint64_t(0) - s.sh_offset + 1; // wrap target = 0
    std::memcpy(elf.data() + shdr_off, &s, sizeof(s));

    std::vector<FuncmapDiagnostic> diags;
    auto sec = extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags);
    EXPECT_FALSE(sec);
    EXPECT_TRUE(has_error(diags));
}
