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

TEST(DecodeMarkerValue, DecodesFlagCombinations)
{
    struct Case
    {
        const char* name;
        uint32_t raw;
        uint32_t id;
        bool isEnter;
        bool exitsPrevious;
    };
    for (const Case& test : {
             Case{"enter",      (42u << 2) | 0b10u, 42, true,  false},
             Case{"exit",       1u,                 0,  false, true },
             Case{"point",      7u << 2,            7,  false, false},
             Case{"exit_enter", (9u << 2) | 0b11u,  9,  true,  true }
    })
    {
        SCOPED_TRACE(test.name);
        MarkerValue marker = decode_marker_value(test.raw);
        EXPECT_EQ(marker.id, test.id);
        EXPECT_EQ(marker.is_enter, test.isEnter);
        EXPECT_EQ(marker.exit_prev, test.exitsPrevious);
    }
}

TEST(MarkerEncoding, ComputesMasksAndDecodesClock)
{
    struct Case
    {
        const char* name;
        MarkerEncoding encoding;
        bool hasClock;
        uint32_t idMask;
        uint32_t packedClockMask;
        uint32_t sourceClockMask;
    };
    for (const Case& test : {
             Case{"packed", {12, 4}, true,  0x000FFFFCu, 0xFFF00000u, 0x0000FFF0u},
             Case{"legacy", {},      false, 0xFFFFFFFCu, 0u,          0u         },
             Case{"high_bit", {1, 31}, true, 0x7FFFFFFCu, 0x80000000u, 0x80000000u},
             Case{"wide", {29, 3},    true, 0x00000004u, 0xFFFFFFF8u, 0xFFFFFFF8u},
         })
    {
        SCOPED_TRACE(test.name);
        EXPECT_TRUE(test.encoding.is_valid());
        EXPECT_EQ(test.encoding.has_shader_clock(), test.hasClock);
        EXPECT_EQ(test.encoding.marker_id_mask(), test.idMask);
        EXPECT_EQ(test.encoding.packed_shader_clock_mask(), test.packedClockMask);
        EXPECT_EQ(test.encoding.shader_clock_source_mask(), test.sourceClockMask);
    }

    MarkerEncoding packed{12, 4};
    uint32_t raw = (0xABCu << 20) | (42u << 2) | 0b10u;
    MarkerValue marker = decode_marker_value(raw, packed);
    EXPECT_EQ(marker.id, 42u);
    EXPECT_EQ(packed.decode_shader_clock(raw), 0xABCu);
    EXPECT_TRUE(marker.is_enter);
    EXPECT_FALSE(marker.exit_prev);
    EXPECT_EQ((raw & packed.marker_id_mask()) >> 2, 42u);
    EXPECT_EQ(raw & packed.packed_shader_clock_mask(), 0xABC00000u);

    Funcmap fm;
    fm.marker_encoding = packed;
    EXPECT_EQ(decode_marker_value(raw, fm).id, raw >> 2);

    uint32_t legacyRaw = (0x3FFFFFFFu << 2) | 0b01u;
    EXPECT_EQ(decode_marker_value(legacyRaw, MarkerEncoding{}).id, 0x3FFFFFFFu);
    EXPECT_EQ(MarkerEncoding{}.decode_shader_clock(legacyRaw), 0u);
    MarkerEncoding invalid{30, 0};
    EXPECT_FALSE(invalid.is_valid());
    EXPECT_EQ(decode_marker_value(legacyRaw, invalid).id, 0x3FFFFFFFu);
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
struct ElfSection
{
    std::string name;
    std::string data;
};

std::vector<uint8_t> buildElf(const std::vector<ElfSection>& sections)
{
    std::string names(1, '\0');
    std::vector<detail::Elf64Shdr> headers(sections.size() + 2);
    uint64_t offset = sizeof(detail::Elf64Ehdr);
    size_t index = 1; // Header 0 is SHT_NULL.
    for (const ElfSection& section : sections)
    {
        detail::Elf64Shdr& header = headers[index++];
        header.sh_name = static_cast<uint32_t>(names.size());
        names += section.name;
        names += '\0';
        header.sh_offset = offset;
        header.sh_size = section.data.size();
        offset += section.data.size();
    }

    detail::Elf64Shdr& stringTable = headers.back();
    stringTable.sh_name = static_cast<uint32_t>(names.size());
    names += ".shstrtab";
    names += '\0';
    stringTable.sh_offset = offset;
    stringTable.sh_size = names.size();
    const uint64_t sectionHeadersOffset = offset + names.size();

    std::vector<uint8_t> elf(sectionHeadersOffset + headers.size() * sizeof(detail::Elf64Shdr), 0);
    detail::Elf64Ehdr header{};
    std::memcpy(header.e_ident, detail::ElfMagic, detail::ElfMagicSize);
    header.e_ident[detail::EiClass] = detail::ElfClass64;
    header.e_shoff = sectionHeadersOffset;
    header.e_ehsize = sizeof(header);
    header.e_shentsize = sizeof(detail::Elf64Shdr);
    header.e_shnum = static_cast<uint16_t>(headers.size());
    header.e_shstrndx = static_cast<uint16_t>(headers.size() - 1);
    std::memcpy(elf.data(), &header, sizeof(header));

    index = 1;
    for (const ElfSection& section : sections)
    {
        const detail::Elf64Shdr& sectionHeader = headers[index++];
        if (!section.data.empty())
            std::memcpy(elf.data() + sectionHeader.sh_offset, section.data.data(), section.data.size());
    }
    std::memcpy(elf.data() + stringTable.sh_offset, names.data(), names.size());
    std::memcpy(elf.data() + sectionHeadersOffset, headers.data(), headers.size() * sizeof(detail::Elf64Shdr));
    return elf;
}

void expectFuncmapExtractionError(const char* data, size_t size)
{
    std::vector<FuncmapDiagnostic> diagnostics;
    EXPECT_FALSE(extract_elf_section(data, size, ".sqtt_funcmap", diagnostics));
    EXPECT_TRUE(has_error(diagnostics));
}

void expectFuncmapExtractionError(const std::vector<uint8_t>& elf)
{
    expectFuncmapExtractionError(reinterpret_cast<const char*>(elf.data()), elf.size());
}

enum class ElfCorruption
{
    InvalidClass,
    InvalidSectionHeaderSize,
    SectionTableOutOfRange,
    MissingSectionNameTable,
    OverflowingSectionSize,
};

std::vector<uint8_t> malformedFuncmapElf(ElfCorruption corruption)
{
    auto elf = buildElf({
        {".sqtt_funcmap", "F:1:x\n"}
    });
    detail::Elf64Ehdr header{};
    std::memcpy(&header, elf.data(), sizeof(header));

    switch (corruption)
    {
        case ElfCorruption::InvalidClass:
            header.e_ident[detail::EiClass] = static_cast<uint8_t>(~header.e_ident[detail::EiClass]);
            break;
        case ElfCorruption::InvalidSectionHeaderSize:
            header.e_shentsize = static_cast<uint16_t>(sizeof(detail::Elf64Shdr) - 1);
            break;
        case ElfCorruption::SectionTableOutOfRange: header.e_shoff = elf.size() + sizeof(header); break;
        case ElfCorruption::MissingSectionNameTable: header.e_shstrndx = detail::ShnUndef; break;
        case ElfCorruption::OverflowingSectionSize:
        {
            detail::Elf64Shdr section;
            size_t offset = header.e_shoff + sizeof(section); // NULL=0, funcmap=1
            std::memcpy(&section, elf.data() + offset, sizeof(section));
            section.sh_size = ~uint64_t(0) - section.sh_offset + 1;
            std::memcpy(elf.data() + offset, &section, sizeof(section));
            break;
        }
    }
    std::memcpy(elf.data(), &header, sizeof(header));
    return elf;
}
} // namespace

TEST(ExtractElfSection, HandlesBasicSectionLayouts)
{
    struct Case
    {
        const char* name;
        std::vector<ElfSection> sections;
        const char* expected;
        bool warning;
    };
    for (const Case& test : {
             Case{"find", {{".sqtt_funcmap", "F:1:foo\n"}}, "F:1:foo\n", false},
             Case{"absent", {{".text", "xx"}}, nullptr, false},
             Case{"empty", {{".sqtt_funcmap", {}}}, nullptr, true},
             Case{"multiple", {{".text", "xxx"}, {".rodata", "hello"}, {".sqtt_funcmap", "F:1:hit\n"}},
                  "F:1:hit\n", false},
         })
    {
        SCOPED_TRACE(test.name);
        auto elf = buildElf(test.sections);
        std::vector<FuncmapDiagnostic> diags;
        auto section = extract_elf_section(
            reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap", diags
        );
        if (test.expected)
        {
            ASSERT_TRUE(section);
            EXPECT_EQ(*section, test.expected);
        }
        else
            EXPECT_FALSE(section);
        if (test.warning)
        {
            ASSERT_EQ(diags.size(), 1u);
            EXPECT_EQ(diags[0].severity, FuncmapDiagnostic::Severity::Warning);
        }
        else
            EXPECT_TRUE(diags.empty());
    }
}

TEST(ExtractElfSection, RejectsMalformedHeadersAndBounds)
{
    struct Case
    {
        const char* name;
        ElfCorruption corruption;
    };
    for (const Case& test : {
             Case{"non_elf64_class",     ElfCorruption::InvalidClass            },
             Case{"bad_shentsize",       ElfCorruption::InvalidSectionHeaderSize},
             Case{"shoff_out_of_range",  ElfCorruption::SectionTableOutOfRange  },
             Case{"missing_shstrndx",    ElfCorruption::MissingSectionNameTable },
             Case{"overflowing_sh_size", ElfCorruption::OverflowingSectionSize  }
    })
    {
        SCOPED_TRACE(test.name);
        expectFuncmapExtractionError(malformedFuncmapElf(test.corruption));
    }
}

TEST(ExtractElfSection, RejectsTinyAndNullBuffers)
{
    char tiny[4] = {0};
    expectFuncmapExtractionError(tiny, sizeof(tiny));
    expectFuncmapExtractionError(nullptr, 0);
}
