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

#include <rocprofiler-sdk/cxx/codeobj/funcmap.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

using namespace rocprofiler::sdk::codeobj::funcmap;

namespace
{
// RAII redirect of std::cerr -> an internal ostringstream.
class CerrCapture
{
public:
    CerrCapture()
    : prev(std::cerr.rdbuf(buf.rdbuf()))
    {}
    ~CerrCapture() { std::cerr.rdbuf(prev); }
    std::string str() const { return buf.str(); }

private:
    std::ostringstream buf;
    std::streambuf*    prev;
};
}  // namespace

// --- decode_marker_value ----------------------------------------------------

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

// --- parse_funcmap_section --------------------------------------------------

TEST(ParseFuncmap, EachRowKind)
{
    std::string blob = "F:1:my_device_fn@/p/foo.cpp:42\n"
                       "K:my_kernel\n"
                       "U:2:my_scope\n"
                       "P:3:vmem_load@a.cpp:7\n"
                       "W:64\n";

    CerrCapture cap;
    Funcmap     m = parse_funcmap_section(blob);

    EXPECT_EQ(m.entries.size(), 4u);
    EXPECT_EQ(m.wave_size, 64u);
    EXPECT_TRUE(cap.str().empty()) << "unexpected diagnostics emitted: " << cap.str();

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

TEST(ParseFuncmap, ToleratesBlankLinesCRLFAndTrailingNUL)
{
    std::string blob = "F:5:foo\r\n\r\nK:k1\r\n";
    blob.push_back('\0');  // common: ConstantDataArray::getString(AddNull=true)

    CerrCapture cap;
    Funcmap     m = parse_funcmap_section(blob);
    ASSERT_EQ(m.entries.size(), 2u);
    EXPECT_TRUE(cap.str().empty()) << cap.str();
    auto f = m.find(5);
    ASSERT_TRUE(f);
    EXPECT_EQ(f->name, "foo");
}

TEST(ParseFuncmap, SourceLocPreservesColonsAndAtSigns)
{
    // First `@` splits name from source_loc; subsequent `@` and `:` survive in source_loc.
    std::string blob = "F:1:my_fn@/p/file.cpp:10:5\n";
    Funcmap     m    = parse_funcmap_section(blob, /*silent=*/true);
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
    Funcmap     m = parse_funcmap_section(blob);

    EXPECT_EQ(m.entries.size(), 2u);  // both rows retained in entries
    auto e = m.find(7);
    ASSERT_TRUE(e);
    EXPECT_EQ(e->name, "second");  // last-writer-wins in by_id
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
    Funcmap     m = parse_funcmap_section(blob);

    EXPECT_TRUE(m.find(1));
    EXPECT_TRUE(m.find(2));
    EXPECT_FALSE(m.find(9));

    const std::string out = cap.str();
    EXPECT_NE(out.find("malformed row"), std::string::npos);
    EXPECT_NE(out.find("malformed F:"), std::string::npos);
    EXPECT_NE(out.find("unknown row prefix"), std::string::npos);

    // Each diagnostic must carry the offending line content + 1-based line no.
    EXPECT_NE(out.find("line 2"), std::string::npos);
    EXPECT_NE(out.find("garbage line"), std::string::npos);

    EXPECT_FALSE(out.empty());  // echoed to cerr by default
}

TEST(ParseFuncmap, SilentSuppressesCerr)
{
    std::string blob = "garbage\n"
                       "F:7:first\n"
                       "F:7:second\n";

    CerrCapture cap;
    Funcmap     m = parse_funcmap_section(blob, /*silent=*/true);

    EXPECT_TRUE(cap.str().empty()) << "expected silent mode to leave cerr untouched";
    // Despite suppressed warnings, parsing still proceeds and records both rows.
    EXPECT_EQ(m.entries.size(), 2u);
    auto e = m.find(7);
    ASSERT_TRUE(e);
    EXPECT_EQ(e->name, "second");
}

TEST(ParseFuncmap, FindReturnsSameInstanceAcrossLookups)
{
    std::string blob = "F:1:foo\n";
    Funcmap     m    = parse_funcmap_section(blob, /*silent=*/true);
    auto        a    = m.find(1);
    auto        b    = m.find(1);
    ASSERT_TRUE(a);
    ASSERT_TRUE(b);
    EXPECT_EQ(a.get(), b.get()) << "Funcmap::find should not allocate per call";
}

TEST(ParseFuncmap, EmptyBlobYieldsEmptyFuncmap)
{
    CerrCapture cap;
    Funcmap     m = parse_funcmap_section("");
    EXPECT_TRUE(m.entries.empty());
    EXPECT_TRUE(cap.str().empty());
    EXPECT_EQ(m.wave_size, 0u);
}

TEST(ParseFuncmap, MalformedWaveSizeWarns)
{
    std::string blob = "W:notanumber\n";
    CerrCapture cap;
    Funcmap     m = parse_funcmap_section(blob);
    EXPECT_EQ(m.wave_size, 0u);
    EXPECT_NE(cap.str().find("malformed W:"), std::string::npos);
}

// --- extract_elf_section: build a synthetic ELF64 in memory -----------------

namespace
{
// Lay out: ehdr | sections (data) | shstrtab data | shdr table.
// Keeps offsets simple to compute deterministically.
struct ElfBuilder
{
    std::vector<uint8_t> elf;

    struct SectionDesc
    {
        std::string          name;
        uint32_t             type;
        std::vector<uint8_t> data;
    };

    // Builds an ELF with `sections` plus an implicit `.shstrtab`. Returns
    // the byte buffer. The first synthetic section is reserved as the SHT_NULL
    // entry (index 0), as per ELF convention.
    static std::vector<uint8_t> build(const std::vector<SectionDesc>& sections,
                                      uint8_t                         elf_class = ELFCLASS64,
                                      uint16_t                        shentsize_override = 0,
                                      bool                            zero_shstrndx      = false,
                                      bool                            shoff_oob          = false)
    {
        // Layout:
        // [Elf64_Ehdr][section data, packed]  [shstrtab data]  [shdr table (NULL + sections +
        // shstrtab)]
        std::vector<uint8_t> shstrtab;
        shstrtab.push_back(0);  // index 0 must be NUL

        std::vector<uint32_t> name_offsets(sections.size(), 0);
        for(size_t i = 0; i < sections.size(); ++i)
        {
            name_offsets[i] = uint32_t(shstrtab.size());
            for(char c : sections[i].name)
                shstrtab.push_back(uint8_t(c));
            shstrtab.push_back(0);
        }
        uint32_t          shstrtab_name_off = uint32_t(shstrtab.size());
        const std::string shstrtab_name     = ".shstrtab";
        for(char c : shstrtab_name)
            shstrtab.push_back(uint8_t(c));
        shstrtab.push_back(0);

        // Compute offsets.
        size_t                cursor = sizeof(Elf64_Ehdr);
        std::vector<uint64_t> data_offsets(sections.size(), 0);
        for(size_t i = 0; i < sections.size(); ++i)
        {
            data_offsets[i] = cursor;
            cursor += sections[i].data.size();
        }
        uint64_t shstrtab_off = cursor;
        cursor += shstrtab.size();

        uint64_t shoff = cursor;
        // 1 SHT_NULL entry at index 0, then `sections.size()` user sections,
        // then 1 .shstrtab section. So total = sections.size() + 2.
        uint16_t shnum    = uint16_t(sections.size() + 2);
        uint16_t shstrndx = uint16_t(sections.size() + 1);
        if(zero_shstrndx) shstrndx = SHN_UNDEF;

        size_t shdr_size = sizeof(Elf64_Shdr) * shnum;

        std::vector<uint8_t> out(shoff + shdr_size, 0);

        Elf64_Ehdr ehdr{};
        std::memcpy(ehdr.e_ident, ELFMAG, SELFMAG);
        ehdr.e_ident[EI_CLASS]   = elf_class;
        ehdr.e_ident[EI_DATA]    = ELFDATA2LSB;
        ehdr.e_ident[EI_VERSION] = EV_CURRENT;
        ehdr.e_type              = ET_REL;
        ehdr.e_machine           = EM_NONE;
        ehdr.e_version           = EV_CURRENT;
        ehdr.e_shoff             = shoff_oob ? out.size() + 0x10000 : shoff;
        ehdr.e_ehsize            = sizeof(Elf64_Ehdr);
        ehdr.e_shentsize =
            shentsize_override != 0 ? shentsize_override : uint16_t(sizeof(Elf64_Shdr));
        ehdr.e_shnum    = shnum;
        ehdr.e_shstrndx = shstrndx;
        std::memcpy(out.data(), &ehdr, sizeof(ehdr));

        // Pack section data.
        for(size_t i = 0; i < sections.size(); ++i)
            std::memcpy(
                out.data() + data_offsets[i], sections[i].data.data(), sections[i].data.size());
        std::memcpy(out.data() + shstrtab_off, shstrtab.data(), shstrtab.size());

        // SHT_NULL entry at index 0 (already zero).
        // User sections at indices [1..sections.size()].
        for(size_t i = 0; i < sections.size(); ++i)
        {
            Elf64_Shdr s{};
            s.sh_name   = name_offsets[i];
            s.sh_type   = sections[i].type;
            s.sh_flags  = 0;
            s.sh_offset = data_offsets[i];
            s.sh_size   = sections[i].data.size();
            std::memcpy(out.data() + shoff + (i + 1) * sizeof(Elf64_Shdr), &s, sizeof(s));
        }
        // .shstrtab at index sections.size() + 1.
        Elf64_Shdr s_str{};
        s_str.sh_name   = shstrtab_name_off;
        s_str.sh_type   = SHT_STRTAB;
        s_str.sh_offset = shstrtab_off;
        s_str.sh_size   = shstrtab.size();
        std::memcpy(
            out.data() + shoff + (sections.size() + 1) * sizeof(Elf64_Shdr), &s_str, sizeof(s_str));

        return out;
    }
};

std::vector<uint8_t>
bytes_of(const std::string& s)
{
    return std::vector<uint8_t>(s.begin(), s.end());
}
}  // namespace

TEST(ExtractElfSection, FindsSection)
{
    std::string body = "F:1:foo\n";
    auto        elf  = ElfBuilder::build({
        {".sqtt_funcmap", SHT_PROGBITS, bytes_of(body)},
    });

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    ASSERT_TRUE(sec);
    EXPECT_EQ(*sec, body);
}

TEST(ExtractElfSection, AbsentSectionReturnsNullopt)
{
    auto elf = ElfBuilder::build({{".text", SHT_PROGBITS, {0x90, 0x90}}});

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);  // common case -- non-instrumented binary
}

TEST(ExtractElfSection, EmptySectionReturnsNullopt)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, {}}});

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, MultipleProgbitsSelectByName)
{
    auto elf = ElfBuilder::build({
        {".text", SHT_PROGBITS, {0x90, 0x90, 0x90}},
        {".rodata", SHT_PROGBITS, bytes_of("hello")},
        {".sqtt_funcmap", SHT_PROGBITS, bytes_of("F:1:hit\n")},
    });

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    ASSERT_TRUE(sec);
    EXPECT_EQ(*sec, "F:1:hit\n");
}

TEST(ExtractElfSection, RejectsElf32)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, bytes_of("x")}}, ELFCLASS32);

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, RejectsBadShentsize)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, bytes_of("x")}},
                                 ELFCLASS64,
                                 /*shentsize_override=*/13);

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, RejectsShoffOutOfRange)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, bytes_of("x")}},
                                 ELFCLASS64,
                                 /*shentsize_override=*/0,
                                 /*zero_shstrndx=*/false,
                                 /*shoff_oob=*/true);

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, RejectsNoShstrndx)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, bytes_of("x")}},
                                 ELFCLASS64,
                                 /*shentsize_override=*/0,
                                 /*zero_shstrndx=*/true);

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, RejectsTinyBuffer)
{
    char tiny[4] = {0};
    auto sec     = extract_elf_section(tiny, sizeof(tiny), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

TEST(ExtractElfSection, RejectsNullBuffer)
{
    auto sec = extract_elf_section(nullptr, 0, ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}

// Adversarial sh_size near 2^64 must not wrap and pass the bound check.
TEST(ExtractElfSection, RejectsAdversarialShSizeWrap)
{
    auto elf = ElfBuilder::build({{".sqtt_funcmap", SHT_PROGBITS, bytes_of("F:1:x\n")}});

    // Find the .sqtt_funcmap shdr (index 1: NULL=0, our section=1) and clobber sh_size.
    Elf64_Ehdr ehdr;
    std::memcpy(&ehdr, elf.data(), sizeof(ehdr));
    Elf64_Shdr s;
    size_t     shdr_off = ehdr.e_shoff + 1 * sizeof(Elf64_Shdr);
    std::memcpy(&s, elf.data() + shdr_off, sizeof(s));
    // sh_offset is small and valid; sh_size huge so sh_offset + sh_size wraps.
    s.sh_size = ~uint64_t(0) - s.sh_offset + 1;  // wrap target = 0
    std::memcpy(elf.data() + shdr_off, &s, sizeof(s));

    auto sec =
        extract_elf_section(reinterpret_cast<const char*>(elf.data()), elf.size(), ".sqtt_funcmap");
    EXPECT_FALSE(sec);
}
