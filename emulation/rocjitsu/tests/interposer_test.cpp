// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file interposer_test.cpp
/// @brief Verifies librocjitsu.so exports the LD_PRELOAD interposer ABI.
///
/// @details The KMD interposer (kmd/linux/interposer.cpp) overrides a fixed set
/// of libc entry points (open/ioctl/mmap/stat/...) so an LD_PRELOAD of
/// librocjitsu.so can redirect the amdgpu/KFD syscall surface to the simulated
/// driver. Those overrides only take effect while they remain *exported*
/// dynamic symbols: the project-wide -fvisibility=hidden preset plus
/// --exclude-libs (see the rocjitsu_shared link options in CMakeLists.txt) strip
/// almost everything, so a mistake there silently turns the preload into a
/// no-op (the library loads but never redirects a single syscall).
///
/// This test parses the ELF dynamic symbol table of the freshly built
/// librocjitsu.so and asserts every interposed libc symbol is present as a
/// defined, globally-bound export, that the internal fcntl helpers stay hidden,
/// and that the extern "C" surface is not over-exposed. It intentionally
/// inspects the ELF image directly rather than dlopen()-ing the library so the
/// interposer's constructors (which install a SIGSEGV handler and may contact
/// the daemon) never run in the test process.
///
/// The expected set of interposed libc symbols is not hand-maintained: it is
/// auto-detected at run time by scanning the interposer source
/// (kmd/linux/interposer.cpp, whose path is baked in via
/// RJ_INTERPOSER_SOURCE_PATH) for the RJ_INTERPOSER_EXPORT marker, so adding or
/// removing an override needs no change here.

#include <gtest/gtest.h>

#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <regex>
#include <set>
#include <string>
#include <vector>

#include <elf.h>

namespace {

// Internal interposer helpers that must NOT leak into the dynamic symbol table.
constexpr const char *kInternalHelpers[] = {
    "fcntl_impl",
    "fcntl_arg_kind",
};

// Auto-detect the libc entry points the interposer overrides straight from the
// interposer source, so the expected export set is never hand-maintained. Each
// override in kmd/linux/interposer.cpp is written as:
//
//     RJ_INTERPOSER_EXPORT <return type> <name>(<args>) { ... }
//
// so the interposed symbol is the identifier immediately preceding the '(' that
// follows the RJ_INTERPOSER_EXPORT marker. Scanning for that marker keeps this
// test in lock-step with interposer.cpp: add or remove an override there and the
// expectations below follow automatically.
bool ParseInterposedSymbols(const std::string &source, std::set<std::string> &out,
                            std::string &err) {
  static constexpr char kMarker[] = "RJ_INTERPOSER_EXPORT";
  const size_t marker_len = std::strlen(kMarker);
  auto is_ident = [](char c) {
    return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
  };

  for (size_t pos = 0; (pos = source.find(kMarker, pos)) != std::string::npos;) {
    const size_t marker_begin = pos;
    pos += marker_len;
    // Match RJ_INTERPOSER_EXPORT as a whole token so we never pick up a longer
    // identifier that merely contains the marker text: the preceding character
    // must not be part of an identifier and the next must be whitespace.
    const bool boundary_before = marker_begin == 0 || !is_ident(source[marker_begin - 1]);
    const bool boundary_after =
        pos < source.size() && std::isspace(static_cast<unsigned char>(source[pos])) != 0;
    if (!boundary_before || !boundary_after)
      continue;

    // The parameter list opens at the first '(' after the marker.
    const size_t paren = source.find('(', pos);
    if (paren == std::string::npos)
      break;
    // A real definition has only "<return type> <name>" on the marker line
    // between the marker and '('. A statement terminator or line break there
    // means this is not an interposer definition (e.g. a mention in a comment).
    if (source.substr(pos, paren - pos).find_first_of(";{}#\n") != std::string::npos)
      continue;

    // The function name is the last identifier token before '('.
    size_t name_end = paren;
    while (name_end > pos && std::isspace(static_cast<unsigned char>(source[name_end - 1])) != 0)
      --name_end;
    size_t name_begin = name_end;
    while (name_begin > pos && is_ident(source[name_begin - 1]))
      --name_begin;
    if (name_begin < name_end)
      out.insert(source.substr(name_begin, name_end - name_begin));
  }

  if (out.empty()) {
    err = "found no RJ_INTERPOSER_EXPORT definitions in the interposer source";
    return false;
  }
  return true;
}

// Defined, globally-visible dynamic symbols of an ELF object, i.e. the set that
// `nm --dynamic --defined-only` reports as `T`/`W`.
class DynamicSymbols {
public:
  bool is_exported(const std::string &name) const { return exported_.count(name) != 0; }

  const std::set<std::string> &exported() const { return exported_; }

  void insert(std::string name) { exported_.insert(std::move(name)); }

private:
  std::set<std::string> exported_;
};

// Parse the exported dynamic symbols out of an in-memory ELF64 image. Every
// offset is bounds-checked against `buf` before use so a malformed file yields
// a clean failure rather than an out-of-bounds read.
bool ParseExportedSymbols(const std::vector<uint8_t> &buf, DynamicSymbols &out, std::string &err) {
  auto fail = [&](const char *message) {
    err = message;
    return false;
  };

  if (buf.size() < sizeof(Elf64_Ehdr))
    return fail("file smaller than an ELF64 header");
  if (std::memcmp(buf.data(), ELFMAG, SELFMAG) != 0)
    return fail("not an ELF file (bad magic)");
  if (buf[EI_CLASS] != ELFCLASS64)
    return fail("not an ELF64 object");

  Elf64_Ehdr eh;
  std::memcpy(&eh, buf.data(), sizeof(eh));
  // Copy the 16-bit header fields into wider unsigned locals so the bounds and
  // loop comparisons below stay same-signedness (avoids -Wsign-compare).
  const uint64_t shoff = eh.e_shoff;
  const unsigned shnum = eh.e_shnum;
  const uint64_t shentsize = eh.e_shentsize;
  if (shoff == 0 || shentsize < sizeof(Elf64_Shdr))
    return fail("no section headers");

  const uint64_t sh_table_end = shoff + static_cast<uint64_t>(shnum) * shentsize;
  if (sh_table_end > buf.size())
    return fail("section header table out of bounds");

  auto read_section = [&](unsigned index) {
    Elf64_Shdr sh;
    std::memcpy(&sh, buf.data() + shoff + static_cast<uint64_t>(index) * shentsize, sizeof(sh));
    return sh;
  };

  // Locate the dynamic symbol table (.dynsym) and its string table (.dynstr).
  Elf64_Shdr dynsym{};
  Elf64_Shdr dynstr{};
  bool found = false;
  for (unsigned i = 0; i < shnum; ++i) {
    Elf64_Shdr sh = read_section(i);
    if (sh.sh_type == SHT_DYNSYM) {
      if (sh.sh_link >= shnum)
        return fail(".dynsym has an invalid sh_link");
      dynsym = sh;
      dynstr = read_section(sh.sh_link);
      found = true;
      break;
    }
  }
  if (!found)
    return fail("no .dynsym section (library has no dynamic symbol table)");
  if (dynsym.sh_entsize < sizeof(Elf64_Sym))
    return fail(".dynsym has an invalid entry size");
  if (dynsym.sh_offset + dynsym.sh_size > buf.size())
    return fail(".dynsym out of bounds");
  if (dynstr.sh_offset + dynstr.sh_size > buf.size())
    return fail(".dynstr out of bounds");

  const char *strtab = reinterpret_cast<const char *>(buf.data() + dynstr.sh_offset);
  const uint64_t strtab_size = dynstr.sh_size;
  const uint64_t count = dynsym.sh_size / dynsym.sh_entsize;
  for (uint64_t i = 0; i < count; ++i) {
    Elf64_Sym sym;
    std::memcpy(&sym, buf.data() + dynsym.sh_offset + i * dynsym.sh_entsize, sizeof(sym));

    const unsigned bind = ELF64_ST_BIND(sym.st_info);
    const unsigned vis = ELF64_ST_VISIBILITY(sym.st_other);
    const bool defined = sym.st_shndx != SHN_UNDEF;
    const bool global = bind == STB_GLOBAL || bind == STB_WEAK;
    const bool visible = vis == STV_DEFAULT || vis == STV_PROTECTED;
    if (!defined || !global || !visible)
      continue;
    if (sym.st_name >= strtab_size)
      continue; // malformed name offset; skip defensively
    const char *name = strtab + sym.st_name;
    if (std::memchr(name, '\0', strtab_size - sym.st_name) == nullptr)
      continue; // name is not NUL-terminated within the string table
    if (name[0] != '\0')
      out.insert(name);
  }
  return true;
}

// Resolve the path to the librocjitsu.so under test. The build system bakes in
// the freshly built path via RJ_LIBROCJITSU_SO_PATH; RJ_LIBROCJITSU_SO can
// override it (e.g. to point at a different build).
const char *LibrocjitsuPath() {
  if (const char *env = std::getenv("RJ_LIBROCJITSU_SO"))
    if (env[0] != '\0')
      return env;
#ifdef RJ_LIBROCJITSU_SO_PATH
  return RJ_LIBROCJITSU_SO_PATH;
#else
  return nullptr;
#endif
}

// Resolve the path to the interposer source whose RJ_INTERPOSER_EXPORT markers
// define the expected export set. The build bakes in RJ_INTERPOSER_SOURCE_PATH;
// RJ_INTERPOSER_SOURCE can override it (e.g. to point at a different checkout).
const char *InterposerSourcePath() {
  if (const char *env = std::getenv("RJ_INTERPOSER_SOURCE"))
    if (env[0] != '\0')
      return env;
#ifdef RJ_INTERPOSER_SOURCE_PATH
  return RJ_INTERPOSER_SOURCE_PATH;
#else
  return nullptr;
#endif
}

class InterposerExportsTest : public ::testing::Test {
protected:
  static void SetUpTestSuite() {
    const char *path = LibrocjitsuPath();
    ASSERT_NE(path, nullptr) << "librocjitsu.so path not configured (set RJ_LIBROCJITSU_SO or the "
                                "RJ_LIBROCJITSU_SO_PATH compile definition).";

    std::ifstream file(path, std::ios::binary);
    ASSERT_TRUE(file.good()) << "cannot open librocjitsu.so at: " << path;
    std::vector<uint8_t> buf((std::istreambuf_iterator<char>(file)),
                             std::istreambuf_iterator<char>());
    ASSERT_FALSE(buf.empty()) << "librocjitsu.so is empty: " << path;

    std::string err;
    ASSERT_TRUE(ParseExportedSymbols(buf, symbols_, err))
        << "failed to parse ELF " << path << ": " << err;
    path_ = path;

    // Auto-detect the interposed libc entry points from the interposer source.
    const char *src = InterposerSourcePath();
    ASSERT_NE(src, nullptr) << "interposer source path not configured (set RJ_INTERPOSER_SOURCE or "
                               "the RJ_INTERPOSER_SOURCE_PATH compile definition).";
    std::ifstream src_file(src, std::ios::binary);
    ASSERT_TRUE(src_file.good()) << "cannot open interposer source at: " << src;
    const std::string source((std::istreambuf_iterator<char>(src_file)),
                             std::istreambuf_iterator<char>());
    ASSERT_TRUE(ParseInterposedSymbols(source, interposed_, err))
        << "failed to auto-detect interposed symbols from " << src << ": " << err;
  }

  static DynamicSymbols symbols_;
  static std::set<std::string> interposed_;
  static std::string path_;
};

DynamicSymbols InterposerExportsTest::symbols_;
std::set<std::string> InterposerExportsTest::interposed_;
std::string InterposerExportsTest::path_;

// The fcntl helpers are implementation details and must stay hidden.
TEST_F(InterposerExportsTest, InternalHelpersStayHidden) {
  for (const char *name : kInternalHelpers) {
    EXPECT_FALSE(symbols_.is_exported(name))
        << "internal interposer helper '" << name << "' must not be exported from librocjitsu.so.";
  }
}

// Sanity check that the ELF we parsed is actually librocjitsu.so: its public
// rj_* C API should be present in the export table.
TEST_F(InterposerExportsTest, PublicCApiIsExported) {
  bool any_rj = false;
  for (const std::string &name : symbols_.exported()) {
    if (name.rfind("rj_", 0) == 0) {
      any_rj = true;
      break;
    }
  }
  EXPECT_TRUE(any_rj) << "expected at least one rj_* C API symbol among the exports of " << path_
                      << " (sanity check that the correct library was inspected).";
}

// The auto-detection itself must find the interposer surface: a broken scan
// (e.g. the marker spelling or the source layout changed) would otherwise
// silently turn the expectations above into a vacuous pass over an empty set.
TEST_F(InterposerExportsTest, AutoDetectedInterposerSurfaceLooksSane) {
  ASSERT_FALSE(interposed_.empty())
      << "no interposed symbols were auto-detected from the interposer source; the "
         "RJ_INTERPOSER_EXPORT scan in ParseInterposedSymbols is likely broken.";
  // A few core entry points that must always be part of the redirected surface.
  for (const char *anchor : {"open", "close", "ioctl", "mmap", "munmap"}) {
    EXPECT_TRUE(interposed_.count(anchor) != 0)
        << "expected core libc entry point '" << anchor
        << "' among the auto-detected interposed symbols.";
  }
}

// The library's real ABI is entirely extern "C": the rj_* C API plus the
// interposed libc entry points. Without a linker version script the dynamic
// table cannot be made perfectly pristine -- libstdc++ marks namespace std as
// _GLIBCXX_VISIBILITY(default), so std:: template instantiations emitted into
// our own objects stay exported even under -fvisibility=hidden. Those residuals
// are all std::-rooted Itanium C++ manglings, so we tolerate exactly that shape
// (a positive std:: match) while still flagging any unmangled (extern "C") leak
// or any other mangled C++ symbol (e.g. an accidentally exported rocjitsu:: one).
TEST_F(InterposerExportsTest, ExportSurfaceIsNotOverExposed) {
  const std::regex std_mangled("^_Z[A-Z]*St");
  std::vector<std::string> unexpected;
  for (const std::string &name : symbols_.exported()) {
    if (name.rfind("rj_", 0) == 0)
      continue; // public C API
    if (interposed_.count(name) != 0)
      continue; // interposed libc entry point
    // Tolerate C++ standard-library runtime instantiations only: their mangled
    // names are std::-rooted (^_Z[A-Z]*St). Anything else -- an unmangled extern
    // "C" symbol or a mangled non-std (e.g. rocjitsu::) symbol -- is a real leak.
    if (std::regex_search(name, std_mangled))
      continue;
    unexpected.push_back(name);
  }

  std::string sample;
  for (size_t i = 0; i < unexpected.size() && i < 10; ++i)
    sample += (i == 0 ? "" : ", ") + unexpected[i];

  EXPECT_TRUE(unexpected.empty())
      << "librocjitsu.so exports " << unexpected.size()
      << " unexpected dynamic symbol(s); the extern \"C\" surface should be "
         "exactly the rj_* C API plus the interposed libc entry points (C++ "
         "standard-library runtime instantiations excepted). Sample: "
      << sample;
}

} // namespace
