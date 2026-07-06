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

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <set>
#include <string>
#include <vector>

#include <elf.h>

namespace {

// The complete set of libc entry points the interposer overrides. This MUST be
// kept in sync with the extern "C" definitions in
// lib/rocjitsu/src/rocjitsu/kmd/linux/interposer.cpp. When you add or remove an
// interposed libc function, update both.
constexpr const char *kInterposedLibcSymbols[] = {
    "access",     "close",      "dup",      "dup2",       "dup3",         "fcntl",    "fcntl64",
    "fopen",      "fopen64",    "fork",     "freopen",    "freopen64",    "fstat",    "fstat64",
    "__fxstat",   "__fxstat64", "ioctl",    "lstat",      "lstat64",      "__lxstat", "__lxstat64",
    "madvise",    "mmap",       "mprotect", "munmap",     "open",         "open64",   "__open_2",
    "__open64_2", "openat",     "openat64", "__openat_2", "__openat64_2", "opendir",  "readlink",
    "stat",       "stat64",     "__xstat",  "__xstat64",
};

// Internal interposer helpers that must NOT leak into the dynamic symbol table.
constexpr const char *kInternalHelpers[] = {
    "fcntl_impl",
    "fcntl_arg_kind",
};

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
  }

  static DynamicSymbols symbols_;
  static std::string path_;
};

DynamicSymbols InterposerExportsTest::symbols_;
std::string InterposerExportsTest::path_;

// Every libc entry point the LD_PRELOAD shim overrides must be an exported,
// defined dynamic symbol, otherwise the preload cannot intercept that syscall.
TEST_F(InterposerExportsTest, AllInterposedLibcSymbolsExported) {
  for (const char *name : kInterposedLibcSymbols) {
    EXPECT_TRUE(symbols_.is_exported(name))
        << "librocjitsu.so must export interposed libc symbol '" << name
        << "'. Without it the LD_PRELOAD shim cannot intercept that part of "
           "the amdgpu/KFD syscall surface. Check the RJ_INTERPOSER_EXPORT "
           "marker on its definition in interposer.cpp.";
  }
}

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

// The library's real ABI is entirely extern "C": the rj_* C API plus the
// interposed libc entry points. Without a linker version script the dynamic
// table cannot be made perfectly pristine -- libstdc++ marks namespace std as
// _GLIBCXX_VISIBILITY(default), so std::/__gnu_cxx:: template instantiations
// emitted into our own objects stay exported even under -fvisibility=hidden.
// Those residuals are all Itanium-C++-mangled (_Z...) and never live in our own
// namespace, so we tolerate them while still flagging any unmangled (extern
// "C") leak or any accidentally exported rocjitsu:: C++ symbol.
TEST_F(InterposerExportsTest, ExportSurfaceIsNotOverExposed) {
  const std::set<std::string> allowed(std::begin(kInterposedLibcSymbols),
                                      std::end(kInterposedLibcSymbols));
  std::vector<std::string> unexpected;
  for (const std::string &name : symbols_.exported()) {
    if (name.rfind("rj_", 0) == 0)
      continue; // public C API
    if (allowed.count(name) != 0)
      continue; // interposed libc entry point
    // Tolerate C++ standard-library runtime instantiations: they are mangled
    // (_Z...) and never in our own namespace. Anything else -- an unmangled
    // extern "C" symbol or a mangled rocjitsu:: symbol -- is a real leak.
    const bool cxx_mangled = name.rfind("_Z", 0) == 0;
    const bool ours = name.find("rocjitsu") != std::string::npos;
    if (cxx_mangled && !ours)
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
