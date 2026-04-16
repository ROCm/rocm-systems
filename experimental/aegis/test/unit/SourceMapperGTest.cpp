//===-- SourceMapperGTest.cpp - SourceMapper Tests ---------------*- C++ -*-===//
//
// Part of the AegisBit Project
//
//===----------------------------------------------------------------------===//
///
/// \file
/// Unit tests for SourceMapper: DWARF source mapping and SourceLocation
/// helper logic.  Uses the gfx950 GEMM ELF fixture (which lacks debug info)
/// to exercise the nullptr-return path and tests SourceLocation::shortFile()
/// directly.
///
//===----------------------------------------------------------------------===//

#include "aegisbit/SourceMapper.h"
#include "fixtures/gemm_gfx950_elf.h"
#include <gtest/gtest.h>

using namespace aegisbit;
using namespace llvm;

//===----------------------------------------------------------------------===//
// S-001: SourceLocation::shortFile — various path formats
//===----------------------------------------------------------------------===//

TEST(SourceMapper, ShortFileUnixPath) {
  SourceLocation Loc;
  Loc.File = "/home/user/foo.py";
  EXPECT_EQ(Loc.shortFile(), "foo.py");
}

TEST(SourceMapper, ShortFileBareName) {
  SourceLocation Loc;
  Loc.File = "bar.cpp";
  EXPECT_EQ(Loc.shortFile(), "bar.cpp");
}

TEST(SourceMapper, ShortFileEmpty) {
  SourceLocation Loc;
  EXPECT_EQ(Loc.shortFile(), "");
}

TEST(SourceMapper, ShortFileDeepPath) {
  SourceLocation Loc;
  Loc.File = "/a/b/c/d.h";
  EXPECT_EQ(Loc.shortFile(), "d.h");
}

TEST(SourceMapper, ShortFileWindowsPath) {
  SourceLocation Loc;
  Loc.File = "C:\\Users\\file.txt";
  EXPECT_EQ(Loc.shortFile(), "file.txt");
}

TEST(SourceMapper, ShortFileTrailingSlash) {
  SourceLocation Loc;
  Loc.File = "/a/b/c/";
  EXPECT_EQ(Loc.shortFile(), "");
}

//===----------------------------------------------------------------------===//
// S-002: SourceLocation::isValid
//===----------------------------------------------------------------------===//

TEST(SourceMapper, IsValidTrueWhenLineSet) {
  SourceLocation Loc;
  Loc.Line = 42;
  EXPECT_TRUE(Loc.isValid());
}

TEST(SourceMapper, IsValidFalseWhenDefault) {
  SourceLocation Loc;
  EXPECT_FALSE(Loc.isValid());
}

//===----------------------------------------------------------------------===//
// S-003: create() – error/fallback paths
//===----------------------------------------------------------------------===//

TEST(SourceMapper, CreateEmptyBytesReturnsNull) {
  std::vector<uint8_t> Empty;
  auto Mapper = SourceMapper::create(Empty);
  EXPECT_EQ(Mapper, nullptr);
}

TEST(SourceMapper, CreateSmallBytesReturnsNull) {
  std::vector<uint8_t> Small(32, 0);
  auto Mapper = SourceMapper::create(Small);
  EXPECT_EQ(Mapper, nullptr);
}

TEST(SourceMapper, CreateGarbageBytesReturnsNull) {
  std::vector<uint8_t> Garbage(256, 0xCC);
  auto Mapper = SourceMapper::create(Garbage);
  EXPECT_EQ(Mapper, nullptr);
}

TEST(SourceMapper, CreateValidELFWithoutDebugInfoReturnsNull) {
  // The GEMM fixture ELF was compiled without -g, so no DWARF debug info.
  ArrayRef<uint8_t> ELF(gemm_gfx950_elf, gemm_gfx950_elf_len);
  auto Mapper = SourceMapper::create(ELF);
  // Should return nullptr because there's no .debug_line section.
  EXPECT_EQ(Mapper, nullptr);
}

//===----------------------------------------------------------------------===//
// S-004: lookup on nullptr mapper (client code pattern)
//===----------------------------------------------------------------------===//

TEST(SourceMapper, LookupGracefulWhenNoDebugInfo) {
  // Verify the typical client pattern: check for nullptr before lookup.
  ArrayRef<uint8_t> ELF(gemm_gfx950_elf, gemm_gfx950_elf_len);
  auto Mapper = SourceMapper::create(ELF);
  if (Mapper) {
    auto Loc = Mapper->lookup(0);
    // If somehow the fixture has debug info, the lookup should still work.
    (void)Loc;
  }
  // No crash = success.
  SUCCEED();
}
