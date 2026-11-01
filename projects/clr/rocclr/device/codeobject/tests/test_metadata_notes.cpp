/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// NT_AMDGPU_METADATA notes: comgr's rules for merging several notes (Tensile code objects carry
// one per kernel), and rejection of malformed or conflicting notes. The kernel callback here only
// records .name; the field interpretation is the runtime's (devkernel.cpp).

#include "device/codeobject/metadata_notes.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <string_view>
#include <vector>

using namespace amd::codeobject;

namespace {

//! Minimal MessagePack writer for the notes
class Pack {
 public:
  Pack& map(uint32_t n) { return header(0x80, 0xde, n); }
  Pack& array(uint32_t n) { return header(0x90, 0xdc, n); }
  Pack& str(std::string_view s) {
    if (s.size() < 32) {
      bytes_.push_back(static_cast<char>(0xa0u | s.size()));
    } else {
      bytes_.push_back(static_cast<char>(0xd9));
      bytes_.push_back(static_cast<char>(s.size()));
    }
    bytes_.insert(bytes_.end(), s.begin(), s.end());
    return *this;
  }
  Pack& uint(uint8_t v) {
    bytes_.push_back(static_cast<char>(0xcc));
    bytes_.push_back(static_cast<char>(v));
    return *this;
  }
  Pack& boolean(bool v) {
    bytes_.push_back(static_cast<char>(v ? 0xc3 : 0xc2));
    return *this;
  }
  //! A kernel map with .name and a few other fields
  Pack& kernel(std::string_view name) {
    return map(3).str(".name").str(name).str(".sgpr_count").uint(12).str(".args").array(1)
        .map(2).str(".size").uint(8).str(".value_kind").str("global_buffer");
  }
  //! amdhsa.version [major, minor]
  Pack& version(uint8_t major, uint8_t minor) {
    return str("amdhsa.version").array(2).uint(major).uint(minor);
  }
  const std::vector<char>& bytes() const { return bytes_; }

 private:
  Pack& header(uint8_t fix, uint8_t tag16, uint32_t n) {
    if (n < 16) {
      bytes_.push_back(static_cast<char>(fix | n));
    } else {
      bytes_.push_back(static_cast<char>(tag16));
      bytes_.push_back(static_cast<char>(n >> 8u));
      bytes_.push_back(static_cast<char>(n));
    }
    return *this;
  }

  std::vector<char> bytes_;  //!< Encoded bytes
};

//! Notes parser whose kernel callback records each kernel's .name
struct Fixture {
  std::vector<std::string> printf;
  std::vector<std::string> kernels;
  bool failKernels = false;
  MetadataNotes notes{printf, [this](MsgPackReader& reader) {
                        if (failKernels) return false;
                        uint32_t fields = 0;
                        if (!reader.readMapSize(&fields)) return false;
                        for (uint32_t i = 0; i < fields; ++i) {
                          std::string_view key;
                          if (!reader.readString(&key)) return false;
                          std::string_view name;
                          if (key == ".name") {
                            if (!reader.readString(&name)) return false;
                            kernels.emplace_back(name);
                          } else if (!reader.skip()) {
                            return false;
                          }
                        }
                        return true;
                      }};

  bool parse(const Pack& note) {
    return notes.parse(note.bytes().data(), static_cast<uint32_t>(note.bytes().size()));
  }
};

Pack Note(uint8_t minor, std::vector<std::string_view> kernels,
          std::vector<std::string_view> printf = {}) {
  Pack p;
  p.map(printf.empty() ? 2 : 3).version(1, minor).str("amdhsa.kernels").array(kernels.size());
  for (const std::string_view name : kernels) p.kernel(name);
  if (!printf.empty()) {
    p.str("amdhsa.printf").array(printf.size());
    for (const std::string_view format : printf) p.str(format);
  }
  return p;
}

}  // namespace

TEST_CASE("One note: version, kernels and printf", "[notes]") {
  Fixture f;
  REQUIRE(f.parse(Note(2, {"a", "b"}, {"1:1:4:%d\\n"})));
  CHECK(f.notes.hasVersion());
  CHECK(f.notes.version() == std::make_pair(uint64_t{1}, uint64_t{2}));
  CHECK(f.notes.hasKernels());
  CHECK(f.kernels == std::vector<std::string>{"a", "b"});
  CHECK(f.printf == std::vector<std::string>{"1:1:4:%d\\n"});
}

TEST_CASE("Notes are merged: kernels concatenate, the first printf list is kept", "[notes]") {
  Fixture f;
  REQUIRE(f.parse(Note(2, {"a"}, {"p0", "p1"})));
  REQUIRE(f.parse(Note(2, {"b", "c"}, {"p0", "p1"})));
  REQUIRE(f.parse(Note(2, {"d"})));
  CHECK(f.kernels == std::vector<std::string>{"a", "b", "c", "d"});
  CHECK(f.printf == std::vector<std::string>{"p0", "p1"});
}

TEST_CASE("printf comes from the first note that has it", "[notes]") {
  Fixture f;
  REQUIRE(f.parse(Note(2, {"a"})));
  REQUIRE(f.parse(Note(2, {"b"}, {"q"})));
  CHECK(f.printf == std::vector<std::string>{"q"});
}

TEST_CASE("Conflicting notes are rejected", "[notes]") {
  Fixture f;
  REQUIRE(f.parse(Note(2, {"a"}, {"p0", "p1"})));
  SECTION("different version") {
    CHECK_FALSE(f.parse(Note(1, {"b"})));
    CHECK(f.notes.error() == "Code object metadata notes have different versions");
  }
  SECTION("different printf string") {
    CHECK_FALSE(f.parse(Note(2, {"b"}, {"p0", "pX"})));
    CHECK(f.notes.error() == "Code object metadata notes have different printf strings");
  }
  SECTION("different printf count") {
    CHECK_FALSE(f.parse(Note(2, {"b"}, {"p0"})));
    CHECK(f.notes.error() == "Code object metadata notes have different printf strings");
  }
  // The first note's kernels were built before the conflict
  CHECK(f.kernels.front() == "a");
}

TEST_CASE("A malformed later note fails after the earlier ones were read", "[notes]") {
  Fixture f;
  REQUIRE(f.parse(Note(2, {"a"})));
  const Pack later = Note(2, {"b", "c"});
  const std::vector<char> cut(later.bytes().begin(), later.bytes().end() - 3);
  CHECK_FALSE(f.notes.parse(cut.data(), static_cast<uint32_t>(cut.size())));
  CHECK(f.notes.error() == "Malformed code object metadata");
}

TEST_CASE("Unknown keys are skipped at the top level", "[notes]") {
  Fixture f;
  Pack p;
  p.map(4)
      .str("amdhsa.target").str("amdgcn-amd-amdhsa--gfx1100")
      .str("amdhsa.future").map(2).str("x").array(2).uint(1).boolean(true).str("y").map(0)
      .version(1, 2)
      .str("amdhsa.kernels").array(1).kernel("k");
  REQUIRE(f.parse(p));
  CHECK(f.kernels == std::vector<std::string>{"k"});
}

TEST_CASE("Kernel lists: missing, empty, failing", "[notes]") {
  Fixture f;
  SECTION("no amdhsa.kernels") {
    Pack p;
    p.map(1).version(1, 2);
    REQUIRE(f.parse(p));
    CHECK_FALSE(f.notes.hasKernels());
  }
  SECTION("empty amdhsa.kernels") {
    REQUIRE(f.parse(Note(2, {})));
    CHECK(f.notes.hasKernels());
    CHECK(f.kernels.empty());
  }
  SECTION("a kernel map the callback rejects") {
    f.failKernels = true;
    CHECK_FALSE(f.parse(Note(2, {"a"})));
    CHECK(f.notes.error() == "Malformed code object metadata");
  }
}

TEST_CASE("amdhsa.version shapes", "[notes]") {
  Fixture f;
  SECTION("extra elements are skipped") {
    Pack p;
    p.map(1).str("amdhsa.version").array(3).uint(1).uint(2).str("x");
    REQUIRE(f.parse(p));
    CHECK(f.notes.version() == std::make_pair(uint64_t{1}, uint64_t{2}));
  }
  SECTION("one element") {
    Pack p;
    p.map(1).str("amdhsa.version").array(1).uint(1);
    CHECK_FALSE(f.parse(p));
  }
  SECTION("not an array") {
    Pack p;
    p.map(1).str("amdhsa.version").str("1.2");
    CHECK_FALSE(f.parse(p));
  }
  SECTION("no version at all") {
    REQUIRE(f.parse(Pack().map(0)));
    CHECK_FALSE(f.notes.hasVersion());
  }
}

TEST_CASE("A count larger than the note fails without allocating for it", "[notes]") {
  // { "amdhsa.printf": array32(0xffffffff) } followed by one string: the count would reserve
  // ~137 GB of std::string if it were trusted
  Pack p;
  p.map(1).str("amdhsa.printf");
  std::vector<char> note = p.bytes();
  for (const unsigned char byte : {0xddu, 0xffu, 0xffu, 0xffu, 0xffu}) {
    note.push_back(static_cast<char>(byte));
  }
  note.push_back(static_cast<char>(0xa1));
  note.push_back('x');
  Fixture f;
  bool ok = true;
  CHECK_NOTHROW(ok = f.notes.parse(note.data(), static_cast<uint32_t>(note.size())));
  CHECK_FALSE(ok);
  CHECK(f.printf.capacity() <= note.size());
}

TEST_CASE("A note that isn't a map is malformed", "[notes]") {
  Fixture f;
  Pack p;
  p.array(1).uint(1);
  CHECK_FALSE(f.parse(p));
  CHECK(f.notes.error() == "Malformed code object metadata");
}

TEST_CASE("No prefix of a note parses", "[notes]") {
  const Pack note = Note(2, {"a", "b"}, {"p0"});
  const std::vector<char>& full = note.bytes();
  for (size_t size = 0; size < full.size(); ++size) {
    INFO("size " << size);
    Fixture f;
    const std::vector<char> prefix(full.begin(),
                                   full.begin() + static_cast<std::ptrdiff_t>(size));
    CHECK_FALSE(f.notes.parse(prefix.data(), static_cast<uint32_t>(prefix.size())));
  }
}
