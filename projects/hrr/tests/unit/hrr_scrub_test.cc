/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Code Object Scrub
 * @{
 * @ingroup HRRTest
 * GPU-free unit tests for capture's code object scrubber (CLR's
 * hip_capture_scrub.cpp, compiled in on its own). It parses the application's
 * memory, so each case feeds it a malformed image and checks that it writes
 * nothing outside a section it has fully validated.
 */

#include "hrr_test_common.hh"

#ifdef HRR_HAVE_CAPTURE_SCRUB
#include "hip_capture_scrub.h"

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

using hrr_cap::scrub::scrub_code_object;

namespace {

constexpr uint32_t kProgbits = 1, kStrtab = 3, kNobits = 8;

template <typename T>
void put(std::vector<uint8_t>& v, size_t at, T x) {
  std::memcpy(v.data() + at, &x, sizeof(x));
}
template <typename T>
T get(const std::vector<uint8_t>& v, size_t at) {
  T x;
  std::memcpy(&x, v.data() + at, sizeof(x));
  return x;
}

struct Section {
  std::string name, data;
};

// ELF64 LSB: header, section contents, .shstrtab, then the section headers
// (null, `sections` in order, .shstrtab last). The last section's name is the
// last string in the table.
std::vector<uint8_t> make_elf(const std::vector<Section>& sections) {
  std::string names = std::string(1, '\0') + ".shstrtab" + '\0';
  struct Header { uint32_t name, type; uint64_t offset, size; };
  std::vector<Header> headers{{0, 0, 0, 0}};
  std::vector<uint8_t> image(64, 0);
  for (const auto& s : sections) {
    headers.push_back({uint32_t(names.size()), kProgbits, image.size(), s.data.size()});
    names += s.name + '\0';
    image.insert(image.end(), s.data.begin(), s.data.end());
  }
  headers.push_back({1, kStrtab, image.size(), names.size()});
  image.insert(image.end(), names.begin(), names.end());
  const uint64_t shoff = image.size();
  image.resize(shoff + headers.size() * 64, 0);
  for (size_t i = 0; i < headers.size(); ++i) {
    put(image, shoff + i * 64, headers[i].name);
    put(image, shoff + i * 64 + 4, headers[i].type);
    put(image, shoff + i * 64 + 24, headers[i].offset);
    put(image, shoff + i * 64 + 32, headers[i].size);
  }
  std::memcpy(image.data(), "\x7f" "ELF\x02\x01", 6);  // ELFCLASS64, ELFDATA2LSB
  put(image, 0x28, shoff);
  put(image, 0x3a, uint16_t{64});
  put(image, 0x3c, uint16_t(headers.size()));
  put(image, 0x3e, uint16_t(headers.size() - 1));
  return image;
}

size_t header_of(const std::vector<uint8_t>& elf, size_t i) {
  return get<uint64_t>(elf, 0x28) + i * 64;
}

std::string contents(const std::vector<uint8_t>& elf, size_t i) {
  const size_t sh = header_of(elf, i);
  const auto* p = elf.data() + get<uint64_t>(elf, sh + 24);
  return std::string(p, p + get<uint64_t>(elf, sh + 32));
}

// An uncompressed offload bundle with one entry, `elf`, whose id is `id_size` long.
std::vector<uint8_t> make_bundle(const std::vector<uint8_t>& elf, uint64_t id_size) {
  const std::string id = "hipv4-amdgcn-amd-amdhsa--gfx950";
  std::vector<uint8_t> b(24 + 8 + 24, 0);
  std::memcpy(b.data(), "__CLANG_OFFLOAD_BUNDLE__", 24);
  put(b, 24, uint64_t{1});
  put(b, 32, uint64_t(b.size() + id.size()));
  put(b, 40, uint64_t(elf.size()));
  put(b, 48, id_size);
  b.insert(b.end(), id.begin(), id.end());
  b.insert(b.end(), elf.begin(), elf.end());
  return b;
}

const std::vector<Section> kLeaky{
    {".text", "\x01\x02\x03\x04"}, {".debug_str", "/app/build/k.hip"}, {".comment", "clang version"}};

}  // namespace

HRR_TEST_CASE(Unit_HRR_Scrub_ZeroesAndRetypesBuildInfo) {
  auto elf = make_elf(kLeaky);
  REQUIRE(scrub_code_object(elf.data(), elf.size()) == 2);
  CHECK(contents(elf, 1) == kLeaky[0].data);
  CHECK(get<uint32_t>(elf, header_of(elf, 1) + 4) == kProgbits);
  for (size_t i : {size_t{2}, size_t{3}}) {
    CHECK(contents(elf, i) == std::string(kLeaky[i - 1].data.size(), '\0'));
    CHECK(get<uint32_t>(elf, header_of(elf, i) + 4) == kNobits);
  }

  auto bundle = make_bundle(make_elf(kLeaky), 31);
  REQUIRE(scrub_code_object(bundle.data(), bundle.size()) == 2);
}

// The table declares ".comment" without its terminator, so the name is
// malformed even though the byte after the table is a NUL.
HRR_TEST_CASE(Unit_HRR_Scrub_SkipsUnterminatedName) {
  auto elf = make_elf(kLeaky);
  const size_t strtab = header_of(elf, 4);
  put(elf, strtab + 32, get<uint64_t>(elf, strtab + 32) - 1);
  CHECK(scrub_code_object(elf.data(), elf.size()) == 1);  // .debug_str only
  CHECK(contents(elf, 3) == kLeaky[2].data);
  CHECK(get<uint32_t>(elf, header_of(elf, 3) + 4) == kProgbits);
  CHECK(get<uint32_t>(elf, header_of(elf, 2) + 4) == kNobits);
}

HRR_TEST_CASE(Unit_HRR_Scrub_RefusesOutOfBoundsTables) {
  const auto good = make_elf(kLeaky);
  auto unchanged = [](std::vector<uint8_t> elf) {
    const auto before = elf;
    return scrub_code_object(elf.data(), elf.size()) == 0 && elf == before;
  };

  // .comment runs one byte past the end of the image: only .debug_str changes.
  auto elf = good;
  put(elf, header_of(elf, 3) + 24, uint64_t(elf.size() - kLeaky[2].data.size() + 1));
  auto expected = elf;
  const size_t debug = header_of(expected, 2);
  std::memset(expected.data() + get<uint64_t>(expected, debug + 24), 0, kLeaky[1].data.size());
  put(expected, debug + 4, kNobits);
  CHECK(scrub_code_object(elf.data(), elf.size()) == 1);
  CHECK(elf == expected);

  elf = good;  // the string table runs past the end
  put(elf, header_of(elf, 4) + 32, uint64_t(elf.size()));
  CHECK(unchanged(elf));
  elf = good;  // its offset wraps around
  put(elf, header_of(elf, 4) + 24, ~uint64_t{0});
  CHECK(unchanged(elf));

  elf = good;  // the section table's offset wraps around
  put(elf, 0x28, ~uint64_t{0} - 63);
  CHECK(unchanged(elf));
  elf = good;  // the section table ends one byte past the image
  put(elf, 0x28, get<uint64_t>(elf, 0x28) + 1);
  CHECK(unchanged(elf));

  // A bundle entry's id length runs past the end, and wraps the cursor if added.
  CHECK(unchanged(make_bundle(good, ~uint64_t{0} - 40)));
  CHECK(unchanged(make_bundle(good, 1u << 20)));

  // Every truncation of the image, and of the bundle, is refused.
  const auto bundle = make_bundle(good, 31);
  for (const auto* whole : {&good, &bundle})
    for (size_t n = 0; n < whole->size(); ++n) {
      std::vector<uint8_t> cut(whole->begin(), whole->begin() + n);
      CHECK(scrub_code_object(cut.data(), cut.size()) == 0);
    }
}

#else
// projects/hrr was checked out without the CLR sources (tests/unit/CMakeLists.txt).
#define HRR_SCRUB_NOT_BUILT(name) \
  HRR_TEST_CASE(name) { HRR_SKIP("built without CLR's hip_capture_scrub.cpp"); }
HRR_SCRUB_NOT_BUILT(Unit_HRR_Scrub_ZeroesAndRetypesBuildInfo)
HRR_SCRUB_NOT_BUILT(Unit_HRR_Scrub_SkipsUnterminatedName)
HRR_SCRUB_NOT_BUILT(Unit_HRR_Scrub_RefusesOutOfBoundsTables)
#endif

/**
 * @}
 */
