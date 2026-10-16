/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "hip_capture_scrub.h"

#include <cstring>

// Field offsets are spelled out rather than taken from <elf.h>, which Windows
// does not have and whose Elf64_* typedefs collide with CLR's own ELF headers.
// The image is the application's memory, so every read is bounds-checked and
// copied out with memcpy, whatever its alignment.

namespace hrr_cap {
namespace scrub {

namespace {

constexpr size_t kElfHeaderSize = 64;
constexpr size_t kSectionHeaderSize = 64;
constexpr uint8_t kElfClass64 = 2;
constexpr uint8_t kElfDataLsb = 1;
constexpr uint32_t kShtNobits = 8;
constexpr uint64_t kShfAlloc = 0x2;

constexpr char kBundleMagic[] = "__CLANG_OFFLOAD_BUNDLE__";
constexpr size_t kBundleMagicLen = sizeof(kBundleMagic) - 1;
constexpr size_t kBundleEntryHeaderSize = 3 * sizeof(uint64_t);

template <typename T>
T read(const uint8_t* p, size_t at) {
  T v;
  std::memcpy(&v, p + at, sizeof(v));
  return v;
}

// [offset, offset + size) lies inside [0, limit).
bool in_range(uint64_t offset, uint64_t size, size_t limit) {
  return offset <= limit && size <= limit - offset;
}

bool has_prefix(const char* s, size_t len, const char* prefix) {
  const size_t n = std::strlen(prefix);
  return len >= n && std::memcmp(s, prefix, n) == 0;
}

bool is_scrubbed_section(const char* name, size_t len) {
  return (len == 8 && std::memcmp(name, ".comment", 8) == 0) || has_prefix(name, len, ".debug");
}

size_t scrub_elf(uint8_t* image, size_t size) {
  if (size < kElfHeaderSize || std::memcmp(image, "\x7f" "ELF", 4) != 0 ||
      image[4] != kElfClass64 || image[5] != kElfDataLsb)
    return 0;
  const uint64_t shoff = read<uint64_t>(image, 0x28);
  const uint16_t shentsize = read<uint16_t>(image, 0x3a);
  const uint16_t shnum = read<uint16_t>(image, 0x3c);
  const uint16_t shstrndx = read<uint16_t>(image, 0x3e);
  // shnum == 0 also covers extended section numbering, which code objects never need.
  if (shentsize != kSectionHeaderSize || shnum == 0 || shstrndx >= shnum ||
      !in_range(shoff, uint64_t{shnum} * kSectionHeaderSize, size))
    return 0;

  const uint8_t* shstr_hdr = image + shoff + size_t{shstrndx} * kSectionHeaderSize;
  const uint64_t names_off = read<uint64_t>(shstr_hdr, 24);
  const uint64_t names_size = read<uint64_t>(shstr_hdr, 32);
  if (!in_range(names_off, names_size, size)) return 0;
  const char* names = reinterpret_cast<const char*>(image + names_off);

  size_t zeroed = 0;
  for (uint16_t i = 0; i < shnum; ++i) {
    const uint8_t* sh = image + shoff + size_t{i} * kSectionHeaderSize;
    const uint32_t name = read<uint32_t>(sh, 0);
    const uint32_t type = read<uint32_t>(sh, 4);
    const uint64_t flags = read<uint64_t>(sh, 8);
    const uint64_t offset = read<uint64_t>(sh, 24);
    const uint64_t sec_size = read<uint64_t>(sh, 32);
    if (name >= names_size || type == kShtNobits || (flags & kShfAlloc) != 0) continue;
    const size_t name_len = strnlen(names + name, names_size - name);
    if (!is_scrubbed_section(names + name, name_len)) continue;
    if (sec_size == 0 || !in_range(offset, sec_size, size)) continue;
    std::memset(image + offset, 0, sec_size);
    ++zeroed;
  }
  return zeroed;
}

size_t scrub_bundle(uint8_t* image, size_t size) {
  size_t pos = kBundleMagicLen;
  if (size < pos + sizeof(uint64_t)) return 0;
  const uint64_t count = read<uint64_t>(image, pos);
  pos += sizeof(uint64_t);
  size_t zeroed = 0;
  for (uint64_t i = 0; i < count; ++i) {
    if (size - pos < kBundleEntryHeaderSize) break;
    const uint64_t offset = read<uint64_t>(image, pos);
    const uint64_t entry_size = read<uint64_t>(image, pos + 8);
    const uint64_t id_size = read<uint64_t>(image, pos + 16);
    pos += kBundleEntryHeaderSize;
    if (id_size > size - pos) break;
    pos += id_size;
    if (in_range(offset, entry_size, size)) zeroed += scrub_elf(image + offset, entry_size);
  }
  return zeroed;
}

}  // namespace

size_t scrub_code_object(uint8_t* image, size_t size) {
  if (!image) return 0;
  if (size >= kBundleMagicLen && std::memcmp(image, kBundleMagic, kBundleMagicLen) == 0)
    return scrub_bundle(image, size);
  return scrub_elf(image, size);
}

}  // namespace scrub
}  // namespace hrr_cap
