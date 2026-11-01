/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

// Compressed offload bundles (CCOB, llvm/lib/Object/OffloadBundle.cpp): header versions 1-3,
// zlib and zstd. Current clang writes only versions 2 and 3 with zstd, so every combination is
// built here with the compression libraries the runtime loads.

#include "device/codeobject/compressed_bundle.hpp"
#include "device/codeobject/elf_header.hpp"
#include "os/os.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstring>
#include <string>
#include <vector>

using namespace amd::codeobject;
using amd::Os;

namespace {

constexpr uint16_t kZlib = 0;
constexpr uint16_t kZstd = 1;

#if defined(_WIN32)
constexpr const char* kZstdNames[] = {"zstd.dll", "libzstd.dll"};
constexpr const char* kZlibNames[] = {"zlib1.dll", "zlib.dll"};
#else
constexpr const char* kZstdNames[] = {"librocm_sysdeps_zstd.so.1", "libzstd.so.1"};
constexpr const char* kZlibNames[] = {"librocm_sysdeps_z.so.1", "libz.so.1"};
#endif

template <size_t N>
void* FindSymbol(const char* const (&names)[N], const char* symbol) {
  for (const char* name : names) {
    void* handle = Os::loadLibrary(name);
    if (handle == nullptr) continue;
    if (void* fn = Os::getSymbol(handle, symbol)) return fn;
  }
  return nullptr;
}

//! Compresses @p data with @p method; empty if the library isn't installed
std::vector<char> Compress(uint16_t method, const std::vector<char>& data) {
  std::vector<char> out(data.size() + 1024);
  if (method == kZstd) {
    using Fn = size_t (*)(void*, size_t, const void*, size_t, int);
    auto compress = reinterpret_cast<Fn>(FindSymbol(kZstdNames, "ZSTD_compress"));
    if (compress == nullptr) return {};
    out.resize(compress(out.data(), out.size(), data.data(), data.size(), 3));
  } else {
    using Fn = int (*)(unsigned char*, unsigned long*, const unsigned char*, unsigned long, int);
    auto compress = reinterpret_cast<Fn>(FindSymbol(kZlibNames, "compress2"));
    if (compress == nullptr) return {};
    unsigned long size = static_cast<unsigned long>(out.size());
    REQUIRE(compress(reinterpret_cast<unsigned char*>(out.data()), &size,
                     reinterpret_cast<const unsigned char*>(data.data()), data.size(), 6) == 0);
    out.resize(size);
  }
  return out;
}

void AppendLe(std::vector<char>& out, uint64_t value, size_t bytes) {
  for (size_t i = 0; i < bytes; ++i) out.push_back(static_cast<char>(value >> (8 * i)));
}

//! A CCOB of @p version around @p compressed (uncompressed size @p rawSize)
std::vector<char> MakeCcob(uint16_t version, uint16_t method, const std::vector<char>& compressed,
                           uint64_t rawSize) {
  const size_t headerSize = (version == 1) ? 20 : (version == 2) ? 24 : 32;
  const uint64_t totalSize = headerSize + compressed.size();
  std::vector<char> ccob = {'C', 'C', 'O', 'B'};
  AppendLe(ccob, version, 2);
  AppendLe(ccob, method, 2);
  if (version == 2) AppendLe(ccob, totalSize, 4);
  if (version >= 3) AppendLe(ccob, totalSize, 8);
  AppendLe(ccob, rawSize, version >= 3 ? 8 : 4);
  AppendLe(ccob, 0x1234567890abcdefull, 8);  // hash, not checked
  ccob.insert(ccob.end(), compressed.begin(), compressed.end());
  return ccob;
}

//! Stand-in for an uncompressed bundle: decompression doesn't look at the contents
std::vector<char> Payload() {
  std::string text = "__CLANG_OFFLOAD_BUNDLE__";
  for (int i = 0; i < 2000; ++i) text += "kernel_" + std::to_string(i) + ";";
  return std::vector<char>(text.begin(), text.end());
}

}  // namespace

TEST_CASE("Every header version and method decompresses", "[ccob]") {
  const std::vector<char> payload = Payload();
  for (const uint16_t method : {kZlib, kZstd}) {
    const std::vector<char> compressed = Compress(method, payload);
    if (compressed.empty()) {
      WARN("compression library for method " << method << " not installed");
      continue;
    }
    for (const uint16_t version : {1, 2, 3}) {
      INFO("version " << version << " method " << method);
      const std::vector<char> ccob = MakeCcob(version, method, compressed, payload.size());
      CHECK(IsCompressedBundle(ccob.data(), ccob.size()));
      CHECK(IsKnownCompressedBundle(ccob.data(), ccob.size()));
      std::vector<char> out;
      std::string error;
      REQUIRE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
      CHECK(out == payload);
      CHECK(CompressedBundleSize(ccob.data(), ccob.size()) == (version == 1 ? 0 : ccob.size()));
    }
  }
}

TEST_CASE("A version 1 bundle stops at the end of its own data", "[ccob]") {
  // v1 records no size. A registered fat binary gives no bound either, and what follows the
  // compressed data (padding, the next translation unit's CCOB) isn't part of it.
  const std::vector<char> payload = Payload();
  for (const uint16_t method : {kZlib, kZstd}) {
    const std::vector<char> compressed = Compress(method, payload);
    if (compressed.empty()) continue;
    INFO("method " << method);
    std::vector<char> image = MakeCcob(1, method, compressed, payload.size());
    image.resize(image.size() + 4096 - image.size() % 4096, '\0');
    const std::vector<char> next = MakeCcob(1, method, compressed, payload.size());
    image.insert(image.end(), next.begin(), next.end());

    std::vector<char> out;
    std::string error;
    REQUIRE(DecompressBundle(image.data(), image.size(), out, &error));
    CHECK(out == payload);
    out.clear();
    REQUIRE(DecompressBundle(image.data(), kUnknownSize, out, &error));
    CHECK(out == payload);
  }
}

TEST_CASE("Newer formats are left to comgr", "[ccob]") {
  const std::vector<char> payload = Payload();
  const std::vector<char> compressed = Compress(kZstd, payload);
  if (compressed.empty()) SKIP("zstd not installed");
  std::vector<char> out;
  std::string error;

  SECTION("unknown version") {
    std::vector<char> ccob = MakeCcob(3, kZstd, compressed, payload.size());
    ccob[4] = 4;
    CHECK(IsCompressedBundle(ccob.data(), ccob.size()));
    CHECK_FALSE(IsKnownCompressedBundle(ccob.data(), ccob.size()));
    CHECK(CompressedBundleSize(ccob.data(), ccob.size()) == 0);
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
    CHECK(error == "unknown compressed bundle version 4");
  }
  SECTION("unknown method") {
    std::vector<char> ccob = MakeCcob(3, 2, compressed, payload.size());
    CHECK_FALSE(IsKnownCompressedBundle(ccob.data(), ccob.size()));
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
    CHECK(error == "unknown compression method 2");
  }
}

TEST_CASE("A zlib size beyond deflate's ratio is rejected before allocating", "[ccob]") {
  const std::vector<char> payload = Payload();
  const std::vector<char> compressed = Compress(kZlib, payload);
  if (compressed.empty()) SKIP("zlib not installed");
  // 4 GB claimed from a few KB of deflate data (v2 sizes are 32-bit)
  const std::vector<char> bad = MakeCcob(2, kZlib, compressed, 0xffffffffu);
  std::vector<char> out;
  std::string error;
  bool ok = true;
  CHECK_NOTHROW(ok = DecompressBundle(bad.data(), bad.size(), out, &error));
  CHECK_FALSE(ok);
  CHECK(out.capacity() < (size_t{1} << 30));
}

TEST_CASE("Malformed compressed bundles are rejected", "[ccob]") {
  const std::vector<char> payload = Payload();
  const std::vector<char> compressed = Compress(kZstd, payload);
  if (compressed.empty()) SKIP("zstd not installed");
  std::vector<char> ccob = MakeCcob(3, kZstd, compressed, payload.size());
  std::vector<char> out;
  std::string error;

  SECTION("truncated header") {
    CHECK_FALSE(DecompressBundle(ccob.data(), 20, out, &error));
    CHECK(CompressedBundleSize(ccob.data(), 20) == 0);
  }
  SECTION("total size past the image") {
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size() - 1, out, &error));
    CHECK(CompressedBundleSize(ccob.data(), ccob.size() - 1) == 0);
  }
  SECTION("total size smaller than the header") {
    std::memset(ccob.data() + 8, 0, 8);
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
  }
  SECTION("wrong uncompressed size") {
    std::vector<char> bad = MakeCcob(3, kZstd, compressed, payload.size() + 1);
    CHECK_FALSE(DecompressBundle(bad.data(), bad.size(), out, &error));
    CHECK(error == "zstd frame size doesn't match the bundle header");
  }
  SECTION("uncompressed size the data can't produce is rejected before allocating") {
    // 2^62 bytes: resizing for it would throw out of the runtime
    std::vector<char> bad = MakeCcob(3, kZstd, compressed, uint64_t{1} << 62);
    bool ok = true;
    CHECK_NOTHROW(ok = DecompressBundle(bad.data(), bad.size(), out, &error));
    CHECK_FALSE(ok);
    CHECK(out.capacity() < (size_t{1} << 30));
  }
  SECTION("corrupt data") {
    ccob[32] = static_cast<char>(static_cast<unsigned char>(ccob[32]) ^ 0x5au);  // zstd magic
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
  }
  SECTION("not a CCOB") {
    ccob[0] = 'X';
    CHECK_FALSE(IsCompressedBundle(ccob.data(), ccob.size()));
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
  }
}
