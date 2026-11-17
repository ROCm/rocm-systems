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
#include <catch2/generators/catch_generators.hpp>

#include <cstring>
#include <filesystem>
#include <string>
#include <utility>
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

//! A well-formed zstd frame whose header claims @p contentSize: single segment, an 8-byte
//! content size, then one last raw block of 8 bytes
std::vector<char> ZstdFrameClaiming(uint64_t contentSize) {
  std::vector<char> frame = {'\x28', '\xb5', '\x2f', '\xfd', '\xe0'};
  AppendLe(frame, contentSize, 8);
  AppendLe(frame, (8u << 3) | 1u, 3);  // block size 8, raw, last
  frame.insert(frame.end(), 8, '\0');
  return frame;
}

//! Sets the v3 total size of @p ccob to @p totalSize
void SetTotalSizeV3(std::vector<char>& ccob, uint64_t totalSize) {
  for (size_t i = 0; i < 8; ++i) ccob[8 + i] = static_cast<char>(totalSize >> (8 * i));
}

}  // namespace

TEST_CASE("Every header version and method decompresses", "[ccob]") {
  const std::vector<char> payload = Payload();
  const uint16_t method = GENERATE(kZlib, kZstd);
  const std::vector<char> compressed = Compress(method, payload);
  if (compressed.empty()) SKIP("compression library for method " << method << " not installed");
  for (const uint16_t version : {1, 2, 3}) {
    INFO("version " << version << " method " << method);
    const std::vector<char> ccob = MakeCcob(version, method, compressed, payload.size());
    CHECK(IsCompressedBundle(ccob.data(), ccob.size()));
    CHECK(IsKnownCompressedBundle(ccob.data(), ccob.size()));
    CHECK(CanDecompressBundle(ccob.data(), ccob.size()));
    std::vector<char> out;
    std::string error;
    REQUIRE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
    CHECK(out == payload);
    CHECK(CompressedBundleSize(ccob.data(), ccob.size()) == (version == 1 ? 0 : ccob.size()));
  }
}

TEST_CASE("zlib output larger than the first buffer decompresses", "[ccob]") {
  // The zlib buffer starts at 16 MiB and doubles while uncompress fills it
  std::vector<char> payload(40u << 20);
  for (size_t i = 0; i < payload.size(); ++i) payload[i] = static_cast<char>(i * 7 / 4096);
  const std::vector<char> compressed = Compress(kZlib, payload);
  if (compressed.empty()) SKIP("zlib not installed");
  const std::vector<char> ccob = MakeCcob(3, kZlib, compressed, payload.size());
  const size_t bound = GENERATE(size_t{0}, kUnknownSize);
  INFO("bound " << (bound == kUnknownSize ? "unknown" : "known"));
  std::vector<char> out;
  std::string error;
  REQUIRE(DecompressBundle(ccob.data(), bound == 0 ? ccob.size() : bound, out, &error));
  CHECK(out == payload);
}

TEST_CASE("Sizes a pointer load can't bound are checked against the data", "[ccob]") {
  // Without a bound the compressed size is the header's claim: 1 TiB claimed for both headers
  // here, so the ratio checks against it pass
  std::vector<char> out;
  std::string error;
  bool ok = true;
  SECTION("zstd measures the frame") {
    if (Compress(kZstd, Payload()).empty()) SKIP("zstd not installed");
    constexpr uint64_t kClaimed = uint64_t{1} << 40;
    std::vector<char> ccob = MakeCcob(3, kZstd, ZstdFrameClaiming(kClaimed), kClaimed);
    SetTotalSizeV3(ccob, kClaimed * 2);
    CHECK_NOTHROW(ok = DecompressBundle(ccob.data(), kUnknownSize, out, &error));
    CHECK_FALSE(ok);
    CHECK(error.find("exceeds what its zstd data can hold") != std::string::npos);
  }
  SECTION("zlib grows the output only as the data decompresses") {
    const std::vector<char> compressed = Compress(kZlib, Payload());
    if (compressed.empty()) SKIP("zlib not installed");
    constexpr uint64_t kClaimed = uint64_t{1} << 39;
    std::vector<char> ccob = MakeCcob(3, kZlib, compressed, kClaimed);
    SetTotalSizeV3(ccob, kClaimed * 2);
    CHECK_NOTHROW(ok = DecompressBundle(ccob.data(), kUnknownSize, out, &error));
    CHECK_FALSE(ok);
    CHECK(error == "zlib decompression failed");
  }
  CHECK(out.capacity() < (size_t{1} << 30));
}

TEST_CASE("Compression libraries can be kept out of the current directory", "[ccob]") {
#if defined(_WIN32)
  SKIP("LoadLibrary's search order has the current directory in it on its own");
#else
  // A copy of the installed zstd under a name only the current directory has
  namespace fs = std::filesystem;
  void* symbol = FindSymbol(kZstdNames, "ZSTD_decompress");
  std::string path;
  size_t offset = 0;
  if (symbol == nullptr || !Os::FindFileNameFromAddress(symbol, &path, &offset)) {
    SKIP("zstd not installed");
  }
  struct TempCwd {
    fs::path saved = fs::current_path();
    fs::path dir =
        fs::temp_directory_path() / ("codeobject_cwd_" + std::to_string(Os::getProcessId()));
    TempCwd() {
      fs::create_directories(dir);
      fs::current_path(dir);
    }
    ~TempCwd() {
      std::error_code ec;
      fs::current_path(saved, ec);
      fs::remove_all(dir, ec);
    }
  } cwd;
  constexpr const char* kName = "codeobject_cwd_probe.so.1";
  fs::copy_file(path, cwd.dir / kName);

  CHECK(Os::loadLibrary(kName, /*searchCwd=*/false) == nullptr);
  void* handle = Os::loadLibrary(kName);
  CHECK(handle != nullptr);
  if (handle != nullptr) Os::unloadLibrary(handle);
#endif
}

TEST_CASE("A version 1 bundle stops at the end of its own data", "[ccob]") {
  // v1 records no size. A registered fat binary gives no bound either, and what follows the
  // compressed data (padding, the next translation unit's CCOB) isn't part of it.
  const std::vector<char> payload = Payload();
  const uint16_t method = GENERATE(kZlib, kZstd);
  const std::vector<char> compressed = Compress(method, payload);
  if (compressed.empty()) SKIP("compression library for method " << method << " not installed");
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
    CHECK_FALSE(CanDecompressBundle(ccob.data(), ccob.size()));
    CHECK(CompressedBundleSize(ccob.data(), ccob.size()) == 0);
    CHECK_FALSE(DecompressBundle(ccob.data(), ccob.size(), out, &error));
    CHECK(error == "unknown compressed bundle version 4");
  }
  SECTION("unknown method") {
    std::vector<char> ccob = MakeCcob(3, 2, compressed, payload.size());
    CHECK_FALSE(IsKnownCompressedBundle(ccob.data(), ccob.size()));
    CHECK_FALSE(CanDecompressBundle(ccob.data(), ccob.size()));
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
  SECTION("zstd's content size sentinels in both headers are rejected before allocating") {
    // Invalid data makes ZSTD_getFrameContentSize return ZSTD_CONTENTSIZE_ERROR; a frame
    // without a size returns ZSTD_CONTENTSIZE_UNKNOWN. Either matches a bundle header that
    // holds the same value.
    const std::vector<char> notZstd(compressed.size(), 'x');
    std::vector<char> frameNoSize = {'\x28', '\xb5', '\x2f', '\xfd', '\x00', '\x58'};
    frameNoSize.insert(frameNoSize.end(), 16, '\0');
    for (const auto& [data, rawSize] :
         {std::make_pair(notZstd, ~uint64_t(0) - 1), std::make_pair(frameNoSize, ~uint64_t(0))}) {
      std::vector<char> bad = MakeCcob(3, kZstd, data, rawSize);
      bool ok = true;
      CHECK_NOTHROW(ok = DecompressBundle(bad.data(), bad.size(), out, &error));
      CHECK_FALSE(ok);
      CHECK(error == "zstd frame has no valid content size");
      CHECK(out.capacity() < (size_t{1} << 30));
    }
  }
  SECTION("a zstd frame header claiming more than its blocks can hold is rejected") {
    // Both headers agree on 1 TiB, but an 8-byte block can't expand to that size
    constexpr uint64_t kClaimed = uint64_t{1} << 40;
    std::vector<char> bad = MakeCcob(3, kZstd, ZstdFrameClaiming(kClaimed), kClaimed);
    bool ok = true;
    CHECK_NOTHROW(ok = DecompressBundle(bad.data(), bad.size(), out, &error));
    CHECK_FALSE(ok);
    CHECK(error.find("exceeds what its zstd data can hold") != std::string::npos);
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
