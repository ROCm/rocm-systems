/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#include "device/codeobject/compressed_bundle.hpp"

#include "os/os.hpp"

#include <cstring>
#include <mutex>

namespace amd::codeobject {

namespace {

constexpr char kCompressedMagic[] = "CCOB";
constexpr size_t kCompressedMagicSize = sizeof(kCompressedMagic) - 1;

// Compression method stored in the header (llvm::compression::Format)
constexpr uint16_t kMethodZlib = 0;
constexpr uint16_t kMethodZstd = 1;

// Header sizes: magic (4), version (2), method (2), then per version
//   v1: uncompressed size u32, hash u64
//   v2: file size u32, uncompressed size u32, hash u64
//   v3: file size u64, uncompressed size u64, hash u64
constexpr size_t kCommonHeaderSize = 8;
constexpr size_t kHeaderSizeV1 = kCommonHeaderSize + 4 + 8;
constexpr size_t kHeaderSizeV2 = kCommonHeaderSize + 4 + 4 + 8;
constexpr size_t kHeaderSizeV3 = kCommonHeaderSize + 8 + 8 + 8;

uint64_t LoadLe(const char* p, size_t bytes) {
  uint64_t v = 0;
  for (size_t i = 0; i < bytes; ++i) {
    v |= static_cast<uint64_t>(static_cast<uint8_t>(p[i])) << (8 * i);
  }
  return v;
}

// Library names tried in order; the first that loads wins
#if defined(_WIN32)
constexpr const char* kZstdNames[] = {"zstd.dll", "libzstd.dll"};
constexpr const char* kZlibNames[] = {"zlib1.dll", "zlib.dll"};
#else
constexpr const char* kZstdNames[] = {"librocm_sysdeps_zstd.so.1", "libzstd.so.1"};
constexpr const char* kZlibNames[] = {"librocm_sysdeps_z.so.1", "libz.so.1"};
#endif

//! zstd entry points, loaded with Os::loadLibrary on first use
struct Zstd {
  using DecompressFn = size_t (*)(void*, size_t, const void*, size_t);
  using IsErrorFn = unsigned (*)(size_t);
  using FrameSizeFn = size_t (*)(const void*, size_t);
  using ContentSizeFn = unsigned long long (*)(const void*, size_t);

  DecompressFn decompress = nullptr;    //!< ZSTD_decompress
  IsErrorFn isError = nullptr;          //!< ZSTD_isError
  FrameSizeFn frameSize = nullptr;      //!< ZSTD_findFrameCompressedSize
  ContentSizeFn contentSize = nullptr;  //!< ZSTD_getFrameContentSize

  static const Zstd& Get() {
    static Zstd zstd;
    static std::once_flag once;
    std::call_once(once, [] {
      for (const char* name : kZstdNames) {
        void* handle = Os::loadLibrary(name);
        if (handle == nullptr) continue;
        zstd.decompress = reinterpret_cast<DecompressFn>(Os::getSymbol(handle, "ZSTD_decompress"));
        zstd.isError = reinterpret_cast<IsErrorFn>(Os::getSymbol(handle, "ZSTD_isError"));
        zstd.frameSize = reinterpret_cast<FrameSizeFn>(
            Os::getSymbol(handle, "ZSTD_findFrameCompressedSize"));
        zstd.contentSize = reinterpret_cast<ContentSizeFn>(
            Os::getSymbol(handle, "ZSTD_getFrameContentSize"));
        if (zstd.decompress != nullptr && zstd.isError != nullptr && zstd.frameSize != nullptr &&
            zstd.contentSize != nullptr) {
          return;
        }
        zstd = Zstd{};
        Os::unloadLibrary(handle);
      }
    });
    return zstd;
  }
};

//! zlib entry points, loaded with Os::loadLibrary on first use
struct Zlib {
  using UncompressFn = int (*)(unsigned char*, unsigned long*, const unsigned char*,
                               unsigned long);

  UncompressFn uncompress = nullptr;  //!< uncompress

  static const Zlib& Get() {
    static Zlib zlib;
    static std::once_flag once;
    std::call_once(once, [] {
      for (const char* name : kZlibNames) {
        void* handle = Os::loadLibrary(name);
        if (handle == nullptr) continue;
        zlib.uncompress = reinterpret_cast<UncompressFn>(Os::getSymbol(handle, "uncompress"));
        if (zlib.uncompress != nullptr) return;
        Os::unloadLibrary(handle);
      }
    });
    return zlib;
  }
};

}  // namespace

// ================================================================================================
bool IsCompressedBundle(const void* image, size_t imageBound) {
  return imageBound >= kCompressedMagicSize &&
         std::memcmp(image, kCompressedMagic, kCompressedMagicSize) == 0;
}

// ================================================================================================
bool IsKnownCompressedBundle(const void* image, size_t imageBound) {
  if (!IsCompressedBundle(image, imageBound) || imageBound < kCommonHeaderSize) {
    return false;
  }
  const char* p = static_cast<const char*>(image);
  const uint64_t version = LoadLe(p + 4, 2);
  const uint64_t method = LoadLe(p + 6, 2);
  return version >= 1 && version <= 3 && (method == kMethodZlib || method == kMethodZstd);
}

// ================================================================================================
uint64_t CompressedBundleSize(const void* image, size_t imageBound) {
  const char* p = static_cast<const char*>(image);
  if (!IsCompressedBundle(image, imageBound) || imageBound < kCommonHeaderSize) {
    return 0;
  }
  uint64_t fileSize = 0;
  switch (LoadLe(p + 4, 2)) {
    case 2:
      if (imageBound < kHeaderSizeV2) return 0;
      fileSize = LoadLe(p + 8, 4);
      return (fileSize >= kHeaderSizeV2 && fileSize <= imageBound) ? fileSize : 0;
    case 3:
      if (imageBound < kHeaderSizeV3) return 0;
      fileSize = LoadLe(p + 8, 8);
      return (fileSize >= kHeaderSizeV3 && fileSize <= imageBound) ? fileSize : 0;
    default:
      return 0;  // version 1 doesn't record the size; unknown versions can't be read
  }
}

// ================================================================================================
bool DecompressBundle(const void* image, size_t imageBound, std::vector<char>& out,
                      std::string* error) {
  const char* p = static_cast<const char*>(image);
  if (!IsCompressedBundle(image, imageBound) || imageBound < kCommonHeaderSize) {
    *error = "not a compressed offload bundle";
    return false;
  }
  const uint16_t version = static_cast<uint16_t>(LoadLe(p + 4, 2));
  const uint16_t method = static_cast<uint16_t>(LoadLe(p + 6, 2));

  size_t headerSize = 0;
  uint64_t fileSize = 0;  // 0 for v1, which doesn't record it
  uint64_t uncompressedSize = 0;
  switch (version) {
    case 1:
      headerSize = kHeaderSizeV1;
      if (imageBound < headerSize) break;
      uncompressedSize = LoadLe(p + 8, 4);
      break;
    case 2:
      headerSize = kHeaderSizeV2;
      if (imageBound < headerSize) break;
      fileSize = LoadLe(p + 8, 4);
      uncompressedSize = LoadLe(p + 12, 4);
      break;
    case 3:
      headerSize = kHeaderSizeV3;
      if (imageBound < headerSize) break;
      fileSize = LoadLe(p + 8, 8);
      uncompressedSize = LoadLe(p + 16, 8);
      break;
    default:
      *error = "unknown compressed bundle version " + std::to_string(version);
      return false;
  }
  if (imageBound < headerSize) {
    *error = "compressed bundle header is truncated";
    return false;
  }

  // v2/v3 record the total size; v1 runs to the end of the readable range
  uint64_t compressedSize = imageBound - headerSize;
  if (version != 1) {
    if (fileSize < headerSize || fileSize > imageBound) {
      *error = "compressed bundle size " + std::to_string(fileSize) + " is out of bounds";
      return false;
    }
    compressedSize = fileSize - headerSize;
  }
  const char* compressed = p + headerSize;

  // The uncompressed size comes from the header; it is checked against the compressed data
  // before anything is allocated for it
  if (method == kMethodZstd) {
    const Zstd& zstd = Zstd::Get();
    if (zstd.decompress == nullptr) {
      *error = "zstd library not found";
      return false;
    }
    if (version == 1) {
      // v1 records no size, and what follows the frame (padding, the next translation unit's
      // bundle) isn't zstd data: decompress the one frame only
      const size_t frameSize = zstd.frameSize(compressed, static_cast<size_t>(compressedSize));
      if (zstd.isError(frameSize)) {
        *error = "zstd frame is malformed";
        return false;
      }
      compressedSize = frameSize;
    }
    // LLVM's bundler records the content size in the zstd frame header as well
    if (zstd.contentSize(compressed, static_cast<size_t>(compressedSize)) != uncompressedSize) {
      *error = "zstd frame size doesn't match the bundle header";
      return false;
    }
    out.resize(static_cast<size_t>(uncompressedSize));
    const size_t result = zstd.decompress(out.data(), out.size(), compressed, compressedSize);
    if (zstd.isError(result) || result != out.size()) {
      *error = "zstd decompression failed";
      return false;
    }
  } else if (method == kMethodZlib) {
    const Zlib& zlib = Zlib::Get();
    if (zlib.uncompress == nullptr) {
      *error = "zlib library not found";
      return false;
    }
    // Deflate expands at most 1032:1
    constexpr uint64_t kMaxDeflateRatio = 1032;
    if (compressedSize < ~uint64_t(0) / kMaxDeflateRatio &&
        uncompressedSize > compressedSize * kMaxDeflateRatio) {
      *error = "compressed bundle size " + std::to_string(uncompressedSize) +
               " exceeds what its zlib data can hold";
      return false;
    }
    out.resize(static_cast<size_t>(uncompressedSize));
    unsigned long outSize = static_cast<unsigned long>(out.size());
    if (zlib.uncompress(reinterpret_cast<unsigned char*>(out.data()), &outSize,
                        reinterpret_cast<const unsigned char*>(compressed),
                        static_cast<unsigned long>(compressedSize)) != 0 ||
        outSize != out.size()) {
      *error = "zlib decompression failed";
      return false;
    }
  } else {
    *error = "unknown compression method " + std::to_string(method);
    return false;
  }
  return true;
}

}  // namespace amd::codeobject
