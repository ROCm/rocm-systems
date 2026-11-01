/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace amd::codeobject {

//! @returns True if @p image starts with the compressed offload bundle magic "CCOB".
bool IsCompressedBundle(const void* image, size_t imageBound);

/**
 * @returns True if @p image is a compressed bundle whose header version (1-3) and compression
 *          method (zlib, zstd) DecompressBundle reads. A newer format is left to comgr, which
 *          follows LLVM's bundler.
 */
bool IsKnownCompressedBundle(const void* image, size_t imageBound);

/**
 * @returns The size of the compressed bundle at @p image as recorded in its header (versions 2
 *          and 3), or 0 if the header is malformed, the header or the recorded size exceeds
 *          @p imageBound, or version 1, which doesn't record it.
 */
uint64_t CompressedBundleSize(const void* image, size_t imageBound);

/**
 * Decompresses the first compressed offload bundle (CCOB) at @p image into @p out, which then
 * holds an uncompressed clang offload bundle.
 *
 * Header versions 1, 2 and 3 are read; methods are zlib and zstd, each loaded with
 * Os::loadLibrary on first use. The stored hash is not checked, as in clang's OffloadBundler.
 * The uncompressed size in the header is checked against the compressed data (the zstd frame's
 * content size, deflate's maximum ratio) before it is allocated.
 *
 * @param image       Start of the compressed bundle.
 * @param imageBound  Readable bytes at @p image.
 * @param out         Decompressed bundle.
 * @param error       Reason on failure.
 * @returns False for a malformed header, an unknown version or method, a missing compression
 *          library, or a decompression error.
 */
bool DecompressBundle(const void* image, size_t imageBound, std::vector<char>& out,
                      std::string* error);

}  // namespace amd::codeobject
