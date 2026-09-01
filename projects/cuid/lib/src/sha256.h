// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

//
// SHA-256 (FIPS 180-4) and HMAC-SHA-256 (FIPS 198-1 / RFC 2104).
//
// This is the implementation already carried in this repository at
// projects/rocprofiler-sdk/source/lib/common/sha256.{hpp,cpp}, copied here so
// that CUID has no dependency beyond the C++17 standard library. The class API
// is kept source-compatible with that original -- same members, same method
// names, same private helpers -- so the two copies can be collapsed into one
// shared component later without touching either caller. The namespace is
// deliberately neutral rather than cuid-specific for the same reason: the move
// should be a file move, not an API change.
//
// The complete set of deviations from the rocprofiler-sdk original, for
// whoever reconciles the two:
//
//   1. The rocprofiler-internal includes (defines.hpp / mpl.hpp / logging.hpp)
//      and the two ROCP_CI_LOG_IF diagnostics in update() are dropped. Those
//      diagnostics only logged; the `if(m_finalized) return;` guards they sat
//      next to are retained.
//   2. reset() additionally clears m_data and m_finalized. In the original,
//      reset() is private and only ever runs from the constructor, so the
//      omission is unobservable there; clearing them makes the method correct
//      if it is ever exposed for object reuse.
//   3. transform() casts each byte to uint32_t before shifting. The original
//      relies on integer promotion to int; that is well defined for these
//      values, so this is legibility, not a fix.
//   4. transform() scrubs its message schedule before returning.
//   5. digest(), sha256_digest(), hmac_sha256() and secure_zero() are new.
//
// Deviations 1-4 are behaviour-preserving: the two implementations were run
// side by side over the padding-boundary lengths (0, 55, 56, 63, 64, 65, 127,
// 128, 129) plus randomised lengths and randomised chunked updates, and agree
// bit for bit.
//
// Not constant-time and not side-channel hardened. It exists for identifier
// derivation (CUID), not for bulk secret processing.
//

#ifndef ROCM_SHA2_SHA256_H
#define ROCM_SHA2_SHA256_H

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>

namespace rocm {
namespace sha2 {

/// Size of a SHA-256 digest, in bytes.
constexpr size_t SHA256_DIGEST_SIZE = 32;
/// SHA-256 compression block size, in bytes. Also the HMAC key block size.
constexpr size_t SHA256_BLOCK_SIZE = 64;

/// Overwrite a buffer in a way the optimiser is not permitted to elide.
void secure_zero(void* p, size_t n);

// --- SHA-256 Implementation ---
class sha256 {
 public:
  sha256();
  explicit sha256(const std::string& data);

  void update(const uint8_t* data, size_t len);
  void update(const std::string& data);

  void finalize();
  std::string hexdigest();
  std::array<uint32_t, 8> rawdigest();

  /// Serialise the digest big-endian: the FIPS 180-4 byte order, and the one
  /// every other SHA-256 implementation emits. Finalises if not already done.
  std::array<uint8_t, SHA256_DIGEST_SIZE> digest();

 private:
  bool m_finalized = false;
  std::array<uint8_t, 64> m_data = {};
  std::array<uint32_t, 8> m_state = {};
  uint32_t m_datalen = 0;
  uint64_t m_bitlen = 0;

  static constexpr std::array<uint32_t, 64> m_k = {
      0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4,
      0xab1c5ed5, 0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe,
      0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
      0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7,
      0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967, 0x27b70a85, 0x2e1b2138, 0x4d2c6dfc,
      0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b,
      0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116,
      0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
      0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7,
      0xc67178f2};

  static uint32_t rotr(uint32_t x, uint32_t n);
  static uint32_t ch(uint32_t x, uint32_t y, uint32_t z);
  static uint32_t maj(uint32_t x, uint32_t y, uint32_t z);
  static uint32_t sig0(uint32_t x);
  static uint32_t sig1(uint32_t x);
  static uint32_t theta0(uint32_t x);
  static uint32_t theta1(uint32_t x);

  void transform();
  void reset();
};

/// One-shot SHA-256 over a single buffer.
void sha256_digest(const uint8_t* data, size_t data_len, uint8_t out[SHA256_DIGEST_SIZE]);

/// HMAC-SHA-256 (FIPS 198-1 / RFC 2104). Any key length is accepted: keys
/// longer than the 64-byte block are hashed down first, shorter keys are
/// zero-padded, per the standard.
void hmac_sha256(const uint8_t* key, size_t key_len, const uint8_t* msg, size_t msg_len,
                 uint8_t out[SHA256_DIGEST_SIZE]);

}  // namespace sha2
}  // namespace rocm

#endif  // ROCM_SHA2_SHA256_H
