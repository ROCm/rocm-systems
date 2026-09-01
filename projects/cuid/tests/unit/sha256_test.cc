/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

// Pins the CUID derivation to FIPS 180-4 / RFC 4231. A regression here changes
// every CUID ever issued.

#include "unit/sha256_test.h"

#include <gtest/gtest.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/hmac.h"
#include "src/sha256.h"

namespace {

std::string to_hex(const uint8_t* data, size_t len) {
  static const char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(len * 2);
  for (size_t i = 0; i < len; ++i) {
    out.push_back(kDigits[data[i] >> 4]);
    out.push_back(kDigits[data[i] & 0x0F]);
  }
  return out;
}

std::string sha256_hex(const std::string& msg) {
  uint8_t digest[rocm::sha2::SHA256_DIGEST_SIZE];
  rocm::sha2::sha256_digest(reinterpret_cast<const uint8_t*>(msg.data()), msg.size(), digest);
  return to_hex(digest, sizeof(digest));
}

std::string hmac_hex(const std::vector<uint8_t>& key, const std::vector<uint8_t>& msg) {
  uint8_t mac[rocm::sha2::SHA256_DIGEST_SIZE];
  rocm::sha2::hmac_sha256(key.data(), key.size(), msg.data(), msg.size(), mac);
  return to_hex(mac, sizeof(mac));
}

std::vector<uint8_t> repeated(uint8_t byte, size_t count) {
  return std::vector<uint8_t>(count, byte);
}

std::vector<uint8_t> bytes_of(const char* s) {
  return std::vector<uint8_t>(reinterpret_cast<const uint8_t*>(s),
                              reinterpret_cast<const uint8_t*>(s) + std::strlen(s));
}

}  // namespace

// =============================================================================
// SHA-256 (FIPS 180-4)
// =============================================================================

TestSha256Kat::TestSha256Kat() {
  SetTitle("SHA-256 Known-Answer Vectors");
  SetDescription(
      "Verify the in-tree SHA-256 reproduces the FIPS 180-4 published digests, "
      "including the multi-block and length-encoding edge cases.");
}

void TestSha256Kat::SetUp() {}

void TestSha256Kat::Run() {
  // FIPS 180-4 published examples.
  EXPECT_EQ(sha256_hex(""), "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  EXPECT_EQ(sha256_hex("abc"), "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  EXPECT_EQ(sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq"),
            "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
  EXPECT_EQ(sha256_hex("abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmn"
                       "hijklmnoijklmnopjklmnopqklmnopqrlmnopqrsmnopqrstnopqrstu"),
            "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1");

  // Message lengths that straddle the padding boundary: a 55-byte message is
  // the largest that still fits its length field in the same block, 56 bytes
  // forces an extra block, 64 bytes is exactly one block.
  EXPECT_EQ(sha256_hex(std::string(55, 'a')),
            "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318");
  EXPECT_EQ(sha256_hex(std::string(56, 'a')),
            "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
  EXPECT_EQ(sha256_hex(std::string(64, 'a')),
            "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb");

  // One million 'a' -- the classic long-message vector. Fed in 1000-byte
  // chunks, which also exercises update() across arbitrary boundaries.
  {
    rocm::sha2::sha256 hasher;
    const std::string chunk(1000, 'a');
    for (int i = 0; i < 1000; ++i) hasher.update(chunk);
    const auto digest = hasher.digest();
    EXPECT_EQ(to_hex(digest.data(), digest.size()),
              "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  }

  // Streaming byte-at-a-time must equal the one-shot digest.
  {
    const std::string msg = "The quick brown fox jumps over the lazy dog";
    rocm::sha2::sha256 hasher;
    for (char c : msg) hasher.update(reinterpret_cast<const uint8_t*>(&c), 1);
    const auto streamed = hasher.digest();
    EXPECT_EQ(to_hex(streamed.data(), streamed.size()), sha256_hex(msg));
    EXPECT_EQ(sha256_hex(msg), "d7a8fbb307d7809469ca9abcb0082e4f8d5651e46d3cdb762d02d0bf37c9e592");
  }

  // digest() must be idempotent and agree with hexdigest().
  {
    rocm::sha2::sha256 hasher(std::string("abc"));
    const auto first = hasher.digest();
    EXPECT_EQ(first, hasher.digest());  // idempotent
    EXPECT_EQ(hasher.hexdigest(), to_hex(first.data(), first.size()));
  }

  // The public wrapper the rest of the library calls.
  {
    uint8_t out[32];
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS,
              sha256_unkeyed(reinterpret_cast<const uint8_t*>("abc"), 3, out));
    EXPECT_EQ(to_hex(out, sizeof(out)),
              "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    EXPECT_EQ(AMDCUID_STATUS_INVALID_ARGUMENT, sha256_unkeyed(nullptr, 4, out));
  }

  IF_VERB(1) { printf("  SHA-256 vectors verified\n"); }
}

void TestSha256Kat::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestSha256Kat::DisplayResults() const { TestBase::DisplayResults(); }
void TestSha256Kat::Close() {}

// =============================================================================
// HMAC-SHA-256 (RFC 4231)
// =============================================================================

TestHmacSha256Kat::TestHmacSha256Kat() {
  SetTitle("HMAC-SHA-256 Known-Answer Vectors");
  SetDescription(
      "Verify HMAC-SHA-256 reproduces the RFC 4231 test cases, including the "
      "short-key padding and long-key hash-down paths, and that cuid_hmac "
      "produces the same MAC through the library API.");
}

void TestHmacSha256Kat::SetUp() {}

void TestHmacSha256Kat::Run() {
  // RFC 4231 section 4.2 -- 20-byte key.
  EXPECT_EQ(hmac_hex(repeated(0x0b, 20), bytes_of("Hi There")),
            "b0344c61d8db38535ca8afceaf0bf12b881dc200c9833da726e9376c2e32cff7");

  // 4.3 -- 4-byte key (heavily zero-padded).
  EXPECT_EQ(hmac_hex(bytes_of("Jefe"), bytes_of("what do ya want for nothing?")),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843");

  // 4.4 -- 20-byte key, 50-byte message.
  EXPECT_EQ(hmac_hex(repeated(0xaa, 20), repeated(0xdd, 50)),
            "773ea91e36800e46854db8ebd09181a72959098b3ef8c122d9635514ced565fe");

  // 4.5 -- 25-byte key 0x01..0x19, 50-byte message.
  {
    std::vector<uint8_t> key(25);
    for (size_t i = 0; i < key.size(); ++i) key[i] = static_cast<uint8_t>(i + 1);
    EXPECT_EQ(hmac_hex(key, repeated(0xcd, 50)),
              "82558a389a443c0ea4cc819899f2083a85f0faa3e578f8077a2e3ff46729665b");
  }

  // 4.7 -- 131-byte key, longer than the 64-byte block, so it is hashed down
  // first. This is the branch most hand-rolled HMACs get wrong.
  EXPECT_EQ(hmac_hex(repeated(0xaa, 131),
                     bytes_of("Test Using Larger Than Block-Size Key - Hash Key First")),
            "60e431591ee0b67f0d8a26aacbf5b77f8e0bc6213728c5140546040f0ee37f54");

  // 4.8 -- long key and long message together.
  EXPECT_EQ(hmac_hex(repeated(0xaa, 131),
                     bytes_of("This is a test using a larger than block-size key and a "
                              "larger than block-size data. The key needs to be hashed "
                              "before being used by the HMAC algorithm.")),
            "9b09ffa71b942fcb27635fbcd5b0e944bfdc63644f0713938a7f51535c3a35e2");

  // Exactly one block: neither padded nor hashed down.
  EXPECT_EQ(hmac_hex(repeated(0x0b, 64), bytes_of("Hi There")),
            "21cd586aeca0579d99a1c938127c92525a371f807bc5ba6eb78bc825bd4f2be3");

  // Empty key and empty message.
  EXPECT_EQ(hmac_hex({}, {}), "b613679a0814d9ec772f95d778c35fc5ff1697c493715653c6c712144292c5ad");

  // The same MAC, produced through the class the library actually uses. The
  // 32-byte key path is the one CUID derivation takes.
  {
    uint8_t key[key_length];
    std::memset(key, 0x0b, sizeof(key));
    cuid_hmac hmac(key);
    EXPECT_TRUE(hmac.is_valid());
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.set_hmac_algorithm("SHA256"));

    uint8_t mac[hash_length];
    size_t mac_len = 0;
    EXPECT_EQ(
        AMDCUID_STATUS_SUCCESS,
        hmac.generate_hmac_sha256(reinterpret_cast<const uint8_t*>("Hi There"), 8, mac, &mac_len));
    EXPECT_EQ(mac_len, sizeof(mac));
    EXPECT_EQ(to_hex(mac, sizeof(mac)), hmac_hex(repeated(0x0b, key_length), bytes_of("Hi There")));

    // Only SHA-256 is offered; a wider digest would overrun the caller's
    // 32-byte output buffer, so it must be refused rather than accepted.
    EXPECT_EQ(AMDCUID_STATUS_HMAC_ERROR, hmac.set_hmac_algorithm("SHA512"));
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.set_hmac_algorithm("SHA-256"));
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.set_hmac_algorithm(nullptr));
  }

  // set_hmac_key is in-memory only: it must not create or touch a key file.
  // store_key is what persists, atomically, at mode 0600.
  {
    const std::string dir = "/tmp/amdcuid_store_key_test";
    const std::string path = dir + "/hmac_key.bin";
    ::mkdir(dir.c_str(), 0755);
    ::unlink(path.c_str());
    ::setenv("AMDCUID_HMAC_KEY_PATH", path.c_str(), 1);

    {
      cuid_hmac hmac;  // no key file yet
      // With nothing provisioned the object is usable, keyed with the public
      // default seed, and says so.
      EXPECT_TRUE(hmac.is_valid());
      EXPECT_TRUE(hmac.is_using_default_key());
      EXPECT_EQ(hmac.get_key_file_path(), path);

      // The default seed must match the kernel's CUID_DEFAULT_SEED byte for
      // byte, or an unprovisioned machine derives one secondary CUID from
      // sysfs and a different one from this library.
      {
        static const char kKernelDefaultSeed[] = "AMD-CUID-DEFAULT-SEED-v1";
        uint8_t via_lib[hash_length];
        size_t via_lib_len = 0;
        const uint8_t msg[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
        EXPECT_EQ(AMDCUID_STATUS_SUCCESS,
                  hmac.generate_hmac_sha256(msg, sizeof(msg), via_lib, &via_lib_len));

        uint8_t expected[rocm::sha2::SHA256_DIGEST_SIZE];
        rocm::sha2::hmac_sha256(reinterpret_cast<const uint8_t*>(kKernelDefaultSeed),
                                sizeof(kKernelDefaultSeed) - 1, msg, sizeof(msg), expected);
        EXPECT_EQ(to_hex(via_lib, sizeof(via_lib)), to_hex(expected, sizeof(expected)));
      }

      uint8_t k[key_length];
      std::memset(k, 0x5a, sizeof(k));
      EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.set_hmac_key(k));
      EXPECT_TRUE(hmac.is_valid());
      EXPECT_FALSE(hmac.is_using_default_key());

      // In-memory only: still nothing on disk.
      struct stat st;
      EXPECT_NE(0, ::stat(path.c_str(), &st));

      EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.store_key(k));
      ASSERT_EQ(0, ::stat(path.c_str(), &st));
      EXPECT_EQ(static_cast<off_t>(key_length), st.st_size);
      EXPECT_EQ(0600, st.st_mode & 07777);

      // No temporary file left behind.
      struct stat tst;
      EXPECT_NE(0, ::stat((path + ".new").c_str(), &tst));

      // Overwriting an existing key must succeed (the old code unlinked first).
      std::memset(k, 0xa5, sizeof(k));
      EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.store_key(k));
      ASSERT_EQ(0, ::stat(path.c_str(), &st));
      EXPECT_EQ(static_cast<off_t>(key_length), st.st_size);
      EXPECT_EQ(0600, st.st_mode & 07777);
    }

    // A fresh instance must load exactly what was stored and derive the same
    // MAC as an instance keyed directly.
    {
      uint8_t k[key_length];
      std::memset(k, 0xa5, sizeof(k));
      cuid_hmac from_file;
      EXPECT_TRUE(from_file.is_valid());
      EXPECT_FALSE(from_file.is_using_default_key());
      cuid_hmac from_mem(k);

      uint8_t a[hash_length], c[hash_length];
      size_t la = 0, lc = 0;
      const uint8_t msg[4] = {1, 2, 3, 4};
      EXPECT_EQ(AMDCUID_STATUS_SUCCESS, from_file.generate_hmac_sha256(msg, sizeof(msg), a, &la));
      EXPECT_EQ(AMDCUID_STATUS_SUCCESS, from_mem.generate_hmac_sha256(msg, sizeof(msg), c, &lc));
      EXPECT_EQ(0, std::memcmp(a, c, sizeof(a)));
    }

    // A wrong-sized key file must be rejected rather than half-loaded.
    {
      std::ofstream short_key(path, std::ios::binary | std::ios::trunc);
      short_key.write("\x01\x02\x03", 3);
      short_key.close();
      // Corruption is not absence: a wrong-sized key file must be refused
      // rather than silently falling back to the default, which would change
      // every CUID on the machine.
      cuid_hmac bad;
      EXPECT_FALSE(bad.is_valid());
      EXPECT_FALSE(bad.is_using_default_key());
      uint8_t out[hash_length];
      size_t n = 0;
      const uint8_t msg[1] = {0};
      EXPECT_EQ(AMDCUID_STATUS_KEY_ERROR, bad.generate_hmac_sha256(msg, 1, out, &n));
    }

    ::unlink(path.c_str());
    ::rmdir(dir.c_str());
    ::unsetenv("AMDCUID_HMAC_KEY_PATH");
  }

  // Null arguments must be rejected, not memcpy'd.
  {
    cuid_hmac hmac;
    EXPECT_EQ(AMDCUID_STATUS_INVALID_ARGUMENT, hmac.set_hmac_key(nullptr));
    EXPECT_EQ(AMDCUID_STATUS_INVALID_ARGUMENT, hmac.store_key(nullptr));
  }

  // generate_key must return distinct, non-trivial key material.
  {
    cuid_hmac hmac;
    uint8_t a[key_length] = {};
    uint8_t b[key_length] = {};
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.generate_key(a));
    EXPECT_EQ(AMDCUID_STATUS_SUCCESS, hmac.generate_key(b));
    EXPECT_NE(0, std::memcmp(a, b, sizeof(a)));

    uint8_t zeros[key_length] = {};
    EXPECT_NE(0, std::memcmp(a, zeros, sizeof(a)));
    EXPECT_EQ(AMDCUID_STATUS_INVALID_ARGUMENT, hmac.generate_key(nullptr));
  }

  IF_VERB(1) { printf("  HMAC-SHA-256 vectors verified\n"); }
}

void TestHmacSha256Kat::DisplayTestInfo() { TestBase::DisplayTestInfo(); }
void TestHmacSha256Kat::DisplayResults() const { TestBase::DisplayResults(); }
void TestHmacSha256Kat::Close() {}
