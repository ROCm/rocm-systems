// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef HMAC_H
#define HMAC_H

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <string>

#include "include/amd_cuid.h"

// Length of a provisioned secret, in bytes. Not a maximum: the specification
// defines a 256-bit shared secret, and any other size is rejected as corrupt.
#define key_length 32
#define hash_length 32

// Fixed public key for auxiliary ("temporary") CUIDs. Exactly these 20 ASCII
// bytes: no NUL terminator and no padding. Do not zero-extend it to key_length;
// HMAC-SHA-256 pads a short key to its own 64-byte block, so padding first
// yields a different key. It is not a secret: an auxiliary CUID is built from
// non-privileged data so out-of-band consumers can reproduce it.
constexpr char kTemporaryKey[] = "AMD-CUID-TEMP-KEY-v1";
constexpr size_t kTemporaryKeyLen = sizeof(kTemporaryKey) - 1;

// Used when no secret is provisioned. Byte-identical to CUID_DEFAULT_SEED in
// the kernel's amdgpu_cuid.c, so sysfs and this library agree on an
// unprovisioned machine. 24 bytes, no NUL, unpadded, for the same reason as
// above. Not a substitute for provisioning; callers can check
// is_using_default_key().
constexpr char kDefaultSeed[] = "AMD-CUID-DEFAULT-SEED-v1";
constexpr size_t kDefaultSeedLen = sizeof(kDefaultSeed) - 1;

// cuid_hmac wraps HMAC-SHA-256 over the in-tree SHA-256 (see sha256.h). There
// is one implementation on every platform; the pimpl is retained only to keep
// the class layout stable for existing callers.
class cuid_hmac {
 private:
  struct Impl;  // defined in hmac.cc
  void use_default_key();
  Impl* impl_;
  uint8_t* key;
  size_t key_len;
  bool valid;
  bool using_default_key;
  amdcuid_status_t key_store_status_;
  std::string key_file_path;
  // Guards key, key_len, valid, using_default_key and key_store_status_
  // against concurrent readers (generate_hmac_sha256, key_fingerprint,
  // key_store_status) and writers (set_hmac_key, store_key).
  mutable std::mutex key_mutex_;

 public:
  cuid_hmac();
  cuid_hmac(uint8_t key_data[key_length]);
  // Key of an explicit length, for the fixed public keys that are not 32 bytes.
  // Does not read or write the key file.
  cuid_hmac(const char* key_data, size_t len);
  ~cuid_hmac();
  bool is_valid() const {
    std::lock_guard<std::mutex> lock(key_mutex_);
    return valid;
  }

  // True when no key file was found and the public default seed is in use, so
  // a caller can distinguish a provisioned identity from an unprovisioned one.
  bool is_using_default_key() const {
    std::lock_guard<std::mutex> lock(key_mutex_);
    return using_default_key;
  }

  // What the key store had to say, as distinct from what key ended up in use:
  //
  //   SUCCESS           a provisioned key was loaded, or none exists and the
  //                     public default seed stands in. is_using_default_key()
  //                     tells the two apart.
  //   PERMISSION_DENIED a key file exists but this caller cannot open it, which
  //                     is where every unprivileged caller on a provisioned
  //                     node lands (the file is 0600). The default seed is used
  //                     so derivation keeps working, but the node IS
  //                     provisioned and must not be reported otherwise.
  //   KEY_ERROR         a key file exists and is not a key.
  amdcuid_status_t key_store_status() const {
    std::lock_guard<std::mutex> lock(key_mutex_);
    return key_store_status_;
  }

  amdcuid_status_t generate_hmac_sha256(const uint8_t* data, size_t data_len, uint8_t* out_hash,
                                        size_t* out_len);
  amdcuid_status_t set_hmac_algorithm(const char* digest_name);

  // Replace the in-memory key. Purely an in-memory operation: it does not read
  // or write the key file. Pair it with store_key() to change the key on disk.
  amdcuid_status_t set_hmac_key(const uint8_t key_data[key_length]);

  // Persist a key to key_file_path, replacing any existing one. The file is
  // created 0600 and swapped into place atomically, so a concurrent reader
  // sees either the whole old key or the whole new one, never a partial write
  // and never a moment where the key is readable by other users. Requires
  // privileges to write the config directory.
  amdcuid_status_t store_key(const uint8_t key_data[key_length]);

  // First 8 octets of the unkeyed SHA-256 of the key in use, so a caller can
  // confirm two nodes carry the same seed without the seed crossing an ABI
  // boundary. Truncated: it answers that question and nothing else.
  amdcuid_status_t key_fingerprint(uint8_t out[8]) const;

  amdcuid_status_t generate_key(uint8_t key[key_length]);
  std::string get_key_file_path() const { return key_file_path; }
};

// Unkeyed SHA-256 digest of data into a 32-byte output buffer.
//
// Namespaced for the same reason cuid::get_hash_from_raw is: this archive is
// linked into libamd_smi.so, where a name this generic at global scope invites
// a collision. Declared here rather than in cuid_util.h to keep hmac.h
// self-contained; the definition is in hmac.cc, beside the rocm::sha2 calls.
namespace CuidUtilities {
amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]);
}  // namespace CuidUtilities

#endif  // HMAC_H
