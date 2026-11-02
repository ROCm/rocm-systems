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

std::recursive_mutex& cuid_operation_mutex();

// HMAC-SHA-256 over the in-tree SHA-256 (see sha256.h), keyed with a
// caller-supplied key.
class cuid_hmac {
 private:
  uint8_t* key;
  size_t key_len;
  bool valid;
  // Guards key, key_len and valid against concurrent readers
  // (generate_hmac_sha256) and writers (set_hmac_key).
  mutable std::mutex key_mutex_;

 public:
  cuid_hmac();
  cuid_hmac(uint8_t key_data[key_length]);
  // Key of an explicit length; used by tests.
  cuid_hmac(const char* key_data, size_t len);
  ~cuid_hmac();
  bool is_valid() const {
    std::lock_guard<std::mutex> lock(key_mutex_);
    return valid;
  }

  amdcuid_status_t generate_hmac_sha256(const uint8_t* data, size_t data_len, uint8_t* out_hash,
                                        size_t* out_len);
  amdcuid_status_t set_hmac_algorithm(const char* digest_name);

  // Replace the in-memory key. Used by tests.
  amdcuid_status_t set_hmac_key(const uint8_t key_data[key_length]);

  amdcuid_status_t generate_key(uint8_t key[key_length]);
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
