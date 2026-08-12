// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef HMAC_H
#define HMAC_H

#include "include/amd_cuid.h"
#include <cstddef>
#include <cstdint>
#include <string>

#define key_length 32
#define hash_length 32

// cuid_hmac wraps HMAC-SHA-256 over the in-tree SHA-256 (see sha256.h). There
// is one implementation on every platform; the pimpl is retained only to keep
// the class layout stable for existing callers.
class cuid_hmac {
private:
  struct Impl; // defined in hmac.cc
  Impl *impl_;
  uint8_t *key;
  size_t key_len;
  bool valid;
  std::string key_file_path;

public:
  cuid_hmac();
  cuid_hmac(uint8_t key_data[key_length]);
  ~cuid_hmac();
  bool is_valid() const { return valid; }

  amdcuid_status_t generate_hmac_sha256(const uint8_t *data, size_t data_len,
                                        uint8_t *out_hash, size_t *out_len);
  amdcuid_status_t set_hmac_algorithm(const char *digest_name);
  amdcuid_status_t set_hmac_key(const uint8_t key_data[key_length]);
  amdcuid_status_t generate_key(uint8_t key[key_length]);
  std::string get_key_file_path() const { return key_file_path; }
};

// Unkeyed SHA-256 digest of data into a 32-byte output buffer.
amdcuid_status_t sha256_unkeyed(const uint8_t *data, size_t data_len,
                                uint8_t out[32]);

#endif // HMAC_H
