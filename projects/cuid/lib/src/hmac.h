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

#ifndef HMAC_H
#define HMAC_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "include/amd_cuid.h"

#define key_length 32
#define hash_length 32

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
  std::string key_file_path;

 public:
  cuid_hmac();
  cuid_hmac(uint8_t key_data[key_length]);
  ~cuid_hmac();
  bool is_valid() const { return valid; }

  // True when no key file was found and the public default seed is in use, so
  // a caller can distinguish a provisioned identity from an unprovisioned one.
  bool is_using_default_key() const { return using_default_key; }

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

  amdcuid_status_t generate_key(uint8_t key[key_length]);
  std::string get_key_file_path() const { return key_file_path; }
};

// Unkeyed SHA-256 digest of data into a 32-byte output buffer.
amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]);

#endif  // HMAC_H
