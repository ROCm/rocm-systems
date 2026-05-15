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

#include "include/amd_cuid.h"
#include <cstddef>
#include <string>

#define key_length 32
#define hash_length 32

// cuid_hmac uses a pimpl to hide the crypto-backend context entirely.
// The active backend is selected at compile time:
//   _WIN32                              -> Windows CNG  (BCrypt)
//   OpenSSL >= 3.0  (all other platforms) -> EVP_MAC API
//   OpenSSL <  3.0  (all other platforms) -> HMAC_CTX API
class cuid_hmac {
private:
  struct Impl; // defined in hmac.cc per-backend
  Impl *impl_;
  uint8_t *key;
  size_t key_len;
  bool valid;

public:
  cuid_hmac();
  ~cuid_hmac();
  bool is_valid() const { return valid; }

  amdcuid_status_t generate_hmac_sha256(const uint8_t *data, size_t data_len,
                                        uint8_t *out_hash, size_t *out_len);
  amdcuid_status_t set_hmac_key(const uint8_t key_data[key_length]);
  amdcuid_status_t generate_key(uint8_t key[key_length]);

  std::string key_file_path = "/opt/amdcuid/etc/hmac_key.bin";
};

#endif // HMAC_H
