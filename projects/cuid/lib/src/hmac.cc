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

// Backend selection (compile-time):
//   _WIN32              -> Windows CNG  (BCrypt)
//   all other platforms -> vendored cifra (SHA-256 only, see third_party/cifra)

#include "hmac.h"

#include <stdlib.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>

#ifndef AMDCUID_CONFIG_DIR
#error "AMDCUID_CONFIG_DIR must be defined via CMake"
#endif

// ─────────────────────────────────────────────────────────────────────────────
// Windows CNG backend (BCrypt)
// ─────────────────────────────────────────────────────────────────────────────
#if defined(_WIN32)

#define WIN32_LEAN_AND_MEAN
#include <bcrypt.h>
#include <windows.h>

struct cuid_hmac::Impl {
  BCRYPT_ALG_HANDLE hAlg;
  std::string digest_name;

  // Map a cuid_hmac digest name (see set_hmac_algorithm) to the BCrypt HMAC provider ID.
  static const wchar_t* to_bcrypt_alg(const char* name) {
    if (!name) return BCRYPT_SHA256_ALGORITHM;
    if (_stricmp(name, "SHA256") == 0 || _stricmp(name, "SHA-256") == 0)
      return BCRYPT_SHA256_ALGORITHM;
    if (_stricmp(name, "SHA1") == 0 || _stricmp(name, "SHA-1") == 0) return BCRYPT_SHA1_ALGORITHM;
    if (_stricmp(name, "SHA384") == 0 || _stricmp(name, "SHA-384") == 0)
      return BCRYPT_SHA384_ALGORITHM;
    if (_stricmp(name, "SHA512") == 0 || _stricmp(name, "SHA-512") == 0)
      return BCRYPT_SHA512_ALGORITHM;
    return nullptr;
  }
};

cuid_hmac::cuid_hmac() : impl_(nullptr), key(nullptr), key_len(key_length), valid(false) {
  const char* env_path = std::getenv("AMDCUID_HMAC_KEY_PATH");
  key_file_path = (env_path && env_path[0]) ? env_path : AMDCUID_CONFIG_DIR "/hmac_key.bin";
  impl_ = new Impl();
  impl_->digest_name = "SHA256";
  impl_->hAlg = nullptr;

  NTSTATUS status = BCryptOpenAlgorithmProvider(&impl_->hAlg, BCRYPT_SHA256_ALGORITHM, nullptr,
                                                BCRYPT_ALG_HANDLE_HMAC_FLAG);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "Error opening BCrypt HMAC algorithm provider" << std::endl;
    delete impl_;
    impl_ = nullptr;
    return;
  }

  std::ifstream key_file_stream(key_file_path, std::ios::binary);
  if (!key_file_stream.is_open()) {
    return;
  }
  key_file_stream.seekg(0, std::ios::end);
  key_len = static_cast<size_t>(key_file_stream.tellg());
  if (key_len == 0 || key_len != key_length) {  // sanity check on key length
    std::cerr << "Invalid key length in key file" << std::endl;
    key_file_stream.close();
    return;
  }
  key_file_stream.seekg(0, std::ios::beg);
  key = new uint8_t[key_len];
  key_file_stream.read(reinterpret_cast<char*>(key), key_len);
  key_file_stream.close();

  valid = true;
}

cuid_hmac::cuid_hmac(uint8_t key_data[key_length])
    : impl_(nullptr), key(nullptr), key_len(key_length), valid(false) {
  impl_ = new Impl();
  impl_->digest_name = "SHA256";
  impl_->hAlg = nullptr;

  NTSTATUS status = BCryptOpenAlgorithmProvider(&impl_->hAlg, BCRYPT_SHA256_ALGORITHM, nullptr,
                                                BCRYPT_ALG_HANDLE_HMAC_FLAG);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "Error opening BCrypt HMAC algorithm provider" << std::endl;
    delete impl_;
    impl_ = nullptr;
    return;
  }

  key = new uint8_t[key_length];
  std::memcpy(key, key_data, key_length);

  valid = true;
}

cuid_hmac::~cuid_hmac() {
  if (impl_) {
    if (impl_->hAlg) BCryptCloseAlgorithmProvider(impl_->hAlg, 0);
    delete impl_;
  }
  delete[] key;
}

amdcuid_status_t cuid_hmac::generate_hmac_sha256(const uint8_t* data, size_t data_len,
                                                 uint8_t* out_hash, size_t* out_len) {
  if (!impl_ || !impl_->hAlg) {
    std::cerr << "BCrypt backend not initialized" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  BCRYPT_HASH_HANDLE hHash = nullptr;
  NTSTATUS status = BCryptCreateHash(impl_->hAlg, &hHash, nullptr, 0, reinterpret_cast<PUCHAR>(key),
                                     static_cast<ULONG>(key_len), 0);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "BCryptCreateHash failed" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  status = BCryptHashData(hHash, const_cast<PUCHAR>(reinterpret_cast<const UCHAR*>(data)),
                          static_cast<ULONG>(data_len), 0);
  if (!BCRYPT_SUCCESS(status)) {
    BCryptDestroyHash(hHash);
    std::cerr << "BCryptHashData failed" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  DWORD hash_len = 0, result_size = 0;
  status = BCryptGetProperty(impl_->hAlg, BCRYPT_HASH_LENGTH, reinterpret_cast<PUCHAR>(&hash_len),
                             sizeof(hash_len), &result_size, 0);
  if (!BCRYPT_SUCCESS(status)) {
    BCryptDestroyHash(hHash);
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  status = BCryptFinishHash(hHash, reinterpret_cast<PUCHAR>(out_hash), hash_len, 0);
  BCryptDestroyHash(hHash);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "BCryptFinishHash failed" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  *out_len = hash_len;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_algorithm(const char* digest_name) {
  if (!impl_) return AMDCUID_STATUS_HMAC_ERROR;

  const char* name = digest_name ? digest_name : "SHA256";
  const wchar_t* alg = Impl::to_bcrypt_alg(name);
  if (!alg) {
    std::cerr << "Unsupported digest: " << name << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  if (impl_->hAlg) BCryptCloseAlgorithmProvider(impl_->hAlg, 0);
  impl_->hAlg = nullptr;

  NTSTATUS status =
      BCryptOpenAlgorithmProvider(&impl_->hAlg, alg, nullptr, BCRYPT_ALG_HANDLE_HMAC_FLAG);
  if (!BCRYPT_SUCCESS(status)) {
    std::cerr << "Error opening BCrypt HMAC algorithm provider" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  impl_->digest_name = name;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_key(const uint8_t key_data[key_length]) {
  delete[] key;
  key = new uint8_t[key_length];
  key_len = key_length;
  std::memcpy(key, key_data, key_length);

  // Remove existing key file if it exists, ignoring error if it doesn't exist
  if (std::remove(key_file_path.c_str()) != 0 && errno != ENOENT) return AMDCUID_STATUS_KEY_ERROR;

  std::ofstream key_file(key_file_path, std::ios::out | std::ios::binary);
  if (!key_file) return AMDCUID_STATUS_KEY_ERROR;

  key_file.write(reinterpret_cast<const char*>(key), key_length);
  if (!key_file) {
    key_file.close();
    return AMDCUID_STATUS_KEY_ERROR;
  }
  key_file.close();

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::generate_key(uint8_t out_key[key_length]) {
  if (!out_key) return AMDCUID_STATUS_INVALID_ARGUMENT;
  NTSTATUS status = BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(out_key), key_length,
                                    BCRYPT_USE_SYSTEM_PREFERRED_RNG);
  return BCRYPT_SUCCESS(status) ? AMDCUID_STATUS_SUCCESS : AMDCUID_STATUS_KEY_ERROR;
}

amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]) {
  BCRYPT_ALG_HANDLE hAlg = nullptr;
  BCRYPT_HASH_HANDLE hHash = nullptr;
  if (!BCRYPT_SUCCESS(BCryptOpenAlgorithmProvider(&hAlg, BCRYPT_SHA256_ALGORITHM, nullptr, 0)))
    return AMDCUID_STATUS_HMAC_ERROR;
  bool ok = BCRYPT_SUCCESS(BCryptCreateHash(hAlg, &hHash, nullptr, 0, nullptr, 0, 0)) &&
            BCRYPT_SUCCESS(
                BCryptHashData(hHash, const_cast<PUCHAR>(data), static_cast<ULONG>(data_len), 0)) &&
            BCRYPT_SUCCESS(BCryptFinishHash(hHash, out, 32, 0));
  if (hHash) BCryptDestroyHash(hHash);
  BCryptCloseAlgorithmProvider(hAlg, 0);
  return ok ? AMDCUID_STATUS_SUCCESS : AMDCUID_STATUS_HMAC_ERROR;
}

// ─────────────────────────────────────────────────────────────────────────────
// cifra backend (Linux / macOS) — vendored SHA-256 + HMAC, no external
// dependency. Source: third_party/cifra (ctz/cifra, CC0-1.0, commit 90f7631).
// ─────────────────────────────────────────────────────────────────────────────
#else  // !defined(_WIN32)

#include <strings.h>
#include <sys/random.h>
#include <sys/stat.h>
#include <unistd.h>

extern "C" {
#include "third_party/cifra/hmac.h"
#include "third_party/cifra/sha2.h"
}

struct cuid_hmac::Impl {
  std::string digest_name;
};

cuid_hmac::cuid_hmac() : impl_(nullptr), key(nullptr), key_len(key_length), valid(false) {
  const char* env_path = std::getenv("AMDCUID_HMAC_KEY_PATH");
  key_file_path = (env_path && env_path[0]) ? env_path : AMDCUID_CONFIG_DIR "/hmac_key.bin";
  impl_ = new Impl();
  impl_->digest_name = "SHA256";

  std::ifstream key_file_stream(key_file_path, std::ios::binary);
  if (!key_file_stream.is_open()) {
    return;
  }

  key_file_stream.seekg(0, std::ios::end);
  key_len = static_cast<size_t>(key_file_stream.tellg());
  if (key_len == 0 || key_len != key_length) {  // sanity check on key length
    std::cerr << "Invalid key length in key file" << std::endl;
    key_file_stream.close();
    return;
  }
  key_file_stream.seekg(0, std::ios::beg);

  key = new uint8_t[key_length];
  key_file_stream.read(reinterpret_cast<char*>(key), key_length);
  key_file_stream.close();

  valid = true;
}

cuid_hmac::cuid_hmac(uint8_t key_data[key_length])
    : impl_(nullptr), key(nullptr), key_len(key_length), valid(false) {
  impl_ = new Impl();
  impl_->digest_name = "SHA256";

  key = new uint8_t[key_length];
  std::memcpy(key, key_data, key_length);

  valid = true;
}

cuid_hmac::~cuid_hmac() {
  delete impl_;
  delete[] key;
}

amdcuid_status_t cuid_hmac::generate_hmac_sha256(const uint8_t* data, size_t data_len,
                                                 uint8_t* out_hash, size_t* out_len) {
  if (!impl_) {
    std::cerr << "HMAC context is not initialized" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  cf_hmac(key, key_len, data, data_len, out_hash, &cf_sha256);
  *out_len = CF_SHA256_HASHSZ;
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_algorithm(const char* digest_name) {
  if (!impl_) {
    std::cerr << "HMAC context is not initialized" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  // Only SHA-256 is vendored; this is the only digest ever requested via the
  // public API (see cuid_device.cc), so no other algorithm needs support here.
  const char* name = digest_name ? digest_name : "SHA256";
  if (strcasecmp(name, "SHA256") != 0 && strcasecmp(name, "SHA-256") != 0) {
    std::cerr << "Unsupported digest: " << name << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  impl_->digest_name = "SHA256";
  return AMDCUID_STATUS_SUCCESS;
}

// ─────────────────────────────────────────────────────────────────────────────
// set_hmac_key / generate_key / sha256_unkeyed
// ─────────────────────────────────────────────────────────────────────────────

amdcuid_status_t cuid_hmac::set_hmac_key(const uint8_t key_data[key_length]) {
  if (geteuid() != 0) return AMDCUID_STATUS_PERMISSION_DENIED;

  delete[] key;
  key = new uint8_t[key_length];
  key_len = key_length;
  std::memcpy(key, key_data, key_length);

  if (std::remove(key_file_path.c_str()) != 0 && errno != ENOENT) return AMDCUID_STATUS_KEY_ERROR;

  // TODO: This is backwards, we're never actually reading the stored key file,
  // but we're always overwriting it. Need to rethink this Maybe have public
  // facing API write the stored key file and then call this function to set the
  // in-memory key, and have this function only update the in-memory key without
  // touching the file system? That way we can ensure the file is only written
  // to when we actually want to change the key, and not every time we start the
  // daemon? Remember to also adjust the Windows CNG implementation to match any
  // changes made here for consistency.
  std::ofstream key_file(key_file_path, std::ios::out | std::ios::binary);
  if (!key_file) return AMDCUID_STATUS_KEY_ERROR;
  key_file.write(reinterpret_cast<const char*>(key), key_length);
  if (!key_file) {
    key_file.close();
    return AMDCUID_STATUS_KEY_ERROR;
  }
  key_file.close();

  if (chmod(key_file_path.c_str(), S_IRUSR | S_IWUSR) != 0) return AMDCUID_STATUS_PERMISSION_DENIED;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::generate_key(uint8_t out_key[key_length]) {
  if (!out_key) return AMDCUID_STATUS_INVALID_ARGUMENT;

  ssize_t n = getrandom(out_key, key_length, 0);
  if (n != static_cast<ssize_t>(key_length)) {
    std::cerr << "Error generating random bytes for HMAC key" << std::endl;
    return AMDCUID_STATUS_KEY_ERROR;
  }

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]) {
  cf_sha256_context ctx;
  cf_sha256_init(&ctx);
  cf_sha256_update(&ctx, data, data_len);
  cf_sha256_digest(&ctx, out);
  return AMDCUID_STATUS_SUCCESS;
}

#endif  // _WIN32
