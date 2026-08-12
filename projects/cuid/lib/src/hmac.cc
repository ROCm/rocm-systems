// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// HMAC-SHA-256 over the in-tree SHA-256 in sha256.{h,cc}.
//
// There is one code path on every platform. The previous three compile-time
// backends (Windows CNG, OpenSSL 3 EVP_MAC, OpenSSL 1.x HMAC_CTX) existed only
// to reach a primitive that fits in ~200 lines of standard C++, at the cost of
// making libamdcuid -- and therefore everything that links it statically,
// including libhsa-runtime64.so and libamd_smi.so -- depend on libcrypto.
//
// The only platform-specific code left is the CSPRNG used to mint a new key.

#include "hmac.h"

#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>

#include "sha256.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// bcrypt.h must follow windows.h.
#include <bcrypt.h>
#else
#include <unistd.h>
// getrandom(2) needs glibc >= 2.25 (or musl); where it is missing, and where
// the syscall itself is missing (pre-3.17 kernels, some containers/seccomp
// profiles), fill_random() falls back to reading /dev/urandom. Define
// AMDCUID_HAVE_GETRANDOM=0 on the command line to force the fallback path.
#if !defined(AMDCUID_HAVE_GETRANDOM) && defined(__has_include)
#if __has_include(<sys/random.h>)
#define AMDCUID_HAVE_GETRANDOM 1
#endif
#endif
#if AMDCUID_HAVE_GETRANDOM
#include <sys/random.h>
#endif
#endif

#ifndef AMDCUID_CONFIG_DIR
#error "AMDCUID_CONFIG_DIR must be defined via CMake"
#endif

namespace {

// The only digest the CUID specification uses. set_hmac_algorithm() accepts it
// (and its hyphenated spelling) and rejects everything else: generate_hmac_sha256()
// writes into a caller-supplied 32-byte buffer, so a wider digest would overrun it.
bool is_sha256_name(const char* name) {
  if (!name) return true;  // nullptr means "the default", which is SHA-256
  return std::strcmp(name, "SHA256") == 0 || std::strcmp(name, "SHA-256") == 0 ||
         std::strcmp(name, "sha256") == 0 || std::strcmp(name, "sha-256") == 0;
}

// Fill buf with cryptographically secure random bytes.
bool fill_random(uint8_t* buf, size_t len) {
#if defined(_WIN32)
  return BCRYPT_SUCCESS(BCryptGenRandom(nullptr, reinterpret_cast<PUCHAR>(buf),
                                        static_cast<ULONG>(len), BCRYPT_USE_SYSTEM_PREFERRED_RNG));
#else
#if AMDCUID_HAVE_GETRANDOM
  size_t off = 0;
  while (off < len) {
    ssize_t n = getrandom(buf + off, len - off, 0);
    if (n < 0) {
      if (errno == EINTR) continue;
      break;  // ENOSYS on a pre-3.17 kernel; fall through to /dev/urandom
    }
    off += static_cast<size_t>(n);
  }
  if (off == len) return true;
#endif
  std::ifstream urandom("/dev/urandom", std::ios::binary);
  if (!urandom) return false;
  urandom.read(reinterpret_cast<char*>(buf), static_cast<std::streamsize>(len));
  return urandom.gcount() == static_cast<std::streamsize>(len);
#endif
}

}  // namespace

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
  if (key_len != key_length) {  // sanity check on key length
    std::cerr << "Invalid key length in key file" << std::endl;
    key_file_stream.close();
    key_len = key_length;
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
  if (key) {
    rocm::sha2::secure_zero(key, key_len);
    delete[] key;
  }
}

amdcuid_status_t cuid_hmac::generate_hmac_sha256(const uint8_t* data, size_t data_len,
                                                 uint8_t* out_hash, size_t* out_len) {
  if (!impl_ || !out_hash) {
    std::cerr << "HMAC context is not initialized" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }
  if (!key) {
    std::cerr << "No HMAC key is set" << std::endl;
    return AMDCUID_STATUS_KEY_ERROR;
  }

  rocm::sha2::hmac_sha256(key, key_len, data, data_len, out_hash);
  if (out_len) *out_len = rocm::sha2::SHA256_DIGEST_SIZE;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_algorithm(const char* digest_name) {
  if (!impl_) {
    std::cerr << "HMAC context is not initialized" << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  if (!is_sha256_name(digest_name)) {
    std::cerr << "Unsupported digest: " << digest_name << " (only SHA-256 is supported)"
              << std::endl;
    return AMDCUID_STATUS_HMAC_ERROR;
  }

  impl_->digest_name = "SHA256";
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_key(const uint8_t key_data[key_length]) {
#if !defined(_WIN32)
  if (geteuid() != 0) return AMDCUID_STATUS_PERMISSION_DENIED;
#endif

  if (key) {
    rocm::sha2::secure_zero(key, key_len);
    delete[] key;
  }
  key = new uint8_t[key_length];
  key_len = key_length;
  std::memcpy(key, key_data, key_length);
  valid = true;

  if (std::remove(key_file_path.c_str()) != 0 && errno != ENOENT) return AMDCUID_STATUS_KEY_ERROR;

  // TODO: This is backwards, we're never actually reading the stored key file,
  // but we're always overwriting it. Need to rethink this Maybe have public
  // facing API write the stored key file and then call this function to set the
  // in-memory key, and have this function only update the in-memory key without
  // touching the file system? That way we can ensure the file is only written
  // to when we actually want to change the key, and not every time we start the
  // daemon?
  std::ofstream key_file(key_file_path, std::ios::out | std::ios::binary);
  if (!key_file) return AMDCUID_STATUS_KEY_ERROR;
  key_file.write(reinterpret_cast<const char*>(key), key_length);
  if (!key_file) {
    key_file.close();
    return AMDCUID_STATUS_KEY_ERROR;
  }
  key_file.close();

#if !defined(_WIN32)
  if (chmod(key_file_path.c_str(), S_IRUSR | S_IWUSR) != 0) return AMDCUID_STATUS_PERMISSION_DENIED;
#endif

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::generate_key(uint8_t out_key[key_length]) {
  if (!out_key) return AMDCUID_STATUS_INVALID_ARGUMENT;

  if (!fill_random(out_key, key_length)) {
    std::cerr << "Error generating random bytes for HMAC key" << std::endl;
    return AMDCUID_STATUS_KEY_ERROR;
  }

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t sha256_unkeyed(const uint8_t* data, size_t data_len, uint8_t out[32]) {
  if (!out || (!data && data_len > 0)) return AMDCUID_STATUS_INVALID_ARGUMENT;
  rocm::sha2::sha256_digest(data, data_len, out);
  return AMDCUID_STATUS_SUCCESS;
}
