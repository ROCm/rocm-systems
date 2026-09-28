// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// HMAC-SHA-256 over the in-tree SHA-256 (sha256.h). One code path on every
// platform; only the CSPRNG below is platform-specific.

#include "hmac.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>

#include "cuid_util.h"
#include "rocm/sha2/log.h"
#include "rocm/sha2/sha256.h"

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

namespace {

// rocm::sha2 has no logging dependency of its own; it reports diagnostics
// (e.g. update() after finalize(), which should never happen in this library)
// through a swappable handler that writes straight to stderr by default. Route
// it into cuid's own Logger instead, so it is subject to the same level
// filtering as the rest of the library and downstream apps aren't surprised by
// unconditional stderr output from a dependency they don't call directly.
void sha2_log_handler(const char* message) { LOG(ERROR, message); }

// Idempotent; called from every cuid_hmac constructor so the handler is
// installed before any sha256 use regardless of construction order.
void init_sha2_logging() {
  static std::once_flag once;
  std::call_once(once, [] { rocm::sha2::set_log_handler(&sha2_log_handler); });
}

// The only digest CUID uses. A wider one would overrun the caller's 32-byte
// output buffer, so set_hmac_algorithm() rejects everything else.
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

std::recursive_mutex& cuid_operation_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

cuid_hmac::cuid_hmac() : key(nullptr), key_len(key_length), valid(false) { init_sha2_logging(); }

cuid_hmac::cuid_hmac(uint8_t key_data[key_length])
    : key(nullptr), key_len(key_length), valid(false) {
  init_sha2_logging();
  key = new uint8_t[key_length];
  std::memcpy(key, key_data, key_length);

  valid = true;
}

cuid_hmac::cuid_hmac(const char* key_data, size_t len) : key(nullptr), key_len(len), valid(false) {
  init_sha2_logging();
  if (!key_data || len == 0) {
    key_len = key_length;
    return;  // leaves valid == false
  }

  key = new uint8_t[len];
  std::memcpy(key, key_data, len);

  valid = true;
}

cuid_hmac::~cuid_hmac() {
  if (key) {
    rocm::sha2::secure_zero(key, key_len);
    delete[] key;
  }
}

amdcuid_status_t cuid_hmac::generate_hmac_sha256(const uint8_t* data, size_t data_len,
                                                 uint8_t* out_hash, size_t* out_len) {
  if (!out_hash) return AMDCUID_STATUS_HMAC_ERROR;

  std::lock_guard<std::mutex> lock(key_mutex_);
  if (!key) {
    LOG(ERROR, "No HMAC key is set");
    return AMDCUID_STATUS_KEY_ERROR;
  }

  rocm::sha2::hmac_sha256(key, key_len, data, data_len, out_hash);
  if (out_len) *out_len = rocm::sha2::SHA256_DIGEST_SIZE;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_algorithm(const char* digest_name) {
  if (!is_sha256_name(digest_name)) {
    LOG(ERROR, "Unsupported digest: " << digest_name << " (only SHA-256 is supported)");
    return AMDCUID_STATUS_HMAC_ERROR;
  }
  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::set_hmac_key(const uint8_t key_data[key_length]) {
  if (!key_data) return AMDCUID_STATUS_INVALID_ARGUMENT;

  std::lock_guard<std::mutex> lock(key_mutex_);
  if (key) {
    rocm::sha2::secure_zero(key, key_len);
    delete[] key;
  }
  key = new uint8_t[key_length];
  key_len = key_length;
  std::memcpy(key, key_data, key_length);
  valid = true;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::generate_key(uint8_t out_key[key_length]) {
  if (!out_key) return AMDCUID_STATUS_INVALID_ARGUMENT;

  if (!fill_random(out_key, key_length)) {
    LOG(ERROR, "Error generating random bytes for HMAC key");
    return AMDCUID_STATUS_KEY_ERROR;
  }

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidUtilities::sha256_unkeyed(const uint8_t* data, size_t data_len,
                                               uint8_t out[32]) {
  if (!out || (!data && data_len > 0)) return AMDCUID_STATUS_INVALID_ARGUMENT;
  rocm::sha2::sha256_digest(data, data_len, out);
  return AMDCUID_STATUS_SUCCESS;
}
