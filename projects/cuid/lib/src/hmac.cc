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
#include <iostream>

#include "sha256.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// bcrypt.h must follow windows.h.
#include <bcrypt.h>
#include <direct.h>
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

// Used when no secret is provisioned. Byte-identical to CUID_DEFAULT_SEED in
// the kernel's amdgpu_cuid.c, so sysfs and this library agree on an
// unprovisioned machine. Not a substitute for provisioning; callers can check
// is_using_default_key().
constexpr char kDefaultSeed[] = "AMD-CUID-DEFAULT-SEED-v1";
constexpr size_t kDefaultSeedLen = sizeof(kDefaultSeed) - 1;

// The only digest CUID uses. A wider one would overrun the caller's 32-byte
// output buffer, so set_hmac_algorithm() rejects everything else.
bool is_sha256_name(const char* name) {
  if (!name) return true;  // nullptr means "the default", which is SHA-256
  return std::strcmp(name, "SHA256") == 0 || std::strcmp(name, "SHA-256") == 0 ||
         std::strcmp(name, "sha256") == 0 || std::strcmp(name, "sha-256") == 0;
}

// Directory portion of a path, or "." when there is none.
std::string parent_dir(const std::string& path) {
  const size_t slash = path.find_last_of('/');
  if (slash == std::string::npos) return ".";
  if (slash == 0) return "/";
  return path.substr(0, slash);
}

// Create the key directory (0755, as the packaging script does) so the API
// works on a machine where the post-install script never ran.
bool ensure_parent_dir(const std::string& path) {
  const std::string dir = parent_dir(path);
  struct stat st;
  if (stat(dir.c_str(), &st) == 0) return S_ISDIR(st.st_mode);
#if defined(_WIN32)
  return _mkdir(dir.c_str()) == 0 || errno == EEXIST;
#else
  return mkdir(dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP | S_IROTH | S_IXOTH) == 0 ||
         errno == EEXIST;
#endif
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

cuid_hmac::cuid_hmac()
    : impl_(nullptr), key(nullptr), key_len(key_length), valid(false), using_default_key(false) {
  // getenv races only against setenv, which this library never calls.
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  const char* env_path = std::getenv("AMDCUID_HMAC_KEY_PATH");
  key_file_path = (env_path && env_path[0]) ? env_path : AMDCUID_CONFIG_DIR "/hmac_key.bin";
  impl_ = new Impl();
  impl_->digest_name = "SHA256";

  std::ifstream key_file_stream(key_file_path, std::ios::binary);
  if (!key_file_stream.is_open()) {
    // No key provisioned: use the public default seed, as the kernel does,
    // rather than failing every privileged lookup on an unprovisioned machine.
    use_default_key();
    return;
  }

  key_file_stream.seekg(0, std::ios::end);
  const size_t file_len = static_cast<size_t>(key_file_stream.tellg());
  if (file_len != key_length) {
    // Wrong size is corruption, not absence: refuse rather than silently
    // changing every CUID on the machine.
    std::cerr << "Invalid key length in " << key_file_path << " (" << file_len
              << " bytes, expected " << key_length << ")" << std::endl;
    key_file_stream.close();
    return;
  }
  key_file_stream.seekg(0, std::ios::beg);

  key = new uint8_t[key_length];
  key_file_stream.read(reinterpret_cast<char*>(key), key_length);
  if (!key_file_stream) {
    std::cerr << "Failed to read " << key_file_path << std::endl;
    key_file_stream.close();
    delete[] key;
    key = nullptr;
    return;
  }
  key_file_stream.close();

  valid = true;
}

void cuid_hmac::use_default_key() {
  delete[] key;
  key = new uint8_t[kDefaultSeedLen];
  std::memcpy(key, kDefaultSeed, kDefaultSeedLen);
  key_len = kDefaultSeedLen;
  using_default_key = true;
  valid = true;
}

cuid_hmac::cuid_hmac(uint8_t key_data[key_length])
    : impl_(nullptr), key(nullptr), key_len(key_length), valid(false), using_default_key(false) {
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
  if (!key_data) return AMDCUID_STATUS_INVALID_ARGUMENT;

  if (key) {
    rocm::sha2::secure_zero(key, key_len);
    delete[] key;
  }
  key = new uint8_t[key_length];
  key_len = key_length;
  std::memcpy(key, key_data, key_length);
  valid = true;
  using_default_key = false;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::store_key(const uint8_t key_data[key_length]) {
  if (!key_data) return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Write a sibling temp file and rename over the target: atomic, so a reader
  // never sees a truncated key and a failed write cannot destroy the old one.
  if (!ensure_parent_dir(key_file_path)) {
    std::cerr << "Cannot create directory for " << key_file_path << std::endl;
    return AMDCUID_STATUS_KEY_ERROR;
  }

  const std::string tmp_path = key_file_path + ".new";

#if defined(_WIN32)
  {
    std::ofstream tmp(tmp_path, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!tmp) return AMDCUID_STATUS_KEY_ERROR;
    tmp.write(reinterpret_cast<const char*>(key_data), key_length);
    tmp.flush();
    if (!tmp) {
      tmp.close();
      std::remove(tmp_path.c_str());
      return AMDCUID_STATUS_KEY_ERROR;
    }
  }
  // rename() refuses to clobber an existing file on Windows.
  if (!MoveFileExA(tmp_path.c_str(), key_file_path.c_str(), MOVEFILE_REPLACE_EXISTING)) {
    std::remove(tmp_path.c_str());
    return AMDCUID_STATUS_KEY_ERROR;
  }
  return AMDCUID_STATUS_SUCCESS;
#else
  // O_EXCL so we never write into a pre-created file; 0600 from creation.
  int fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
  if (fd < 0 && errno == EEXIST) {
    // Stale temporary from an interrupted run; it is ours to reclaim.
    if (unlink(tmp_path.c_str()) != 0) return AMDCUID_STATUS_KEY_ERROR;
    fd = open(tmp_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, S_IRUSR | S_IWUSR);
  }
  if (fd < 0) {
    return (errno == EACCES || errno == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                               : AMDCUID_STATUS_KEY_ERROR;
  }

  auto fail = [&]() {
    close(fd);
    unlink(tmp_path.c_str());
    return AMDCUID_STATUS_KEY_ERROR;
  };

  size_t written = 0;
  while (written < key_length) {
    ssize_t n = write(fd, key_data + written, key_length - written);
    if (n < 0) {
      if (errno == EINTR) continue;
      return fail();
    }
    written += static_cast<size_t>(n);
  }

  // Durable before the rename, or a crash can leave the file present but empty.
  if (fsync(fd) != 0) return fail();
  if (close(fd) != 0) {
    unlink(tmp_path.c_str());
    return AMDCUID_STATUS_KEY_ERROR;
  }

  if (rename(tmp_path.c_str(), key_file_path.c_str()) != 0) {
    unlink(tmp_path.c_str());
    return AMDCUID_STATUS_KEY_ERROR;
  }

  // Persist the directory entry so the rename survives power loss.
  const size_t slash = key_file_path.find_last_of('/');
  const std::string dir = (slash == std::string::npos) ? "." : key_file_path.substr(0, slash);
  int dir_fd = open(dir.empty() ? "/" : dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (dir_fd >= 0) {
    (void)fsync(dir_fd);
    close(dir_fd);
  }

  return AMDCUID_STATUS_SUCCESS;
#endif
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
