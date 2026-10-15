// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// HMAC-SHA-256 over the in-tree SHA-256 (sha256.h). One code path on every
// platform; only the CSPRNG and the key-source scan below are
// platform-specific.

#include "hmac.h"

#include <fcntl.h>
#include <sys/stat.h>

#include <array>
#include <cerrno>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <mutex>
#include <new>

#include "cuid_util.h"
#include "rocm/sha2/log.h"
#include "rocm/sha2/sha256.h"

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
// bcrypt.h must follow windows.h.
#include <bcrypt.h>
#else
#include <dirent.h>
#include <sys/mman.h>
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

#if !defined(_WIN32) && !defined(MADV_WIPEONFORK)
#define MADV_WIPEONFORK 18
#endif

#if !defined(_WIN32)
size_t key_mapping_size(size_t len) {
  const size_t page = static_cast<size_t>(sysconf(_SC_PAGESIZE));
  return (len + page - 1) / page * page;
}
#endif

// Key material lives in its own anonymous mapping, which the kernel leaves out
// of core dumps (MADV_DONTDUMP) and gives a fork()ed child zero-filled
// (MADV_WIPEONFORK, Linux 4.14). A fork from another thread while a call holds
// the key copies none of it. Where the kernel refuses either, the node key
// (`protect`) gets no storage, nullptr; any other key gets the mapping without
// that protection, which is still zeroed when it is released.
uint8_t* new_key_storage(size_t len, bool protect) {
#if defined(_WIN32)
  (void)protect;
  return new uint8_t[len];
#else
  const size_t size = key_mapping_size(len);
  void* mapping = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (mapping == MAP_FAILED) {
    LOG(ERROR, "cannot map key storage: " << CuidUtilities::errno_string(errno));
    return nullptr;
  }
  for (const int advice : {MADV_WIPEONFORK, MADV_DONTDUMP}) {
    if (madvise(mapping, size, advice) != 0 && protect) {
      LOG(ERROR, "cannot protect key storage (madvise "
                     << advice << "): " << CuidUtilities::errno_string(errno));
      munmap(mapping, size);
      return nullptr;
    }
  }
  return static_cast<uint8_t*>(mapping);
#endif
}

void delete_key_storage(uint8_t* storage, size_t len) {
  if (!storage) return;
  rocm::sha2::secure_zero(storage, len);
#if defined(_WIN32)
  delete[] storage;
#else
  munmap(storage, key_mapping_size(len));
#endif
}

// HMAC-SHA-256 (RFC 2104) over rocm::sha2's streaming SHA-256, with the whole
// working state -- the padded key, the pad and the hash context -- in key
// storage. rocm::sha2::hmac_sha256() keeps them on the stack, where a fork()
// from another thread would copy them.
struct HmacWork {
  std::array<uint8_t, rocm::sha2::SHA256_DIGEST_SIZE> digest;
  uint8_t k0[rocm::sha2::SHA256_BLOCK_SIZE];
  uint8_t pad[rocm::sha2::SHA256_BLOCK_SIZE];
  rocm::sha2::sha256 hash;
};

bool hmac_sha256_in_key_storage(const uint8_t* key, size_t key_len, const uint8_t* msg,
                                size_t msg_len, uint8_t out[rocm::sha2::SHA256_DIGEST_SIZE],
                                bool protect) {
  uint8_t* const storage = new_key_storage(sizeof(HmacWork), protect);
  if (!storage) return false;
  auto* work = new (storage) HmacWork{};

  if (key_len > rocm::sha2::SHA256_BLOCK_SIZE) {
    work->hash.update(key, key_len);
    new (&work->digest) std::array<uint8_t, rocm::sha2::SHA256_DIGEST_SIZE>(work->hash.digest());
    std::memcpy(work->k0, work->digest.data(), work->digest.size());
  } else if (key_len > 0) {
    std::memcpy(work->k0, key, key_len);
  }

  for (size_t i = 0; i < sizeof(work->pad); ++i) work->pad[i] = work->k0[i] ^ 0x36;
  new (&work->hash) rocm::sha2::sha256();
  work->hash.update(work->pad, sizeof(work->pad));
  if (msg_len > 0) work->hash.update(msg, msg_len);
  new (&work->digest) std::array<uint8_t, rocm::sha2::SHA256_DIGEST_SIZE>(work->hash.digest());

  for (size_t i = 0; i < sizeof(work->pad); ++i) work->pad[i] = work->k0[i] ^ 0x5c;
  new (&work->hash) rocm::sha2::sha256();
  work->hash.update(work->pad, sizeof(work->pad));
  work->hash.update(work->digest.data(), work->digest.size());
  new (&work->digest) std::array<uint8_t, rocm::sha2::SHA256_DIGEST_SIZE>(work->hash.digest());
  std::memcpy(out, work->digest.data(), work->digest.size());

  delete_key_storage(storage, sizeof(HmacWork));
  return true;
}

// SHA-256 of key material, with the context, which keeps the last partial
// block, in key storage rather than in a stack frame left behind on return.
bool sha256_in_key_storage(const uint8_t* data, size_t len,
                           uint8_t out[rocm::sha2::SHA256_DIGEST_SIZE], bool protect) {
  uint8_t* const storage = new_key_storage(sizeof(rocm::sha2::sha256), protect);
  if (!storage) return false;
  auto* hash = new (storage) rocm::sha2::sha256();
  if (len > 0) hash->update(data, len);
  const auto digest = hash->digest();
  std::memcpy(out, digest.data(), digest.size());
  delete_key_storage(storage, sizeof(rocm::sha2::sha256));
  return true;
}

#if !defined(_WIN32)
// Read up to `len` bytes of `path`. Returns the number of bytes read, or -1
// with errno from the failed open() or read().
ssize_t read_file(const std::string& path, uint8_t* buf, size_t len) {
  const int fd = open(path.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0) return -1;
  size_t got = 0;
  while (got < len) {
    const ssize_t n = read(fd, buf + got, len - got);
    if (n < 0) {
      if (errno == EINTR) continue;
      const int err = errno;
      close(fd);
      errno = err;
      return -1;
    }
    if (n == 0) break;
    got += static_cast<size_t>(n);
  }
  close(fd);
  return static_cast<ssize_t>(got);
}

// Whether `device` is bound to amdgpu, whose cuid_seed is the only key source:
// 1 if it is, 0 if it has no driver or another one, -1 with errno when its
// driver link cannot be read.
int bound_to_amdgpu(const std::string& device) {
  char target[PATH_MAX];
  const ssize_t n = readlink((device + "/driver").c_str(), target, sizeof(target) - 1);
  if (n < 0) return errno == ENOENT || errno == ENOTDIR ? 0 : -1;
  target[n] = '\0';
  const char* name = std::strrchr(target, '/');
  return std::strcmp(name ? name + 1 : target, "amdgpu") == 0 ? 1 : 0;
}
#endif  // !_WIN32

}  // namespace

std::recursive_mutex& cuid_operation_mutex() {
  static std::recursive_mutex mutex;
  return mutex;
}

cuid_hmac::cuid_hmac() : key(nullptr), key_len(key_length), valid(false) { init_sha2_logging(); }

amdcuid_status_t cuid_hmac::reload_key() {
#if !defined(_WIN32)
  if (geteuid() == 0) return reload_key("/sys/bus/pci/devices");
#endif
  clear_key();
  return AMDCUID_STATUS_SUCCESS;
}

void cuid_hmac::clear_key() {
  std::lock_guard<std::mutex> lock(key_mutex_);
  delete_key_storage(key, key_len);
  key = nullptr;
  valid = false;
  node_key = false;
  key_len = key_length;
}

amdcuid_status_t cuid_hmac::reload_key(const std::string& devices_dir) {
#if defined(_WIN32)
  (void)devices_dir;
  clear_key();
  return AMDCUID_STATUS_SUCCESS;
#else
  bool denied = false;
  bool failed = false;
  DIR* dir = opendir(devices_dir.c_str());
  if (!dir) {
    const int err = errno;
    if (err != ENOENT) {
      LOG(ERROR, "cannot list " << devices_dir << ": " << CuidUtilities::errno_string(err));
      return AMDCUID_STATUS_FILE_ERROR;
    }
    clear_key();
    return AMDCUID_STATUS_SUCCESS;
  }

  // One extra byte, so a cuid_seed longer than a key reads as the wrong length.
  constexpr size_t read_length = key_length + 1;
  uint8_t* const bytes = new_key_storage(read_length, true);
  if (!bytes) {
    closedir(dir);
    clear_key();
    return AMDCUID_STATUS_KEY_ERROR;
  }
  bool found = false;
  for (;;) {
    errno = 0;
    // This call site owns its DIR*, which is all POSIX requires; readdir_r is
    // deprecated and must not be adopted.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const struct dirent* entry = readdir(dir);
    if (!entry) {
      const int err = errno;
      if (err != 0) {
        LOG(ERROR, "cannot list " << devices_dir << ": " << CuidUtilities::errno_string(err));
        failed = true;
      }
      break;
    }
    if (entry->d_name[0] == '.') continue;
    const std::string device = devices_dir + "/" + entry->d_name;
    const int amdgpu = bound_to_amdgpu(device);
    if (amdgpu < 0) {
      const int err = errno;
      if (err == EACCES || err == EPERM) {
        denied = true;
      } else {
        LOG(ERROR, "cannot read " << device << "/driver: " << CuidUtilities::errno_string(err));
        failed = true;
      }
    }
    if (amdgpu != 1) continue;
    const std::string path = device + "/cuid_seed";
    const ssize_t got = read_file(path, bytes, read_length);
    const int err = errno;
    if (got == static_cast<ssize_t>(key_length)) {
      found = true;
      break;
    }
    if (got >= 0) {
      LOG(ERROR, path << " does not hold a " << key_length << "-octet key");
      failed = true;
    } else if (err == EACCES || err == EPERM) {
      denied = true;
    } else if (err != ENOENT && err != ENODATA) {
      LOG(ERROR, "cannot read " << path << ": " << CuidUtilities::errno_string(err));
      failed = true;
    }
  }
  closedir(dir);

  const amdcuid_status_t stored = found ? set_hmac_key(bytes) : AMDCUID_STATUS_SUCCESS;
  delete_key_storage(bytes, read_length);
  if (found) return stored;
  if (denied) {
    clear_key();
    return AMDCUID_STATUS_PERMISSION_DENIED;
  }
  if (failed) return AMDCUID_STATUS_FILE_ERROR;
  clear_key();
  return AMDCUID_STATUS_SUCCESS;
#endif
}

cuid_hmac::cuid_hmac(uint8_t key_data[key_length])
    : key(nullptr), key_len(key_length), valid(false) {
  init_sha2_logging();
  key = new_key_storage(key_length, false);
  if (!key) return;
  std::memcpy(key, key_data, key_length);

  valid = true;
}

cuid_hmac::cuid_hmac(const char* key_data, size_t len) : key(nullptr), key_len(len), valid(false) {
  init_sha2_logging();
  if (!key_data || len == 0) {
    key_len = key_length;
    return;  // leaves valid == false
  }

  key = new_key_storage(len, false);
  if (!key) return;
  std::memcpy(key, key_data, len);

  valid = true;
}

cuid_hmac::~cuid_hmac() { delete_key_storage(key, key_len); }

amdcuid_status_t cuid_hmac::generate_hmac_sha256(const uint8_t* data, size_t data_len,
                                                 uint8_t* out_hash, size_t* out_len) {
  if (!out_hash) return AMDCUID_STATUS_HMAC_ERROR;

  std::lock_guard<std::mutex> lock(key_mutex_);
  if (!key) {
    LOG(ERROR, "No HMAC key is set");
    return AMDCUID_STATUS_KEY_ERROR;
  }

  if (!hmac_sha256_in_key_storage(key, key_len, data, data_len, out_hash, node_key))
    return AMDCUID_STATUS_HMAC_ERROR;
  if (out_len) *out_len = rocm::sha2::SHA256_DIGEST_SIZE;

  return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t cuid_hmac::get_key_info(amdcuid_key_info_t* info) const {
  if (!info) return AMDCUID_STATUS_INVALID_ARGUMENT;

  // Hashed under the lock, so a concurrent reload_key()/set_hmac_key() can't
  // land mid-read, and in key storage, so no copy of the key is made.
  std::memset(info, 0, sizeof(*info));
  uint8_t digest[32];
  {
    std::lock_guard<std::mutex> lock(key_mutex_);
    if (!key || !valid) return AMDCUID_STATUS_SUCCESS;
    if (!sha256_in_key_storage(key, key_len, digest, node_key)) return AMDCUID_STATUS_KEY_ERROR;
  }

  std::memcpy(info->fingerprint, digest, sizeof(info->fingerprint));
  rocm::sha2::secure_zero(digest, sizeof(digest));
  info->provisioned = 1;
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
  delete_key_storage(key, key_len);
  key = new_key_storage(key_length, true);
  key_len = key_length;
  valid = key != nullptr;
  node_key = valid;
  if (!key) return AMDCUID_STATUS_KEY_ERROR;
  std::memcpy(key, key_data, key_length);

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

bool CuidUtilities::is_rejected_key(const uint8_t key[key_length]) {
  bool all_equal = true;
  for (size_t i = 1; i < key_length; ++i) all_equal = all_equal && key[i] == key[0];
  if (all_equal) return true;

  const auto padded_equals = [key](const char* text) {
    uint8_t padded[key_length] = {};
    std::memcpy(padded, text, std::strlen(text));
    return std::memcmp(key, padded, key_length) == 0;
  };
  if (padded_equals("AMD-CUID-DEFAULT-SEED-v1") || padded_equals("AMD-CUID-TEMP-KEY-v1"))
    return true;

  // The two conformance-vector keys: 00..1f and 0xa5 ^ n.
  uint8_t counting[key_length];
  uint8_t xored[key_length];
  for (size_t i = 0; i < key_length; ++i) {
    counting[i] = static_cast<uint8_t>(i);
    xored[i] = static_cast<uint8_t>(0xA5 ^ i);
  }
  return std::memcmp(key, counting, key_length) == 0 || std::memcmp(key, xored, key_length) == 0;
}

amdcuid_status_t CuidUtilities::find_cuid_seed(const std::string& devices_dir, std::string& bdf) {
#if defined(_WIN32)
  (void)devices_dir;
  (void)bdf;
  return AMDCUID_STATUS_UNSUPPORTED;
#else
  DIR* dir = opendir(devices_dir.c_str());
  if (!dir) {
    const int err = errno;
    if (err == ENOENT) return AMDCUID_STATUS_UNSUPPORTED;
    LOG(ERROR, "cannot list " << devices_dir << ": " << CuidUtilities::errno_string(err));
    return (err == EACCES || err == EPERM) ? AMDCUID_STATUS_PERMISSION_DENIED
                                           : AMDCUID_STATUS_FILE_ERROR;
  }

  bool found = false;
  bool denied = false;
  bool failed = false;
  for (;;) {
    errno = 0;
    // This call site owns its DIR*, which is all POSIX requires; readdir_r is
    // deprecated and must not be adopted.
    // NOLINTNEXTLINE(concurrency-mt-unsafe)
    const struct dirent* entry = readdir(dir);
    if (!entry) {
      const int err = errno;
      if (err != 0) {
        LOG(ERROR, "cannot list " << devices_dir << ": " << CuidUtilities::errno_string(err));
        failed = true;
      }
      break;
    }
    if (entry->d_name[0] == '.') continue;
    const std::string device = devices_dir + "/" + entry->d_name;
    const int amdgpu = bound_to_amdgpu(device);
    if (amdgpu < 0) {
      const int err = errno;
      if (err == EACCES || err == EPERM) {
        denied = true;
      } else {
        LOG(ERROR, "cannot read " << device << "/driver: " << CuidUtilities::errno_string(err));
        failed = true;
      }
    }
    if (amdgpu != 1) continue;
    struct stat st{};
    const std::string path = device + "/cuid_seed";
    if (stat(path.c_str(), &st) == 0) {
      bdf = entry->d_name;
      found = true;
      break;
    }
    const int err = errno;
    if (err == EACCES || err == EPERM) {
      denied = true;
    } else if (err != ENOENT && err != ENOTDIR) {
      LOG(ERROR, "cannot stat " << path << ": " << CuidUtilities::errno_string(err));
      failed = true;
    }
  }
  closedir(dir);
  if (found) return AMDCUID_STATUS_SUCCESS;
  if (denied) return AMDCUID_STATUS_PERMISSION_DENIED;
  if (failed) return AMDCUID_STATUS_FILE_ERROR;
  return AMDCUID_STATUS_UNSUPPORTED;
#endif
}
