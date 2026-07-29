// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file hooks/translation_disk_cache.cpp
/// @brief Session-scoped store for translated gfx1250 code objects.

#include "rocjitsu/hooks/translation_disk_cache.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocjitsu::hotswap {
namespace {

/// @brief Bumped by hand to invalidate every entry a deployment already holds.
/// @details The build identity below already separates differently-built
/// translators. This exists for the case where an operator needs to discard
/// entries without a rebuild.
constexpr uint32_t kCacheEpoch = 1;

/// @brief Layout revision, carried in the path so a change starts a new tree.
constexpr std::string_view kSchemaDir = "v1";

/// @brief Prefix distinguishing this digest from a plain hash of the source.
constexpr std::string_view kKeyDomain = "rjc1";

constexpr std::string_view kManifestMagic = "rjcache1";

/// @brief Largest store, before the proportional and headroom limits apply.
constexpr uint64_t kAbsoluteCapacityBytes = 256ull << 20;

/// @brief Share of the filesystem this store may occupy.
constexpr uint64_t kCapacityPermilleOfTotal = 100; // 10%

/// @brief Share of currently free space this store may take.
constexpr uint64_t kCapacityPermilleOfFree = 500; // 50%

/// @brief Below this much free space the store stops accepting writes.
/// @details The runtime directory is shared with the daemon socket and config,
/// and is memory-backed, so exhausting it breaks more than caching.
constexpr uint64_t kHeadroomBytes = 64ull << 20;

/// @brief Fraction of capacity a single entry may occupy.
constexpr uint64_t kMaxEntryDivisor = 16;

/// @brief Largest object a read will materialise.
/// @details Derived from the write-side per-entry limit so a reader never has to
/// buffer something the writer would have refused. It also bounds the damage
/// from a corrupt size field before the digest gets a chance to reject it.
constexpr uint64_t kMaxObjectBytes = kAbsoluteCapacityBytes / kMaxEntryDivisor;

/// @brief Evict down to this share of capacity so eviction is amortised.
constexpr uint64_t kEvictTargetPermille = 900; // 90%

/// @brief Age at which an unrenamed temporary is treated as abandoned.
/// @details A writer killed between creating its temporary and renaming it
/// leaves bytes that no entry accounts for and nothing would ever reclaim. The
/// threshold only has to exceed the time one write can take, which is a single
/// buffered write and an fsync, so it is generous by orders of magnitude.
constexpr time_t kAbandonedTempSeconds = 600;

// ---------------------------------------------------------------------------
// SHA-256. Self-contained: this library links no crypto dependency, and the
// store needs a digest strong enough that two distinct code objects cannot be
// made to collide by accident.
// ---------------------------------------------------------------------------

struct Sha256 {
  uint32_t state[8] = {0x6a09e667u, 0xbb67ae85u, 0x3c6ef372u, 0xa54ff53au,
                       0x510e527fu, 0x9b05688cu, 0x1f83d9abu, 0x5be0cd19u};
  uint64_t bit_count = 0;
  uint8_t buffer[64] = {};
  size_t buffered = 0;

  static uint32_t rotr(uint32_t v, uint32_t n) { return (v >> n) | (v << (32u - n)); }

  void compress(const uint8_t *block) {
    static constexpr uint32_t k[64] = {
        0x428a2f98u, 0x71374491u, 0xb5c0fbcfu, 0xe9b5dba5u, 0x3956c25bu, 0x59f111f1u, 0x923f82a4u,
        0xab1c5ed5u, 0xd807aa98u, 0x12835b01u, 0x243185beu, 0x550c7dc3u, 0x72be5d74u, 0x80deb1feu,
        0x9bdc06a7u, 0xc19bf174u, 0xe49b69c1u, 0xefbe4786u, 0x0fc19dc6u, 0x240ca1ccu, 0x2de92c6fu,
        0x4a7484aau, 0x5cb0a9dcu, 0x76f988dau, 0x983e5152u, 0xa831c66du, 0xb00327c8u, 0xbf597fc7u,
        0xc6e00bf3u, 0xd5a79147u, 0x06ca6351u, 0x14292967u, 0x27b70a85u, 0x2e1b2138u, 0x4d2c6dfcu,
        0x53380d13u, 0x650a7354u, 0x766a0abbu, 0x81c2c92eu, 0x92722c85u, 0xa2bfe8a1u, 0xa81a664bu,
        0xc24b8b70u, 0xc76c51a3u, 0xd192e819u, 0xd6990624u, 0xf40e3585u, 0x106aa070u, 0x19a4c116u,
        0x1e376c08u, 0x2748774cu, 0x34b0bcb5u, 0x391c0cb3u, 0x4ed8aa4au, 0x5b9cca4fu, 0x682e6ff3u,
        0x748f82eeu, 0x78a5636fu, 0x84c87814u, 0x8cc70208u, 0x90befffau, 0xa4506cebu, 0xbef9a3f7u,
        0xc67178f2u};
    uint32_t w[64];
    for (int i = 0; i < 16; ++i)
      w[i] = (static_cast<uint32_t>(block[i * 4]) << 24) |
             (static_cast<uint32_t>(block[i * 4 + 1]) << 16) |
             (static_cast<uint32_t>(block[i * 4 + 2]) << 8) |
             static_cast<uint32_t>(block[i * 4 + 3]);
    for (int i = 16; i < 64; ++i) {
      const uint32_t s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >> 3);
      const uint32_t s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >> 10);
      w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = state[0], b = state[1], c = state[2], d = state[3];
    uint32_t e = state[4], f = state[5], g = state[6], h = state[7];
    for (int i = 0; i < 64; ++i) {
      const uint32_t s1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
      const uint32_t ch = (e & f) ^ (~e & g);
      const uint32_t t1 = h + s1 + ch + k[i] + w[i];
      const uint32_t s0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
      const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
      const uint32_t t2 = s0 + maj;
      h = g;
      g = f;
      f = e;
      e = d + t1;
      d = c;
      c = b;
      b = a;
      a = t1 + t2;
    }
    state[0] += a;
    state[1] += b;
    state[2] += c;
    state[3] += d;
    state[4] += e;
    state[5] += f;
    state[6] += g;
    state[7] += h;
  }

  void update(const void *data, size_t size) {
    const auto *p = static_cast<const uint8_t *>(data);
    bit_count += static_cast<uint64_t>(size) * 8u;
    while (size > 0) {
      const size_t take = std::min(size, sizeof(buffer) - buffered);
      std::memcpy(buffer + buffered, p, take);
      buffered += take;
      p += take;
      size -= take;
      if (buffered == sizeof(buffer)) {
        compress(buffer);
        buffered = 0;
      }
    }
  }

  void update(std::string_view text) { update(text.data(), text.size()); }

  std::array<uint8_t, 32> finish() {
    const uint64_t bits = bit_count;
    const uint8_t pad = 0x80;
    update(&pad, 1);
    const uint8_t zero = 0;
    while (buffered != 56)
      update(&zero, 1);
    uint8_t tail[8];
    for (int i = 0; i < 8; ++i)
      tail[i] = static_cast<uint8_t>(bits >> (56 - i * 8));
    // Length is appended directly: routing it through update() would grow
    // bit_count again and corrupt the padding invariant.
    std::memcpy(buffer + buffered, tail, sizeof(tail));
    compress(buffer);
    std::array<uint8_t, 32> out{};
    for (int i = 0; i < 8; ++i) {
      out[i * 4] = static_cast<uint8_t>(state[i] >> 24);
      out[i * 4 + 1] = static_cast<uint8_t>(state[i] >> 16);
      out[i * 4 + 2] = static_cast<uint8_t>(state[i] >> 8);
      out[i * 4 + 3] = static_cast<uint8_t>(state[i]);
    }
    return out;
  }
};

[[nodiscard]] std::string to_hex(std::span<const uint8_t> bytes) {
  static constexpr char kDigits[] = "0123456789abcdef";
  std::string out;
  out.reserve(bytes.size() * 2);
  for (uint8_t b : bytes) {
    out.push_back(kDigits[b >> 4]);
    out.push_back(kDigits[b & 0xf]);
  }
  return out;
}

// ---------------------------------------------------------------------------
// Build identity
// ---------------------------------------------------------------------------

/// @brief GNU build identity of this shared object.
///
/// @details The linker derives it from object content, so any rebuild that
/// changes emitted bytes changes it without anyone having to remember. That is
/// the property the store depends on to never serve an entry produced by a
/// different translator. Returns empty when the note is absent, which disables
/// the store rather than falling back to a weaker identity.
[[nodiscard]] std::string read_own_build_id() {
  Dl_info info{};
  if (dladdr(reinterpret_cast<const void *>(&read_own_build_id), &info) == 0 ||
      info.dli_fbase == nullptr) {
    return {};
  }

  struct Query {
    const void *base;
    std::string result;
  } query{info.dli_fbase, {}};

  dl_iterate_phdr(
      [](struct dl_phdr_info *phdr_info, size_t, void *data) -> int {
        auto *q = static_cast<Query *>(data);
        if (reinterpret_cast<const void *>(phdr_info->dlpi_addr) != q->base)
          return 0;
        for (int i = 0; i < phdr_info->dlpi_phnum; ++i) {
          const ElfW(Phdr) &segment = phdr_info->dlpi_phdr[i];
          if (segment.p_type != PT_NOTE)
            continue;
          const auto *cursor =
              reinterpret_cast<const uint8_t *>(phdr_info->dlpi_addr + segment.p_vaddr);
          const uint8_t *end = cursor + segment.p_memsz;
          while (cursor + sizeof(ElfW(Nhdr)) <= end) {
            ElfW(Nhdr) note{};
            std::memcpy(&note, cursor, sizeof(note));
            const size_t name_size = (note.n_namesz + 3u) & ~3u;
            const size_t desc_size = (note.n_descsz + 3u) & ~3u;
            const uint8_t *name = cursor + sizeof(note);
            const uint8_t *desc = name + name_size;
            if (desc + desc_size > end)
              break;
            if (note.n_type == NT_GNU_BUILD_ID && note.n_namesz == 4 &&
                std::memcmp(name, "GNU", 4) == 0) {
              q->result = to_hex({desc, note.n_descsz});
              return 1;
            }
            cursor = desc + desc_size;
          }
        }
        return 1;
      },
      &query);
  return query.result;
}

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

[[nodiscard]] std::string runtime_dir() {
  // Mirrors rpc_default_runtime_dir(): an env var that is set but empty is
  // treated as unset, since returning "" would root every derived path at "/".
  if (const char *xdg = getenv("XDG_RUNTIME_DIR"); xdg != nullptr && *xdg != '\0')
    return std::string(xdg) + "/rocjitsu";
  return "/tmp/rocjitsu-" + std::to_string(getuid());
}

[[nodiscard]] bool make_directory_path(const std::string &path) {
  std::string partial;
  partial.reserve(path.size());
  for (size_t i = 1; i <= path.size(); ++i) {
    if (i != path.size() && path[i] != '/')
      continue;
    partial.assign(path, 0, i);
    if (mkdir(partial.c_str(), 0700) != 0 && errno != EEXIST)
      return false;
  }
  return true;
}

/// @brief Open @p path and confirm it is a private directory we own.
///
/// @details The preferred location is created private by the session manager,
/// but the fallback sits in a world-writable directory where another user can
/// pre-create the path. A directory that fails these checks is refused outright
/// rather than repaired: the wrong mode is a signal, and fixing it is precisely
/// what someone planting a directory would want.
[[nodiscard]] int open_verified_directory(const std::string &path) {
  const int fd = open(path.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return -1;
  struct stat info {};
  if (fstat(fd, &info) != 0 || !S_ISDIR(info.st_mode) || info.st_uid != geteuid() ||
      (info.st_mode & (S_IWGRP | S_IWOTH)) != 0) {
    close(fd);
    return -1;
  }
  return fd;
}

struct Entry {
  std::string name;
  uint64_t size = 0;
  time_t used = 0;
};

class Store {
public:
  static Store &instance() {
    static Store store;
    return store;
  }

  [[nodiscard]] bool available() {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    return dir_fd_ >= 0 && !build_id_.empty();
  }

  [[nodiscard]] std::string build_id() {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    return build_id_;
  }

  [[nodiscard]] std::vector<uint8_t> lookup(const std::string &key,
                                            const TranslationIdentity &identity);
  void store(const std::string &key, std::span<const uint8_t> object,
             const TranslationIdentity &identity);

#if defined(RJ_HOTSWAP_TEST_HOOKS)
  void set_root_for_test(const char *root) {
    std::lock_guard lock(mutex_);
    if (dir_fd_ >= 0)
      close(dir_fd_);
    dir_fd_ = -1;
    opened_ = false;
    hits_ = 0;
    root_override_ = root == nullptr ? std::string{} : std::string(root);
  }
  void set_capacity_for_test(uint64_t bytes) {
    std::lock_guard lock(mutex_);
    capacity_override_ = bytes;
  }
  void set_headroom_for_test(uint64_t bytes) {
    std::lock_guard lock(mutex_);
    headroom_override_ = bytes;
  }
  [[nodiscard]] uint64_t hits_for_test() {
    std::lock_guard lock(mutex_);
    return hits_;
  }
  [[nodiscard]] uint64_t size_for_test() {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    if (dir_fd_ < 0)
      return 0;
    uint64_t total = 0;
    for (const Entry &entry : scan_locked())
      total += entry.size;
    return total;
  }
#endif

private:
  void ensure_open_locked() {
    if (opened_)
      return;
    opened_ = true;
    build_id_ = read_own_build_id();
    if (build_id_.empty())
      return; // No stable identity: refuse to key on anything weaker.

    std::string base;
#if defined(RJ_HOTSWAP_TEST_HOOKS)
    base = root_override_.empty() ? runtime_dir() : root_override_;
#else
    base = runtime_dir();
#endif
    const std::string path = base + "/gfx1250-b0-a0/" + std::string(kSchemaDir);
    if (!make_directory_path(path))
      return;
    dir_fd_ = open_verified_directory(path);
    // Once per process, and the only moment at which a temporary left by a
    // process that died mid-write is unambiguously abandoned.
    sweep_abandoned_locked();
  }

  [[nodiscard]] std::vector<Entry> scan_locked() const;
  [[nodiscard]] uint64_t capacity_locked() const;
  [[nodiscard]] bool reserve_space_locked(uint64_t needed);
  void sweep_abandoned_locked() const;

  [[nodiscard]] uint64_t headroom_locked() const {
#if defined(RJ_HOTSWAP_TEST_HOOKS)
    if (headroom_override_ != 0)
      return headroom_override_;
#endif
    return kHeadroomBytes;
  }

  std::mutex mutex_;
  bool opened_ = false;
  int dir_fd_ = -1;
  std::string build_id_;
#if defined(RJ_HOTSWAP_TEST_HOOKS)
  std::string root_override_;
  uint64_t capacity_override_ = 0;
  uint64_t headroom_override_ = 0;
  uint64_t hits_ = 0;
#endif
};

std::vector<Entry> Store::scan_locked() const {
  std::vector<Entry> entries;
  if (dir_fd_ < 0)
    return entries;
  const int scan_fd = openat(dir_fd_, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (scan_fd < 0)
    return entries;
  DIR *dir = fdopendir(scan_fd);
  if (dir == nullptr) {
    close(scan_fd);
    return entries;
  }
  while (const dirent *item = readdir(dir)) {
    const std::string_view name(item->d_name);
    if (!name.ends_with(".obj"))
      continue;
    struct stat info {};
    if (fstatat(dir_fd_, item->d_name, &info, AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode))
      continue;
    entries.push_back({std::string(name), static_cast<uint64_t>(info.st_size), info.st_mtime});
  }
  closedir(dir);
  return entries;
}

void Store::sweep_abandoned_locked() const {
  if (dir_fd_ < 0)
    return;
  const int scan_fd = openat(dir_fd_, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (scan_fd < 0)
    return;
  DIR *dir = fdopendir(scan_fd);
  if (dir == nullptr) {
    close(scan_fd);
    return;
  }
  const time_t now = time(nullptr);
  while (const dirent *item = readdir(dir)) {
    if (std::string_view(item->d_name).find(".tmp.") == std::string_view::npos)
      continue;
    struct stat info {};
    if (fstatat(dir_fd_, item->d_name, &info, AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode))
      continue;
    if (now - info.st_mtime > kAbandonedTempSeconds)
      unlinkat(dir_fd_, item->d_name, 0);
  }
  closedir(dir);
}

uint64_t Store::capacity_locked() const {
  struct statvfs vfs {};
  if (dir_fd_ < 0 || fstatvfs(dir_fd_, &vfs) != 0)
    return 0;
  const uint64_t total = static_cast<uint64_t>(vfs.f_blocks) * vfs.f_frsize;
  const uint64_t free_now = static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize;
  // Checked ahead of any override, because the headroom floor is the one rule
  // that protects the rest of the runtime directory rather than the store.
  if (free_now < headroom_locked())
    return 0;
#if defined(RJ_HOTSWAP_TEST_HOOKS)
  if (capacity_override_ != 0)
    return capacity_override_;
#endif
  uint64_t cap = kAbsoluteCapacityBytes;
  cap = std::min(cap, total / 1000u * kCapacityPermilleOfTotal);
  cap = std::min(cap, free_now / 1000u * kCapacityPermilleOfFree);
  return cap;
}

bool Store::reserve_space_locked(uint64_t needed) {
  const uint64_t cap = capacity_locked();
  if (cap == 0 || needed > cap / kMaxEntryDivisor)
    return false;

  std::vector<Entry> entries = scan_locked();
  uint64_t used = 0;
  for (const Entry &entry : entries)
    used += entry.size;
  if (used + needed <= cap)
    return true;

  // Least recently used first. Reads refresh the timestamp, so this approximates
  // usage rather than age; if that refresh ever fails the order degrades to
  // insertion order, which is acceptable for uniformly sized entries.
  std::sort(entries.begin(), entries.end(),
            [](const Entry &a, const Entry &b) { return a.used < b.used; });
  const uint64_t target = cap / 1000u * kEvictTargetPermille;
  for (const Entry &entry : entries) {
    if (used + needed <= target)
      break;
    std::string manifest = entry.name.substr(0, entry.name.size() - 4) + ".man";
    unlinkat(dir_fd_, entry.name.c_str(), 0);
    unlinkat(dir_fd_, manifest.c_str(), 0);
    used -= std::min(used, entry.size);
  }
  return used + needed <= cap;
}

/// @brief Serialise the fields a reader re-checks before trusting an object.
[[nodiscard]] std::string build_manifest(const std::string &key, std::span<const uint8_t> object,
                                         const TranslationIdentity &identity,
                                         const std::string &build_id) {
  Sha256 hash;
  hash.update(object.data(), object.size());
  std::string text;
  text += std::string(kManifestMagic) + "\n";
  text += "key=" + key + "\n";
  text += "build=" + build_id + "\n";
  text += "epoch=" + std::to_string(kCacheEpoch) + "\n";
  text += "profile=" + std::to_string(identity.profile_id) + "\n";
  text += "in_rev=" + std::to_string(identity.input_revision) + "\n";
  text += "out_rev=" + std::to_string(identity.output_revision) + "\n";
  text += "isa=" + std::string(identity.target_isa) + "\n";
  text += "size=" + std::to_string(object.size()) + "\n";
  text += "sha=" + to_hex(hash.finish()) + "\n";
  return text;
}

[[nodiscard]] std::optional<std::string> manifest_field(std::string_view text,
                                                        std::string_view name) {
  std::string needle = "\n" + std::string(name) + "=";
  const size_t at = text.find(needle);
  if (at == std::string_view::npos)
    return std::nullopt;
  const size_t begin = at + needle.size();
  const size_t end = text.find('\n', begin);
  if (end == std::string_view::npos)
    return std::nullopt;
  return std::string(text.substr(begin, end - begin));
}

[[nodiscard]] std::optional<std::string> read_whole(int dir_fd, const std::string &name,
                                                    uint64_t limit) {
  const int fd = openat(dir_fd, name.c_str(), O_RDONLY | O_NOFOLLOW | O_CLOEXEC);
  if (fd < 0)
    return std::nullopt;
  struct stat info {};
  if (fstat(fd, &info) != 0 || !S_ISREG(info.st_mode) ||
      static_cast<uint64_t>(info.st_size) > limit) {
    close(fd);
    return std::nullopt;
  }
  std::string data;
  data.resize(static_cast<size_t>(info.st_size));
  size_t done = 0;
  while (done < data.size()) {
    const ssize_t got = read(fd, data.data() + done, data.size() - done);
    if (got <= 0) {
      close(fd);
      return std::nullopt;
    }
    done += static_cast<size_t>(got);
  }
  close(fd);
  return data;
}

[[nodiscard]] bool write_atomically(int dir_fd, const std::string &name, const void *data,
                                    size_t size) {
  static std::atomic<uint64_t> counter{0};
  const std::string temp = name + ".tmp." +
                           std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + "." +
                           std::to_string(getpid());
  const int fd =
      openat(dir_fd, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  if (fd < 0)
    return false;
  const auto *cursor = static_cast<const uint8_t *>(data);
  size_t left = size;
  while (left > 0) {
    const ssize_t put = write(fd, cursor, left);
    if (put <= 0) {
      close(fd);
      unlinkat(dir_fd, temp.c_str(), 0);
      return false;
    }
    cursor += put;
    left -= static_cast<size_t>(put);
  }
  if (fsync(fd) != 0) {
    close(fd);
    unlinkat(dir_fd, temp.c_str(), 0);
    return false;
  }
  close(fd);
  if (renameat(dir_fd, temp.c_str(), dir_fd, name.c_str()) != 0) {
    unlinkat(dir_fd, temp.c_str(), 0);
    return false;
  }
  return true;
}

std::vector<uint8_t> Store::lookup(const std::string &key, const TranslationIdentity &identity) {
  std::lock_guard lock(mutex_);
  ensure_open_locked();
  if (dir_fd_ < 0)
    return {};

  const std::string manifest_name = key + ".man";
  const std::string object_name = key + ".obj";
  constexpr uint64_t kManifestLimit = 4096;
  const std::optional<std::string> manifest = read_whole(dir_fd_, manifest_name, kManifestLimit);
  if (!manifest || !manifest->starts_with(kManifestMagic))
    return {};

  // The key already covers these, so a mismatch means a digest collision or a
  // file left behind by a defect. Either way the object is not ours to use.
  const auto recorded_size = manifest_field(*manifest, "size");
  const auto recorded_sha = manifest_field(*manifest, "sha");
  if (!recorded_size || !recorded_sha)
    return {};
  const std::pair<std::string_view, std::string> expected[] = {
      {"key", key},
      {"build", build_id_},
      {"epoch", std::to_string(kCacheEpoch)},
      {"profile", std::to_string(identity.profile_id)},
      {"in_rev", std::to_string(identity.input_revision)},
      {"out_rev", std::to_string(identity.output_revision)},
      {"isa", std::string(identity.target_isa)},
  };
  for (const auto &[name, want] : expected) {
    const auto got = manifest_field(*manifest, name);
    if (!got || *got != want)
      return {};
  }

  const std::optional<std::string> object = read_whole(dir_fd_, object_name, kMaxObjectBytes);
  if (!object || std::to_string(object->size()) != *recorded_size)
    return {};

  Sha256 hash;
  hash.update(object->data(), object->size());
  if (to_hex(hash.finish()) != *recorded_sha)
    return {};

  // Refresh for the eviction order. Failure only costs ordering accuracy.
  const timespec now[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
  (void)utimensat(dir_fd_, object_name.c_str(), now, AT_SYMLINK_NOFOLLOW);

#if defined(RJ_HOTSWAP_TEST_HOOKS)
  ++hits_;
#endif
  return std::vector<uint8_t>(object->begin(), object->end());
}

void Store::store(const std::string &key, std::span<const uint8_t> object,
                  const TranslationIdentity &identity) {
  std::lock_guard lock(mutex_);
  ensure_open_locked();
  if (dir_fd_ < 0 || object.empty())
    return;

  const std::string manifest = build_manifest(key, object, identity, build_id_);
  if (!reserve_space_locked(object.size() + manifest.size()))
    return;

  // Manifest first: a visible object must always have something to verify
  // against. A manifest with no object simply reads as a miss.
  if (!write_atomically(dir_fd_, key + ".man", manifest.data(), manifest.size()))
    return;
  if (!write_atomically(dir_fd_, key + ".obj", object.data(), object.size()))
    unlinkat(dir_fd_, (key + ".man").c_str(), 0);
}

} // namespace

CacheKey cache_key_for(std::span<const uint8_t> source, const TranslationIdentity &identity) {
  CacheKey key;
  if (source.empty())
    return key;
  const std::string build_id = Store::instance().build_id();
  if (build_id.empty() || !Store::instance().available())
    return key;

  Sha256 hash;
  hash.update(kKeyDomain);
  hash.update(build_id);
  const uint32_t fields[] = {kCacheEpoch, identity.profile_id, identity.input_revision,
                             identity.output_revision};
  hash.update(fields, sizeof(fields));
  hash.update(identity.target_isa);
  const uint64_t source_size = source.size();
  hash.update(&source_size, sizeof(source_size));
  hash.update(source.data(), source.size());
  key.digest = hash.finish();
  key.valid = true;
  return key;
}

std::vector<uint8_t> cache_lookup(const CacheKey &key, const TranslationIdentity &identity) {
  if (!key.valid)
    return {};
  return Store::instance().lookup(to_hex(key.digest), identity);
}

void cache_store(const CacheKey &key, std::span<const uint8_t> translated,
                 const TranslationIdentity &identity) {
  if (!key.valid || translated.empty())
    return;
  Store::instance().store(to_hex(key.digest), translated, identity);
}

#if defined(RJ_HOTSWAP_TEST_HOOKS)
void set_cache_root_for_test(const char *root) { Store::instance().set_root_for_test(root); }
uint64_t cache_size_for_test() { return Store::instance().size_for_test(); }
uint64_t cache_hits_for_test() { return Store::instance().hits_for_test(); }
void set_cache_capacity_for_test(uint64_t bytes) { Store::instance().set_capacity_for_test(bytes); }
void set_cache_headroom_for_test(uint64_t bytes) { Store::instance().set_headroom_for_test(bytes); }
#endif

} // namespace rocjitsu::hotswap
