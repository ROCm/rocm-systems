// Copyright (c) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/// @file dbt/translation_store.cpp
/// @brief On-disk store for translated code objects.

#include "rocjitsu/code/dbt/translation_store.h"

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <mutex>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>

#include <dirent.h>
#include <dlfcn.h>
#include <elf.h>
#include <fcntl.h>
#include <link.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/types.h>
#include <unistd.h>

namespace rocjitsu {
namespace {

/// @brief Stable identity for caches distributed with their consumer.
/// @details Changing the portable cache contract requires changing this nonce
/// or the cache epoch. It is deliberately not a wildcard: manifests and keys
/// still agree on one explicit producer identity.
constexpr std::string_view kPortablePrebuiltNonce = "unversioned-prebuilt-v1";

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
constexpr uint64_t operator""_MiB(unsigned long long value) { return value << 20; }
constexpr uint64_t operator""_GiB(unsigned long long value) { return value << 30; }

/// @brief Below this much free space the store stops accepting writes.
/// @details The runtime directory is shared with the daemon socket and config,
/// and is memory-backed, so exhausting it breaks more than caching.
constexpr uint64_t kHeadroomBytes = 64_MiB;

/// @brief Largest object the store will read or write.
///
/// @details The store exists to hold exactly the objects too large to be worth
/// translating repeatedly -- a device library of a few hundred megabytes takes
/// minutes -- so this is not a capacity policy but a sanity bound, so a corrupt
/// size never turns into an allocation of that size.
constexpr uint64_t kMaxObjectBytes = 4_GiB;

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

/// @brief GNU build identity of the object holding this code.
///
/// @details The linker derives it from object content, so any rebuild that
/// changes emitted bytes changes it without anyone having to remember. That is
/// the property the store depends on to never serve an entry produced by a
/// different translator. Returns empty when the note is absent, which disables
/// the store rather than falling back to a weaker identity.
///
/// The object is identified by the loaded segment that contains @p address
/// rather than by comparing load bases, because a load base only equals the
/// iterator's reported address for a position-independent object. Whether a
/// consumer is a shared library or a non-relocatable executable is not something
/// this component should care about.
///
/// Probing by a caller-supplied address rather than by this function's own is
/// what lets the key name the TRANSLATOR instead of whichever binary happens to
/// embed the store. Two programs calling the same translator derive the same id
/// and so agree on keys; one linking a different build of it derives a different
/// id and simply misses.
[[nodiscard]] std::string read_build_id_of(const void *address) {
  struct Query {
    const uint8_t *self;
    std::string result;
  } query{static_cast<const uint8_t *>(address), {}};

  dl_iterate_phdr(
      [](struct dl_phdr_info *phdr_info, size_t, void *data) -> int {
        auto *q = static_cast<Query *>(data);
        bool holds_self = false;
        for (int i = 0; i < phdr_info->dlpi_phnum && !holds_self; ++i) {
          const ElfW(Phdr) &segment = phdr_info->dlpi_phdr[i];
          if (segment.p_type != PT_LOAD)
            continue;
          const auto *begin =
              reinterpret_cast<const uint8_t *>(phdr_info->dlpi_addr + segment.p_vaddr);
          holds_self = q->self >= begin && q->self < begin + segment.p_memsz;
        }
        if (!holds_self)
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

} // namespace

std::string shared_translation_root(const void *address) {
  Dl_info info{};
  if (dladdr(address, &info) == 0 || info.dli_fname == nullptr)
    return {};
  // dladdr() preserves the spelling used by the loader, including relative
  // components. The secure descriptor walk intentionally rejects those, so
  // derive the installed root from the resolved module path instead.
  char *resolved = realpath(info.dli_fname, nullptr);
  if (resolved == nullptr)
    return {};
  const std::string resolved_path(resolved);
  free(resolved);
  const std::string_view path(resolved_path);
  const size_t file_at = path.rfind('/');
  if (file_at == std::string_view::npos)
    return {}; // A bare name means the loader resolved it by search; no prefix.
  const std::string_view library_dir = path.substr(0, file_at);
  const size_t prefix_at = library_dir.rfind('/');
  if (prefix_at == std::string_view::npos)
    return {};

  // Only an install layout gets a root. Stripping two components unconditionally
  // means a module loaded from anywhere at all names a sibling `share` -- a copy
  // under /tmp would derive /tmp/share/rocjitsu/translations, inventing a cache
  // location in a world-writable directory purely because that is where the
  // library happened to sit. Requiring the containing directory to be a library
  // directory keeps the derivation to the case it was designed for; anything
  // else reports no root, and the caller treats that as "not pre-translated".
  const std::string_view library_dir_name = library_dir.substr(prefix_at + 1);
  const bool is_library_dir = library_dir_name == "lib" || library_dir_name == "lib64" ||
                              // Debian/Ubuntu multiarch, e.g. lib/x86_64-linux-gnu, whose prefix is
                              // one level further up than the two-component strip above assumes.
                              library_dir_name.find('-') != std::string_view::npos;
  if (!is_library_dir)
    return {};
  std::string_view prefix = library_dir.substr(0, prefix_at);
  if (library_dir_name.find('-') != std::string_view::npos) {
    const size_t multiarch_prefix_at = prefix.rfind('/');
    if (multiarch_prefix_at == std::string_view::npos ||
        prefix.substr(multiarch_prefix_at + 1) != "lib")
      return {};
    prefix = prefix.substr(0, multiarch_prefix_at);
  }
  return std::string(prefix) + "/share/rocjitsu/translations";
}

namespace {

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

/// @brief Create @p path and every missing parent with mode @p mode.
///
/// @details The mode is explicit because the two tiers need opposite things. The
/// session tree is private to one user. The shared tree is typically written by
/// root during an image build and read later by an unprivileged process, so a
/// private mode would produce a tree nobody can use.
///
/// mkdir applies the umask, so a component this call creates has its mode set
/// again afterwards. A component that already existed is left exactly as found:
/// correcting one would quietly repair the very thing the directory checks below
/// exist to notice, and repairing it is precisely what someone who planted it
/// would want.
[[nodiscard]] std::vector<std::string> path_components(std::string_view path) {
  std::vector<std::string> components;
  size_t at = 0;
  while (at < path.size()) {
    const size_t next = path.find('/', at);
    const std::string_view component =
        path.substr(at, next == std::string_view::npos ? std::string_view::npos : next - at);
    if (!component.empty())
      components.emplace_back(component);
    if (next == std::string_view::npos)
      break;
    at = next + 1;
  }
  return components;
}

/// @brief Whether a store root is an absolute, descended path.
/// @details The descriptor walk begins at `/`, so accepting a relative root
/// would silently reinterpret it as absolute. Requiring at least one ordinary
/// component also rejects `/` before trust-boundary index arithmetic.
[[nodiscard]] bool is_descended_absolute_path(std::string_view path) {
  if (path.size() < 2 || path.front() != '/' || path.find('\0') != std::string_view::npos)
    return false;
  const auto components = path_components(path);
  return !components.empty() &&
         std::none_of(components.begin(), components.end(), [](const std::string &component) {
           return component == "." || component == "..";
         });
}

/// @brief Which owners a directory the store relies on may have.
enum class DirectoryTrust {
  /// Sole ownership by the caller. The session tier is writable by the process
  /// that checks it, and exclusivity is what makes its own writes safe to trust.
  kOwnedByCaller,
  /// Owner root or the caller -- the standard trusted-system-directory test. The
  /// shared tier is read at run time and written only by an installer, so the
  /// question is not exclusivity but whether anyone outside the trust boundary
  /// could have placed an entry.
  kOwnedByRootOrCaller,
};

/// @brief Whether @p info names a directory an acceptable user owns.
///
/// @details Asked of the directories the store is handed rather than the ones it
/// creates. The fallback root lives in a world-writable directory, so another
/// user can pre-create it, and ownership is what rules that out. Their MODE is
/// not the store's business: $XDG_RUNTIME_DIR/rocjitsu is made by the daemon and
/// is group-writable, and demanding otherwise would disable the session tier
/// wherever the daemon has run.
[[nodiscard]] bool directory_owner_is_trusted(const struct stat &info, DirectoryTrust trust) {
  if (!S_ISDIR(info.st_mode))
    return false;
  if (trust == DirectoryTrust::kOwnedByCaller)
    return info.st_uid == geteuid();
  return info.st_uid == 0 || info.st_uid == geteuid();
}

/// @brief Whether @p info names a directory only its owner can write.
///
/// @details Asked of the directories holding entries, which the store creates
/// itself and whose contents it is about to believe. Group or other write there
/// means someone outside the trust boundary could have placed an entry.
[[nodiscard]] bool directory_is_trusted(const struct stat &info, DirectoryTrust trust) {
  return directory_owner_is_trusted(info, trust) && (info.st_mode & (S_IWGRP | S_IWOTH)) == 0;
}

/// @brief Walk to @p path one component at a time, following no symlink.
///
/// @details Opening the assembled path in one call and checking only what it
/// lands on proves nothing about how it got there: any component along the way
/// can be a symlink, and the check then describes the target rather than the
/// place the caller named. A domain pre-created as a symlink was enough to send
/// the writer outside the root it was given and have the reader find the entry
/// there, with every check passing.
///
/// So each component is opened relative to its already-verified parent with
/// O_NOFOLLOW, and the checks apply to that descriptor. A symlink anywhere is
/// refused instead of followed. The final descriptor is then held for the
/// process lifetime, so nothing that happens to the path afterwards changes
/// which directory is used.
///
/// @param create_mode Non-zero creates missing components with that mode. mkdir
///        applies the umask, so a component this call creates has its mode set
///        again through its own descriptor. A component that already existed is
///        left exactly as found: correcting one would quietly repair the very
///        thing these checks exist to notice.
/// @param trusted_from Index of the first component the trust policy applies to.
///        The directories above a store root are the system's -- /tmp is
///        world-writable by design and would fail any ownership test -- so they
///        are walked without following symlinks but are not otherwise judged.
[[nodiscard]] int open_descended_directory(const std::string &path, mode_t create_mode,
                                           DirectoryTrust trust, size_t owner_checked_from,
                                           size_t trusted_from) {
  const std::vector<std::string> components = path_components(path);
  if (components.empty())
    return -1;

  int parent = open("/", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  if (parent < 0)
    return -1;

  for (size_t index = 0; index < components.size(); ++index) {
    const std::string &component = components[index];
    // "." and ".." would step outside the chain of descriptors this walk has
    // verified, which is the whole point of walking it.
    if (component == "." || component == "..") {
      close(parent);
      return -1;
    }
    const bool created = create_mode != 0 && mkdirat(parent, component.c_str(), create_mode) == 0;
    if (!created && create_mode != 0 && errno != EEXIST) {
      close(parent);
      return -1;
    }
    const int child =
        openat(parent, component.c_str(), O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
    close(parent);
    if (child < 0)
      return -1;
    if (created)
      (void)fchmod(child, create_mode);
    if (index >= owner_checked_from) {
      struct stat info {};
      const bool ok = fstat(child, &info) == 0 &&
                      (index >= trusted_from ? directory_is_trusted(info, trust)
                                             : directory_owner_is_trusted(info, trust));
      if (!ok) {
        close(child);
        return -1;
      }
    }
    parent = child;
  }
  return parent;
}

/// @brief Holds the domain directory's lock for a write.
///
/// @details The in-process mutex covers one handle, which is the wrong scope for
/// a capacity decision: separate processes each scan the same pre-write usage,
/// each concludes there is room, and each writes. Forty-eight writers with
/// distinct keys left 194592 bytes under a 65536-byte cap, and no same-key test
/// can expose it because the collision is in the arithmetic rather than the
/// bytes.
///
/// The scan, the eviction it drives and the publication that consumes the result
/// therefore happen under one lock on the directory itself. Readers do not take
/// it: publication is already atomic through rename, so a lookup either sees a
/// complete entry or misses, and making hits wait behind a writer would trade a
/// real cost for no benefit.
///
/// An advisory lock is enough because every writer is this code. It is released
/// by the kernel if the holder dies, which matters more here than in a
/// longer-lived design: a killed pre-translation run must not wedge the store
/// for everyone after it.
class DirectoryWriteLock {
public:
  enum class Mode { kBlocking, kNonBlocking };

  explicit DirectoryWriteLock(int dir_fd, Mode mode = Mode::kBlocking) : dir_fd_(dir_fd) {
    const int operation = LOCK_EX | (mode == Mode::kNonBlocking ? LOCK_NB : 0);
    held_ = dir_fd_ >= 0 && flock(dir_fd_, operation) == 0;
  }
  ~DirectoryWriteLock() {
    if (held_)
      (void)flock(dir_fd_, LOCK_UN);
  }
  DirectoryWriteLock(const DirectoryWriteLock &) = delete;
  DirectoryWriteLock &operator=(const DirectoryWriteLock &) = delete;

  /// @details A store that could not take the lock declines the write rather
  /// than proceeding unsynchronised, which is the same best-effort outcome as
  /// any other refusal.
  [[nodiscard]] bool held() const { return held_; }

private:
  int dir_fd_;
  bool held_ = false;
};

struct Entry {
  /// Key without the .obj/.man suffix, so one entry covers the pair.
  std::string name;
  /// Object plus manifest, which is what the entry actually occupies.
  uint64_t size = 0;
  time_t used = 0;
  bool has_object = false;
};

/// @brief True when @p domain names exactly one directory beneath the root.
///
/// @details The domain reaches a path, so a caller that passed a separator or a
/// traversal component could place entries outside the verified directory. It is
/// a compile-time constant at every call site today, which is why rejecting it
/// outright is preferable to sanitising it.
[[nodiscard]] bool is_single_path_component(std::string_view domain) {
  // A NUL is as much a separator problem as a slash: the C path APIs stop at it,
  // so "pair\0a" and "pair\0b" are distinct string_views that name one directory
  // and silently share entries, which is exactly the isolation the domain exists
  // to provide.
  return !domain.empty() && domain != "." && domain != ".." &&
         domain.find('/') == std::string_view::npos && domain.find('\0') == std::string_view::npos;
}

} // namespace

struct TranslationStore::Impl {
  Impl(std::string_view domain, const void *translator, std::string_view root, Access access,
       KeyMode key_mode)
      : domain_(domain), translator_(translator), root_(root), access_(access),
        key_mode_(key_mode) {}

  ~Impl() {
    if (dir_fd_ >= 0)
      close(dir_fd_);
  }

  [[nodiscard]] bool available() {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    return dir_fd_ >= 0 && !build_nonce_.empty();
  }

  [[nodiscard]] std::string build_nonce() {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    return build_nonce_;
  }

  [[nodiscard]] std::vector<uint8_t> lookup(const std::string &key,
                                            const TranslationIdentity &identity);
  void store(const std::string &key, std::span<const uint8_t> object,
             const TranslationIdentity &identity);

#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
  void set_root_for_test(const char *root) {
    std::lock_guard lock(mutex_);
    if (dir_fd_ >= 0)
      close(dir_fd_);
    dir_fd_ = -1;
    opened_ = false;
    hits_ = 0;
    root_override_ = root == nullptr ? std::string{} : std::string(root);
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

  [[nodiscard]] bool writable() const { return access_ == Access::kReadWrite; }

  [[nodiscard]] uint64_t max_object_bytes() const { return kMaxObjectBytes; }

  void ensure_open_locked() {
    if (opened_)
      return;
    opened_ = true;
    if (!is_single_path_component(domain_))
      return;
    if (key_mode_ == KeyMode::kPortablePrebuilt) {
      build_nonce_ = kPortablePrebuiltNonce;
    } else {
      build_nonce_ = read_build_id_of(translator_);
      if (build_nonce_.empty())
        return; // No stable identity: refuse to key on anything weaker.
    }

    std::string base;
#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
    base = root_override_;
#endif
    if (base.empty()) {
      // An empty root disables the store rather than defaulting somewhere: a
      // caller that could not locate its install has not told us where to look.
      if (root_.empty())
        return;
      base = root_;
    }
    if (!is_descended_absolute_path(base))
      return;
    const std::string path = base + "/" + domain_ + "/" + std::string(kSchemaDir);
    // The trust policy starts at the store's own root. Everything above it
    // belongs to the system -- /tmp is world-writable by design, and judging it
    // by the store's rules would refuse every fallback location -- so those are
    // walked without following symlinks but are not otherwise judged. From the
    // root down, every component must pass, because those are the directories
    // whose contents this store is about to believe.
    // Ownership is checked from the store root down, because a root someone else
    // owns is a root someone else controls -- that is what protects the /tmp
    // fallback. The stricter no-group-or-other-write rule starts one level
    // lower, at the directories this store creates and reads entries from: the
    // root itself may be the daemon's, and it is group-writable.
    const size_t owner_checked_from = path_components(base).size() - 1;
    const size_t trusted_from = path_components(base).size();
    // A read-only store never creates its tree. An absent shared tier means
    // nobody pre-translated, which is a miss and not something to repair -- and
    // a directory this process created could not be trusted by the next one
    // anyway, since it would fail its own ownership test under a different user.
    const mode_t create_mode = writable() ? 0755 : 0;
    dir_fd_ = open_descended_directory(path, create_mode, DirectoryTrust::kOwnedByRootOrCaller,
                                       owner_checked_from, trusted_from);
    if (writable() && dir_fd_ >= 0) {
      // Once per process, and the only moment at which a temporary left by a
      // process that died mid-write is unambiguously abandoned.
      // Opening and lookup must never wait behind a publisher. The sweep is
      // opportunistic; losing this race only leaves old files for the next
      // process to reclaim.
      const DirectoryWriteLock directory_lock(dir_fd_, DirectoryWriteLock::Mode::kNonBlocking);
      if (directory_lock.held())
        sweep_abandoned_locked();
    }
  }

  [[nodiscard, maybe_unused]] std::vector<Entry> scan_locked() const;
  [[nodiscard]] bool reserve_space_locked(uint64_t needed);
  void sweep_abandoned_locked() const;

  [[nodiscard]] uint64_t headroom_locked() const {
#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
    if (headroom_override_ != 0)
      return headroom_override_;
#endif
    return kHeadroomBytes;
  }

  const std::string domain_;
  // An address inside the translator whose output these entries are. Exact mode
  // resolves its module's build ID; portable mode deliberately does not.
  const void *const translator_;
  // Root of the tree, supplied by the caller from its own install location.
  const std::string root_;
  const Access access_;
  const KeyMode key_mode_;
  std::mutex mutex_;
  bool opened_ = false;
  int dir_fd_ = -1;
  std::string build_nonce_;
#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
  std::string root_override_;
  uint64_t headroom_override_ = 0;
  uint64_t hits_ = 0;
#endif
};

std::vector<Entry> TranslationStore::Impl::scan_locked() const {
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
  // An entry is the object AND its manifest. Counting only the object lets the
  // manifests accumulate outside the cap entirely -- 400 one-byte objects under
  // an 8192-byte cap left 100400 bytes on disk while the store reported 400 --
  // and makes eviction give back less than it appears to.
  //
  // A manifest whose object is missing is counted too, under its own name. That
  // is what a process killed between the two renames leaves behind: it belongs
  // to no entry, is never a lookup hit, and nothing would otherwise reclaim it.
  std::unordered_map<std::string, Entry> by_key;
  while (const dirent *item = readdir(dir)) {
    const std::string_view name(item->d_name);
    const bool is_object = name.ends_with(".obj");
    if (!is_object && !name.ends_with(".man"))
      continue;
    struct stat info {};
    if (fstatat(dir_fd_, item->d_name, &info, AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode))
      continue;
    std::string key(name.substr(0, name.size() - 4));
    Entry &entry = by_key[key];
    if (entry.name.empty())
      entry.name = key;
    entry.size += static_cast<uint64_t>(info.st_size);
    // The object's timestamp is the one lookups refresh, so it decides eviction
    // order whenever there is an object at all.
    if (is_object || entry.used == 0) {
      entry.used = info.st_mtime;
      entry.has_object = entry.has_object || is_object;
    }
  }
  closedir(dir);
  entries.reserve(by_key.size());
  for (auto &[key, entry] : by_key)
    entries.push_back(std::move(entry));
  return entries;
}

void TranslationStore::Impl::sweep_abandoned_locked() const {
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
    const std::string_view name(item->d_name);
    // A manifest published without its object is the other way a writer can die
    // mid-entry: the manifest is renamed first, so a kill between the two leaves
    // one that can never satisfy a lookup and that eviction only ever reaches
    // under cap pressure. The same age threshold applies for the same reason --
    // below it, the writer may simply still be working.
    const bool orphan_manifest =
        name.ends_with(".man") &&
        faccessat(dir_fd_, (std::string(name.substr(0, name.size() - 4)) + ".obj").c_str(), F_OK,
                  0) != 0;
    if (!orphan_manifest && name.find(".tmp.") == std::string_view::npos)
      continue;
    struct stat info {};
    if (fstatat(dir_fd_, item->d_name, &info, AT_SYMLINK_NOFOLLOW) != 0 || !S_ISREG(info.st_mode))
      continue;
    if (now - info.st_mtime > kAbandonedTempSeconds)
      unlinkat(dir_fd_, item->d_name, 0);
  }
  closedir(dir);
}

bool TranslationStore::Impl::reserve_space_locked(uint64_t needed) {
  // The store is curated rather than cached: someone decided which objects
  // belong in it, so evicting one to fit another would silently undo that
  // decision and leave a pre-translation pass with unpredictable output. It gets
  // a per-entry ceiling and the free-space floor, and nothing else.
  struct statvfs vfs {};
  if (dir_fd_ < 0 || fstatvfs(dir_fd_, &vfs) != 0)
    return false;
  const uint64_t free_now = static_cast<uint64_t>(vfs.f_bavail) * vfs.f_frsize;
  return needed <= kMaxObjectBytes && free_now >= needed + headroom_locked();
}

namespace {

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

template <typename Buffer>
[[nodiscard]] std::optional<Buffer> read_whole(int dir_fd, const std::string &name,
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
  Buffer data;
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
                                    size_t size, mode_t mode) {
  static std::atomic<uint64_t> counter{0};
  const std::string temp = name + ".tmp." +
                           std::to_string(counter.fetch_add(1, std::memory_order_relaxed)) + "." +
                           std::to_string(getpid());
  const int fd =
      openat(dir_fd, temp.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
  if (fd < 0)
    return false;
  // open() applies the umask, which for a shared entry written during an image
  // build would leave a file the eventual reader cannot open. Set the mode on the
  // descriptor instead, before the rename makes the name visible.
  if (fchmod(fd, mode) != 0) {
    close(fd);
    unlinkat(dir_fd, temp.c_str(), 0);
    return false;
  }
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

} // namespace

std::vector<uint8_t> TranslationStore::Impl::lookup(const std::string &key,
                                                    const TranslationIdentity &identity) {
  // Only opening needs the lock. Holding it across the file reads and the
  // digest made independent hits serialise against each other -- repeated
  // lookups of one 8 MiB entry went from 320 ms on one thread to 2561 ms on
  // eight -- and none of that work touches shared state. Once the descriptor
  // and the build identity are settled they do not change again, and a
  // concurrent publication is already invisible: rename is atomic, so a reader
  // sees the previous entry or the new one, and anything else fails the
  // manifest and digest checks and degrades to a miss.
  int dir_fd = -1;
  std::string build_nonce;
  {
    std::lock_guard lock(mutex_);
    ensure_open_locked();
    dir_fd = dir_fd_;
    build_nonce = build_nonce_;
  }
  if (dir_fd < 0)
    return {};

  const std::string manifest_name = key + ".man";
  const std::string object_name = key + ".obj";
  constexpr uint64_t kManifestLimit = 4096;
  const std::optional<std::string> manifest =
      read_whole<std::string>(dir_fd, manifest_name, kManifestLimit);
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
      {"build", build_nonce},
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

  std::optional<std::vector<uint8_t>> object =
      read_whole<std::vector<uint8_t>>(dir_fd, object_name, max_object_bytes());
  if (!object || std::to_string(object->size()) != *recorded_size)
    return {};

  Sha256 hash;
  hash.update(object->data(), object->size());
  if (to_hex(hash.finish()) != *recorded_sha)
    return {};

  // Refresh for the eviction order. Failure only costs ordering accuracy, and
  // the shared tier never evicts, so it has no order to keep.
  if (writable()) {
    const timespec now[2] = {{0, UTIME_NOW}, {0, UTIME_NOW}};
    (void)utimensat(dir_fd, object_name.c_str(), now, AT_SYMLINK_NOFOLLOW);
  }

#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
  {
    std::lock_guard lock(mutex_);
    ++hits_;
  }
#endif
  return std::move(*object);
}

void TranslationStore::Impl::store(const std::string &key, std::span<const uint8_t> object,
                                   const TranslationIdentity &identity) {
  std::lock_guard lock(mutex_);
  if (!writable())
    return;
  ensure_open_locked();
  if (dir_fd_ < 0 || object.empty())
    return;

  // Everything from here to publication is one critical section across
  // processes, because the reservation below is only meaningful if nobody else
  // writes between the scan and the rename that consumes its result.
  const DirectoryWriteLock directory_lock(dir_fd_);
  if (!directory_lock.held())
    return;

  // Refuse what cannot fit before hashing it. build_manifest() digests the whole
  // object, so a refused 64 MiB entry spent 234 ms on a SHA-256 whose result was
  // discarded -- with the directory locked and the mutex held, so every other
  // writer and every opening reader waited behind it. The bound only has to be
  // no smaller than the real manifest; the reservation below still uses the
  // exact size.
  // The object alone already exceeding the per-entry share is enough to refuse:
  // the manifest only makes it larger, so this can never reject something the
  // exact check below would have accepted. Anything subtler belongs to that
  // check, which still runs with the real size.
  if (object.size() > kMaxObjectBytes)
    return;

  const std::string manifest = build_manifest(key, object, identity, build_nonce_);
  if (!reserve_space_locked(object.size() + manifest.size()))
    return;

  // Entries must be readable by whoever eventually runs; the directory's
  // ownership, not the file's mode, is what keeps them trustworthy.
  constexpr mode_t mode = 0644;
  // Manifest first: a visible object must always have something to verify
  // against. A manifest with no object simply reads as a miss.
  if (!write_atomically(dir_fd_, key + ".man", manifest.data(), manifest.size(), mode))
    return;
  if (!write_atomically(dir_fd_, key + ".obj", object.data(), object.size(), mode))
    unlinkat(dir_fd_, (key + ".man").c_str(), 0);
}

TranslationStore::TranslationStore(std::string_view domain, const void *translator,
                                   std::string_view root, Access access, KeyMode key_mode)
    : impl_(std::make_unique<Impl>(domain, translator, root, access, key_mode)) {}

bool TranslationStore::available() { return impl_->available(); }

TranslationStore::~TranslationStore() = default;

// Every public operation is contained here rather than at each internal step.
// The documented contract is best effort -- a caller translates instead -- but
// these paths allocate strings, vectors and file buffers, and this component is
// reached from an HSA load callback. A std::bad_alloc escaping key_for() fails
// that load outright, turning a cache that could not answer into a program that
// cannot run. Returning an invalid key, a miss, or a no-op keeps the promise.
CacheKey TranslationStore::key_for(std::span<const uint8_t> source,
                                   const TranslationIdentity &identity) try {
  CacheKey key;
  if (source.empty())
    return key;
  const std::string build_nonce = impl_->build_nonce();
  if (build_nonce.empty() || !impl_->available())
    return key;

  // The manifest is line oriented, so a newline here forges an earlier size= or
  // sha= field. The object still gets written, every later lookup rejects it,
  // and the entry is retried forever: a write-only cache slot. The one identity
  // this hook presents is safe, but the type accepts any string_view.
  if (identity.target_isa.find('\n') != std::string_view::npos ||
      identity.target_isa.find('\r') != std::string_view::npos)
    return key;

  Sha256 hash;
  hash.update(kKeyDomain);
  hash.update(build_nonce);
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
} catch (...) {
  return CacheKey{};
}

std::vector<uint8_t> TranslationStore::lookup(const CacheKey &key,
                                              const TranslationIdentity &identity) try {
  if (!key.valid)
    return {};
  return impl_->lookup(to_hex(key.digest), identity);
} catch (...) {
  return {};
}

void TranslationStore::store(const CacheKey &key, std::span<const uint8_t> translated,
                             const TranslationIdentity &identity) noexcept try {
  if (!key.valid || translated.empty())
    return;
  impl_->store(to_hex(key.digest), translated, identity);
} catch (...) {
  // Declared never to throw, and now actually is.
}

#if defined(RJ_TRANSLATION_STORE_TEST_HOOKS)
void TranslationStore::set_root_for_test(const char *root) { impl_->set_root_for_test(root); }
uint64_t TranslationStore::size_for_test() { return impl_->size_for_test(); }
uint64_t TranslationStore::hits_for_test() { return impl_->hits_for_test(); }
void TranslationStore::set_headroom_for_test(uint64_t bytes) {
  impl_->set_headroom_for_test(bytes);
}
#endif

} // namespace rocjitsu
