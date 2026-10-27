/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#include "hrr_archive_sign.h"

#include "monocypher-ed25519.h"
#include "rocm/sha2/sha256.h"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

#if !defined(_WIN32)
#include <dirent.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace hrr_sign {
namespace {

std::string hex(const uint8_t* p, size_t n) {
  static const char* d = "0123456789abcdef";
  std::string s;
  for (size_t i = 0; i < n; ++i) {
    s += d[p[i] >> 4];
    s += d[p[i] & 15];
  }
  return s;
}

bool unhex(const std::string& s, uint8_t* out, size_t n) {
  if (s.size() != 2 * n) return false;
  for (size_t i = 0; i < n; ++i) {
    unsigned v = 0;
    for (int k = 0; k < 2; ++k) {
      const char c = s[2 * i + k];
      unsigned x;
      if (c >= '0' && c <= '9') x = c - '0';
      else if (c >= 'a' && c <= 'f') x = c - 'a' + 10;
      else if (c >= 'A' && c <= 'F') x = c - 'A' + 10;
      else return false;
      v = v * 16 + x;
    }
    out[i] = static_cast<uint8_t>(v);
  }
  return true;
}

#if defined(_WIN32)
bool sha256_file(const fs::path& p, std::string* hex_out) {
  FILE* f = fopen(p.string().c_str(), "rb");
  if (!f) return false;
  rocm::sha2::sha256 h;
  std::vector<uint8_t> buf(1u << 20);
  size_t n;
  while ((n = fread(buf.data(), 1, buf.size(), f)) > 0) h.update(buf.data(), n);
  const bool ok = !ferror(f);
  fclose(f);
  if (!ok) return false;
  *hex_out = h.hexdigest();
  return true;
}
#endif

bool read_all(const fs::path& p, uintmax_t max, std::string* out) {
  std::error_code ec;
  const uintmax_t size = fs::file_size(p, ec);
  if (ec || size > max) return false;
  FILE* f = fopen(p.string().c_str(), "rb");
  if (!f) return false;
  out->resize(size);
  const bool ok = fread(out->data(), 1, size, f) == size;
  fclose(f);
  return ok;
}

bool is_sign_file(const std::string& rel) { return rel == kManifestName || rel == kSignatureName; }

#if defined(_WIN32)
// Relative paths (generic form) of the files in the archive, sorted. Fails on
// anything that is not a regular file or directory, naming it.
bool list_files(const fs::path& dir, std::vector<std::string>* out, std::string* err) {
  std::error_code ec;
  fs::recursive_directory_iterator it(dir, fs::directory_options::none, ec), end;
  for (; !ec && it != end; it.increment(ec)) {
    const std::string rel = it->path().lexically_relative(dir).generic_string();
    const fs::file_status st = it->symlink_status(ec);
    if (ec) break;
    if (fs::is_directory(st)) continue;
    if (!fs::is_regular_file(st)) {
      *err = rel + ": not a regular file (a link or special file cannot be signed or verified)";
      return false;
    }
    if (is_sign_file(rel)) continue;
    out->push_back(rel);
  }
  if (ec) {
    *err = dir.string() + ": cannot list: " + ec.message();
    return false;
  }
  std::sort(out->begin(), out->end());
  return true;
}
#endif

bool write_file(const fs::path& p, const std::string& data, std::string* err,
                unsigned mode = 0600) {
  std::error_code ec;
  fs::remove(p, ec);
  FILE* f = fopen(p.string().c_str(), "wb");
#if !defined(_WIN32)
  if (f) fchmod(fileno(f), mode);  // 0600 like the rest of the archive
#endif
  const bool ok = f && fwrite(data.data(), 1, data.size(), f) == data.size();
  if (f && fclose(f) != 0) { *err = p.string() + ": write failed"; return false; }
  if (!ok) *err = p.string() + ": cannot write";
  return ok;
}

#if !defined(_WIN32)
// Signing works relative to a directory descriptor, like the capture writer: every
// component opened is a single name under an already-open directory, with
// O_NOFOLLOW, so renaming or replacing a directory on the path cannot make the
// walk hash or write files outside the archive that `fd` names.
bool sha256_at(int dir, const char* name, std::string* hex_out) {
  const int fd = openat(dir, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return false;
  rocm::sha2::sha256 h;
  std::vector<uint8_t> buf(1u << 20);
  bool ok = true;
  for (;;) {
    const ssize_t n = read(fd, buf.data(), buf.size());
    if (n > 0) h.update(buf.data(), static_cast<size_t>(n));
    else if (n == 0) break;
    else if (errno != EINTR) { ok = false; break; }
  }
  close(fd);
  if (!ok) return false;
  *hex_out = h.hexdigest();
  return true;
}

// Hashes every regular file under the directory `dir` (borrowed), keyed by its
// '/'-separated path below it. `prefix` is that path for `dir` ("" or "a/b/").
bool hash_tree_at(int dir, const std::string& prefix, std::map<std::string, std::string>* out,
                  std::string* err) {
  const int dup_fd = openat(dir, ".", O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  DIR* dp = dup_fd >= 0 ? fdopendir(dup_fd) : nullptr;
  if (!dp) {
    if (dup_fd >= 0) close(dup_fd);
    *err = prefix + ": cannot list";
    return false;
  }
  bool ok = true;
  while (ok) {
    errno = 0;
    const dirent* e = readdir(dp);
    if (!e) {
      if (errno != 0) { *err = prefix + ": cannot list"; ok = false; }
      break;
    }
    const std::string name = e->d_name;
    if (name == "." || name == "..") continue;
    const std::string rel = prefix + name;
    struct stat st;
    if (fstatat(dirfd(dp), name.c_str(), &st, AT_SYMLINK_NOFOLLOW) != 0) {
      *err = rel + ": cannot stat";
      ok = false;
    } else if (S_ISDIR(st.st_mode)) {
      const int sub = openat(dirfd(dp), name.c_str(),
                             O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
      if (sub < 0) {
        *err = rel + ": cannot open";
        ok = false;
      } else {
        ok = hash_tree_at(sub, rel + "/", out, err);
        close(sub);
      }
    } else if (!S_ISREG(st.st_mode)) {
      *err = rel + ": not a regular file (a link or special file cannot be signed or verified)";
      ok = false;
    } else if (!is_sign_file(rel)) {
      std::string h;
      if (!sha256_at(dirfd(dp), name.c_str(), &h)) {
        *err = rel + ": cannot read";
        ok = false;
      } else {
        (*out)[rel] = h;
      }
    }
  }
  closedir(dp);
  return ok;
}

// `name` below the directory `dir`, whole, at most `max` bytes. A link is refused.
bool read_all_at(int dir, const char* name, uintmax_t max, std::string* out) {
  const int fd = openat(dir, name, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC);
  if (fd < 0) return false;
  struct stat st;
  bool ok = fstat(fd, &st) == 0 && S_ISREG(st.st_mode) &&
            static_cast<uintmax_t>(st.st_size) <= max;
  if (ok) {
    out->resize(static_cast<size_t>(st.st_size));
    for (size_t off = 0; ok && off < out->size();) {
      const ssize_t n = read(fd, out->data() + off, out->size() - off);
      if (n > 0) off += static_cast<size_t>(n);
      else ok = n < 0 && errno == EINTR;
    }
  }
  close(fd);
  return ok;
}

bool write_at(int dir, const char* name, const std::string& data, std::string* err) {
  unlinkat(dir, name, 0);
  const int fd = openat(dir, name, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0600);
  bool ok = fd >= 0;
  for (size_t off = 0; ok && off < data.size();) {
    const ssize_t n = write(fd, data.data() + off, data.size() - off);
    if (n > 0) off += static_cast<size_t>(n);
    else if (n < 0 && errno != EINTR) ok = false;
  }
  if (fd >= 0 && close(fd) != 0) ok = false;
  if (!ok) *err = std::string(name) + ": cannot write";
  return ok;
}
#endif

#if !defined(_WIN32)
// The caller's descriptor when it has one; otherwise one opened here. The walk
// and any writes then share a single directory.
struct DirFd {
  int fd;
  int own;
  DirFd(const std::string& dir, int borrowed)
      : fd(borrowed >= 0 ? borrowed : open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC)),
        own(borrowed >= 0 ? -1 : fd) {}
  ~DirFd() { if (own >= 0) close(own); }
};
#endif

// The one walk and hash of an archive, for signing and verifying alike: every
// file but the signature pair, keyed by its '/'-separated path.
bool hash_tree(const std::string& dir, [[maybe_unused]] int fd,
               std::map<std::string, std::string>* out, std::string* err) {
#if !defined(_WIN32)
  return hash_tree_at(fd, "", out, err);
#else
  const fs::path d(dir);
  std::vector<std::string> files;
  if (!list_files(d, &files, err)) return false;
  for (const std::string& rel : files) {
    if (!sha256_file(d / rel, &(*out)[rel])) {
      *err = rel + ": cannot read";
      return false;
    }
  }
  return true;
#endif
}

}  // namespace

bool sign_archive(const std::string& dir, const uint8_t seed[kSeedSize], std::string* err,
                  int dir_fd, uintmax_t max_manifest_bytes, const ExpectedTree* expected) {
  [[maybe_unused]] const fs::path d(dir);
  std::map<std::string, std::string> hashes;  // sorted by path, as the manifest is
#if !defined(_WIN32)
  const DirFd dfd(dir, dir_fd);
  const int fd = dfd.fd;
  if (fd < 0) {
    *err = dir + ": cannot open";
    return false;
  }
#else
  (void)dir_fd;
  const int fd = -1;
#endif
  if (!hash_tree(dir, fd, &hashes, err)) return false;
  if (expected) {
    // Only what the caller rewrote may differ from what it verified: any other
    // file changed, added or removed since is refused rather than signed.
    for (const auto& kv : hashes) {
      if (expected->rewritten.count(kv.first)) continue;
      const auto it = expected->digests.find(kv.first);
      if (it == expected->digests.end()) {
        *err = kv.first + ": not in the archive as it was verified";
        return false;
      }
      if (it->second != kv.second) {
        *err = kv.first + ": changed after the archive was verified";
        return false;
      }
    }
    for (const auto& kv : expected->digests) {
      if (!expected->rewritten.count(kv.first) && !hashes.count(kv.first)) {
        *err = kv.first + ": removed after the archive was verified";
        return false;
      }
    }
  }
  std::string manifest;
  for (const auto& kv : hashes) {
    // A name with a newline would split into lines that never verify.
    if (kv.first.find('\n') != std::string::npos) {
      *err = kv.first + ": file name holds a newline";
      return false;
    }
    manifest += kv.second + "  " + kv.first + "\n";
  }
  if (manifest.size() > max_manifest_bytes) {
    *err = std::string(kManifestName) + " would be " + std::to_string(manifest.size()) +
           " bytes (" + std::to_string(hashes.size()) + " files), over the " +
           std::to_string(max_manifest_bytes) +
           "-byte limit verification accepts; the archive cannot be signed";
    return false;
  }

  uint8_t secret[64], pub[32], s[32];
  memcpy(s, seed, 32);
  crypto_ed25519_key_pair(secret, pub, s);  // wipes s
  uint8_t sig[64];
  crypto_ed25519_sign(sig, secret, reinterpret_cast<const uint8_t*>(manifest.data()),
                      manifest.size());
  crypto_wipe(secret, sizeof(secret));

  // Remove the old signature first: a crash between the two writes then leaves an
  // unsigned archive, not one whose old signature meets a new manifest.
  const std::string sigbytes(reinterpret_cast<char*>(sig), 64);
#if !defined(_WIN32)
  unlinkat(fd, kSignatureName, 0);
  return write_at(fd, kManifestName, manifest, err) && write_at(fd, kSignatureName, sigbytes, err);
#else
  std::error_code ec;
  fs::remove(d / kSignatureName, ec);
  if (!write_file(d / kManifestName, manifest, err)) return false;
  return write_file(d / kSignatureName, sigbytes, err);
#endif
}

VerifyResult verify_archive(const std::string& dir, const uint8_t public_key[kPublicKeySize],
                            std::string* err, std::map<std::string, std::string>* verified,
                            uintmax_t max_manifest_bytes, int dir_fd) {
  [[maybe_unused]] const fs::path d(dir);
#if !defined(_WIN32)
  // One directory for the signature, the manifest and the tree walk.
  const DirFd dfd(dir, dir_fd);
  if (dfd.fd < 0) {
    *err = dir + ": cannot open";
    return VerifyResult::kRefused;
  }
  const int fd = dfd.fd;
  struct stat sst;
  if (fstatat(fd, kSignatureName, &sst, AT_SYMLINK_NOFOLLOW) != 0 && errno == ENOENT) {
    *err = dir + ": no " + kSignatureName;
    return VerifyResult::kUnsigned;
  }
  auto read_member = [&](const char* name, uintmax_t max, std::string* out) {
    return read_all_at(fd, name, max, out);
  };
#else
  (void)dir_fd;
  const int fd = -1;
  std::error_code ec;
  if (!fs::exists(fs::symlink_status(d / kSignatureName, ec))) {
    *err = dir + ": no " + kSignatureName;
    return VerifyResult::kUnsigned;
  }
  auto read_member = [&](const char* name, uintmax_t max, std::string* out) {
    return read_all(d / name, max, out);
  };
#endif
  std::string sig, manifest;
  if (!read_member(kSignatureName, 64, &sig) || sig.size() != 64) {
    *err = std::string(kSignatureName) + ": not a 64-byte Ed25519 signature";
    return VerifyResult::kRefused;
  }
  if (!read_member(kManifestName, max_manifest_bytes, &manifest)) {
    *err = std::string(kManifestName) + ": missing or unreadable";
    return VerifyResult::kRefused;
  }
  if (crypto_ed25519_check(reinterpret_cast<const uint8_t*>(sig.data()), public_key,
                           reinterpret_cast<const uint8_t*>(manifest.data()),
                           manifest.size()) != 0) {
    *err = std::string(kManifestName) + ": signature does not verify under the trusted key";
    return VerifyResult::kRefused;
  }

  // The manifest is authentic. Parse it, then hold the tree to it. A digest that
  // is not hex cannot equal one we compute, so it fails the comparison below.
  std::map<std::string, std::string> expected;
  size_t pos = 0;
  while (pos < manifest.size()) {
    const size_t nl = manifest.find('\n', pos);
    const std::string line = manifest.substr(pos, nl == std::string::npos ? nl : nl - pos);
    pos = nl == std::string::npos ? manifest.size() : nl + 1;
    if (line.size() < 67 || line[64] != ' ' || line[65] != ' ') {
      *err = std::string(kManifestName) + ": malformed line \"" + line.substr(0, 80) + "\"";
      return VerifyResult::kRefused;
    }
    expected[line.substr(66)] = line.substr(0, 64);
  }

  std::map<std::string, std::string> actual;
  if (!hash_tree(dir, fd, &actual, err)) return VerifyResult::kRefused;
  for (const auto& kv : actual) {
    const auto it = expected.find(kv.first);
    if (it == expected.end()) {
      *err = kv.first + ": not in the signed manifest";
      return VerifyResult::kRefused;
    }
    if (kv.second != it->second) {
      *err = kv.first + ": SHA-256 differs from the signed manifest";
      return VerifyResult::kRefused;
    }
    expected.erase(it);
  }
  if (!expected.empty()) {
    *err = expected.begin()->first + ": listed in the signed manifest but missing";
    return VerifyResult::kRefused;
  }
  if (verified) *verified = std::move(actual);
  return VerifyResult::kOk;
}

bool public_key_from_seed(const uint8_t seed[kSeedSize], uint8_t public_key[kPublicKeySize]) {
  uint8_t secret[64], s[32];
  memcpy(s, seed, 32);
  crypto_ed25519_key_pair(secret, public_key, s);  // wipes s
  crypto_wipe(secret, sizeof(secret));
  return true;
}

bool read_key_file(const std::string& path, bool is_seed, uint8_t out[32], std::string* err) {
#if !defined(_WIN32)
  struct stat st;
  if (stat(path.c_str(), &st) != 0) {
    *err = path + ": cannot read";
    return false;
  }
  if (is_seed && (st.st_mode & 077) != 0) {
    *err = path + ": signing key is readable by group or others; chmod 600 it";
    return false;
  }
#endif
  std::string text;
  if (!read_all(path, 256, &text)) {
    *err = path + ": cannot read";
    return false;
  }
  while (!text.empty() && (text.back() == '\n' || text.back() == '\r')) text.pop_back();
  const bool parsed = unhex(text, out, 32);
  rocm::sha2::secure_zero(text.data(), text.size());
  if (!parsed) {
    *err = path + ": expected 64 hex digits (a 32-byte " + (is_seed ? "seed" : "public key") + ")";
    return false;
  }
  return true;
}

bool generate_key_pair(const std::string& seed_path, const std::string& public_path,
                       std::string* err) {
#if defined(_WIN32)
  *err = "key generation is not supported on Windows";
  return false;
#else
  uint8_t seed[32], secret[64], pub[32], s[32];
  FILE* r = fopen("/dev/urandom", "rb");
  if (!r || fread(seed, 1, 32, r) != 32) {
    if (r) fclose(r);
    *err = "cannot read /dev/urandom";
    return false;
  }
  fclose(r);
  memcpy(s, seed, 32);
  crypto_ed25519_key_pair(secret, pub, s);
  crypto_wipe(secret, sizeof(secret));

  std::string seed_hex = hex(seed, 32) + "\n";
  crypto_wipe(seed, sizeof(seed));
  // Both files are created exclusively: an existing key is never replaced, and a
  // failure leaves neither file behind, so the same paths can be tried again.
  const auto create = [](const std::string& path, const std::string& text, mode_t mode) {
    const int fd = open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, mode);
    if (fd < 0) return false;
    bool ok = write(fd, text.data(), text.size()) == static_cast<ssize_t>(text.size());
    ok = (mode == 0600 || fchmod(fd, mode) == 0) && ok;  // not narrowed by the umask
    if (close(fd) != 0) ok = false;
    if (!ok) unlink(path.c_str());
    return ok;
  };
  const bool seed_ok = create(seed_path, seed_hex, 0600);
  rocm::sha2::secure_zero(seed_hex.data(), seed_hex.size());
  if (!seed_ok) {
    *err = seed_path + ": cannot create (it must not exist)";
    return false;
  }
  if (!create(public_path, hex(pub, 32) + "\n", 0644)) {  // meant to be shared
    unlink(seed_path.c_str());
    *err = public_path + ": cannot create (it must not exist); no key pair written";
    return false;
  }
  return true;
#endif
}

}  // namespace hrr_sign
