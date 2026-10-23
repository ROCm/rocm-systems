/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#include "hrr_archive_sign.h"

#include "monocypher-ed25519.h"
#include "rocm/sha2/sha256.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;

namespace hrr_sign {
namespace {

// A manifest line is 64 + 2 + path; bound the file so a forged one cannot make
// the verifier allocate without limit before the signature is checked.
constexpr uintmax_t kMaxManifestBytes = 256u << 20;

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
  const auto d = h.digest();
  *hex_out = hex(d.data(), d.size());
  return true;
}

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
    if (rel.find('\n') != std::string::npos) {
      *err = rel + ": file name holds a newline";
      return false;
    }
    out->push_back(rel);
  }
  if (ec) {
    *err = dir.string() + ": cannot list: " + ec.message();
    return false;
  }
  std::sort(out->begin(), out->end());
  return true;
}

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

}  // namespace

void wipe(void* p, size_t n) { crypto_wipe(p, n); }

bool sign_archive(const std::string& dir, const uint8_t seed[kSeedSize], std::string* err) {
  const fs::path d(dir);
  std::vector<std::string> files;
  if (!list_files(d, &files, err)) return false;

  std::string manifest;
  for (const std::string& rel : files) {
    std::string h;
    if (!sha256_file(d / rel, &h)) {
      *err = rel + ": cannot read";
      return false;
    }
    manifest += h + "  " + rel + "\n";
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
  std::error_code ec;
  fs::remove(d / kSignatureName, ec);
  if (!write_file(d / kManifestName, manifest, err)) return false;
  return write_file(d / kSignatureName, std::string(reinterpret_cast<char*>(sig), 64), err);
}

VerifyResult verify_archive(const std::string& dir, const uint8_t public_key[kPublicKeySize],
                            std::string* err) {
  const fs::path d(dir);
  std::error_code ec;
  if (!fs::exists(fs::symlink_status(d / kSignatureName, ec))) {
    *err = dir + ": no " + kSignatureName;
    return VerifyResult::kUnsigned;
  }
  std::string sig, manifest;
  if (!read_all(d / kSignatureName, 64, &sig) || sig.size() != 64) {
    *err = std::string(kSignatureName) + ": not a 64-byte Ed25519 signature";
    return VerifyResult::kRefused;
  }
  if (!read_all(d / kManifestName, kMaxManifestBytes, &manifest)) {
    *err = std::string(kManifestName) + ": missing or unreadable";
    return VerifyResult::kRefused;
  }
  if (crypto_ed25519_check(reinterpret_cast<const uint8_t*>(sig.data()), public_key,
                           reinterpret_cast<const uint8_t*>(manifest.data()),
                           manifest.size()) != 0) {
    *err = std::string(kManifestName) + ": signature does not verify under the trusted key";
    return VerifyResult::kRefused;
  }

  // The manifest is authentic. Parse it, then hold the tree to it.
  std::map<std::string, std::string> expected;
  size_t pos = 0;
  while (pos < manifest.size()) {
    const size_t nl = manifest.find('\n', pos);
    const std::string line = manifest.substr(pos, nl == std::string::npos ? nl : nl - pos);
    pos = nl == std::string::npos ? manifest.size() : nl + 1;
    uint8_t raw[32];
    if (line.size() < 67 || line[64] != ' ' || line[65] != ' ' ||
        !unhex(line.substr(0, 64), raw, 32)) {
      *err = std::string(kManifestName) + ": malformed line \"" + line.substr(0, 80) + "\"";
      return VerifyResult::kRefused;
    }
    expected[line.substr(66)] = line.substr(0, 64);
  }

  std::vector<std::string> files;
  if (!list_files(d, &files, err)) return VerifyResult::kRefused;
  for (const std::string& rel : files) {
    const auto it = expected.find(rel);
    if (it == expected.end()) {
      *err = rel + ": not in the signed manifest";
      return VerifyResult::kRefused;
    }
    std::string h;
    if (!sha256_file(d / rel, &h)) {
      *err = rel + ": cannot read";
      return VerifyResult::kRefused;
    }
    if (h != it->second) {
      *err = rel + ": SHA-256 differs from the signed manifest";
      return VerifyResult::kRefused;
    }
    expected.erase(it);
  }
  if (!expected.empty()) {
    *err = expected.begin()->first + ": listed in the signed manifest but missing";
    return VerifyResult::kRefused;
  }
  return VerifyResult::kOk;
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
  wipe(text.data(), text.size());
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
  const int fd = open(seed_path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
  if (fd < 0) {
    *err = seed_path + ": cannot create (it must not exist)";
    return false;
  }
  const bool ok = write(fd, seed_hex.data(), seed_hex.size()) ==
                  static_cast<ssize_t>(seed_hex.size());
  close(fd);
  wipe(seed_hex.data(), seed_hex.size());
  if (!ok) {
    unlink(seed_path.c_str());
    *err = seed_path + ": write failed";
    return false;
  }
  return write_file(public_path, hex(pub, 32) + "\n", err, 0644);  // meant to be shared
#endif
}

}  // namespace hrr_sign
