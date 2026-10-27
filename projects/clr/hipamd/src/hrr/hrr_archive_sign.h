/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */
#pragma once

// Signing and verification of one process archive (a pid-<pid>/ directory).
//
// Archive integrity is two steps. A manifest, `hrr.manifest`, lists the SHA-256 of
// every file in the archive in `sha256sum` format ("<64 hex>  <relative path>",
// sorted by path, '/' separators). An Ed25519 (RFC 8032) signature over the exact
// bytes of that manifest is stored beside it as `hrr.sig` (64 raw bytes).
//
// The private key is a 32-byte seed in a file that stays with the capturing site;
// the public key is 32 bytes and is what a replaying site is given. See DESIGN.md,
// "Archive signing", for which keys replay trusts and how a recipient gets one.
//
// This file is compiled into libamdhip64 (capture signs) and into hrr-playback
// (replay verifies, and repair re-signs). It uses only the C++ standard library,
// shared/sha256 and the vendored Monocypher; it never links libcrypto.

#include <cstdint>
#include <map>
#include <set>
#include <string>

namespace hrr_sign {

constexpr const char* kManifestName = "hrr.manifest";
constexpr const char* kSignatureName = "hrr.sig";
constexpr size_t kSeedSize = 32;
constexpr size_t kPublicKeySize = 32;

// A manifest line is 64 + 2 + path. Verification reads the manifest whole before
// it checks the signature, so it refuses one over this size rather than allocate
// without limit for a forged file; signing refuses to write one, so an archive
// that is signed can always be verified.
constexpr uintmax_t kMaxManifestBytes = 256u << 20;

// Signs the finished archive in `dir`: writes hrr.manifest and hrr.sig, replacing
// any earlier pair, so a rewriter (repair, scrub) can sign again. Refuses an
// archive holding a link, a file that is not regular, or a name with a newline.
// On POSIX the archive is read and written relative to one directory descriptor,
// `dir_fd` when it is >= 0 (the capture writer's checked descriptor for `dir`),
// else one opened from `dir`; a directory renamed or replaced on the path
// afterwards cannot redirect the signing to other files. `dir` is then only used
// in messages. Refuses when the manifest would exceed `max_manifest_bytes`.
//
// `expected` (when given) is an archive as an earlier verify_archive saw it. Signing
// then refuses any file whose SHA-256 differs from it, that is not listed in it or
// that is gone, except the paths in `rewritten`, which the caller changed itself
// (repair rewrites events.bin and manifest.json). Without it the directory is
// signed as found.
struct ExpectedTree {
  std::map<std::string, std::string> digests;  // path -> SHA-256 hex, as verified
  std::set<std::string> rewritten;             // paths allowed to differ
};
bool sign_archive(const std::string& dir, const uint8_t seed[kSeedSize], std::string* err,
                  int dir_fd = -1, uintmax_t max_manifest_bytes = kMaxManifestBytes,
                  const ExpectedTree* expected = nullptr);

enum class VerifyResult {
  kOk,        // signature is valid under the key and every file matches
  kUnsigned,  // dir holds no hrr.sig; nothing was checked
  kRefused,   // signed, but does not verify; *err names the file
};

// Checks `dir` against its signature under `public_key`. Every file in the
// archive must be listed and match, and every listed file must exist. `err` names
// the first offending file, in path order. On kOk `*verified` (when given) holds
// the SHA-256 (lowercase hex) of each file as checked, by path below `dir`, for a
// reader that opens the files again later and must compare what it reads. On
// POSIX the signature, the manifest and the files are read relative to one
// directory descriptor, `dir_fd` when it is >= 0, else one opened from `dir`.
VerifyResult verify_archive(const std::string& dir, const uint8_t public_key[kPublicKeySize],
                            std::string* err, std::map<std::string, std::string>* verified = nullptr,
                            uintmax_t max_manifest_bytes = kMaxManifestBytes, int dir_fd = -1);

// The public key for a seed (what a replaying site is given). The seed is not
// modified. Always true; the bool leaves room to refuse a key that is not valid.
bool public_key_from_seed(const uint8_t seed[kSeedSize], uint8_t public_key[kPublicKeySize]);

// A key file holds 64 hex digits and an optional newline. A seed file is refused
// when group or others can read it (POSIX), as ssh does for a private key.
bool read_key_file(const std::string& path, bool is_seed, uint8_t out[32], std::string* err);

// Creates a new seed file (mode 0600) and the matching public key file, neither
// overwriting an existing file; if either cannot be created, neither is left
// behind. POSIX only; the random bytes come from /dev/urandom.
bool generate_key_pair(const std::string& seed_path, const std::string& public_path,
                       std::string* err);

}  // namespace hrr_sign
