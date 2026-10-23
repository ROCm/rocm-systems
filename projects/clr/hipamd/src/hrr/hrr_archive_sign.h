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
#include <string>

namespace hrr_sign {

constexpr const char* kManifestName = "hrr.manifest";
constexpr const char* kSignatureName = "hrr.sig";
constexpr size_t kSeedSize = 32;
constexpr size_t kPublicKeySize = 32;

// Signs the finished archive in `dir`: writes hrr.manifest and hrr.sig, replacing
// any earlier pair, so a rewriter (repair, scrub) can sign again. Refuses an
// archive holding a link, a file that is not regular, or a name with a newline.
bool sign_archive(const std::string& dir, const uint8_t seed[kSeedSize], std::string* err);

enum class VerifyResult {
  kOk,        // signature is valid under the key and every file matches
  kUnsigned,  // dir holds no hrr.sig; nothing was checked
  kRefused,   // signed, but does not verify; *err names the file
};

// Checks `dir` against its signature under `public_key`. Every file in the
// archive must be listed and match, and every listed file must exist. `err` names
// the first offending file, in path order.
VerifyResult verify_archive(const std::string& dir, const uint8_t public_key[kPublicKeySize],
                            std::string* err);

// Overwrites key material in a way the optimiser may not drop.
void wipe(void* p, size_t n);

// A key file holds 64 hex digits and an optional newline. A seed file is refused
// when group or others can read it (POSIX), as ssh does for a private key.
bool read_key_file(const std::string& path, bool is_seed, uint8_t out[32], std::string* err);

// Creates a new seed file (mode 0600, never overwriting) and the matching public
// key file. POSIX only; the random bytes come from /dev/urandom.
bool generate_key_pair(const std::string& seed_path, const std::string& public_path,
                       std::string* err);

}  // namespace hrr_sign
