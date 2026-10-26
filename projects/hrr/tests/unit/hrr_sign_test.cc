/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Archive Signing
 * @{
 * @ingroup HRRTest
 * GPU-free tests of archive signing (CLR's hrr_archive_sign.cpp, compiled in on
 * its own): the Ed25519 primitive against an RFC 8032 vector, the manifest, and
 * that verification refuses every kind of change to a signed archive, naming the
 * file. Two cases run `hrr-playback` to show replay and --repair honour it.
 */

#include "hrr_test_common.hh"

#ifdef HRR_HAVE_ARCHIVE_SIGN
#include "hrr_archive_sign.h"
#include "hrr/hrr_api_args.h"
#include "monocypher-ed25519.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#ifndef _WIN32
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>
#endif

namespace fs = std::filesystem;
using hrr_sign::VerifyResult;

namespace {

std::vector<uint8_t> unhex(const std::string& s) {
  std::vector<uint8_t> v;
  for (size_t i = 0; i + 1 < s.size(); i += 2)
    v.push_back(static_cast<uint8_t>(std::stoi(s.substr(i, 2), nullptr, 16)));
  return v;
}

void write_bytes(const fs::path& p, const std::string& s) {
  fs::create_directories(p.parent_path());
  std::ofstream f(p, std::ios::binary);
  f << s;
}

void flip_first_byte(const fs::path& p) {
  std::string s = read_text_file(p);
  s[0] = static_cast<char>(s[0] ^ 0x01);
  write_bytes(p, s);
}

// A whole one-process archive: an events.bin with a trailer, a manifest, a blob
// under blobs/<xx>/ and a code object, so a replay would find every kind of file.
struct Archive {
  ScopedDir scope;  // removes the tree on destruction, also when a REQUIRE fails
  fs::path root, dir;
  uint8_t seed[32], pub[32];

  explicit Archive(const std::string& name, bool sign = true)
      : scope(fs::temp_directory_path() / ("hrr_sign_" + name)) {
    root = scope.path;
    dir = root / "pid-4242";
    {
      std::ofstream f(dir / "events.bin", std::ios::binary | std::ios::out);
      if (!f) {
        fs::create_directories(dir);
        f.open(dir / "events.bin", std::ios::binary);
      }
      hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
      f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
      hrr_eof_record rec = hrr_make_eof_record(0, 0);
      f.write(reinterpret_cast<const char*>(&rec), sizeof(rec));
    }
    write_bytes(dir / "manifest.json",
                "{\n  \"pid\": 4242,\n  \"parent_pid\": 1,\n  \"complete\": true,\n"
                "  \"event_count\": 0,\n  \"blob_count\": 1\n}\n");
    write_bytes(dir / "blobs/ab/abcdef.blob", "blob contents");
    write_bytes(dir / "code_objects/0123.hsaco", "code object");
    for (int i = 0; i < 32; ++i) seed[i] = static_cast<uint8_t>(i + 1);
    uint8_t s[32], secret[64];
    std::memcpy(s, seed, 32);
    crypto_ed25519_key_pair(secret, pub, s);
    if (sign) {
      std::string err;
      REQUIRE(hrr_sign::sign_archive(dir.string(), seed, &err));
    }
  }

  VerifyResult verify(std::string* err, const uint8_t* key = nullptr) const {
    return hrr_sign::verify_archive(dir.string(), key ? key : pub, err);
  }
  std::string hex_pub() const {
    static const char* d = "0123456789abcdef";
    std::string s;
    for (uint8_t b : pub) { s += d[b >> 4]; s += d[b & 15]; }
    return s + "\n";
  }
};

}  // namespace

/**
 * Test Description
 * ----------------
 *   - The vendored Ed25519 is the RFC 8032 one (SHA-512), not Monocypher's
 *     BLAKE2b variant: section 7.1 test 1 (empty message) yields the published
 *     public key and signature, so a signature from another Ed25519 tool verifies.
 */
HRR_TEST_CASE(Unit_HRR_Sign_Rfc8032Vector) {
  auto seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
  uint8_t secret[64], pub[32];
  crypto_ed25519_key_pair(secret, pub, seed.data());
  REQUIRE(std::vector<uint8_t>(pub, pub + 32) ==
          unhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"));
  uint8_t sig[64];
  crypto_ed25519_sign(sig, secret, nullptr, 0);
  REQUIRE(std::vector<uint8_t>(sig, sig + 64) ==
          unhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
                "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b"));
  REQUIRE(crypto_ed25519_check(sig, pub, nullptr, 0) == 0);
}

/**
 * Test Description
 * ----------------
 *   - A signed archive verifies. hrr.manifest lists every file, and no other, as
 *     "<sha-256>  <path>" in sorted order (the sha256sum format), with the digest
 *     of "blob contents" a known value; hrr.sig is 64 bytes.
 *   - Signing again (a rewriter re-signing) replaces the pair and still verifies.
 */
HRR_TEST_CASE(Unit_HRR_Sign_SignedArchiveVerifies) {
  Archive a("ok");
  std::string err;
  REQUIRE(a.verify(&err) == VerifyResult::kOk);
  INFO(err);

  const std::string manifest = read_text_file(a.dir / hrr_sign::kManifestName);
  std::istringstream in(manifest);
  std::vector<std::string> paths;
  for (std::string line; std::getline(in, line);) {
    REQUIRE(line.size() > 66);
    REQUIRE(line.substr(64, 2) == "  ");
    paths.push_back(line.substr(66));
  }
  REQUIRE(paths == std::vector<std::string>{"blobs/ab/abcdef.blob", "code_objects/0123.hsaco",
                                            "events.bin", "manifest.json"});
  // sha256sum of "blob contents"; the digest is of the bytes, not the name.
  CHECK(manifest.find("507c1a291c63d7e46cce06cb843de4e7835b2afa4bae4f2ed33e2f7e28796c9c  blobs/ab/abcdef.blob\n") != std::string::npos);
  CHECK(fs::file_size(a.dir / hrr_sign::kSignatureName) == 64);

  write_bytes(a.dir / "blobs/ab/abcdef.blob", "new contents");
  REQUIRE(a.verify(&err) == VerifyResult::kRefused);
  REQUIRE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));
  REQUIRE(a.verify(&err) == VerifyResult::kOk);
}

/**
 * Test Description
 * ----------------
 *   Each change to a signed archive is refused, and the message names the file:
 *   one flipped byte in a blob, in a code object, in events.bin and in
 *   manifest.json; a file added; a file deleted; a link planted (POSIX).
 *   A manifest or signature edit is refused too, as a signature that does not
 *   verify. The negative control is the untouched archive in the first case.
 */
HRR_TEST_CASE(Unit_HRR_Sign_TamperIsRefusedNamingFile) {
  std::string err;
  for (const char* file : {"blobs/ab/abcdef.blob", "code_objects/0123.hsaco", "events.bin",
                           "manifest.json"}) {
    Archive a(std::string("flip_") + std::to_string(std::hash<std::string>{}(file) % 1000));
    REQUIRE(a.verify(&err) == VerifyResult::kOk);  // control
    flip_first_byte(a.dir / file);
    INFO(file);
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find(file) == 0);
  }
  {
    Archive a("extra");
    write_bytes(a.dir / "blobs/cd/planted.blob", "x");
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find("blobs/cd/planted.blob") == 0);
  }
  {
    Archive a("missing");
    fs::remove(a.dir / "code_objects/0123.hsaco");
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find("code_objects/0123.hsaco") == 0);
  }
  {
    Archive a("manifest_edit");
    // Replace a file's recorded digest with another's: the signature fails.
    std::string m = read_text_file(a.dir / hrr_sign::kManifestName);
    m[0] = m[0] == '0' ? '1' : '0';
    write_bytes(a.dir / hrr_sign::kManifestName, m);
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find(hrr_sign::kManifestName) == 0);
  }
  {
    Archive a("sig_cut");
    write_bytes(a.dir / hrr_sign::kSignatureName, std::string(63, 'x'));
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find(hrr_sign::kSignatureName) == 0);
  }
#ifndef _WIN32
  {
    Archive a("link");
    REQUIRE(::symlink("/etc/passwd", (a.dir / "blobs/ab/link.blob").c_str()) == 0);
    REQUIRE(a.verify(&err) == VerifyResult::kRefused);
    CHECK(err.find("blobs/ab/link.blob") == 0);
    REQUIRE_FALSE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));
  }
#endif
}

/**
 * Test Description
 * ----------------
 *   - An archive with no signature reports kUnsigned, not kOk, so the caller that
 *     asked for a signature can refuse it.
 *   - A signature under another key is refused.
 */
HRR_TEST_CASE(Unit_HRR_Sign_UnsignedAndWrongKey) {
  std::string err;
  {
    Archive a("unsigned", /*sign=*/false);
    CHECK(a.verify(&err) == VerifyResult::kUnsigned);
  }
  Archive a("wrongkey");
  uint8_t other[32], secret[64], s[32];
  for (int i = 0; i < 32; ++i) s[i] = static_cast<uint8_t>(100 + i);
  crypto_ed25519_key_pair(secret, other, s);
  CHECK(a.verify(&err, other) == VerifyResult::kRefused);
}

/**
 * Test Description
 * ----------------
 *   - generate_key_pair writes a seed (mode 0600) and the matching public key,
 *     and refuses to overwrite an existing seed. An archive signed with the
 *     seed read back verifies under the public key read back.
 *   - A seed readable by others is refused (POSIX); a malformed key file is too.
 */
HRR_TEST_CASE(Unit_HRR_Sign_KeyFiles) {
  ScopedDir keys{fs::temp_directory_path() / "hrr_sign_keys"};
  const fs::path& d = keys.path;
  fs::create_directories(d);
  std::string err;
  const std::string seed_path = (d / "seed").string(), pub_path = (d / "pub").string();
#ifndef _WIN32
  REQUIRE(hrr_sign::generate_key_pair(seed_path, pub_path, &err));
  struct stat st;
  REQUIRE(::stat(seed_path.c_str(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  CHECK_FALSE(hrr_sign::generate_key_pair(seed_path, pub_path, &err));

  // An existing public key file is never replaced, and the refusal leaves no new
  // seed behind; a public key that cannot be created takes the seed with it.
  const std::string seed2 = (d / "seed2").string(), pub_before = read_text_file(pub_path);
  CHECK_FALSE(hrr_sign::generate_key_pair(seed2, pub_path, &err));
  CHECK_FALSE(fs::exists(seed2));
  CHECK(read_text_file(pub_path) == pub_before);
  CHECK_FALSE(hrr_sign::generate_key_pair(seed2, (d / "no_such_dir" / "pub").string(), &err));
  CHECK_FALSE(fs::exists(seed2));
  REQUIRE(hrr_sign::generate_key_pair(seed2, (d / "pub2").string(), &err));  // retry works
  CHECK(fs::exists(seed2));

  uint8_t seed[32], pub[32];
  REQUIRE(hrr_sign::read_key_file(seed_path, true, seed, &err));
  REQUIRE(hrr_sign::read_key_file(pub_path, false, pub, &err));
  Archive a("keyfiles", /*sign=*/false);
  REQUIRE(hrr_sign::sign_archive(a.dir.string(), seed, &err));
  CHECK(a.verify(&err, pub) == VerifyResult::kOk);

  REQUIRE(::chmod(seed_path.c_str(), 0644) == 0);
  CHECK_FALSE(hrr_sign::read_key_file(seed_path, true, seed, &err));
  CHECK(hrr_sign::read_key_file(seed_path, false, seed, &err));  // a public key may be world-readable
#endif
  write_bytes(d / "bad", "not hex\n");
  CHECK_FALSE(hrr_sign::read_key_file((d / "bad").string(), false, nullptr, &err));
}

/**
 * Test Description
 * ----------------
 *   - sign_archive with a directory descriptor signs the directory that descriptor
 *     names, not whatever the path names by then: the archive is renamed away and
 *     another directory takes its path. The signed manifest covers the original
 *     files and the stand-in directory is left untouched (POSIX).
 */
HRR_TEST_CASE(Unit_HRR_Sign_SigningFollowsTheDirectoryDescriptor) {
#ifndef _WIN32
  Archive a("dirfd", /*sign=*/false);
  const int fd = ::open(a.dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC);
  REQUIRE(fd >= 0);
  const fs::path moved = a.root / "pid-moved";
  fs::rename(a.dir, moved);
  write_bytes(a.dir / "events.bin", "someone else's file");
  std::string err;
  const bool ok = hrr_sign::sign_archive(a.dir.string(), a.seed, &err, fd);
  ::close(fd);
  INFO(err);
  REQUIRE(ok);
  CHECK(fs::exists(moved / "hrr.sig"));
  CHECK(read_text_file(moved / "hrr.manifest").find("blobs/ab/abcdef.blob") != std::string::npos);
  CHECK_FALSE(fs::exists(a.dir / "hrr.sig"));
  CHECK_FALSE(fs::exists(a.dir / "hrr.manifest"));
  CHECK(hrr_sign::verify_archive(moved.string(), a.pub, &err) == VerifyResult::kOk);
#else
  HRR_SKIP("directory descriptors are POSIX");
#endif
}

/**
 * Test Description
 * ----------------
 *   hrr-playback --info on a signed archive, which loads it as replay does:
 *   - with the right --trusted-key: succeeds;
 *   - after one byte of a blob changes: fails and names the blob;
 *   - with another key: fails; with no key: still loads, the signature unchecked;
 *   - an unsigned archive with a key: fails.
 */
HRR_TEST_CASE(Unit_HRR_Sign_PlaybackRefusesTamperedArchive) {
  Archive a("playback");
  const fs::path key = a.root / "trusted.pub";
  write_bytes(key, a.hex_pub());
  const std::string args = "--info --trusted-key " + key.string();

  auto [ok, out_ok] = hrr_playback_merged(a.dir, args);
  INFO(out_ok);
  REQUIRE(ok == 0);

  REQUIRE(hrr_playback_merged(a.dir, "--info").first == 0);

  write_bytes(a.root / "other.pub", std::string(64, '1') + "\n");
  CHECK(hrr_playback_merged(a.dir, "--info --trusted-key " + (a.root / "other.pub").string()).first != 0);

  flip_first_byte(a.dir / "blobs/ab/abcdef.blob");
  auto [bad, out_bad] = hrr_playback_merged(a.dir, args);
  INFO(out_bad);
  CHECK(bad != 0);
  CHECK(out_bad.find("blobs/ab/abcdef.blob") != std::string::npos);

  // The archive root takes the same path: --info there must not print the
  // unchecked process summary and exit 0 when a key was given.
  auto [root_bad, out_root_bad] = hrr_playback_merged(a.root, args);
  INFO(out_root_bad);
  CHECK(root_bad != 0);
  CHECK(out_root_bad.find("blobs/ab/abcdef.blob") != std::string::npos);
  CHECK(hrr_playback_merged(a.root, "--info").first == 0);  // no key: summary, unchecked

  Archive u("playback_unsigned", /*sign=*/false);
  write_bytes(u.root / "trusted.pub", u.hex_pub());
  auto [uns, out_uns] = hrr_playback_merged(u.dir, "--info --trusted-key " + (u.root / "trusted.pub").string());
  INFO(out_uns);
  CHECK(uns != 0);
}

/**
 * Test Description
 * ----------------
 *   - A key option with no value, or with an empty one, is an error: replay never
 *     runs unchecked because a flag was dropped. Same for an empty
 *     HIP_HRR_TRUSTED_KEY, and for --trusted-key with --repair.
 *   - --repair --signing-key on an already complete archive signs it.
 */
HRR_TEST_CASE(Unit_HRR_Sign_KeyOptionsAreNotIgnored) {
  Archive a("options", /*sign=*/false);
  CHECK(hrr_playback_merged(a.dir, "--info --trusted-key").first != 0);
  CHECK(hrr_playback_merged(a.dir, "--info", {{"HIP_HRR_TRUSTED_KEY", ""}}).first != 0);
  write_bytes(a.root / "trusted.pub", a.hex_pub());
  // A signed archive that verifies under the key: only the --repair guard can
  // refuse it, since load_archive would accept it.
  {
    Archive sa("options_signed");
    write_bytes(sa.root / "trusted.pub", sa.hex_pub());
    const std::string before = read_text_file(sa.dir / "manifest.json");
    auto [gret, gout] = hrr_playback_merged(
        sa.dir, "--info --trusted-key " + (sa.root / "trusted.pub").string());
    INFO(gout);
    REQUIRE(gret == 0);  // the archive does verify
    auto [rret, rout] = hrr_playback_merged(
        sa.dir, "--repair --trusted-key " + (sa.root / "trusted.pub").string());
    INFO(rout);
    CHECK(rret != 0);
    CHECK(rout.find("does not check a signature") != std::string::npos);
    CHECK(read_text_file(sa.dir / "manifest.json") == before);
  }
  CHECK(hrr_playback_merged(a.dir, "--info --signing-key x").first != 0);

  std::string hexseed, err;
  for (uint8_t b : a.seed) { char t[3]; snprintf(t, sizeof(t), "%02x", b); hexseed += t; }
  write_bytes(a.root / "seed", hexseed + "\n");
#ifndef _WIN32
  REQUIRE(::chmod((a.root / "seed").c_str(), 0600) == 0);
#endif
  auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + (a.root / "seed").string());
  INFO(out);
  REQUIRE(ret == 0);
  CHECK(a.verify(&err) == VerifyResult::kOk);
}

/**
 * Test Description
 * ----------------
 *   --repair rewrites events.bin and manifest.json, so a signature over the old
 *   ones is stale:
 *   - without --signing-key, the stale hrr.sig and hrr.manifest are removed;
 *   - with --signing-key, the repaired archive is signed and verifies.
 */
HRR_TEST_CASE(Unit_HRR_Sign_RepairSignsOrDropsStaleSignature) {
  auto torn = [](Archive& a) {  // drop the trailer: the archive now needs repair
    std::string e = read_text_file(a.dir / "events.bin");
    write_bytes(a.dir / "events.bin", e.substr(0, e.size() - sizeof(hrr_eof_record)));
    std::string err;
    REQUIRE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));  // signed, then torn: stale
    write_bytes(a.dir / "manifest.json", "{\n  \"pid\": 4242,\n  \"complete\": false\n}\n");
  };
  std::string err;
  {
    Archive a("repair_drop");
    torn(a);
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair");
    INFO(out);
    REQUIRE(ret == 0);
    CHECK(a.verify(&err) == VerifyResult::kUnsigned);
  }
  {
    Archive a("repair_sign");
    torn(a);
    std::string hexseed;
    for (uint8_t b : a.seed) { char t[3]; snprintf(t, sizeof(t), "%02x", b); hexseed += t; }
    write_bytes(a.root / "seed", hexseed + "\n");
#ifndef _WIN32
    REQUIRE(::chmod((a.root / "seed").c_str(), 0600) == 0);
#endif
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + (a.root / "seed").string());
    INFO(out);
    REQUIRE(ret == 0);
    CHECK(a.verify(&err) == VerifyResult::kOk);
    INFO(err);
  }
}

#else
// projects/hrr was checked out without CLR's hrr_archive_sign.cpp or shared/sha256
// (playback/CMakeLists.txt).
#define HRR_SIGN_NOT_BUILT(name) \
  HRR_TEST_CASE(name) { HRR_SKIP("built without hrr_archive_sign.cpp"); }
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_Rfc8032Vector)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_SignedArchiveVerifies)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_TamperIsRefusedNamingFile)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_UnsignedAndWrongKey)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_KeyFiles)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_PlaybackRefusesTamperedArchive)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_RepairSignsOrDropsStaleSignature)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_KeyOptionsAreNotIgnored)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_SigningFollowsTheDirectoryDescriptor)
#endif

/**
 * @}
 */
