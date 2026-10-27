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
 * its own): sign_archive against an RFC 8032 vector, the manifest, and
 * that verification refuses every kind of change to a signed archive, naming the
 * file. Two cases run `hrr-playback` to show replay and --repair honour it.
 */

#include "hrr_test_common.hh"

#ifdef HRR_HAVE_ARCHIVE_SIGN
#include "hrr_archive_sign.h"
#include "hrr/hrr_api_args.h"
#include "hrr_reader.h"

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <set>
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
    hrr_sign::public_key_from_seed(seed, pub);
    if (sign) {
      std::string err;
      REQUIRE(hrr_sign::sign_archive(dir.string(), seed, &err));
    }
  }

  VerifyResult verify(std::string* err, const uint8_t* key = nullptr) const {
    return hrr_sign::verify_archive(dir.string(), key ? key : pub, err);
  }
  // The seed as a 0600 key file under root, for --signing-key.
  fs::path write_seed_file(const std::string& name = "seed") const {
    std::string hexseed;
    for (uint8_t b : seed) { char t[3]; snprintf(t, sizeof(t), "%02x", b); hexseed += t; }
    write_bytes(root / name, hexseed + "\n");
#ifndef _WIN32
    REQUIRE(::chmod((root / name).c_str(), 0600) == 0);
#endif
    return root / name;
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
 *   - sign_archive signs with the RFC 8032 Ed25519 (SHA-512), not Monocypher's
 *     BLAKE2b variant: an archive with no files has an empty manifest, so section
 *     7.1 test 1 (empty message) gives the published public key and hrr.sig, and a
 *     signature from another Ed25519 tool verifies. Through sign_archive, so a
 *     switch of the call there to the other variant fails here.
 */
HRR_TEST_CASE(Unit_HRR_Sign_Rfc8032Vector) {
  ScopedDir scope(fs::temp_directory_path() / "hrr_sign_rfc8032");
  fs::create_directories(scope.path);
  auto seed = unhex("9d61b19deffd5a60ba844af492ec2cc44449c5697b326919703bac031cae7f60");
  uint8_t pub[32];
  hrr_sign::public_key_from_seed(seed.data(), pub);
  REQUIRE(std::vector<uint8_t>(pub, pub + 32) ==
          unhex("d75a980182b10ab7d54bfed3c964073a0ee172f3daa62325af021a68f707511a"));
  std::string err;
  REQUIRE(hrr_sign::sign_archive(scope.path.string(), seed.data(), &err));
  const std::string sig = read_text_file(scope.path / "hrr.sig");
  const auto want = unhex("e5564300c360ac729086e2cc806e828a84877f1eb8e5d974d873e06522490155"
                          "5fb8821590a33bacc61e39701cf9b46bd25bf5f0595bbe24655141438e7a100b");
  REQUIRE(sig.size() == 64);
  REQUIRE(std::vector<uint8_t>(sig.begin(), sig.end()) == want);
  CHECK(hrr_sign::verify_archive(scope.path.string(), pub, &err) == VerifyResult::kOk);
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
  uint8_t other[32], s[32];
  for (int i = 0; i < 32; ++i) s[i] = static_cast<uint8_t>(100 + i);
  hrr_sign::public_key_from_seed(s, other);
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
 *   - with --signing-key, a crashed (never signed) archive is repaired and signed,
 *     and the result verifies.
 */
HRR_TEST_CASE(Unit_HRR_Sign_RepairSignsOrDropsStaleSignature) {
  auto torn = [](Archive& a, bool stale_signature) {  // drop the trailer: needs repair
    std::string e = read_text_file(a.dir / "events.bin");
    write_bytes(a.dir / "events.bin", e.substr(0, e.size() - sizeof(hrr_eof_record)));
    std::string err;
    if (stale_signature)
      REQUIRE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));  // signed, then torn: stale
    write_bytes(a.dir / "manifest.json", "{\n  \"pid\": 4242,\n  \"complete\": false\n}\n");
  };
  std::string err;
  {
    Archive a("repair_drop");
    torn(a, /*stale_signature=*/true);
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair");
    INFO(out);
    REQUIRE(ret == 0);
    CHECK(a.verify(&err) == VerifyResult::kUnsigned);
  }
  {
    Archive a("repair_sign", /*sign=*/false);  // what a crash leaves: no signature
    torn(a, /*stale_signature=*/false);
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + a.write_seed_file().string());
    INFO(out);
    REQUIRE(ret == 0);
    CHECK(a.verify(&err) == VerifyResult::kOk);
    INFO(err);
  }
}

/**
 * Test Description
 * ----------------
 *   --repair --signing-key signs whatever is in the directory, so it must not turn
 *   a signed archive that was altered into one signed by the user's key:
 *   - a signed archive with a blob changed is refused, naming the blob, before
 *     anything is rewritten: hrr.sig, hrr.manifest and the blob are as they were;
 *   - the same when the signature is under another key;
 *   - the same archive unaltered (a signature that verifies) is signed again, and
 *     an unsigned one is signed, so the refusal is the altered signature, not the flag.
 */
HRR_TEST_CASE(Unit_HRR_Sign_RepairRefusesToSignAnAlteredSignedArchive) {
  std::string err;
  {
    Archive a("repair_altered");
    const fs::path seedfile = a.write_seed_file();
    const std::string sig_before = read_text_file(a.dir / "hrr.sig");
    const std::string manifest_before = read_text_file(a.dir / "hrr.manifest");

    // Negative control: nothing altered, so the signature verifies and repair signs it again.
    auto [okret, okout] = hrr_playback_merged(a.dir, "--repair --signing-key " + seedfile.string());
    INFO(okout);
    CHECK(okret == 0);
    CHECK(a.verify(&err) == VerifyResult::kOk);

    flip_first_byte(a.dir / "blobs/ab/abcdef.blob");
    const std::string blob_after = read_text_file(a.dir / "blobs/ab/abcdef.blob");
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + seedfile.string());
    INFO(out);
    CHECK(ret != 0);
    CHECK(out.find("blobs/ab/abcdef.blob") != std::string::npos);
    CHECK(read_text_file(a.dir / "hrr.sig") == sig_before);  // deterministic Ed25519, same key
    CHECK(read_text_file(a.dir / "hrr.manifest") == manifest_before);
    CHECK(read_text_file(a.dir / "blobs/ab/abcdef.blob") == blob_after);  // not re-signed over
    CHECK(a.verify(&err) == VerifyResult::kRefused);
  }
  {
    Archive a("repair_other_key");
    Archive other("repair_other_key_owner");
    other.seed[0] ^= 0x55;  // another user's seed
    const fs::path seedfile = other.write_seed_file();
    const std::string sig_before = read_text_file(a.dir / "hrr.sig");
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + seedfile.string());
    INFO(out);
    CHECK(ret != 0);
    CHECK(read_text_file(a.dir / "hrr.sig") == sig_before);
  }
  {
    Archive a("repair_unsigned", /*sign=*/false);
    auto [ret, out] = hrr_playback_merged(a.dir, "--repair --signing-key " + a.write_seed_file().string());
    INFO(out);
    CHECK(ret == 0);
    CHECK(a.verify(&err) == VerifyResult::kOk);
    CHECK(out.find("not verified") != std::string::npos);  // said, since nothing was checked
  }
}

/**
 * Test Description
 * ----------------
 *   HIP_HRR_TRUSTED_KEY is for replay, as --help says; --repair checks no signature:
 *   - with it exported, --repair --signing-key still signs a crashed archive (the
 *     documented path), and a plain --repair still repairs;
 *   - the flag --repair --trusted-key is still refused (Unit_HRR_Sign_KeyOptionsAreNotIgnored);
 *   - negative control: the same variable on --info still verifies, so it was not dropped.
 */
HRR_TEST_CASE(Unit_HRR_Sign_TrustedKeyVariableDoesNotBlockRepair) {
  Archive a("repair_env", /*sign=*/false);
  write_bytes(a.root / "trusted.pub", a.hex_pub());
  const std::vector<std::pair<std::string, std::string>> env = {
      {"HIP_HRR_TRUSTED_KEY", (a.root / "trusted.pub").string()}};
  std::string e = read_text_file(a.dir / "events.bin");
  write_bytes(a.dir / "events.bin", e.substr(0, e.size() - sizeof(hrr_eof_record)));  // crashed

  // The variable still applies to --info: this unsigned archive is refused.
  auto [info_ret, info_out] = hrr_playback_merged(a.dir, "--info", env);
  INFO(info_out);
  CHECK(info_ret != 0);

  auto [ret, out] = hrr_playback_merged(
      a.dir, "--repair --signing-key " + a.write_seed_file().string(), env);
  INFO(out);
  CHECK(ret == 0);
  std::string err;
  CHECK(a.verify(&err) == VerifyResult::kOk);
  CHECK(hrr_playback_merged(a.dir, "--info", env).first == 0);  // now signed: --info accepts it

  Archive b("repair_env_plain", /*sign=*/false);
  e = read_text_file(b.dir / "events.bin");
  write_bytes(b.dir / "events.bin", e.substr(0, e.size() - sizeof(hrr_eof_record)));
  CHECK(hrr_playback_merged(b.dir, "--repair", env).first == 0);
}

/**
 * Test Description
 * ----------------
 *   The manifest limit is one number for both sides, so an archive that is signed
 *   can be verified:
 *   - sign_archive refuses a manifest over the limit, saying so and why, and
 *     leaves no hrr.sig or hrr.manifest;
 *   - verify_archive refuses a manifest over the limit it is given;
 *   - negative control: under the default limit the same archive signs and verifies.
 */
HRR_TEST_CASE(Unit_HRR_Sign_ManifestLimitAppliesToSigningToo) {
  Archive a("manifest_cap", /*sign=*/false);
  std::string err;
  const uintmax_t small = 100;  // the manifest here is 4 lines of 66+ bytes
  CHECK_FALSE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err, -1, small));
  INFO(err);
  CHECK(err.find("limit") != std::string::npos);
  CHECK(err.find("cannot be signed") != std::string::npos);
  CHECK_FALSE(fs::exists(a.dir / "hrr.sig"));
  CHECK_FALSE(fs::exists(a.dir / "hrr.manifest"));

  REQUIRE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));
  CHECK(fs::file_size(a.dir / "hrr.manifest") > small);
  CHECK(a.verify(&err) == VerifyResult::kOk);
  CHECK(hrr_sign::verify_archive(a.dir.string(), a.pub, &err, nullptr, small) ==
        VerifyResult::kRefused);
  CHECK(err.find("hrr.manifest") != std::string::npos);
}

namespace {
// An hrr::ArchiveCheck that verifies like hrr-playback's, then runs `after_check`,
// standing in for whoever can write the archive between the check and the reads.
uint8_t g_check_key[32];
std::function<void(const std::string&)> g_after_check;
bool test_check(const std::string& dir, hrr::VerifiedFiles* files, std::string* err) {
  if (hrr_sign::verify_archive(dir, g_check_key, err, files) != VerifyResult::kOk) return false;
  if (g_after_check) g_after_check(dir);
  return true;
}
struct CheckGuard {  // the check is process-wide: leave none installed for other cases
  explicit CheckGuard(const Archive& a, std::function<void(const std::string&)> after = {}) {
    std::memcpy(g_check_key, a.pub, 32);
    g_after_check = std::move(after);
    hrr::set_archive_check(test_check);
  }
  ~CheckGuard() {
    hrr::set_archive_check(nullptr);
    g_after_check = nullptr;
  }
};
}  // namespace

/**
 * Test Description
 * ----------------
 *   The archive is checked once and its files are opened again later (events.bin,
 *   then code objects at each module's first launch and blobs at each use, then
 *   region streams), so a file swapped after the check must not be used:
 *   - negative control: an archive nobody touches loads, keeps the digest of every
 *     file in Archive::verified, and its files read back through them;
 *   - events.bin replaced between the check and the load: load_archive refuses it;
 *   - a code object added between the check and the load: refused, naming it;
 *   - after a load, a code object or blob replaced: read_archive_file refuses it
 *     (*refused set, no bytes), as does a file the check never listed.
 */
HRR_TEST_CASE(Unit_HRR_Sign_FilesAreHeldToTheVerifiedDigestsWhenRead) {
  {
    Archive a("toctou_ok");
    CheckGuard guard(a);
    hrr::Archive loaded;
    REQUIRE(hrr::load_archive(a.dir.string(), loaded));
    CHECK(loaded.checked);
    CHECK(loaded.verified.count("events.bin") == 1);
    CHECK(loaded.verified.count("blobs/ab/abcdef.blob") == 1);
    CHECK(loaded.verified.count("code_objects/0123.hsaco") == 1);
    CHECK(loaded.verified.count("hrr.sig") == 0);
    std::vector<uint8_t> data;
    std::string err;
    bool refused = true;
    REQUIRE(hrr::read_archive_file(a.dir.string(), "code_objects/0123.hsaco", &loaded.verified,
                                   data, &err, &refused));
    CHECK_FALSE(refused);
    CHECK(std::string(data.begin(), data.end()) == "code object");
  }
  {
    Archive a("toctou_events");
    CheckGuard guard(a, [&](const std::string& dir) {
      std::ofstream f(fs::path(dir) / "events.bin", std::ios::binary | std::ios::app);
      f << "x";  // a byte past the trailer: parsing alone would not notice
    });
    hrr::Archive loaded;
    CHECK_FALSE(hrr::load_archive(a.dir.string(), loaded));
  }
  {
    Archive a("toctou_events_swap");
    CheckGuard guard(a, [&](const std::string& dir) {
      // Same size, one byte different, in the middle of the trailer's payload.
      const fs::path p = fs::path(dir) / "events.bin";
      std::string e = read_text_file(p);
      e[e.size() - 4] = static_cast<char>(e[e.size() - 4] ^ 0x01);
      write_bytes(p, e);
    });
    hrr::Archive loaded;
    CHECK_FALSE(hrr::load_archive(a.dir.string(), loaded));
  }
  {
    Archive a("toctou_added");
    CheckGuard guard(a, [&](const std::string& dir) {
      write_bytes(fs::path(dir) / "code_objects/evil.hsaco", "unsigned gpu code");
    });
    hrr::Archive loaded;
    CHECK_FALSE(hrr::load_archive(a.dir.string(), loaded));
  }
  {
    Archive a("toctou_after_load");
    CheckGuard guard(a);
    hrr::Archive loaded;
    REQUIRE(hrr::load_archive(a.dir.string(), loaded));
    write_bytes(a.dir / "code_objects/0123.hsaco", "code objecT");  // same size, one byte
    write_bytes(a.dir / "blobs/ab/abcdef.blob", "something else");
    write_bytes(a.dir / "code_objects/new.hsaco", "never listed");
    for (const char* rel : {"code_objects/0123.hsaco", "blobs/ab/abcdef.blob",
                            "code_objects/new.hsaco"}) {
      std::vector<uint8_t> data;
      std::string err;
      bool refused = false;
      INFO(rel);
      CHECK_FALSE(hrr::read_archive_file(a.dir.string(), rel, &loaded.verified, data, &err,
                                         &refused));
      CHECK(refused);
      CHECK(data.empty());
      CHECK(err.find(rel) != std::string::npos);
    }
    // A listed file that is gone is a refusal too, not an absent blob.
    fs::remove(a.dir / "code_objects/0123.hsaco");
    std::vector<uint8_t> gone;
    std::string gone_err;
    bool gone_refused = false;
    CHECK_FALSE(hrr::read_archive_file(a.dir.string(), "code_objects/0123.hsaco", &loaded.verified,
                                       gone, &gone_err, &gone_refused));
    CHECK(gone_refused);
    // Without the digests the same swapped file is read: the check is what refuses it.
    std::vector<uint8_t> data;
    CHECK(hrr::read_archive_file(a.dir.string(), "blobs/ab/abcdef.blob", nullptr, data));
    // Without the digests a missing file is only missing: not a refusal.
    bool missing_refused = true;
    std::string missing_err;
    CHECK_FALSE(hrr::read_archive_file(a.dir.string(), "code_objects/0123.hsaco", nullptr, data,
                                       &missing_err, &missing_refused));
    CHECK_FALSE(missing_refused);
  }
}

/**
 * Test Description
 * ----------------
 *   --repair --signing-key verifies the old signature, then signs; between the two
 *   the files must not change:
 *   - negative control: an untouched archive signs again against the digests the
 *     verify saw, and so does one whose events.bin and manifest.json were
 *     rewritten (the two files repair rewrites, named in `rewritten`);
 *   - a blob or code object altered, added or removed after the verify: sign_archive
 *     refuses, naming the file, and leaves hrr.sig and hrr.manifest as they were;
 *   - events.bin altered when it was not declared rewritten (a complete archive,
 *     where repair writes nothing): refused;
 *   - verify_archive given a directory descriptor reads that directory, not
 *     whatever now stands at the path.
 */
HRR_TEST_CASE(Unit_HRR_Sign_SigningHoldsFilesToTheVerifiedTree) {
  std::string err;
  auto held_for = [&](const Archive& a, std::set<std::string> rewritten) {
    hrr_sign::ExpectedTree t;
    REQUIRE(hrr_sign::verify_archive(a.dir.string(), a.pub, &err, &t.digests) == VerifyResult::kOk);
    t.rewritten = std::move(rewritten);
    return t;
  };
  {
    Archive a("hold_ok");
    hrr_sign::ExpectedTree t = held_for(a, {});
    CHECK(hrr_sign::sign_archive(a.dir.string(), a.seed, &err, -1, hrr_sign::kMaxManifestBytes, &t));
    hrr_sign::ExpectedTree r = held_for(a, {"events.bin", "manifest.json"});
    write_bytes(a.dir / "manifest.json", "{ \"pid\": 4242, \"complete\": true }\n");
    write_bytes(a.dir / "events.bin", read_text_file(a.dir / "events.bin") + "x");
    CHECK(hrr_sign::sign_archive(a.dir.string(), a.seed, &err, -1, hrr_sign::kMaxManifestBytes, &r));
    CHECK(a.verify(&err) == VerifyResult::kOk);
  }
  for (const char* what : {"altered", "added", "removed", "events"}) {
    Archive a(std::string("hold_") + what);
    hrr_sign::ExpectedTree t = held_for(a, {});
    const std::string sig_before = read_text_file(a.dir / "hrr.sig");
    const std::string manifest_before = read_text_file(a.dir / "hrr.manifest");
    std::string file;
    if (!strcmp(what, "altered")) { file = "blobs/ab/abcdef.blob"; flip_first_byte(a.dir / file); }
    else if (!strcmp(what, "added")) { file = "code_objects/evil.hsaco"; write_bytes(a.dir / file, "x"); }
    else if (!strcmp(what, "removed")) { file = "code_objects/0123.hsaco"; fs::remove(a.dir / file); }
    else { file = "events.bin"; write_bytes(a.dir / file, read_text_file(a.dir / file) + "x"); }
    INFO(what);
    CHECK_FALSE(hrr_sign::sign_archive(a.dir.string(), a.seed, &err, -1, hrr_sign::kMaxManifestBytes, &t));
    CHECK(err.find(file) != std::string::npos);
    CHECK(read_text_file(a.dir / "hrr.sig") == sig_before);
    CHECK(read_text_file(a.dir / "hrr.manifest") == manifest_before);
    // Negative control: without the digests the same directory is signed as found.
    if (strcmp(what, "removed"))
      CHECK(hrr_sign::sign_archive(a.dir.string(), a.seed, &err));
  }
#ifndef _WIN32
  {
    Archive a("hold_fd");
    const int fd = ::open(a.dir.c_str(), O_RDONLY | O_DIRECTORY);
    REQUIRE(fd >= 0);
    const fs::path moved = a.root / "pid-moved";
    fs::rename(a.dir, moved);
    fs::create_directories(a.dir);  // another, empty directory now stands at the path
    CHECK(hrr_sign::verify_archive(a.dir.string(), a.pub, &err, nullptr,
                                   hrr_sign::kMaxManifestBytes, fd) == VerifyResult::kOk);
    CHECK(hrr_sign::verify_archive(a.dir.string(), a.pub, &err) == VerifyResult::kUnsigned);
    ::close(fd);
  }
#endif
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
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_RepairRefusesToSignAnAlteredSignedArchive)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_TrustedKeyVariableDoesNotBlockRepair)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_ManifestLimitAppliesToSigningToo)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_FilesAreHeldToTheVerifiedDigestsWhenRead)
HRR_SIGN_NOT_BUILT(Unit_HRR_Sign_SigningHoldsFilesToTheVerifiedTree)
#endif

/**
 * @}
 */
