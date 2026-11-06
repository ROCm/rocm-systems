/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Repair
 * @{
 * @ingroup HRRTest
 */

#include "hrr_test_common.hh"
#include "hrr_reader.h"
#include "hrr/hrr_api_args.h"

#ifndef _WIN32

#include "hrr_safe_fs.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>

namespace fs = std::filesystem;

namespace {

hrr_event_header make_record(uint64_t seq) {
  hrr_event_header h{};
  std::memset(&h, 0, sizeof(h));
  h.event_type     = static_cast<uint16_t>(HRR_API_HIPDEVICESYNCHRONIZE);
  h.sequence_id    = seq;
  h.timestamp_ns   = 1000 + seq;
  h.thread_id      = 42;
  h.payload_length = static_cast<uint16_t>(sizeof(hrr_event_header));
  return h;
}

// A crash-truncated events.bin: file header, `n` whole records, no trailer, and
// half of one more record header.
void write_torn_events(const fs::path& file, int n) {
  std::ofstream f(file, std::ios::binary);
  hrr_file_header fh{HRR_MAGIC, HRR_VERSION, 0};
  f.write(reinterpret_cast<const char*>(&fh), sizeof(fh));
  for (int i = 0; i < n; ++i) {
    hrr_event_header h = make_record(static_cast<uint64_t>(i));
    f.write(reinterpret_cast<const char*>(&h), sizeof(h));
  }
  hrr_event_header partial = make_record(static_cast<uint64_t>(n));
  f.write(reinterpret_cast<const char*>(&partial), sizeof(partial) / 2);
}

void write_text(const fs::path& p, const std::string& s) {
  std::ofstream f(p, std::ios::binary);
  f << s;
}

std::string read_text(const fs::path& p) {
  std::ifstream f(p, std::ios::binary);
  std::stringstream ss;
  ss << f.rdbuf();
  return ss.str();
}

struct Scratch {
  fs::path root;     // the scratch directory
  fs::path archive;  // root/archive: a torn single-process archive
  fs::path victim;   // root/victim.txt: must never change

  explicit Scratch(const std::string& name) {
    root = fs::temp_directory_path() / ("hrr_repair_" + name);
    fs::remove_all(root);
    archive = root / "archive";
    fs::create_directories(archive / "blobs");
    fs::create_directories(archive / "code_objects");
    write_torn_events(archive / "events.bin", 4);
    victim = root / "victim.txt";
    write_text(victim, "victim contents\n");
  }
  ~Scratch() { fs::remove_all(root); }
};

std::set<std::string> repair_leftovers(const fs::path& dir) {
  std::set<std::string> out;
  for (const auto& e : fs::directory_iterator(dir)) {
    const std::string n = e.path().filename().string();
    if (n.find(".repair") != std::string::npos) out.insert(n);
  }
  return out;
}

}  // namespace

HRR_TEST_CASE(Unit_HRR_Repair_RecoversTornEventsWithoutLeftovers) {
  Scratch s("clean");
  struct stat before;
  REQUIRE(::stat((s.archive / "events.bin").c_str(), &before) == 0);
  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  INFO(out);
  REQUIRE(ret == 0);

  hrr::Archive a;
  REQUIRE(hrr::load_archive(s.archive.string(), a));
  CHECK(a.complete);
  CHECK(a.events.size() == 4);
  CHECK(repair_leftovers(s.archive).empty());

  struct stat st;
  REQUIRE(::stat((s.archive / "events.bin").c_str(), &st) == 0);
  CHECK((st.st_mode & 07777) == (before.st_mode & 07777));
}

HRR_TEST_CASE(Unit_HRR_Repair_LinkAtOldTempNameIsLeftAlone) {
  Scratch s("tmplink");
  const fs::path link = s.archive / "events.bin.repair.tmp";
  fs::create_symlink(s.victim, link);

  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  INFO(out);
  CHECK(ret == 0);
  CHECK(read_text(s.victim) == "victim contents\n");
  CHECK(fs::is_symlink(link));

  hrr::Archive a;
  REQUIRE(hrr::load_archive(s.archive.string(), a));
  CHECK(a.complete);
}

HRR_TEST_CASE(Unit_HRR_Repair_ManifestLinkIsRefused) {
  Scratch s("manifestlink");
  const std::string torn_before = read_text(s.archive / "events.bin");
  fs::create_symlink(s.victim, s.archive / "manifest.json");

  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  CHECK(ret != 0);
  CHECK(out.find("manifest.json") != std::string::npos);
  CHECK(read_text(s.victim) == "victim contents\n");
  CHECK(fs::is_symlink(s.archive / "manifest.json"));
  CHECK(read_text(s.archive / "events.bin") == torn_before);
  CHECK(repair_leftovers(s.archive).empty());
}

HRR_TEST_CASE(Unit_HRR_Repair_EventsLinkIsRefused) {
  Scratch s("eventslink");
  const fs::path real = s.root / "real_events.bin";
  fs::rename(s.archive / "events.bin", real);
  fs::create_symlink(real, s.archive / "events.bin");
  const std::string before = read_text(real);

  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  CHECK(ret != 0);
  CHECK(out.find("events.bin") != std::string::npos);
  CHECK(read_text(real) == before);
  CHECK(fs::is_symlink(s.archive / "events.bin"));
}

HRR_TEST_CASE(Unit_HRR_Repair_PidDirectoryLinkIsRefused) {
  Scratch s("pidlink");
  const fs::path elsewhere = s.root / "elsewhere";
  fs::create_directories(elsewhere);
  write_torn_events(elsewhere / "events.bin", 3);
  const std::string before = read_text(elsewhere / "events.bin");

  const fs::path rootdir = s.root / "capture";
  fs::create_directories(rootdir / "pid-100");
  write_torn_events(rootdir / "pid-100" / "events.bin", 2);
  fs::create_directory_symlink(elsewhere, rootdir / "pid-200");

  auto [ret, out] = hrr_playback_merged(rootdir, "--repair");
  INFO(out);
  CHECK(ret != 0);
  CHECK(read_text(elsewhere / "events.bin") == before);
  CHECK(!fs::exists(elsewhere / "manifest.json"));
  CHECK(repair_leftovers(elsewhere).empty());

  hrr::Archive a;
  REQUIRE(hrr::load_archive((rootdir / "pid-100").string(), a));
  CHECK(a.complete);
}

HRR_TEST_CASE(Unit_HRR_Repair_UserNamedLinkIsResolved) {
  Scratch s("userlink");
  const fs::path link = s.root / "latest";
  fs::create_directory_symlink(s.archive, link);

  auto [ret, out] = hrr_playback_merged(link, "--repair");
  INFO(out);
  CHECK(ret == 0);

  hrr::Archive a;
  REQUIRE(hrr::load_archive(s.archive.string(), a));
  CHECK(a.complete);
}

HRR_TEST_CASE(Unit_HRR_SafeFs_OpenDirRefusesLinks) {
  Scratch s("opendir");
  fs::create_directory_symlink(s.archive, s.root / "link");

  hrr::safefs::Fd parent(::open(s.root.c_str(), O_RDONLY | O_DIRECTORY));
  REQUIRE(parent.valid());
  std::string err;
  CHECK_FALSE(hrr::safefs::open_dir_at(parent.get(), "link", &err).valid());
  CHECK(err.find("link") != std::string::npos);
  CHECK_FALSE(hrr::safefs::open_dir_at(parent.get(), "victim.txt", &err).valid());
  CHECK(hrr::safefs::open_dir_at(parent.get(), "archive", &err).valid());
  CHECK_FALSE(hrr::safefs::open_dir_at(parent.get(), "../x", &err).valid());
}

HRR_TEST_CASE(Unit_HRR_SafeFs_TempIsExclusiveAndReplaceDoesNotFollow) {
  Scratch s("temp");
  std::string err;
  hrr::safefs::Fd dir = hrr::safefs::open_dir_nofollow_leaf(s.archive.string(), &err);
  REQUIRE(dir.valid());

  std::string n1, n2;
  hrr::safefs::Fd t1 = hrr::safefs::create_exclusive_temp(dir.get(), "x", &n1, &err);
  hrr::safefs::Fd t2 = hrr::safefs::create_exclusive_temp(dir.get(), "x", &n2, &err);
  REQUIRE(t1.valid());
  REQUIRE(t2.valid());
  CHECK(n1 != n2);
  struct stat st;
  REQUIRE(::fstat(t1.get(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);

  fs::create_symlink(s.victim, s.archive / "final");
  REQUIRE(::write(t1.get(), "new", 3) == 3);
  t1.reset();
  REQUIRE(hrr::safefs::commit_replace(dir.get(), n1, "final", &err));
  CHECK_FALSE(fs::is_symlink(s.archive / "final"));
  CHECK(read_text(s.archive / "final") == "new");
  CHECK(read_text(s.victim) == "victim contents\n");

  CHECK(hrr::safefs::stat_entry(dir.get(), "final") == hrr::safefs::Entry::Regular);
  CHECK(hrr::safefs::stat_entry(dir.get(), "nope") == hrr::safefs::Entry::Absent);
  fs::create_symlink(s.victim, s.archive / "lnk");
  CHECK(hrr::safefs::stat_entry(dir.get(), "lnk") == hrr::safefs::Entry::Other);
}

HRR_TEST_CASE(Unit_HRR_Repair_KeepsFilePermissions) {
  Scratch s("modes");
  write_text(s.archive / "manifest.json", "{\"pid\": 7, \"complete\": false}\n");
  REQUIRE(::chmod((s.archive / "events.bin").c_str(), 0640) == 0);
  REQUIRE(::chmod((s.archive / "manifest.json").c_str(), 0640) == 0);

  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  INFO(out);
  REQUIRE(ret == 0);
  struct stat st;
  REQUIRE(::stat((s.archive / "events.bin").c_str(), &st) == 0);
  CHECK((st.st_mode & 07777) == 0640);
  REQUIRE(::stat((s.archive / "manifest.json").c_str(), &st) == 0);
  CHECK((st.st_mode & 07777) == 0640);
}

HRR_TEST_CASE(Unit_HRR_Repair_DropsSetuidAndSetgid) {
  Scratch s("setid");
  write_text(s.archive / "manifest.json", "{\"pid\": 7, \"complete\": false}\n");
  REQUIRE(::chmod((s.archive / "events.bin").c_str(), 04750) == 0);
  REQUIRE(::chmod((s.archive / "manifest.json").c_str(), 02750) == 0);

  auto [ret, out] = hrr_playback_merged(s.archive, "--repair");
  INFO(out);
  REQUIRE(ret == 0);
  struct stat st;
  REQUIRE(::stat((s.archive / "events.bin").c_str(), &st) == 0);
  CHECK((st.st_mode & 07777) == 0750);
  REQUIRE(::stat((s.archive / "manifest.json").c_str(), &st) == 0);
  CHECK((st.st_mode & 07777) == 0750);
}

HRR_TEST_CASE(Unit_HRR_Repair_RootManifestLinkIsRefusedBeforeFanOut) {
  Scratch s("rootmanifest");
  const fs::path rootdir = s.root / "capture";
  fs::create_directories(rootdir / "pid-100");
  fs::create_directories(rootdir / "pid-200");
  write_torn_events(rootdir / "pid-100" / "events.bin", 2);
  write_torn_events(rootdir / "pid-200" / "events.bin", 3);
  const std::string before100 = read_text(rootdir / "pid-100" / "events.bin");
  const std::string before200 = read_text(rootdir / "pid-200" / "events.bin");
  fs::create_symlink(s.victim, rootdir / "manifest.json");

  auto [ret, out] = hrr_playback_merged(rootdir, "--repair");
  CHECK(ret != 0);
  CHECK(out.find("manifest.json") != std::string::npos);
  CHECK(read_text(s.victim) == "victim contents\n");
  CHECK(read_text(rootdir / "pid-100" / "events.bin") == before100);
  CHECK(read_text(rootdir / "pid-200" / "events.bin") == before200);
}

HRR_TEST_CASE(Unit_HRR_Repair_LinkedPidDirectoryIsNotIndexed) {
  Scratch s("pidindex");
  const fs::path elsewhere = s.root / "elsewhere";
  fs::create_directories(elsewhere);
  write_torn_events(elsewhere / "events.bin", 3);

  const fs::path rootdir = s.root / "capture";
  fs::create_directories(rootdir / "pid-100");
  fs::create_directories(rootdir / "pid-150");
  write_torn_events(rootdir / "pid-100" / "events.bin", 2);
  write_torn_events(rootdir / "pid-150" / "events.bin", 2);
  fs::create_directory_symlink(elsewhere, rootdir / "pid-200");

  auto [ret, out] = hrr_playback_merged(rootdir, "--repair");
  INFO(out);
  CHECK(ret != 0);  // the linked entry is a failure
  const std::string index = read_text(rootdir / "manifest.json");
  CHECK(index.find("\"pid\": 100") != std::string::npos);
  CHECK(index.find("\"pid\": 150") != std::string::npos);
  CHECK(index.find("\"pid\": 200") == std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_SafeFs_DotDotAfterLinkIsResolvedByTheKernel) {
  Scratch s("dotdot");
  fs::create_directories(s.root / "real" / "sub");
  fs::create_directories(s.root / "real" / "x");
  fs::create_directories(s.root / "x");  // the lexical answer; must NOT be opened
  fs::create_directory_symlink(s.root / "real" / "sub", s.root / "link");

  std::string err;
  hrr::safefs::Fd d = hrr::safefs::open_dir_nofollow_leaf((s.root / "link" / ".." / "x").string(), &err);
  REQUIRE(d.valid());
  struct stat got, want_kernel, want_lexical;
  REQUIRE(::fstat(d.get(), &got) == 0);
  REQUIRE(::stat((s.root / "real" / "x").c_str(), &want_kernel) == 0);
  REQUIRE(::stat((s.root / "x").c_str(), &want_lexical) == 0);
  CHECK(got.st_ino == want_kernel.st_ino);
  CHECK(got.st_ino != want_lexical.st_ino);
}

HRR_TEST_CASE(Unit_HRR_SafeFs_FdOwnsItsDescriptor) {
  hrr::safefs::Fd none;
  CHECK_FALSE(none.valid());

  int raw = ::open("/dev/null", O_RDONLY);
  REQUIRE(raw >= 0);
  hrr::safefs::Fd a(raw);
  CHECK(a.valid());
  CHECK(a.get() == raw);

  hrr::safefs::Fd b(std::move(a));
  CHECK_FALSE(a.valid());
  CHECK(b.get() == raw);

  hrr::safefs::Fd c;
  c = std::move(b);
  CHECK_FALSE(b.valid());
  CHECK(c.get() == raw);

  const int released = c.release();
  CHECK(released == raw);
  CHECK_FALSE(c.valid());
  CHECK(::fcntl(released, F_GETFD) != -1);  // release() did not close it

  hrr::safefs::Fd d(released);
  d.reset();
  CHECK_FALSE(d.valid());
  CHECK(::fcntl(released, F_GETFD) == -1);  // reset() did
}

HRR_TEST_CASE(Unit_HRR_SafeFs_RandomHexAndErrnoText) {
  std::string a, b;
  REQUIRE(hrr::safefs::random_hex(&a));
  REQUIRE(hrr::safefs::random_hex(&b));
  CHECK(a.size() == 16);
  CHECK(a.find_first_not_of("0123456789abcdef") == std::string::npos);
  CHECK(a != b);

  errno = ENOENT;
  const std::string text = hrr::safefs::errno_text("cannot open x");
  CHECK(text.find("cannot open x") != std::string::npos);
  CHECK(text.find(std::strerror(ENOENT)) != std::string::npos);
}

HRR_TEST_CASE(Unit_HRR_SafeFs_UnlinkAtRemovesTheNameNotTheTarget) {
  Scratch s("unlink");
  std::string err;
  hrr::safefs::Fd dir = hrr::safefs::open_dir_nofollow_leaf(s.archive.string(), &err);
  REQUIRE(dir.valid());

  write_text(s.archive / "plain", "x");
  hrr::safefs::unlink_at(dir.get(), "plain");
  CHECK_FALSE(fs::exists(s.archive / "plain"));

  fs::create_symlink(s.victim, s.archive / "lnk");
  hrr::safefs::unlink_at(dir.get(), "lnk");
  CHECK_FALSE(fs::is_symlink(s.archive / "lnk"));
  CHECK(read_text(s.victim) == "victim contents\n");

  hrr::safefs::unlink_at(dir.get(), "absent");  // no effect, no crash
}

HRR_TEST_CASE(Unit_HRR_SafeFs_TempFileModeAndName) {
  Scratch s("tempmode");
  std::string err;
  hrr::safefs::Fd dir = hrr::safefs::open_dir_nofollow_leaf(s.archive.string(), &err);
  REQUIRE(dir.valid());
  const mode_t old_umask = ::umask(022);

  std::string n_default, n_group;
  hrr::safefs::Fd t1 = hrr::safefs::create_exclusive_temp(dir.get(), "events.bin.repair", &n_default, &err);
  hrr::safefs::Fd t2 = hrr::safefs::create_exclusive_temp(dir.get(), "events.bin.repair", &n_group, &err, 0666);
  ::umask(old_umask);
  REQUIRE(t1.valid());
  REQUIRE(t2.valid());
  CHECK(n_default.rfind("events.bin.repair.", 0) == 0);
  CHECK(n_default.size() == std::string("events.bin.repair.").size() + 16);

  struct stat st;
  REQUIRE(::fstat(t1.get(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0600);
  REQUIRE(::fstat(t2.get(), &st) == 0);
  CHECK((st.st_mode & 0777) == 0644);  // 0666 & ~022
}

#endif  // !_WIN32

/** @} */
