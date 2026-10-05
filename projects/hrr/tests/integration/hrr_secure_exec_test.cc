/*
 * Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
 *
 * SPDX-License-Identifier: MIT
 */

/**
 * @addtogroup HRR HRR Capture Gate
 * @{
 * @ingroup HRRTest
 * When HIP_HRR_CAPTURE_OUTPUT must not arm capture:
 *
 *   Unit_HRR_BlankCaptureOutputCapturesNothing:
 *     An exported empty variable reaches CLR as a single space. The workload
 *     runs with it and must leave no archive under a directory named " ".
 *
 *   Unit_HRR_SecureExecIgnoresCaptureOutput (Linux):
 *     A set-group-ID copy of this binary, for one of the user's supplementary
 *     groups, starts in secure-execution mode. It must write no archive and
 *     print the notice. No root is needed. The case skips when the filesystem
 *     is mounted nosuid, when no_new_privs is set, and when the loader cannot
 *     reach the capture runtime without LD_LIBRARY_PATH, which secure-execution
 *     mode ignores.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#if defined(__linux__)
#include <cstdio>
#include <cstring>
#include <fstream>
#include <link.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif


#undef HRR_SKIP
#define HRR_SKIP(m) FAIL("DEBUG SKIP: " << m)

#if defined(__linux__)
static std::string dbg_status() {
  std::string r;
  std::ifstream st("/proc/self/status");
  std::string l;
  while (std::getline(st, l))
    if (l.rfind("NoNewPrivs", 0) == 0 || l.rfind("Seccomp", 0) == 0 || l.rfind("Cap", 0) == 0 ||
        l.rfind("Uid", 0) == 0 || l.rfind("Gid", 0) == 0 || l.rfind("Groups", 0) == 0)
      r += l + "\n";
  return r;
}

TEST_CASE("Unit_HRR_SecureExecProbe_Direct", "[.][hrr-direct]") {
  std::printf("PROBE AT_SECURE=%lu uid=%d euid=%d gid=%d egid=%d\n%s", getauxval(AT_SECURE),
              (int)getuid(), (int)geteuid(), (int)getgid(), (int)getegid(), dbg_status().c_str());
  int n = 0;
  (void)hipGetDeviceCount(&n);
  dl_iterate_phdr(
      [](dl_phdr_info* i, size_t, void*) {
        if (std::strstr(i->dlpi_name, "amdhip") || std::strstr(i->dlpi_name, "hsa-runtime"))
          std::printf("PROBE lib %s\n", i->dlpi_name);
        return 0;
      },
      nullptr);
  std::printf("PROBE devices=%d\n", n);
  std::fflush(stdout);
}
#endif

HRR_TEST_CASE(Unit_HRR_BlankCaptureOutputCapturesNothing) {
  // The child inherits this working directory, so a blank value taken as a
  // path would write under " " here.
  const fs::path blank = fs::current_path() / " ";
  if (fs::exists(blank)) HRR_SKIP("a directory named \" \" already exists here");
  ScopedDir guard{blank};

  hrr::test::SpawnProc proc(hrr_test_exe(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", "");
  set_proc_search_path(proc);
  const int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
  INFO("Workload output:\n" << proc.getOutput());
  REQUIRE(ret == 0);
  CHECK(hrr_process_archives(blank).empty());
  CHECK_FALSE(fs::exists(blank));
}

#if defined(__linux__)
namespace {

constexpr const char* kSecureExecNotice = "HIP_HRR_CAPTURE_OUTPUT ignored";

// A group the file can be given that differs from the effective one, so a
// set-group-ID exec changes the effective group and the kernel sets AT_SECURE.
// The owner of a file may give it any of their supplementary groups; root may
// give it any group at all.
bool pick_other_group(gid_t* out) {
  const gid_t egid = getegid();
  const int n = getgroups(0, nullptr);
  if (n > 0) {
    std::vector<gid_t> groups(static_cast<size_t>(n));
    const int got = getgroups(n, groups.data());
    for (int i = 0; i < got; ++i) {
      if (groups[i] != egid) {
        *out = groups[i];
        return true;
      }
    }
  }
  if (geteuid() == 0) {
    *out = egid == 65534 ? 65533 : 65534;
    return true;
  }
  return false;
}

bool loader_failed(const std::string& out) {
  return out.find("error while loading shared libraries") != std::string::npos;
}

}  // namespace
#endif

HRR_TEST_CASE(Unit_HRR_SecureExecIgnoresCaptureOutput) {
#if !defined(__linux__)
  HRR_SKIP("secure-execution mode is Linux only");
#else
  ScopedDir dir{fs::temp_directory_path() / "hrr_secure_exec"};
  fs::create_directories(dir.path);

  struct statvfs vfs {};
  if (statvfs(dir.path.c_str(), &vfs) != 0 || (vfs.f_flag & ST_NOSUID))
    HRR_SKIP("the temporary directory is on a nosuid filesystem");
  if (prctl(PR_GET_NO_NEW_PRIVS, 0, 0, 0, 0) == 1)
    HRR_SKIP("no_new_privs is set, so exec would ignore the set-group-ID bit");
  gid_t gid = 0;
  if (!pick_other_group(&gid)) HRR_SKIP("the user has no supplementary group to use");

  std::string mounts;
  {
    std::ifstream mi("/proc/self/mountinfo");
    std::string l;
    while (std::getline(mi, l)) mounts += l + "\n";
  }
  INFO("Parent: dir=" << dir.path << " gid=" << gid << "\n" << dbg_status() << "mountinfo:\n" << mounts);
  const fs::path exe = dir.path / "hrr-integration-tests";
  std::error_code ec;
  fs::copy_file("/proc/self/exe", exe, ec);
  if (ec) HRR_SKIP("cannot copy this test binary: " << ec.message());
  if (chown(exe.c_str(), static_cast<uid_t>(-1), gid) != 0 || chmod(exe.c_str(), 0755) != 0)
    HRR_SKIP("cannot give the copy group " << gid);

  // Control: the same file without the set-group-ID bit, and with no
  // LD_LIBRARY_PATH, as the loader will treat it in secure-execution mode.
  // It has to capture, or a missing archive below would prove nothing.
  {
    const fs::path cap = dir.path / "control";
    hrr::test::SpawnProc proc(exe.string(), /*capture_stdout=*/true, /*capture_stderr=*/true);
    proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
    proc.setEnv("LD_LIBRARY_PATH", "");
    set_proc_search_path(proc);
    const int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
    const std::string out = proc.getOutput();
    INFO("Control output:\n" << out);
    if (ret != 0 || hrr_process_archives(cap).empty())
      HRR_SKIP("the capture runtime is not reachable without LD_LIBRARY_PATH (exit "
               << ret << (loader_failed(out) ? ", loader error" : "") << ")");
  }

  std::string probe_plain;
  {
    hrr::test::SpawnProc pp(exe.string(), true, true);
    pp.setEnv("LD_LIBRARY_PATH", "");
    probe_plain = (pp.run("\"Unit_HRR_SecureExecProbe_Direct\""), pp.getOutput());
  }
  INFO("Probe plain:\n" << probe_plain);
  struct stat st {};
  if (chmod(exe.c_str(), 02755) != 0 || stat(exe.c_str(), &st) != 0 || !(st.st_mode & S_ISGID))
    HRR_SKIP("cannot set the set-group-ID bit on the copy");

  const fs::path cap = dir.path / "secure";
  hrr::test::SpawnProc proc(exe.string(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  proc.setEnv("LD_LIBRARY_PATH", "");
  set_proc_search_path(proc);
  const int ret = proc.run("\"Unit_HRR_GpuWorkload_Direct\"");
  const std::string out = proc.getOutput();
  INFO("Set-group-ID output:\n" << out);
  std::string probe_sgid;
  {
    hrr::test::SpawnProc pp(exe.string(), true, true);
    pp.setEnv("LD_LIBRARY_PATH", "");
    probe_sgid = (pp.run("\"Unit_HRR_SecureExecProbe_Direct\""), pp.getOutput());
  }
  INFO("Probe setgid (mode " << std::oct << st.st_mode << std::dec << " gid " << st.st_gid << "):\n" << probe_sgid);
  CHECK(false);
  if (loader_failed(out)) HRR_SKIP("the loader refused the set-group-ID copy");
  REQUIRE(ret == 0);
  CHECK(hrr_process_archives(cap).empty());
  CHECK(out.find(kSecureExecNotice) != std::string::npos);
#endif
}

/**
 * End of HRR group
 * @}
 */
