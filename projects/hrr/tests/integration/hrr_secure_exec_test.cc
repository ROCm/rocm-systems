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
 *     print the notice. No root is needed. Secure-execution mode ignores
 *     LD_LIBRARY_PATH, so the copy only tests this build when the loader finds
 *     the same libamdhip64 without it (an installed ROCm, or a loader cache
 *     that lists the build). The case skips when it does not, when the
 *     filesystem is mounted nosuid, when no_new_privs is set, and when the
 *     kernel does not mark the copy secure.
 */

#include "hrr_test_common.hh"
#include "hrr_test_process.hh"

// Cross-check only: make every skip visible in the job log.
#undef HRR_SKIP
#define HRR_SKIP(m) FAIL("SKIP: " << m)

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#if defined(__linux__)
#include <link.h>
#include <sys/auxv.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
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
constexpr const char* kProbeRuntime = "hrr-probe hip-runtime=";
constexpr const char* kProbeSecure = "hrr-probe at-secure=";

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

// The libamdhip64 the loader gave this process, as a canonical path.
std::string loaded_hip_runtime() {
  std::string found;
  dl_iterate_phdr(
      [](dl_phdr_info* info, size_t, void* data) {
        if (info->dlpi_name == nullptr || std::strstr(info->dlpi_name, "libamdhip64") == nullptr)
          return 0;
        std::error_code ec;
        *static_cast<std::string*>(data) = fs::canonical(info->dlpi_name, ec).string();
        return 1;
      },
      &found);
  return found;
}

// The rest of the line after `key` in a child's output, or "" if absent.
std::string probe_value(const std::string& out, const char* key) {
  const size_t at = out.find(key);
  if (at == std::string::npos) return {};
  const size_t begin = at + std::strlen(key);
  return out.substr(begin, out.find('\n', begin) - begin);
}

struct CopyRun {
  int ret;
  std::string out;
};

// Runs the probe and the GPU workload in the copy, capturing to `cap`, with no
// LD_LIBRARY_PATH: secure-execution mode ignores it, so the control has to do
// without it too.
CopyRun run_copy(const fs::path& exe, const fs::path& cap) {
  hrr::test::SpawnProc proc(exe.string(), /*capture_stdout=*/true, /*capture_stderr=*/true);
  proc.setEnv("HIP_HRR_CAPTURE_OUTPUT", cap.string());
  proc.setEnv("LD_LIBRARY_PATH", "");
  set_proc_search_path(proc);
  // Unquoted: Catch2 reads a quoted filter as one name, comma included.
  const int ret = proc.run("Unit_HRR_SecureExecProbe_Direct,Unit_HRR_GpuWorkload_Direct");
  return {ret, proc.getOutput()};
}

}  // namespace

// Hidden: reports what the loader and the kernel gave this process, for the
// case below. It makes no HIP call.
TEST_CASE("Unit_HRR_SecureExecProbe_Direct", "[.][hrr-direct]") {
  std::printf("%s%s\n%s%lu\n", kProbeRuntime, loaded_hip_runtime().c_str(), kProbeSecure,
              getauxval(AT_SECURE));
  std::fflush(stdout);
}
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

  const fs::path exe = dir.path / "hrr-integration-tests";
  std::error_code ec;
  fs::copy_file("/proc/self/exe", exe, ec);
  if (ec) HRR_SKIP("cannot copy this test binary: " << ec.message());
  if (chown(exe.c_str(), static_cast<uid_t>(-1), gid) != 0 || chmod(exe.c_str(), 0755) != 0)
    HRR_SKIP("cannot give the copy group " << gid);

  // Control: the same file without the set-group-ID bit. It has to load this
  // build's runtime and capture, or a missing archive below would prove nothing.
  // Every skip comes before the first assertion, so a pass means the gate ran.
  const std::string runtime = loaded_hip_runtime();
  const fs::path control_cap = dir.path / "control";
  const CopyRun control = run_copy(exe, control_cap);
  INFO("Control output:\n" << control.out);
  const std::string control_loaded = probe_value(control.out, kProbeRuntime);
  if (control_loaded.empty())
    HRR_SKIP("the copy does not start without LD_LIBRARY_PATH (exit " << control.ret << ")");
  if (control_loaded != runtime)
    HRR_SKIP("without LD_LIBRARY_PATH the copy loads " << control_loaded << ", not " << runtime);

  struct stat st {};
  if (chmod(exe.c_str(), 02755) != 0 || stat(exe.c_str(), &st) != 0 || !(st.st_mode & S_ISGID))
    HRR_SKIP("cannot set the set-group-ID bit on the copy");

  const fs::path cap = dir.path / "secure";
  const CopyRun run = run_copy(exe, cap);
  INFO("Set-group-ID output:\n" << run.out);
  const std::string loaded = probe_value(run.out, kProbeRuntime);
  if (loaded.empty()) HRR_SKIP("the loader refused the set-group-ID copy (exit " << run.ret << ")");
  if (probe_value(run.out, kProbeSecure) != "1")
    HRR_SKIP("the set-group-ID copy did not start in secure-execution mode");
  if (loaded != runtime)
    HRR_SKIP("in secure-execution mode the copy loads " << loaded << ", not " << runtime);

  REQUIRE(control.ret == 0);
  REQUIRE_FALSE(hrr_process_archives(control_cap).empty());
  REQUIRE(run.ret == 0);
  CHECK(hrr_process_archives(cap).empty());
  CHECK(run.out.find(kSecureExecNotice) != std::string::npos);
#endif
}

/**
 * End of HRR group
 * @}
 */
