// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_common.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "src/cuid_util.h"

CUIDTstGlobals sCUIDGlvalues;

namespace {

// The temporary record directory, or nullptr when this process is root.
//
// A bare pointer with static (zero) initialisation, because it is written from
// a constructor function that runs before dynamic initialisation: a std::string
// here would be constructed after the value had already been stored into it.
std::string* g_record_dir = nullptr;

// Create the directory and publish it in the environment before anything else
// in this binary initialises.
//
// cuid.cc has a namespace-scope reference that constructs CuidDeviceManager
// during static initialisation; its members are default-initialised from
// CuidUtilities::cuid_file(), which caches record_dir() in a function-local
// static on that first call. By the time main() runs the store path is already
// decided and a setenv() there has no effect at all. A constructor priority in
// the reserved 101-65535 range runs ahead of every unprioritised initialiser,
// and RecordStoreEnvironment::SetUp() asserts the override took, so a
// regression in initialisation order fails the run rather than sending the
// unprivileged suite at the node's real store.
__attribute__((constructor(101))) void CreateTemporaryRecordDir() {
  // Root uses the real store: record_dir() ignores the environment for euid 0,
  // and the privileged tests exercise the real path.
  if (geteuid() == 0) return;

  // NOLINTNEXTLINE(concurrency-mt-unsafe) - single-threaded, before main()
  const char* tmp = std::getenv("TMPDIR");
  std::string tmpl = std::string((tmp && tmp[0]) ? tmp : "/tmp") + "/amdcuid_test.XXXXXX";
  std::vector<char> buf(tmpl.begin(), tmpl.end());
  buf.push_back('\0');
  if (mkdtemp(buf.data()) == nullptr) {
    std::fprintf(stderr, "amdcuid_test: cannot create a temporary record directory: %s\n",
                 std::strerror(errno));
    return;
  }

  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  if (setenv("AMDCUID_RECORD_DIR", buf.data(), 1) != 0) {
    std::fprintf(stderr, "amdcuid_test: cannot set AMDCUID_RECORD_DIR: %s\n", std::strerror(errno));
    (void)rmdir(buf.data());
    return;
  }
  g_record_dir = new std::string(buf.data());
}

}  // namespace

void RecordStoreEnvironment::SetUp() {
  // The override is ignored for root, which is what leaves the privileged tests
  // on the real /var/lib/amdcuid.
  if (geteuid() == 0) return;

  ASSERT_NE(g_record_dir, nullptr) << "the temporary record directory was never created";
  dir_ = *g_record_dir;

  ASSERT_EQ(CuidUtilities::record_dir(), dir_)
      << "the library resolved its record store before the suite could redirect it; "
         "the unprivileged tests would read and write the node's real records";
  ASSERT_EQ(CuidUtilities::cuid_file(), dir_ + "/cuid");
  ASSERT_EQ(CuidUtilities::priv_cuid_file(), dir_ + "/priv_cuid");
}

void RecordStoreEnvironment::TearDown() {
  if (dir_.empty()) return;

  // cuid, priv_cuid, their .lock files and any temp file a failed write
  // abandoned. One level deep is all there ever is.
  if (DIR* d = opendir(dir_.c_str())) {
    while (const dirent* e = readdir(d)) {
      const std::string name = e->d_name;
      if (name == "." || name == "..") continue;
      (void)std::remove((dir_ + "/" + name).c_str());
    }
    closedir(d);
  }
  (void)rmdir(dir_.c_str());
  dir_.clear();
}

static void print_help() {
  printf(
      "amdcuid_test: CUID test suite\n"
      "\n"
      "Usage: amdcuid_test [options] [gtest options]\n"
      "\n"
      "Options:\n"
      "  -v, --verbose      Increase output verbosity (may be repeated)\n"
      "  -f, --dont_fail    Continue on assertion failures instead of "
      "aborting\n"
      "  -h, --help         Show this help message\n"
      "\n"
      "GoogleTest flags must use --gtest_*; all other unrecognised options are ignored.\n");
}

void ProcessCmdline(CUIDTstGlobals* globals, int argc, char** argv) {
  for (int i = 1; i < argc; ++i) {
    if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--verbose") == 0) {
      globals->verbosity++;
    } else if (strcmp(argv[i], "-f") == 0 || strcmp(argv[i], "--dont_fail") == 0) {
      globals->dont_fail = true;
    } else if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
      print_help();
      exit(0);
    }
    // Unrecognised flags are currently ignored (GoogleTest is initialized before ProcessCmdline()).
  }
}
