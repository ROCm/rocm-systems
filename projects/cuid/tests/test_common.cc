// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_common.h"

#include <dirent.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cctype>
#include <cerrno>
#include <climits>
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

// ---------------------------------------------------------------------------
// ColdLookupEnvironment
// ---------------------------------------------------------------------------

namespace {

std::vector<ColdLookup> g_cold_lookups;

// The GPUs the machine actually has, read out of sysfs rather than assumed.
// The tests this feeds used to name "0000:03:00.0" and "/dev/dri/renderD128"
// literally, which is a different device on every machine and no device at all
// on most, and then accepted DEVICE_NOT_FOUND as a pass -- so they could not
// fail, and did not, while the lookup they cover was returning DEVICE_NOT_FOUND
// for every GPU on the node.
std::vector<ColdLookup> DiscoverGpusFromSysfs() {
  std::vector<ColdLookup> found;

  DIR* drm = opendir("/sys/class/drm");
  if (!drm) return found;

  while (const dirent* e = readdir(drm)) {
    const std::string name(e->d_name);
    // cardN and nothing else: cardN-DP-1 and the like are connectors.
    if (name.compare(0, 4, "card") != 0 || name.size() < 5) continue;
    bool digits = true;
    for (size_t i = 4; i < name.size(); ++i) {
      if (!isdigit(static_cast<unsigned char>(name[i]))) digits = false;
    }
    if (!digits) continue;

    ColdLookup entry;
    entry.card_path = "/sys/class/drm/" + name;

    // The BDF is the name of the directory the device symlink lands in.
    char resolved[PATH_MAX];
    if (realpath((entry.card_path + "/device").c_str(), resolved) == nullptr) continue;
    const std::string pci_dir(resolved);
    const size_t slash = pci_dir.find_last_of('/');
    if (slash == std::string::npos) continue;
    entry.bdf = pci_dir.substr(slash + 1);
    // A BDF is domain:bus:device.function; anything else is not a PCI device.
    if (entry.bdf.size() != 12 || entry.bdf[4] != ':' || entry.bdf[7] != ':') continue;

    // The render node beside it, where the driver publishes one.
    if (DIR* drm_dir = opendir((pci_dir + "/drm").c_str())) {
      while (const dirent* r = readdir(drm_dir)) {
        if (std::strncmp(r->d_name, "renderD", 7) == 0) {
          entry.node_path = std::string("/dev/dri/") + r->d_name;
          break;
        }
      }
      closedir(drm_dir);
    }

    found.push_back(entry);
  }
  closedir(drm);
  return found;
}

void RecordColdLookup(amdcuid_status_t status, const amdcuid_id_t& handle,
                      amdcuid_status_t* status_out, std::string* cuid_out) {
  *status_out = status;
  if (status != AMDCUID_STATUS_SUCCESS) return;
  const char* text = amdcuid_id_to_string(handle);
  *cuid_out = text ? text : "";
}

}  // namespace

void ColdLookupEnvironment::SetUp() {
  g_cold_lookups = DiscoverGpusFromSysfs();

  for (ColdLookup& entry : g_cold_lookups) {
    amdcuid_id_t handle = {};
    RecordColdLookup(
        amdcuid_get_handle_by_bdf(entry.bdf.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle), handle,
        &entry.bdf_status, &entry.bdf_cuid);

    // Immediately, on the handle just returned and before anything else has
    // enumerated: the manager's index is rebuilt wholesale by discovery, so a
    // query made after that would be answered from the rebuilt index and could
    // not tell whether the cold path had indexed the device at all.
    if (entry.bdf_status == AMDCUID_STATUS_SUCCESS) {
      uint32_t length = sizeof(entry.bdf_query_type);
      entry.bdf_query_status = amdcuid_query_device_property(
          handle, AMDCUID_QUERY_DEVICE_TYPE, &entry.bdf_query_type, &length);
    }

    handle = amdcuid_id_t{};
    RecordColdLookup(
        amdcuid_get_handle_by_dev_path(entry.card_path.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle),
        handle, &entry.card_status, &entry.card_cuid);

    if (entry.node_path.empty()) continue;
    handle = amdcuid_id_t{};
    RecordColdLookup(
        amdcuid_get_handle_by_dev_path(entry.node_path.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle),
        handle, &entry.node_status, &entry.node_cuid);
  }
}

const std::vector<ColdLookup>& ColdLookupEnvironment::results() { return g_cold_lookups; }
