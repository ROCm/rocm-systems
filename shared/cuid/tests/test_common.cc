// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include "test_common.h"

#include <dirent.h>
#include <ftw.h>
#include <unistd.h>

#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include "src/cuid_util.h"

CUIDTstGlobals sCUIDGlvalues;

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

// Enumerate AMD GPUs in sysfs independently of the library so missing GPUs fail
// the tests.
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

    char resolved[PATH_MAX];
    if (realpath((entry.card_path + "/device").c_str(), resolved) == nullptr) continue;
    const std::string pci_dir(resolved);
    const size_t slash = pci_dir.find_last_of('/');
    if (slash == std::string::npos) continue;
    entry.bdf = pci_dir.substr(slash + 1);
    // A BDF is domain:bus:device.function; anything else is not a PCI device.
    if (entry.bdf.size() != 12 || entry.bdf[4] != ':' || entry.bdf[7] != ':') continue;
    std::ifstream vendor_file(pci_dir + "/vendor");
    std::string vendor;
    if (!(vendor_file >> vendor) || vendor != "0x1002") continue;

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
    RecordColdLookup(amdcuid_get_handle_by_bdf(entry.bdf.c_str(), AMDCUID_DEVICE_TYPE_GPU, &handle),
                     handle, &entry.bdf_status, &entry.bdf_cuid);

    // Query immediately so a later enumeration cannot hide a missing index entry.
    if (entry.bdf_status == AMDCUID_STATUS_SUCCESS) {
      uint32_t length = sizeof(entry.bdf_query_type);
      entry.bdf_query_status = amdcuid_query_device_property(handle, AMDCUID_QUERY_DEVICE_TYPE,
                                                             &entry.bdf_query_type, &length);
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

// std::filesystem needs -lstdc++fs on GCC 8, which some CI images still use.
ScopedTempDir::ScopedTempDir(const char* prefix) {
  const char* base = std::getenv("TMPDIR");
  std::string tmpl = std::string(base && *base ? base : "/tmp") + "/" + prefix + "XXXXXX";
  if (mkdtemp(&tmpl[0])) path_ = tmpl;
}

ScopedTempDir::~ScopedTempDir() {
  if (path_.empty()) return;
  nftw(
      path_.c_str(),
      [](const char* path, const struct stat*, int, struct FTW*) { return ::remove(path); }, 16,
      FTW_DEPTH | FTW_PHYS);
}

namespace {

template <typename T>
bool QueryValue(const amdcuid_id_t& handle, amdcuid_query_t query, T& value) {
  uint32_t length = sizeof(value);
  return amdcuid_query_device_property(handle, query, &value, &length) == AMDCUID_STATUS_SUCCESS;
}

}  // namespace

bool RebuiltTemporaryCuid(const amdcuid_id_t& handle, amdcuid_id_t& out) {
  amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
  if (!QueryValue(handle, AMDCUID_QUERY_DEVICE_TYPE, type)) return false;
  CuidUtilities::AuxiliaryInput input;
  input.component_type = static_cast<uint8_t>(type);
  if (type == AMDCUID_DEVICE_TYPE_PLATFORM) {
    input.format = CuidUtilities::kAuxFormatCpu;
  } else {
    uint16_t vendor_id = 0, device_id = 0, unit_id = 0;
    uint8_t revision_id = 0;
    if (!QueryValue(handle, AMDCUID_QUERY_VENDOR_ID, vendor_id) ||
        !QueryValue(handle, AMDCUID_QUERY_DEVICE_ID, device_id) ||
        !QueryValue(handle, AMDCUID_QUERY_REVISION_ID, revision_id) ||
        !QueryValue(handle, AMDCUID_QUERY_UNIT_ID, unit_id))
      return false;
    input.vendor_id = vendor_id;
    input.device_id = device_id;
    input.revision_id = revision_id;
    input.unit_id = unit_id;
    if (type == AMDCUID_DEVICE_TYPE_CPU) {
      uint16_t physical_id = 0;
      if (!QueryValue(handle, AMDCUID_QUERY_PHYSICAL_ID, physical_id)) return false;
      input.format = CuidUtilities::kAuxFormatCpu;
      input.routing_id = physical_id;
    } else {
      char bdf[32] = {0};
      uint32_t length = sizeof(bdf);
      if (amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, bdf, &length) !=
          AMDCUID_STATUS_SUCCESS)
        return false;
      input.format = CuidUtilities::kAuxFormatPcie;
      CuidUtilities::set_pcie_routing(input, bdf);
    }
  }
  amdcuid_derived_id derived{};
  if (CuidUtilities::temporary_cuid(input, &derived) != AMDCUID_STATUS_SUCCESS) return false;
  out = derived.UUIDv8_representation;
  return true;
}

bool GpuPartition(const amdcuid_id_t& handle) {
  amdcuid_device_type_t type = AMDCUID_DEVICE_TYPE_NONE;
  if (!QueryValue(handle, AMDCUID_QUERY_DEVICE_TYPE, type) || type != AMDCUID_DEVICE_TYPE_GPU)
    return false;
  char bdf[32] = {0};
  uint32_t length = sizeof(bdf);
  return amdcuid_query_device_property(handle, AMDCUID_QUERY_BDF, bdf, &length) !=
         AMDCUID_STATUS_SUCCESS;
}

bool TemporaryWithoutPrimary(const amdcuid_id_t& handle) {
  bool temporary = false;
  if (!QueryValue(handle, AMDCUID_QUERY_TEMPORARY_CUID, temporary) || !temporary) return false;
  amdcuid_id_t primary;
  std::memset(&primary, 0xA5, sizeof(primary));
  const amdcuid_id_t untouched = primary;
  uint32_t length = sizeof(primary);
  EXPECT_EQ(amdcuid_query_device_property(handle, AMDCUID_QUERY_PRIMARY_CUID, &primary, &length),
            AMDCUID_STATUS_UNSUPPORTED)
      << amdcuid_id_to_string(handle);
  EXPECT_EQ(std::memcmp(&primary, &untouched, sizeof(primary)), 0)
      << "a temporary component's refused primary query wrote " << amdcuid_id_to_string(primary);
  return true;
}
