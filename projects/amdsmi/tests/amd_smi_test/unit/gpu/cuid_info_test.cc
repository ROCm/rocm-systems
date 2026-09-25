// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

/*
 * Tests for the CUID API surface.
 *
 * These run with or without CUID support compiled in, and with or without a
 * GPU: most of them assert that the ABI behaves the same either way, which is
 * what lets a consumer call these entry points unconditionally. Checks that
 * need a device are skipped rather than failed.
 */

#include <fcntl.h>
#include <gtest/gtest.h>
#include <linux/capability.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"

namespace {

bool HasCapSysAdmin() {
  __user_cap_header_struct header{_LINUX_CAPABILITY_VERSION_3, 0};
  __user_cap_data_struct data[_LINUX_CAPABILITY_U32S_3] = {};
  if (syscall(SYS_capget, &header, data) != 0) return false;
  return (data[CAP_SYS_ADMIN / 32].effective & (1U << (CAP_SYS_ADMIN % 32))) != 0;
}

// A CUID is rendered as the standard 8-4-4-4-12 UUID string.
bool LooksLikeUuid(const char* value) {
  const std::string s(value);
  if (s.size() != 36) return false;
  for (size_t i = 0; i < s.size(); ++i) {
    const char c = s[i];
    if (i == 8 || i == 13 || i == 18 || i == 23) {
      if (c != '-') return false;
    } else if (!std::isxdigit(static_cast<unsigned char>(c))) {
      return false;
    }
  }
  return true;
}

// Payload bit 117 (the Auxiliary Value Identifier, payload octet 14 mask 0x20)
// read back out of a rendered UUIDv8 string. The framing puts payload bits
// 120:121 in the low two bits of the last rendered octet and shifts the rest
// along by six: that octet is
// `((payload[14] & 0x3F) << 2) | (payload[15] & 0x03)`, so payload bit 117
// lands in bit 7 of the last rendered octet.
//
// Checked against all thirteen vectors in
// shared/cuid/tests/vectors/cuid_vectors.txt.
bool AuxiliaryBitFromUuidString(const std::string& uuid, bool* aux) {
  std::string hex;
  for (char c : uuid) {
    if (c != '-') hex.push_back(c);
  }
  if (hex.size() != 32) return false;

  const int last = std::stoi(hex.substr(30, 2), nullptr, 16);
  *aux = (last & 0x80) != 0;
  return true;
}

// The BDF string the driver's sysfs directory is named after, and the one
// amd-smi hands libamdcuid. Mirrors stringify_bdf() in
// src/amd_smi/amd_smi_utils.cc, which is internal to the library.
std::string BdfString(const amdsmi_bdf_t& bdf) {
  char out[32] = {};
  snprintf(out, sizeof(out), "%04x:%02x:%02x.%x", static_cast<unsigned>(bdf.domain_number),
           static_cast<unsigned>(bdf.bus_number), static_cast<unsigned>(bdf.device_number),
           static_cast<unsigned>(bdf.function_number));
  return out;
}

// A private directory under TMPDIR, removed when the test finishes.
class ScopedTempDir {
 public:
  ScopedTempDir() {
    const char* tmp = std::getenv("TMPDIR");
    std::string tmpl = std::string(tmp && tmp[0] ? tmp : "/tmp") + "/amdsmi-cuid-XXXXXX";
    std::vector<char> buf(tmpl.begin(), tmpl.end());
    buf.push_back('\0');
    path_ = mkdtemp(buf.data()) ? buf.data() : "";
  }
  ~ScopedTempDir() {
    if (!path_.empty()) {
      // Only ever holds the fabricated sysfs tree created below.
      std::error_code ignored;
      std::filesystem::remove_all(path_, ignored);
    }
  }
  ScopedTempDir(const ScopedTempDir&) = delete;
  ScopedTempDir& operator=(const ScopedTempDir&) = delete;

  const std::string& path() const { return path_; }
  bool valid() const { return !path_.empty(); }

 private:
  std::string path_;
};

// Sets AMDSMI_CUID_SYSFS_ROOT for the duration of a test and restores whatever
// was there, so one case cannot change what a later one sees.
class ScopedSysfsRoot {
 public:
  explicit ScopedSysfsRoot(const std::string& root) {
    const char* previous = std::getenv(kVar);
    had_previous_ = previous != nullptr;
    if (had_previous_) previous_ = previous;
    setenv(kVar, root.c_str(), 1);
  }
  ~ScopedSysfsRoot() {
    if (had_previous_) {
      setenv(kVar, previous_.c_str(), 1);
    } else {
      unsetenv(kVar);
    }
  }
  ScopedSysfsRoot(const ScopedSysfsRoot&) = delete;
  ScopedSysfsRoot& operator=(const ScopedSysfsRoot&) = delete;

 private:
  static constexpr const char* kVar = "AMDSMI_CUID_SYSFS_ROOT";
  bool had_previous_ = false;
  std::string previous_;
};

// mkdir -p, for the fabricated sysfs tree.
bool MakeDirs(const std::string& path) {
  std::string partial;
  for (size_t i = 1; i <= path.size(); ++i) {
    if (i == path.size() || path[i] == '/') {
      partial = path.substr(0, i);
      if (mkdir(partial.c_str(), 0755) != 0 && errno != EEXIST) return false;
    }
  }
  return true;
}

constexpr uint32_t kNoPartition = 0xFFFFFFFF;

// SPX partition zero shares the PCI device but has its own publication. Without
// XCP publication, only the first/unpartitioned processor may use the whole GPU.
std::string DriverPublished(const std::string& root, const std::string& bdf, uint32_t render,
                            uint32_t partition) {
  std::string path =
      root + "/class/drm/renderD" + std::to_string(render) + "/device/xcp/cuid_derived";
  if (render == kNoPartition || access(path.c_str(), F_OK) != 0) {
    if (partition != 0 && partition != kNoPartition) return "";
    path = root + "/bus/pci/devices/" + bdf + "/cuid_derived";
  }
  std::ifstream in(path);
  if (!in) return "";
  std::string value;
  std::getline(in, value);
  while (!value.empty() && (value.back() == '\n' || value.back() == '\r' || value.back() == ' ')) {
    value.pop_back();
  }
  return value;
}

std::string DriverPublished(amdsmi_processor_handle handle, const std::string& bdf) {
  amdsmi_enumeration_info_t enumeration = {};
  if (amdsmi_get_gpu_enumeration_info(handle, &enumeration) != AMDSMI_STATUS_SUCCESS)
    enumeration.drm_render = kNoPartition;
  amdsmi_kfd_info_t kfd = {};
  if (amdsmi_get_gpu_kfd_info(handle, &kfd) != AMDSMI_STATUS_SUCCESS)
    kfd.current_partition_id = kNoPartition;
  return DriverPublished("/sys", bdf, enumeration.drm_render, kfd.current_partition_id);
}

std::vector<amdsmi_processor_handle> GpuHandles() {
  std::vector<amdsmi_processor_handle> handles;

  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) return handles;
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return handles;
  }

  for (auto socket : sockets) {
    uint32_t count = 0;
    if (amdsmi_get_processor_handles(socket, &count, nullptr) != AMDSMI_STATUS_SUCCESS) continue;
    std::vector<amdsmi_processor_handle> procs(count);
    if (amdsmi_get_processor_handles(socket, &count, procs.data()) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    for (auto proc : procs) {
      processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(proc, &type) != AMDSMI_STATUS_SUCCESS) continue;
      if (type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) handles.push_back(proc);
    }
  }
  return handles;
}

// current_partition_id is also 0 on an unpartitioned card; the profile's
// partition count tells the two apart.
uint32_t PartitionId(amdsmi_processor_handle handle) {
  amdsmi_accelerator_partition_profile_t profile = {};
  uint32_t ids[AMDSMI_MAX_ACCELERATOR_PARTITIONS] = {};
  if (amdsmi_get_gpu_accelerator_partition_profile(handle, &profile, ids) !=
      AMDSMI_STATUS_SUCCESS) {
    return kNoPartition;
  }
  if (profile.num_partitions <= 1) return kNoPartition;

  amdsmi_kfd_info_t kfd_info = {};
  if (amdsmi_get_gpu_kfd_info(handle, &kfd_info) != AMDSMI_STATUS_SUCCESS) return kNoPartition;
  return kfd_info.current_partition_id;
}

// On any failure amdsmi_get_gpu_cuid_info() must report nothing at all: in a
// partly filled struct every field reads as determined, and auxiliary = 0 in
// particular is indistinguishable from a device that is not auxiliary.
void ExpectNoCuidReported(const amdsmi_cuid_info_t& info) {
  EXPECT_EQ(info.derived[0], '\0') << "a failed query must not report a derived CUID";
  EXPECT_EQ(info.primary[0], '\0') << "a failed query must not report a primary CUID";
  EXPECT_EQ(info.auxiliary, 0) << "a failed query must not report an auxiliary verdict";
  EXPECT_EQ(info.component_type, AMDSMI_CUID_COMPONENT_UNKNOWN);
  EXPECT_EQ(info.source, AMDSMI_CUID_SOURCE_UNKNOWN)
      << "a failed query must not claim a provenance";
}

// RAII init/shutdown. A fixture would be the obvious way, but every other test
// under unit/gpu/ uses the bare TEST() macro against the GpuUnit suite, and
// GTest rejects a suite that mixes TEST and TEST_F.
class AmdSmiSession {
 public:
  AmdSmiSession() : status_(amdsmi_init(AMDSMI_INIT_AMD_GPUS)) {}
  ~AmdSmiSession() {
    if (status_ == AMDSMI_STATUS_SUCCESS) amdsmi_shut_down();
  }
  AmdSmiSession(const AmdSmiSession&) = delete;
  AmdSmiSession& operator=(const AmdSmiSession&) = delete;

  amdsmi_status_t status() const { return status_; }

 private:
  amdsmi_status_t status_;
};

// Whether this build linked libamdcuid. The ABI is identical either way, so
// nothing at runtime distinguishes "built without the library" from "the
// library is here but has nothing to say about this device". The build system
// passes BUILD_CUID to this target so the cases below can assert
// not-supported where it is required rather than skipping on it.
#ifdef BUILD_CUID
constexpr bool kCuidBuiltIn = true;
#else
constexpr bool kCuidBuiltIn = false;
#endif

constexpr char kKeyVariable[] =
    "/sys/firmware/efi/efivars/AmdCuidKey-e41c1f7f-63cb-46b9-bf27-36a55f92a06d";

bool ReadExactly(const std::string& path, std::string& out, size_t size) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  out.assign(size + 1, '\0');
  file.read(&out[0], static_cast<std::streamsize>(out.size()));
  if (static_cast<size_t>(file.gcount()) != size) return false;
  out.resize(size);
  return true;
}

// Whether libamdcuid can see a node key from this process, and whether an
// administrator set it: any amdgpu cuid_seed, else a well-formed AmdCuidKey.
bool NodeKey(bool* provisioned) {
  if (geteuid() != 0) return false;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices", ec)) {
    std::string seed;
    if (!ReadExactly(entry.path().string() + "/cuid_seed", seed, AMDSMI_CUID_SEED_SIZE)) continue;
    std::string state;
    std::ifstream(entry.path().string() + "/cuid_seed_state") >> state;
    *provisioned = state == "provisioned";
    return true;
  }
  std::string variable;
  if (!ReadExactly(kKeyVariable, variable, 4 + 36)) return false;
  const auto* payload = reinterpret_cast<const uint8_t*>(variable.data()) + 4;
  if (payload[0] != 1 || (payload[1] & 0xfe) != 0 || payload[2] != 0 || payload[3] != 0)
    return false;
  *provisioned = (payload[1] & 1) != 0;
  return true;
}

// A non-root caller is refused rather than told there is no key, and a host
// without one fails rather than reporting a made-up state.
amdsmi_status_t ExpectedSeedInfoStatus() {
  if (geteuid() != 0) return AMDSMI_STATUS_NO_PERM;
  bool provisioned = false;
  return NodeKey(&provisioned) ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_API_FAILED;
}

// On any failure the output struct must come back untouched. The dangerous
// wrong answer is not a garbage fingerprint but a plausible one next to
// provisioned = 0, which reads as a key amdgpu generated.
void ExpectNothingReported(const amdsmi_cuid_seed_info_t& info) {
  EXPECT_EQ(info.provisioned, 0) << "a failed query must not report a provisioning state";
  for (size_t i = 0; i < sizeof(info.fingerprint); ++i) {
    EXPECT_EQ(info.fingerprint[i], 0)
        << "a failed query must not report a fingerprint (octet " << i << ")";
  }
}

}  // namespace

// A null argument is a caller error, not a "not supported": this must be true
// whether or not the CUID library was linked, so that a caller cannot tell the
// two apart by passing garbage.
TEST(GpuUnit, CuidNullArgumentsRejected) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  EXPECT_EQ(amdsmi_get_gpu_cuid_info(nullptr, nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_set_cuid_seed(nullptr), AMDSMI_STATUS_INVAL);
  EXPECT_EQ(amdsmi_get_cuid_seed_info(nullptr), AMDSMI_STATUS_INVAL);
  amdsmi_cuid_component_t component = {};
  EXPECT_EQ(amdsmi_get_cuid_components(nullptr, &component), AMDSMI_STATUS_INVAL);
}

// Built without libamdcuid, every CUID entry point reports not-supported, with
// valid arguments, so this is not the null-argument case in disguise. The
// symbols are exported either way; only the answer differs. This is the one
// build where the feature can regress to a link error or to a status nobody
// checks.
TEST(GpuUnit, CuidEntryPointsNotSupportedWithoutTheLibrary) {
  if (kCuidBuiltIn) GTEST_SKIP() << "built with CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_cuid_seed_info_t seed_info = {};
  EXPECT_EQ(amdsmi_get_cuid_seed_info(&seed_info), AMDSMI_STATUS_NOT_SUPPORTED);

  const uint8_t seed[AMDSMI_CUID_SEED_SIZE] = {};
  EXPECT_EQ(amdsmi_set_cuid_seed(seed), AMDSMI_STATUS_NOT_SUPPORTED);

  uint32_t count = 7;
  EXPECT_EQ(amdsmi_get_cuid_components(&count, nullptr), AMDSMI_STATUS_NOT_SUPPORTED);
  EXPECT_EQ(count, 0u);

  // A real handle where there is one. Without the library neither device entry
  // point dereferences the handle before returning, so a GPU-less machine
  // still exercises them.
  auto handles = GpuHandles();
  if (handles.empty()) handles.push_back(reinterpret_cast<amdsmi_processor_handle>(&seed_info));

  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    EXPECT_EQ(amdsmi_get_gpu_cuid_info(handle, &info), AMDSMI_STATUS_NOT_SUPPORTED);

    char cuid[AMDSMI_GPU_CUID_SIZE] = {};
    unsigned int length = sizeof(cuid);
    EXPECT_EQ(amdsmi_get_gpu_device_cuid(handle, &length, cuid), AMDSMI_STATUS_NOT_SUPPORTED);
  }
}

// Built without libamdcuid the entry points still exist and report
// not-supported. Built with it they answer, and which answer is right is
// decided by the node key and the caller, so assert against those.
TEST(GpuUnit, CuidSeedInfoAnsweredOrUnsupported) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_cuid_seed_info_t info = {};
  const amdsmi_status_t status = amdsmi_get_cuid_seed_info(&info);

  if (!kCuidBuiltIn) {
    ASSERT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
    return;
  }

  ASSERT_EQ(status, ExpectedSeedInfoStatus()) << "euid " << geteuid();
  if (status != AMDSMI_STATUS_SUCCESS) {
    ExpectNothingReported(info);
    return;
  }

  bool any_set = false;
  for (uint8_t octet : info.fingerprint) {
    if (octet != 0) any_set = true;
  }
  EXPECT_TRUE(any_set) << "seed fingerprint should never be all zeroes";

  bool provisioned = false;
  ASSERT_TRUE(NodeKey(&provisioned));
  EXPECT_EQ(info.provisioned != 0, provisioned);
}

// The seed is write-only through this API. If a future change adds a way to
// read it back, the struct grows a field and this stops being true.
TEST(GpuUnit, CuidSeedInfoCarriesNoSeedMaterial) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  // The shape of the struct is the requirement, and it holds in every state
  // and in a build without the library: 8 octets of digest, and nothing else
  // that could hold 32 octets of secret.
  amdsmi_cuid_seed_info_t info = {};
  EXPECT_EQ(sizeof(info.fingerprint), 8u);
  EXPECT_LT(sizeof(info.fingerprint), 32u);

  const amdsmi_status_t status = amdsmi_get_cuid_seed_info(&info);
  if (!kCuidBuiltIn) {
    ASSERT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
    return;
  }

  ASSERT_EQ(status, ExpectedSeedInfoStatus()) << "euid " << geteuid();

  // A failed query discloses less, never more.
  if (status != AMDSMI_STATUS_SUCCESS) ExpectNothingReported(info);

  // The reserved space is 32 octets, exactly the width of a seed, and the one
  // place a future change could park seed material without widening the
  // fingerprint. Assert it comes back as it went in, in every state.
  for (size_t i = 0; i < sizeof(info.reserved_flags); ++i) {
    EXPECT_EQ(info.reserved_flags[i], 0) << "reserved_flags[" << i << "] carries something";
  }
  for (size_t i = 0; i < sizeof(info.reserved) / sizeof(info.reserved[0]); ++i) {
    EXPECT_EQ(info.reserved[i], 0u) << "reserved[" << i << "] carries something";
  }
}

// A root run would reach the node key if the refusal ever regressed, so the
// refusals themselves are covered by the library's unit tests.
TEST(GpuUnit, CuidSetSeedNeedsRoot) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
  if (geteuid() == 0) GTEST_SKIP() << "only an ordinary user can call it safely";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  uint8_t seed[AMDSMI_CUID_SEED_SIZE];
  for (size_t i = 0; i < sizeof(seed); ++i) seed[i] = static_cast<uint8_t>(0x40 + i * 7);
  EXPECT_EQ(amdsmi_set_cuid_seed(seed), AMDSMI_STATUS_NO_PERM);
}

// The decoder the auxiliary check below relies on, against the published
// conformance vectors: a decoder that reads the wrong octet agrees with a
// producer that writes the wrong octet. Needs neither a GPU nor libamdcuid.
//
// Payload octet 14 of each vector, mask 0x20, is the Auxiliary Value
// Identifier; the expectation below is that bit taken from
// shared/cuid/tests/vectors/cuid_vectors.txt.
TEST(GpuUnit, CuidAuxiliaryBitDecoderMatchesConformanceVectors) {
  const struct {
    const char* name;
    const char* uuid;
    bool auxiliary;
  } kVectors[] = {
      {"P-1", "d4abaad3-9b34-8c50-9800-028dcc084200", false},
      {"P-2", "ffeb5272-7771-88c8-b800-028dcc084200", false},
      {"U-1", "d4abaad3-9b34-8c50-988c-028dcc084204", false},
      {"T-PLATFORM", "d4abaad3-9b34-8c50-9800-028dcc084000", false},
      {"T-CPU", "d4abaad3-9b34-8c50-9800-028dcc084100", false},
      {"T-GPU", "d4abaad3-9b34-8c50-9800-028dcc084200", false},
      {"T-NIC", "d4abaad3-9b34-8c50-9800-028dcc084300", false},
      {"T-NPU", "d4abaad3-9b34-8c50-9800-028dcc084001", false},
      {"T-OTHER", "d4abaad3-9b34-8c50-9800-028dcc084303", false},
      {"D-1", "10133e37-3995-80bc-a402-7074c5c1c44c", false},
      {"D-2", "73488f9e-ea52-86ce-8401-2627fa41b068", false},
      {"A-1", "96f3618e-bffd-8ff3-9000-028dcc084280", true},
      {"A-2", "5afaa441-5d39-8795-bc02-5e6010667cf0", true},
  };

  for (const auto& vector : kVectors) {
    bool decoded = false;
    ASSERT_TRUE(AuxiliaryBitFromUuidString(vector.uuid, &decoded)) << vector.name;
    EXPECT_EQ(decoded, vector.auxiliary) << vector.name << ": " << vector.uuid;
  }
}

TEST(GpuUnit, CuidSnapshotIsSelfConsistent) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
    if (status == AMDSMI_STATUS_NOT_SUPPORTED) {
      GTEST_SKIP() << (kCuidBuiltIn ? "no CUID for this device"
                                    : "built without CUID support; asserted by "
                                      "CuidEntryPointsNotSupportedWithoutTheLibrary");
    }
    // NO_PERM leaves no snapshot to check for consistency.
    if (status == AMDSMI_STATUS_NO_PERM) {
      GTEST_SKIP() << "this caller may not read the whole snapshot; run as root";
    }
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);

    // The derived CUID is the value an unprivileged caller is meant to get, so
    // it is always populated on success.
    EXPECT_TRUE(LooksLikeUuid(info.derived)) << "derived: " << info.derived;

    // Version nibble is always 8, auxiliary or not. A consumer parses every
    // CUID with one code path.
    EXPECT_EQ(info.derived[14], '8') << "derived: " << info.derived;

    EXPECT_EQ(info.component_type, AMDSMI_CUID_COMPONENT_GPU);

    // The reported auxiliary flag must agree with payload bit 117 of the value
    // it describes. A producer reading the flag out of the wrong place emits an
    // auxiliary value that reports itself as canonical.
    bool aux_from_value = false;
    ASSERT_TRUE(AuxiliaryBitFromUuidString(info.derived, &aux_from_value));
    EXPECT_EQ(static_cast<bool>(info.auxiliary), aux_from_value);

    // libamdcuid gates its primary query on an effective UID of zero, and the
    // driver's cuid_primary on CAP_SYS_ADMIN, so a non-root caller must get the
    // empty string and a root caller holding CAP_SYS_ADMIN must get a value.
    // Asserting only "empty or well-formed" would accept both answers from both
    // callers, including a snapshot handing an unprivileged process the
    // serial-bearing primary. Each run asserts the branch it is in; root
    // without the capability (a default container) may get either.
    if (geteuid() == 0) {
      if (HasCapSysAdmin())
        EXPECT_NE(info.primary[0], '\0') << "root caller should receive the primary CUID";
      if (info.primary[0] != '\0') {
        EXPECT_TRUE(LooksLikeUuid(info.primary)) << "primary: " << info.primary;
        EXPECT_EQ(info.primary[14], '8');
      }
    } else {
      EXPECT_EQ(info.primary[0], '\0')
          << "unprivileged caller should receive an empty primary, got: " << info.primary;
    }
  }
}

// Where the driver really does publish cuid_derived, the derived CUID that
// comes back is the driver's value verbatim and the source says so: the kernel
// and the library producing different values for one device is the failure the
// driver stage exists to prevent. Only checkable where the attribute is real,
// so it is skipped where it is not; no fabricated root, no build flag.
TEST(GpuUnit, CuidDriverPublishedValueIsUsedVerbatim) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  size_t checked = 0;
  for (auto handle : handles) {
    amdsmi_bdf_t bdf = {};
    ASSERT_EQ(amdsmi_get_gpu_device_bdf(handle, &bdf), AMDSMI_STATUS_SUCCESS);
    const std::string bdf_str = BdfString(bdf);

    const std::string driver_value = DriverPublished(handle, bdf_str);
    if (driver_value.empty()) continue;

    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
    if (status == AMDSMI_STATUS_NOT_SUPPORTED || status == AMDSMI_STATUS_NO_PERM) {
      GTEST_SKIP() << bdf_str << ": amd-smi reports no CUID (run as root)";
    }
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS) << bdf_str;
    EXPECT_EQ(info.source, AMDSMI_CUID_SOURCE_DRIVER) << bdf_str;
    EXPECT_EQ(std::string(info.derived), driver_value)
        << bdf_str << ": amd-smi and the driver disagree about the derived CUID";
    ++checked;
  }

  if (checked == 0) GTEST_SKIP() << "no device publishes cuid_derived";
}

TEST(GpuUnit, CuidDriverExpectationFollowsProcessorRenderNode) {
  ScopedTempDir sysfs;
  ASSERT_TRUE(sysfs.valid());
  const std::string bdf = "0000:03:00.0";
  const std::string pci = sysfs.path() + "/bus/pci/devices/" + bdf;
  const std::string xcp = sysfs.path() + "/devices/platform/amdgpu_xcp.1";
  ASSERT_TRUE(MakeDirs(pci + "/xcp"));
  ASSERT_TRUE(MakeDirs(xcp + "/xcp"));
  ASSERT_TRUE(MakeDirs(sysfs.path() + "/class/drm/renderD128"));
  ASSERT_TRUE(MakeDirs(sysfs.path() + "/class/drm/renderD140"));
  ASSERT_EQ(symlink(pci.c_str(), (sysfs.path() + "/class/drm/renderD128/device").c_str()), 0);
  ASSERT_EQ(symlink(xcp.c_str(), (sysfs.path() + "/class/drm/renderD140/device").c_str()), 0);
  const char* whole = "61ffe99a-b3e0-8e16-a802-4b1d515d5438";
  const char* first = "73488f9e-ea52-86ce-8401-2627fa41b068";
  const char* second = "3395667e-f8fa-840e-bc00-028dcc084280";
  std::ofstream(pci + "/cuid_derived") << whole << '\n';
  std::ofstream(pci + "/xcp/cuid_derived") << first << '\n';
  std::ofstream(xcp + "/xcp/cuid_derived") << second << '\n';

  // No profile-count gate: a single SPX partition is still its own component.
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 128, 0), first);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 128, kNoPartition), first);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 140, 1), second);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, kNoPartition, kNoPartition), whole);

  // A present but empty publication must not silently select the parent.
  std::ofstream(pci + "/xcp/cuid_derived", std::ios::trunc).close();
  EXPECT_TRUE(DriverPublished(sysfs.path(), bdf, 128, 0).empty());
  ASSERT_EQ(unlink((pci + "/xcp/cuid_derived").c_str()), 0);
  ASSERT_EQ(unlink((xcp + "/xcp/cuid_derived").c_str()), 0);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 128, 0), whole);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 128, kNoPartition), whole);
  EXPECT_TRUE(DriverPublished(sysfs.path(), bdf, 140, 1).empty());
  EXPECT_TRUE(DriverPublished(sysfs.path(), bdf, kNoPartition, 1).empty());
  ASSERT_EQ(rmdir((pci + "/xcp").c_str()), 0);
  EXPECT_EQ(DriverPublished(sysfs.path(), bdf, 128, 0), whole);
}

TEST(GpuUnit, CuidSourceNamesTheStageThatAnswered) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  size_t answered = 0;
  for (auto handle : handles) {
    amdsmi_bdf_t bdf = {};
    ASSERT_EQ(amdsmi_get_gpu_device_bdf(handle, &bdf), AMDSMI_STATUS_SUCCESS);
    const std::string bdf_str = BdfString(bdf);

    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
    if (status == AMDSMI_STATUS_NOT_SUPPORTED || status == AMDSMI_STATUS_NO_PERM) continue;
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS) << bdf_str;
    ++answered;

    EXPECT_NE(info.source, AMDSMI_CUID_SOURCE_UNKNOWN)
        << bdf_str << ": a snapshot that succeeded should say which stage produced it";

    // A non-UNKNOWN source alone would not catch false DRIVER attribution.
    const bool driver_published = !DriverPublished(handle, bdf_str).empty();
    EXPECT_EQ(info.source == AMDSMI_CUID_SOURCE_DRIVER, driver_published)
        << bdf_str << ": source=" << static_cast<int>(info.source) << " while the driver "
        << (driver_published ? "does" : "does not") << " publish cuid_derived";
  }

  if (answered == 0) GTEST_SKIP() << "no device reported a CUID";
}

// A driver-published partition node is what names a partition, so amd-smi must
// answer from it, verbatim and as DRIVER, rather than from the whole card. No
// GPU here need be partitionable: the node is fabricated under
// AMDSMI_CUID_SYSFS_ROOT, which a library honours only when built with
// AMDSMI_CUID_TEST_SYSFS_OVERRIDE, because a relocatable root would let any
// unprivileged user forge the source field.
TEST(GpuUnit, CuidPartitionNodeIsUsedVerbatimAsDriverSourced) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
#ifndef AMDSMI_CUID_TEST_SYSFS_OVERRIDE
  GTEST_SKIP() << "configure with -DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON to exercise this case";
#endif

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  const amdsmi_processor_handle handle = handles.front();
  amdsmi_enumeration_info_t enumeration = {};
  ASSERT_EQ(amdsmi_get_gpu_enumeration_info(handle, &enumeration), AMDSMI_STATUS_SUCCESS);

  ScopedTempDir sysfs;
  ASSERT_TRUE(sysfs.valid());
  const std::string node =
      sysfs.path() + "/class/drm/renderD" + std::to_string(enumeration.drm_render) + "/device/xcp";
  ASSERT_TRUE(MakeDirs(node));
  // Differs from anything a real device publishes, so an answer taken from
  // the whole card cannot match it.
  const char* partition_cuid = "73488f9e-ea52-86ce-8401-2627fa41b068";
  std::ofstream(node + "/cuid_derived") << partition_cuid << '\n';

  const ScopedSysfsRoot root(sysfs.path());
  amdsmi_cuid_info_t info = {};
  ASSERT_EQ(amdsmi_get_gpu_cuid_info(handle, &info), AMDSMI_STATUS_SUCCESS);
  EXPECT_STREQ(info.derived, partition_cuid);
  EXPECT_EQ(info.source, AMDSMI_CUID_SOURCE_DRIVER);
}

// One value, one lookup path: two paths to one identifier is how the kernel and
// the library came to disagree.
TEST(GpuUnit, CuidSingleStringCallMatchesSnapshot) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_cuid_info(handle, &info) != AMDSMI_STATUS_SUCCESS) continue;

    char cuid[AMDSMI_GPU_CUID_SIZE] = {};
    unsigned int length = sizeof(cuid);
    ASSERT_EQ(amdsmi_get_gpu_device_cuid(handle, &length, cuid), AMDSMI_STATUS_SUCCESS);
    EXPECT_STREQ(cuid, info.derived);
  }
}

// A partition must never be handed the whole card's identifier, since all
// partitions share one BDF. With per-partition driver CUIDs each partition
// reports its own; without them it reports NOT_SUPPORTED. Two partitions of one
// card must never report the same value.
//
// Needs partitioned hardware to exercise, so it skips where there is none.
TEST(GpuUnit, CuidForAPartitionIsItsOwnOrIsRefused) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  std::map<std::string, std::vector<std::string>> derived_by_card;
  size_t checked = 0;

  for (auto handle : handles) {
    const uint32_t partition = PartitionId(handle);
    if (partition == kNoPartition) continue;

    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);

    char cuid[AMDSMI_GPU_CUID_SIZE] = {};
    unsigned int length = sizeof(cuid);
    const amdsmi_status_t string_status = amdsmi_get_gpu_device_cuid(handle, &length, cuid);

    if (status == AMDSMI_STATUS_SUCCESS) {
      EXPECT_EQ(info.source, AMDSMI_CUID_SOURCE_DRIVER)
          << "partition " << partition
          << " reported a CUID that did not come from the driver; nothing else can name a "
             "partition";
      EXPECT_EQ(string_status, AMDSMI_STATUS_SUCCESS) << "partition " << partition;
      EXPECT_STREQ(cuid, info.derived) << "partition " << partition;

      amdsmi_bdf_t bdf = {};
      ASSERT_EQ(amdsmi_get_gpu_device_bdf(handle, &bdf), AMDSMI_STATUS_SUCCESS);
      char bdf_str[32] = {};
      snprintf(bdf_str, sizeof(bdf_str), "%04x:%02x:%02x.%01x",
               static_cast<unsigned>(bdf.domain_number), static_cast<unsigned>(bdf.bus_number),
               static_cast<unsigned>(bdf.device_number),
               static_cast<unsigned>(bdf.function_number));
      derived_by_card[bdf_str].emplace_back(info.derived);
    } else {
      EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED)
          << "partition " << partition << " failed for a reason other than having no identifier";
      ExpectNoCuidReported(info);
      EXPECT_EQ(string_status, AMDSMI_STATUS_NOT_SUPPORTED)
          << "partition " << partition << ": the string form rounded the refusal back to a value";
    }
    ++checked;
  }

  for (const auto& card : derived_by_card) {
    for (size_t i = 0; i < card.second.size(); ++i) {
      for (size_t j = i + 1; j < card.second.size(); ++j) {
        EXPECT_NE(card.second[i], card.second[j])
            << "two partitions of " << card.first << " report the same derived CUID";
      }
    }
  }

  if (checked == 0) {
    GTEST_SKIP() << "no partitioned device present: this case needs a GPU in a multi-partition "
                    "mode, such as an MI300X or MI350X in CPX";
  }
}

// Two processors reporting one derived CUID is the same defect seen from the
// other side, and is visible on any machine with more than one device,
// partitioned or not.
TEST(GpuUnit, CuidDerivedValuesDoNotCollideAcrossProcessors) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.size() < 2) GTEST_SKIP() << "fewer than two GPU processors present";

  std::vector<std::string> derived;
  std::vector<uint32_t> partitions;
  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_cuid_info(handle, &info) != AMDSMI_STATUS_SUCCESS) continue;
    derived.emplace_back(info.derived);
    partitions.push_back(PartitionId(handle));
  }
  if (derived.size() < 2) GTEST_SKIP() << "fewer than two processors report a CUID";

  for (size_t i = 0; i < derived.size(); ++i) {
    for (size_t j = i + 1; j < derived.size(); ++j) {
      EXPECT_NE(derived[i], derived[j])
          << "processors " << i << " (partition " << partitions[i] << ") and " << j
          << " (partition " << partitions[j] << ") report the same derived CUID";
    }
  }
}

// A failed snapshot reports nothing. The field that makes this matter is
// auxiliary: it has no "undetermined" encoding, so a query that could not read
// it and returned success would state that a value is canonical on no evidence.
// An unprivileged caller on a driver publishing cuid_primary as 0400 is exactly
// that case, and must see the permission failure instead.
//
// Runs wherever a call fails for any reason; where every call succeeds there is
// nothing here to check.
TEST(GpuUnit, CuidFailedSnapshotReportsNothing) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  size_t checked = 0;
  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
    if (status == AMDSMI_STATUS_SUCCESS) continue;

    ExpectNoCuidReported(info);
    ++checked;
  }

  if (checked == 0) GTEST_SKIP() << "every device answered; no failure to inspect";
}

// The legacy device UUID is retained and is a different value. Redefining it to
// return a CUID would change the meaning of a published ABI under consumers
// that have already recorded its output.
TEST(GpuUnit, CuidLegacyUuidIsStillItsOwnValue) {
  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  const auto handles = GpuHandles();
  if (handles.empty()) GTEST_SKIP() << "no GPU present";

  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_cuid_info(handle, &info) != AMDSMI_STATUS_SUCCESS) continue;

    char uuid[AMDSMI_GPU_UUID_SIZE] = {};
    unsigned int length = sizeof(uuid);
    if (amdsmi_get_gpu_device_uuid(handle, &length, uuid) != AMDSMI_STATUS_SUCCESS) continue;

    EXPECT_TRUE(LooksLikeUuid(uuid)) << "uuid: " << uuid;
    EXPECT_STRNE(uuid, info.derived);
  }
}

namespace {

std::vector<amdsmi_cuid_component_t> CuidComponents(amdsmi_status_t* status) {
  uint32_t count = 0;
  *status = amdsmi_get_cuid_components(&count, nullptr);
  if (*status != AMDSMI_STATUS_SUCCESS) return {};
  std::vector<amdsmi_cuid_component_t> components(count);
  *status = amdsmi_get_cuid_components(&count, components.data());
  components.resize(count);
  return components;
}

}  // namespace

// The node-wide list is a full inventory: every entry is a well-formed CUID of
// a known type, no two share a derived CUID, entries come sorted by type, and
// every GPU amd-smi manages appears with the same CUID it reports per device.
TEST(GpuUnit, CuidComponentListIsConsistent) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  const auto components = CuidComponents(&status);
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  if (components.empty()) GTEST_SKIP() << "no component has a CUID";

  std::set<std::string> derived;
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& c = components[i];
    EXPECT_TRUE(LooksLikeUuid(c.info.derived)) << "entry " << i << ": " << c.info.derived;
    EXPECT_NE(c.info.component_type, AMDSMI_CUID_COMPONENT_UNKNOWN) << "entry " << i;
    EXPECT_NE(c.info.source, AMDSMI_CUID_SOURCE_UNKNOWN) << "entry " << i;
    EXPECT_TRUE(derived.insert(c.info.derived).second) << "duplicate " << c.info.derived;
    bool aux_from_value = false;
    ASSERT_TRUE(AuxiliaryBitFromUuidString(c.info.derived, &aux_from_value));
    EXPECT_EQ(static_cast<bool>(c.info.auxiliary), aux_from_value) << c.info.derived;
    if (geteuid() != 0) EXPECT_EQ(c.info.primary[0], '\0') << "entry " << i;
    if (i > 0) EXPECT_LE(components[i - 1].info.component_type, c.info.component_type);
  }

  for (auto handle : GpuHandles()) {
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_cuid_info(handle, &info) != AMDSMI_STATUS_SUCCESS) continue;
    EXPECT_EQ(derived.count(info.derived), 1u) << info.derived << " is missing from the list";
  }
}

// A buffer one entry short is filled as far as it goes, and the call reports
// the full count with INSUFFICIENT_SIZE rather than a silently truncated list.
TEST(GpuUnit, CuidComponentListReportsAShortBuffer) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  uint32_t total = 0;
  ASSERT_EQ(amdsmi_get_cuid_components(&total, nullptr), AMDSMI_STATUS_SUCCESS);
  if (total < 2) GTEST_SKIP() << "fewer than two components";

  std::vector<amdsmi_cuid_component_t> components(total - 1);
  uint32_t count = total - 1;
  EXPECT_EQ(amdsmi_get_cuid_components(&count, components.data()), AMDSMI_STATUS_INSUFFICIENT_SIZE);
  EXPECT_EQ(count, total);
  EXPECT_TRUE(LooksLikeUuid(components.back().info.derived));
}
