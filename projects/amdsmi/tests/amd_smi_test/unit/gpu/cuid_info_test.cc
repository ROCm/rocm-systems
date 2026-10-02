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
#include <sched.h>
#include <sys/mount.h>
#include <sys/prctl.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "amd_smi_cuid_seed.h"

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
  explicit AmdSmiSession(uint64_t flags = AMDSMI_INIT_AMD_GPUS) : status_(amdsmi_init(flags)) {}
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

bool ReadExactly(const std::string& path, std::string& out, size_t size) {
  std::ifstream file(path, std::ios::binary);
  if (!file) return false;
  out.assign(size + 1, '\0');
  file.read(&out[0], static_cast<std::streamsize>(out.size()));
  if (static_cast<size_t>(file.gcount()) != size) return false;
  out.resize(size);
  return true;
}

// Whether amdgpu holds a node key this process can read: any cuid_seed that
// returns 32 octets. amdgpu fails the read with ENODATA while it holds none.
bool NodeKey() {
  if (geteuid() != 0) return false;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices", ec)) {
    std::string seed;
    if (ReadExactly(entry.path().string() + "/cuid_seed", seed, AMDSMI_CUID_SEED_SIZE)) return true;
  }
  return false;
}

// A non-root caller is refused rather than told there is no key.
amdsmi_status_t ExpectedSeedInfoStatus() {
  return geteuid() == 0 ? AMDSMI_STATUS_SUCCESS : AMDSMI_STATUS_NO_PERM;
}

// Without a key, and on any failure, the struct must come back zeroed: a
// fingerprint beside provisioned = 0 would name a key that is not there.
void ExpectNothingReported(const amdsmi_cuid_seed_info_t& info) {
  EXPECT_EQ(info.provisioned, 0) << "no key must not report a provisioning state";
  for (size_t i = 0; i < sizeof(info.fingerprint); ++i) {
    EXPECT_EQ(info.fingerprint[i], 0) << "no key must not report a fingerprint (octet " << i << ")";
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

// The output is cleared on every failure, including a call made before
// amdsmi_init(), so a caller's earlier contents are never mistaken for an answer.
TEST(GpuUnit, CuidSeedInfoIsClearedBeforeInit) {
  amdsmi_cuid_seed_info_t info;
  std::memset(&info, 0xa5, sizeof(info));
  EXPECT_EQ(amdsmi_get_cuid_seed_info(&info), AMDSMI_STATUS_NOT_INIT);
  ExpectNothingReported(info);
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
  if (status != AMDSMI_STATUS_SUCCESS || !NodeKey()) {
    ExpectNothingReported(info);
    return;
  }

  EXPECT_NE(info.provisioned, 0);
  bool any_set = false;
  for (uint8_t octet : info.fingerprint) {
    if (octet != 0) any_set = true;
  }
  EXPECT_TRUE(any_set) << "seed fingerprint should never be all zeroes";
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

// AMDSMI_STATUS_IO promises the key is now the seed. That takes a key known
// before the call, different after it and equal to the seed, so a build that
// cannot compute the seed's fingerprint never claims it.
TEST(GpuUnit, CuidSetSeedCommitNeedsEvidenceOfTheSeed) {
  using amd::smi::detail::cuid_seed_committed;
  constexpr size_t kSize = AMDSMI_CUID_SEED_FINGERPRINT_SIZE;
  const uint8_t old_key[kSize] = {1, 2, 3, 4, 5, 6, 7, 8};
  const uint8_t seed[kSize] = {9, 9, 9, 9, 9, 9, 9, 9};

  EXPECT_FALSE(cuid_seed_committed(false, false, old_key, old_key, nullptr, kSize));
  EXPECT_FALSE(cuid_seed_committed(false, false, old_key, seed, nullptr, kSize));
  EXPECT_FALSE(cuid_seed_committed(true, true, old_key, old_key, nullptr, kSize));
  EXPECT_FALSE(cuid_seed_committed(true, true, old_key, seed, nullptr, kSize));
  EXPECT_FALSE(cuid_seed_committed(true, false, old_key, seed, nullptr, kSize));

  EXPECT_FALSE(cuid_seed_committed(false, false, old_key, old_key, seed, kSize));
  EXPECT_FALSE(cuid_seed_committed(false, false, old_key, seed, seed, kSize));
  EXPECT_FALSE(cuid_seed_committed(true, true, seed, seed, seed, kSize));
  EXPECT_FALSE(cuid_seed_committed(true, true, old_key, old_key, seed, kSize));
  EXPECT_TRUE(cuid_seed_committed(true, true, old_key, seed, seed, kSize));
  EXPECT_TRUE(cuid_seed_committed(true, false, old_key, seed, seed, kSize));
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

  size_t checked = 0;
  for (auto handle : handles) {
    amdsmi_cuid_info_t info = {};
    const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
    // A handle without a CUID, such as a DPX partition without a node key, or
    // one this caller may not read in full, leaves nothing to check.
    if (status == AMDSMI_STATUS_NOT_SUPPORTED || status == AMDSMI_STATUS_NO_PERM) continue;
    ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
    ++checked;

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
      if (HasCapSysAdmin()) {
        EXPECT_NE(info.primary[0], '\0') << "root caller should receive the primary CUID";
      }
      if (info.primary[0] != '\0') {
        EXPECT_TRUE(LooksLikeUuid(info.primary)) << "primary: " << info.primary;
        EXPECT_EQ(info.primary[14], '8');
      }
    } else {
      EXPECT_EQ(info.primary[0], '\0')
          << "unprivileged caller should receive an empty primary, got: " << info.primary;
    }
  }

  if (checked == 0) {
    GTEST_SKIP() << (kCuidBuiltIn ? "no GPU handle reports a CUID this caller can read"
                                  : "built without CUID support; asserted by "
                                    "CuidEntryPointsNotSupportedWithoutTheLibrary");
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

// The whole GPU's entry in the component list, for a handle whose BDF names
// one: a whole GPU or partition 0. Not the handle's own CUID, which on a
// partitionable GPU is a partition's.
bool WholeGpuCuid(amdsmi_processor_handle handle, amdsmi_cuid_info_t* info) {
  amdsmi_bdf_t bdf = {};
  uint32_t count = 0;
  if (amdsmi_get_gpu_device_bdf(handle, &bdf) != AMDSMI_STATUS_SUCCESS ||
      amdsmi_get_cuid_components(&count, nullptr) != AMDSMI_STATUS_SUCCESS)
    return false;
  std::vector<amdsmi_cuid_component_t> components(count);
  if (amdsmi_get_cuid_components(&count, components.data()) != AMDSMI_STATUS_SUCCESS) return false;
  const std::string wanted = BdfString(bdf);
  for (const auto& c : components) {
    const std::string path = c.device_path;
    if (c.info.component_type == AMDSMI_CUID_COMPONENT_GPU && wanted == c.bdf &&
        (path.size() < 4 || path.compare(path.size() - 4, 4, "/xcp") != 0)) {
      *info = c.info;
      return true;
    }
  }
  return false;
}

// An MI300-class GPU publishes a partition node, <render device>/xcp, in every
// compute mode, SPX included. This fabricates one for `handle` under
// AMDSMI_CUID_SYSFS_ROOT, which a library honours only when built with
// AMDSMI_CUID_TEST_SYSFS_OVERRIDE, because a relocatable root would let any
// unprivileged user forge the source field. With `derived` it also publishes
// the partition's cuid_derived, as amdgpu does once it holds a node key.
// `mode` is the compute mode and `unit_id` the partition's UnitID; without one
// there is no partition node, as with a driver that publishes no CUID at all.
class FabricatedPartition {
 public:
  FabricatedPartition(amdsmi_processor_handle handle, const char* mode, const char* unit_id,
                      bool derived) {
    amdsmi_enumeration_info_t enumeration = {};
    if (!sysfs_.valid() ||
        amdsmi_get_gpu_enumeration_info(handle, &enumeration) != AMDSMI_STATUS_SUCCESS)
      return;
    const std::string device =
        sysfs_.path() + "/class/drm/renderD" + std::to_string(enumeration.drm_render) + "/device";
    if (!MakeDirs(unit_id ? device + "/xcp" : device)) return;
    std::ofstream(device + "/current_compute_partition") << mode << '\n';
    if (unit_id) std::ofstream(device + "/xcp/cuid_unit_id") << unit_id << '\n';
    if (derived) std::ofstream(device + "/xcp/cuid_derived") << kPartitionCuid << '\n';
    root_ = std::make_unique<ScopedSysfsRoot>(sysfs_.path());
  }
  bool valid() const { return root_ != nullptr; }

  // Differs from anything a real device publishes, so an answer taken from
  // the whole GPU cannot match it.
  static constexpr const char* kPartitionCuid = "73488f9e-ea52-86ce-8401-2627fa41b068";

 private:
  ScopedTempDir sysfs_;
  std::unique_ptr<ScopedSysfsRoot> root_;
};

// A partition's own driver CUID is what its handle reports, verbatim and as
// DRIVER, in every mode. Without one, the one partition of a GPU in SPX is the
// whole GPU and reports the GPU's CUID, source and primary; in DPX and above it
// has none.
TEST(GpuUnit, CuidPartitionNodeInSpxAndDpx) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
#ifndef AMDSMI_CUID_TEST_SYSFS_OVERRIDE
  GTEST_SKIP() << "configure with -DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON to exercise this case";
#endif

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_processor_handle handle = nullptr;
  amdsmi_cuid_info_t whole = {};
  for (auto candidate : GpuHandles()) {
    if (WholeGpuCuid(candidate, &whole)) {
      handle = candidate;
      break;
    }
  }
  if (handle == nullptr) GTEST_SKIP() << "no GPU handle names a whole GPU with a CUID";

  for (const bool derived : {false, true}) {
    SCOPED_TRACE(derived ? "partition cuid_derived published" : "no partition cuid_derived");
    for (const auto& mode : {std::make_pair("SPX", "512"), std::make_pair("DPX", "256")}) {
      SCOPED_TRACE(mode.first);
      const FabricatedPartition partition(handle, mode.first, mode.second, derived);
      ASSERT_TRUE(partition.valid());
      amdsmi_cuid_info_t info = {};
      const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
      if (derived) {
        ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
        EXPECT_STREQ(info.derived, FabricatedPartition::kPartitionCuid);
        EXPECT_EQ(info.source, AMDSMI_CUID_SOURCE_DRIVER);
      } else if (std::string(mode.first) == "SPX") {
        ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
        EXPECT_STREQ(info.derived, whole.derived);
        EXPECT_STREQ(info.primary, whole.primary);
        EXPECT_EQ(info.source, whole.source);
        EXPECT_EQ(info.auxiliary, whole.auxiliary);
      } else {
        EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
        ExpectNoCuidReported(info);
      }
    }
  }
}

// A driver that publishes no CUID files leaves no partition node, but the
// compute mode still says whether a handle is the whole GPU: in SPX it reports
// the GPU's CUID, in DPX it reports none, partition 0's handle included.
// Handles of other partitions are also refused by their KFD partition index.
TEST(GpuUnit, CuidWithoutDriverFilesFollowsTheComputeMode) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
#ifndef AMDSMI_CUID_TEST_SYSFS_OVERRIDE
  GTEST_SKIP() << "configure with -DAMDSMI_CUID_TEST_SYSFS_OVERRIDE=ON to exercise this case";
#endif

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_processor_handle handle = nullptr;
  amdsmi_cuid_info_t whole = {};
  for (auto candidate : GpuHandles()) {
    if (WholeGpuCuid(candidate, &whole)) {
      handle = candidate;
      break;
    }
  }
  if (handle == nullptr) GTEST_SKIP() << "no GPU handle names a whole GPU with a CUID";

  {
    const FabricatedPartition spx(handle, "SPX", nullptr, false);
    ASSERT_TRUE(spx.valid());
    amdsmi_cuid_info_t info = {};
    ASSERT_EQ(amdsmi_get_gpu_cuid_info(handle, &info), AMDSMI_STATUS_SUCCESS);
    EXPECT_STREQ(info.derived, whole.derived);
    EXPECT_STREQ(info.primary, whole.primary);
    EXPECT_EQ(info.source, whole.source);
  }
  {
    const FabricatedPartition dpx(handle, "DPX", nullptr, false);
    ASSERT_TRUE(dpx.valid());
    amdsmi_cuid_info_t info = {};
    EXPECT_EQ(amdsmi_get_gpu_cuid_info(handle, &info), AMDSMI_STATUS_NOT_SUPPORTED);
    ExpectNoCuidReported(info);
  }
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
      derived_by_card[BdfString(bdf)].emplace_back(info.derived);
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
// a known type, entries come sorted by type, each names its own function, only
// NIC functions of one card share a derived CUID, and every GPU amd-smi
// manages appears with the same CUID it reports per device.
TEST(GpuUnit, CuidComponentListIsConsistent) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

  amdsmi_status_t status = AMDSMI_STATUS_SUCCESS;
  const auto components = CuidComponents(&status);
  ASSERT_EQ(status, AMDSMI_STATUS_SUCCESS);
  if (components.empty()) GTEST_SKIP() << "no component has a CUID";

  std::map<std::string, std::string> card_by_derived;
  std::set<std::string> functions;
  for (size_t i = 0; i < components.size(); ++i) {
    const auto& c = components[i];
    EXPECT_TRUE(LooksLikeUuid(c.info.derived)) << "entry " << i << ": " << c.info.derived;
    EXPECT_NE(c.info.component_type, AMDSMI_CUID_COMPONENT_UNKNOWN) << "entry " << i;
    EXPECT_NE(c.info.source, AMDSMI_CUID_SOURCE_UNKNOWN) << "entry " << i;
    EXPECT_TRUE(functions.insert(std::string(c.bdf) + " " + c.device_path).second)
        << "listed twice: " << c.bdf << " " << c.device_path;
    // A NIC card is the BDF without its function number.
    const std::string card = c.info.component_type == AMDSMI_CUID_COMPONENT_NIC
                                 ? std::string(c.bdf).substr(0, 10)
                                 : std::string(c.bdf) + " " + c.device_path;
    const auto shared = card_by_derived.emplace(c.info.derived, card);
    EXPECT_TRUE(shared.second || shared.first->second == card)
        << "two components share " << c.info.derived;
    bool aux_from_value = false;
    ASSERT_TRUE(AuxiliaryBitFromUuidString(c.info.derived, &aux_from_value));
    EXPECT_EQ(static_cast<bool>(c.info.auxiliary), aux_from_value) << c.info.derived;
    if (geteuid() != 0) {
      EXPECT_EQ(c.info.primary[0], '\0') << "entry " << i;
    }
    if (i > 0) {
      EXPECT_LE(components[i - 1].info.component_type, c.info.component_type);
    }
  }

  for (auto handle : GpuHandles()) {
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_cuid_info(handle, &info) != AMDSMI_STATUS_SUCCESS) continue;
    EXPECT_EQ(card_by_derived.count(info.derived), 1u)
        << info.derived << " is missing from the list";
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

namespace {

constexpr const char* kRerunVar = "AMDSMI_CUID_TEST_RERUN";
constexpr int kNoNamespace = 125;

// Whether this process is the rerun RerunHiding() started for `test`.
bool InRerun(const char* test) {
  const char* rerun = std::getenv(kRerunVar);
  return rerun != nullptr && std::strcmp(rerun, test) == 0;
}

// Reruns GpuUnit.`test` in a fresh process, which no earlier discovery has
// primed, in a private mount namespace (and, for an ordinary user, a user
// namespace) where `dirs` are empty and `files` read as empty, and with
// `extra_env` set when it is not empty. Returns the rerun's exit status, or
// kNoNamespace where no namespace can be created.
int RerunHiding(const char* test, std::initializer_list<const char*> dirs,
                std::initializer_list<const char*> files, const std::string& extra_env = "",
                bool drop_sys_admin = false) {
  ScopedTempDir scratch;
  if (!scratch.valid()) return kNoNamespace;
  const std::string empty_file = scratch.path() + "/empty";
  std::ofstream(empty_file).close();

  std::vector<std::string> env;
  for (char** e = environ; *e != nullptr; ++e) env.emplace_back(*e);
  env.emplace_back(std::string(kRerunVar) + "=" + test);
  if (!extra_env.empty()) env.push_back(extra_env);
  std::vector<char*> envp;
  for (auto& e : env) envp.push_back(&e[0]);
  envp.push_back(nullptr);
  std::string filter = std::string("--gtest_filter=GpuUnit.") + test;
  char* const argv[] = {const_cast<char*>("/proc/self/exe"), &filter[0], nullptr};

  const pid_t pid = fork();
  if (pid < 0) return kNoNamespace;
  if (pid == 0) {
    // Root needs no user namespace, and in one could no longer read files it
    // does not own.
    const uid_t uid = geteuid();
    if (unshare(uid == 0 ? CLONE_NEWNS : CLONE_NEWUSER | CLONE_NEWNS) != 0) _exit(kNoNamespace);
    if (uid != 0) {
      char map[64];
      const int length = snprintf(map, sizeof(map), "%u %u 1", uid, uid);
      const int fd = open("/proc/self/uid_map", O_WRONLY);
      if (fd < 0 || write(fd, map, static_cast<size_t>(length)) != length) _exit(kNoNamespace);
      close(fd);
    }
    if (mount(nullptr, "/", nullptr, MS_REC | MS_PRIVATE, nullptr) != 0) _exit(kNoNamespace);
    for (const char* dir : dirs) {
      if (access(dir, F_OK) == 0 && mount("none", dir, "tmpfs", 0, nullptr) != 0)
        _exit(kNoNamespace);
    }
    for (const char* file : files) {
      if (access(file, F_OK) == 0 &&
          mount(empty_file.c_str(), file, nullptr, MS_BIND, nullptr) != 0)
        _exit(kNoNamespace);
    }
    if (drop_sys_admin && prctl(PR_CAPBSET_DROP, CAP_SYS_ADMIN, 0, 0, 0) != 0) _exit(kNoNamespace);
    execve(argv[0], argv, envp.data());
    _exit(127);
  }
  int status = 0;
  if (waitpid(pid, &status, 0) != pid || !WIFEXITED(status)) return -1;
  return WEXITSTATUS(status);
}

// Whether libamdcuid finds a machine identity here, read as it reads one:
// /var/lib/dbus/machine-id only where /etc/machine-id does not exist, and 32
// hex digits that are not all zero. Without one there is no temporary CUID.
bool HostHasMachineId() {
  const char* path =
      access("/etc/machine-id", F_OK) == 0 ? "/etc/machine-id" : "/var/lib/dbus/machine-id";
  std::ifstream file(path);
  std::string line;
  if (!std::getline(file, line) || line.size() != 32) return false;
  bool nonzero = false;
  for (char c : line) {
    if (!std::isxdigit(static_cast<unsigned char>(c))) return false;
    if (c != '0') nonzero = true;
  }
  return nonzero;
}

}  // namespace

// Without a machine identity there is no temporary CUID, so a GPU that would
// get one has none: NOT_SUPPORTED with nothing reported, from both entry
// points, as amdsmi_get_cuid_components() leaves it out. A GPU that still
// answers has a CUID from its driver or the node key, never a temporary one.
TEST(GpuUnit, CuidWithoutAMachineIdThereIsNoTemporaryCuid) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  if (InRerun("CuidWithoutAMachineIdThereIsNoTemporaryCuid")) {
    ASSERT_FALSE(HostHasMachineId());
    const AmdSmiSession session;
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
    for (auto handle : GpuHandles()) {
      amdsmi_cuid_info_t info = {};
      const amdsmi_status_t status = amdsmi_get_gpu_cuid_info(handle, &info);
      char cuid[AMDSMI_GPU_CUID_SIZE] = {};
      unsigned int length = sizeof(cuid);
      const amdsmi_status_t string_status = amdsmi_get_gpu_device_cuid(handle, &length, cuid);
      if (status == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(info.auxiliary, 0) << info.derived;
        continue;
      }
      if (status == AMDSMI_STATUS_NO_PERM && geteuid() != 0) continue;
      EXPECT_EQ(status, AMDSMI_STATUS_NOT_SUPPORTED);
      ExpectNoCuidReported(info);
      EXPECT_EQ(string_status, status);
    }
    uint32_t count = 0;
    const amdsmi_status_t list_status = amdsmi_get_cuid_components(&count, nullptr);
    if (list_status == AMDSMI_STATUS_NO_PERM && geteuid() != 0) return;
    ASSERT_EQ(list_status, AMDSMI_STATUS_SUCCESS);
    std::vector<amdsmi_cuid_component_t> components(count);
    ASSERT_EQ(amdsmi_get_cuid_components(&count, components.data()), AMDSMI_STATUS_SUCCESS);
    for (const auto& c : components) {
      EXPECT_EQ(c.info.auxiliary, 0) << c.bdf << " " << c.device_path << ": " << c.info.derived;
    }
    return;
  }

  const int status = RerunHiding("CuidWithoutAMachineIdThereIsNoTemporaryCuid", {},
                                 {"/etc/machine-id", "/var/lib/dbus/machine-id"});
  if (status == kNoNamespace) GTEST_SKIP() << "cannot create a mount namespace here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}

// A root caller whose primary read fails behind a canonical CUID has hit a
// fault, not a privilege boundary, and must see it rather than an empty
// primary. One GPU's cuid_primary is made to read back empty, which the
// library rejects as malformed.
TEST(GpuUnit, CuidUnreadablePrimaryFailsTheSnapshot) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
  if (geteuid() != 0 || !HasCapSysAdmin()) GTEST_SKIP() << "needs root with CAP_SYS_ADMIN";

  if (InRerun("CuidUnreadablePrimaryFailsTheSnapshot")) {
    const AmdSmiSession session;
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
    size_t checked = 0;
    for (auto handle : GpuHandles()) {
      amdsmi_bdf_t bdf = {};
      ASSERT_EQ(amdsmi_get_gpu_device_bdf(handle, &bdf), AMDSMI_STATUS_SUCCESS);
      struct stat primary = {};
      if (stat(("/sys/bus/pci/devices/" + BdfString(bdf) + "/cuid_primary").c_str(), &primary) !=
              0 ||
          primary.st_size != 0) {
        continue;
      }
      amdsmi_cuid_info_t info = {};
      EXPECT_EQ(amdsmi_get_gpu_cuid_info(handle, &info), AMDSMI_STATUS_UNEXPECTED_DATA);
      ExpectNoCuidReported(info);
      ++checked;
    }
    EXPECT_EQ(checked, 1u);
    uint32_t count = 0;
    EXPECT_EQ(amdsmi_get_cuid_components(&count, nullptr), AMDSMI_STATUS_UNEXPECTED_DATA);
    return;
  }

  const AmdSmiSession session;
  ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
  std::string primary;
  for (auto handle : GpuHandles()) {
    amdsmi_bdf_t bdf = {};
    amdsmi_cuid_info_t info = {};
    if (amdsmi_get_gpu_device_bdf(handle, &bdf) == AMDSMI_STATUS_SUCCESS &&
        amdsmi_get_gpu_cuid_info(handle, &info) == AMDSMI_STATUS_SUCCESS &&
        info.source == AMDSMI_CUID_SOURCE_DRIVER && info.auxiliary == 0 &&
        info.primary[0] != '\0') {
      primary = "/sys/bus/pci/devices/" + BdfString(bdf) + "/cuid_primary";
      break;
    }
  }
  if (primary.empty()) GTEST_SKIP() << "no GPU whose driver publishes a canonical CUID and primary";

  const int status = RerunHiding("CuidUnreadablePrimaryFailsTheSnapshot", {}, {primary.c_str()});
  if (status == kNoNamespace) GTEST_SKIP() << "cannot create a mount namespace here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}

// A temporary CUID can stand where there is no primary at all: root refused
// the node key for want of CAP_SYS_ADMIN, with SMBIOS hidden, so a CPU without
// a readable PPIN has no serial. The CPU keeps its machine-id CUID with an
// empty primary rather than failing the inventory.
TEST(GpuUnit, CuidTemporaryComponentWithoutAPrimaryIsListed) {
  if (!HostHasMachineId())
    GTEST_SKIP() << "no machine identity, so no temporary CUID; asserted by "
                    "CuidWithoutAMachineIdThereIsNoTemporaryCuid";
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
  if (geteuid() != 0) GTEST_SKIP() << "needs root to drop CAP_SYS_ADMIN from";

  if (InRerun("CuidTemporaryComponentWithoutAPrimaryIsListed")) {
    ASSERT_FALSE(HasCapSysAdmin());
    const AmdSmiSession session;
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
    uint32_t count = 0;
    ASSERT_EQ(amdsmi_get_cuid_components(&count, nullptr), AMDSMI_STATUS_SUCCESS);
    std::vector<amdsmi_cuid_component_t> components(count);
    ASSERT_EQ(amdsmi_get_cuid_components(&count, components.data()), AMDSMI_STATUS_SUCCESS);
    size_t cpus = 0;
    for (const auto& c : components) {
      if (c.info.component_type != AMDSMI_CUID_COMPONENT_CPU) continue;
      EXPECT_TRUE(LooksLikeUuid(c.info.derived)) << "derived: " << c.info.derived;
      EXPECT_EQ(c.info.auxiliary, 1) << c.device_path;
      ++cpus;
    }
    EXPECT_GT(cpus, 0u);
    return;
  }

  if (!HasCapSysAdmin()) GTEST_SKIP() << "root already lacks CAP_SYS_ADMIN";
  const int status = RerunHiding("CuidTemporaryComponentWithoutAPrimaryIsListed",
                                 {"/sys/firmware", "/sys/devices/virtual/dmi"}, {}, "", true);
  if (status == kNoNamespace) GTEST_SKIP() << "cannot drop CAP_SYS_ADMIN here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}

// A node where no component has a CUID, such as a container with no GPU and no
// machine-id, has an empty inventory: success with a count of zero.
TEST(GpuUnit, CuidComponentListOfAnEmptyNodeIsEmpty) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";

  if (InRerun("CuidComponentListOfAnEmptyNodeIsEmpty")) {
    // amdsmi_init() for GPUs fails with /sys/class/drm hidden.
    const AmdSmiSession session(AMDSMI_INIT_AMD_NICS);
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
    uint32_t count = 7;
    EXPECT_EQ(amdsmi_get_cuid_components(&count, nullptr), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(count, 0u);
    return;
  }

  const int status = RerunHiding("CuidComponentListOfAnEmptyNodeIsEmpty",
                                 {"/sys/class/drm", "/sys/class/accel", "/sys/class/net",
                                  "/sys/bus/pci/devices", "/sys/firmware"},
                                 {"/etc/machine-id", "/var/lib/dbus/machine-id"});
  if (status == kNoNamespace) GTEST_SKIP() << "cannot create a mount namespace here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}

// Re-submitting the node key already in place changes nothing, so a refresh
// failing after it is a plain failure rather than a key that changed without
// being published. Every build tells the two apart the same way, by whether the
// fingerprint changed. The key comes back unchanged, so this is safe on a host
// that has one; the refresh is made to fail by hiding /proc/cpuinfo.
TEST(GpuUnit, CuidSetSeedRefreshFailureWithTheKeyAlreadyInPlace) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
  if (geteuid() != 0) GTEST_SKIP() << "setting the node key needs root";

  if (InRerun("CuidSetSeedRefreshFailureWithTheKeyAlreadyInPlace")) {
    const AmdSmiSession session;
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);
    std::string key;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices", ec)) {
      if (ReadExactly(entry.path().string() + "/cuid_seed", key, AMDSMI_CUID_SEED_SIZE)) break;
      key.clear();
    }
    ASSERT_FALSE(key.empty());
    amdsmi_cuid_seed_info_t before = {};
    ASSERT_EQ(amdsmi_get_cuid_seed_info(&before), AMDSMI_STATUS_SUCCESS);

    EXPECT_EQ(amdsmi_set_cuid_seed(reinterpret_cast<const uint8_t*>(key.data())),
              AMDSMI_STATUS_API_FAILED);

    amdsmi_cuid_seed_info_t after = {};
    ASSERT_EQ(amdsmi_get_cuid_seed_info(&after), AMDSMI_STATUS_SUCCESS);
    EXPECT_EQ(after.provisioned, before.provisioned);
    EXPECT_EQ(memcmp(after.fingerprint, before.fingerprint, sizeof(after.fingerprint)), 0);
    return;
  }

  const char* allow = std::getenv("AMDCUID_TEST_ALLOW_SET_KEY");
  if (!allow || std::strcmp(allow, "1") != 0)
    GTEST_SKIP() << "writes the host's node key back to cuid_seed; "
                    "set AMDCUID_TEST_ALLOW_SET_KEY=1 to run it.";
  if (!NodeKey()) GTEST_SKIP() << "amdgpu holds no node key to re-submit";
  const int status =
      RerunHiding("CuidSetSeedRefreshFailureWithTheKeyAlreadyInPlace", {}, {"/proc/cpuinfo"});
  if (status == kNoNamespace) GTEST_SKIP() << "cannot create a mount namespace here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}

// amdgpu serves cuid_seed only to CAP_SYS_ADMIN, so root without it is refused
// by the driver rather than by the library's euid check, and must still see
// NO_PERM. The key re-submitted is the one in place, so nothing changes even if
// a driver let the write through.
TEST(GpuUnit, CuidSeedNeedsCapSysAdmin) {
  if (!kCuidBuiltIn) GTEST_SKIP() << "built without CUID support";
  if (geteuid() != 0) GTEST_SKIP() << "needs root to drop CAP_SYS_ADMIN from";
  constexpr const char* kKeyFileVar = "AMDSMI_CUID_TEST_KEY_FILE";

  if (InRerun("CuidSeedNeedsCapSysAdmin")) {
    ASSERT_FALSE(HasCapSysAdmin());
    const AmdSmiSession session;
    ASSERT_EQ(session.status(), AMDSMI_STATUS_SUCCESS);

    amdsmi_cuid_seed_info_t info = {};
    EXPECT_EQ(amdsmi_get_cuid_seed_info(&info), AMDSMI_STATUS_NO_PERM);
    ExpectNothingReported(info);

    std::string key;
    const char* key_file = std::getenv(kKeyFileVar);
    ASSERT_TRUE(key_file && ReadExactly(key_file, key, AMDSMI_CUID_SEED_SIZE));
    EXPECT_EQ(amdsmi_set_cuid_seed(reinterpret_cast<const uint8_t*>(key.data())),
              AMDSMI_STATUS_NO_PERM);
    return;
  }

  if (!HasCapSysAdmin()) GTEST_SKIP() << "root already lacks CAP_SYS_ADMIN";
  std::string key;
  std::error_code ec;
  for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices", ec)) {
    if (ReadExactly(entry.path().string() + "/cuid_seed", key, AMDSMI_CUID_SEED_SIZE)) break;
    key.clear();
  }
  if (key.empty()) GTEST_SKIP() << "amdgpu holds no node key to re-submit";
  ScopedTempDir scratch;
  ASSERT_TRUE(scratch.valid());
  const std::string key_file = scratch.path() + "/key";
  std::ofstream(key_file, std::ios::binary) << key;

  const int status = RerunHiding("CuidSeedNeedsCapSysAdmin", {}, {},
                                 std::string(kKeyFileVar) + "=" + key_file, true);
  if (status == kNoNamespace) GTEST_SKIP() << "cannot drop CAP_SYS_ADMIN here";
  EXPECT_EQ(status, 0) << "see the rerun above";
}
