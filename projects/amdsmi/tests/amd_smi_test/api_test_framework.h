// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#ifndef TESTS_AMD_SMI_TEST_API_TEST_FRAMEWORK_H_
#define TESTS_AMD_SMI_TEST_API_TEST_FRAMEWORK_H_

#include <gtest/gtest.h>

#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_utils.h"
#include "test_base.h"
#include "test_common.h"

namespace amdsmi {
namespace test {

// Tests always emit the DISPLAY_AMDSMI_API/STATUS trace.
inline constexpr bool kVerbose = true;

// Sentinel used for invalid-handle negative tests.
inline constexpr amdsmi_processor_handle kInvalidHandle = nullptr;

// Statuses that mean the feature is genuinely absent on this platform -- the
// driver, the interface, or the privilege it needs is missing. A positive test
// counts these as a SKIPPED device: never a pass, never a failure. Treating
// them as passes is what let a suite stay green against a library that answers
// AMDSMI_STATUS_NOT_SUPPORTED for everything.
inline bool IsFeatureAbsent(amdsmi_status_t status) {
  return status == AMDSMI_STATUS_NOT_SUPPORTED || status == AMDSMI_STATUS_NOT_YET_IMPLEMENTED ||
         status == AMDSMI_STATUS_NO_PERM ||
         // Required kernel driver / interface not present on this system.
         status == AMDSMI_STATUS_NO_HSMP_MSG_SUP || status == AMDSMI_STATUS_NO_HSMP_SUP ||
         status == AMDSMI_STATUS_NO_HSMP_DRV || status == AMDSMI_STATUS_NO_ENERGY_DRV ||
         status == AMDSMI_STATUS_NO_MSR_DRV || status == AMDSMI_STATUS_NO_DRV ||
         status == AMDSMI_STATUS_DRIVER_NOT_LOADED || status == AMDSMI_STATUS_NON_AMD_CPU;
}

// A switch is on only for an explicitly true value, so FOO=0 reads as off.
inline bool EnvFlagEnabled(const char* name) {
  const char* v = std::getenv(name);
  if (v == nullptr) return false;
  return std::strcmp(v, "1") == 0 || std::strcmp(v, "true") == 0 || std::strcmp(v, "TRUE") == 0 ||
         std::strcmp(v, "yes") == 0;
}

// Tests whose API is tracked in known_failures.md are skipped by default.
// Set AMDSMI_RUN_KNOWN_FAILURES=1 to run them and see the current behavior.
inline bool RunKnownFailures() {
  static const bool run = EnvFlagEnabled("AMDSMI_RUN_KNOWN_FAILURES");
  return run;
}

// Human-readable "N (NAME)" for a status code.
inline std::string AmdsmiStatusLabel(amdsmi_status_t err) {
  const char* name = nullptr;
  amdsmi_status_code_to_string(err, &name);
  std::string out = std::to_string(static_cast<int>(err));
  if (name != nullptr) out += std::string(" (") + name + ")";
  return out;
}

// err is one of the caller-listed codes. Used by negative cases, where the set
// of acceptable codes is exactly what the amdsmi.h contract names.
template <typename... Args>
inline bool AmdsmiStatusMatches(amdsmi_status_t err, Args... expected) {
  const amdsmi_status_t codes[] = {expected...};
  for (amdsmi_status_t c : codes) {
    if (err == c) return true;
  }
  return false;
}

// The caller-listed codes plus any status meaning the call could not be
// exercised here. Only for negative cases: an API that bails before reaching
// its argument check makes that check unobservable, not wrong.
template <typename... Args>
inline bool AmdsmiStatusIsExpected(amdsmi_status_t err, Args... expected) {
  return AmdsmiStatusMatches(err, expected...) || IsFeatureAbsent(err);
}

// Getters are handed a struct pre-filled with this byte so a SUCCESS that left
// the output untouched is detectable. A plain all-zero check cannot do that: an
// idle GPU's activity counters and a healthy board's ECC counts are legitimately
// zero.
inline constexpr unsigned char kOutputPoison = 0xA5;

inline void PoisonOutput(void* out, size_t size) { memset(out, kOutputPoison, size); }

inline bool OutputWasWritten(const void* out, size_t size) {
  const auto* bytes = static_cast<const unsigned char*>(out);
  for (size_t i = 0; i < size; ++i) {
    if (bytes[i] != kOutputPoison) return true;
  }
  return false;
}

// True when buf holds a non-empty, NUL-terminated string within size.
inline bool IsValidString(const char* buf, size_t size) {
  if (buf == nullptr || size == 0) return false;
  size_t len = strnlen(buf, size);
  return len > 0 && len < size;
}

// Enumerates and classifies the processors AMD SMI reports. Acquire() and
// Release() are driven from a fixture's SetUpTestSuite/TearDownTestSuite, never
// from a constructor or destructor, so the GTest assertions inside
// TestBase::SetUp() and TestBase::Close() run where GTest supports them.
class DeviceInventory : public TestBase {
 public:
  void Acquire() {
    // The inventory outlives a single acquire/release cycle, so drop the
    // previous cycle's handles: amdsmi_shut_down() invalidated them and a stale
    // one answers AMDSMI_STATUS_NOT_FOUND on every call.
    gpus_.clear();
    cpus_.clear();
    cpu_cores_.clear();
    nics_.clear();
    cpu_supported_ = false;
    TestBase::SetUp(AMDSMI_INIT_ALL_PROCESSORS);
    Classify();
  }

  void Release() { TestBase::Close(); }

  bool initialized() const { return !setup_failed_; }
  // False when the ESMI/HSMP stack never came up, so CPU positive cases skip.
  bool cpu_supported() const { return cpu_supported_; }
  const std::vector<amdsmi_socket_handle>& sockets() const { return sockets_; }
  const std::vector<amdsmi_processor_handle>& gpus() const { return gpus_; }
  const std::vector<amdsmi_processor_handle>& cpus() const { return cpus_; }
  const std::vector<amdsmi_processor_handle>& cpu_cores() const { return cpu_cores_; }
  const std::vector<amdsmi_processor_handle>& nics() const { return nics_; }

 private:
  // The socket walk and amdsmi_get_cpu_handles() both hand back CPU handles on a
  // host whose ESMI/HSMP driver never came up, and every call through them then
  // fails with AMDSMI_STATUS_NOT_INIT. Probe a socketless CPU getter so positive
  // cases skip; negative cases take kInvalidHandle and stay live.
  static bool CpuStackUsable() {
    uint32_t threads_per_core = 0;
    return amdsmi_get_threads_per_core(&threads_per_core) != AMDSMI_STATUS_NOT_INIT;
  }

  void Classify() {
    for (uint32_t i = 0; i < num_monitor_devs_; ++i) {
      amdsmi_processor_type_t type = AMDSMI_PROCESSOR_TYPE_UNKNOWN;
      if (amdsmi_get_processor_type(processor_handles_[i], &type) != AMDSMI_STATUS_SUCCESS)
        continue;
      switch (type) {
        case AMDSMI_PROCESSOR_TYPE_AMD_GPU:
          gpus_.push_back(processor_handles_[i]);
          break;
        case AMDSMI_PROCESSOR_TYPE_AMD_CPU:
          cpus_.push_back(processor_handles_[i]);
          break;
        case AMDSMI_PROCESSOR_TYPE_AMD_CPU_CORE:
          cpu_cores_.push_back(processor_handles_[i]);
          break;
        case AMDSMI_PROCESSOR_TYPE_AMD_NIC:
        case AMDSMI_PROCESSOR_TYPE_BRCM_NIC:
          nics_.push_back(processor_handles_[i]);
          break;
        default:
          break;
      }
    }

    // Fallbacks through the dedicated enumeration APIs when the socket walk did
    // not surface CPU/NIC processors on this platform.
    if (cpus_.empty()) {
      uint32_t cpu_count = 0;
      if (amdsmi_get_cpu_handles(&cpu_count, nullptr) == AMDSMI_STATUS_SUCCESS && cpu_count > 0) {
        cpus_.resize(cpu_count);
        if (amdsmi_get_cpu_handles(&cpu_count, cpus_.data()) != AMDSMI_STATUS_SUCCESS)
          cpus_.clear();
      }
    }
    if (cpu_cores_.empty()) {
      uint32_t core_count = 0;
      if (amdsmi_get_cpucore_handles(&core_count, nullptr) == AMDSMI_STATUS_SUCCESS &&
          core_count > 0) {
        cpu_cores_.resize(core_count);
        if (amdsmi_get_cpucore_handles(&core_count, cpu_cores_.data()) != AMDSMI_STATUS_SUCCESS)
          cpu_cores_.clear();
      }
    }
    cpu_supported_ = CpuStackUsable();
    if (!cpu_supported_) {
      cpus_.clear();
      cpu_cores_.clear();
    }
    // NIC handles are socket-scoped; amdsmi_get_nic_processor_handles is NIC-specific
    // unlike the CPU/core APIs which are socketless.
    if (nics_.empty()) {
      for (auto socket : sockets_) {
        uint32_t nic_count = 0;
        if (amdsmi_get_nic_processor_handles(socket, &nic_count, nullptr) !=
                AMDSMI_STATUS_SUCCESS ||
            nic_count == 0)
          continue;
        std::vector<amdsmi_processor_handle> nic_procs(nic_count);
        if (amdsmi_get_nic_processor_handles(socket, &nic_count, nic_procs.data()) ==
            AMDSMI_STATUS_SUCCESS)
          nics_.insert(nics_.end(), nic_procs.begin(), nic_procs.end());
      }
    }
  }

  bool cpu_supported_ = false;
  std::vector<amdsmi_processor_handle> gpus_;
  std::vector<amdsmi_processor_handle> cpus_;
  std::vector<amdsmi_processor_handle> cpu_cores_;
  std::vector<amdsmi_processor_handle> nics_;
};

// One inventory shared by every suite that asks for it, reference counted so a
// run pays for amdsmi_init once per suite instead of once per test.
inline DeviceInventory& SharedInventory() {
  static DeviceInventory inv;
  return inv;
}

inline int& SharedInventoryRefs() {
  static int refs = 0;
  return refs;
}

inline void AcquireSharedInventory() {
  if (SharedInventoryRefs()++ > 0) return;
  DeviceInventory& inv = SharedInventory();
  inv.set_title("AMD SMI C API suite");
  inv.set_description("Shared device enumeration for the AMD SMI C API suites.");
  uint32_t level = GetTestVerbosity();
  if (level < TestBase::VERBOSE_STANDARD) level = TestBase::VERBOSE_STANDARD;
  inv.set_verbosity(level);
  inv.DisplayTestInfo();
  inv.Acquire();
  inv.Run();
}

inline void ReleaseSharedInventory() {
  if (SharedInventoryRefs() == 0) return;
  if (--SharedInventoryRefs() > 0) return;
  SharedInventory().DisplayResults();
  SharedInventory().Release();
}

// Accumulates the inputs whose call returned an unexpected result across a
// multi-input (id / enum) loop, then fails the test once, listing every failed
// input, instead of letting a bad code pass silently.
class StatusCollector {
 public:
  explicit StatusCollector(std::string api) : api_(std::move(api)) {}

  ~StatusCollector() {
    // Stay quiet when the test is already unwinding from its own failure or a
    // GTEST_SKIP(); a missing ExpectNoFailures() is then a symptom, not a cause.
    if (total_ > 0 && !reported_ && !::testing::Test::HasFailure() && !::testing::Test::IsSkipped())
      ADD_FAILURE() << api_ << ": StatusCollector destroyed without calling ExpectNoFailures()";
  }

  StatusCollector(const StatusCollector&) = delete;
  StatusCollector& operator=(const StatusCollector&) = delete;

  const std::string& api() const { return api_; }

  // Negative case: expected is the caller's own contract check.
  void Record(const std::string& input, amdsmi_status_t err, bool expected) {
    ++total_;
    if (expected) {
      ++passed_;
    } else {
      failures_.push_back(input + " -> returned " + AmdsmiStatusLabel(err));
    }
  }

  // Positive case. SUCCESS counts only when the call actually wrote its output;
  // a feature-absent status counts the device as skipped; anything else fails.
  void RecordPositive(const std::string& input, amdsmi_status_t err, bool wrote_output) {
    ++total_;
    if (err == AMDSMI_STATUS_SUCCESS) {
      if (wrote_output) {
        ++passed_;
      } else {
        failures_.push_back(input + " -> returned SUCCESS but left the output untouched");
      }
      return;
    }
    if (IsFeatureAbsent(err)) {
      ++skipped_;
      return;
    }
    failures_.push_back(input + " -> returned " + AmdsmiStatusLabel(err) +
                        ", expected SUCCESS or a feature-absent status");
  }

  // Status-only positive case, for a call whose output is validated separately
  // (or whose "written" state cannot be judged from the bytes alone).
  void RecordPositive(const std::string& input, amdsmi_status_t err) {
    RecordPositive(input, err, true);
  }

  // Flags a value-level problem on a call that reported SUCCESS.
  void RecordBadOutput(const std::string& input, const std::string& detail) {
    failures_.push_back(input + " -> returned SUCCESS but " + detail);
  }

  // True when every input was feature-absent, so the test proved nothing here.
  bool NothingExercised() const { return passed_ == 0 && failures_.empty() && skipped_ > 0; }

  void ExpectNoFailures() {
    reported_ = true;
    if (failures_.empty()) return;
    std::string msg = api_ + ": " + std::to_string(failures_.size()) + " of " +
                      std::to_string(total_) + " input(s) returned an unexpected result:";
    for (const auto& f : failures_) msg += "\n    " + f;
    ADD_FAILURE() << msg;
  }

 private:
  std::string api_;
  std::size_t total_ = 0;
  std::size_t passed_ = 0;
  std::size_t skipped_ = 0;
  bool reported_ = false;
  std::vector<std::string> failures_;
};

// Fixture base for the suites that drive the live AMD SMI C API. The shared
// inventory is acquired once per suite rather than once per test: the only test
// that cares about the init refcount balances its own extra init/shut_down pair.
class ApiTest : public ::testing::Test {
 public:
  static void SetUpTestSuite() { AcquireSharedInventory(); }
  static void TearDownTestSuite() { ReleaseSharedInventory(); }

 protected:
  // On a host where amdsmi_init() failed -- no amdgpu driver, for instance --
  // every call would otherwise return AMDSMI_STATUS_NOT_INIT and each test
  // would report that same failure. Skip instead, and let the init test report
  // the real cause once. Only suites that acquired the shared inventory are
  // gated, so a self-managed test still owns its own init sequence.
  void SetUp() override {
    if (SharedInventoryRefs() > 0 && !SharedInventory().initialized())
      GTEST_SKIP() << "AMD SMI could not initialize on this host";
  }

  void TearDown() override {
    if (!test_scoped_ref_) return;
    ReleaseSharedInventory();
    test_scoped_ref_ = false;
  }

  // Acquire for a suite whose SetUpTestSuite deliberately does not, so a test
  // that owns its own amdsmi_init sequence is never initialized behind its back.
  void RequireInit() {
    if (SharedInventoryRefs() > 0) return;
    AcquireSharedInventory();
    test_scoped_ref_ = true;
  }

  DeviceInventory& devices() {
    RequireInit();
    return SharedInventory();
  }

  bool initialized() { return devices().initialized(); }
  bool cpu_supported() { return devices().cpu_supported(); }
  const std::vector<amdsmi_socket_handle>& sockets() { return devices().sockets(); }
  const std::vector<amdsmi_processor_handle>& gpus() { return devices().gpus(); }
  const std::vector<amdsmi_processor_handle>& cpus() { return devices().cpus(); }
  const std::vector<amdsmi_processor_handle>& cpu_cores() { return devices().cpu_cores(); }
  const std::vector<amdsmi_processor_handle>& nics() { return devices().nics(); }

  // Handle for a negative test that needs to reach an API's argument check on a
  // host that has no such processor. Argument validation does not depend on the
  // device, so falling back to the invalid-handle sentinel keeps invalid-input
  // cases running everywhere instead of skipping them.
  amdsmi_processor_handle any_gpu() { return gpus().empty() ? kInvalidHandle : gpus()[0]; }
  amdsmi_processor_handle any_cpu() { return cpus().empty() ? kInvalidHandle : cpus()[0]; }
  amdsmi_processor_handle any_cpu_core() {
    return cpu_cores().empty() ? kInvalidHandle : cpu_cores()[0];
  }
  amdsmi_processor_handle any_nic() { return nics().empty() ? kInvalidHandle : nics()[0]; }

 private:
  bool test_scoped_ref_ = false;
};

// Base for suites shared with tests that own their amdsmi_init()/shut_down()
// sequence (the mutual-exclusion and cross-process cases in main.cc). Devices
// are acquired only if the test actually asks for them.
class SelfManagedApiTest : public ApiTest {
 public:
  static void SetUpTestSuite() {}
  static void TearDownTestSuite() {}
};

}  // namespace test
}  // namespace amdsmi

// A TEST_F() suite name is its fixture class name. The three tiers are:
//
//   <Component>Unit          no device and no amdsmi_init -- pure logic and
//                            static data; declared in unit_fixtures.h
//   <Component>Integration   every API's invalid-input cases, plus getters
//                            driven with valid input and checked for valid output
//   <Component>Functional*   setters, and APIs that need setup from another API
//
class GpuIntegration : public amdsmi::test::ApiTest {};
class CpuIntegration : public amdsmi::test::ApiTest {};
class NicIntegration : public amdsmi::test::ApiTest {};
class SystemIntegration : public amdsmi::test::ApiTest {};

class GpuFunctionalReadOnly : public amdsmi::test::SelfManagedApiTest {};
class GpuFunctionalReadWrite : public amdsmi::test::SelfManagedApiTest {};
class CpuFunctionalReadWrite : public amdsmi::test::SelfManagedApiTest {};
class NicFunctionalReadOnly : public amdsmi::test::SelfManagedApiTest {};
class SystemFunctionalReadOnly : public amdsmi::test::SelfManagedApiTest {};
class IfoeFunctionalReadOnly : public amdsmi::test::SelfManagedApiTest {};
class IfoeFunctionalReadWrite : public amdsmi::test::SelfManagedApiTest {};

// Assert an API returned the status it SHOULD for a null pointer argument.
// A feature-absent status is tolerated because an unimplemented or unsupported
// API never reaches its argument check. SUCCESS never is -- that is the real
// contract violation these cases exist to catch.
#define AMDSMI_EXPECT_NULL_ARG(actual)                                              \
  EXPECT_TRUE(::amdsmi::test::AmdsmiStatusIsExpected((actual), AMDSMI_STATUS_INVAL, \
                                                     AMDSMI_STATUS_ARG_PTR_NULL))   \
      << "returned " << ::amdsmi::test::AmdsmiStatusLabel(actual)                   \
      << ", expected AMDSMI_STATUS_INVAL or AMDSMI_STATUS_ARG_PTR_NULL "            \
      << "(or a status meaning the feature is absent here)"

// Assert an API rejected an invalid processor handle. The out-param handed in
// with it is valid, so AMDSMI_STATUS_ARG_PTR_NULL would be a wrong answer about a
// non-null pointer and is not accepted here. A feature-absent status is, because
// the call can bail before reaching the handle check. SUCCESS is never acceptable.
#define AMDSMI_EXPECT_INVALID_HANDLE(actual)                                         \
  EXPECT_TRUE(::amdsmi::test::AmdsmiStatusIsExpected((actual), AMDSMI_STATUS_INVAL)) \
      << "returned " << ::amdsmi::test::AmdsmiStatusLabel(actual)                    \
      << ", expected AMDSMI_STATUS_INVAL for an invalid handle "                     \
      << "(or a status meaning the feature is absent here)"

// Non-fatal check for a single negative call: fails if err is neither one of the
// listed acceptable codes nor a genuine platform limitation.
#define AMDSMI_EXPECT_STATUS(err, ...)                                    \
  EXPECT_TRUE(::amdsmi::test::AmdsmiStatusIsExpected((err), __VA_ARGS__)) \
      << "returned " << ::amdsmi::test::AmdsmiStatusLabel(err)            \
      << ", which is not an expected status"

// Close out a multi-input positive loop: report every bad input, then skip when
// the feature was absent on every one so the pass is not mistaken for coverage.
#define AMDSMI_FINISH_POSITIVE(col)                                        \
  do {                                                                     \
    (col).ExpectNoFailures();                                              \
    if ((col).NothingExercised())                                          \
      GTEST_SKIP() << (col).api() << ": not supported on any device here"; \
  } while (0)

// A device write needs the privilege to perform it. Most *FunctionalReadWrite
// tests record the original value and restore it; the setters the API cannot
// read back are named in known_failures.md and the suite README.
#define AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED()                                                      \
  do {                                                                                             \
    if (::amdsmi::test::EnvFlagEnabled("AMDSMI_NON_PRIVILEGED"))                                   \
      GTEST_SKIP() << "device write skipped; AMDSMI_NON_PRIVILEGED is set";                        \
    if (!amd::smi::is_sudo_user()) GTEST_SKIP() << "device write skipped; must run as super user"; \
  } while (0)

// Skip a test blocked by an entry in known_failures.md. Stream the symptom onto
// it; AMDSMI_RUN_KNOWN_FAILURES=1 runs the test instead of skipping.
// Expands to a bare if -- callers stream onto it, so do/while(0) is unavailable.
// Never place it inside an unbraced if/else, where a later else would bind here.
#define AMDSMI_SKIP_KNOWN_FAILURE() \
  if (!::amdsmi::test::RunKnownFailures()) GTEST_SKIP()

// Handle for a negative case on a host with no such processor: argument checks
// do not depend on the device, so fall back to the invalid-handle sentinel.
#define AMDSMI_ANY_HANDLE(HANDLES) \
  (HANDLES().empty() ? ::amdsmi::test::kInvalidHandle : HANDLES()[0])

// The null-output / invalid-handle / all-processors trio below is the shape
// almost every read-only getter is tested with. Generating it keeps one copy of
// the expectations -- including the SUCCESS-path output check -- instead of one
// per API. FIXTURE/HANDLES/LABEL select the component, so CPU and NIC getters
// use the same generator as GPU ones. Use the long form only where an API needs
// different inputs.
#define AMDSMI_INTEGRATION_STRUCT_GETTER(FIXTURE, HANDLES, LABEL, TESTBASE, APINAME, STRUCT)       \
  TEST_F(FIXTURE, TESTBASE##_NullOutput) {                                                         \
    DISPLAY_AMDSMI_API(#APINAME, "out=nullptr", ::amdsmi::test::kVerbose);                         \
    amdsmi_status_t err = APINAME(AMDSMI_ANY_HANDLE(HANDLES), nullptr);                            \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_NULL_ARG(err);                                                                   \
  }                                                                                                \
  TEST_F(FIXTURE, TESTBASE##_InvalidHandle) {                                                      \
    STRUCT info;                                                                                   \
    ::amdsmi::test::PoisonOutput(&info, sizeof(info));                                             \
    DISPLAY_AMDSMI_API(#APINAME, "handle=invalid", ::amdsmi::test::kVerbose);                      \
    amdsmi_status_t err = APINAME(::amdsmi::test::kInvalidHandle, &info);                          \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_INVALID_HANDLE(err);                                                             \
  }                                                                                                \
  TEST_F(FIXTURE, TESTBASE##_All##LABEL) {                                                         \
    ::amdsmi::test::StatusCollector col(#APINAME);                                                 \
    if (HANDLES().empty()) GTEST_SKIP() << "No " #LABEL " processors";                             \
    for (size_t i = 0; i < HANDLES().size(); ++i) {                                                \
      STRUCT info;                                                                                 \
      ::amdsmi::test::PoisonOutput(&info, sizeof(info));                                           \
      const std::string in = #LABEL "=" + std::to_string(i);                                       \
      DISPLAY_AMDSMI_API(#APINAME, in, ::amdsmi::test::kVerbose);                                  \
      amdsmi_status_t err = APINAME(HANDLES()[i], &info);                                          \
      DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err,                     \
                            AMDSMI_STATUS_SUCCESS);                                                \
      col.RecordPositive(in, err, ::amdsmi::test::OutputWasWritten(&info, sizeof(info)));          \
    }                                                                                              \
    AMDSMI_FINISH_POSITIVE(col);                                                                   \
  }

#define AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(TESTBASE, APINAME, STRUCT) \
  AMDSMI_INTEGRATION_STRUCT_GETTER(GpuIntegration, gpus, Gpus, TESTBASE, APINAME, STRUCT)

#define AMDSMI_INTEGRATION_CPU_STRUCT_GETTER(TESTBASE, APINAME, STRUCT) \
  AMDSMI_INTEGRATION_STRUCT_GETTER(CpuIntegration, cpus, Cpus, TESTBASE, APINAME, STRUCT)

#define AMDSMI_INTEGRATION_NIC_STRUCT_GETTER(TESTBASE, APINAME, STRUCT) \
  AMDSMI_INTEGRATION_STRUCT_GETTER(NicIntegration, nics, Nics, TESTBASE, APINAME, STRUCT)

// Same trio for getters that fill a caller-supplied character buffer. The
// SUCCESS path checks the buffer holds a real string, not just that the call
// reported success.
#define AMDSMI_INTEGRATION_BUFFER_GETTER(FIXTURE, HANDLES, LABEL, TESTBASE, APINAME, BUFSIZE)      \
  TEST_F(FIXTURE, TESTBASE##_NullOutput) {                                                         \
    DISPLAY_AMDSMI_API(#APINAME, "out=nullptr", ::amdsmi::test::kVerbose);                         \
    amdsmi_status_t err = APINAME(AMDSMI_ANY_HANDLE(HANDLES), nullptr, BUFSIZE);                   \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_NULL_ARG(err);                                                                   \
  }                                                                                                \
  TEST_F(FIXTURE, TESTBASE##_InvalidHandle) {                                                      \
    char buf[BUFSIZE];                                                                             \
    memset(buf, 0, sizeof(buf));                                                                   \
    DISPLAY_AMDSMI_API(#APINAME, "handle=invalid", ::amdsmi::test::kVerbose);                      \
    amdsmi_status_t err = APINAME(::amdsmi::test::kInvalidHandle, buf, sizeof(buf));               \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_INVALID_HANDLE(err);                                                             \
  }                                                                                                \
  TEST_F(FIXTURE, TESTBASE##_All##LABEL) {                                                         \
    ::amdsmi::test::StatusCollector col(#APINAME);                                                 \
    if (HANDLES().empty()) GTEST_SKIP() << "No " #LABEL " processors";                             \
    for (size_t i = 0; i < HANDLES().size(); ++i) {                                                \
      char buf[BUFSIZE];                                                                           \
      memset(buf, 0, sizeof(buf));                                                                 \
      const std::string in = #LABEL "=" + std::to_string(i);                                       \
      DISPLAY_AMDSMI_API(#APINAME, in, ::amdsmi::test::kVerbose);                                  \
      amdsmi_status_t err = APINAME(HANDLES()[i], buf, sizeof(buf));                               \
      DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err,                     \
                            AMDSMI_STATUS_SUCCESS);                                                \
      col.RecordPositive(in, err, ::amdsmi::test::IsValidString(buf, sizeof(buf)));                \
    }                                                                                              \
    AMDSMI_FINISH_POSITIVE(col);                                                                   \
  }

#define AMDSMI_INTEGRATION_GPU_BUFFER_GETTER(TESTBASE, APINAME, BUFSIZE) \
  AMDSMI_INTEGRATION_BUFFER_GETTER(GpuIntegration, gpus, Gpus, TESTBASE, APINAME, BUFSIZE)

#endif  // TESTS_AMD_SMI_TEST_API_TEST_FRAMEWORK_H_
