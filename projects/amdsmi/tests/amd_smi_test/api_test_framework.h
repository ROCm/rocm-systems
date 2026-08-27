/*
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

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

// Enumerates and classifies the processors AMD SMI reports. Acquire() and
// Release() are driven from a fixture's test body and TearDown, never from a
// constructor or destructor, so the GTest assertions inside TestBase::SetUp()
// and TestBase::Close() run where GTest supports them.
class DeviceInventory : public TestBase {
 public:
  void Acquire() {
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

// Fixture base for the suites that drive the live AMD SMI C API. Initialization
// and enumeration run in SetUp() and the shutdown in TearDown(), which is where
// GTest supports the assertions TestBase::SetUp()/Close() make. Each test still
// gets a fresh init so the reference-count tests see a zero refcount between
// cases.
class ApiTest : public ::testing::Test {
 protected:
  // Suites that also carry tests driving amdsmi_init() themselves override this
  // so the fixture never initializes behind their back.
  virtual bool AcquireInSetUp() const { return true; }

  void SetUp() override {
    if (AcquireInSetUp()) EnsureAcquired();
  }

  void TearDown() override {
    if (!acquired_) return;
    inv_.DisplayResults();
    inv_.Release();
    acquired_ = false;
  }

  DeviceInventory& devices() {
    EnsureAcquired();
    return inv_;
  }

  // For tests that call the API without going through a device handle, on a
  // suite whose fixture does not initialize by itself.
  void RequireInit() { EnsureAcquired(); }

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
  void EnsureAcquired() {
    if (acquired_) return;

    const ::testing::TestInfo* info = ::testing::UnitTest::GetInstance()->current_test_info();
    const std::string full_name =
        std::string(info->test_suite_name()) + "." + std::string(info->name());
    inv_.set_title(full_name);
    inv_.set_description("Verification of the AMD SMI C API exercised by " + full_name + ".");

    uint32_t level = GetTestVerbosity();
    if (level < TestBase::VERBOSE_STANDARD) level = TestBase::VERBOSE_STANDARD;
    inv_.set_verbosity(level);

    acquired_ = true;  // set first so TearDown still releases a partial init
    inv_.DisplayTestInfo();
    inv_.Acquire();
    inv_.Run();
  }

  DeviceInventory inv_;
  bool acquired_ = false;
};

// Base for suites shared with tests that own their amdsmi_init()/shut_down()
// sequence (the mutual-exclusion and cross-process cases in main.cc). Devices
// are acquired only if the test actually asks for them.
class SelfManagedApiTest : public ApiTest {
 protected:
  bool AcquireInSetUp() const override { return false; }
};

// Statuses that mean a case could not be exercised on this platform (feature
// absent, stack not brought up, or insufficient privilege) rather than a real
// contract violation.
inline bool IsUntestableHere(amdsmi_status_t status) {
  return status == AMDSMI_STATUS_NOT_SUPPORTED || status == AMDSMI_STATUS_NOT_YET_IMPLEMENTED ||
         status == AMDSMI_STATUS_NO_PERM ||
         // Required kernel driver / interface not present on this system.
         status == AMDSMI_STATUS_NO_HSMP_MSG_SUP || status == AMDSMI_STATUS_NO_HSMP_SUP ||
         status == AMDSMI_STATUS_NO_HSMP_DRV || status == AMDSMI_STATUS_NO_ENERGY_DRV ||
         status == AMDSMI_STATUS_NO_MSR_DRV || status == AMDSMI_STATUS_NO_DRV ||
         status == AMDSMI_STATUS_DRIVER_NOT_LOADED || status == AMDSMI_STATUS_NON_AMD_CPU;
}

// Strict match: err must be one of the caller-listed codes.
template <typename... Args>
inline bool AmdsmiStatusMatches(amdsmi_status_t err, Args... expected) {
  const amdsmi_status_t codes[] = {expected...};
  for (amdsmi_status_t c : codes) {
    if (err == c) return true;
  }
  return false;
}

// Lenient match: the caller-listed codes plus any status meaning the call could
// not be exercised here.
template <typename... Args>
inline bool AmdsmiStatusIsExpected(amdsmi_status_t err, Args... expected) {
  return AmdsmiStatusMatches(err, expected...) || IsUntestableHere(err);
}

// Human-readable "N (NAME)" for a status code.
inline std::string AmdsmiStatusLabel(amdsmi_status_t err) {
  const char* name = nullptr;
  amdsmi_status_code_to_string(err, &name);
  std::string out = std::to_string(static_cast<int>(err));
  if (name != nullptr) out += std::string(" (") + name + ")";
  return out;
}

// True when a zero-initialized output buffer was actually written. A call that
// reports SUCCESS without touching its out-param violates its contract in a way
// no status-only check can see.
inline bool WroteOutput(const void* out, size_t size) {
  const unsigned char* p = static_cast<const unsigned char*>(out);
  for (size_t i = 0; i < size; ++i) {
    if (p[i] != 0) return true;
  }
  return false;
}

// True when buf holds a non-empty, NUL-terminated string within size.
inline bool IsValidString(const char* buf, size_t size) {
  if (buf == nullptr || size == 0) return false;
  size_t len = strnlen(buf, size);
  return len > 0 && len < size;
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

  void Record(const std::string& input, amdsmi_status_t err, bool expected) {
    ++total_;
    if (!expected) {
      failures_.push_back(input + " -> returned " + AmdsmiStatusLabel(err));
    }
  }

  // Flags a value-level problem on a call that reported SUCCESS.
  void RecordBadOutput(const std::string& input, const std::string& detail) {
    failures_.push_back(input + " -> returned SUCCESS but " + detail);
  }

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
  bool reported_ = false;
  std::vector<std::string> failures_;
};

// Puts a device value back when the test leaves scope, so a functional setter
// test restores the hardware even if an assertion fails partway through.
// Construct only after the original value has been read successfully.
template <typename T>
class ScopedRestore {
 public:
  using Restore = amdsmi_status_t (*)(amdsmi_processor_handle, T);

  ScopedRestore(amdsmi_processor_handle handle, T original, Restore restore)
      : handle_(handle), original_(original), restore_(restore) {}

  ~ScopedRestore() {
    if (!armed_) return;
    amdsmi_status_t err = restore_(handle_, original_);
    if (err != AMDSMI_STATUS_SUCCESS && !IsUntestableHere(err))
      ADD_FAILURE() << "failed to restore original value: " << AmdsmiStatusLabel(err);
  }

  ScopedRestore(const ScopedRestore&) = delete;
  ScopedRestore& operator=(const ScopedRestore&) = delete;

  T original() const { return original_; }
  void disarm() { armed_ = false; }

 private:
  amdsmi_processor_handle handle_;
  T original_;
  Restore restore_;
  bool armed_ = true;
};

}  // namespace test
}  // namespace amdsmi

// A TEST_F() suite name is its fixture class name. The three tiers are:
//
//   <Component>Unit          no device and no amdsmi_init -- pure logic and
//                            static data
//   <Component>Integration   every API's invalid-input cases, plus getters
//                            driven with valid input and checked for valid output
//   <Component>Functional*   setters, and APIs that need setup from another API
//
class GpuUnit : public ::testing::Test {};
class SystemUnit : public ::testing::Test {};

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
// amdsmi.h documents both outcomes for a null out-param: AMDSMI_STATUS_INVAL when
// the function is supported with the given arguments, AMDSMI_STATUS_NOT_SUPPORTED
// when it is not -- so a feature-absent status is a pass, not a miss.
// SUCCESS is never acceptable.
#define AMDSMI_EXPECT_NULL_ARG(actual)                                              \
  EXPECT_TRUE(::amdsmi::test::AmdsmiStatusIsExpected((actual), AMDSMI_STATUS_INVAL, \
                                                     AMDSMI_STATUS_ARG_PTR_NULL))   \
      << "returned " << ::amdsmi::test::AmdsmiStatusLabel(actual)                   \
      << ", expected AMDSMI_STATUS_INVAL or AMDSMI_STATUS_NOT_SUPPORTED"

// Non-fatal check for a single valid call: fails if err is neither one of the
// listed acceptable codes nor a genuine platform limitation.
#define AMDSMI_EXPECT_STATUS(err, ...)                                    \
  EXPECT_TRUE(::amdsmi::test::AmdsmiStatusIsExpected((err), __VA_ARGS__)) \
      << "returned " << ::amdsmi::test::AmdsmiStatusLabel(err)            \
      << ", which is not an expected status"

// Destructive hardware writes are opt-in: off unless AMDSMI_TEST_ALLOW_MUTATION
// is set, and skipped without the privilege the write needs, so a default run
// never disturbs a shared GPU/CPU.
#define AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED()                                                     \
  do {                                                                                            \
    if (std::getenv("AMDSMI_TEST_ALLOW_MUTATION") == nullptr)                                     \
      GTEST_SKIP() << "destructive write skipped; set AMDSMI_TEST_ALLOW_MUTATION to run";         \
    if (std::getenv("AMDSMI_NON_PRIVILEGED") != nullptr)                                          \
      GTEST_SKIP() << "Skipped in non-privileged mode";                                           \
    if (!amd::smi::is_sudo_user()) GTEST_SKIP() << "Invalid permission - Must run as super user"; \
  } while (0)

// The null-output / invalid-handle / all-GPUs trio below is the shape almost
// every read-only getter is tested with. Generating it keeps one copy of the
// expectations -- including the SUCCESS-path output check -- instead of one per
// API. Use the explicit long form only where an API needs different inputs.
#define AMDSMI_INTEGRATION_GPU_STRUCT_GETTER(TESTBASE, APINAME, STRUCT)                            \
  TEST_F(GpuIntegration, TESTBASE##_NullOutput) {                                                  \
    DISPLAY_AMDSMI_API(#APINAME, "gpu=0 out=nullptr", ::amdsmi::test::kVerbose);                   \
    amdsmi_status_t err = APINAME(any_gpu(), nullptr);                                             \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_NULL_ARG(err);                                                                   \
  }                                                                                                \
  TEST_F(GpuIntegration, TESTBASE##_InvalidHandle) {                                               \
    STRUCT info;                                                                                   \
    memset(&info, 0, sizeof(info));                                                                \
    DISPLAY_AMDSMI_API(#APINAME, "handle=invalid", ::amdsmi::test::kVerbose);                      \
    amdsmi_status_t err = APINAME(::amdsmi::test::kInvalidHandle, &info);                          \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,  \
                          AMDSMI_STATUS_NOT_SUPPORTED);                                            \
    EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);                                                         \
  }                                                                                                \
  TEST_F(GpuIntegration, TESTBASE##_AllGpus) {                                                     \
    ::amdsmi::test::StatusCollector col(#APINAME);                                                 \
    if (gpus().empty()) GTEST_SKIP() << "No GPU processors";                                       \
    for (size_t i = 0; i < gpus().size(); ++i) {                                                   \
      STRUCT info;                                                                                 \
      memset(&info, 0, sizeof(info));                                                              \
      const std::string in = "gpu=" + std::to_string(i);                                           \
      DISPLAY_AMDSMI_API(#APINAME, in, ::amdsmi::test::kVerbose);                                  \
      amdsmi_status_t err = APINAME(gpus()[i], &info);                                             \
      DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err,                     \
                            AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,                    \
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED);                                    \
      col.Record(in, err,                                                                          \
                 ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,                \
                                                        AMDSMI_STATUS_NOT_SUPPORTED,               \
                                                        AMDSMI_STATUS_NOT_YET_IMPLEMENTED));       \
      if (err == AMDSMI_STATUS_SUCCESS && !::amdsmi::test::WroteOutput(&info, sizeof(info)))       \
        col.RecordBadOutput(in, "left the " #STRUCT " output untouched");                          \
    }                                                                                              \
    col.ExpectNoFailures();                                                                        \
  }

// Same trio for getters that fill a caller-supplied character buffer.
#define AMDSMI_INTEGRATION_GPU_BUFFER_GETTER(TESTBASE, APINAME, BUFSIZE)                           \
  TEST_F(GpuIntegration, TESTBASE##_NullOutput) {                                                  \
    DISPLAY_AMDSMI_API(#APINAME, "gpu=0 out=nullptr", ::amdsmi::test::kVerbose);                   \
    amdsmi_status_t err = APINAME(any_gpu(), nullptr, BUFSIZE);                                    \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL); \
    AMDSMI_EXPECT_NULL_ARG(err);                                                                   \
  }                                                                                                \
  TEST_F(GpuIntegration, TESTBASE##_InvalidHandle) {                                               \
    char buf[BUFSIZE];                                                                             \
    memset(buf, 0, sizeof(buf));                                                                   \
    DISPLAY_AMDSMI_API(#APINAME, "handle=invalid", ::amdsmi::test::kVerbose);                      \
    amdsmi_status_t err = APINAME(::amdsmi::test::kInvalidHandle, buf, sizeof(buf));               \
    DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_INVAL,  \
                          AMDSMI_STATUS_NOT_SUPPORTED);                                            \
    EXPECT_NE(err, AMDSMI_STATUS_SUCCESS);                                                         \
  }                                                                                                \
  TEST_F(GpuIntegration, TESTBASE##_AllGpus) {                                                     \
    ::amdsmi::test::StatusCollector col(#APINAME);                                                 \
    if (gpus().empty()) GTEST_SKIP() << "No GPU processors";                                       \
    for (size_t i = 0; i < gpus().size(); ++i) {                                                   \
      char buf[BUFSIZE];                                                                           \
      memset(buf, 0, sizeof(buf));                                                                 \
      const std::string in = "gpu=" + std::to_string(i);                                           \
      DISPLAY_AMDSMI_API(#APINAME, in, ::amdsmi::test::kVerbose);                                  \
      amdsmi_status_t err = APINAME(gpus()[i], buf, sizeof(buf));                                  \
      DISPLAY_AMDSMI_STATUS(::amdsmi::test::kVerbose, __FILE__, __LINE__, err,                     \
                            AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,                    \
                            AMDSMI_STATUS_NOT_YET_IMPLEMENTED);                                    \
      col.Record(in, err,                                                                          \
                 ::amdsmi::test::AmdsmiStatusIsExpected(err, AMDSMI_STATUS_SUCCESS,                \
                                                        AMDSMI_STATUS_NOT_SUPPORTED,               \
                                                        AMDSMI_STATUS_NOT_YET_IMPLEMENTED));       \
      if (err == AMDSMI_STATUS_SUCCESS && !::amdsmi::test::IsValidString(buf, sizeof(buf)))        \
        col.RecordBadOutput(in, "did not fill a NUL-terminated string");                           \
    }                                                                                              \
    col.ExpectNoFailures();                                                                        \
  }

#endif  // TESTS_AMD_SMI_TEST_API_TEST_FRAMEWORK_H_
