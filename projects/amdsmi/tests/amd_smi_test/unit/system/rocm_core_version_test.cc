// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Exercises amdsmi_get_rocm_version()'s librocm-core.so candidate search and
// status mapping directly, without a real librocm-core.so on the host. No
// GPU required.

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <optional>
#include <string>
#include <vector>

#include "amd_smi/impl/amd_smi_rocm_core_testing.h"

namespace {

// Saves an env var on construction and restores its prior value (set or
// absent) on destruction, so tests can freely setenv/unsetenv ROCM_HOME and
// ROCM_PATH without leaking state into other tests.
class EnvVarGuard {
 public:
  explicit EnvVarGuard(const char* name) : name_(name) {
    const char* value = std::getenv(name);
    if (value != nullptr) {
      prior_value_ = std::string(value);
    }
  }

  ~EnvVarGuard() {
    if (prior_value_.has_value()) {
      setenv(name_, prior_value_->c_str(), /*overwrite=*/1);
    } else {
      unsetenv(name_);
    }
  }

 private:
  const char* name_;
  std::optional<std::string> prior_value_;
};

bool Contains(const std::vector<std::string>& candidates, const std::string& value) {
  return std::find(candidates.begin(), candidates.end(), value) != candidates.end();
}

TEST(SystemUnit, RocmCoreCandidatesIncludesOptRocmAndBareSonames) {
  EnvVarGuard home_guard("ROCM_HOME");
  EnvVarGuard path_guard("ROCM_PATH");
  unsetenv("ROCM_HOME");
  unsetenv("ROCM_PATH");

  std::vector<std::string> candidates = rocm_core_candidates();

  EXPECT_TRUE(Contains(candidates, "/opt/rocm/lib/librocm-core.so.1"));
  EXPECT_TRUE(Contains(candidates, "/opt/rocm/lib/librocm-core.so"));
  ASSERT_GE(candidates.size(), 2u);
  EXPECT_EQ(candidates[candidates.size() - 2], "librocm-core.so.1");
  EXPECT_EQ(candidates[candidates.size() - 1], "librocm-core.so");
}

TEST(SystemUnit, RocmCoreCandidatesSearchesRocmHomeDir) {
  EnvVarGuard home_guard("ROCM_HOME");
  EnvVarGuard path_guard("ROCM_PATH");
  setenv("ROCM_HOME", "/tmp/fake-rocm-home", /*overwrite=*/1);
  unsetenv("ROCM_PATH");

  std::vector<std::string> candidates = rocm_core_candidates();

  EXPECT_TRUE(Contains(candidates, "/tmp/fake-rocm-home/lib/librocm-core.so.1"));
  EXPECT_TRUE(Contains(candidates, "/tmp/fake-rocm-home/lib/librocm-core.so"));
}

TEST(SystemUnit, RocmCoreCandidatesFallsBackToRocmPathDir) {
  EnvVarGuard home_guard("ROCM_HOME");
  EnvVarGuard path_guard("ROCM_PATH");
  unsetenv("ROCM_HOME");
  setenv("ROCM_PATH", "/tmp/fake-rocm-path", /*overwrite=*/1);

  std::vector<std::string> candidates = rocm_core_candidates();

  EXPECT_TRUE(Contains(candidates, "/tmp/fake-rocm-path/lib/librocm-core.so.1"));
  EXPECT_TRUE(Contains(candidates, "/tmp/fake-rocm-path/lib/librocm-core.so"));
}

TEST(SystemUnit, RocmCoreStatusToAmdsmiMapsSuccess) {
  amdsmi_version_t version{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};
  std::string build_str;

  amdsmi_status_t status =
      rocm_core_status_to_amdsmi(/*rocm_core_status=*/0, /*major=*/6, /*minor=*/2,
                                 /*release=*/1, &version, &build_str);

  EXPECT_EQ(status, AMDSMI_STATUS_SUCCESS);
  EXPECT_EQ(version.major, 6u);
  EXPECT_EQ(version.minor, 2u);
  EXPECT_EQ(version.release, 1u);
  ASSERT_NE(version.build, nullptr);
  EXPECT_STREQ(version.build, "6.2.1");
}

TEST(SystemUnit, RocmCoreStatusToAmdsmiMapsFailure) {
  amdsmi_version_t version{0xFFFFFFFF, 0xFFFFFFFF, 0xFFFFFFFF, nullptr};
  std::string build_str;

  amdsmi_status_t status =
      rocm_core_status_to_amdsmi(/*rocm_core_status=*/1, /*major=*/6, /*minor=*/2,
                                 /*release=*/1, &version, &build_str);

  EXPECT_EQ(status, AMDSMI_STATUS_API_FAILED);
  // A failed getROCmVersion() call leaves the out-params untouched.
  EXPECT_EQ(version.major, 0xFFFFFFFFu);
  EXPECT_EQ(version.build, nullptr);
  EXPECT_TRUE(build_str.empty());
}

}  // namespace
