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
#include "ampp_read_write.h"

#include <gtest/gtest.h>
#include <stddef.h>
#include <stdint.h>

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "test_common.h"

namespace {

// RAII helper to set and automatically restore an environment variable,
// mirroring the pattern already used by memory_read_write_test.cc for
// exercising AMDSMI_DRY_RUN-gated write paths without root/hardware access.
class ScopedEnvVar {
 public:
  ScopedEnvVar(const char* name, const char* value) : name_(name) {
    const char* old_val = std::getenv(name);
    if (old_val) {
      old_value_ = old_val;
      has_old_value_ = true;
    } else {
      has_old_value_ = false;
    }
    setenv(name, value, 1);
  }
  ~ScopedEnvVar() {
    if (has_old_value_) {
      setenv(name_.c_str(), old_value_.c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

 private:
  std::string name_;
  std::string old_value_;
  bool has_old_value_;
};

}  // namespace

TestAmppReadWrite::TestAmppReadWrite() : TestBase() {
  set_title("AMDSMI AMPP (Power Profile) Read/Write Test");
  set_description(
      "The AMPP tests verify that power profile recipes can be enumerated, "
      "queried, and activated/configured (via AMDSMI_DRY_RUN) properly.");
}

TestAmppReadWrite::~TestAmppReadWrite(void) {}

void TestAmppReadWrite::SetUp(void) {
  TestBase::SetUp();

  return;
}

void TestAmppReadWrite::DisplayTestInfo(void) { TestBase::DisplayTestInfo(); }

void TestAmppReadWrite::DisplayResults(void) const {
  TestBase::DisplayResults();
  return;
}

void TestAmppReadWrite::Close() {
  // This will close handles opened within amdsmitst utility calls and call
  // amdsmi_shut_down(), so it should be done after other hsa cleanup
  TestBase::Close();
}

void TestAmppReadWrite::Run(void) {
  amdsmi_status_t ret;

  TestBase::Run();
  PRINT_VERBOSITY();
  if (setup_failed_) {
    std::cout << "** SetUp Failed for this test. Skipping.**" << std::endl;
    return;
  }

  for (uint32_t dv_ind = 0; dv_ind < num_monitor_devs(); ++dv_ind) {
    PrintDeviceHeader(processor_handles_[dv_ind]);
    amdsmi_processor_handle handle = processor_handles_[dv_ind];

    // num_profiles == nullptr must be rejected regardless of app_modes/
    // support.
    DISPLAY_AMDSMI_API("amdsmi_get_ampp_profiles", "gpu=" + std::to_string(dv_ind) + ", NULL",
                       VERB(STANDARD));
    ret = amdsmi_get_ampp_profiles(handle, nullptr, nullptr, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // Query size first (profiles == nullptr). version may be non-NULL even
    // on the sizing call -- it is a single tree-wide string, not sized by
    // num_profiles.
    char version[AMDSMI_MAX_STRING_LENGTH] = {};
    uint32_t num_profiles = 0;
    DISPLAY_AMDSMI_API("amdsmi_get_ampp_profiles", "gpu=" + std::to_string(dv_ind) + " (size)",
                       VERB(STANDARD));
    ret = amdsmi_get_ampp_profiles(handle, version, nullptr, &num_profiles);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
      IF_VERB(STANDARD) {
        std::cout << "\t**AMPP not supported on this ASIC/VBIOS. Skipping." << std::endl;
      }
      continue;
    }
    CHK_ERR_ASRT(ret)
    IF_VERB(STANDARD) { std::cout << "\t  profile_abi version=" << version << std::endl; }

    // An under-sized caller buffer must be rejected with OUT_OF_RESOURCES
    // instead of overflowing it, and num_profiles must still be updated to
    // the true required count so the caller can retry correctly.
    if (num_profiles > 0) {
      // Fixed 1-element buffer (not sized to num_profiles - 1): an empty
      // vector's .data() returns nullptr, which the API treats as a sizing
      // call rather than an undersized buffer, so this must stay non-null.
      amdsmi_ampp_profile_t undersized[1];
      uint32_t requested = num_profiles - 1;
      DISPLAY_AMDSMI_API("amdsmi_get_ampp_profiles",
                         "gpu=" + std::to_string(dv_ind) + " (undersized buffer)", VERB(STANDARD));
      ret = amdsmi_get_ampp_profiles(handle, version, undersized, &requested);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret,
                            AMDSMI_STATUS_OUT_OF_RESOURCES);
      ASSERT_EQ(ret, AMDSMI_STATUS_OUT_OF_RESOURCES);
      ASSERT_EQ(requested, num_profiles);
    }

    std::vector<amdsmi_ampp_profile_t> profiles(num_profiles);
    DISPLAY_AMDSMI_API("amdsmi_get_ampp_profiles", "gpu=" + std::to_string(dv_ind) + " (fill)",
                       VERB(STANDARD));
    ret = amdsmi_get_ampp_profiles(handle, version, profiles.data(), &num_profiles);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
    CHK_ERR_ASRT(ret)

    int active_count = 0;
    std::string configured_profile_name;
    std::string unconfigured_writable_profile_name;
    for (const auto& p : profiles) {
      IF_VERB(STANDARD) {
        std::cout << "\t  profile=" << p.name << " index=" << p.index << " active=" << p.is_active
                  << " writable=" << p.is_writable << " configured=" << p.is_configured
                  << std::endl;
      }
      if (p.is_active) {
        active_count++;
      }
      if (p.is_configured && configured_profile_name.empty()) {
        configured_profile_name = p.name;
      }
      if (p.is_writable && !p.is_configured && unconfigured_writable_profile_name.empty()) {
        unconfigured_writable_profile_name = p.name;
      }
    }
    // Exactly one profile should be marked active.
    ASSERT_EQ(active_count, 1);

    // num_fields == nullptr must be rejected.
    DISPLAY_AMDSMI_API("amdsmi_get_ampp_fields", "gpu=" + std::to_string(dv_ind) + ", NULL",
                       VERB(STANDARD));
    ret = amdsmi_get_ampp_fields(handle, "profile_0", nullptr, nullptr);
    DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
    ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

    // A profile_name that does not parse as "profile_<N>" (e.g. a sysfs
    // control directory, or a path-traversal attempt) must be rejected
    // with INVAL before any filesystem lookup is attempted.
    for (const char* bad_name : {"config", "limits", "../../etc", "profile_"}) {
      uint32_t nf = 0;
      DISPLAY_AMDSMI_API("amdsmi_get_ampp_fields",
                         "gpu=" + std::to_string(dv_ind) + ", profile_name=" + bad_name,
                         VERB(STANDARD));
      ret = amdsmi_get_ampp_fields(handle, bad_name, &nf, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);
    }

    if (!configured_profile_name.empty()) {
      uint32_t num_fields = 0;
      DISPLAY_AMDSMI_API(
          "amdsmi_get_ampp_fields",
          "gpu=" + std::to_string(dv_ind) + ", profile=" + configured_profile_name + " (size)",
          VERB(STANDARD));
      ret = amdsmi_get_ampp_fields(handle, configured_profile_name.c_str(), &num_fields, nullptr);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      ASSERT_GT(num_fields, 0u);

      // Same under-sized-buffer contract as amdsmi_get_ampp_profiles above;
      // fixed 1-element buffer for the same nullptr-.data() reason.
      amdsmi_ampp_field_t undersized_fields[1];
      uint32_t requested_fields = num_fields - 1;
      DISPLAY_AMDSMI_API("amdsmi_get_ampp_fields",
                         "gpu=" + std::to_string(dv_ind) + ", profile=" + configured_profile_name +
                             " (undersized buffer)",
                         VERB(STANDARD));
      ret = amdsmi_get_ampp_fields(handle, configured_profile_name.c_str(), &requested_fields,
                                   undersized_fields);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret,
                            AMDSMI_STATUS_OUT_OF_RESOURCES);
      ASSERT_EQ(ret, AMDSMI_STATUS_OUT_OF_RESOURCES);
      ASSERT_EQ(requested_fields, num_fields);

      std::vector<amdsmi_ampp_field_t> fields(num_fields);
      DISPLAY_AMDSMI_API(
          "amdsmi_get_ampp_fields",
          "gpu=" + std::to_string(dv_ind) + ", profile=" + configured_profile_name + " (fill)",
          VERB(STANDARD));
      ret = amdsmi_get_ampp_fields(handle, configured_profile_name.c_str(), &num_fields,
                                   fields.data());
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
      CHK_ERR_ASRT(ret)
      for (const auto& f : fields) {
        IF_VERB(STANDARD) {
          std::cout << "\t    field=" << f.name << " value=" << f.value << " unit=" << f.unit
                    << std::endl;
        }
        ASSERT_GT(strnlen(f.name, sizeof(f.name)), 0u);
      }
    }

    if (!unconfigured_writable_profile_name.empty()) {
      uint32_t num_fields = 5;
      amdsmi_ampp_field_t fields[5];
      DISPLAY_AMDSMI_API("amdsmi_get_ampp_fields",
                         "gpu=" + std::to_string(dv_ind) +
                             ", profile=" + unconfigured_writable_profile_name + " (unconfigured)",
                         VERB(STANDARD));
      ret = amdsmi_get_ampp_fields(handle, unconfigured_writable_profile_name.c_str(), &num_fields,
                                   fields);
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_NO_DATA);
      ASSERT_EQ(ret, AMDSMI_STATUS_NO_DATA);
      ASSERT_EQ(num_fields, 0u);
    }

    // --- Write-path exercised only via AMDSMI_DRY_RUN, matching the
    // precedent in memory_read_write_test.cc / test_memory.py for
    // privileged sysfs writes that would otherwise require root. ---
    {
      ScopedEnvVar dry_run("AMDSMI_DRY_RUN", "1");

      if (!profiles.empty()) {
        DISPLAY_AMDSMI_API(
            "amdsmi_activate_ampp_profile",
            "gpu=" + std::to_string(dv_ind) + ", ACTIVATE " + profiles[0].name + " (DRY_RUN)",
            VERB(STANDARD));
        ret = amdsmi_activate_ampp_profile(handle, profiles[0].name);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        CHK_ERR_ASRT(ret)
      }

      // ACTIVATE / CONFIGURE of a name that doesn't resolve to a published
      // profile_N must fail with INVAL, not silently succeed.
      DISPLAY_AMDSMI_API("amdsmi_activate_ampp_profile",
                         "gpu=" + std::to_string(dv_ind) + ", ACTIVATE ../../etc (DRY_RUN)",
                         VERB(STANDARD));
      ret = amdsmi_activate_ampp_profile(handle, "../../etc");
      DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
      ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

      if (!unconfigured_writable_profile_name.empty()) {
        // CONFIGURE with fields == nullptr but num_fields > 0 must be
        // rejected with INVAL rather than dereferencing a null pointer.
        DISPLAY_AMDSMI_API("amdsmi_configure_ampp_profile",
                           "gpu=" + std::to_string(dv_ind) + ", CONFIGURE " +
                               unconfigured_writable_profile_name + " fields=NULL,num=1 (DRY_RUN)",
                           VERB(STANDARD));
        ret = amdsmi_configure_ampp_profile(handle, unconfigured_writable_profile_name.c_str(),
                                            nullptr, 1);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
        ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

        // A commit with nothing ever staged (fields == nullptr, num_fields
        // == 0) must be rejected with INVAL: the driver's config/commit
        // rejects -EINVAL in this case, and AMDSMI defensively requires at
        // least one staged field before issuing any sysfs writes at all.
        DISPLAY_AMDSMI_API("amdsmi_configure_ampp_profile",
                           "gpu=" + std::to_string(dv_ind) + ", CONFIGURE " +
                               unconfigured_writable_profile_name + " (empty commit, DRY_RUN)",
                           VERB(STANDARD));
        ret = amdsmi_configure_ampp_profile(handle, unconfigured_writable_profile_name.c_str(),
                                            nullptr, 0);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_INVAL);
        ASSERT_EQ(ret, AMDSMI_STATUS_INVAL);

        // A CONFIGURE with at least one staged field must succeed.
        amdsmi_ampp_field_t stage_field;
        memset(&stage_field, 0, sizeof(stage_field));
        snprintf(stage_field.name, sizeof(stage_field.name), "%s", "PPT0_Limit");
        stage_field.value = 300;
        DISPLAY_AMDSMI_API("amdsmi_configure_ampp_profile",
                           "gpu=" + std::to_string(dv_ind) + ", CONFIGURE " +
                               unconfigured_writable_profile_name + " PPT0_Limit=300 (DRY_RUN)",
                           VERB(STANDARD));
        ret = amdsmi_configure_ampp_profile(handle, unconfigured_writable_profile_name.c_str(),
                                            &stage_field, 1);
        DISPLAY_AMDSMI_STATUS(VERB(STANDARD), __FILE__, __LINE__, ret, AMDSMI_STATUS_SUCCESS);
        CHK_ERR_ASRT(ret)
      }
    }
  }
}
