// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

// Coverage for the primary/non-primary gating behavior of the two AMPP call
// sites that consult amd::smi::is_node_scoped_field() (rocm_smi_ampp.cc):
// rsmi_dev_ampp_fields_get() (read path, via amdsmi_get_ampp_fields) and
// rsmi_dev_ampp_profile_configure() (write path, via
// amdsmi_configure_ampp_profile()).
//
// AMPP's app_modes/ root is anchored to the real device's own sysfs path
// (Device::get_ampp_root_path()), unlike NPM's board_path (an arbitrary
// caller-supplied directory amdsmi_npm_limit_test.cc fabricates via a temp
// dir) -- there is no equivalent fabrication path here, and no test-injection
// hook for is_primary_gpu() analogous to amdsmi_test_register_node_handle()
// (deliberately not added; see the review this test responds to). These
// tests therefore run against whatever real/mocked device state is present at
// run time and GTEST_SKIP_ with a specific reason when a precondition (a
// non-primary device, a primary device with a node_-scoped field, a writable
// profile, root) is not satisfiable -- the same "best effort against real
// hardware" convention already used by GetNpmInfoCurrentNodePowerRoundTrip in
// amdsmi_npm_limit_test.cc.
//
// The write-path rejection test (AmppNodeScopedFieldWriteRejectedOnNonPrimaryGpu)
// is the one exception that does not depend on the device actually having a
// field literally named "node_power_limit": field_is_stageable() in
// rocm_smi_ampp.cc rejects any "node_"-prefixed name on a non-primary GPU
// before ever checking whether that name is a real, recognized field for the
// profile, so the assertion holds unconditionally on any non-primary device
// with a writable slot.
#include <gtest/gtest.h>

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

#include "amd_smi/amdsmi.h"
#include "rocm_smi/rocm_smi_ampp.h"
#include "rocm_smi/rocm_smi_utils.h"

namespace {

// Small RAII wrapper identical in spirit to the one in amdsmi_npm_limit_test.cc;
// kept local to this file to avoid coupling across unit-test translation units.
class ScopedAmdSmiInit {
 public:
  ScopedAmdSmiInit() { status_ = amdsmi_init(AMDSMI_INIT_AMD_GPUS); }
  ~ScopedAmdSmiInit() {
    if (status_ == AMDSMI_STATUS_SUCCESS) {
      amdsmi_shut_down();
    }
  }
  amdsmi_status_t status() const { return status_; }

 private:
  amdsmi_status_t status_;
};

#define REQUIRE_AMDSMI_INIT(init)                             \
  do {                                                        \
    if ((init).status() == AMDSMI_STATUS_DRIVER_NOT_LOADED) { \
      GTEST_SKIP() << "amdgpu driver not loaded";             \
    }                                                         \
    ASSERT_EQ((init).status(), AMDSMI_STATUS_SUCCESS);        \
  } while (0)

// amdsmi_get_gpu_asic_info()'s oam_id is populated from the same
// rsmi_dev_xgmi_physical_id_get() primitive is_primary_gpu() uses (see
// amd_smi.cc:3319-3322); it is left at whatever *out_oam_id was initialized
// to when retrieval fails/is unsupported, so the sentinel must be pre-set to
// a non-zero value for "unknown" to correctly read as "not primary" --
// mirroring is_primary_gpu()'s own fail-closed semantics (any failure is
// treated as non-primary).
bool GetIsPrimary(amdsmi_processor_handle handle, bool* out_is_primary) {
  amdsmi_asic_info_t asic_info{};
  asic_info.oam_id = std::numeric_limits<uint32_t>::max();
  if (amdsmi_get_gpu_asic_info(handle, &asic_info) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  *out_is_primary = (asic_info.oam_id == 0);
  return true;
}

bool GetProcessorHandles(std::vector<amdsmi_processor_handle>* out_handles) {
  uint32_t socket_count = 0;
  if (amdsmi_get_socket_handles(&socket_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  if (socket_count > 0 &&
      amdsmi_get_socket_handles(&socket_count, sockets.data()) != AMDSMI_STATUS_SUCCESS) {
    return false;
  }
  for (auto socket : sockets) {
    uint32_t proc_count = 0;
    if (amdsmi_get_processor_handles(socket, &proc_count, nullptr) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    std::vector<amdsmi_processor_handle> procs(proc_count);
    if (proc_count > 0 &&
        amdsmi_get_processor_handles(socket, &proc_count, procs.data()) != AMDSMI_STATUS_SUCCESS) {
      continue;
    }
    out_handles->insert(out_handles->end(), procs.begin(), procs.end());
  }
  return true;
}

bool GetProfiles(amdsmi_processor_handle handle, std::vector<amdsmi_ampp_profile_t>* out_profiles) {
  char version[AMDSMI_MAX_STRING_LENGTH] = {};
  uint32_t num_profiles = 0;
  amdsmi_status_t ret = amdsmi_get_ampp_profiles(handle, version, &num_profiles, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS || num_profiles == 0) {
    return false;
  }
  out_profiles->resize(num_profiles);
  return amdsmi_get_ampp_profiles(handle, version, &num_profiles, out_profiles->data()) ==
         AMDSMI_STATUS_SUCCESS;
}

bool GetFields(amdsmi_processor_handle handle, const char* profile_name,
               std::vector<amdsmi_ampp_field_t>* out_fields) {
  uint32_t num_fields = 0;
  amdsmi_status_t ret = amdsmi_get_ampp_fields(handle, profile_name, &num_fields, nullptr);
  if (ret != AMDSMI_STATUS_SUCCESS || num_fields == 0) {
    return false;
  }
  out_fields->resize(num_fields);
  return amdsmi_get_ampp_fields(handle, profile_name, &num_fields, out_fields->data()) ==
         AMDSMI_STATUS_SUCCESS;
}

std::string FieldName(const amdsmi_ampp_field_t& f) {
  return std::string(f.name, strnlen(f.name, sizeof(f.name)));
}

}  // namespace

// ---------------------------------------------------------------------
// Read path: rsmi_dev_ampp_fields_get() via amdsmi_get_ampp_fields()
// ---------------------------------------------------------------------

TEST(GpuUnit, AmppNodeScopedFieldsHiddenOnNonPrimaryGpu) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  std::vector<amdsmi_processor_handle> handles;
  ASSERT_TRUE(GetProcessorHandles(&handles));
  if (handles.empty()) {
    GTEST_SKIP_("No processors discovered on this system");
  }

  bool exercised_non_primary_device = false;

  for (auto handle : handles) {
    bool is_primary = false;
    if (!GetIsPrimary(handle, &is_primary) || is_primary) {
      continue;
    }

    std::vector<amdsmi_ampp_profile_t> profiles;
    if (!GetProfiles(handle, &profiles)) {
      continue;  // AMPP not supported, or no profiles published
    }

    for (const auto& p : profiles) {
      std::vector<amdsmi_ampp_field_t> fields;
      if (!GetFields(handle, p.name, &fields)) {
        continue;  // empty/unconfigured/not-writable -- nothing to check here
      }
      exercised_non_primary_device = true;
      for (const auto& f : fields) {
        const std::string name = FieldName(f);
        EXPECT_FALSE(amd::smi::is_node_scoped_field(name))
            << "node-scoped field '" << name << "' (profile " << p.name
            << ") must not be visible via amdsmi_get_ampp_fields() on a non-primary GPU";
      }
    }
  }

  if (!exercised_non_primary_device) {
    GTEST_SKIP_(
        "No non-primary GPU (oam_id != 0) with any readable AMPP profile fields was found on "
        "this system");
  }
}

TEST(GpuUnit, AmppNodeScopedFieldsVisibleOnPrimaryGpuWhenPresent) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  std::vector<amdsmi_processor_handle> handles;
  ASSERT_TRUE(GetProcessorHandles(&handles));
  if (handles.empty()) {
    GTEST_SKIP_("No processors discovered on this system");
  }

  bool found_node_scoped_field_on_primary = false;

  for (auto handle : handles) {
    bool is_primary = false;
    if (!GetIsPrimary(handle, &is_primary) || !is_primary) {
      continue;
    }

    std::vector<amdsmi_ampp_profile_t> profiles;
    if (!GetProfiles(handle, &profiles)) {
      continue;
    }

    for (const auto& p : profiles) {
      std::vector<amdsmi_ampp_field_t> fields;
      if (!GetFields(handle, p.name, &fields)) {
        continue;
      }
      for (const auto& f : fields) {
        if (amd::smi::is_node_scoped_field(FieldName(f))) {
          found_node_scoped_field_on_primary = true;
        }
      }
    }
  }

  // Primary-GPU behavior is unchanged by this fix (no filtering is applied),
  // so this is a sanity check, not a regression guard on its own -- it only
  // confirms a node_-scoped field published by the driver is not
  // accidentally dropped on the GPU where it is supposed to remain visible.
  if (!found_node_scoped_field_on_primary) {
    GTEST_SKIP_(
        "No primary GPU (oam_id == 0) with a node_-scoped AMPP field was found on this system");
  }
  SUCCEED();
}

// ---------------------------------------------------------------------
// Write path: rsmi_dev_ampp_profile_configure() via
// amdsmi_configure_ampp_profile()
// ---------------------------------------------------------------------

TEST(GpuUnit, AmppNodeScopedFieldWriteRejectedOnNonPrimaryGpu) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  std::vector<amdsmi_processor_handle> handles;
  ASSERT_TRUE(GetProcessorHandles(&handles));
  if (handles.empty()) {
    GTEST_SKIP_("No processors discovered on this system");
  }

  bool exercised = false;

  for (auto handle : handles) {
    bool is_primary = false;
    if (!GetIsPrimary(handle, &is_primary) || is_primary) {
      continue;
    }

    std::vector<amdsmi_ampp_profile_t> profiles;
    if (!GetProfiles(handle, &profiles)) {
      continue;
    }

    for (const auto& p : profiles) {
      if (!p.is_writable) {
        continue;
      }
      exercised = true;

      // field_is_stageable() rejects any "node_"-prefixed name on a
      // non-primary GPU unconditionally, before ever checking whether the
      // name is one of this profile's real config/ fields -- so this holds
      // regardless of whether "node_power_limit" happens to be a real field
      // here.
      amdsmi_ampp_field_t field{};
      std::snprintf(field.name, sizeof(field.name), "%s", "node_power_limit");
      field.value = 0;

      amdsmi_status_t ret = amdsmi_configure_ampp_profile(handle, p.name, &field, 1);
      EXPECT_EQ(ret, AMDSMI_STATUS_INVAL)
          << "writing a node_-prefixed field to profile " << p.name
          << " on a non-primary GPU must be rejected with AMDSMI_STATUS_INVAL";
      break;  // one writable profile per device is enough to exercise this
    }
  }

  if (!exercised) {
    GTEST_SKIP_(
        "No non-primary GPU (oam_id != 0) with a writable AMPP profile was found on this "
        "system");
  }
}

TEST(GpuUnit, AmppNodeScopedFieldWriteAllowedOnPrimaryGpuWhenPresent) {
  ScopedAmdSmiInit init;
  REQUIRE_AMDSMI_INIT(init);

  if (!amd::smi::is_sudo_user()) {
    GTEST_SKIP_("Invalid permission - Must run as super user");
  }

  std::vector<amdsmi_processor_handle> handles;
  ASSERT_TRUE(GetProcessorHandles(&handles));
  if (handles.empty()) {
    GTEST_SKIP_("No processors discovered on this system");
  }

  bool exercised = false;

  for (auto handle : handles) {
    bool is_primary = false;
    if (!GetIsPrimary(handle, &is_primary) || !is_primary) {
      continue;
    }

    std::vector<amdsmi_ampp_profile_t> profiles;
    if (!GetProfiles(handle, &profiles)) {
      continue;
    }

    for (const auto& p : profiles) {
      if (!p.is_writable) {
        continue;
      }
      std::vector<amdsmi_ampp_field_t> fields;
      if (!GetFields(handle, p.name, &fields)) {
        continue;
      }
      for (const auto& f : fields) {
        if (!amd::smi::is_node_scoped_field(FieldName(f))) {
          continue;
        }
        // Round-trip the field's own current value back onto itself -- a
        // real, recognized node_-scoped field on a writable, primary-GPU
        // profile must not be rejected with INVAL merely for being
        // node_-prefixed (it still may legitimately fail for other reasons,
        // e.g. PMFW range clamping, which is out of scope here).
        amdsmi_ampp_field_t write_field{};
        std::snprintf(write_field.name, sizeof(write_field.name), "%s", f.name);
        write_field.value = f.value;

        amdsmi_status_t ret = amdsmi_configure_ampp_profile(handle, p.name, &write_field, 1);
        EXPECT_NE(ret, AMDSMI_STATUS_INVAL)
            << "writing node_-scoped field " << FieldName(f) << " back to its own current "
            << "value on profile " << p.name << " of a primary GPU must not be rejected as "
            << "INVAL";
        exercised = true;
        break;  // one real config/commit write is enough to exercise this
      }
      if (exercised) break;
    }
    if (exercised) break;
  }

  if (!exercised) {
    GTEST_SKIP_(
        "No primary GPU (oam_id == 0) with a writable AMPP profile containing a "
        "node_-scoped field was found on this system");
  }
}
