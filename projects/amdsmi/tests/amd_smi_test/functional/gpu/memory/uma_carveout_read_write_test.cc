// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT

#include <cstring>
#include <string>

#include "api_test_framework.h"

using amdsmi::test::kInvalidHandle;
using amdsmi::test::kVerbose;

// amdsmi_get_gpu_uma_carveout_info / amdsmi_set_gpu_uma_carveout (option index).
TEST_F(GpuFunctionalReadWrite, UmaCarveout_SetVerifyRestore) {
  AMDSMI_SKIP_UNLESS_MUTATION_ALLOWED();
  if (gpus().empty()) GTEST_SKIP() << "No GPU processors";
  amdsmi::test::StatusCollector col("amdsmi_set_gpu_uma_carveout");
  for (size_t i = 0; i < gpus().size(); ++i) {
    amdsmi_uma_carveout_info_t info;
    memset(&info, 0, sizeof(info));
    if (amdsmi_get_gpu_uma_carveout_info(gpus()[i], &info) != AMDSMI_STATUS_SUCCESS) continue;
    if (info.num_options < 2 || info.current_index >= info.num_options) continue;

    const uint32_t initial = info.current_index;
    // The setter rejects an option carrying no description, so both ends need one.
    if (info.options[initial].description[0] == '\0') continue;

    uint32_t target = info.num_options;
    for (uint32_t o = 0; o < info.num_options; ++o) {
      if (o != initial && info.options[o].description[0] != '\0') {
        target = o;
        break;
      }
    }
    if (target == info.num_options) continue;
    const std::string label = "gpu=" + std::to_string(i);

    DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout", label + " set=" + std::to_string(target),
                       kVerbose);
    amdsmi_status_t err = amdsmi_set_gpu_uma_carveout(gpus()[i], target);
    DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, err, AMDSMI_STATUS_SUCCESS,
                          AMDSMI_STATUS_NOT_SUPPORTED, AMDSMI_STATUS_NOT_YET_IMPLEMENTED,
                          AMDSMI_STATUS_NO_PERM);
    col.Record(label, err,
               ::amdsmi::test::AmdsmiStatusIsExpected(
                   err, AMDSMI_STATUS_SUCCESS, AMDSMI_STATUS_NOT_SUPPORTED,
                   AMDSMI_STATUS_NOT_YET_IMPLEMENTED, AMDSMI_STATUS_NO_PERM));

    if (err == AMDSMI_STATUS_SUCCESS) {
      // The sysfs node reports the pending index right away; the reboot the API
      // documents is only needed for the allocation itself to change.
      amdsmi_uma_carveout_info_t readback;
      memset(&readback, 0, sizeof(readback));
      if (amdsmi_get_gpu_uma_carveout_info(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current_index, target) << label << " set did not take effect";
      }

      DISPLAY_AMDSMI_API("amdsmi_set_gpu_uma_carveout",
                         label + " restore=" + std::to_string(initial), kVerbose);
      amdsmi_status_t rerr = amdsmi_set_gpu_uma_carveout(gpus()[i], initial);
      DISPLAY_AMDSMI_STATUS(kVerbose, __FILE__, __LINE__, rerr, AMDSMI_STATUS_SUCCESS);
      EXPECT_EQ(rerr, AMDSMI_STATUS_SUCCESS) << label << " failed to restore uma carveout";
      if (rerr == AMDSMI_STATUS_SUCCESS &&
          amdsmi_get_gpu_uma_carveout_info(gpus()[i], &readback) == AMDSMI_STATUS_SUCCESS) {
        EXPECT_EQ(readback.current_index, initial) << label << " restore did not take effect";
      }
    }
  }
  col.ExpectNoFailures();
}
