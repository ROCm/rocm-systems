/*
 * Copyright (C) Advanced Micro Devices. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy of
 * this software and associated documentation files (the "Software"), to deal in
 * the Software without restriction, including without limitation the rights to
 * use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
 * the Software, and to permit persons to whom the Software is furnished to do so,
 * subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
 * FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
 * COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
 * IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
 * CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 */

// Hardware-free unit test for the fwupd UMA carveout adapter. On a host without
// the fwupd carveout BIOS setting -- any CI machine: no APU firmware exposes it,
// and typically no fwupd daemon or D-Bus system bus is present -- both entry
// points must report AMDSMI_STATUS_NOT_SUPPORTED so the amdgpu sysfs node stays
// the sole UMA carveout interface. Needs no GPU, no root, and no fwupd daemon.

#include "fwupd_carveout.h"

#include <gtest/gtest.h>

#include <cstdlib>

#include "amd_smi/amdsmi.h"

// On a typical CI host no APU firmware exposes the carveout setting (and usually
// there is no fwupd daemon or D-Bus system bus), so the adapter reports
// NOT_SUPPORTED. On real affected hardware it can succeed; accept that too and
// sanity-check the result rather than hard-failing.
TEST(FwupdCarveout, GetReportsNotSupportedOrSaneInfo) {
  amdsmi_uma_carveout_info_t info{};
  const amdsmi_status_t ret = amd::smi::fwupd_get_carveout_info(&info);
  if (ret == AMDSMI_STATUS_SUCCESS) {
    EXPECT_GT(info.num_options, 0u);
    EXPECT_LE(info.num_options, static_cast<uint32_t>(AMDSMI_MAX_CARVEOUT_OPTIONS));
    EXPECT_LE(info.current_index, info.num_options);
  } else {
    EXPECT_EQ(ret, AMDSMI_STATUS_NOT_SUPPORTED);
  }
}

// Force dry-run so the test can never change a real BIOS setting on affected
// hardware. CI hosts still get NOT_SUPPORTED (no carveout); affected hardware
// resolves the setting and returns SUCCESS (or NO_PERM) without writing.
TEST(FwupdCarveout, SetReportsNotSupportedWithoutMutating) {
  setenv("AMDSMI_DRY_RUN", "1", 1);
  const amdsmi_status_t ret = amd::smi::fwupd_set_carveout(0);
  unsetenv("AMDSMI_DRY_RUN");
  EXPECT_TRUE(ret == AMDSMI_STATUS_NOT_SUPPORTED || ret == AMDSMI_STATUS_SUCCESS ||
              ret == AMDSMI_STATUS_NO_PERM)
      << "unexpected status: " << ret;
}
