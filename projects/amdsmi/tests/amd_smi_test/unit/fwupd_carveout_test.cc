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

#include "amd_smi/amdsmi.h"

TEST(FwupdCarveout, GetReportsNotSupported) {
  amdsmi_uma_carveout_info_t info{};
  EXPECT_EQ(amd::smi::fwupd_get_carveout_info(&info), AMDSMI_STATUS_NOT_SUPPORTED);
}

TEST(FwupdCarveout, SetReportsNotSupported) {
  EXPECT_EQ(amd::smi::fwupd_set_carveout(0), AMDSMI_STATUS_NOT_SUPPORTED);
}
