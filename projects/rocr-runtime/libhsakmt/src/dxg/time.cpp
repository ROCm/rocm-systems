/*
 * Copyright © 2014 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use, copy,
 * modify, merge, publish, distribute, sublicense, and/or sell copies
 * of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice (including
 * the next paragraph) shall be included in all copies or substantial
 * portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS IN THE SOFTWARE.
 */
#include <iostream>
#include <ctime>
#include <cstring>
#include <cassert>
#include "impl/wddm/device.h"
#include "util/os.h"

static HSAKMT_STATUS read_clock_counters(HSAuint32 NodeId, HsaClockCounters *Counters) {
  HSAKMT_STATUS result = HSAKMT_STATUS_SUCCESS;

  CHECK_DXG_OPEN();

  std::memset(Counters, 0, sizeof(*Counters));

  wsl::thunk::WDDMDevice *device_ = get_wddmdev(NodeId);
  assert(device_);
  device_->GetClockCounters(&Counters->GPUClockCounter, &Counters->CPUClockCounter);

  Counters->SystemClockCounter = rocr::os::TimeNanos();
  Counters->SystemClockFrequencyHz = 1000000000;

  return result;
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCounters(HSAuint32 NodeId,
                                               HsaClockCounters *Counters) {
  return read_clock_counters(NodeId, Counters);
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCountersPrecise(HSAuint32 NodeId,
                                                     HsaClockCounters *Counters) {
  HsaClockCounters best{};
  uint64_t best_elapsed = UINT64_MAX;

  for (unsigned int i = 0; i < 4; i++) {
    HsaClockCounters sample;
    uint64_t before = rocr::os::TimeNanos();
    HSAKMT_STATUS result = read_clock_counters(NodeId, &sample);
    uint64_t after = rocr::os::TimeNanos();
    if (result != HSAKMT_STATUS_SUCCESS)
      return result;

    uint64_t elapsed = after - before;
    if (elapsed < best_elapsed) {
      best_elapsed = elapsed;
      best = sample;
    }
  }

  *Counters = best;
  return HSAKMT_STATUS_SUCCESS;
}
