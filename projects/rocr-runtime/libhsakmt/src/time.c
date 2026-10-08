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

#include "libhsakmt.h"
#include "hsakmt/linux/kfd_ioctl.h"


#define CLOCKFD 3
#define FD_TO_CLOCKID(fd) ((~(clockid_t)(fd) << 3) | CLOCKFD)

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif

#include <sys/time.h>
#include <fcntl.h>
#include <unistd.h>
#include <stdlib.h>

static HSAKMT_STATUS get_clockdev_ns(const char* dev, HSAuint64* clk_ns)
{
  static int fd = 0;
  if (fd == 0)
  {
    fd = open(dev, O_RDONLY);
    if (fd < 0) return HSAKMT_STATUS_ERROR;
  }

  clockid_t clkid = FD_TO_CLOCKID(fd);
  struct timespec ts;
  if (clock_gettime(clkid, &ts)) return HSAKMT_STATUS_ERROR;

  *clk_ns = (HSAuint64)ts.tv_sec * 1000000000ULL + (HSAuint64)ts.tv_nsec;
  return HSAKMT_STATUS_SUCCESS;
}

static HSAKMT_STATUS check_clockdev(HsaClockCounters *Counters)
{
    char* clockdev = getenv("HSA_ROCM_TIMESYNC_CLOCKDEV");
    if (clockdev == NULL)
        return HSAKMT_STATUS_SUCCESS;

    // replace system clock counter with a read of the rocm_timesync clockdevice
    return get_clockdev_ns(clockdev, &(Counters->SystemClockCounter));
}

HSAKMT_STATUS HSAKMTAPI hsaKmtGetClockCounters(HSAuint32 NodeId,
					       HsaClockCounters *Counters)
{
	HSAKMT_STATUS result;
	uint32_t gpu_id;
	struct kfd_ioctl_get_clock_counters_args args = {0};
	int err;

	CHECK_KFD_OPEN();

	result = hsakmt_validate_nodeid(NodeId, &gpu_id);
	if (result != HSAKMT_STATUS_SUCCESS)
		return result;

	args.gpu_id = gpu_id;

	err = hsakmt_ioctl(hsakmt_primary_kfd_ctx.fd, AMDKFD_IOC_GET_CLOCK_COUNTERS, &args);
	if (err < 0) {
		result = HSAKMT_STATUS_ERROR;
	} else {
		/* At this point the result is already HSAKMT_STATUS_SUCCESS */
		Counters->GPUClockCounter = args.gpu_clock_counter;
		Counters->CPUClockCounter = args.cpu_clock_counter;
		Counters->SystemClockCounter = args.system_clock_counter;
		Counters->SystemClockFrequencyHz = args.system_clock_freq;
	}

    if (result == HSAKMT_STATUS_ERROR)
        return result;

    return check_clockdev(Counters);
}
