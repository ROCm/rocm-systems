/* SPDX-License-Identifier: MIT */
/*
 * Copyright (c) OdinLink-Five contributors
 * https://github.com/Geramy/OdinLink-Five
 *
 * OdinLink Thunderbolt 5 - Peer Discovery
 */
#include "odl_tb5_priv.h"
#include <odl_tb5/odl_tb5_ioctl.h>
#include <errno.h>
#include <time.h>
#include <sys/ioctl.h>

int odl_tb5_get_peer(odl_tb5_t handle, struct odl_tb5_peer_info *info)
{
	if (!handle || !info)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_GET_PEER, info) < 0)
		return -errno;

	return 0;
}

int odl_tb5_get_link_info(odl_tb5_t handle, struct odl_tb5_link_info *info)
{
	if (!handle || !info)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_GET_LINK_INFO, info) < 0)
		return -errno;

	return 0;
}

int odl_tb5_wait_peer(odl_tb5_t handle, int timeout_ms)
{
	struct odl_tb5_peer_info info;
	struct timespec ts_start, ts_now;
	int ret;

	if (!handle)
		return -EINVAL;

	clock_gettime(CLOCK_MONOTONIC, &ts_start);

	for (;;) {
		ret = odl_tb5_get_peer(handle, &info);
		if (ret < 0)
			return ret;

		if (info.state == ODL_TB5_STATE_READY)
			return 0;

		if (info.state == ODL_TB5_STATE_ERROR)
			return -EIO;

		if (timeout_ms > 0) {
			clock_gettime(CLOCK_MONOTONIC, &ts_now);
			long elapsed_ms = (ts_now.tv_sec - ts_start.tv_sec) * 1000 +
					  (ts_now.tv_nsec - ts_start.tv_nsec) / 1000000;
			if (elapsed_ms >= timeout_ms)
				return -ETIMEDOUT;
		}

		struct timespec sleep_ts = { .tv_sec = 0, .tv_nsec = 10000000 };
		nanosleep(&sleep_ts, NULL);
	}
}
