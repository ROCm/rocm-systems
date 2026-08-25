/*
 * OdinLink Thunderbolt 5 - Completion Polling and Waiting
 */
#include "odl_tb5_priv.h"
#include <odl_tb5/odl_tb5_ioctl.h>
#include <errno.h>
#include <sys/ioctl.h>

int odl_tb5_poll(odl_tb5_t handle, struct odl_tb5_completion *comp)
{
	if (!handle || !comp)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_POLL_COMPLETION, comp) < 0)
		return -errno;

	return 0;
}

int odl_tb5_wait_tx(odl_tb5_t handle, struct odl_tb5_completion *comp)
{
	if (!handle || !comp)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_WAIT_TX, comp) < 0)
		return -errno;

	return 0;
}

int odl_tb5_wait_rx(odl_tb5_t handle, struct odl_tb5_completion *comp)
{
	if (!handle || !comp)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_WAIT_RX, comp) < 0)
		return -errno;

	return 0;
}
