/*
 * OdinLink Thunderbolt 5 - Data Transfer
 */
#include "odl_tb5_priv.h"
#include <odl_tb5/odl_tb5_ioctl.h>
#include <errno.h>
#include <sys/ioctl.h>

int odl_tb5_send(odl_tb5_t handle, size_t offset, size_t len)
{
	struct odl_tb5_xfer_request req = {
		.offset = offset,
		.len = len,
		.flags = 0,
	};

	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_SEND, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_send_ctrl(odl_tb5_t handle, size_t offset, size_t len)
{
	struct odl_tb5_xfer_request req = {
		.offset = offset,
		.len = len,
		.flags = ODL_TB5_XFER_FLAG_CTRL,
	};

	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_SEND, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_recv(odl_tb5_t handle, size_t offset, size_t len)
{
	struct odl_tb5_xfer_request req = {
		.offset = offset,
		.len = len,
		.flags = 0,
	};

	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_RECV, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_swap_tx(odl_tb5_t handle)
{
	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_SWAP_TX_BUF) < 0)
		return -errno;

	handle->tx_back = 1 - handle->tx_back;
	return 0;
}

int odl_tb5_swap_rx(odl_tb5_t handle)
{
	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_SWAP_RX_BUF) < 0)
		return -errno;

	handle->rx_back = 1 - handle->rx_back;
	return 0;
}

int odl_tb5_send_dmabuf(odl_tb5_t handle, int dmabuf_fd,
			off_t offset, size_t len)
{
	struct odl_tb5_ring_request req = {
		.dmabuf_fd = dmabuf_fd,
		.offset = offset,
		.len = len,
	};

	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_SEND_DMABUF, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_recv_dmabuf(odl_tb5_t handle, int dmabuf_fd,
			off_t offset, size_t len)
{
	struct odl_tb5_ring_request req = {
		.dmabuf_fd = dmabuf_fd,
		.offset = offset,
		.len = len,
	};

	if (!handle)
		return -EINVAL;

	if (ioctl(handle->fd, ODL_TB5_IOCTL_RECV_DMABUF, &req) < 0)
		return -errno;

	return 0;
}
