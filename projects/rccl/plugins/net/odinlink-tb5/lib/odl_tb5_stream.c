// SPDX-License-Identifier: MIT
/*
 * OdinLink Thunderbolt 5 - Stream-Based Multiplexed I/O
 */
#include <odl_tb5/odl_tb5.h>
#include <odl_tb5/odl_tb5_ioctl.h>
#include <errno.h>
#include <string.h>
#include <sys/ioctl.h>

int odl_tb5_stream_open(odl_tb5_t handle, uint8_t filter_id,
			uint8_t *stream_id_out)
{
	struct odl_tb5_stream_req req = { .stream_id = filter_id };

	if (!handle || !stream_id_out)
		return -EINVAL;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_OPEN, &req) < 0)
		return -errno;

	*stream_id_out = req.stream_id;
	return 0;
}

int odl_tb5_stream_close(odl_tb5_t handle, uint8_t stream_id)
{
	struct odl_tb5_stream_req req = { .stream_id = stream_id };

	if (!handle)
		return -EINVAL;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_CLOSE, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_stream_send(odl_tb5_t handle, uint8_t stream_id,
			uint8_t dst_id, const void *data, uint32_t len)
{
	struct odl_tb5_stream_xfer req;

	if (!handle || !data || len == 0)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.dst_id = dst_id;
	req.data = (uint64_t)(uintptr_t)data;
	req.len = len;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_SEND, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_stream_recv(odl_tb5_t handle, uint8_t stream_id,
			void *buf, uint32_t buf_len,
			uint8_t *src_id, uint32_t *actual_len)
{
	struct odl_tb5_stream_xfer req;

	if (!handle || !buf || buf_len == 0)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.data = (uint64_t)(uintptr_t)buf;
	req.len = buf_len;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_RECV, &req) < 0)
		return -errno;

	if (src_id)
		*src_id = req.src_id;
	if (actual_len)
		*actual_len = req.actual_len;

	return 0;
}

int odl_tb5_stream_wait_tx(odl_tb5_t handle, uint8_t stream_id,
			   uint32_t timeout_ms)
{
	struct odl_tb5_stream_wait req;

	if (!handle)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.timeout_ms = timeout_ms;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_WAIT_TX, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_stream_wait_rx(odl_tb5_t handle, uint8_t stream_id,
			   uint32_t timeout_ms)
{
	struct odl_tb5_stream_wait req;

	if (!handle)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.timeout_ms = timeout_ms;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_WAIT_RX, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_stream_send_dmabuf(odl_tb5_t handle, uint8_t stream_id,
			       uint8_t dst_id, int dmabuf_fd,
			       uint64_t offset, uint64_t len)
{
	struct odl_tb5_stream_dmabuf req;

	if (!handle)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.dst_id = dst_id;
	req.dmabuf_fd = dmabuf_fd;
	req.offset = offset;
	req.len = len;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_SEND_DMABUF, &req) < 0)
		return -errno;

	return 0;
}

int odl_tb5_stream_recv_dmabuf(odl_tb5_t handle, uint8_t stream_id,
			       int dmabuf_fd, uint64_t offset, uint64_t len)
{
	struct odl_tb5_stream_dmabuf req;

	if (!handle)
		return -EINVAL;

	memset(&req, 0, sizeof(req));
	req.stream_id = stream_id;
	req.dmabuf_fd = dmabuf_fd;
	req.offset = offset;
	req.len = len;

	if (ioctl(odl_tb5_get_fd(handle), ODL_TB5_IOCTL_STREAM_RECV_DMABUF, &req) < 0)
		return -errno;

	return 0;
}
