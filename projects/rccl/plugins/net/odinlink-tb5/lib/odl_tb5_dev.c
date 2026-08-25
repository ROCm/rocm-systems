/*
 * OdinLink Thunderbolt 5 - Device Lifecycle
 */
#include "odl_tb5_priv.h"
#include <odl_tb5/odl_tb5_ioctl.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <sys/mman.h>
#include <sys/ioctl.h>

static int odl_tb5_mmap_buffers(odl_tb5_t h)
{
	struct odl_tb5_buf_info info;
	int ret;

	ret = ioctl(h->fd, ODL_TB5_IOCTL_GET_BUF_INFO, &info);
	if (ret < 0)
		return -errno;

	h->tx_buf_size = info.tx_buf_size;
	h->rx_buf_size = info.rx_buf_size;

	h->tx_bufs[0] = mmap(NULL, h->tx_buf_size, PROT_READ | PROT_WRITE,
			      MAP_SHARED, h->fd, ODL_TB5_MMAP_TX_BUF0);
	if (h->tx_bufs[0] == MAP_FAILED)
		return -errno;

	h->tx_bufs[1] = mmap(NULL, h->tx_buf_size, PROT_READ | PROT_WRITE,
			      MAP_SHARED, h->fd, ODL_TB5_MMAP_TX_BUF1);
	if (h->tx_bufs[1] == MAP_FAILED)
		goto err_unmap_tx0;

	h->rx_bufs[0] = mmap(NULL, h->rx_buf_size, PROT_READ,
			      MAP_SHARED, h->fd, ODL_TB5_MMAP_RX_BUF0);
	if (h->rx_bufs[0] == MAP_FAILED)
		goto err_unmap_tx1;

	h->rx_bufs[1] = mmap(NULL, h->rx_buf_size, PROT_READ,
			      MAP_SHARED, h->fd, ODL_TB5_MMAP_RX_BUF1);
	if (h->rx_bufs[1] == MAP_FAILED)
		goto err_unmap_rx0;

	h->tx_back = 1;
	h->rx_back = 1;

	return 0;

err_unmap_rx0:
	{ int saved = errno;
	munmap(h->rx_bufs[0], h->rx_buf_size);
	errno = saved; }
err_unmap_tx1:
	{ int saved = errno;
	munmap(h->tx_bufs[1], h->tx_buf_size);
	errno = saved; }
err_unmap_tx0:
	{ int saved = errno;
	munmap(h->tx_bufs[0], h->tx_buf_size);
	errno = saved; }
	return -errno;
}

static void odl_tb5_munmap_buffers(odl_tb5_t h)
{
	if (h->tx_bufs[0] && h->tx_bufs[0] != MAP_FAILED)
		munmap(h->tx_bufs[0], h->tx_buf_size);
	if (h->tx_bufs[1] && h->tx_bufs[1] != MAP_FAILED)
		munmap(h->tx_bufs[1], h->tx_buf_size);
	if (h->rx_bufs[0] && h->rx_bufs[0] != MAP_FAILED)
		munmap(h->rx_bufs[0], h->rx_buf_size);
	if (h->rx_bufs[1] && h->rx_bufs[1] != MAP_FAILED)
		munmap(h->rx_bufs[1], h->rx_buf_size);
}

int odl_tb5_open(odl_tb5_t *handle, int index)
{
	char path[64];
	snprintf(path, sizeof(path), "/dev/%s_%d", ODL_TB5_DEVICE_NAME, index);
	return odl_tb5_open_path(handle, path);
}

int odl_tb5_open_path(odl_tb5_t *handle, const char *path)
{
	odl_tb5_t h;
	int ret;

	if (!handle || !path)
		return -EINVAL;

	h = calloc(1, sizeof(*h));
	if (!h)
		return -ENOMEM;

	h->fd = open(path, O_RDWR);
	if (h->fd < 0) {
		ret = -errno;
		free(h);
		return ret;
	}

	ret = odl_tb5_mmap_buffers(h);
	if (ret < 0) {
		close(h->fd);
		free(h);
		return ret;
	}

	*handle = h;
	return 0;
}

void odl_tb5_close(odl_tb5_t handle)
{
	if (!handle)
		return;

	odl_tb5_munmap_buffers(handle);

	if (handle->fd >= 0)
		close(handle->fd);

	free(handle);
}

int odl_tb5_get_fd(odl_tb5_t handle)
{
	if (!handle)
		return -EINVAL;
	return handle->fd;
}

void *odl_tb5_tx_buffer(odl_tb5_t handle, size_t *size)
{
	if (!handle)
		return NULL;
	if (size)
		*size = handle->tx_buf_size;
	return handle->tx_bufs[handle->tx_back];
}

void *odl_tb5_rx_buffer(odl_tb5_t handle, size_t *size)
{
	if (!handle)
		return NULL;
	if (size)
		*size = handle->rx_buf_size;
	return handle->rx_bufs[handle->rx_back];
}

int odl_tb5_get_buf_info(odl_tb5_t handle, uint64_t *tx_size,
			 uint64_t *rx_size)
{
	if (!handle)
		return -EINVAL;
	if (tx_size)
		*tx_size = handle->tx_buf_size;
	if (rx_size)
		*rx_size = handle->rx_buf_size;
	return 0;
}
