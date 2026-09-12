/* SPDX-License-Identifier: MIT */
/*
 * Copyright (c) OdinLink-Five contributors
 * https://github.com/Geramy/OdinLink-Five
 *
 * OdinLink Thunderbolt 5 - Public Userspace API
 */
#ifndef ODL_TB5_H
#define ODL_TB5_H

#include <odl_tb5/odl_tb5_types.h>
#include <stdint.h>
#include <stddef.h>
#include <sys/types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle */
typedef struct odl_tb5_handle *odl_tb5_t;

/* Open device by index (opens /dev/odl_tb5_N, mmaps double buffers) */
int odl_tb5_open(odl_tb5_t *handle, int index);

/* Open device by path */
int odl_tb5_open_path(odl_tb5_t *handle, const char *path);

/* Close device, unmap all buffers */
void odl_tb5_close(odl_tb5_t handle);

/* Get info about the connected peer (non-blocking) */
int odl_tb5_get_peer(odl_tb5_t handle, struct odl_tb5_peer_info *info);

/* Get local and peer identity for this link (non-blocking) */
int odl_tb5_get_link_info(odl_tb5_t handle, struct odl_tb5_link_info *info);

/* Wait for peer connection (blocking, timeout_ms=0 means forever) */
int odl_tb5_wait_peer(odl_tb5_t handle, int timeout_ms);

/* Get pointer to the back TX buffer (userspace writes here) */
void *odl_tb5_tx_buffer(odl_tb5_t handle, size_t *size);

/* Get pointer to the back RX buffer (userspace reads here) */
void *odl_tb5_rx_buffer(odl_tb5_t handle, size_t *size);

/* Submit data from TX front buffer for DMA transfer */
int odl_tb5_send(odl_tb5_t handle, size_t offset, size_t len);

/* Submit control message from TX front buffer */
int odl_tb5_send_ctrl(odl_tb5_t handle, size_t offset, size_t len);

/* Post receive into RX front buffer */
int odl_tb5_recv(odl_tb5_t handle, size_t offset, size_t len);

/* Swap TX double buffer (front <-> back) */
int odl_tb5_swap_tx(odl_tb5_t handle);

/* Swap RX double buffer (front <-> back) */
int odl_tb5_swap_rx(odl_tb5_t handle);

/* Send from a DMA-buf (GPU memory, zero-copy) */
int odl_tb5_send_dmabuf(odl_tb5_t handle, int dmabuf_fd,
			off_t offset, size_t len);

/* Receive into a DMA-buf (GPU memory, zero-copy) */
int odl_tb5_recv_dmabuf(odl_tb5_t handle, int dmabuf_fd,
			off_t offset, size_t len);

/* Poll for completions (non-blocking) */
int odl_tb5_poll(odl_tb5_t handle, struct odl_tb5_completion *comp);

/* Wait for at least one TX completion (blocking, resets TX counter) */
int odl_tb5_wait_tx(odl_tb5_t handle, struct odl_tb5_completion *comp);

/* Wait for at least one RX completion (blocking, resets RX counter) */
int odl_tb5_wait_rx(odl_tb5_t handle, struct odl_tb5_completion *comp);

/* Get double buffer sizes */
int odl_tb5_get_buf_info(odl_tb5_t handle, uint64_t *tx_size,
			 uint64_t *rx_size);

/* Get raw file descriptor (for advanced use) */
int odl_tb5_get_fd(odl_tb5_t handle);

/* ── Stream-based multiplexed I/O ──────────────────────────────────── */

/* Open a stream on the device. filter_id=0 for auto-assign, 1-255 for specific. */
int odl_tb5_stream_open(odl_tb5_t handle, uint8_t filter_id,
			uint8_t *stream_id_out);

/* Close a stream */
int odl_tb5_stream_close(odl_tb5_t handle, uint8_t stream_id);

/* Send data on a stream (blocking until TX complete, kernel-managed DMA) */
int odl_tb5_stream_send(odl_tb5_t handle, uint8_t stream_id,
			uint8_t dst_id, const void *data, uint32_t len);

/* Receive data from a stream (blocking until data arrives) */
int odl_tb5_stream_recv(odl_tb5_t handle, uint8_t stream_id,
			void *buf, uint32_t buf_len,
			uint8_t *src_id, uint32_t *actual_len);

/* Wait for TX completion on a stream (timeout_ms=0 means forever) */
int odl_tb5_stream_wait_tx(odl_tb5_t handle, uint8_t stream_id,
			   uint32_t timeout_ms);

/* Wait for RX data on a stream (timeout_ms=0 means forever) */
int odl_tb5_stream_wait_rx(odl_tb5_t handle, uint8_t stream_id,
			   uint32_t timeout_ms);

/* Send from a DMA-buf on a stream (zero-copy GPU TX) */
int odl_tb5_stream_send_dmabuf(odl_tb5_t handle, uint8_t stream_id,
			       uint8_t dst_id, int dmabuf_fd,
			       uint64_t offset, uint64_t len);

/* Receive into a DMA-buf on a stream (zero-copy GPU RX) */
int odl_tb5_stream_recv_dmabuf(odl_tb5_t handle, uint8_t stream_id,
			       int dmabuf_fd, uint64_t offset, uint64_t len);

#ifdef __cplusplus
}
#endif

#endif /* ODL_TB5_H */
