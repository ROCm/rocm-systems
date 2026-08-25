/*
 * OdinLink Thunderbolt 5 - Private Library Header
 *
 * Internal handle definition shared across library source files.
 * NOT part of the public API.
 */
#ifndef ODL_TB5_PRIV_H
#define ODL_TB5_PRIV_H

#include <odl_tb5/odl_tb5.h>
#include <stddef.h>

struct odl_tb5_handle {
	int    fd;
	void  *tx_bufs[2];
	void  *rx_bufs[2];
	size_t tx_buf_size;
	size_t rx_buf_size;
	int    tx_back;
	int    rx_back;
};

#endif /* ODL_TB5_PRIV_H */
