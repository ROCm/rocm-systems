/*
 * Copyright 2015 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE COPYRIGHT HOLDER(S) OR AUTHOR(S) BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */
#ifndef AMDP2PTEST_H_
#define AMDP2PTEST_H_

#include <linux/ioctl.h>

#define AMDP2PTEST_IOCTL_MAGIC 'A'


#define AMDP2PTEST_DEVICE_NAME "amdp2ptest"
#define AMDP2PTEST_DEVICE_PATH "/dev/amdp2ptest"

struct AMDRDMA_IOCTL_GET_PAGE_SIZE_PARAM {
	/* Input parameters */
	uint64_t addr;
	uint64_t length;

	/* Output parameters */
	uint64_t page_size;
};

struct AMDRDMA_IOCTL_GET_PAGES_PARAM {
	/* Input parameters */
	uint64_t addr;
	uint64_t length;
	uint64_t is_local;	/* 1 if this is the pointer to local
				   allocation */

	/* Output parameters */
	uint64_t cpu_ptr;
};


struct AMDRDMA_IOCTL_PUT_PAGES_PARAM {
	/* Input parameters */
	uint64_t addr;
	uint64_t length;
};


#define AMD2P2PTEST_IOCTL_GET_PAGE_SIZE	\
_IOWR(AMDP2PTEST_IOCTL_MAGIC, 1, struct AMDRDMA_IOCTL_GET_PAGE_SIZE_PARAM *)

#define AMD2P2PTEST_IOCTL_GET_PAGES \
_IOWR(AMDP2PTEST_IOCTL_MAGIC, 2, struct AMDRDMA_IOCTL_GET_PAGES_PARAM *)

#define AMD2P2PTEST_IOCTL_PUT_PAGES	\
_IOW(AMDP2PTEST_IOCTL_MAGIC, 3, struct AMDRDMA_IOCTL_PUT_PAGES_PARAM *)


#endif  /* AMDP2PTEST_H */
