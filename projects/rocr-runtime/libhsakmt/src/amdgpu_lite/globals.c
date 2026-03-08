/*
 * Copyright (c) 2026 Advanced Micro Devices, Inc.
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

#include <unistd.h>

#include "libhsakmt.h"
#include "amdgpu_lite_device.h"

/* HSAKMT global data - mirrors src/globals.c */

static pid_t parent_pid = -1;

int hsakmt_udmabuf_dev_fd = -1;
unsigned long hsakmt_kfd_open_count;
pthread_mutex_t hsakmt_mutex = PTHREAD_MUTEX_INITIALIZER;
bool hsakmt_is_dgpu;
bool hsakmt_forked;

int hsakmt_page_size;
int hsakmt_page_shift;

bool hsakmt_is_svm_api_supported;
int hsakmt_zfb_support;

int hsakmt_debug_level;

HsaKFDContext hsakmt_primary_kfd_ctx = {.fd = -1};

/* Global amdgpu_lite device instance */
struct amdgpu_lite_device g_amdgpu_lite_dev = {
	.fd = -1,
	.is_open = false,
};

bool hsakmt_is_forked_child(void)
{
	pid_t cur_pid;

	if (hsakmt_forked)
		return true;

	cur_pid = getpid();

	if (parent_pid == -1) {
		parent_pid = cur_pid;
		return false;
	}

	if (parent_pid != cur_pid) {
		hsakmt_forked = true;
		return true;
	}

	return false;
}
