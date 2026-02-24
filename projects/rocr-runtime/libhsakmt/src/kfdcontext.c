/*
 * Copyright © 2025 Advanced Micro Devices, Inc.
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

#include "kfdcontext.h"
#include "libhsakmt.h"
#include <stdlib.h>
#include <stddef.h>
#include <assert.h>
#include <stdio.h>
#include <errno.h>

void hsakmt_kfdcontext_init_context(int fd, HsaKFDContext *ctx)
{
    assert(fd >= 0);
    assert(ctx);

    ctx->fd = fd;
    ctx->topology_context = NULL;
    ctx->queue_context = NULL;
    ctx->fmm_context = NULL;
    ctx->event_context = NULL;
    ctx->debug_context = NULL;
    ctx->perf_context = NULL;
}

void hsakmt_kfdcontext_clear_context(HsaKFDContext *ctx)
{
    if (!ctx)
        return;

    if (ctx->topology_context) {
        free(ctx->topology_context);
        ctx->topology_context = NULL;
    }
    if (ctx->queue_context) {
        free(ctx->queue_context);
        ctx->queue_context = NULL;
    }
    if (ctx->fmm_context) {
        free(ctx->fmm_context);
        ctx->fmm_context = NULL;
    }
    if (ctx->event_context) {
        free(ctx->event_context);
        ctx->event_context = NULL;
    }
    if (ctx->debug_context) {
        free(ctx->debug_context);
        ctx->debug_context = NULL;
    }
    if (ctx->perf_context) {
        free(ctx->perf_context);
        ctx->perf_context = NULL;
    }
    ctx->fd = -1;
}
