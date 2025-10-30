/*
 * Copyright © Advanced Micro Devices, Inc., or its affiliates. 
 * 
 * SPDX-License-Identifier: MIT
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
    ctx->queue_context = NULL;
    ctx->fmm_context = NULL;
    ctx->event_context = NULL;
}

void hsakmt_kfdcontext_clear_context(HsaKFDContext *ctx)
{
    if (!ctx)
        return;

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
    ctx->fd = -1;
}
