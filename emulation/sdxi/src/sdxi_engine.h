/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDXI descriptor-queue engine -- state and API visible to the server.
 */

#ifndef SDXI_ENGINE_H
#define SDXI_ENGINE_H

#include <stdbool.h>
#include <stdint.h>

#include "sdxi_regs.h"

struct vfu_ctx;   /* forward-declared; engine takes vfu_ctx_t * */
typedef struct vfu_ctx vfu_ctx_t;

/* ------------------------------------------------------------------ */
/* Per-context state                                                   */
/* ------------------------------------------------------------------ */

typedef enum {
    SDXI_CXT_IDLE    = 0,  /* allocated but not started */
    SDXI_CXT_ACTIVE  = 1,  /* CXT_START received; descriptors processed */
    SDXI_CXT_STOPPED = 2,  /* CXT_STOP received; no further processing */
} sdxi_cxt_state_t;

typedef struct {
    sdxi_cxt_state_t state;
    uint32_t         read_idx;          /* consumer index into the descriptor ring */
    uint64_t         bytes_moved;       /* cumulative DMAB COPY bytes */
    uint64_t         bytes_compressed;  /* cumulative COMPRESS input bytes */
    uint64_t         bytes_decompressed;/* cumulative DECOMPRESS output bytes */
    uint64_t         num_copy;          /* DMAB COPY descriptors processed */
    uint64_t         num_nop;           /* NOP descriptors processed */
    uint64_t         num_sync;          /* SYNC/fence descriptors processed */
    uint64_t         num_compress;      /* COMPRESS descriptors processed */
    uint64_t         num_decompress;    /* DECOMPRESS descriptors processed */
    uint64_t         num_intr;          /* INTR descriptors processed */
    uint64_t         num_cxt_start;     /* CXT_START admin descriptors issued */
    uint64_t         num_cxt_stop;      /* CXT_STOP admin descriptors issued */
    uint64_t         num_errors;        /* total descriptors that completed with error */
} sdxi_cxt_t;

/* ------------------------------------------------------------------ */
/* Per-queue state                                                     */
/* ------------------------------------------------------------------ */

typedef struct sdxi_queue {
    uint32_t id;
    bool     enabled;
    bool     intr_enabled;
    uint32_t intr_vec;      /* MSI-X vector */

    /* SQ ring */
    uint32_t sq_size;       /* capacity in entries */
    uint32_t sq_head;       /* next entry the engine will consume */
    uint32_t sq_tail;       /* last entry the guest has written (doorbell) */
    uint64_t sq_base;       /* GPA of the SQ ring buffer */

    /* CQ ring */
    uint32_t cq_size;       /* capacity in entries */
    uint32_t cq_head;       /* next entry software will read */
    uint32_t cq_tail;       /* next slot the engine will write */
    uint64_t cq_base;       /* GPA of the CQ ring buffer */
    uint8_t  cq_phase;      /* phase bit, toggles on every CQ wrap */
} sdxi_queue_t;

/* ------------------------------------------------------------------ */
/* Device-level engine state (embedded in the top-level device struct) */
/* ------------------------------------------------------------------ */

typedef struct sdxi_engine {
    sdxi_queue_t queues[SDXI_MAX_QUEUES];
    uint32_t     num_queues;   /* number allocated (== SDXI_MAX_QUEUES) */
} sdxi_engine_t;

/* ------------------------------------------------------------------ */
/* API                                                                 */
/* ------------------------------------------------------------------ */

/* Initialise engine state to power-on defaults. */
void sdxi_engine_init(sdxi_engine_t *eng);

/* Reset one queue to its power-on defaults, preserving no guest state. */
void sdxi_engine_queue_reset(sdxi_engine_t *eng, uint32_t qid);

/*
 * Doorbell write: guest advanced SQ tail to new_tail for queue qid.
 * Recorded now; work is drained by sdxi_engine_poll().
 */
void sdxi_engine_doorbell(sdxi_engine_t *eng, uint32_t qid, uint32_t new_tail);

/*
 * Poll all enabled queues for pending descriptors.  For each pending SQE
 * the engine executes the DMA operation using vfu_addr_to_sgl /
 * vfu_sgl_read / vfu_sgl_write, posts a CQE, and optionally fires MSI-X.
 *
 * Call this in the main loop after each vfu_run_ctx() invocation.
 *
 * Returns the number of descriptors processed (0 if idle).
 */
int sdxi_engine_poll(sdxi_engine_t *eng, vfu_ctx_t *vfu_ctx);

/*
 * Process v3 descriptors for context cid.  Called from the BAR2 doorbell
 * handler when the driver is in context-table mode (mmio_l2_ptr != 0).
 *
 *   l2_gpa        -- L2 context-table GPA (from SDXI_MMIO_CXT_L2 register)
 *   cid           -- context ID (doorbell offset / SDXI_DB_STRIDE)
 *   new_write_idx -- new producer index from the doorbell value
 *   cxt_arr       -- array of per-context state (indexed by cid)
 *   num_cxt       -- length of cxt_arr (== SDXI_MAX_CXT)
 */
void sdxi_engine_process_cxt(vfu_ctx_t *vfu_ctx, uint64_t l2_gpa,
                              uint32_t cid, uint32_t new_write_idx,
                              sdxi_cxt_t *cxt_arr, uint32_t num_cxt);

#endif /* SDXI_ENGINE_H */
