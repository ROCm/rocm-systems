/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * SDXI descriptor-queue engine.
 *
 * The engine processes SQ descriptors posted by the guest (via doorbell
 * writes or direct SQ tail register updates) and writes CQ completion
 * entries back to guest memory, then optionally fires MSI-X.
 *
 * DMA is performed entirely through the libvfio-user scatter/gather API
 * (vfu_addr_to_sgl + vfu_sgl_read/write), which safely maps guest
 * physical memory via the DMA regions the client registered.
 */

#include "sdxi_engine.h"
#include "sdxi_log.h"
#include "sdxi_regs.h"

#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/mman.h>
#include <zlib.h>
#include <lz4.h>
#include <zstd.h>

#include <vfio-user/libvfio-user.h>

/* Maximum contiguous SG segments we allocate on the stack. */
#define SDXI_MAX_SG_SEGS 64

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

static inline uint32_t queue_avail(const sdxi_queue_t *q)
{
    /* Number of unconsumed SQ entries */
    return (q->sq_tail - q->sq_head) % q->sq_size;
}

static inline bool cq_full(const sdxi_queue_t *q)
{
    return ((q->cq_tail + 1U) % q->cq_size) == q->cq_head;
}

/* ------------------------------------------------------------------ */
/* Initialisation                                                      */
/* ------------------------------------------------------------------ */

void sdxi_engine_init(sdxi_engine_t *eng)
{
    memset(eng, 0, sizeof(*eng));
    eng->num_queues = SDXI_MAX_QUEUES;
    for (uint32_t i = 0; i < SDXI_MAX_QUEUES; i++)
        eng->queues[i].id = i;
}

void sdxi_engine_queue_reset(sdxi_engine_t *eng, uint32_t qid)
{
    if (qid >= SDXI_MAX_QUEUES)
        return;
    sdxi_queue_t *q = &eng->queues[qid];
    uint32_t id = q->id;
    memset(q, 0, sizeof(*q));
    q->id = id;
}

/* ------------------------------------------------------------------ */
/* Doorbell                                                            */
/* ------------------------------------------------------------------ */

void sdxi_engine_doorbell(sdxi_engine_t *eng, uint32_t qid, uint32_t new_tail)
{
    if (qid >= SDXI_MAX_QUEUES) {
        sdxi_warn("doorbell for invalid queue %u", qid);
        return;
    }
    sdxi_queue_t *q = &eng->queues[qid];
    if (!q->enabled) {
        sdxi_warn("doorbell for disabled queue %u (tail=%u)", qid, new_tail);
        return;
    }
    if (q->sq_size == 0) {
        sdxi_warn("doorbell for zero-size queue %u", qid);
        return;
    }
    q->sq_tail = new_tail % q->sq_size;
    sdxi_debug("queue %u doorbell: sq_tail=%u", qid, q->sq_tail);
}

/* ------------------------------------------------------------------ */
/* CQE posting                                                         */
/* ------------------------------------------------------------------ */

static int post_cqe(sdxi_engine_t *eng __attribute__((unused)), vfu_ctx_t *vfu_ctx,
                    sdxi_queue_t *q, uint32_t cid, uint8_t status)
{
    if (cq_full(q)) {
        sdxi_error("queue %u CQ is full, dropping completion for cid=%u",
                   q->id, cid);
        return -1;
    }

    uint64_t cqe_gpa = q->cq_base +
        (uint64_t)q->cq_tail * SDXI_CQE_SIZE;

    struct sdxi_cqe cqe = {
        .status   = status,
        .cid      = cid,
        .sq_head  = (uint16_t)(q->sq_head % q->sq_size),
        .sq_id    = (uint16_t)q->id,
        .phase    = q->cq_phase,
        .reserved = {0},
    };

    /* Allocate SG for the CQE write target */
    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;

    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)cqe_gpa,
                              SDXI_CQE_SIZE, sg, 1, PROT_WRITE);
    if (nr < 0) {
        sdxi_error("queue %u: vfu_addr_to_sgl for CQE failed: %s",
                   q->id, strerror(errno));
        return -1;
    }

    int rc = vfu_sgl_write(vfu_ctx, sg, (size_t)nr,
                           &cqe, VFU_SGL_DIRECT_ACCESS);
    if (rc != 0) {
        sdxi_error("queue %u: vfu_sgl_write CQE failed: %s",
                   q->id, strerror(errno));
        return -1;
    }

    q->cq_tail = (q->cq_tail + 1U) % q->cq_size;
    if (q->cq_tail == 0)
        q->cq_phase ^= 1U;   /* phase toggles on wrap */

    if (q->intr_enabled) {
        int irq_rc = vfu_irq_trigger(vfu_ctx, q->intr_vec);
        if (irq_rc != 0)
            sdxi_warn("queue %u: MSI-X trigger failed: %s",
                      q->id, strerror(errno));
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Zero-copy SGL helpers                                               */
/* ------------------------------------------------------------------ */

/*
 * Copy len bytes from src SGL to dst SGL.
 * Fast path: map both via vfu_sgl_get() and memcpy between iov_base ptrs.
 * Slow path (any segment not mappable): malloc bounce + read/write.
 */
static int sgl_memcpy(vfu_ctx_t *ctx,
                      dma_sg_t *src_sgl, int src_nr,
                      dma_sg_t *dst_sgl, int dst_nr,
                      size_t len)
{
    bool mappable = true;
    for (int i = 0; i < src_nr && mappable; i++)
        mappable = vfu_sg_is_mappable(ctx,
            (dma_sg_t *)(void *)((uint8_t *)src_sgl + (size_t)i * dma_sg_size()));
    for (int i = 0; i < dst_nr && mappable; i++)
        mappable = vfu_sg_is_mappable(ctx,
            (dma_sg_t *)(void *)((uint8_t *)dst_sgl + (size_t)i * dma_sg_size()));

    if (mappable) {
        struct iovec src_iov[SDXI_MAX_SG_SEGS];
        struct iovec dst_iov[SDXI_MAX_SG_SEGS];
        if (vfu_sgl_get(ctx, src_sgl, src_iov, (size_t)src_nr, 0) == 0 &&
            vfu_sgl_get(ctx, dst_sgl, dst_iov, (size_t)dst_nr, 0) == 0) {
            /* Walk iov pairs and memcpy segment by segment. */
            int si = 0, di = 0;
            size_t soff = 0, doff = 0, done = 0;
            while (done < len && si < src_nr && di < dst_nr) {
                size_t sc = src_iov[si].iov_len - soff;
                size_t dc = dst_iov[di].iov_len - doff;
                size_t chunk = sc < dc ? sc : dc;
                if (done + chunk > len)
                    chunk = len - done;
                memcpy((uint8_t *)dst_iov[di].iov_base + doff,
                       (uint8_t *)src_iov[si].iov_base + soff, chunk);
                done += chunk; soff += chunk; doff += chunk;
                if (soff == src_iov[si].iov_len) { soff = 0; si++; }
                if (doff == dst_iov[di].iov_len) { doff = 0; di++; }
            }
            vfu_sgl_put(ctx, dst_sgl, dst_iov, (size_t)dst_nr);
            vfu_sgl_put(ctx, src_sgl, src_iov, (size_t)src_nr);
            return 0;
        }
        /* vfu_sgl_get failed — fall through to bounce */
    }

    uint8_t *bounce = malloc(len);
    if (!bounce)
        return -1;
    int rc = vfu_sgl_read(ctx, src_sgl, (size_t)src_nr, bounce,
                          VFU_SGL_DIRECT_ACCESS);
    if (rc == 0)
        rc = vfu_sgl_write(ctx, dst_sgl, (size_t)dst_nr, bounce,
                           VFU_SGL_DIRECT_ACCESS);
    free(bounce);
    return rc;
}

/*
 * Fill len bytes of dst SGL with pattern.
 * Fast path: map via vfu_sgl_get() and memset iov_base ptrs.
 * Slow path: malloc bounce + write.
 */
static int sgl_memset(vfu_ctx_t *ctx,
                      dma_sg_t *dst_sgl, int dst_nr,
                      uint8_t pattern, size_t len)
{
    bool mappable = true;
    for (int i = 0; i < dst_nr && mappable; i++)
        mappable = vfu_sg_is_mappable(ctx,
            (dma_sg_t *)(void *)((uint8_t *)dst_sgl + (size_t)i * dma_sg_size()));

    if (mappable) {
        struct iovec dst_iov[SDXI_MAX_SG_SEGS];
        if (vfu_sgl_get(ctx, dst_sgl, dst_iov, (size_t)dst_nr, 0) == 0) {
            size_t done = 0;
            for (int i = 0; i < dst_nr && done < len; i++) {
                size_t chunk = dst_iov[i].iov_len;
                if (done + chunk > len)
                    chunk = len - done;
                memset(dst_iov[i].iov_base, pattern, chunk);
                done += chunk;
            }
            vfu_sgl_put(ctx, dst_sgl, dst_iov, (size_t)dst_nr);
            return 0;
        }
    }

    uint8_t *bounce = malloc(len);
    if (!bounce)
        return -1;
    memset(bounce, pattern, len);
    int rc = vfu_sgl_write(ctx, dst_sgl, (size_t)dst_nr, bounce,
                           VFU_SGL_DIRECT_ACCESS);
    free(bounce);
    return rc;
}

/* ------------------------------------------------------------------ */
/* Descriptor execution                                                */
/* ------------------------------------------------------------------ */

static int exec_memcpy(vfu_ctx_t *vfu_ctx, const struct sdxi_sqe *sqe)
{
    if (sqe->length == 0)
        return 0;

    uint8_t  src_sg_buf[SDXI_MAX_SG_SEGS * dma_sg_size()];
    uint8_t  dst_sg_buf[SDXI_MAX_SG_SEGS * dma_sg_size()];
    dma_sg_t *src_sgl = (dma_sg_t *)(void *)src_sg_buf;
    dma_sg_t *dst_sgl = (dma_sg_t *)(void *)dst_sg_buf;

    int src_nr = vfu_addr_to_sgl(vfu_ctx,
                     (vfu_dma_addr_t)(uintptr_t)sqe->src_addr,
                     (size_t)sqe->length, src_sgl, SDXI_MAX_SG_SEGS,
                     PROT_READ);
    if (src_nr < 0) {
        sdxi_error("memcpy: src vfu_addr_to_sgl failed (gpa=0x%llx len=0x%llx): %s",
                   (unsigned long long)sqe->src_addr,
                   (unsigned long long)sqe->length, strerror(errno));
        return SDXI_CQE_STATUS_FAULT;
    }

    int dst_nr = vfu_addr_to_sgl(vfu_ctx,
                     (vfu_dma_addr_t)(uintptr_t)sqe->dst_addr,
                     (size_t)sqe->length, dst_sgl, SDXI_MAX_SG_SEGS,
                     PROT_WRITE);
    if (dst_nr < 0) {
        sdxi_error("memcpy: dst vfu_addr_to_sgl failed (gpa=0x%llx len=0x%llx): %s",
                   (unsigned long long)sqe->dst_addr,
                   (unsigned long long)sqe->length, strerror(errno));
        return SDXI_CQE_STATUS_FAULT;
    }

    if (sgl_memcpy(vfu_ctx, src_sgl, src_nr, dst_sgl, dst_nr,
                   (size_t)sqe->length) != 0) {
        sdxi_error("memcpy: sgl_memcpy failed: %s", strerror(errno));
        return SDXI_CQE_STATUS_FAULT;
    }

    sdxi_debug("memcpy cid=%u: 0x%llx -> 0x%llx len=0x%llx",
               sqe->cid, (unsigned long long)sqe->src_addr,
               (unsigned long long)sqe->dst_addr,
               (unsigned long long)sqe->length);
    return SDXI_CQE_STATUS_OK;
}

static int exec_memset(vfu_ctx_t *vfu_ctx, const struct sdxi_sqe *sqe)
{
    if (sqe->length == 0)
        return 0;

    uint8_t pattern = (uint8_t)(sqe->src_addr & 0xFFU);

    uint8_t dst_sg_buf[SDXI_MAX_SG_SEGS * dma_sg_size()];
    dma_sg_t *dst_sgl = (dma_sg_t *)(void *)dst_sg_buf;

    int dst_nr = vfu_addr_to_sgl(vfu_ctx,
                     (vfu_dma_addr_t)(uintptr_t)sqe->dst_addr,
                     (size_t)sqe->length, dst_sgl, SDXI_MAX_SG_SEGS,
                     PROT_WRITE);
    if (dst_nr < 0) {
        sdxi_error("memset: vfu_addr_to_sgl failed: %s", strerror(errno));
        return SDXI_CQE_STATUS_FAULT;
    }

    if (sgl_memset(vfu_ctx, dst_sgl, dst_nr, pattern, (size_t)sqe->length) != 0) {
        sdxi_error("memset: sgl_memset failed: %s", strerror(errno));
        return SDXI_CQE_STATUS_FAULT;
    }

    sdxi_debug("memset cid=%u: 0x%llx pattern=0x%02x len=0x%llx",
               sqe->cid, (unsigned long long)sqe->dst_addr,
               pattern, (unsigned long long)sqe->length);
    return SDXI_CQE_STATUS_OK;
}

/* Forward declarations for helpers defined after the compress functions. */
static int read_guest(vfu_ctx_t *vfu_ctx, uint64_t gpa, void *buf, size_t len);
static int write_guest(vfu_ctx_t *vfu_ctx, uint64_t gpa, const void *buf, size_t len);

/*
 * Compress src_len bytes at src_gpa into dst_gpa.
 * algo: SDXI_COMP_ALGO_ZLIB/LZ4/ZSTD  level: 0=default 1-9=algo-specific.
 * Returns actual compressed size on success, -1 on failure.
 * On success writes result_len into the CSB at csb_gpa (if non-zero).
 */
static ssize_t exec_compress(vfu_ctx_t *vfu_ctx, uint64_t src_gpa,
                              uint32_t src_len, uint64_t dst_gpa,
                              uint32_t dst_cap, int level, uint32_t algo,
                              uint64_t csb_gpa)
{
    if (src_len == 0)
        return 0;

    uint8_t *src_buf = malloc(src_len);
    uint8_t *dst_buf = malloc(dst_cap);
    if (!src_buf || !dst_buf) {
        free(src_buf); free(dst_buf);
        return -1;
    }

    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;

    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)src_gpa,
                              src_len, sg, 1, PROT_READ);
    int rc = (nr < 0) ? -1 : vfu_sgl_read(vfu_ctx, sg, (size_t)nr, src_buf,
                                            VFU_SGL_DIRECT_ACCESS);
    if (rc != 0) {
        sdxi_error("compress: failed to read src (gpa=0x%llx len=%u)",
                   (unsigned long long)src_gpa, src_len);
        free(src_buf); free(dst_buf);
        return -1;
    }

    size_t out_len;
    switch (algo) {
    case SDXI_COMP_ALGO_LZ4: {
        int lrc = LZ4_compress_default((const char *)src_buf, (char *)dst_buf,
                                       (int)src_len, (int)dst_cap);
        free(src_buf);
        if (lrc <= 0) {
            sdxi_error("compress: LZ4 error (src=%u dst_cap=%u)", src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = (size_t)lrc;
        break;
    }
    case SDXI_COMP_ALGO_ZSTD: {
        int zstd_level = (level == 0) ? ZSTD_CLEVEL_DEFAULT : level;
        size_t zrc = ZSTD_compress(dst_buf, (size_t)dst_cap, src_buf,
                                   (size_t)src_len, zstd_level);
        free(src_buf);
        if (ZSTD_isError(zrc)) {
            sdxi_error("compress: ZSTD error: %s (src=%u dst_cap=%u)",
                       ZSTD_getErrorName(zrc), src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = zrc;
        break;
    }
    default: { /* SDXI_COMP_ALGO_ZLIB */
        uLongf zlib_out = (uLongf)dst_cap;
        int zrc = compress2(dst_buf, &zlib_out, src_buf, (uLong)src_len,
                            level == 0 ? Z_DEFAULT_COMPRESSION : level);
        free(src_buf);
        if (zrc != Z_OK) {
            sdxi_error("compress: zlib error %d (src=%u dst_cap=%u)", zrc, src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = (size_t)zlib_out;
        break;
    }
    }

    nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)dst_gpa,
                          out_len, sg, 1, PROT_WRITE);
    rc = (nr < 0) ? -1 : vfu_sgl_write(vfu_ctx, sg, (size_t)nr, dst_buf,
                                         VFU_SGL_DIRECT_ACCESS);
    free(dst_buf);
    if (rc != 0) {
        sdxi_error("compress: failed to write dst (gpa=0x%llx len=%zu)",
                   (unsigned long long)dst_gpa, out_len);
        return -1;
    }

    sdxi_debug("compress: src=%u algo=%u -> out=%zu bytes", src_len, algo, out_len);

    if (csb_gpa != 0ULL) {
        struct sdxi_cst_blk csb;
        memset(&csb, 0, sizeof(csb));
        uint64_t result = (uint64_t)out_len;
        memcpy((uint8_t *)&csb + SDXI_CSB_RESULT_LEN_OFF, &result, sizeof(result));
        write_guest(vfu_ctx, csb_gpa, &csb, sizeof(csb));
        vfu_irq_trigger(vfu_ctx, 0);
    }

    return (ssize_t)out_len;
}

/*
 * Decompress src_len bytes at src_gpa into dst_gpa.
 * algo must match the value used to compress (SDXI_COMP_ALGO_ZLIB/LZ4/ZSTD).
 * Returns actual decompressed size on success, -1 on failure.
 */
static ssize_t exec_decompress(vfu_ctx_t *vfu_ctx, uint64_t src_gpa,
                                uint32_t src_len, uint64_t dst_gpa,
                                uint32_t dst_cap, uint32_t algo,
                                uint64_t csb_gpa)
{
    if (src_len == 0)
        return 0;

    uint8_t *src_buf = malloc(src_len);
    uint8_t *dst_buf = malloc(dst_cap);
    if (!src_buf || !dst_buf) {
        free(src_buf); free(dst_buf);
        return -1;
    }

    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;

    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)src_gpa,
                              src_len, sg, 1, PROT_READ);
    int rc = (nr < 0) ? -1 : vfu_sgl_read(vfu_ctx, sg, (size_t)nr, src_buf,
                                            VFU_SGL_DIRECT_ACCESS);
    if (rc != 0) {
        sdxi_error("decompress: failed to read src (gpa=0x%llx len=%u)",
                   (unsigned long long)src_gpa, src_len);
        free(src_buf); free(dst_buf);
        return -1;
    }

    size_t out_len;
    switch (algo) {
    case SDXI_COMP_ALGO_LZ4: {
        int lrc = LZ4_decompress_safe((const char *)src_buf, (char *)dst_buf,
                                      (int)src_len, (int)dst_cap);
        free(src_buf);
        if (lrc < 0) {
            sdxi_error("decompress: LZ4 error %d (src=%u dst_cap=%u)", lrc, src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = (size_t)lrc;
        break;
    }
    case SDXI_COMP_ALGO_ZSTD: {
        size_t zrc = ZSTD_decompress(dst_buf, (size_t)dst_cap, src_buf, (size_t)src_len);
        free(src_buf);
        if (ZSTD_isError(zrc)) {
            sdxi_error("decompress: ZSTD error: %s (src=%u dst_cap=%u)",
                       ZSTD_getErrorName(zrc), src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = zrc;
        break;
    }
    default: { /* SDXI_COMP_ALGO_ZLIB */
        uLongf zlib_out = (uLongf)dst_cap;
        int zrc = uncompress(dst_buf, &zlib_out, src_buf, (uLong)src_len);
        free(src_buf);
        if (zrc != Z_OK) {
            sdxi_error("decompress: zlib error %d (src=%u dst_cap=%u)", zrc, src_len, dst_cap);
            free(dst_buf);
            return -1;
        }
        out_len = (size_t)zlib_out;
        break;
    }
    }

    nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)dst_gpa,
                          out_len, sg, 1, PROT_WRITE);
    rc = (nr < 0) ? -1 : vfu_sgl_write(vfu_ctx, sg, (size_t)nr, dst_buf,
                                         VFU_SGL_DIRECT_ACCESS);
    free(dst_buf);
    if (rc != 0) {
        sdxi_error("decompress: failed to write dst (gpa=0x%llx len=%zu)",
                   (unsigned long long)dst_gpa, out_len);
        return -1;
    }

    sdxi_debug("decompress: src=%u algo=%u -> out=%zu bytes", src_len, algo, out_len);

    if (csb_gpa != 0ULL) {
        struct sdxi_cst_blk csb;
        memset(&csb, 0, sizeof(csb));
        uint64_t result = (uint64_t)out_len;
        memcpy((uint8_t *)&csb + SDXI_CSB_RESULT_LEN_OFF, &result, sizeof(result));
        write_guest(vfu_ctx, csb_gpa, &csb, sizeof(csb));
        vfu_irq_trigger(vfu_ctx, 0);
    }

    return (ssize_t)out_len;
}

/* ------------------------------------------------------------------ */
/* Elementwise arithmetic helpers — rocm-ernic p4_fp8.c pattern:      */
/* memcpy for type punning, float as the intermediate arithmetic type. */
/* ------------------------------------------------------------------ */

static float bf16_to_float(uint16_t v) {
    uint32_t f32 = (uint32_t)v << 16;
    float r; memcpy(&r, &f32, 4); return r;
}
static uint16_t bf16_from_float(float v) {
    uint32_t b; memcpy(&b, &v, 4); return (uint16_t)(b >> 16);
}

static float fp16_to_float(uint16_t v) {
    uint32_t s = (v >> 15) & 1U, e = (v >> 10) & 0x1FU, m = v & 0x3FFU;
    uint32_t f32;
    if (e == 0x1FU) {
        f32 = (s << 31) | 0x7F800000U | (m << 13);
    } else if (e == 0 && m == 0) {
        f32 = s << 31;
    } else if (e == 0) {
        float fv = (float)m * 5.9604644775390625e-8f;
        return s ? -fv : fv;
    } else {
        uint32_t fe = e - 15 + 127;
        f32 = (s << 31) | (fe << 23) | (m << 13);
    }
    float r; memcpy(&r, &f32, 4); return r;
}
static uint16_t fp16_from_float(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint32_t s = (b >> 31) & 1U, abs_b = b & 0x7FFFFFFFU;
    uint32_t fe = (abs_b >> 23) & 0xFFU;
    if (fe == 0xFFU)
        return (uint16_t)((s << 15) | 0x7C00U | ((abs_b & 0x7FFFFFU) ? 0x200U : 0U));
    if (abs_b > 0x477FE000U) return (uint16_t)((s << 15) | 0x7C00U);
    if (abs_b == 0) return (uint16_t)(s << 15);
    int exp32 = (int)fe - 127, exp16 = exp32 + 15;
    uint32_t man32 = abs_b & 0x7FFFFFU;
    if (exp16 <= 0) {
        uint32_t shift = (uint32_t)(1 - exp16) + 13;
        if (shift > 23) return (uint16_t)(s << 15);
        uint32_t m = (0x800000U | man32) >> shift;
        return (uint16_t)((s << 15) | (m & 0x3FFU));
    }
    return (uint16_t)((s << 15) | ((uint32_t)exp16 << 10) | (man32 >> 13));
}

/* FP8 E4M3: OCP convention — bias=7, max=448, NaN=0x7F/0xFF, no infinities. */
static float e4m3_to_float(uint8_t v) {
    uint8_t s = v >> 7, e = (v >> 3) & 0xFU, m = v & 0x7U;
    if (e == 0xFU && m == 0x7U) {
        uint32_t nb = ((uint32_t)s << 31) | 0x7FC00000U;
        float r; memcpy(&r, &nb, 4); return r;
    }
    if (e == 0 && m == 0) return s ? -0.0f : 0.0f;
    if (e == 0) {
        float fv = (float)m * 0.001953125f; /* m × 2^(-9) */
        return s ? -fv : fv;
    }
    uint32_t fe = (uint32_t)e - 7 + 127;
    uint32_t f32 = ((uint32_t)s << 31) | (fe << 23) | ((uint32_t)m << 20);
    float r; memcpy(&r, &f32, 4); return r;
}
static uint8_t e4m3_from_float(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint8_t sign = (b >> 31) & 1U;
    uint32_t abs_b = b & 0x7FFFFFFFU;
    if ((abs_b >> 23) == 0xFFU) return sign ? 0xFFU : 0x7FU; /* NaN/Inf → NaN */
    if (abs_b > 0x43E00000U) return sign ? (uint8_t)(0x80U | 0x7EU) : 0x7EU; /* >448 → sat */
    if (abs_b == 0) return sign ? 0x80U : 0x00U;
    int exp32 = (int)((abs_b >> 23) & 0xFFU) - 127, exp8 = exp32 + 7;
    uint32_t man32 = abs_b & 0x7FFFFFU;
    if (exp8 <= 0) {
        uint32_t shift = (uint32_t)(1 - exp8) + 20;
        if (shift > 23) return sign ? 0x80U : 0x00U;
        uint32_t m = (0x800000U | man32) >> shift;
        if (!m) return sign ? 0x80U : 0x00U;
        return (uint8_t)((sign << 7) | (m & 0x7U));
    }
    uint8_t e8 = (uint8_t)exp8, m8 = (uint8_t)(man32 >> 20);
    return (uint8_t)((sign << 7) | (e8 << 3) | (m8 & 0x7U));
}

/* FP8 E5M2: OCP convention — bias=15, has infinities, NaN when e=11111 m!=0. */
static float e5m2_to_float(uint8_t v) {
    uint8_t s = v >> 7, e = (v >> 2) & 0x1FU, m = v & 0x3U;
    uint32_t f32;
    if (e == 0x1FU) {
        f32 = m ? (((uint32_t)s << 31) | 0x7FC00000U)
                : (((uint32_t)s << 31) | 0x7F800000U);
    } else if (e == 0 && m == 0) {
        f32 = (uint32_t)s << 31;
    } else if (e == 0) {
        float fv = (float)m * (1.0f / 65536.0f); /* m × 2^(-16) */
        return s ? -fv : fv;
    } else {
        uint32_t fe = (uint32_t)e - 15 + 127;
        f32 = ((uint32_t)s << 31) | (fe << 23) | ((uint32_t)m << 21);
    }
    float r; memcpy(&r, &f32, 4); return r;
}
static uint8_t e5m2_from_float(float v) {
    uint32_t b; memcpy(&b, &v, 4);
    uint8_t sign = (b >> 31) & 1U;
    uint32_t abs_b = b & 0x7FFFFFFFU;
    if ((abs_b >> 23) == 0xFFU) {
        if (!(abs_b & 0x7FFFFFU)) return sign ? 0xFCU : 0x7CU; /* Inf */
        return sign ? 0xFFU : 0x7FU; /* NaN */
    }
    if (abs_b > 0x47600000U) return sign ? 0xFCU : 0x7CU; /* overflow → Inf */
    if (abs_b == 0) return sign ? 0x80U : 0x00U;
    int exp32 = (int)((abs_b >> 23) & 0xFFU) - 127, exp5 = exp32 + 15;
    uint32_t man32 = abs_b & 0x7FFFFFU;
    if (exp5 <= 0) {
        uint32_t shift = (uint32_t)(1 - exp5) + 21;
        if (shift > 23) return sign ? 0x80U : 0x00U;
        uint32_t m = (0x800000U | man32) >> shift;
        return (uint8_t)((sign << 7) | (m & 0x3U));
    }
    uint8_t e5 = (uint8_t)exp5, m5 = (uint8_t)(man32 >> 21);
    return (uint8_t)((sign << 7) | (e5 << 2) | (m5 & 0x3U));
}

/* INT4: nibble-packed, low nibble first, signed two's complement -8..7. */
static int int4_from_nibble(uint8_t n) {
    n &= 0xFU; return (n >= 8) ? (int)n - 16 : (int)n;
}
static uint8_t int4_to_nibble(int v) {
    if (v < -8) v = -8; if (v > 7) v = 7;
    return (uint8_t)(v & 0xFU);
}

static int exec_elemwise(vfu_ctx_t *vfu_ctx,
                         uint64_t src_gpa, uint32_t src_len,
                         uint64_t dst_gpa,
                         uint32_t dtype, uint32_t op,
                         uint64_t scalar_bits,
                         uint64_t csb_gpa)
{
    if (src_len == 0)
        return -1;

    /* Determine element size; INT4 packs 2 elements per byte. */
    size_t elem_sz;
    size_t n_elems;
    switch (dtype) {
    case SDXI_DTYPE_FP32:  case SDXI_DTYPE_INT32:  elem_sz = 4; n_elems = src_len / 4; break;
    case SDXI_DTYPE_FP16:  case SDXI_DTYPE_BF16:   elem_sz = 2; n_elems = src_len / 2; break;
    case SDXI_DTYPE_INT8:  case SDXI_DTYPE_FP8_E4M3:
    case SDXI_DTYPE_FP8_E5M2:                       elem_sz = 1; n_elems = src_len;     break;
    case SDXI_DTYPE_INT4:                            elem_sz = 1; n_elems = src_len * 2; break;
    default: return -1;
    }

    bool is_reduce = (op == SDXI_ELEMOP_REDUCE_MIN || op == SDXI_ELEMOP_REDUCE_MAX);
    size_t dst_len = is_reduce ? elem_sz : (size_t)src_len;

    uint8_t *src_buf = malloc(src_len);
    uint8_t *dst_buf = calloc(1, dst_len);
    if (!src_buf || !dst_buf) { free(src_buf); free(dst_buf); return -1; }

    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;

    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)src_gpa,
                              src_len, sg, 1, PROT_READ);
    int rc = (nr < 0) ? -1 : vfu_sgl_read(vfu_ctx, sg, (size_t)nr, src_buf,
                                            VFU_SGL_DIRECT_ACCESS);
    if (rc != 0) {
        sdxi_error("elemwise: failed to read src (gpa=0x%llx len=%u)",
                   (unsigned long long)src_gpa, src_len);
        free(src_buf); free(dst_buf); return -1;
    }

    /* Decode scalar to float for arithmetic; scalar_bits is dtype-native. */
    float scalar_f = 0.0f;
    switch (dtype) {
    case SDXI_DTYPE_FP32: { float f; memcpy(&f, &scalar_bits, 4); scalar_f = f; break; }
    case SDXI_DTYPE_FP16: scalar_f = fp16_to_float((uint16_t)scalar_bits); break;
    case SDXI_DTYPE_BF16: scalar_f = bf16_to_float((uint16_t)scalar_bits); break;
    case SDXI_DTYPE_INT8: scalar_f = (float)(int8_t)(scalar_bits & 0xFFU); break;
    case SDXI_DTYPE_INT32: { int32_t i; memcpy(&i, &scalar_bits, 4); scalar_f = (float)i; break; }
    case SDXI_DTYPE_FP8_E4M3: scalar_f = e4m3_to_float((uint8_t)scalar_bits); break;
    case SDXI_DTYPE_FP8_E5M2: scalar_f = e5m2_to_float((uint8_t)scalar_bits); break;
    case SDXI_DTYPE_INT4: scalar_f = (float)int4_from_nibble((uint8_t)scalar_bits); break;
    }

    float accum = (op == SDXI_ELEMOP_REDUCE_MIN) ? 1e38f : -1e38f;

    for (size_t i = 0; i < n_elems; i++) {
        /* Decode element i to float. */
        float elem = 0.0f;
        switch (dtype) {
        case SDXI_DTYPE_FP32: { float f; memcpy(&f, src_buf + i * 4, 4); elem = f; break; }
        case SDXI_DTYPE_FP16: { uint16_t h; memcpy(&h, src_buf + i * 2, 2); elem = fp16_to_float(h); break; }
        case SDXI_DTYPE_BF16: { uint16_t b; memcpy(&b, src_buf + i * 2, 2); elem = bf16_to_float(b); break; }
        case SDXI_DTYPE_INT8:  elem = (float)(int8_t)src_buf[i]; break;
        case SDXI_DTYPE_INT32: { int32_t v; memcpy(&v, src_buf + i * 4, 4); elem = (float)v; break; }
        case SDXI_DTYPE_FP8_E4M3: elem = e4m3_to_float(src_buf[i]); break;
        case SDXI_DTYPE_FP8_E5M2: elem = e5m2_to_float(src_buf[i]); break;
        case SDXI_DTYPE_INT4: {
            uint8_t byte = src_buf[i / 2];
            elem = (float)int4_from_nibble((i & 1) ? (byte >> 4) : (byte & 0xFU));
            break;
        }
        }

        /* Apply operation. */
        float result = elem;
        switch (op) {
        case SDXI_ELEMOP_ADD: result = elem + scalar_f; break;
        case SDXI_ELEMOP_MIN: result = elem < scalar_f ? elem : scalar_f; break;
        case SDXI_ELEMOP_MAX: result = elem > scalar_f ? elem : scalar_f; break;
        case SDXI_ELEMOP_REDUCE_MIN: accum = elem < accum ? elem : accum; continue;
        case SDXI_ELEMOP_REDUCE_MAX: accum = elem > accum ? elem : accum; continue;
        }

        /* Encode result back to dtype and store. */
        switch (dtype) {
        case SDXI_DTYPE_FP32: { memcpy(dst_buf + i * 4, &result, 4); break; }
        case SDXI_DTYPE_FP16: { uint16_t h = fp16_from_float(result); memcpy(dst_buf + i * 2, &h, 2); break; }
        case SDXI_DTYPE_BF16: { uint16_t b = bf16_from_float(result); memcpy(dst_buf + i * 2, &b, 2); break; }
        case SDXI_DTYPE_INT8: dst_buf[i] = (uint8_t)(int8_t)(int)result; break;
        case SDXI_DTYPE_INT32: { int32_t v = (int32_t)result; memcpy(dst_buf + i * 4, &v, 4); break; }
        case SDXI_DTYPE_FP8_E4M3: dst_buf[i] = e4m3_from_float(result); break;
        case SDXI_DTYPE_FP8_E5M2: dst_buf[i] = e5m2_from_float(result); break;
        case SDXI_DTYPE_INT4: {
            uint8_t nib = int4_to_nibble((int)result);
            if (i & 1) dst_buf[i / 2] = (uint8_t)((dst_buf[i / 2] & 0x0FU) | (nib << 4));
            else        dst_buf[i / 2] = (uint8_t)((dst_buf[i / 2] & 0xF0U) | (nib & 0x0FU));
            break;
        }
        }
    }

    /* For reduce ops, store the accumulated result as a single element. */
    if (is_reduce) {
        switch (dtype) {
        case SDXI_DTYPE_FP32: { memcpy(dst_buf, &accum, 4); break; }
        case SDXI_DTYPE_FP16: { uint16_t h = fp16_from_float(accum); memcpy(dst_buf, &h, 2); break; }
        case SDXI_DTYPE_BF16: { uint16_t b = bf16_from_float(accum); memcpy(dst_buf, &b, 2); break; }
        case SDXI_DTYPE_INT8: dst_buf[0] = (uint8_t)(int8_t)(int)accum; break;
        case SDXI_DTYPE_INT32: { int32_t v = (int32_t)accum; memcpy(dst_buf, &v, 4); break; }
        case SDXI_DTYPE_FP8_E4M3: dst_buf[0] = e4m3_from_float(accum); break;
        case SDXI_DTYPE_FP8_E5M2: dst_buf[0] = e5m2_from_float(accum); break;
        case SDXI_DTYPE_INT4: dst_buf[0] = int4_to_nibble((int)accum); break;
        }
    }

    free(src_buf);

    nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)dst_gpa,
                          dst_len, sg, 1, PROT_WRITE);
    rc = (nr < 0) ? -1 : vfu_sgl_write(vfu_ctx, sg, (size_t)nr, dst_buf,
                                         VFU_SGL_DIRECT_ACCESS);
    free(dst_buf);
    if (rc != 0) {
        sdxi_error("elemwise: failed to write dst (gpa=0x%llx len=%zu)",
                   (unsigned long long)dst_gpa, dst_len);
        return -1;
    }

    sdxi_debug("elemwise: dtype=%u op=%u src_len=%u -> dst_len=%zu", dtype, op, src_len, dst_len);

    if (csb_gpa != 0ULL) {
        struct sdxi_cst_blk csb;
        memset(&csb, 0, sizeof(csb));
        uint64_t result = (uint64_t)dst_len;
        memcpy((uint8_t *)&csb + SDXI_CSB_RESULT_LEN_OFF, &result, sizeof(result));
        write_guest(vfu_ctx, csb_gpa, &csb, sizeof(csb));
        vfu_irq_trigger(vfu_ctx, 0);
    }

    return 0;
}

/* ------------------------------------------------------------------ */
/* Poll                                                                */
/* ------------------------------------------------------------------ */

static int process_queue(sdxi_engine_t *eng, vfu_ctx_t *vfu_ctx,
                         sdxi_queue_t *q)
{
    int processed = 0;

    while (queue_avail(q) > 0 && !cq_full(q)) {
        uint64_t sqe_gpa = q->sq_base +
            (uint64_t)q->sq_head * SDXI_SQE_SIZE;

        /* Read SQE from guest memory */
        uint8_t sg_buf[dma_sg_size()];
        dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;

        int nr = vfu_addr_to_sgl(vfu_ctx,
                     (vfu_dma_addr_t)(uintptr_t)sqe_gpa,
                     SDXI_SQE_SIZE, sg, 1, PROT_READ);
        if (nr < 0) {
            sdxi_error("queue %u: failed to map SQE at gpa=0x%llx: %s",
                       q->id, (unsigned long long)sqe_gpa, strerror(errno));
            break;
        }

        struct sdxi_sqe sqe;
        int rc = vfu_sgl_read(vfu_ctx, sg, (size_t)nr,
                              &sqe, VFU_SGL_DIRECT_ACCESS);
        if (rc != 0) {
            sdxi_error("queue %u: vfu_sgl_read SQE failed: %s",
                       q->id, strerror(errno));
            break;
        }

        /* Advance SQ head to mark descriptor as consumed */
        q->sq_head = (q->sq_head + 1U) % q->sq_size;

        /* Execute */
        uint8_t status;
        switch (sqe.opcode) {
        case SDXI_OP_MEMCPY:
            status = (uint8_t)exec_memcpy(vfu_ctx, &sqe);
            break;
        case SDXI_OP_MEMSET:
            status = (uint8_t)exec_memset(vfu_ctx, &sqe);
            break;
        case SDXI_OP_FENCE:
            /* Ordering fence: no data movement; always completes OK */
            status = SDXI_CQE_STATUS_OK;
            sdxi_debug("queue %u fence cid=%u", q->id, sqe.cid);
            break;
        default:
            sdxi_warn("queue %u: unknown opcode 0x%x cid=%u",
                      q->id, sqe.opcode, sqe.cid);
            status = SDXI_CQE_STATUS_INVAL;
            break;
        }

        post_cqe(eng, vfu_ctx, q, sqe.cid, status);
        processed++;
    }

    return processed;
}

int sdxi_engine_poll(sdxi_engine_t *eng, vfu_ctx_t *vfu_ctx)
{
    int total = 0;
    for (uint32_t i = 0; i < SDXI_MAX_QUEUES; i++) {
        sdxi_queue_t *q = &eng->queues[i];
        if (!q->enabled || q->sq_size == 0 || q->cq_size == 0)
            continue;
        if (queue_avail(q) == 0)
            continue;
        total += process_queue(eng, vfu_ctx, q);
    }
    return total;
}

/* ------------------------------------------------------------------ */
/* v3 context-table DMA processing                                     */
/* ------------------------------------------------------------------ */

static int read_guest(vfu_ctx_t *vfu_ctx, uint64_t gpa, void *buf, size_t len)
{
    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;
    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)gpa,
                              len, sg, 1, PROT_READ);
    if (nr < 0)
        return -1;
    return vfu_sgl_read(vfu_ctx, sg, (size_t)nr, buf, VFU_SGL_DIRECT_ACCESS);
}

static int write_guest(vfu_ctx_t *vfu_ctx, uint64_t gpa, const void *buf, size_t len)
{
    uint8_t sg_buf[dma_sg_size()];
    dma_sg_t *sg = (dma_sg_t *)(void *)sg_buf;
    int nr = vfu_addr_to_sgl(vfu_ctx, (vfu_dma_addr_t)(uintptr_t)gpa,
                              len, sg, 1, PROT_WRITE);
    if (nr < 0)
        return -1;
    return vfu_sgl_write(vfu_ctx, sg, (size_t)nr, (void *)(uintptr_t)buf, VFU_SGL_DIRECT_ACCESS);
}

void sdxi_engine_process_cxt(vfu_ctx_t *vfu_ctx, uint64_t l2_gpa,
                              uint32_t cid, uint32_t new_write_idx,
                              sdxi_cxt_t *cxt_arr, uint32_t num_cxt)
{
    if (cid >= num_cxt) {
        sdxi_warn("cxt %u: out of range (num_cxt=%u)", cid, num_cxt);
        return;
    }
    sdxi_cxt_t *cxt = &cxt_arr[cid];

    /* Admin context auto-activates on first doorbell. */
    if (cid == SDXI_ADMIN_CXT_ID && cxt->state == SDXI_CXT_IDLE) {
        cxt->state = SDXI_CXT_ACTIVE;
        sdxi_debug("admin context auto-activated");
    }

    if (cxt->state != SDXI_CXT_ACTIVE) {
        sdxi_warn("cxt %u: not active (state=%d), doorbell ignored", cid, cxt->state);
        return;
    }

    /* Walk L2 -> L1 -> cxt_ctl to reach the descriptor ring. */
    uint64_t l2_entry;
    if (read_guest(vfu_ctx, l2_gpa + (cid / SDXI_L1_TABLE_ENTRIES) * 8U,
                   &l2_entry, 8U) != 0) {
        sdxi_error("cxt %u: failed to read L2 entry", cid);
        return;
    }
    if (!(l2_entry & SDXI_CXT_L2_ENT_VL)) {
        sdxi_warn("cxt %u: L2 entry not valid (0x%016llx)", cid,
                  (unsigned long long)l2_entry);
        return;
    }
    uint64_t l1_gpa = l2_entry & SDXI_CXT_L2_PTR_MASK;

    uint64_t l1_entry;
    if (read_guest(vfu_ctx,
                   l1_gpa + (cid % SDXI_L1_TABLE_ENTRIES) * SDXI_CXT_L1_ENTRY_SIZE,
                   &l1_entry, 8U) != 0) {
        sdxi_error("cxt %u: failed to read L1 entry", cid);
        return;
    }
    if (!(l1_entry & SDXI_CXT_L1_ENT_VL)) {
        sdxi_warn("cxt %u: L1 entry not valid (0x%016llx)", cid,
                  (unsigned long long)l1_entry);
        return;
    }
    uint64_t cxt_ctl_gpa = l1_entry & SDXI_CXT_L1_PTR_MASK;

    uint8_t cxt_ctl[64];
    if (read_guest(vfu_ctx, cxt_ctl_gpa, cxt_ctl, sizeof(cxt_ctl)) != 0) {
        sdxi_error("cxt %u: failed to read cxt_ctl at 0x%llx", cid,
                   (unsigned long long)cxt_ctl_gpa);
        return;
    }
    uint64_t ds_ring_ptr, ds_ring_sz;
    memcpy(&ds_ring_ptr, cxt_ctl + 0, 8U);
    memcpy(&ds_ring_sz,  cxt_ctl + 8, 8U);

    if (!(ds_ring_ptr & SDXI_CXT_CTL_DS_RING_VL)) {
        sdxi_warn("cxt %u: ds_ring_ptr not valid (0x%016llx)", cid,
                  (unsigned long long)ds_ring_ptr);
        return;
    }
    if (ds_ring_sz == 0) {
        sdxi_warn("cxt %u: ds_ring_sz is 0", cid);
        return;
    }
    uint64_t ds_ring_gpa = ds_ring_ptr & SDXI_CXT_CTL_DS_RING_PTR_MASK;

    while (cxt->read_idx != new_write_idx) {
        uint64_t dsc_gpa = ds_ring_gpa +
            (uint64_t)(cxt->read_idx % (uint32_t)ds_ring_sz) * 64U;

        uint8_t dsc[64];
        if (read_guest(vfu_ctx, dsc_gpa, dsc, 64U) != 0) {
            sdxi_error("cxt %u: failed to read descriptor at 0x%llx", cid,
                       (unsigned long long)dsc_gpa);
            break;
        }

        uint32_t opcode;
        memcpy(&opcode, dsc + 0, 4U);

        if (!(opcode & SDXI_DSC_VL)) {
            sdxi_warn("cxt %u: descriptor at 0x%llx VL=0, stopping", cid,
                      (unsigned long long)dsc_gpa);
            break;
        }

        uint32_t type    = SDXI_DSC_GET_TYPE(opcode);
        uint32_t subtype = SDXI_DSC_GET_SUBTYPE(opcode);
        int      csb_ok  = 0;
        bool     write_csb = true;
        uint64_t csb_ptr_val, csb_gpa;
        memcpy(&csb_ptr_val, dsc + SDXI_COPY_CSBPTR_OFF, 8U);
        csb_gpa = (csb_ptr_val >> SDXI_DSC_CSBPTR_SHIFT) << SDXI_DSC_CSBPTR_SHIFT;

        switch (type) {
        case SDXI_DSC_OP_TYPE_DMAB:
            switch (subtype) {
            case SDXI_DSC_OP_SUBTYPE_NOP:
                sdxi_debug("cxt %u DMAB NOP", cid);
                cxt->num_nop++;
                break;
            case SDXI_DSC_OP_SUBTYPE_SYNC:
                /* Ordering fence: single-threaded emulator provides this for free. */
                sdxi_debug("cxt %u DMAB SYNC", cid);
                cxt->num_sync++;
                break;
            case SDXI_DSC_OP_SUBTYPE_COPY: {
                uint32_t size_m1;
                uint64_t addr0, addr1;
                memcpy(&size_m1, dsc + SDXI_COPY_SIZE_OFF,  4U);
                memcpy(&addr0,   dsc + SDXI_COPY_ADDR0_OFF, 8U);
                memcpy(&addr1,   dsc + SDXI_COPY_ADDR1_OFF, 8U);
                uint32_t copy_len = size_m1 + 1U;
                sdxi_debug("cxt %u COPY: 0x%llx -> 0x%llx len=%u",
                           cid, (unsigned long long)addr0,
                           (unsigned long long)addr1, copy_len);

                uint8_t src_sg_buf[SDXI_MAX_SG_SEGS * dma_sg_size()];
                uint8_t dst_sg_buf[SDXI_MAX_SG_SEGS * dma_sg_size()];
                dma_sg_t *src_sgl = (dma_sg_t *)(void *)src_sg_buf;
                dma_sg_t *dst_sgl = (dma_sg_t *)(void *)dst_sg_buf;

                int src_nr = vfu_addr_to_sgl(vfu_ctx,
                                 (vfu_dma_addr_t)(uintptr_t)addr0,
                                 (size_t)copy_len, src_sgl, SDXI_MAX_SG_SEGS,
                                 PROT_READ);
                int dst_nr = (src_nr >= 0) ? vfu_addr_to_sgl(vfu_ctx,
                                 (vfu_dma_addr_t)(uintptr_t)addr1,
                                 (size_t)copy_len, dst_sgl, SDXI_MAX_SG_SEGS,
                                 PROT_WRITE) : -1;

                if (src_nr < 0 || dst_nr < 0) {
                    sdxi_error("cxt %u: sgl map failed (src=%d dst=%d): %s",
                               cid, src_nr, dst_nr, strerror(errno));
                    csb_ok = -1;
                } else if (sgl_memcpy(vfu_ctx, src_sgl, src_nr, dst_sgl, dst_nr,
                                      (size_t)copy_len) != 0) {
                    sdxi_error("cxt %u: sgl_memcpy failed: %s", cid, strerror(errno));
                    csb_ok = -1;
                } else {
                    cxt->bytes_moved += copy_len;
                    cxt->num_copy++;
                }
                break;
            }
            case SDXI_DSC_OP_SUBTYPE_COMPRESS: {
                uint32_t src_len, dst_cap, flags;
                uint64_t src_addr, dst_addr;
                memcpy(&src_len,  dsc + SDXI_COMP_SRC_LEN_OFF,  4U);
                memcpy(&dst_cap,  dsc + SDXI_COMP_DST_CAP_OFF,  4U);
                memcpy(&flags,    dsc + SDXI_COMP_FLAGS_OFF,     4U);
                memcpy(&src_addr, dsc + SDXI_COMP_SRC_ADDR_OFF, 8U);
                memcpy(&dst_addr, dsc + SDXI_COMP_DST_ADDR_OFF, 8U);
                uint32_t level = flags & 0xFFU;
                uint32_t algo  = (flags >> 8) & 0xFFU;
                sdxi_debug("cxt %u COMPRESS: 0x%llx len=%u -> 0x%llx cap=%u level=%u algo=%u",
                           cid, (unsigned long long)src_addr, src_len,
                           (unsigned long long)dst_addr, dst_cap, level, algo);
                ssize_t out = exec_compress(vfu_ctx, src_addr, src_len,
                                            dst_addr, dst_cap,
                                            (int)level, algo, csb_gpa);
                if (out < 0) {
                    csb_ok = -1;
                } else {
                    cxt->bytes_compressed  += src_len;
                    cxt->num_compress++;
                    write_csb = false;   /* exec_compress already wrote the CSB */
                }
                break;
            }
            case SDXI_DSC_OP_SUBTYPE_DECOMPRESS: {
                uint32_t src_len, dst_cap, dflags;
                uint64_t src_addr, dst_addr;
                memcpy(&src_len,  dsc + SDXI_COMP_SRC_LEN_OFF,  4U);
                memcpy(&dst_cap,  dsc + SDXI_COMP_DST_CAP_OFF,  4U);
                memcpy(&dflags,   dsc + SDXI_COMP_FLAGS_OFF,     4U);
                memcpy(&src_addr, dsc + SDXI_COMP_SRC_ADDR_OFF, 8U);
                memcpy(&dst_addr, dsc + SDXI_COMP_DST_ADDR_OFF, 8U);
                uint32_t algo = (dflags >> 8) & 0xFFU;
                sdxi_debug("cxt %u DECOMPRESS: 0x%llx len=%u -> 0x%llx cap=%u algo=%u",
                           cid, (unsigned long long)src_addr, src_len,
                           (unsigned long long)dst_addr, dst_cap, algo);
                ssize_t out = exec_decompress(vfu_ctx, src_addr, src_len,
                                              dst_addr, dst_cap, algo, csb_gpa);
                if (out < 0) {
                    csb_ok = -1;
                } else {
                    cxt->bytes_decompressed += (uint64_t)out;
                    cxt->num_decompress++;
                    write_csb = false;   /* exec_decompress already wrote the CSB */
                }
                break;
            }
            default:
                sdxi_warn("cxt %u: unknown DMAB subtype 0x%x", cid, subtype);
                csb_ok = -1;
                break;
            }
            break;

        case SDXI_DSC_OP_TYPE_ADMIN:
            switch (subtype) {
            case SDXI_DSC_OP_SUBTYPE_NOP:
                sdxi_debug("cxt %u ADMIN NOP", cid);
                cxt->num_nop++;
                break;
            case SDXI_DSC_OP_SUBTYPE_CXT_START: {
                uint32_t target_cid = 0;
                memcpy(&target_cid, dsc + 4, 4U);
                if (target_cid < num_cxt && target_cid != SDXI_ADMIN_CXT_ID) {
                    cxt_arr[target_cid].state = SDXI_CXT_ACTIVE;
                    cxt->num_cxt_start++;
                    sdxi_debug("cxt %u CXT_START target=%u", cid, target_cid);
                } else {
                    sdxi_warn("cxt %u: CXT_START invalid target %u", cid, target_cid);
                    csb_ok = -1;
                }
                break;
            }
            case SDXI_DSC_OP_SUBTYPE_CXT_STOP: {
                uint32_t target_cid = 0;
                memcpy(&target_cid, dsc + 4, 4U);
                if (target_cid < num_cxt && target_cid != SDXI_ADMIN_CXT_ID) {
                    cxt_arr[target_cid].state = SDXI_CXT_STOPPED;
                    cxt->num_cxt_stop++;
                    sdxi_debug("cxt %u CXT_STOP target=%u", cid, target_cid);
                } else {
                    sdxi_warn("cxt %u: CXT_STOP invalid target %u", cid, target_cid);
                    csb_ok = -1;
                }
                break;
            }
            default:
                sdxi_warn("cxt %u: unknown ADMIN subtype 0x%x", cid, subtype);
                csb_ok = -1;
                break;
            }
            break;

        case SDXI_DSC_OP_TYPE_INTR: {
            uint16_t vec = 0;
            memcpy(&vec, dsc + SDXI_INTR_VEC_OFF, sizeof(vec));
            sdxi_debug("cxt %u INTR vector=%u", cid, vec);
            int irq_rc = vfu_irq_trigger(vfu_ctx, vec);
            if (irq_rc != 0)
                sdxi_warn("cxt %u: MSI-X trigger vector=%u failed: %s",
                          cid, vec, strerror(errno));
            cxt->num_intr++;
            write_csb = false;   /* INTR descriptors carry no CSB pointer */
            break;
        }

        case SDXI_DSC_OP_TYPE_COMPUTE:
            switch (subtype) {
            case SDXI_DSC_OP_SUBTYPE_ELEMWISE: {
                uint32_t src_len, dtype, op;
                uint64_t src_addr, dst_addr, scalar;
                memcpy(&src_len,  dsc + SDXI_ELEM_SRC_LEN_OFF,  4U);
                memcpy(&dtype,    dsc + SDXI_ELEM_DTYPE_OFF,     4U);
                memcpy(&op,       dsc + SDXI_ELEM_OP_OFF,        4U);
                memcpy(&src_addr, dsc + SDXI_ELEM_SRC_ADDR_OFF,  8U);
                memcpy(&dst_addr, dsc + SDXI_ELEM_DST_ADDR_OFF,  8U);
                memcpy(&scalar,   dsc + SDXI_ELEM_SCALAR_OFF,    8U);
                int er = exec_elemwise(vfu_ctx, src_addr, src_len, dst_addr,
                                       dtype, op, scalar, csb_gpa);
                if (er < 0) { csb_ok = -1; }
                else { cxt->num_elemwise++; cxt->bytes_elemwise += src_len; write_csb = false; }
                break;
            }
            default:
                sdxi_warn("cxt %u: unknown COMPUTE subtype 0x%x", cid, subtype);
                csb_ok = -1;
                break;
            }
            break;

        default:
            sdxi_warn("cxt %u: unknown descriptor type 0x%x subtype 0x%x",
                      cid, type, subtype);
            csb_ok = -1;
            break;
        }

        if (csb_ok < 0)
            cxt->num_errors++;

        /* Write CSB: signal=0 means success; set error flag on failure. */
        if (write_csb && csb_gpa != 0ULL) {
            struct sdxi_cst_blk csb;
            memset(&csb, 0, sizeof(csb));
            if (csb_ok < 0)
                csb.flags = SDXI_CST_BLK_ER_BIT;
            write_guest(vfu_ctx, csb_gpa, &csb, sizeof(csb));
            vfu_irq_trigger(vfu_ctx, 0);
        }

        cxt->read_idx++;
    }
}
