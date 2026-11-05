/* Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *
 * rocm-sdxi: vfio-user server emulating the AMD SDXI (SNIA Data Exchange
 * Interface) copy-engine for the AMD CPU SoC.
 *
 * The emulated device presents itself as a PCIe peripheral with:
 *   BAR0  64 KiB  MMIO control/status registers + MSI-X table/PBA
 *   BAR2   4 MiB  Doorbell aperture (4 KiB per queue)
 *
 * Architecture follows rocm-ernic/src/rocm_ernic_server.c:
 *   vfu_create_ctx -> PCI init -> BAR setup -> DMA/reset callbacks
 *   -> vfu_realize_ctx -> attach/serve loop -> engine poll
 */

#include <errno.h>
#include <getopt.h>
#include <signal.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include <vfio-user/libvfio-user.h>
#include <vfio-user/pci_defs.h>

#include "sdxi_engine.h"
#include "sdxi_log.h"
#include "sdxi_regs.h"

/* ------------------------------------------------------------------ */
/* Build-time version string embedded by CMake                        */
/* ------------------------------------------------------------------ */
#ifndef GIT_SHA_SHORT
#define GIT_SHA_SHORT "unknown"
#endif

#define SDXI_DEFAULT_SOCKET "/tmp/vfio-user-rocm-sdxi.sock"

/* ------------------------------------------------------------------ */
/* Device state                                                        */
/* ------------------------------------------------------------------ */

typedef struct sdxi_dev {
    vfu_ctx_t    *vfu_ctx;

    /* BAR0 register shadow -- written by BAR0 access callbacks */
    uint32_t      reg_ctrl;
    uint32_t      reg_status;
    uint32_t      reg_q_select;   /* currently selected queue index */

    /* v3 driver MMIO register shadows (64-bit readq/writeq) */
    uint64_t      mmio_ctl0;   /* CTL0: FN_GSR in bits[1:0] */
    uint64_t      mmio_ctl2;   /* CTL2: config written by driver at probe */
    uint64_t      mmio_sts0;   /* STS0: FN_GSV in bits[2:0]; mirrors CTL0[1:0] */
    uint64_t      mmio_l2_ptr; /* CXT_L2: context-table L2 pointer from driver */
    sdxi_cxt_t    cxt[SDXI_MAX_CXT]; /* per-context state (state, read_idx, bytes_moved) */
    uint32_t      ctr_select;         /* CTR_SELECT: context selected for counter reads */

    /* Descriptor-queue engine */
    sdxi_engine_t engine;

    bool          dma_registered;  /* true once the guest has mapped DMA */
} sdxi_dev_t;

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */

static volatile bool g_shutdown_requested;

static void signal_handler(int sig)
{
    (void)sig;
    g_shutdown_requested = true;
}

/* ------------------------------------------------------------------ */
/* BAR0 register access                                                */
/* ------------------------------------------------------------------ */

/*
 * Return the register value for the currently selected queue.
 * reg is a SDXI_REG_Q_* or SDXI_REG_CQ_* offset.
 */
static uint32_t read_queue_reg(sdxi_dev_t *dev, uint16_t reg)
{
    uint32_t qid = dev->reg_q_select;
    if (qid >= SDXI_MAX_QUEUES)
        return 0U;
    const sdxi_queue_t *q = &dev->engine.queues[qid];

    switch (reg) {
    case SDXI_REG_Q_SELECT:  return qid;
    case SDXI_REG_Q_CTRL:
        return (q->enabled    ? SDXI_QCTRL_ENABLE   : 0U) |
               (q->intr_enabled ? SDXI_QCTRL_INTR_EN : 0U);
    case SDXI_REG_Q_STATUS:
        return q->enabled ? SDXI_QSTAT_READY : 0U;
    case SDXI_REG_Q_SIZE:    return q->sq_size;
    case SDXI_REG_Q_HEAD:    return q->sq_head;
    case SDXI_REG_Q_TAIL:    return q->sq_tail;
    case SDXI_REG_Q_BASE_LO: return (uint32_t)(q->sq_base & 0xFFFFFFFFU);
    case SDXI_REG_Q_BASE_HI: return (uint32_t)(q->sq_base >> 32U);
    case SDXI_REG_CQ_SIZE:   return q->cq_size;
    case SDXI_REG_CQ_HEAD:   return q->cq_head;
    case SDXI_REG_CQ_TAIL:   return q->cq_tail;
    case SDXI_REG_CQ_BASE_LO:return (uint32_t)(q->cq_base & 0xFFFFFFFFU);
    case SDXI_REG_CQ_BASE_HI:return (uint32_t)(q->cq_base >> 32U);
    case SDXI_REG_Q_INTR_VEC:return q->intr_vec;
    default:                  return 0U;
    }
}

static void write_queue_reg(sdxi_dev_t *dev, uint16_t reg, uint32_t val)
{
    uint32_t qid = dev->reg_q_select;
    if (qid >= SDXI_MAX_QUEUES)
        return;
    sdxi_queue_t *q = &dev->engine.queues[qid];

    switch (reg) {
    case SDXI_REG_Q_CTRL:
        if (val & SDXI_QCTRL_RESET) {
            sdxi_engine_queue_reset(&dev->engine, qid);
            sdxi_info("queue %u reset", qid);
            return;
        }
        q->enabled      = (val & SDXI_QCTRL_ENABLE)  != 0U;
        q->intr_enabled = (val & SDXI_QCTRL_INTR_EN) != 0U;
        sdxi_debug("queue %u ctrl: enabled=%d intr=%d",
                   qid, q->enabled, q->intr_enabled);
        break;
    case SDXI_REG_Q_SIZE:
        if (!q->enabled)
            q->sq_size = val < SDXI_MAX_QUEUE_DEPTH ? val : SDXI_MAX_QUEUE_DEPTH;
        break;
    case SDXI_REG_Q_TAIL:
        /* BAR0 tail write acts as a doorbell (alternative to BAR2) */
        sdxi_engine_doorbell(&dev->engine, qid, val);
        break;
    case SDXI_REG_Q_BASE_LO:
        q->sq_base = (q->sq_base & 0xFFFFFFFF00000000ULL) | val;
        break;
    case SDXI_REG_Q_BASE_HI:
        q->sq_base = (q->sq_base & 0x00000000FFFFFFFFULL) |
                     ((uint64_t)val << 32U);
        break;
    case SDXI_REG_CQ_SIZE:
        if (!q->enabled)
            q->cq_size = val < SDXI_MAX_QUEUE_DEPTH ? val : SDXI_MAX_QUEUE_DEPTH;
        break;
    case SDXI_REG_CQ_HEAD:
        q->cq_head = val % (q->cq_size ? q->cq_size : 1U);
        break;
    case SDXI_REG_CQ_BASE_LO:
        q->cq_base = (q->cq_base & 0xFFFFFFFF00000000ULL) | val;
        break;
    case SDXI_REG_CQ_BASE_HI:
        q->cq_base = (q->cq_base & 0x00000000FFFFFFFFULL) |
                     ((uint64_t)val << 32U);
        break;
    case SDXI_REG_Q_INTR_VEC:
        if (val < SDXI_MSIX_NUM_VECTORS)
            q->intr_vec = val;
        break;
    default:
        break;
    }
}

/*
 * BAR0 access callback (called by libvfio-user for every guest MMIO
 * read/write in the register region).
 */
static ssize_t bar0_access(vfu_ctx_t *vfu_ctx, char *buf, size_t count,
                            loff_t offset, bool is_write)
{
    sdxi_dev_t *dev = vfu_get_private(vfu_ctx);

    /*
     * 8-byte (readq/writeq) path — v3 kernel driver MMIO registers.
     * The Python test suite uses 4-byte accesses and hits the path below;
     * the kernel driver uses 8-byte accesses and is handled here.
     */
    if (count == 8U && (offset & 7U) == 0) {
        uint32_t reg8 = (uint32_t)(offset & 0x1FFFFU);
        if (is_write) {
            uint64_t val;
            memcpy(&val, buf, 8U);
            switch (reg8) {
            case SDXI_MMIO_CTL0: {
                /* Map FN_GSR → FN_GSV: direct mirror is wrong.
                 * GSR=ACTIVE(3) must yield GSV=ACTIVE(2), not GSV=STOPG_SF(3). */
                static const uint8_t gsr_to_gsv[4] = {
                    SDXI_GSV_STOP,    /* GSR_RESET(0)   -> GSV_STOP(0)    */
                    SDXI_GSV_STOPG_SF,/* GSR_STOP_SF(1) -> GSV_STOPG_SF(3)*/
                    SDXI_GSV_STOPG_HD,/* GSR_STOP_HD(2) -> GSV_STOPG_HD(4)*/
                    SDXI_GSV_ACTIVE,  /* GSR_ACTIVE(3)  -> GSV_ACTIVE(2)  */
                };
                dev->mmio_ctl0 = val;
                uint64_t gsr = val & SDXI_MMIO_CTL0_FN_GSR_MASK;
                dev->mmio_sts0 = (dev->mmio_sts0 & ~SDXI_MMIO_STS0_FN_GSV_MASK) |
                                 gsr_to_gsv[gsr];
                sdxi_debug("CTL0 wr 0x%016llx gsr=%llu sts0=0x%016llx",
                           (unsigned long long)val, (unsigned long long)gsr,
                           (unsigned long long)dev->mmio_sts0);
                break;
            }
            case SDXI_MMIO_CTL2:
                dev->mmio_ctl2 = val;
                sdxi_debug("CTL2 wr 0x%016llx", (unsigned long long)val);
                break;
            case SDXI_MMIO_CXT_L2:
                dev->mmio_l2_ptr = val;
                sdxi_debug("CXT_L2 wr 0x%016llx", (unsigned long long)val);
                break;
            case SDXI_MMIO_CTR_SELECT:
                dev->ctr_select = (uint32_t)val;
                break;
            default:
                sdxi_debug("bar0: 8b write offset=0x%x ignored", reg8);
                break;
            }
        } else {
            uint64_t val = 0ULL;
            switch (reg8) {
            case SDXI_MMIO_CTL0:    val = dev->mmio_ctl0;        break;
            case SDXI_MMIO_CTL2:    val = dev->mmio_ctl2;        break;
            case SDXI_MMIO_STS0:    val = dev->mmio_sts0;        break;
            case SDXI_MMIO_CAP0:    val = SDXI_MMIO_CAP0_VAL;    break;
            case SDXI_MMIO_CAP1:    val = SDXI_MMIO_CAP1_VAL;    break;
            case SDXI_MMIO_VERSION: val = SDXI_MMIO_VERSION_VAL;  break;
            case SDXI_MMIO_CXT_L2:    val = dev->mmio_l2_ptr;                              break;
            case SDXI_MMIO_CTR_SELECT: val = dev->ctr_select; break;
            case SDXI_MMIO_CTR_VAL: {
                uint32_t cid  = dev->ctr_select & SDXI_CTR_CXT_MASK;
                uint32_t stat = (dev->ctr_select & SDXI_CTR_STAT_MASK) >> SDXI_CTR_STAT_SHIFT;
                if (cid >= SDXI_MAX_CXT) break;
                const sdxi_cxt_t *c = &dev->cxt[cid];
                switch (stat) {
                case SDXI_CTR_STAT_BYTES_MOVED:        val = c->bytes_moved;        break;
                case SDXI_CTR_STAT_NUM_COPY:           val = c->num_copy;           break;
                case SDXI_CTR_STAT_NUM_ERRORS:         val = c->num_errors;         break;
                case SDXI_CTR_STAT_NUM_INTR:           val = c->num_intr;           break;
                case SDXI_CTR_STAT_NUM_CXT_START:      val = c->num_cxt_start;      break;
                case SDXI_CTR_STAT_NUM_CXT_STOP:       val = c->num_cxt_stop;       break;
                case SDXI_CTR_STAT_NUM_NOP:            val = c->num_nop;            break;
                case SDXI_CTR_STAT_NUM_SYNC:           val = c->num_sync;           break;
                case SDXI_CTR_STAT_BYTES_COMPRESSED:   val = c->bytes_compressed;   break;
                case SDXI_CTR_STAT_BYTES_DECOMPRESSED: val = c->bytes_decompressed; break;
                case SDXI_CTR_STAT_NUM_COMPRESS:       val = c->num_compress;       break;
                case SDXI_CTR_STAT_NUM_DECOMPRESS:     val = c->num_decompress;     break;
                case SDXI_CTR_STAT_NUM_ELEMWISE:       val = c->num_elemwise;       break;
                case SDXI_CTR_STAT_BYTES_ELEMWISE:     val = c->bytes_elemwise;     break;
                default: break;
                }
                break;
            }
            default:
                sdxi_debug("bar0: 8b read offset=0x%x -> 0", reg8);
                break;
            }
            memcpy(buf, &val, 8U);
        }
        return (ssize_t)count;
    }

    /* Only 32-bit aligned, 4-byte accesses are supported for the old register map. */
    if (count != 4U || (offset & 3U) != 0) {
        sdxi_warn("bar0: unsupported access size=%zu offset=0x%llx",
                  count, (unsigned long long)offset);
        errno = EINVAL;
        return -1;
    }

    uint16_t reg = (uint16_t)(offset & 0xFFFFU);

    if (is_write) {
        uint32_t val;
        memcpy(&val, buf, 4U);

        switch (reg) {
        case SDXI_REG_CTRL:
            if (val & SDXI_CTRL_RESET) {
                sdxi_info("device software reset");
                sdxi_engine_init(&dev->engine);
                dev->reg_q_select = 0U;
                val &= ~(uint32_t)SDXI_CTRL_RESET;
            }
            dev->reg_ctrl = val;
            break;
        case SDXI_REG_Q_SELECT:
            if (val < SDXI_MAX_QUEUES)
                dev->reg_q_select = val;
            break;
        default:
            write_queue_reg(dev, reg, val);
            break;
        }
        return (ssize_t)count;
    }

    /* Read */
    uint32_t val = 0U;
    switch (reg) {
    case SDXI_REG_DEVID:      val = SDXI_DEVID_MAGIC;                break;
    case SDXI_REG_VERSION:    val = SDXI_VERSION_VAL;                break;
    case SDXI_REG_CAP:        val = SDXI_CAP_MEMCPY | SDXI_CAP_MEMSET |
                                    SDXI_CAP_FENCE;                   break;
    case SDXI_REG_NUM_QUEUES: val = SDXI_MAX_QUEUES;                 break;
    case SDXI_REG_STATUS:     val = dev->reg_status;                 break;
    case SDXI_REG_CTRL:       val = dev->reg_ctrl;                   break;
    default:
        val = read_queue_reg(dev, reg);
        break;
    }

    memcpy(buf, &val, 4U);
    return (ssize_t)count;
}

/* ------------------------------------------------------------------ */
/* BAR2 doorbell access                                                */
/* ------------------------------------------------------------------ */

/*
 * BAR2 write: guest rings a queue doorbell by writing the new SQ tail
 * to offset (queue_id * SDXI_DB_STRIDE) within BAR2.
 */
static ssize_t bar2_access(vfu_ctx_t *vfu_ctx, char *buf, size_t count,
                            loff_t offset, bool is_write)
{
    sdxi_dev_t *dev = vfu_get_private(vfu_ctx);

    if (!is_write) {
        /* Reads to the doorbell aperture return zero */
        memset(buf, 0, count);
        return (ssize_t)count;
    }

    if ((count != 4U && count != 8U) || (offset & 3U) != 0) {
        sdxi_warn("bar2: unsupported doorbell access size=%zu offset=0x%llx",
                  count, (unsigned long long)offset);
        errno = EINVAL;
        return -1;
    }

    uint32_t cid = (uint32_t)(offset / SDXI_DB_STRIDE);
    uint32_t new_write_idx;
    if (count == 8U) {
        uint64_t tmp;
        memcpy(&tmp, buf, 8U);
        new_write_idx = (uint32_t)tmp;
    } else {
        memcpy(&new_write_idx, buf, 4U);
    }

    if (dev->mmio_l2_ptr != 0 && cid < SDXI_MAX_CXT) {
        uint64_t l2_gpa = dev->mmio_l2_ptr & SDXI_CXT_L2_PTR_MASK;
        sdxi_engine_process_cxt(vfu_ctx, l2_gpa, cid, new_write_idx,
                                dev->cxt, SDXI_MAX_CXT);
    } else {
        sdxi_engine_doorbell(&dev->engine, cid, new_write_idx);
    }
    return (ssize_t)count;
}

/* ------------------------------------------------------------------ */
/* DMA callbacks                                                       */
/* ------------------------------------------------------------------ */

static void dma_register_cb(vfu_ctx_t *vfu_ctx,
                             vfu_dma_info_t *info)
{
    (void)vfu_ctx;
    sdxi_info("DMA region registered: gpa=0x%llx len=0x%llx",
              (unsigned long long)(uintptr_t)info->iova.iov_base,
              (unsigned long long)info->iova.iov_len);
}

static void dma_unregister_cb(vfu_ctx_t *vfu_ctx,
                               vfu_dma_info_t *info)
{
    (void)vfu_ctx;
    sdxi_info("DMA region unregistered: gpa=0x%llx len=0x%llx",
              (unsigned long long)(uintptr_t)info->iova.iov_base,
              (unsigned long long)info->iova.iov_len);
}

/* ------------------------------------------------------------------ */
/* Reset callback                                                      */
/* ------------------------------------------------------------------ */

static int device_reset_cb(vfu_ctx_t *vfu_ctx, vfu_reset_type_t type)
{
    sdxi_dev_t *dev = vfu_get_private(vfu_ctx);

    const char *type_str = (type == VFU_RESET_DEVICE)      ? "DEVICE" :
                           (type == VFU_RESET_LOST_CONN)    ? "LOST_CONN" :
                           (type == VFU_RESET_PCI_FLR)      ? "PCI_FLR" :
                                                              "UNKNOWN";
    sdxi_info("reset: type=%s", type_str);

    sdxi_engine_init(&dev->engine);
    dev->reg_ctrl      = 0U;
    dev->reg_status    = SDXI_STATUS_READY;
    dev->reg_q_select  = 0U;
    dev->mmio_ctl0     = 0U;
    dev->mmio_ctl2     = 0U;
    dev->mmio_sts0     = 0U;
    dev->mmio_l2_ptr   = 0U;
    memset(dev->cxt, 0, sizeof(dev->cxt));
    dev->ctr_select    = 0U;

    return 0;
}

/* ------------------------------------------------------------------ */
/* Device setup                                                        */
/* ------------------------------------------------------------------ */

static vfu_ctx_t *setup_device(sdxi_dev_t *dev, const char *socket_path)
{
    vfu_ctx_t *vfu_ctx = vfu_create_ctx(VFU_TRANS_SOCK, socket_path,
                                         LIBVFIO_USER_FLAG_ATTACH_NB,
                                         dev, VFU_DEV_TYPE_PCI);
    if (!vfu_ctx) {
        sdxi_error("vfu_create_ctx failed: %s", strerror(errno));
        return NULL;
    }

    /* PCI type and header */
    if (vfu_pci_init(vfu_ctx, VFU_PCI_TYPE_EXPRESS,
                     PCI_HEADER_TYPE_NORMAL, 0) < 0) {
        sdxi_error("vfu_pci_init failed: %s", strerror(errno));
        goto err;
    }

    /* PCI IDs */
    vfu_pci_set_id(vfu_ctx,
                   SDXI_PCI_VENDOR_ID, SDXI_PCI_DEVICE_ID,
                   SDXI_PCI_SUBVENDOR_ID, SDXI_PCI_SUBDEVICE_ID);

    /* PCI class: Processing Accelerator / SDXI (0x120100) */
    vfu_pci_set_class(vfu_ctx, SDXI_PCI_CLASS_BASE,
                      SDXI_PCI_CLASS_SUB, SDXI_PCI_CLASS_PROGIF);

    /* BAR0: control/status registers + MSI-X table */
    if (vfu_setup_region(vfu_ctx, VFU_PCI_DEV_BAR0_REGION_IDX,
                         SDXI_BAR0_SIZE, bar0_access,
                         VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                         NULL, 0, -1, 0) < 0) {
        sdxi_error("BAR0 vfu_setup_region failed: %s", strerror(errno));
        goto err;
    }

    /* BAR2: doorbell aperture */
    if (vfu_setup_region(vfu_ctx, VFU_PCI_DEV_BAR2_REGION_IDX,
                         SDXI_BAR2_SIZE, bar2_access,
                         VFU_REGION_FLAG_RW | VFU_REGION_FLAG_MEM,
                         NULL, 0, -1, 0) < 0) {
        sdxi_error("BAR2 vfu_setup_region failed: %s", strerror(errno));
        goto err;
    }

    /* DMA region callbacks */
    if (vfu_setup_device_dma(vfu_ctx, 1024,
                             dma_register_cb,
                             dma_unregister_cb) < 0) {
        sdxi_error("vfu_setup_device_dma failed: %s", strerror(errno));
        goto err;
    }

    /* Reset callback */
    if (vfu_setup_device_reset_cb(vfu_ctx, device_reset_cb) < 0) {
        sdxi_error("vfu_setup_device_reset_cb failed: %s", strerror(errno));
        goto err;
    }

    /* MSI-X: set up the capability in BAR0 */
    struct {
        uint8_t  id;
        uint8_t  next;
        uint16_t ctrl;
        uint32_t table;
        uint32_t pba;
    } msix_cap = {
        .id   = PCI_CAP_ID_MSIX,
        .next = 0,
        .ctrl = (uint16_t)((SDXI_MSIX_NUM_VECTORS - 1U) & 0x7FFU),
        .table = (SDXI_MSIX_TABLE_OFFSET & 0xFFFFFFF8U) |
                 (SDXI_MSIX_TABLE_BIR & 0x7U),
        .pba   = (SDXI_MSIX_PBA_OFFSET   & 0xFFFFFFF8U) |
                 (SDXI_MSIX_TABLE_BIR & 0x7U),
    };
    if (vfu_pci_add_capability(vfu_ctx, 0, 0, &msix_cap) < 0) {
        sdxi_error("vfu_pci_add_capability (MSI-X) failed: %s",
                   strerror(errno));
        goto err;
    }

    /* Declare MSI-X vector count */
    if (vfu_setup_device_nr_irqs(vfu_ctx, VFU_DEV_MSIX_IRQ,
                                 SDXI_MSIX_NUM_VECTORS) < 0) {
        sdxi_error("vfu_setup_device_nr_irqs failed: %s", strerror(errno));
        goto err;
    }

    /* Realize: freeze configuration, ready for client attachment */
    if (vfu_realize_ctx(vfu_ctx) < 0) {
        sdxi_error("vfu_realize_ctx failed: %s", strerror(errno));
        goto err;
    }

    return vfu_ctx;

err:
    vfu_destroy_ctx(vfu_ctx);
    return NULL;
}

/* ------------------------------------------------------------------ */
/* Main loop                                                           */
/* ------------------------------------------------------------------ */

static void serve(sdxi_dev_t *dev, vfu_ctx_t *vfu_ctx)
{
    struct timespec idle_sleep = { .tv_sec = 0, .tv_nsec = 100000L }; /* 100 µs */

    sdxi_info("listening on socket; waiting for QEMU to connect");

    for (;;) {
        if (g_shutdown_requested) {
            sdxi_info("shutdown requested");
            break;
        }

        /* Attach (non-blocking): wait for QEMU vfio-user client */
        int rc = vfu_attach_ctx(vfu_ctx);
        if (rc < 0) {
            if (errno == EAGAIN) {
                nanosleep(&idle_sleep, NULL);
                continue;
            }
            sdxi_error("vfu_attach_ctx failed: %s", strerror(errno));
            break;
        }

        sdxi_info("client attached");
        dev->reg_status = SDXI_STATUS_READY;

        /* Serve client */
        for (;;) {
            if (g_shutdown_requested)
                break;

            rc = vfu_run_ctx(vfu_ctx);
            if (rc < 0) {
                if (errno == EAGAIN) {
                    /* Poll the engine while idle */
                    sdxi_engine_poll(&dev->engine, vfu_ctx);
                    nanosleep(&idle_sleep, NULL);
                    continue;
                }
                if (errno == ENOTCONN) {
                    sdxi_info("client disconnected");
                    sdxi_engine_init(&dev->engine);
                    dev->reg_status = 0U;
                    break;
                }
                sdxi_error("vfu_run_ctx: %s", strerror(errno));
                return;
            }

            /* vfu_run_ctx processed one or more messages; poll engine */
            sdxi_engine_poll(&dev->engine, vfu_ctx);
        }
    }
}

/* ------------------------------------------------------------------ */
/* CLI                                                                 */
/* ------------------------------------------------------------------ */

static void usage(const char *prog)
{
    fprintf(stderr,
        "Usage: %s [options]\n"
        "\n"
        "  -s, --socket PATH   Unix socket path (default: %s)\n"
        "  -v, --verbose       Increase log verbosity (repeat for debug)\n"
        "  -V, --version       Print version and exit\n"
        "  -h, --help          Print this help\n",
        prog, SDXI_DEFAULT_SOCKET);
}

int main(int argc, char *argv[])
{
    const char *socket_path = SDXI_DEFAULT_SOCKET;
    int verbosity = 0;

    static const struct option long_opts[] = {
        { "socket",  required_argument, NULL, 's' },
        { "verbose", no_argument,       NULL, 'v' },
        { "version", no_argument,       NULL, 'V' },
        { "help",    no_argument,       NULL, 'h' },
        { NULL, 0, NULL, 0 },
    };

    int c;
    while ((c = getopt_long(argc, argv, "s:vVh", long_opts, NULL)) != -1) {
        switch (c) {
        case 's':
            socket_path = optarg;
            break;
        case 'v':
            verbosity++;
            break;
        case 'V':
            printf("rocm-sdxi %s (git %s)\n", "0.1.0", GIT_SHA_SHORT);
            return 0;
        case 'h':
            usage(argv[0]);
            return 0;
        default:
            usage(argv[0]);
            return 1;
        }
    }

    sdxi_log_set_level(verbosity >= 2 ? SDXI_LOG_DEBUG :
                       verbosity >= 1 ? SDXI_LOG_INFO  :
                                        SDXI_LOG_WARN);

    sdxi_info("rocm-sdxi starting (git %s)", GIT_SHA_SHORT);
    sdxi_info("socket: %s", socket_path);

    /* Signal handling */
    struct sigaction sa = { .sa_handler = signal_handler };
    sigemptyset(&sa.sa_mask);
    sigaction(SIGINT,  &sa, NULL);
    sigaction(SIGTERM, &sa, NULL);

    /* Initialise device state */
    sdxi_dev_t dev = {0};
    sdxi_engine_init(&dev.engine);
    dev.reg_status = SDXI_STATUS_READY;

    /* Set up vfio-user context */
    vfu_ctx_t *vfu_ctx = setup_device(&dev, socket_path);
    if (!vfu_ctx) {
        sdxi_error("failed to set up vfio-user device");
        return 1;
    }
    dev.vfu_ctx = vfu_ctx;

    /* Serve until shutdown */
    serve(&dev, vfu_ctx);

    vfu_destroy_ctx(vfu_ctx);
    sdxi_info("rocm-sdxi stopped");
    return 0;
}
