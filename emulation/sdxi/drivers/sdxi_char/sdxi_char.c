// SPDX-License-Identifier: GPL-2.0
/*
 * sdxi_char.c — out-of-tree SDXI userspace char device
 *
 * Registers /dev/sdxi (misc device) and exposes MEMCPY, COMPRESS, and
 * DECOMPRESS to userspace via ioctl.  Operates directly against BAR0/BAR2
 * of the SDXI PCI device (vendor 0x1022, device 0x149f) using the same
 * context-table protocol as the upstream v3 kernel driver.
 *
 * NOTE: mutually exclusive with the upstream sdxi.ko — only one driver
 * may probe the device at a time.  Unbind sdxi.ko before loading this
 * module (echo "0000:00:XX.0" > /sys/bus/pci/drivers/sdxi/unbind).
 *
 * Register layout matches src/sdxi_regs.h and the v3 patchset.
 */

#include <linux/module.h>
#include <linux/moduleparam.h>
#include <linux/pci.h>
#include <linux/miscdevice.h>
#include <linux/fs.h>
#include <linux/dma-buf.h>
#include <linux/dma-mapping.h>
#include <linux/uaccess.h>
#include <linux/mutex.h>
#include <linux/delay.h>
#include <linux/io.h>
#include <linux/slab.h>
#include <linux/jiffies.h>
#include <linux/types.h>
#ifdef CONFIG_IO_URING
#include <linux/io_uring/cmd.h>
#include <linux/interrupt.h>
#include <linux/spinlock.h>
#include <linux/list.h>
#endif

#include "sdxi_user.h"

MODULE_LICENSE("GPL");
MODULE_DESCRIPTION("SDXI userspace char device — MEMCPY, COMPRESS, DECOMPRESS");
MODULE_AUTHOR("Advanced Micro Devices, Inc.");
MODULE_VERSION("1.0");
MODULE_IMPORT_NS("DMA_BUF");

/* PCI identity */
#define SDXI_VENDOR_ID   0x1022
#define SDXI_DEVICE_ID   0x149f

/* BAR0 register offsets (v3 patchset layout) */
#define BAR0_CTL0        0x000UL
#define BAR0_STS0        0x100UL
#define BAR0_CAP0        0x200UL
#define BAR0_CAP1        0x208UL
#define BAR0_VERSION     0x210UL
#define BAR0_CTR_SELECT  0x300UL  /* write (cid & 0x7f) | (selector << 8) */
#define BAR0_CTR_VAL     0x308UL  /* read selected 64-bit counter */
#define BAR0_CXT_L2      0x10000UL  /* outer context-table GPA */

/* BAR2: 4096-byte stride per context */
#define SDXI_DB_STRIDE   4096UL

/* Descriptor ring */
#define DSC_SIZE         64U
#define RING_SZ          16U

/* CSB (struct sdxi_cst_blk) */
#define CSB_SIZE         32U
#define CSB_SIGNAL_OFF   0U    /* LE64: cleared to 0 on completion */
#define CSB_FLAGS_OFF    8U    /* LE32: bit 31 = error */
#define CSB_RESULT_OFF   12U   /* LE64: output length for compress/decompress */
#define CSB_ER_BIT       (1U << 31)
#define CSB_TIMEOUT_MS   5000

/* Context-table geometry (matches sdxi_engine.c) */
#define L1_CTX_PER_GROUP 128U
#define INNER_ENTRY_SZ   32U
#define CTX_CTL_SZ       64U

/* Context IDs */
#define ADMIN_CID        0U
#define CLIENT_CID       1U

/* Descriptor TYPE / SUBTYPE values */
#define TYPE_DMAB          0x001U
#define TYPE_ADMIN         0x002U
#define SUBTYPE_COPY       0x03U
#define SUBTYPE_CXT_START  0x03U   /* same value, disambiguated by TYPE */
#define SUBTYPE_COMPRESS   0x04U
#define SUBTYPE_DECOMPRESS 0x05U
#define TYPE_COMPUTE       0x008U
#define SUBTYPE_ELEMWISE   0x01U

/* Per-operation bounce buffer limit */
#define MAX_OP_SIZE      (16UL << 20)   /* 16 MiB */

/* ------------------------------------------------------------------ */

struct sdxi_cxt {
	void       *ring_cpu;
	dma_addr_t  ring_dma;
	void       *cxt_ctl_cpu;
	dma_addr_t  cxt_ctl_dma;
	void       *csb_cpu;
	dma_addr_t  csb_dma;
	u32         wi;
};

struct sdxi_char_dev {
	struct pci_dev   *pdev;
	void __iomem     *bar0;
	void __iomem     *bar2;

	/* Two-level context table */
	void             *outer_cpu;
	dma_addr_t        outer_dma;
	void             *inner_cpu;
	dma_addr_t        inner_dma;

	struct sdxi_cxt   admin;
	struct sdxi_cxt   client;

	struct mutex      lock;
	struct miscdevice misc;
	char              name[32];
#ifdef CONFIG_IO_URING
	spinlock_t        irq_lock;
	struct list_head  inflight;
	int               msix_irq;
#endif
};

static struct sdxi_char_dev *g_sdev;

/* ------------------------------------------------------------------ */
/* Descriptor helpers                                                  */
/* ------------------------------------------------------------------ */

static inline u32 make_opcode(u32 type, u32 subtype)
{
	return 1U | (subtype << 8) | (type << 16);
}

static inline u8 *ring_slot(struct sdxi_cxt *cxt)
{
	return (u8 *)cxt->ring_cpu + (cxt->wi % RING_SZ) * DSC_SIZE;
}

static void ring_doorbell(struct sdxi_char_dev *sdev, u32 cid, u32 wi)
{
	writeq(wi, sdev->bar2 + (u64)cid * SDXI_DB_STRIDE);
}

static void clear_csb(struct sdxi_cxt *cxt)
{
	memset(cxt->csb_cpu, 0xFF, CSB_SIZE);
}

static int poll_csb(struct sdxi_cxt *cxt, u64 *out_len)
{
	unsigned long deadline = jiffies + msecs_to_jiffies(CSB_TIMEOUT_MS);
	u8 *csb = (u8 *)cxt->csb_cpu;
	u32 sig_lo, flags;

	do {
		sig_lo = le32_to_cpu(*(__le32 *)(csb + CSB_SIGNAL_OFF));
		if (sig_lo != 0xFFFFFFFF) {
			flags = le32_to_cpu(*(__le32 *)(csb + CSB_FLAGS_OFF));
			if (out_len)
				*out_len = le64_to_cpu(*(__le64 *)(csb + CSB_RESULT_OFF));
			return (flags & CSB_ER_BIT) ? -EIO : 0;
		}
		usleep_range(100, 200);
	} while (time_before(jiffies, deadline));

	return -ETIMEDOUT;
}

/* ------------------------------------------------------------------ */
/* Bounce buffer helpers                                               */
/* ------------------------------------------------------------------ */

struct bounce {
	void       *cpu;
	dma_addr_t  dma;
	size_t      size;
};

static int bounce_alloc(struct sdxi_char_dev *sdev, size_t size, struct bounce *b)
{
	b->size = size;
	b->cpu  = dma_alloc_coherent(&sdev->pdev->dev, size, &b->dma, GFP_KERNEL);
	return b->cpu ? 0 : -ENOMEM;
}

static void bounce_free(struct sdxi_char_dev *sdev, struct bounce *b)
{
	if (b->cpu) {
		dma_free_coherent(&sdev->pdev->dev, b->size, b->cpu, b->dma);
		b->cpu = NULL;
	}
}

/* ------------------------------------------------------------------ */
/* ioctl handlers                                                      */
/* ------------------------------------------------------------------ */

static long handle_memcpy(struct sdxi_char_dev *sdev,
			   struct sdxi_ioc_memcpy __user *uioc)
{
	struct sdxi_ioc_memcpy ioc;
	struct bounce src = {}, dst = {};
	u8 *slot;
	long ret;

	if (copy_from_user(&ioc, uioc, sizeof(ioc)))
		return -EFAULT;
	if (ioc.len == 0)
		return 0;
	if (ioc.len > MAX_OP_SIZE)
		return -E2BIG;

	ret = bounce_alloc(sdev, ioc.len, &src);
	if (ret)
		goto out;
	ret = bounce_alloc(sdev, ioc.len, &dst);
	if (ret)
		goto out;

	if (copy_from_user(src.cpu, (void __user *)ioc.src, ioc.len)) {
		ret = -EFAULT;
		goto out;
	}

	clear_csb(&sdev->client);
	slot = ring_slot(&sdev->client);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_COPY));
	*(__le32 *)(slot + 4)  = cpu_to_le32((u32)(ioc.len - 1));  /* size_m1 */
	*(__le64 *)(slot + 16) = cpu_to_le64(src.dma);
	*(__le64 *)(slot + 24) = cpu_to_le64(dst.dma);
	*(__le64 *)(slot + 56) = cpu_to_le64(sdev->client.csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);

	ret = poll_csb(&sdev->client, NULL);
	if (ret)
		goto out;

	if (copy_to_user((void __user *)ioc.dst, dst.cpu, ioc.len))
		ret = -EFAULT;
out:
	bounce_free(sdev, &src);
	bounce_free(sdev, &dst);
	return ret;
}

static long handle_compress(struct sdxi_char_dev *sdev,
			     struct sdxi_ioc_compress __user *uioc)
{
	struct sdxi_ioc_compress ioc;
	struct bounce src = {}, dst = {};
	u64 out_len = 0;
	u8 *slot;
	long ret;

	if (copy_from_user(&ioc, uioc, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0 || ioc.dst_cap == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE || ioc.dst_cap > MAX_OP_SIZE)
		return -E2BIG;

	ret = bounce_alloc(sdev, ioc.src_len, &src);
	if (ret)
		goto out;
	ret = bounce_alloc(sdev, ioc.dst_cap, &dst);
	if (ret)
		goto out;

	if (copy_from_user(src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto out;
	}

	clear_csb(&sdev->client);
	slot = ring_slot(&sdev->client);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_COMPRESS));
	*(__le32 *)(slot + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(slot + 8)  = cpu_to_le32((u32)ioc.dst_cap);
	*(__le32 *)(slot + 12) = cpu_to_le32((ioc.level & 0xFFU) | ((ioc.algo & 0xFFU) << 8U));
	*(__le64 *)(slot + 16) = cpu_to_le64(src.dma);
	*(__le64 *)(slot + 24) = cpu_to_le64(dst.dma);
	*(__le64 *)(slot + 56) = cpu_to_le64(sdev->client.csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);

	ret = poll_csb(&sdev->client, &out_len);
	if (ret)
		goto out;

	if (out_len > ioc.dst_cap) {
		ret = -EIO;
		goto out;
	}
	if (copy_to_user((void __user *)ioc.dst, dst.cpu, (size_t)out_len)) {
		ret = -EFAULT;
		goto out;
	}
	ioc.out_len = out_len;
	if (copy_to_user(uioc, &ioc, sizeof(ioc)))
		ret = -EFAULT;
out:
	bounce_free(sdev, &src);
	bounce_free(sdev, &dst);
	return ret;
}

static long handle_decompress(struct sdxi_char_dev *sdev,
			       struct sdxi_ioc_decompress __user *uioc)
{
	struct sdxi_ioc_decompress ioc;
	struct bounce src = {}, dst = {};
	u64 out_len = 0;
	u8 *slot;
	long ret;

	if (copy_from_user(&ioc, uioc, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0 || ioc.dst_cap == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE || ioc.dst_cap > MAX_OP_SIZE)
		return -E2BIG;

	ret = bounce_alloc(sdev, ioc.src_len, &src);
	if (ret)
		goto out;
	ret = bounce_alloc(sdev, ioc.dst_cap, &dst);
	if (ret)
		goto out;

	if (copy_from_user(src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto out;
	}

	clear_csb(&sdev->client);
	slot = ring_slot(&sdev->client);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_DECOMPRESS));
	*(__le32 *)(slot + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(slot + 8)  = cpu_to_le32((u32)ioc.dst_cap);
	*(__le32 *)(slot + 12) = cpu_to_le32((ioc.algo & 0xFFU) << 8U);
	*(__le64 *)(slot + 16) = cpu_to_le64(src.dma);
	*(__le64 *)(slot + 24) = cpu_to_le64(dst.dma);
	*(__le64 *)(slot + 56) = cpu_to_le64(sdev->client.csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);

	ret = poll_csb(&sdev->client, &out_len);
	if (ret)
		goto out;

	if (out_len > ioc.dst_cap) {
		ret = -EIO;
		goto out;
	}
	if (copy_to_user((void __user *)ioc.dst, dst.cpu, (size_t)out_len)) {
		ret = -EFAULT;
		goto out;
	}
	ioc.out_len = out_len;
	if (copy_to_user(uioc, &ioc, sizeof(ioc)))
		ret = -EFAULT;
out:
	bounce_free(sdev, &src);
	bounce_free(sdev, &dst);
	return ret;
}

static long handle_elemwise(struct sdxi_char_dev *sdev,
			     struct sdxi_ioc_elemwise __user *uioc)
{
	struct sdxi_ioc_elemwise ioc;
	struct bounce src = {}, dst = {};
	u64 out_len = 0;
	u8 *slot;
	long ret;
	size_t elem_sz, dst_len;
	bool is_reduce;

	if (copy_from_user(&ioc, uioc, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE)
		return -E2BIG;

	switch (ioc.dtype) {
	case SDXI_DTYPE_FP32:
	case SDXI_DTYPE_INT32:  elem_sz = 4; break;
	case SDXI_DTYPE_FP16:
	case SDXI_DTYPE_BF16:   elem_sz = 2; break;
	case SDXI_DTYPE_INT8:
	case SDXI_DTYPE_FP8_E4M3:
	case SDXI_DTYPE_FP8_E5M2:
	case SDXI_DTYPE_INT4:   elem_sz = 1; break;
	default: return -EINVAL;
	}

	is_reduce = (ioc.op == SDXI_ELEMOP_REDUCE_MIN ||
		     ioc.op == SDXI_ELEMOP_REDUCE_MAX);
	dst_len = is_reduce ? elem_sz : (size_t)ioc.src_len;

	ret = bounce_alloc(sdev, ioc.src_len, &src);
	if (ret)
		goto out;
	ret = bounce_alloc(sdev, dst_len, &dst);
	if (ret)
		goto out;

	if (copy_from_user(src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto out;
	}

	clear_csb(&sdev->client);
	slot = ring_slot(&sdev->client);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0)  = cpu_to_le32(make_opcode(TYPE_COMPUTE, SUBTYPE_ELEMWISE));
	*(__le32 *)(slot + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(slot + 8)  = cpu_to_le32(ioc.dtype);
	*(__le32 *)(slot + 12) = cpu_to_le32(ioc.op);
	*(__le64 *)(slot + 16) = cpu_to_le64(src.dma);
	*(__le64 *)(slot + 24) = cpu_to_le64(dst.dma);
	*(__le64 *)(slot + 32) = cpu_to_le64(ioc.scalar);
	*(__le64 *)(slot + 56) = cpu_to_le64(sdev->client.csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);

	ret = poll_csb(&sdev->client, &out_len);
	if (ret)
		goto out;

	if (out_len > dst_len) {
		ret = -EIO;
		goto out;
	}
	if (copy_to_user((void __user *)ioc.dst, dst.cpu, (size_t)out_len)) {
		ret = -EFAULT;
		goto out;
	}
	ioc.out_len = out_len;
	if (copy_to_user(uioc, &ioc, sizeof(ioc)))
		ret = -EFAULT;
out:
	bounce_free(sdev, &src);
	bounce_free(sdev, &dst);
	return ret;
}

static long handle_info(struct sdxi_char_dev *sdev,
			struct sdxi_ioc_info __user *uarg)
{
	struct sdxi_ioc_info info;
	u64 cap1;

	info.version   = readl(sdev->bar0 + BAR0_VERSION);
	info.cap0      = readq(sdev->bar0 + BAR0_CAP0);
	cap1           = readq(sdev->bar0 + BAR0_CAP1);
	info.cap1      = cap1;
	info.max_cxt   = (u32)((cap1 >> 16) & 0xFFFFU);
	info.vendor_id = sdev->pdev->vendor;
	info.device_id = sdev->pdev->device;
	info._pad      = 0;

	if (copy_to_user(uarg, &info, sizeof(info)))
		return -EFAULT;
	return 0;
}

static long handle_stats(struct sdxi_char_dev *sdev,
			 struct sdxi_ioc_stats __user *uarg)
{
	struct sdxi_ioc_stats stats;
	int i;

	if (copy_from_user(&stats, uarg, sizeof(stats)))
		return -EFAULT;

	for (i = 0; i < SDXI_NUM_STATS; i++) {
		writeq(((u64)i << 8) | (stats.cid & 0x7FU),
		       sdev->bar0 + BAR0_CTR_SELECT);
		stats.val[i] = readq(sdev->bar0 + BAR0_CTR_VAL);
	}

	if (copy_to_user(uarg, &stats, sizeof(stats)))
		return -EFAULT;
	return 0;
}

static long handle_dmabuf_copy(struct sdxi_char_dev *sdev,
			        struct sdxi_ioc_dmabuf_copy __user *uioc)
{
	struct sdxi_ioc_dmabuf_copy ioc;
	struct dma_buf *src_buf = NULL, *dst_buf = NULL;
	struct dma_buf_attachment *src_attach = NULL, *dst_attach = NULL;
	struct sg_table *src_sgt = NULL, *dst_sgt = NULL;
	dma_addr_t src_dma, dst_dma;
	u8 *slot;
	long ret;

	if (copy_from_user(&ioc, uioc, sizeof(ioc)))
		return -EFAULT;
	if (ioc.len == 0)
		return 0;

	src_buf = dma_buf_get(ioc.src_fd);
	if (IS_ERR(src_buf))
		return PTR_ERR(src_buf);

	dst_buf = dma_buf_get(ioc.dst_fd);
	if (IS_ERR(dst_buf)) {
		ret = PTR_ERR(dst_buf);
		dst_buf = NULL;
		goto out;
	}

	src_attach = dma_buf_attach(src_buf, &sdev->pdev->dev);
	if (IS_ERR(src_attach)) {
		ret = PTR_ERR(src_attach);
		src_attach = NULL;
		goto out;
	}

	dst_attach = dma_buf_attach(dst_buf, &sdev->pdev->dev);
	if (IS_ERR(dst_attach)) {
		ret = PTR_ERR(dst_attach);
		dst_attach = NULL;
		goto out;
	}

	src_sgt = dma_buf_map_attachment_unlocked(src_attach, DMA_TO_DEVICE);
	if (IS_ERR(src_sgt)) {
		ret = PTR_ERR(src_sgt);
		src_sgt = NULL;
		goto out;
	}

	dst_sgt = dma_buf_map_attachment_unlocked(dst_attach, DMA_FROM_DEVICE);
	if (IS_ERR(dst_sgt)) {
		ret = PTR_ERR(dst_sgt);
		dst_sgt = NULL;
		goto out;
	}

	if (src_sgt->nents != 1 || dst_sgt->nents != 1) {
		ret = -ENOTSUPP;
		goto out;
	}

	src_dma = sg_dma_address(src_sgt->sgl) + ioc.src_offset;
	dst_dma = sg_dma_address(dst_sgt->sgl) + ioc.dst_offset;

	clear_csb(&sdev->client);
	slot = ring_slot(&sdev->client);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_COPY));
	*(__le32 *)(slot + 4)  = cpu_to_le32((u32)(ioc.len - 1));
	*(__le64 *)(slot + 16) = cpu_to_le64(src_dma);
	*(__le64 *)(slot + 24) = cpu_to_le64(dst_dma);
	*(__le64 *)(slot + 56) = cpu_to_le64(sdev->client.csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);
	ret = poll_csb(&sdev->client, NULL);
out:
	if (dst_sgt)
		dma_buf_unmap_attachment_unlocked(dst_attach, dst_sgt, DMA_FROM_DEVICE);
	if (src_sgt)
		dma_buf_unmap_attachment_unlocked(src_attach, src_sgt, DMA_TO_DEVICE);
	if (dst_attach)
		dma_buf_detach(dst_buf, dst_attach);
	if (src_attach)
		dma_buf_detach(src_buf, src_attach);
	if (dst_buf)
		dma_buf_put(dst_buf);
	dma_buf_put(src_buf);
	return ret;
}

static long sdxi_char_ioctl(struct file *f, unsigned int cmd, unsigned long arg)
{
	struct sdxi_char_dev *sdev = g_sdev;
	long ret;

	if (!sdev)
		return -ENODEV;

	mutex_lock(&sdev->lock);
	switch (cmd) {
	case SDXI_IOC_MEMCPY:
		ret = handle_memcpy(sdev, (struct sdxi_ioc_memcpy __user *)arg);
		break;
	case SDXI_IOC_COMPRESS:
		ret = handle_compress(sdev, (struct sdxi_ioc_compress __user *)arg);
		break;
	case SDXI_IOC_DECOMPRESS:
		ret = handle_decompress(sdev, (struct sdxi_ioc_decompress __user *)arg);
		break;
	case SDXI_IOC_ELEMWISE:
		ret = handle_elemwise(sdev, (struct sdxi_ioc_elemwise __user *)arg);
		break;
	case SDXI_IOC_INFO:
		ret = handle_info(sdev, (struct sdxi_ioc_info __user *)arg);
		break;
	case SDXI_IOC_STATS:
		ret = handle_stats(sdev, (struct sdxi_ioc_stats __user *)arg);
		break;
	case SDXI_IOC_DMABUF_COPY:
		ret = handle_dmabuf_copy(sdev, (struct sdxi_ioc_dmabuf_copy __user *)arg);
		break;
	default:
		ret = -ENOTTY;
	}
	mutex_unlock(&sdev->lock);
	return ret;
}

#ifdef CONFIG_IO_URING

struct sdxi_inflight {
	struct list_head    node;
	struct io_uring_cmd *cmd;
	unsigned int        issue_flags;
	void               *csb_cpu;
	dma_addr_t          csb_dma;
	struct bounce       src;
	struct bounce       dst;
	bool                use_csb_len;
	bool                aborted;
	u64                 dst_uaddr;
	void __user        *uioc;
	size_t              ioc_size;
	u8                  ioc_buf[48];
};

static void sdxi_cmd_complete(struct io_uring_cmd *cmd, unsigned int issue_flags)
{
	struct sdxi_inflight *inf = *(struct sdxi_inflight **)cmd->pdu;
	struct sdxi_char_dev *sdev = g_sdev;
	u8 *csb;
	u32 flags;
	u64 out_len;
	long ret = 0;

	if (inf->aborted) {
		kfree(inf);
		io_uring_cmd_done(cmd, -ENODEV, issue_flags);
		return;
	}

	csb     = (u8 *)inf->csb_cpu;
	flags   = le32_to_cpu(*(__le32 *)(csb + CSB_FLAGS_OFF));
	out_len = le64_to_cpu(*(__le64 *)(csb + CSB_RESULT_OFF));

	if (flags & CSB_ER_BIT) {
		ret = -EIO;
		goto done;
	}

	if (inf->dst.cpu) {
		size_t copy_len = inf->use_csb_len ? (size_t)out_len : inf->dst.size;
		if (inf->use_csb_len && out_len > inf->dst.size) {
			ret = -EIO;
			goto done;
		}
		if (copy_to_user((void __user *)inf->dst_uaddr,
				 inf->dst.cpu, copy_len)) {
			ret = -EFAULT;
			goto done;
		}
	}

	if (inf->use_csb_len && inf->uioc) {
		*(u64 *)(inf->ioc_buf + 40) = out_len;
		if (copy_to_user(inf->uioc, inf->ioc_buf, inf->ioc_size))
			ret = -EFAULT;
	}
done:
	if (sdev) {
		bounce_free(sdev, &inf->src);
		bounce_free(sdev, &inf->dst);
		if (inf->csb_cpu)
			dma_free_coherent(&sdev->pdev->dev, CSB_SIZE,
					  inf->csb_cpu, inf->csb_dma);
	}
	kfree(inf);
	io_uring_cmd_done(cmd, ret, issue_flags);
}

/*
 * The task-work callback type changed from (cmd, issue_flags) to
 * (io_tw_req, io_tw_token_t). Kernels with the new type also define
 * IO_URING_CMD_TASK_WORK_ISSUE_FLAGS, the flags io_uring_cmd_done() wants
 * when called from task work, so that macro selects the signature.
 */
#ifdef IO_URING_CMD_TASK_WORK_ISSUE_FLAGS
static void sdxi_cmd_finish(struct io_tw_req tw_req, io_tw_token_t tw)
{
	sdxi_cmd_complete(io_uring_cmd_from_tw(tw_req),
			  IO_URING_CMD_TASK_WORK_ISSUE_FLAGS);
}
#else
static void sdxi_cmd_finish(struct io_uring_cmd *cmd, unsigned int issue_flags)
{
	sdxi_cmd_complete(cmd, issue_flags);
}
#endif

static irqreturn_t sdxi_completion_irq(int irq, void *data)
{
	struct sdxi_char_dev *sdev = data;
	struct sdxi_inflight *inf, *tmp;
	LIST_HEAD(done);

	spin_lock(&sdev->irq_lock);
	list_for_each_entry_safe(inf, tmp, &sdev->inflight, node) {
		u32 sig = le32_to_cpu(
			*(__le32 *)((u8 *)inf->csb_cpu + CSB_SIGNAL_OFF));
		if (sig != 0xFFFFFFFF) {
			list_del(&inf->node);
			list_add_tail(&inf->node, &done);
		}
	}
	spin_unlock(&sdev->irq_lock);

	list_for_each_entry_safe(inf, tmp, &done, node) {
		list_del(&inf->node);
		*(struct sdxi_inflight **)inf->cmd->pdu = inf;
		io_uring_cmd_complete_in_task(inf->cmd, sdxi_cmd_finish);
	}
	return IRQ_HANDLED;
}

static int sdxi_submit_inflight(struct sdxi_char_dev *sdev,
				struct sdxi_inflight *inf,
				const u8 *dsc)
{
	unsigned long flags;
	u8 *slot;

	if (!mutex_trylock(&sdev->lock))
		return -EAGAIN;

	slot = ring_slot(&sdev->client);
	memcpy(slot, dsc, 56);
	*(__le64 *)(slot + 56) = cpu_to_le64(inf->csb_dma);
	sdev->client.wi++;
	ring_doorbell(sdev, CLIENT_CID, sdev->client.wi);
	mutex_unlock(&sdev->lock);

	spin_lock_irqsave(&sdev->irq_lock, flags);
	list_add_tail(&inf->node, &sdev->inflight);
	spin_unlock_irqrestore(&sdev->irq_lock, flags);

	return -EIOCBQUEUED;
}

static struct sdxi_inflight *alloc_inflight(struct sdxi_char_dev *sdev,
					    struct io_uring_cmd *cmd,
					    unsigned int issue_flags)
{
	struct sdxi_inflight *inf = kzalloc(sizeof(*inf), GFP_KERNEL);

	if (!inf)
		return NULL;
	inf->cmd         = cmd;
	inf->issue_flags = issue_flags;
	inf->csb_cpu = dma_alloc_coherent(&sdev->pdev->dev, CSB_SIZE,
					  &inf->csb_dma, GFP_KERNEL);
	if (!inf->csb_cpu) {
		kfree(inf);
		return NULL;
	}
	memset(inf->csb_cpu, 0xFF, CSB_SIZE);
	return inf;
}

static void free_inflight(struct sdxi_char_dev *sdev, struct sdxi_inflight *inf)
{
	bounce_free(sdev, &inf->src);
	bounce_free(sdev, &inf->dst);
	if (inf->csb_cpu)
		dma_free_coherent(&sdev->pdev->dev, CSB_SIZE,
				  inf->csb_cpu, inf->csb_dma);
	kfree(inf);
}

static int uring_submit_memcpy(struct sdxi_char_dev *sdev,
			       struct io_uring_cmd *cmd,
			       unsigned int issue_flags)
{
	void __user *arg = u64_to_user_ptr(READ_ONCE(cmd->sqe->addr));
	struct sdxi_ioc_memcpy ioc;
	struct sdxi_inflight *inf;
	u8 dsc[56];
	int ret;

	if (copy_from_user(&ioc, arg, sizeof(ioc)))
		return -EFAULT;
	if (ioc.len == 0) {
		io_uring_cmd_done(cmd, 0, issue_flags);
		return 0;
	}
	if (ioc.len > MAX_OP_SIZE)
		return -E2BIG;

	inf = alloc_inflight(sdev, cmd, issue_flags);
	if (!inf)
		return -ENOMEM;

	ret = bounce_alloc(sdev, ioc.len, &inf->src);
	if (ret)
		goto err;
	ret = bounce_alloc(sdev, ioc.len, &inf->dst);
	if (ret)
		goto err;

	if (copy_from_user(inf->src.cpu, (void __user *)ioc.src, ioc.len)) {
		ret = -EFAULT;
		goto err;
	}

	inf->dst_uaddr   = ioc.dst;
	inf->use_csb_len = false;
	inf->uioc        = NULL;

	memset(dsc, 0, sizeof(dsc));
	*(__le32 *)(dsc + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_COPY));
	*(__le32 *)(dsc + 4)  = cpu_to_le32((u32)(ioc.len - 1));
	*(__le64 *)(dsc + 16) = cpu_to_le64(inf->src.dma);
	*(__le64 *)(dsc + 24) = cpu_to_le64(inf->dst.dma);

	ret = sdxi_submit_inflight(sdev, inf, dsc);
	if (ret != -EIOCBQUEUED)
		goto err;
	return -EIOCBQUEUED;
err:
	free_inflight(sdev, inf);
	return ret;
}

static int uring_submit_compress(struct sdxi_char_dev *sdev,
				 struct io_uring_cmd *cmd,
				 unsigned int issue_flags)
{
	void __user *arg = u64_to_user_ptr(READ_ONCE(cmd->sqe->addr));
	struct sdxi_ioc_compress ioc;
	struct sdxi_inflight *inf;
	u8 dsc[56];
	int ret;

	if (copy_from_user(&ioc, arg, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0 || ioc.dst_cap == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE || ioc.dst_cap > MAX_OP_SIZE)
		return -E2BIG;

	inf = alloc_inflight(sdev, cmd, issue_flags);
	if (!inf)
		return -ENOMEM;

	ret = bounce_alloc(sdev, ioc.src_len, &inf->src);
	if (ret)
		goto err;
	ret = bounce_alloc(sdev, ioc.dst_cap, &inf->dst);
	if (ret)
		goto err;

	if (copy_from_user(inf->src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto err;
	}

	inf->dst_uaddr   = ioc.dst;
	inf->use_csb_len = true;
	inf->uioc        = arg;
	inf->ioc_size    = sizeof(ioc);
	memcpy(inf->ioc_buf, &ioc, sizeof(ioc));

	memset(dsc, 0, sizeof(dsc));
	*(__le32 *)(dsc + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_COMPRESS));
	*(__le32 *)(dsc + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(dsc + 8)  = cpu_to_le32((u32)ioc.dst_cap);
	*(__le32 *)(dsc + 12) = cpu_to_le32((ioc.level & 0xFFU) |
					    ((ioc.algo & 0xFFU) << 8U));
	*(__le64 *)(dsc + 16) = cpu_to_le64(inf->src.dma);
	*(__le64 *)(dsc + 24) = cpu_to_le64(inf->dst.dma);

	ret = sdxi_submit_inflight(sdev, inf, dsc);
	if (ret != -EIOCBQUEUED)
		goto err;
	return -EIOCBQUEUED;
err:
	free_inflight(sdev, inf);
	return ret;
}

static int uring_submit_decompress(struct sdxi_char_dev *sdev,
				   struct io_uring_cmd *cmd,
				   unsigned int issue_flags)
{
	void __user *arg = u64_to_user_ptr(READ_ONCE(cmd->sqe->addr));
	struct sdxi_ioc_decompress ioc;
	struct sdxi_inflight *inf;
	u8 dsc[56];
	int ret;

	if (copy_from_user(&ioc, arg, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0 || ioc.dst_cap == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE || ioc.dst_cap > MAX_OP_SIZE)
		return -E2BIG;

	inf = alloc_inflight(sdev, cmd, issue_flags);
	if (!inf)
		return -ENOMEM;

	ret = bounce_alloc(sdev, ioc.src_len, &inf->src);
	if (ret)
		goto err;
	ret = bounce_alloc(sdev, ioc.dst_cap, &inf->dst);
	if (ret)
		goto err;

	if (copy_from_user(inf->src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto err;
	}

	inf->dst_uaddr   = ioc.dst;
	inf->use_csb_len = true;
	inf->uioc        = arg;
	inf->ioc_size    = sizeof(ioc);
	memcpy(inf->ioc_buf, &ioc, sizeof(ioc));

	memset(dsc, 0, sizeof(dsc));
	*(__le32 *)(dsc + 0)  = cpu_to_le32(make_opcode(TYPE_DMAB, SUBTYPE_DECOMPRESS));
	*(__le32 *)(dsc + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(dsc + 8)  = cpu_to_le32((u32)ioc.dst_cap);
	*(__le32 *)(dsc + 12) = cpu_to_le32((ioc.algo & 0xFFU) << 8U);
	*(__le64 *)(dsc + 16) = cpu_to_le64(inf->src.dma);
	*(__le64 *)(dsc + 24) = cpu_to_le64(inf->dst.dma);

	ret = sdxi_submit_inflight(sdev, inf, dsc);
	if (ret != -EIOCBQUEUED)
		goto err;
	return -EIOCBQUEUED;
err:
	free_inflight(sdev, inf);
	return ret;
}

static int uring_submit_elemwise(struct sdxi_char_dev *sdev,
				 struct io_uring_cmd *cmd,
				 unsigned int issue_flags)
{
	void __user *arg = u64_to_user_ptr(READ_ONCE(cmd->sqe->addr));
	struct sdxi_ioc_elemwise ioc;
	struct sdxi_inflight *inf;
	u8 dsc[56];
	size_t elem_sz, dst_len;
	bool is_reduce;
	int ret;

	if (copy_from_user(&ioc, arg, sizeof(ioc)))
		return -EFAULT;
	if (ioc.src_len == 0)
		return -EINVAL;
	if (ioc.src_len > MAX_OP_SIZE)
		return -E2BIG;

	switch (ioc.dtype) {
	case SDXI_DTYPE_FP32:
	case SDXI_DTYPE_INT32:  elem_sz = 4; break;
	case SDXI_DTYPE_FP16:
	case SDXI_DTYPE_BF16:   elem_sz = 2; break;
	case SDXI_DTYPE_INT8:
	case SDXI_DTYPE_FP8_E4M3:
	case SDXI_DTYPE_FP8_E5M2:
	case SDXI_DTYPE_INT4:   elem_sz = 1; break;
	default: return -EINVAL;
	}

	is_reduce = (ioc.op == SDXI_ELEMOP_REDUCE_MIN ||
		     ioc.op == SDXI_ELEMOP_REDUCE_MAX);
	dst_len = is_reduce ? elem_sz : (size_t)ioc.src_len;

	inf = alloc_inflight(sdev, cmd, issue_flags);
	if (!inf)
		return -ENOMEM;

	ret = bounce_alloc(sdev, ioc.src_len, &inf->src);
	if (ret)
		goto err;
	ret = bounce_alloc(sdev, dst_len, &inf->dst);
	if (ret)
		goto err;

	if (copy_from_user(inf->src.cpu, (void __user *)ioc.src, ioc.src_len)) {
		ret = -EFAULT;
		goto err;
	}

	inf->dst_uaddr   = ioc.dst;
	inf->use_csb_len = true;
	inf->uioc        = arg;
	inf->ioc_size    = sizeof(ioc);
	memcpy(inf->ioc_buf, &ioc, sizeof(ioc));

	memset(dsc, 0, sizeof(dsc));
	*(__le32 *)(dsc + 0)  = cpu_to_le32(make_opcode(TYPE_COMPUTE, SUBTYPE_ELEMWISE));
	*(__le32 *)(dsc + 4)  = cpu_to_le32((u32)ioc.src_len);
	*(__le32 *)(dsc + 8)  = cpu_to_le32(ioc.dtype);
	*(__le32 *)(dsc + 12) = cpu_to_le32(ioc.op);
	*(__le64 *)(dsc + 16) = cpu_to_le64(inf->src.dma);
	*(__le64 *)(dsc + 24) = cpu_to_le64(inf->dst.dma);
	*(__le64 *)(dsc + 32) = cpu_to_le64(ioc.scalar);

	ret = sdxi_submit_inflight(sdev, inf, dsc);
	if (ret != -EIOCBQUEUED)
		goto err;
	return -EIOCBQUEUED;
err:
	free_inflight(sdev, inf);
	return ret;
}

static int sdxi_uring_cmd(struct io_uring_cmd *cmd, unsigned int issue_flags)
{
	struct sdxi_char_dev *sdev = g_sdev;
	void __user *arg = u64_to_user_ptr(READ_ONCE(cmd->sqe->addr));
	long ret;

	if (!sdev)
		return -ENODEV;

	/* INFO/STATS are pure MMIO — always synchronous */
	switch (cmd->cmd_op) {
	case SDXI_IOC_INFO:
		mutex_lock(&sdev->lock);
		ret = handle_info(sdev, arg);
		mutex_unlock(&sdev->lock);
		io_uring_cmd_done(cmd, ret, issue_flags);
		return 0;
	case SDXI_IOC_STATS:
		mutex_lock(&sdev->lock);
		ret = handle_stats(sdev, arg);
		mutex_unlock(&sdev->lock);
		io_uring_cmd_done(cmd, ret, issue_flags);
		return 0;
	case SDXI_IOC_DMABUF_COPY:
		/* dma-buf async lifecycle is complex; keep synchronous */
		if (issue_flags & IO_URING_F_NONBLOCK)
			return -EAGAIN;
		mutex_lock(&sdev->lock);
		ret = handle_dmabuf_copy(sdev, arg);
		mutex_unlock(&sdev->lock);
		io_uring_cmd_done(cmd, ret, issue_flags);
		return 0;
	}

	if (sdev->msix_irq < 0) {
		/* No MSI-X: synchronous fallback via io-wq */
		if (issue_flags & IO_URING_F_NONBLOCK)
			return -EAGAIN;
		mutex_lock(&sdev->lock);
		switch (cmd->cmd_op) {
		case SDXI_IOC_MEMCPY:
			ret = handle_memcpy(sdev, arg);
			break;
		case SDXI_IOC_COMPRESS:
			ret = handle_compress(sdev, arg);
			break;
		case SDXI_IOC_DECOMPRESS:
			ret = handle_decompress(sdev, arg);
			break;
		case SDXI_IOC_ELEMWISE:
			ret = handle_elemwise(sdev, arg);
			break;
		default:
			ret = -EOPNOTSUPP;
		}
		mutex_unlock(&sdev->lock);
		io_uring_cmd_done(cmd, ret, issue_flags);
		return 0;
	}

	/* MSI-X available: async path */
	switch (cmd->cmd_op) {
	case SDXI_IOC_MEMCPY:
		return uring_submit_memcpy(sdev, cmd, issue_flags);
	case SDXI_IOC_COMPRESS:
		return uring_submit_compress(sdev, cmd, issue_flags);
	case SDXI_IOC_DECOMPRESS:
		return uring_submit_decompress(sdev, cmd, issue_flags);
	case SDXI_IOC_ELEMWISE:
		return uring_submit_elemwise(sdev, cmd, issue_flags);
	default:
		return -EOPNOTSUPP;
	}
}
#endif /* CONFIG_IO_URING */

static const struct file_operations sdxi_char_fops = {
	.owner          = THIS_MODULE,
	.unlocked_ioctl = sdxi_char_ioctl,
	.compat_ioctl   = compat_ptr_ioctl,
#ifdef CONFIG_IO_URING
	.uring_cmd      = sdxi_uring_cmd,
#endif
};

/* ------------------------------------------------------------------ */
/* Hardware context-table setup                                         */
/* ------------------------------------------------------------------ */

static int sdxi_alloc_cxt(struct sdxi_char_dev *sdev, struct sdxi_cxt *cxt)
{
	cxt->ring_cpu = dma_alloc_coherent(&sdev->pdev->dev, RING_SZ * DSC_SIZE,
					   &cxt->ring_dma, GFP_KERNEL);
	if (!cxt->ring_cpu)
		return -ENOMEM;

	cxt->cxt_ctl_cpu = dma_alloc_coherent(&sdev->pdev->dev, CTX_CTL_SZ,
					       &cxt->cxt_ctl_dma, GFP_KERNEL);
	if (!cxt->cxt_ctl_cpu)
		return -ENOMEM;

	cxt->csb_cpu = dma_alloc_coherent(&sdev->pdev->dev, CSB_SIZE,
					  &cxt->csb_dma, GFP_KERNEL);
	if (!cxt->csb_cpu)
		return -ENOMEM;

	cxt->wi = 0;
	return 0;
}

static void sdxi_free_cxt(struct sdxi_char_dev *sdev, struct sdxi_cxt *cxt)
{
	if (cxt->csb_cpu) {
		dma_free_coherent(&sdev->pdev->dev, CSB_SIZE,
				  cxt->csb_cpu, cxt->csb_dma);
		cxt->csb_cpu = NULL;
	}
	if (cxt->cxt_ctl_cpu) {
		dma_free_coherent(&sdev->pdev->dev, CTX_CTL_SZ,
				  cxt->cxt_ctl_cpu, cxt->cxt_ctl_dma);
		cxt->cxt_ctl_cpu = NULL;
	}
	if (cxt->ring_cpu) {
		dma_free_coherent(&sdev->pdev->dev, RING_SZ * DSC_SIZE,
				  cxt->ring_cpu, cxt->ring_dma);
		cxt->ring_cpu = NULL;
	}
}

static int sdxi_hw_init(struct sdxi_char_dev *sdev)
{
	u8 *slot;
	int ret;

	/* Outer table: 4096-byte page, one entry pointing to inner. */
	sdev->outer_cpu = dma_alloc_coherent(&sdev->pdev->dev, 4096,
					     &sdev->outer_dma, GFP_KERNEL);
	/* Inner table: 128 × 32 bytes. */
	sdev->inner_cpu = dma_alloc_coherent(&sdev->pdev->dev,
					     L1_CTX_PER_GROUP * INNER_ENTRY_SZ,
					     &sdev->inner_dma, GFP_KERNEL);
	if (!sdev->outer_cpu || !sdev->inner_cpu)
		return -ENOMEM;

	ret = sdxi_alloc_cxt(sdev, &sdev->admin);
	if (ret)
		return ret;
	ret = sdxi_alloc_cxt(sdev, &sdev->client);
	if (ret)
		return ret;

	/* Wire admin context (CID 0) ctx_ctl: ring ptr + ring size. */
	*(__le64 *)((u8 *)sdev->admin.cxt_ctl_cpu + 0) =
		cpu_to_le64(sdev->admin.ring_dma | 1ULL);   /* VL=1 */
	*(__le64 *)((u8 *)sdev->admin.cxt_ctl_cpu + 8) = cpu_to_le64(RING_SZ);

	/* Wire client context (CID 1) ctx_ctl. */
	*(__le64 *)((u8 *)sdev->client.cxt_ctl_cpu + 0) =
		cpu_to_le64(sdev->client.ring_dma | 1ULL);
	*(__le64 *)((u8 *)sdev->client.cxt_ctl_cpu + 8) = cpu_to_le64(RING_SZ);

	/* INNER[ADMIN_CID] → admin ctx_ctl, VL=1 */
	*(__le64 *)((u8 *)sdev->inner_cpu + ADMIN_CID * INNER_ENTRY_SZ) =
		cpu_to_le64(sdev->admin.cxt_ctl_dma | 1ULL);

	/* INNER[CLIENT_CID] → client ctx_ctl, VL=1 */
	*(__le64 *)((u8 *)sdev->inner_cpu + CLIENT_CID * INNER_ENTRY_SZ) =
		cpu_to_le64(sdev->client.cxt_ctl_dma | 1ULL);

	/* OUTER[0] → INNER, VL=1 */
	*(__le64 *)sdev->outer_cpu = cpu_to_le64(sdev->inner_dma | 1ULL);

	/* Point hardware at the outer table. */
	writeq(sdev->outer_dma, sdev->bar0 + BAR0_CXT_L2);

	/* Submit CXT_START for CLIENT_CID via the admin ring. */
	slot = ring_slot(&sdev->admin);
	memset(slot, 0, DSC_SIZE);
	*(__le32 *)(slot + 0) = cpu_to_le32(make_opcode(TYPE_ADMIN, SUBTYPE_CXT_START));
	*(__le32 *)(slot + 4) = cpu_to_le32(CLIENT_CID);
	sdev->admin.wi++;
	ring_doorbell(sdev, ADMIN_CID, sdev->admin.wi);

	msleep(50);  /* give the emulator time to process */
	return 0;
}

static void sdxi_hw_fini(struct sdxi_char_dev *sdev)
{
	sdxi_free_cxt(sdev, &sdev->client);
	sdxi_free_cxt(sdev, &sdev->admin);

	if (sdev->inner_cpu) {
		dma_free_coherent(&sdev->pdev->dev,
				  L1_CTX_PER_GROUP * INNER_ENTRY_SZ,
				  sdev->inner_cpu, sdev->inner_dma);
		sdev->inner_cpu = NULL;
	}
	if (sdev->outer_cpu) {
		dma_free_coherent(&sdev->pdev->dev, 4096,
				  sdev->outer_cpu, sdev->outer_dma);
		sdev->outer_cpu = NULL;
	}
}

/* ------------------------------------------------------------------ */
/* PCI probe / remove                                                  */
/* ------------------------------------------------------------------ */

static int sdxi_char_probe(struct pci_dev *pdev, const struct pci_device_id *id)
{
	struct sdxi_char_dev *sdev;
	int ret;

	sdev = devm_kzalloc(&pdev->dev, sizeof(*sdev), GFP_KERNEL);
	if (!sdev)
		return -ENOMEM;

	sdev->pdev = pdev;
	mutex_init(&sdev->lock);
	pci_set_drvdata(pdev, sdev);

	ret = pci_enable_device(pdev);
	if (ret)
		return ret;

	ret = pci_request_regions(pdev, "sdxi_char");
	if (ret)
		goto err_disable;

	pci_set_master(pdev);

	ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(64));
	if (ret) {
		ret = dma_set_mask_and_coherent(&pdev->dev, DMA_BIT_MASK(32));
		if (ret)
			goto err_regions;
	}

	sdev->bar0 = pci_iomap(pdev, 0, 0);
	if (!sdev->bar0) {
		ret = -EIO;
		goto err_regions;
	}

	sdev->bar2 = pci_iomap(pdev, 2, 0);
	if (!sdev->bar2) {
		ret = -EIO;
		goto err_bar0;
	}

	ret = sdxi_hw_init(sdev);
	if (ret)
		goto err_bar2;

	snprintf(sdev->name, sizeof(sdev->name), "sdxi%d", pdev->devfn);
	sdev->misc.minor  = MISC_DYNAMIC_MINOR;
	sdev->misc.name   = sdev->name;
	sdev->misc.fops   = &sdxi_char_fops;
	sdev->misc.parent = &pdev->dev;

	ret = misc_register(&sdev->misc);
	if (ret)
		goto err_hw;

	g_sdev = sdev;
#ifdef CONFIG_IO_URING
	spin_lock_init(&sdev->irq_lock);
	INIT_LIST_HEAD(&sdev->inflight);
	sdev->msix_irq = -1;
	ret = pci_alloc_irq_vectors(pdev, 1, 1, PCI_IRQ_MSIX);
	if (ret == 1) {
		int irq = pci_irq_vector(pdev, 0);
		ret = request_irq(irq, sdxi_completion_irq, 0, "sdxi_char", sdev);
		if (ret == 0) {
			sdev->msix_irq = irq;
			dev_info(&pdev->dev, "sdxi_char: MSI-X IRQ %d\n", irq);
		} else {
			pci_free_irq_vectors(pdev);
			dev_warn(&pdev->dev,
				 "sdxi_char: request_irq failed (%d), async disabled\n",
				 ret);
		}
		ret = 0;
	} else {
		if (ret > 0)
			pci_free_irq_vectors(pdev);
		dev_warn(&pdev->dev, "sdxi_char: MSI-X unavailable, async disabled\n");
		ret = 0;
	}
#endif
	dev_info(&pdev->dev, "sdxi_char: registered /dev/%s\n", sdev->name);
	return 0;

err_hw:
	sdxi_hw_fini(sdev);
err_bar2:
	pci_iounmap(pdev, sdev->bar2);
err_bar0:
	pci_iounmap(pdev, sdev->bar0);
err_regions:
	pci_release_regions(pdev);
err_disable:
	pci_disable_device(pdev);
	return ret;
}

static void sdxi_char_remove(struct pci_dev *pdev)
{
	struct sdxi_char_dev *sdev = pci_get_drvdata(pdev);

	if (!sdev)
		return;

	g_sdev = NULL;
	misc_deregister(&sdev->misc);
#ifdef CONFIG_IO_URING
	if (sdev->msix_irq >= 0) {
		free_irq(sdev->msix_irq, sdev);
		pci_free_irq_vectors(pdev);
		sdev->msix_irq = -1;
	}
	{
		struct sdxi_inflight *inf, *tmp;
		LIST_HEAD(orphans);
		unsigned long flags;

		spin_lock_irqsave(&sdev->irq_lock, flags);
		list_splice_init(&sdev->inflight, &orphans);
		spin_unlock_irqrestore(&sdev->irq_lock, flags);

		list_for_each_entry_safe(inf, tmp, &orphans, node) {
			list_del(&inf->node);
			bounce_free(sdev, &inf->src);
			bounce_free(sdev, &inf->dst);
			if (inf->csb_cpu)
				dma_free_coherent(&pdev->dev, CSB_SIZE,
						  inf->csb_cpu, inf->csb_dma);
			inf->csb_cpu = NULL;
			inf->src.cpu = NULL;
			inf->dst.cpu = NULL;
			inf->aborted = true;
			*(struct sdxi_inflight **)inf->cmd->pdu = inf;
			io_uring_cmd_complete_in_task(inf->cmd, sdxi_cmd_finish);
		}
	}
#endif
	sdxi_hw_fini(sdev);
	pci_iounmap(pdev, sdev->bar2);
	pci_iounmap(pdev, sdev->bar0);
	pci_release_regions(pdev);
	pci_disable_device(pdev);
	dev_info(&pdev->dev, "sdxi_char: removed\n");
}

static const struct pci_device_id sdxi_char_pci_ids[] = {
	{ PCI_DEVICE(SDXI_VENDOR_ID, SDXI_DEVICE_ID) },
	{}
};
MODULE_DEVICE_TABLE(pci, sdxi_char_pci_ids);

static struct pci_driver sdxi_char_driver = {
	.name     = "sdxi_char",
	.id_table = sdxi_char_pci_ids,
	.probe    = sdxi_char_probe,
	.remove   = sdxi_char_remove,
};

module_pci_driver(sdxi_char_driver);
