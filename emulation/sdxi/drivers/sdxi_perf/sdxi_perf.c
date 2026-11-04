// SPDX-License-Identifier: MIT
/*
 * sdxi_perf — out-of-tree kernel module that exposes per-context performance
 * counters from the SDXI vfio-user emulator via debugfs.
 *
 * The emulator provides two vendor-specific BAR0 registers:
 *   0x300 (CTR_SELECT): write (stat_sel << 8) | cxt_id to select
 *   0x308 (CTR_VAL):    read 64-bit counter value for the selection
 *
 * Usage:
 *   cat /sys/kernel/debug/sdxi_perf/<dev>/cxt<N>_bytes_moved
 *   cat /sys/kernel/debug/sdxi_perf/<dev>/cxt<N>_num_copy
 *   ... (12 files per context)
 *
 * Build:
 *   make -C /lib/modules/$(uname -r)/build M=$(pwd) modules
 */

#include <linux/module.h>
#include <linux/pci.h>
#include <linux/debugfs.h>
#include <linux/io.h>
#include <linux/slab.h>

#define SDXI_VENDOR_ID   0x1022
#define SDXI_DEVICE_ID   0x149f

/* Vendor-specific performance counter register offsets within BAR0. */
#define SDXI_CTR_SELECT  0x300
#define SDXI_CTR_VAL     0x308
#define SDXI_MAX_CXT     128

/* Stat selector values — must match SDXI_CTR_STAT_* in sdxi_regs.h */
#define STAT_BYTES_MOVED        0x00
#define STAT_NUM_COPY           0x01
#define STAT_NUM_ERRORS         0x02
#define STAT_NUM_INTR           0x03
#define STAT_NUM_CXT_START      0x04
#define STAT_NUM_CXT_STOP       0x05
#define STAT_NUM_NOP            0x06
#define STAT_NUM_SYNC           0x07
#define STAT_BYTES_COMPRESSED   0x08
#define STAT_BYTES_DECOMPRESSED 0x09
#define STAT_NUM_COMPRESS       0x0A
#define STAT_NUM_DECOMPRESS     0x0B

struct sdxi_perf_dev {
	void __iomem     *bar0;
	struct pci_dev   *pdev;
	struct dentry    *dbgfs_dir;
};

struct sdxi_cxt_file {
	struct sdxi_perf_dev *priv;
	unsigned int          cid;
	unsigned int          stat_sel;
};

static int cxt_stat_show(struct seq_file *m, void *v)
{
	struct sdxi_cxt_file *cf = m->private;
	struct sdxi_perf_dev *p  = cf->priv;
	u64 sel = ((u64)cf->stat_sel << 8) | cf->cid;

	writeq(sel, p->bar0 + SDXI_CTR_SELECT);
	u64 val = readq(p->bar0 + SDXI_CTR_VAL);
	seq_printf(m, "%llu\n", val);
	return 0;
}

static int cxt_stat_open(struct inode *inode, struct file *file)
{
	return single_open(file, cxt_stat_show, inode->i_private);
}

static const struct file_operations cxt_stat_fops = {
	.owner   = THIS_MODULE,
	.open    = cxt_stat_open,
	.read    = seq_read,
	.llseek  = seq_lseek,
	.release = single_release,
};

static const struct {
	unsigned int sel;
	const char  *suffix;
} sdxi_stats[] = {
	{ STAT_BYTES_MOVED,        "bytes_moved"        },
	{ STAT_NUM_COPY,           "num_copy"           },
	{ STAT_NUM_ERRORS,         "num_errors"         },
	{ STAT_NUM_INTR,           "num_intr"           },
	{ STAT_NUM_CXT_START,      "num_cxt_start"      },
	{ STAT_NUM_CXT_STOP,       "num_cxt_stop"       },
	{ STAT_NUM_NOP,            "num_nop"            },
	{ STAT_NUM_SYNC,           "num_sync"           },
	{ STAT_BYTES_COMPRESSED,   "bytes_compressed"   },
	{ STAT_BYTES_DECOMPRESSED, "bytes_decompressed" },
	{ STAT_NUM_COMPRESS,       "num_compress"       },
	{ STAT_NUM_DECOMPRESS,     "num_decompress"     },
};

static int sdxi_perf_probe(struct pci_dev *pdev,
			   const struct pci_device_id *id)
{
	int err;
	struct sdxi_perf_dev *priv;

	priv = devm_kzalloc(&pdev->dev, sizeof(*priv), GFP_KERNEL);
	if (!priv)
		return -ENOMEM;
	priv->pdev = pdev;

	err = pcim_enable_device(pdev);
	if (err)
		return err;

	err = pcim_iomap_regions(pdev, BIT(0), KBUILD_MODNAME);
	if (err)
		return err;

	priv->bar0 = pcim_iomap_table(pdev)[0];
	if (!priv->bar0)
		return -ENOMEM;

	pci_set_drvdata(pdev, priv);

	priv->dbgfs_dir = debugfs_create_dir(dev_name(&pdev->dev),
					     debugfs_lookup("sdxi_perf", NULL));
	if (IS_ERR(priv->dbgfs_dir))
		priv->dbgfs_dir = NULL;

	for (unsigned int cid = 0; cid < SDXI_MAX_CXT; cid++) {
		for (unsigned int si = 0; si < ARRAY_SIZE(sdxi_stats); si++) {
			struct sdxi_cxt_file *cf;
			char name[48];

			cf = devm_kzalloc(&pdev->dev, sizeof(*cf), GFP_KERNEL);
			if (!cf)
				goto done;
			cf->priv     = priv;
			cf->cid      = cid;
			cf->stat_sel = sdxi_stats[si].sel;

			snprintf(name, sizeof(name), "cxt%u_%s",
				 cid, sdxi_stats[si].suffix);
			debugfs_create_file(name, 0444, priv->dbgfs_dir, cf,
					    &cxt_stat_fops);
		}
	}
done:
	dev_info(&pdev->dev,
		 "sdxi_perf: registered %u contexts × %zu counters\n",
		 SDXI_MAX_CXT, ARRAY_SIZE(sdxi_stats));
	return 0;
}

static void sdxi_perf_remove(struct pci_dev *pdev)
{
	struct sdxi_perf_dev *priv = pci_get_drvdata(pdev);

	debugfs_remove_recursive(priv->dbgfs_dir);
}

static const struct pci_device_id sdxi_perf_ids[] = {
	{ PCI_DEVICE(SDXI_VENDOR_ID, SDXI_DEVICE_ID) },
	{ 0 }
};
MODULE_DEVICE_TABLE(pci, sdxi_perf_ids);

static struct pci_driver sdxi_perf_driver = {
	.name     = "sdxi_perf",
	.id_table = sdxi_perf_ids,
	.probe    = sdxi_perf_probe,
	.remove   = sdxi_perf_remove,
};

static struct dentry *sdxi_perf_root;

static int __init sdxi_perf_init(void)
{
	sdxi_perf_root = debugfs_create_dir("sdxi_perf", NULL);
	return pci_register_driver(&sdxi_perf_driver);
}

static void __exit sdxi_perf_exit(void)
{
	pci_unregister_driver(&sdxi_perf_driver);
	debugfs_remove_recursive(sdxi_perf_root);
}

module_init(sdxi_perf_init);
module_exit(sdxi_perf_exit);

MODULE_LICENSE("MIT");
MODULE_AUTHOR("Advanced Micro Devices, Inc.");
MODULE_DESCRIPTION("SDXI emulator per-context performance counters via debugfs");
