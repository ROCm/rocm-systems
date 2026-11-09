// SPDX-License-Identifier: GPL-2.0-only
/*
 * SDXI PCI device code
 *
 * Copyright Advanced Micro Devices, Inc.
 */

#include <linux/bitfield.h>
#include <linux/dev_printk.h>
#include <linux/dma-mapping.h>
#include <linux/err.h>
#include <linux/io.h>
#include <linux/iomap.h>
#include <linux/module.h>
#include <linux/pci.h>

#include "mmio.h"
#include "sdxi.h"

enum sdxi_mmio_bars {
	SDXI_PCI_BAR_CTL_REGS = 0,
	SDXI_PCI_BAR_DOORBELL = 2,
};

static struct pci_dev *sdxi_to_pci_dev(const struct sdxi_dev *sdxi)
{
	return to_pci_dev(sdxi->dev);
}

static int sdxi_pci_init(struct sdxi_dev *sdxi)
{
	struct pci_dev *pdev = sdxi_to_pci_dev(sdxi);
	struct device *dev = &pdev->dev;
	u32 cap1_max_cxt;
	u64 cap1;
	int vecs, ret;

	ret = pcim_enable_device(pdev);
	if (ret)
		return dev_err_probe(dev, ret, "failed to enable device\n");

	ret = dma_set_mask_and_coherent(dev, DMA_BIT_MASK(64));
	if (ret)
		return dev_err_probe(dev, ret, "failed to set DMA mask\n");

	sdxi->ctrl_regs = pcim_iomap_region(pdev, SDXI_PCI_BAR_CTL_REGS,
					    KBUILD_MODNAME);
	if (IS_ERR(sdxi->ctrl_regs))
		return dev_err_probe(dev, PTR_ERR(sdxi->ctrl_regs),
				     "failed to map control registers\n");

	cap1 = readq(sdxi->ctrl_regs + SDXI_MMIO_CAP1);
	if (!FIELD_GET(SDXI_MMIO_CAP1_MMIO64, cap1))
		return dev_err_probe(dev, -EOPNOTSUPP,
				     "64-bit atomic MMIO access required\n");

	sdxi->dbs = pcim_iomap_region(pdev, SDXI_PCI_BAR_DOORBELL,
				      KBUILD_MODNAME);
	if (IS_ERR(sdxi->dbs))
		return dev_err_probe(dev, PTR_ERR(sdxi->dbs),
				     "failed to map doorbell region\n");

	/* Mailbox registers are reserved without SR-IOV support. */
	sdxi->has_mailbox = pdev->is_virtfn ||
		pci_find_ext_capability(pdev, PCI_EXT_CAP_ID_SRIOV);

	/*
	 * Allocate the minimum required set of vectors plus one for
	 * each client context supported by the function.
	 *
	 * pci_alloc_irq_vectors() handles calling pci_free_irq_vectors()
	 * automatically despite not being called pcim_*. See
	 * pci_setup_msi_context().
	 */
	cap1_max_cxt = FIELD_GET(SDXI_MMIO_CAP1_MAX_CXT,
				 sdxi_read64(sdxi, SDXI_MMIO_CAP1));
	vecs = pci_alloc_irq_vectors(pdev, SDXI_MIN_VECTORS,
				     SDXI_MIN_VECTORS + cap1_max_cxt,
				     PCI_IRQ_MSI | PCI_IRQ_MSIX);
	if (vecs < 0)
		return dev_err_probe(dev, vecs,
				     "failed to allocate MSIs (max_cxt=%u)\n",
				     cap1_max_cxt);

	sdxi->nr_vectors = vecs;
	dev_dbg(sdxi->dev, "allocated %u vectors\n", sdxi->nr_vectors);

	pci_set_master(pdev);
	return 0;
}

static int sdxi_pci_reset(struct sdxi_dev *sdxi)
{
	struct pci_dev *pdev = sdxi_to_pci_dev(sdxi);
	int err;

	/* Probe and remove hold the device lock; preserve PCI configuration. */
	err = pci_reset_function_locked(pdev);
	if (err) {
		dev_err(&pdev->dev,
			"function reset failed (%d), disabling bus mastering\n", err);
		pci_clear_master(pdev);
		if (!pci_wait_for_pending_transaction(pdev))
			dev_warn(&pdev->dev,
				 "PCIe transactions still pending after disabling bus mastering\n");
	}

	return err;
}

static int sdxi_pci_get_irq(struct sdxi_dev *sdxi, unsigned int nr)
{
	return pci_irq_vector(sdxi_to_pci_dev(sdxi), nr);
}

static const struct sdxi_bus_ops sdxi_pci_ops = {
	.init = sdxi_pci_init,
	.reset = sdxi_pci_reset,
	.get_irq = sdxi_pci_get_irq,
};

static int sdxi_pci_probe(struct pci_dev *pdev,
			  const struct pci_device_id *id)
{
	return sdxi_register(&pdev->dev, &sdxi_pci_ops);
}

static void sdxi_pci_remove(struct pci_dev *pdev)
{
	sdxi_unregister(&pdev->dev);
	pci_disable_sriov(pdev);
}

static const struct pci_device_id sdxi_id_table[] = {
	{ PCI_DEVICE_CLASS(PCI_CLASS_ACCELERATOR_SDXI, 0xffffff) },
	{ }
};
MODULE_DEVICE_TABLE(pci, sdxi_id_table);

static struct pci_driver sdxi_driver = {
	.name = "sdxi",
	.id_table = sdxi_id_table,
	.probe = sdxi_pci_probe,
	.remove = sdxi_pci_remove,
	.sriov_configure = pci_sriov_configure_simple,
};

MODULE_IMPORT_NS("SDXI");
MODULE_AUTHOR("Wei Huang");
MODULE_AUTHOR("Nathan Lynch");
MODULE_DESCRIPTION("SDXI PCIe interface driver");
MODULE_LICENSE("GPL");
module_pci_driver(sdxi_driver);
