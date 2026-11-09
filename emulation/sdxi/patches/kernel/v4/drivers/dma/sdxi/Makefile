# SPDX-License-Identifier: GPL-2.0
obj-$(CONFIG_SDXI_CORE) += sdxi-core.o
sdxi-core-y := \
	completion.o  \
	context.o     \
	descriptor.o  \
	device.o      \
	dma.o         \
	dmabuf.o      \
	ring.o

obj-$(CONFIG_SDXI_PCI) += sdxi-pci.o
sdxi-pci-y := pci.o

obj-$(CONFIG_SDXI_KUNIT_TEST) += \
	completion_kunit.o    \
	descriptor_kunit.o    \
	ring_kunit.o
