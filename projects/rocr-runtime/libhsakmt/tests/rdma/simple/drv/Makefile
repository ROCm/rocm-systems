
ifeq ($(src),)
	src := $$PWD
endif

ifneq ($(KERNELRELEASE),)

ifeq ("$(wildcard $(RDMA_HEADER_DIR)/drm/amd_rdma.h)","")
$(error amd_rdma.h not found, please export RDMA_HEADER_DIR=/usr/src/amdgpu-x.x.x/include)
endif

obj-m += amdp2ptest.o

ccflags-y += -I $(RDMA_HEADER_DIR)

else

KDIR ?= /lib/modules/$(shell uname -r)/build

all: default


default:
	@ $(MAKE) -C $(KDIR) M=$$PWD  modules

install:
	$(MAKE) -C $(KDIR) M=$$PWD  modules_install

help:
	$(MAKE) -C $(KDIR) M=$$PWD help

clean:
	rm -rf *.o *.ko* *.mod* .*.cmd Module.symvers modules.order .tmp_versions/ *~ core .depend TAGS

TAGS:
	find $(KERNELDIR) -follow -name \*.h -o -name \*.c  |xargs etags

.PHONY: clean all help install default linksyms

endif
