# Standard kernel module Makefile
obj-m := pmu_stub.o
pmu_stub-y := pmu_main.o pmu_events.o

# Kernel version and build directory
KVERSION := $(shell uname -r)
KERNELDIR ?= /lib/modules/$(KVERSION)/build

# Compiler flags
ccflags-y := -DDEBUG -g -Wall

# Default build target
all:
	@echo "Building kernel module..."
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules
	@if [ -f pmu_stub.ko ]; then \
		echo "Module successfully built: pmu_stub.ko"; \
		echo "Vermagic check:"; \
		strings pmu_stub.ko | grep vermagic || echo "WARNING: No vermagic found"; \
	else \
		echo "Build failed - no .ko file created"; \
		exit 1; \
	fi

# Force the build to ignore __modfinal errors and continue
force-build:
	@echo "Force building with error suppression..."
	-$(MAKE) -C $(KERNELDIR) M=$(PWD) modules 2>&1 | grep -v "__modfinal" || true
	@if [ -f pmu_stub.o ] && [ ! -f pmu_stub.ko ]; then \
		echo "Kernel build system failed to create .ko file"; \
		echo "This is a known bug in kernel 6.17.0-rc4-rocm-gdb"; \
		echo "The build system stops at modfinal stage"; \
		echo "Manual workaround cannot preserve vermagic - module will not load"; \
		exit 1; \
	fi

# Clean target
clean:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) clean

# Install target
install:
	$(MAKE) -C $(KERNELDIR) M=$(PWD) modules_install
	depmod -a

.PHONY: all force-build clean install