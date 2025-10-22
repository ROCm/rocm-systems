# Main Makefile for perf-pmu-stub module

MODULE_NAME := perf-pmu-stub
MODULE_VERSION := 1.0.0
KERNEL_VERSION := $(shell uname -r)

.PHONY: all clean install uninstall dkms-add dkms-build dkms-install dkms-remove test

all:
	$(MAKE) -C src

clean:
	$(MAKE) -C src clean

# DKMS targets
dkms-add:
	sudo cp -R . /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)/
	sudo dkms add -m $(MODULE_NAME) -v $(MODULE_VERSION)

dkms-build: dkms-add
	sudo dkms build -m $(MODULE_NAME) -v $(MODULE_VERSION)

dkms-install: dkms-build
	sudo dkms install -m $(MODULE_NAME) -v $(MODULE_VERSION)

dkms-remove:
	-sudo dkms remove -m $(MODULE_NAME) -v $(MODULE_VERSION) --all
	-sudo rm -rf /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)

# Quick test targets
load:
	sudo insmod src/amdgpu_pmu.ko

unload:
	-sudo rmmod amdgpu_pmu

reload: unload load

test: all
	@echo "Running module tests..."
	@bash test/load_test.sh

perf-test:
	@echo "Running perf integration tests..."
	@bash test/perf_test.sh

# Development helpers
check:
	@echo "Checking kernel headers..."
	@ls -la /lib/modules/$(KERNEL_VERSION)/build

info:
	@echo "Module: $(MODULE_NAME)"
	@echo "Version: $(MODULE_VERSION)"
	@echo "Kernel: $(KERNEL_VERSION)"
	@lsmod | grep amdgpu_pmu || echo "Module not loaded"

dmesg:
	@dmesg | tail -20