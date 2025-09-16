# Linux Kernel Module Development Guide

A comprehensive guide for kernel module development, testing strategies, DKMS setup, and GPU performance monitoring workflows for 2024.

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [DKMS Configuration and Build Process](#dkms-configuration-and-build-process)
3. [Quick Iteration Workflow](#quick-iteration-workflow)
4. [Testing Strategies and Frameworks](#testing-strategies-and-frameworks)
5. [Debugging Tools and Techniques](#debugging-tools-and-techniques)
6. [Build System Examples and Makefiles](#build-system-examples-and-makefiles)
7. [CI/CD Pipeline Suggestions](#cicd-pipeline-suggestions)
8. [Performance Testing Approaches](#performance-testing-approaches)
9. [Code Examples for Common Patterns](#code-examples-for-common-patterns)
10. [GPU-Specific Development Considerations](#gpu-specific-development-considerations)
11. [Security and Stability Best Practices](#security-and-stability-best-practices)

## Development Environment Setup

### Modern DevContainer Approach (2024)

**Benefits:**
- Environment consistency across developers and CI/CD pipelines
- Quick start without complex local dependency configuration
- Isolation from host system
- Pre-configured development tools

**Setup:**
1. Clone a pre-configured Linux kernel development DevContainer project
2. Use virtme-ng for lightweight VM testing with compiled kernels
3. Configure VS Code with DevContainer extensions

```bash
# Example DevContainer configuration
git clone https://github.com/kernel-dev/devcontainer-setup
cd devcontainer-setup
code .  # Opens in DevContainer
```

### Traditional VM Setup

**Requirements:**
- QEMU/KVM for virtualization
- Multiple kernel versions for compatibility testing
- Shared filesystem between host and guest
- Serial console access for debugging

```bash
# VM setup example
qemu-system-x86_64 \
    -enable-kvm \
    -m 4G \
    -smp 4 \
    -kernel /boot/vmlinuz \
    -initrd /boot/initrd.img \
    -append "console=ttyS0" \
    -nographic
```

### Container Development Limitations

**Important Note:** Docker containers share the host kernel and cannot load kernel modules independently. For actual kernel module testing, you need:
- Virtual machines with isolated kernels
- Host system access with appropriate permissions
- Hybrid approach: containers for development, VMs for testing

## DKMS Configuration and Build Process

### DKMS Overview

Dynamic Kernel Module Support (DKMS) enables generating Linux kernel modules whose sources reside outside the kernel source tree. DKMS automatically rebuilds modules when new kernels are installed.

### Installation and Prerequisites

```bash
# Ubuntu/Debian
sudo apt install dkms build-essential linux-headers-$(uname -r)

# RHEL/CentOS
sudo yum install dkms gcc make kernel-devel

# Clear Linux
sudo swupd bundle-add kernel-native-dkms
```

### DKMS Configuration Structure

**Directory Layout:**
```
/usr/src/<modulename>-<version>/
├── dkms.conf
├── Makefile
├── src/
│   ├── module.c
│   └── module.h
└── patches/ (optional)
```

**Basic dkms.conf:**
```ini
PACKAGE_NAME="gpu-monitor"
PACKAGE_VERSION="1.0"
BUILT_MODULE_NAME[0]="gpu_monitor"
DEST_MODULE_LOCATION[0]="/extra"
MAKE[0]="make -C ${kernel_source_dir} M=${dkms_tree}/${PACKAGE_NAME}/${PACKAGE_VERSION}/build"
CLEAN="make -C ${kernel_source_dir} M=${dkms_tree}/${PACKAGE_NAME}/${PACKAGE_VERSION}/build clean"
REMAKE_INITRD="yes"
AUTOINSTALL="yes"
```

### DKMS Workflow Commands

```bash
# Add module to DKMS
sudo dkms add -m gpu-monitor -v 1.0

# Build module
sudo dkms build -m gpu-monitor -v 1.0

# Install module
sudo dkms install -m gpu-monitor -v 1.0

# Remove module
sudo dkms remove -m gpu-monitor -v 1.0 --all

# Status check
dkms status
```

### 2024 Security Considerations

**Module Signing:**
DKMS automatically generates self-signed certificates and signs modules at build time. For Secure Boot systems:

```bash
# Generate Machine Owner Key (MOK)
openssl req -new -x509 -newkey rsa:2048 -keyout MOK.priv -out MOK.der -days 36500 -nodes

# Enroll MOK
sudo mokutil --import MOK.der

# Sign module manually if needed
sudo /usr/src/linux-headers-$(uname -r)/scripts/sign-file sha256 MOK.priv MOK.der module.ko
```

## Quick Iteration Workflow

### Development Cycle

1. **Code Development:**
   ```bash
   # Edit source files
   vim src/gpu_monitor.c

   # Quick syntax check
   make -C /lib/modules/$(uname -r)/build M=$(PWD) modules_check
   ```

2. **Build and Test:**
   ```bash
   # DKMS build
   sudo dkms build -m gpu-monitor -v 1.0

   # Manual build for testing
   make -C /lib/modules/$(uname -r)/build M=$(PWD) modules
   ```

3. **Load and Test:**
   ```bash
   # Unload existing module
   sudo rmmod gpu_monitor

   # Load new version
   sudo insmod gpu_monitor.ko

   # Check logs
   dmesg | tail -20
   ```

4. **Debugging Iteration:**
   ```bash
   # Enable dynamic debugging
   echo 'module gpu_monitor +p' > /sys/kernel/debug/dynamic_debug/control

   # Monitor with ftrace
   echo 1 > /sys/kernel/debug/tracing/events/module/enable
   ```

### Automated Testing Script

```bash
#!/bin/bash
# quick-test.sh

MODULE_NAME="gpu_monitor"
MODULE_VERSION="1.0"

# Build
if ! sudo dkms build -m $MODULE_NAME -v $MODULE_VERSION; then
    echo "Build failed"
    exit 1
fi

# Unload if loaded
if lsmod | grep -q $MODULE_NAME; then
    sudo rmmod $MODULE_NAME
fi

# Install and load
sudo dkms install -m $MODULE_NAME -v $MODULE_VERSION
sudo modprobe $MODULE_NAME

# Basic functionality test
if lsmod | grep -q $MODULE_NAME; then
    echo "Module loaded successfully"
    dmesg | tail -10
else
    echo "Module load failed"
    exit 1
fi
```

## Testing Strategies and Frameworks

### KUnit - Linux Kernel Unit Testing Framework

**Overview:**
KUnit is the primary unit testing framework for the Linux kernel, available since version 5.5. It provides lightweight, in-kernel testing capabilities.

**Setup:**
```bash
# Enable KUnit in kernel config
CONFIG_KUNIT=y
CONFIG_KUNIT_ALL_TESTS=y

# Run tests
./tools/testing/kunit/kunit.py run
```

**Example Test:**
```c
#include <kunit/test.h>

static void gpu_monitor_init_test(struct kunit *test)
{
    int result = gpu_monitor_init();
    KUNIT_EXPECT_EQ(test, result, 0);
}

static struct kunit_case gpu_monitor_test_cases[] = {
    KUNIT_CASE(gpu_monitor_init_test),
    {}
};

static struct kunit_suite gpu_monitor_test_suite = {
    .name = "gpu_monitor",
    .test_cases = gpu_monitor_test_cases,
};

kunit_test_suites(&gpu_monitor_test_suite);
```

### KTF (Kernel Test Framework) by Oracle

**Features:**
- Google Test-like environment for kernel code
- Coverage support with ktfcov
- Tests implemented as kernel modules

**Example:**
```c
#include "ktf.h"

TEST(gpu_monitor, basic_functionality)
{
    int ret = gpu_monitor_function();
    EXPECT_EQ(ret, EXPECTED_VALUE);
}

static struct ktf_test gpu_monitor_tests[] = {
    KTF_TEST_CASE(gpu_monitor, basic_functionality),
    KTF_TEST_CASE_NULL
};

KTF_MODULE_INIT(gpu_monitor_tests);
```

### Integration Testing with kselftest

**Setup:**
```bash
# Run kernel selftests
cd tools/testing/selftests
make run_tests
```

**Custom Test Script:**
```bash
#!/bin/bash
# gpu_monitor_selftest.sh

# Load module
modprobe gpu_monitor

# Test sysfs interfaces
if [ -d /sys/class/gpu_monitor ]; then
    echo "PASS: sysfs interface created"
else
    echo "FAIL: sysfs interface missing"
    exit 1
fi

# Test module parameters
echo "performance" > /sys/module/gpu_monitor/parameters/mode
if [ $? -eq 0 ]; then
    echo "PASS: parameter setting"
else
    echo "FAIL: parameter setting"
    exit 1
fi
```

## Debugging Tools and Techniques

### printk - Foundation Debugging

**Basic Usage:**
```c
#include <linux/kernel.h>

// Different log levels
printk(KERN_DEBUG "Debug message\n");
printk(KERN_INFO "Info message\n");
printk(KERN_WARNING "Warning message\n");
printk(KERN_ERR "Error message\n");

// Dynamic debugging
pr_debug("Debug with dynamic control\n");
```

**Control Dynamic Debugging:**
```bash
# Enable debug messages for module
echo 'module gpu_monitor +p' > /sys/kernel/debug/dynamic_debug/control

# Enable for specific file
echo 'file gpu_monitor.c +p' > /sys/kernel/debug/dynamic_debug/control

# View current settings
cat /sys/kernel/debug/dynamic_debug/control | grep gpu_monitor
```

### ftrace - Advanced Tracing

**Setup:**
```bash
# Mount debugfs if not mounted
mount -t debugfs none /sys/kernel/debug

# Enable function tracing
echo function > /sys/kernel/debug/tracing/current_tracer

# Trace specific functions
echo gpu_monitor_* > /sys/kernel/debug/tracing/set_ftrace_filter

# Start tracing
echo 1 > /sys/kernel/debug/tracing/tracing_on
```

**trace_printk Usage:**
```c
#include <linux/ftrace.h>

static int gpu_monitor_function(void)
{
    trace_printk("Entering gpu_monitor_function\n");

    // Function logic here
    int result = do_something();

    trace_printk("Result: %d\n", result);
    return result;
}
```

**View Trace Output:**
```bash
# Read trace buffer
cat /sys/kernel/debug/tracing/trace

# Live monitoring
cat /sys/kernel/debug/tracing/trace_pipe
```

### KGDB - Interactive Debugging

**Setup Requirements:**
- Two machines (development and target)
- Serial connection or network connection
- Kernel compiled with KGDB support

**Kernel Configuration:**
```
CONFIG_KGDB=y
CONFIG_KGDB_SERIAL_CONSOLE=y
CONFIG_KGDB_KDB=y
```

**Boot Parameters:**
```bash
# Serial connection
linux kgdboc=ttyS0,115200 kgdbwait

# Network connection
linux kgdboe=@192.168.1.100/,@192.168.1.101/
```

**GDB Session:**
```bash
# On development machine
gdb vmlinux
(gdb) target remote /dev/ttyS0
(gdb) continue
```

### Advanced Debugging Techniques

**Memory Debugging:**
```c
#include <linux/slab.h>
#include <linux/kmemleak.h>

// Allocate with debugging
void *ptr = kmalloc(size, GFP_KERNEL);
kmemleak_alloc(ptr, size, 1, GFP_KERNEL);

// Free with debugging
kmemleak_free(ptr);
kfree(ptr);
```

**Lock Debugging:**
```bash
# Enable lock debugging in kernel config
CONFIG_PROVE_LOCKING=y
CONFIG_LOCKDEP=y
CONFIG_LOCK_STAT=y
```

## Build System Examples and Makefiles

### Basic Kernel Module Makefile

```makefile
# Simple module
obj-m := gpu_monitor.o

# Multiple object files
gpu_monitor-objs := main.o utils.o hardware.o

# Build commands
KVERSION := $(shell uname -r)
KDIR := /lib/modules/$(KVERSION)/build
PWD := $(shell pwd)

all:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

# DKMS integration
dkms-add:
	sudo dkms add -m gpu_monitor -v 1.0

dkms-build:
	sudo dkms build -m gpu_monitor -v 1.0

dkms-install:
	sudo dkms install -m gpu_monitor -v 1.0
```

### Advanced Makefile with Cross-Compilation

```makefile
# Cross-compilation support
ifeq ($(ARCH),)
ARCH := $(shell uname -m)
endif

ifeq ($(ARCH),arm64)
CROSS_COMPILE ?= aarch64-linux-gnu-
KDIR ?= /lib/modules/$(KVERSION)/build
endif

# Module information
MODULE_NAME := gpu_monitor
MODULE_VERSION := 1.0

# Source files
$(MODULE_NAME)-objs := main.o device.o performance.o

obj-m := $(MODULE_NAME).o

# Compiler flags
ccflags-y := -DMODULE_VERSION=\"$(MODULE_VERSION)\"
ccflags-y += -DDEBUG

# Build targets
all: modules

modules:
	$(MAKE) -C $(KDIR) M=$(PWD) modules

modules_install:
	$(MAKE) -C $(KDIR) M=$(PWD) modules_install

clean:
	$(MAKE) -C $(KDIR) M=$(PWD) clean

# Testing targets
test: modules
	sudo insmod $(MODULE_NAME).ko
	@echo "Module loaded, checking dmesg..."
	@dmesg | tail -10
	sudo rmmod $(MODULE_NAME)

# DKMS targets
dkms-prepare:
	sudo mkdir -p /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)
	sudo cp -r * /usr/src/$(MODULE_NAME)-$(MODULE_VERSION)/
	sudo dkms add -m $(MODULE_NAME) -v $(MODULE_VERSION)

dkms-build: dkms-prepare
	sudo dkms build -m $(MODULE_NAME) -v $(MODULE_VERSION)

dkms-install: dkms-build
	sudo dkms install -m $(MODULE_NAME) -v $(MODULE_VERSION)

.PHONY: all modules clean test dkms-prepare dkms-build dkms-install
```

### Kbuild File for Complex Projects

```makefile
# Kbuild file for organized build
obj-$(CONFIG_GPU_MONITOR) += gpu_monitor.o

gpu_monitor-y := main.o
gpu_monitor-y += device/device_interface.o
gpu_monitor-y += device/hardware_access.o
gpu_monitor-y += performance/metrics.o
gpu_monitor-y += performance/profiling.o

# Conditional compilation
gpu_monitor-$(CONFIG_GPU_MONITOR_DEBUG) += debug/debug_interface.o
gpu_monitor-$(CONFIG_GPU_MONITOR_TRACE) += debug/trace_support.o

# Include paths
ccflags-y += -I$(src)/include
ccflags-y += -I$(src)/device
ccflags-y += -I$(src)/performance

# Debug flags
ifdef CONFIG_GPU_MONITOR_DEBUG
ccflags-y += -DDEBUG -g
endif
```

### 2024 Build System Updates

**New Linux 6.13 Syntax:**
```makefile
# Traditional approach
all:
	make -C /lib/modules/$(shell uname -r)/build M=$(PWD) modules

# New approach (Linux 6.13+)
all:
	make -f /lib/modules/$(shell uname -r)/build/Makefile M=$(PWD)
```

## CI/CD Pipeline Suggestions

### GitLab CI Pipeline for Kernel Modules

```yaml
# .gitlab-ci.yml
stages:
  - build
  - test
  - deploy

variables:
  MODULE_NAME: "gpu_monitor"
  MODULE_VERSION: "1.0"

before_script:
  - apt-get update -qq
  - apt-get install -y build-essential linux-headers-$(uname -r) dkms

build_module:
  stage: build
  script:
    - make modules
    - make dkms-prepare
    - dkms build -m $MODULE_NAME -v $MODULE_VERSION
  artifacts:
    paths:
      - "*.ko"
    expire_in: 1 hour

test_module:
  stage: test
  script:
    - insmod ${MODULE_NAME}.ko
    - lsmod | grep $MODULE_NAME
    - rmmod $MODULE_NAME
    - echo "Module test passed"
  dependencies:
    - build_module

deploy_dkms:
  stage: deploy
  script:
    - dkms install -m $MODULE_NAME -v $MODULE_VERSION
  only:
    - main
  dependencies:
    - build_module
```

### GitHub Actions Workflow

```yaml
# .github/workflows/kernel-module.yml
name: Kernel Module CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  build-and-test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        kernel: ['5.15', '6.1', '6.6']

    steps:
    - uses: actions/checkout@v4

    - name: Setup kernel headers
      run: |
        sudo apt-get update
        sudo apt-get install -y linux-headers-${{ matrix.kernel }} build-essential dkms

    - name: Build module
      run: |
        make modules

    - name: Test with KUnit
      run: |
        ./tools/testing/kunit/kunit.py run gpu_monitor_test

    - name: DKMS test
      run: |
        sudo make dkms-prepare
        sudo dkms build -m gpu_monitor -v 1.0

    - name: Upload artifacts
      uses: actions/upload-artifact@v4
      with:
        name: kernel-module-${{ matrix.kernel }}
        path: '*.ko'
```

### Continuous Kernel Integration (CKI) Style

**Pipeline Features:**
- Multi-architecture compilation (x86_64, arm64, ppc64le, s390x)
- Multiple kernel version testing
- Automated patch application
- Regression testing
- Performance benchmarking

```bash
#!/bin/bash
# cki-style-pipeline.sh

ARCHITECTURES="x86_64 arm64"
KERNEL_VERSIONS="5.15 6.1 6.6"
MODULE_NAME="gpu_monitor"

for arch in $ARCHITECTURES; do
    for kver in $KERNEL_VERSIONS; do
        echo "Building for $arch kernel $kver"

        # Cross-compile setup
        export ARCH=$arch
        export CROSS_COMPILE=${arch}-linux-gnu-

        # Build
        make -C /lib/modules/$kver/build M=$(pwd) modules

        if [ $? -eq 0 ]; then
            echo "✓ Build successful for $arch-$kver"
        else
            echo "✗ Build failed for $arch-$kver"
            exit 1
        fi
    done
done
```

## Performance Testing Approaches

### XDMoD Application Kernel Module Style

**Continuous Performance Monitoring:**
```bash
#!/bin/bash
# performance-monitor.sh

MODULE_NAME="gpu_monitor"
TEST_DURATION=300  # 5 minutes
RESULTS_DIR="/var/log/kernel-perf"

# Load module
modprobe $MODULE_NAME

# Start performance monitoring
perf record -g -o $RESULTS_DIR/perf-$$.data &
PERF_PID=$!

# Run workload
echo "Starting performance test..."
./gpu_workload_generator --duration=$TEST_DURATION

# Stop monitoring
kill $PERF_PID
wait $PERF_PID

# Generate report
perf report -i $RESULTS_DIR/perf-$$.data > $RESULTS_DIR/report-$$.txt

# Unload module
rmmod $MODULE_NAME

echo "Performance test completed. Results in $RESULTS_DIR/"
```

### Linux Perf Integration

```bash
# Module-specific profiling
perf record -e 'module:*' -g ./test_gpu_monitor

# Function-level profiling
perf record -e 'probe:gpu_monitor_*' -g ./test_gpu_monitor

# Memory access profiling
perf mem record -a ./test_gpu_monitor

# Generate flamegraph
perf script | ./flamegraph.pl > gpu_monitor_flamegraph.svg
```

### MMTests Framework Integration

```bash
# MMTests configuration for kernel module
cat > configs/config-gpu-monitor << EOF
export MMTESTS="gpu-monitor-benchmark"
export RUN_MONITOR=yes
export MONITOR_PERF_EVENTS="cycles,instructions,cache-misses"
export MONITOR_FTRACE_EVENTS="module:*"
EOF

# Run tests
./run-mmtests.sh --config configs/config-gpu-monitor
```

### Benchmarking with Phoronix Test Suite

```bash
# Create custom test profile
phoronix-test-suite build-test gpu-monitor-test

# Run with perf integration
LINUX_PERF=1 phoronix-test-suite run gpu-monitor-test

# Automated comparison
phoronix-test-suite benchmark gpu-monitor-test
```

## Code Examples for Common Patterns

### Module Initialization and Cleanup

```c
#include <linux/init.h>
#include <linux/module.h>
#include <linux/kernel.h>
#include <linux/device.h>
#include <linux/cdev.h>
#include <linux/fs.h>

#define DEVICE_NAME "gpu_monitor"
#define CLASS_NAME "gpu_monitor"

static int major_number;
static struct class* gpu_monitor_class = NULL;
static struct device* gpu_monitor_device = NULL;

static int __init gpu_monitor_init(void)
{
    int ret;

    printk(KERN_INFO "GPU Monitor: Initializing module\n");

    // Register major number
    major_number = register_chrdev(0, DEVICE_NAME, &fops);
    if (major_number < 0) {
        printk(KERN_ALERT "GPU Monitor: Failed to register major number\n");
        return major_number;
    }

    // Create device class
    gpu_monitor_class = class_create(THIS_MODULE, CLASS_NAME);
    if (IS_ERR(gpu_monitor_class)) {
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "GPU Monitor: Failed to create device class\n");
        return PTR_ERR(gpu_monitor_class);
    }

    // Create device
    gpu_monitor_device = device_create(gpu_monitor_class, NULL,
                                     MKDEV(major_number, 0), NULL, DEVICE_NAME);
    if (IS_ERR(gpu_monitor_device)) {
        class_destroy(gpu_monitor_class);
        unregister_chrdev(major_number, DEVICE_NAME);
        printk(KERN_ALERT "GPU Monitor: Failed to create device\n");
        return PTR_ERR(gpu_monitor_device);
    }

    printk(KERN_INFO "GPU Monitor: Module initialized successfully\n");
    return 0;
}

static void __exit gpu_monitor_exit(void)
{
    device_destroy(gpu_monitor_class, MKDEV(major_number, 0));
    class_unregister(gpu_monitor_class);
    class_destroy(gpu_monitor_class);
    unregister_chrdev(major_number, DEVICE_NAME);
    printk(KERN_INFO "GPU Monitor: Module cleanup completed\n");
}

module_init(gpu_monitor_init);
module_exit(gpu_monitor_exit);

MODULE_LICENSE("GPL");
MODULE_AUTHOR("Your Name");
MODULE_DESCRIPTION("GPU Performance Monitor Module");
MODULE_VERSION("1.0");
```

### Sysfs Interface Pattern

```c
#include <linux/sysfs.h>
#include <linux/kobject.h>

static struct kobject *gpu_monitor_kobj;

// Attribute show function
static ssize_t performance_mode_show(struct kobject *kobj,
                                   struct kobj_attribute *attr, char *buf)
{
    return sprintf(buf, "%s\n", current_performance_mode);
}

// Attribute store function
static ssize_t performance_mode_store(struct kobject *kobj,
                                    struct kobj_attribute *attr,
                                    const char *buf, size_t count)
{
    if (strncmp(buf, "high", 4) == 0) {
        set_performance_mode(HIGH_PERFORMANCE);
    } else if (strncmp(buf, "balanced", 8) == 0) {
        set_performance_mode(BALANCED);
    } else if (strncmp(buf, "power_save", 10) == 0) {
        set_performance_mode(POWER_SAVE);
    } else {
        return -EINVAL;
    }

    return count;
}

// Define attribute
static struct kobj_attribute performance_mode_attribute =
    __ATTR(performance_mode, 0664, performance_mode_show, performance_mode_store);

// Create sysfs interface
static int create_sysfs_interface(void)
{
    int retval;

    gpu_monitor_kobj = kobject_create_and_add("gpu_monitor", kernel_kobj);
    if (!gpu_monitor_kobj)
        return -ENOMEM;

    retval = sysfs_create_file(gpu_monitor_kobj, &performance_mode_attribute.attr);
    if (retval)
        kobject_put(gpu_monitor_kobj);

    return retval;
}

// Cleanup sysfs interface
static void cleanup_sysfs_interface(void)
{
    kobject_put(gpu_monitor_kobj);
}
```

### Memory Management Pattern

```c
#include <linux/slab.h>
#include <linux/vmalloc.h>
#include <linux/mm.h>

struct gpu_monitor_data {
    struct list_head list;
    void *buffer;
    size_t size;
    dma_addr_t dma_handle;
};

static LIST_HEAD(gpu_monitor_buffers);
static DEFINE_MUTEX(buffer_mutex);

// Allocate DMA-coherent memory
static int allocate_gpu_buffer(size_t size, struct gpu_monitor_data **data)
{
    struct gpu_monitor_data *new_data;

    new_data = kzalloc(sizeof(*new_data), GFP_KERNEL);
    if (!new_data)
        return -ENOMEM;

    new_data->buffer = dma_alloc_coherent(&pdev->dev, size,
                                        &new_data->dma_handle, GFP_KERNEL);
    if (!new_data->buffer) {
        kfree(new_data);
        return -ENOMEM;
    }

    new_data->size = size;
    INIT_LIST_HEAD(&new_data->list);

    mutex_lock(&buffer_mutex);
    list_add(&new_data->list, &gpu_monitor_buffers);
    mutex_unlock(&buffer_mutex);

    *data = new_data;
    return 0;
}

// Free GPU buffer
static void free_gpu_buffer(struct gpu_monitor_data *data)
{
    mutex_lock(&buffer_mutex);
    list_del(&data->list);
    mutex_unlock(&buffer_mutex);

    dma_free_coherent(&pdev->dev, data->size, data->buffer, data->dma_handle);
    kfree(data);
}

// Cleanup all buffers
static void cleanup_all_buffers(void)
{
    struct gpu_monitor_data *data, *tmp;

    mutex_lock(&buffer_mutex);
    list_for_each_entry_safe(data, tmp, &gpu_monitor_buffers, list) {
        list_del(&data->list);
        dma_free_coherent(&pdev->dev, data->size, data->buffer, data->dma_handle);
        kfree(data);
    }
    mutex_unlock(&buffer_mutex);
}
```

### Interrupt Handling Pattern

```c
#include <linux/interrupt.h>
#include <linux/workqueue.h>

static struct workqueue_struct *gpu_monitor_wq;

// Bottom half handler
static void gpu_monitor_work_handler(struct work_struct *work)
{
    // Process GPU performance data
    process_gpu_metrics();

    // Update statistics
    update_performance_counters();

    // Notify userspace if needed
    sysfs_notify(gpu_monitor_kobj, NULL, "performance_data");
}

static DECLARE_WORK(gpu_monitor_work, gpu_monitor_work_handler);

// Top half interrupt handler
static irqreturn_t gpu_monitor_interrupt(int irq, void *dev_id)
{
    // Acknowledge interrupt
    clear_gpu_interrupt();

    // Schedule bottom half
    queue_work(gpu_monitor_wq, &gpu_monitor_work);

    return IRQ_HANDLED;
}

// Setup interrupt handling
static int setup_interrupt_handling(void)
{
    int ret;

    // Create workqueue
    gpu_monitor_wq = create_singlethread_workqueue("gpu_monitor");
    if (!gpu_monitor_wq)
        return -ENOMEM;

    // Request IRQ
    ret = request_irq(gpu_irq, gpu_monitor_interrupt, IRQF_SHARED,
                     "gpu_monitor", &gpu_monitor_device);
    if (ret) {
        destroy_workqueue(gpu_monitor_wq);
        return ret;
    }

    return 0;
}

// Cleanup interrupt handling
static void cleanup_interrupt_handling(void)
{
    free_irq(gpu_irq, &gpu_monitor_device);
    destroy_workqueue(gpu_monitor_wq);
}
```

## GPU-Specific Development Considerations

### ROCm Integration Patterns

**AMDGPU Driver Interface:**
```c
#include <drm/amd_asic_type.h>
#include <amdgpu.h>

// Check for ROCm-compatible GPU
static bool is_rocm_supported_gpu(struct pci_dev *pdev)
{
    // Check vendor ID (AMD)
    if (pdev->vendor != 0x1002)
        return false;

    // Check device ID for supported GPUs
    switch (pdev->device) {
    case 0x73df:  // RX 6700 XT
    case 0x1002:  // Other supported devices
        return true;
    default:
        return false;
    }
}

// Access GPU performance counters
static int read_gpu_performance_counters(struct gpu_perf_data *data)
{
    // Access through AMDGPU driver interfaces
    struct amdgpu_device *adev = get_amdgpu_device();

    if (!adev)
        return -ENODEV;

    // Read performance metrics
    data->gpu_utilization = amdgpu_get_gpu_utilization(adev);
    data->memory_utilization = amdgpu_get_memory_utilization(adev);
    data->temperature = amdgpu_get_temperature(adev);
    data->power_consumption = amdgpu_get_power_usage(adev);

    return 0;
}
```

**HSA Kernel Driver (KFD) Integration:**
```c
#include <linux/kfd_ioctl.h>

// Monitor HSA compute workloads
static int monitor_hsa_workloads(void)
{
    struct kfd_process *process;
    struct kfd_queue *queue;

    // Iterate through KFD processes
    hash_for_each(kfd_processes_table, process) {
        if (!process)
            continue;

        // Monitor each queue in the process
        list_for_each_entry(queue, &process->pqm.queues, list) {
            // Collect queue statistics
            collect_queue_stats(queue);
        }
    }

    return 0;
}
```

### GPU Memory Management

```c
#include <linux/dma-mapping.h>

// GPU memory allocation for performance monitoring
static int allocate_gpu_monitoring_buffers(void)
{
    struct gpu_monitor_buffers *buffers;

    buffers = kzalloc(sizeof(*buffers), GFP_KERNEL);
    if (!buffers)
        return -ENOMEM;

    // Allocate command buffer
    buffers->cmd_buffer = dma_alloc_coherent(&gpu_pdev->dev,
                                           CMD_BUFFER_SIZE,
                                           &buffers->cmd_dma,
                                           GFP_KERNEL);

    // Allocate result buffer
    buffers->result_buffer = dma_alloc_coherent(&gpu_pdev->dev,
                                              RESULT_BUFFER_SIZE,
                                              &buffers->result_dma,
                                              GFP_KERNEL);

    if (!buffers->cmd_buffer || !buffers->result_buffer) {
        cleanup_gpu_buffers(buffers);
        return -ENOMEM;
    }

    return 0;
}
```

### GPU Target Configuration

```c
// GPU architecture detection
static enum gpu_arch detect_gpu_architecture(void)
{
    struct pci_dev *pdev = gpu_pdev;

    switch (pdev->device) {
    case 0x73df:  // Navi 22 (RDNA2)
        return GPU_ARCH_RDNA2;
    case 0x1681:  // Vega 10
        return GPU_ARCH_VEGA;
    case 0x7310:  // Navi 31 (RDNA3)
        return GPU_ARCH_RDNA3;
    default:
        return GPU_ARCH_UNKNOWN;
    }
}

// Architecture-specific performance monitoring
static int setup_arch_specific_monitoring(enum gpu_arch arch)
{
    switch (arch) {
    case GPU_ARCH_RDNA2:
        return setup_rdna2_monitoring();
    case GPU_ARCH_RDNA3:
        return setup_rdna3_monitoring();
    case GPU_ARCH_VEGA:
        return setup_vega_monitoring();
    default:
        return -ENOTSUP;
    }
}
```

## Security and Stability Best Practices

### Input Validation and Sanitization

```c
// Validate user input from sysfs
static ssize_t config_store(struct kobject *kobj, struct kobj_attribute *attr,
                          const char *buf, size_t count)
{
    long value;
    int ret;

    // Validate input length
    if (count > MAX_CONFIG_LENGTH) {
        pr_err("gpu_monitor: Input too long\n");
        return -EINVAL;
    }

    // Parse and validate numeric input
    ret = kstrtol(buf, 10, &value);
    if (ret) {
        pr_err("gpu_monitor: Invalid numeric input\n");
        return ret;
    }

    // Range validation
    if (value < MIN_CONFIG_VALUE || value > MAX_CONFIG_VALUE) {
        pr_err("gpu_monitor: Value out of range [%d, %d]\n",
               MIN_CONFIG_VALUE, MAX_CONFIG_VALUE);
        return -ERANGE;
    }

    // Apply configuration
    return apply_config(value) ? count : -EIO;
}
```

### Memory Safety

```c
// Safe memory operations
static int copy_performance_data(struct gpu_perf_data __user *user_data,
                               struct gpu_perf_data *kernel_data)
{
    // Validate user pointer
    if (!access_ok(user_data, sizeof(*user_data)))
        return -EFAULT;

    // Use safe copy functions
    if (copy_to_user(user_data, kernel_data, sizeof(*kernel_data)))
        return -EFAULT;

    return 0;
}

// Buffer overflow protection
static int safe_string_copy(char *dest, const char *src, size_t dest_size)
{
    size_t src_len = strlen(src);

    if (src_len >= dest_size)
        return -ENAMETOOLONG;

    strncpy(dest, src, dest_size - 1);
    dest[dest_size - 1] = '\0';

    return 0;
}
```

### Error Handling and Recovery

```c
// Robust error handling
static int gpu_monitor_operation(void)
{
    int ret = 0;
    struct gpu_context *ctx = NULL;

    // Allocate resources
    ctx = kzalloc(sizeof(*ctx), GFP_KERNEL);
    if (!ctx) {
        ret = -ENOMEM;
        goto out;
    }

    // Initialize hardware
    ret = init_gpu_hardware(ctx);
    if (ret) {
        pr_err("gpu_monitor: Hardware initialization failed: %d\n", ret);
        goto cleanup_ctx;
    }

    // Perform operation
    ret = perform_gpu_operation(ctx);
    if (ret) {
        pr_err("gpu_monitor: Operation failed: %d\n", ret);
        goto cleanup_hw;
    }

    // Success path
    goto cleanup_hw;

cleanup_hw:
    cleanup_gpu_hardware(ctx);
cleanup_ctx:
    kfree(ctx);
out:
    return ret;
}
```

### Module Parameter Security

```c
#include <linux/moduleparam.h>

// Secure module parameters
static char *allowed_modes[] = {"performance", "balanced", "power_save"};
static char *mode = "balanced";
static bool debug_enabled = false;
static int max_clients = 16;

// Parameter validation callback
static int validate_mode(const char *val, const struct kernel_param *kp)
{
    int i;

    for (i = 0; i < ARRAY_SIZE(allowed_modes); i++) {
        if (strcmp(val, allowed_modes[i]) == 0)
            return param_set_charp(val, kp);
    }

    pr_err("gpu_monitor: Invalid mode '%s'\n", val);
    return -EINVAL;
}

static int validate_max_clients(const char *val, const struct kernel_param *kp)
{
    int value;
    int ret = kstrtoint(val, 10, &value);

    if (ret)
        return ret;

    if (value < 1 || value > 64) {
        pr_err("gpu_monitor: max_clients must be between 1 and 64\n");
        return -EINVAL;
    }

    return param_set_int(val, kp);
}

static const struct kernel_param_ops mode_ops = {
    .set = validate_mode,
    .get = param_get_charp,
};

static const struct kernel_param_ops max_clients_ops = {
    .set = validate_max_clients,
    .get = param_get_int,
};

module_param_cb(mode, &mode_ops, &mode, 0644);
module_param_cb(max_clients, &max_clients_ops, &max_clients, 0644);
module_param(debug_enabled, bool, 0644);

MODULE_PARM_DESC(mode, "Operating mode: performance, balanced, or power_save");
MODULE_PARM_DESC(max_clients, "Maximum number of concurrent clients (1-64)");
MODULE_PARM_DESC(debug_enabled, "Enable debug output");
```

### Locking and Concurrency

```c
#include <linux/mutex.h>
#include <linux/spinlock.h>
#include <linux/rwlock.h>

// Multiple lock types for different use cases
static DEFINE_MUTEX(gpu_monitor_mutex);        // For configuration changes
static DEFINE_SPINLOCK(stats_lock);            // For statistics updates
static DEFINE_RWLOCK(client_list_lock);        // For client list access

// Safe critical section pattern
static int update_gpu_configuration(struct gpu_config *new_config)
{
    int ret = 0;

    // Use mutex for configuration changes (can sleep)
    if (mutex_lock_interruptible(&gpu_monitor_mutex))
        return -ERESTARTSYS;

    // Validate configuration
    ret = validate_gpu_config(new_config);
    if (ret)
        goto unlock;

    // Apply configuration
    ret = apply_gpu_config(new_config);
    if (ret)
        pr_err("gpu_monitor: Failed to apply configuration: %d\n", ret);

unlock:
    mutex_unlock(&gpu_monitor_mutex);
    return ret;
}

// Fast statistics update (atomic context)
static void update_performance_stats(struct perf_stats *stats)
{
    unsigned long flags;

    spin_lock_irqsave(&stats_lock, flags);

    // Update statistics atomically
    global_stats.samples++;
    global_stats.total_gpu_time += stats->gpu_time;
    global_stats.total_memory_usage += stats->memory_usage;

    spin_unlock_irqrestore(&stats_lock, flags);
}
```

## Conclusion

This comprehensive guide provides a complete framework for Linux kernel module development with a focus on GPU performance monitoring. The combination of modern development environments (DevContainers), robust testing frameworks (KUnit, KTF), advanced debugging tools (ftrace, KGDB), and comprehensive CI/CD pipelines enables rapid development and deployment of stable, secure kernel modules.

Key takeaways for 2024:

1. **Use DKMS** for automatic kernel compatibility across updates
2. **Implement comprehensive testing** with KUnit for unit tests and custom integration tests
3. **Leverage modern debugging tools** like ftrace for performance analysis
4. **Follow security best practices** with input validation and proper error handling
5. **Use CI/CD pipelines** for automated testing across multiple kernel versions and architectures
6. **Consider GPU-specific requirements** for ROCm integration and performance monitoring

The examples and patterns provided in this guide offer practical, actionable code that can be adapted for specific GPU performance monitoring requirements while maintaining high standards for security, stability, and maintainability.