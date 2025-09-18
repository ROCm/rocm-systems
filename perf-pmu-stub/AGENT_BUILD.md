# Build Instructions for perf-pmu-stub Kernel Module
## Machine: kernel-vm
## Generated: 2025-09-17

### System Information
- OS: Linux 6.17.0-rc4-rocm-gdb #1 SMP PREEMPT_DYNAMIC
- Architecture: x86_64
- Shell: /bin/bash
- Working Directory: /home/ben/rocm-systems/perf-pmu-stub

### Project Overview
The perf-pmu-stub is a Linux kernel module that implements a PMU (Performance Monitoring Unit) driver for the perf subsystem. It includes:
- Core PMU driver implementation
- AQL (AMD Queue Language) C library integration
- KFD (Kernel Fusion Driver) integration components
- Performance monitoring capabilities for AMD GPU systems

### Prerequisites

#### Required Packages
- build-essential: 12.10ubuntu1 (installed at /usr/bin/gcc, /usr/bin/make)
- GCC Compiler: 13.3.0 (located at /usr/bin/gcc)
- GNU Make: 4.3 (located at /usr/bin/make)
- DKMS: 3.0.11 (located at /usr/sbin/dkms)

#### Kernel Development Environment
- Kernel Version: 6.17.0-rc4-rocm-gdb
- Kernel Headers: /lib/modules/6.17.0-rc4-rocm-gdb/build → /home/ben/linux
- Kernel Source: /home/ben/linux (complete kernel source tree)

#### Known Build System Issue
**CRITICAL**: This kernel version (6.17.0-rc4-rocm-gdb) has a known build system bug where the `modfinal` stage fails to generate `.ko` files. The project Makefile includes a workaround that automatically handles this issue.

### Directory Structure
```
/home/ben/rocm-systems/perf-pmu-stub/
├── src/                    # Main source directory
│   ├── Makefile           # Kernel module build configuration
│   ├── pmu_main.c         # Core PMU driver implementation
│   ├── pmu_events.c       # PMU event handling
│   ├── pmu_stub.h         # Main header file
│   ├── aql_c/             # AQL C library components
│   │   ├── aql_*.c        # AQL implementation files
│   │   └── aql_*.h        # AQL header files
│   ├── kfd_ioctl_helper.* # KFD integration
│   ├── pmu_kfd_integration.c
│   ├── kfd_file_manager.*
│   └── kfd_manager_test.c
├── Makefile               # Top-level build and management
├── dkms.conf             # DKMS configuration
├── test/                 # Test scripts
│   ├── load_test.sh      # Module load/unload tests
│   └── perf_test.sh      # Performance integration tests
└── docs/                 # Documentation
```

### Build Instructions

#### Method 1: Direct Build (Recommended for Development)

1. **Navigate to project directory:**
```bash
cd /home/ben/rocm-systems/perf-pmu-stub
```

2. **Clean previous builds:**
```bash
/usr/bin/make clean
```

3. **Build the kernel module:**

**Note:** Due to the known kernel build system bug in 6.17.0-rc4-rocm-gdb, use the force-build target:

```bash
cd /home/ben/rocm-systems/perf-pmu-stub/src
/usr/bin/make force-build
```

**Expected Output:**
```
Force building with error suppression...
[... compilation warnings for prototype declarations ...]
  LD [M]  pmu_stub.o
  MODPOST Module.symvers
  CC [M]  pmu_stub.mod.o
  CC [M]  .module-common.o
  LD [M]  pmu_stub.ko
```

4. **Verify build success:**
```bash
ls -la /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko
/usr/bin/file /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko
```

**Expected Output:**
```
-rw-rw-r-- 1 ben ben 1824664 Sep 17 18:31 /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko
/home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko: ELF 64-bit LSB relocatable, x86-64, version 1 (SYSV), with debug_info, not stripped
```

#### Method 2: DKMS Build (Recommended for System Installation)

1. **Add module to DKMS:**
```bash
sudo /usr/sbin/dkms add -m perf-pmu-stub -v 1.0.0 /home/ben/rocm-systems/perf-pmu-stub
```

2. **Build with DKMS:**
```bash
sudo /usr/sbin/dkms build -m perf-pmu-stub -v 1.0.0
```

3. **Install with DKMS:**
```bash
sudo /usr/sbin/dkms install -m perf-pmu-stub -v 1.0.0
```

4. **Verify DKMS installation:**
```bash
/usr/sbin/dkms status perf-pmu-stub
```

### Module Loading and Testing

#### Load Module Manually
```bash
# Load module with debug enabled
sudo /sbin/insmod /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko debug_enable=true timer_period_ms=100

# Verify module is loaded
/sbin/lsmod | grep pmu_stub
```

#### Check Module Information
```bash
/sbin/modinfo /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko
```

**Expected Output:**
```
filename:       /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko
version:        1.0.0
description:    Skeleton PMU driver for Linux perf subsystem
author:         Your Name
license:        GPL
```

#### Verify PMU Registration
```bash
# Check sysfs interface
ls -la /sys/bus/event_source/devices/pmu_stub/

# Check available events
ls -la /sys/bus/event_source/devices/pmu_stub/events/

# Check format attributes
ls -la /sys/bus/event_source/devices/pmu_stub/format/
```

#### Unload Module
```bash
sudo /sbin/rmmod pmu_stub
```

### Automated Testing

#### Basic Load/Unload Test
```bash
cd /home/ben/rocm-systems/perf-pmu-stub
sudo /bin/bash test/load_test.sh
```

**Expected Output:**
```
[INFO] Starting PMU stub module load test...
[INFO] Loading module pmu_stub...
[INFO] Module loaded successfully
[INFO] Checking sysfs interface...
[INFO] Found sysfs directory: /sys/bus/event_source/devices/pmu_stub
[INFO] All basic tests PASSED
[INFO] === ALL TESTS PASSED ===
```

#### Performance Integration Test
```bash
cd /home/ben/rocm-systems/perf-pmu-stub
sudo /bin/bash test/perf_test.sh
```

### Build Verification Commands

#### 1. Successful Build Check
```bash
test -f /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko && echo "BUILD SUCCESS" || echo "BUILD FAILED"
```

#### 2. Module Integrity Check
```bash
/sbin/modinfo /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko >/dev/null 2>&1 && echo "MODULE VALID" || echo "MODULE INVALID"
```

#### 3. Symbol Verification
```bash
/usr/bin/nm /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko | grep -E "(init_module|cleanup_module)" && echo "SYMBOLS OK" || echo "SYMBOLS MISSING"
```

### Troubleshooting

#### Issue 1: Build Fails with "No rule to make target 'pmu_stub.ko'"
**Cause:** Known bug in kernel 6.17.0-rc4-rocm-gdb build system
**Solution:** This is automatically handled by the project Makefile workaround

#### Issue 2: "Operation not permitted" when loading module
**Cause:** Insufficient privileges or secure boot enabled
**Solutions:**
```bash
# Check if running as root
id
# If secure boot is enabled, disable it or sign the module
sudo mokutil --sb-state
```

#### Issue 3: Module loads but no sysfs interface
**Cause:** PMU registration failed
**Debug:**
```bash
# Check kernel messages
dmesg | tail -20 | grep pmu_stub
# Check for error messages during registration
journalctl -k | grep pmu_stub
```

#### Issue 4: Kernel headers not found
**Cause:** Kernel development environment not properly set up
**Solution:**
```bash
# Verify kernel headers location
ls -la /lib/modules/$(uname -r)/build
# Check if kernel source is available
ls -la /home/ben/linux/Makefile
```

#### Issue 5: "Unknown symbol" errors during module load
**Cause:** Module compiled against wrong kernel version
**Solution:**
```bash
# Clean and rebuild
/usr/bin/make clean
/usr/bin/make all
# Verify vermagic matches current kernel
strings /home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko | grep vermagic
uname -r
```

### Clean and Rebuild Procedures

#### Complete Clean Build
```bash
cd /home/ben/rocm-systems/perf-pmu-stub
/usr/bin/make clean
cd src
/usr/bin/make force-build
```

#### DKMS Clean Rebuild
```bash
sudo /usr/sbin/dkms remove -m perf-pmu-stub -v 1.0.0 --all
sudo rm -rf /usr/src/perf-pmu-stub-1.0.0
cd /home/ben/rocm-systems/perf-pmu-stub
/usr/bin/make dkms-install
```

#### Deep Clean (Remove All Generated Files)
```bash
cd /home/ben/rocm-systems/perf-pmu-stub/src
/usr/bin/make clean
rm -f *.o *.ko *.mod *.mod.c *.order *.symvers .*.cmd
rm -f aql_c/*.o aql_c/.*.cmd
```

### Quick Reference Commands

```bash
# Build module
cd /home/ben/rocm-systems/perf-pmu-stub/src && /usr/bin/make force-build

# Load module
sudo /sbin/insmod src/pmu_stub.ko

# Check module status
/sbin/lsmod | grep pmu_stub

# Check sysfs interface
ls /sys/bus/event_source/devices/pmu_stub/

# Unload module
sudo /sbin/rmmod pmu_stub

# Run tests
sudo /bin/bash test/load_test.sh

# Check kernel messages
dmesg | grep pmu_stub

# Clean build
cd /home/ben/rocm-systems/perf-pmu-stub && /usr/bin/make clean && cd src && /usr/bin/make force-build
```

### Success Indicators

#### Build Success
- File exists: `/home/ben/rocm-systems/perf-pmu-stub/src/pmu_stub.ko`
- File size: ~1.8MB (1824664 bytes)
- File type: ELF 64-bit LSB relocatable
- Contains vermagic string matching current kernel

#### Load Success
- Module appears in `lsmod` output
- Sysfs directory created: `/sys/bus/event_source/devices/pmu_stub/`
- No error messages in `dmesg`
- Module parameters accessible in `/sys/module/pmu_stub/parameters/`

#### Test Success
- load_test.sh exits with code 0
- All test phases pass without errors
- Module can be loaded and unloaded multiple times
- sysfs interface properly created and cleaned up

### Development Notes

- The module includes comprehensive debug output when loaded with `debug_enable=true`
- Timer period can be adjusted with `timer_period_ms` parameter
- AQL C library components are statically linked into the module
- KFD integration provides interface to AMD GPU kernel driver
- Module supports both manual and DKMS-based installation

### Performance Testing

To test PMU functionality with perf tools:
```bash
# Load module
sudo /sbin/insmod src/pmu_stub.ko

# List available PMU events
perf list | grep pmu_stub

# Test event counting (if events are registered)
sudo perf stat -e pmu_stub/example_event/ sleep 1

# Unload module
sudo /sbin/rmmod pmu_stub
```

---
**Note:** This documentation is specific to the current machine configuration (kernel-vm) running kernel 6.17.0-rc4-rocm-gdb. Some paths and procedures may need adjustment for different systems.