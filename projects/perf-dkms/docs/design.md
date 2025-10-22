# DKMS Perf Interface Skeleton Module - Design Document

## Overview

This document describes the design and implementation of a DKMS-compatible kernel module that implements a skeleton of the Linux perf interface. The module provides a minimal but complete PMU (Performance Monitoring Unit) driver that demonstrates proper integration with the Linux perf subsystem.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [File Structure](#file-structure)
3. [Core Components](#core-components)
4. [Implementation Details](#implementation-details)
5. [Usage Instructions](#usage-instructions)
6. [Testing Strategy](#testing-strategy)
7. [Extension Points](#extension-points)
8. [Troubleshooting](#troubleshooting)

## Architecture Overview

### Design Goals

- **Educational**: Provide a clear, well-documented example of PMU driver implementation
- **Minimal but Complete**: Include all required components for a working PMU driver
- **DKMS Compatible**: Easy installation across different kernel versions
- **Foundation**: Serve as a base for real GPU performance monitoring implementations

### Key Components

```
┌─────────────────────────────────────────────────────────────┐
│                    User Space                               │
├─────────────────────────────────────────────────────────────┤
│  perf tools  │  sysfs interface  │  /proc/modules          │
├─────────────────────────────────────────────────────────────┤
│                    Kernel Space                             │
├─────────────────────────────────────────────────────────────┤
│              Linux Perf Subsystem                          │
├─────────────────────────────────────────────────────────────┤
│                   PMU Stub Driver                          │
│  ┌─────────────┐ ┌──────────────┐ ┌─────────────────────┐  │
│  │   PMU Core  │ │ Event Handler│ │  Timer & Counters   │  │
│  └─────────────┘ └──────────────┘ └─────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## File Structure

```
perf-pmu-stub/
├── dkms.conf                 # DKMS configuration
├── Makefile                  # Main build file
├── src/
│   ├── Makefile             # Kernel module build file
│   ├── amdgpu_pmu.h           # Main header file
│   ├── pmu_main.c           # Core PMU implementation
│   └── pmu_events.c         # Event handling utilities
├── test/
│   ├── load_test.sh         # Basic module load/unload test
│   └── perf_test.sh         # Perf integration test
└── docs/
    └── design.md            # This document
```

## Core Components

### 1. PMU Structure (`struct amdgpu_pmu`)

The main data structure that represents our PMU driver:

```c
struct amdgpu_pmu {
    struct pmu pmu;                          /* Base PMU structure */
    struct device *dev;                      /* Device for sysfs */

    /* Event management */
    spinlock_t lock;                         /* Protects event_list */
    struct amdgpu_pmu_event events[AMDGPU_PMU_MAX_EVENTS];
    DECLARE_BITMAP(used_mask, AMDGPU_PMU_MAX_EVENTS);
    int num_events;

    /* Timer for simulating counter updates */
    struct hrtimer timer;
    ktime_t timer_period;

    /* Statistics and counters */
    atomic64_t total_events;
    atomic64_t total_samples;
    atomic64_t counter_cycles;
    atomic64_t counter_instructions;
    atomic64_t counter_cache_misses;
    atomic64_t counter_bandwidth;
};
```

### 2. Event Types

Four simulated performance events:

- **cycles**: Simulated CPU cycles
- **instructions**: Simulated instructions retired
- **cache-misses**: Simulated cache misses
- **bandwidth**: Simulated memory bandwidth

### 3. PMU Callbacks

Required callbacks for perf integration:

- `event_init()`: Validate and initialize events
- `add()`: Schedule event on PMU
- `del()`: Remove event from PMU
- `start()`: Start event counting
- `stop()`: Stop event counting
- `read()`: Read current counter values

## Implementation Details

### Module Initialization

1. **Memory Allocation**: Allocate `struct amdgpu_pmu` instance
2. **Lock Initialization**: Set up spinlocks for thread safety
3. **Timer Setup**: Initialize high-resolution timer for counter updates
4. **PMU Registration**: Register with Linux perf subsystem via `perf_pmu_register()`
5. **Sysfs Creation**: Create sysfs entries for event discovery

### Event Management

#### Event Creation Flow

```
perf_event_open() syscall
        ↓
amdgpu_pmu_event_init()
        ↓
Event validation
        ↓
amdgpu_pmu_add()
        ↓
Assign counter slot
        ↓
amdgpu_pmu_start()
        ↓
Begin counting
```

#### Counter Simulation

- **High-Resolution Timer**: Updates counters every 50-100ms
- **Realistic Increments**: Different increment rates per event type
- **Thread-Safe Updates**: Atomic operations for counter updates
- **Overflow Handling**: Proper 64-bit counter management

### DKMS Integration

#### Configuration (`dkms.conf`)

```ini
PACKAGE_NAME="perf-pmu-stub"
PACKAGE_VERSION="1.0.0"
BUILT_MODULE_NAME[0]="pmu_stub"
BUILT_MODULE_LOCATION[0]="src"
DEST_MODULE_LOCATION[0]="/kernel/drivers/perf"
MAKE[0]="make -C ${kernel_source_dir} M=${dkms_tree}/${PACKAGE_NAME}/${PACKAGE_VERSION}/build/src modules"
AUTOINSTALL="yes"
```

#### Build Process

```bash
# Add module to DKMS
sudo dkms add -m perf-pmu-stub -v 1.0.0

# Build for current kernel
sudo dkms build -m perf-pmu-stub -v 1.0.0

# Install module
sudo dkms install -m perf-pmu-stub -v 1.0.0
```

### Sysfs Interface

The module creates the following sysfs structure:

```
/sys/bus/event_source/devices/amdgpu_pmu/
├── events/
│   ├── cycles
│   ├── instructions
│   ├── cache-misses
│   └── bandwidth
├── format/
│   └── config
└── type
```

## Usage Instructions

### Building and Installing

1. **Clone/Download** the module source
2. **Build** using make:
   ```bash
   cd perf-pmu-stub
   make
   ```
3. **Install via DKMS**:
   ```bash
   make dkms-install
   ```

### Loading the Module

```bash
# Manual load
sudo modprobe amdgpu_pmu

# Check if loaded
lsmod | grep amdgpu_pmu

# Check kernel messages
dmesg | tail -10
```

### Using with Perf Tools

#### List Available Events

```bash
perf list | grep amdgpu_pmu
```

Expected output:
```
pmu_stub/bandwidth/                        [Kernel PMU event]
pmu_stub/cache-misses/                     [Kernel PMU event]
pmu_stub/cycles/                           [Kernel PMU event]
pmu_stub/instructions/                     [Kernel PMU event]
```

#### Count Events

```bash
# Single event
perf stat -e amdgpu_pmu/cycles/ sleep 5

# Multiple events
perf stat -e amdgpu_pmu/cycles/,pmu_stub/instructions/ sleep 5

# System-wide monitoring
perf stat -a -e amdgpu_pmu/cache-misses/ sleep 10
```

Example output:
```
Performance counter stats for 'sleep 5':

         5,000      pmu_stub/cycles/

       5.002381063 seconds time elapsed
```

### Module Parameters

- `debug_enable`: Enable debug output (default: false)
- `timer_period_ms`: Timer period in milliseconds (default: 100)

```bash
# Load with debug enabled
sudo modprobe amdgpu_pmu debug_enable=true timer_period_ms=50
```

## Testing Strategy

### Automated Tests

#### 1. Basic Load Test (`test/load_test.sh`)

- Module load/unload functionality
- Sysfs interface validation
- Parameter checking
- Error handling

#### 2. Perf Integration Test (`test/perf_test.sh`)

- Event discovery via `perf list`
- Event counting via `perf stat`
- Multiple event handling
- Invalid event rejection

### Manual Testing

#### Verify Module Loading
```bash
sudo ./test/load_test.sh
```

#### Verify Perf Integration
```bash
sudo ./test/perf_test.sh
```

#### Debug Information
```bash
# Module information
modinfo amdgpu_pmu

# Module parameters
cat /sys/module/amdgpu_pmu/parameters/*

# Event configuration
cat /sys/bus/event_source/devices/amdgpu_pmu/events/*
```

## Extension Points

This skeleton can be extended for real hardware monitoring:

### 1. Hardware Integration

Replace simulated counters with real hardware register access:

```c
// Replace timer-based simulation
static u64 read_hardware_counter(int counter_id)
{
    // Read from actual hardware registers
    return readq(hw_counter_base + counter_id * 8);
}
```

### 2. GPU-Specific Events

Add GPU-specific performance events:

```c
enum gpu_events {
    GPU_SHADER_BUSY = 0x10,
    GPU_MEMORY_BANDWIDTH = 0x11,
    GPU_TEXTURE_CACHE_MISS = 0x12,
    GPU_COMPUTE_UTILIZATION = 0x13,
};
```

### 3. Multi-Instance Support

Support multiple GPU devices:

```c
struct gpu_pmu_instance {
    struct amdgpu_pmu base;
    int gpu_id;
    void __iomem *registers;
    struct pci_dev *pdev;
};
```

### 4. Advanced Features

- **Sampling Support**: Implement interrupt-based sampling
- **Context Switching**: Track per-process GPU usage
- **Power Monitoring**: Integrate with GPU power management
- **Trace Events**: Add ftrace integration

## Troubleshooting

### Common Issues

#### Module Load Fails

**Symptoms**: `insmod` returns error, no sysfs entries created

**Debugging**:
```bash
# Check kernel messages
dmesg | tail -20

# Verify kernel headers
ls /lib/modules/$(uname -r)/build

# Check module dependencies
modprobe --dry-run pmu_stub
```

**Solutions**:
- Install kernel headers: `sudo apt install linux-headers-$(uname -r)`
- Check DKMS build logs: `/var/lib/dkms/perf-pmu-stub/1.0.0/build/make.log`

#### Events Not Visible in Perf

**Symptoms**: `perf list` doesn't show PMU events

**Debugging**:
```bash
# Check PMU registration
ls /sys/bus/event_source/devices/

# Verify events directory
ls /sys/bus/event_source/devices/amdgpu_pmu/events/

# Check perf capabilities
perf --version
```

**Solutions**:
- Ensure module is loaded: `lsmod | grep amdgpu_pmu`
- Check sysfs permissions
- Verify perf tools version compatibility

#### Counters Not Incrementing

**Symptoms**: `perf stat` shows zero counts

**Debugging**:
```bash
# Check timer status
cat /proc/interrupts | grep timer

# Enable debug output
echo 'module amdgpu_pmu +p' > /sys/kernel/debug/dynamic_debug/control

# Monitor kernel messages
dmesg -w
```

**Solutions**:
- Verify timer initialization in module load
- Check for deadlocks in timer handler
- Increase timer frequency (decrease `timer_period_ms`)

### Debug Features

#### Dynamic Debug

Enable runtime debugging:
```bash
# Enable all debug messages
echo 'module amdgpu_pmu +p' > /sys/kernel/debug/dynamic_debug/control

# Enable specific function
echo 'func amdgpu_pmu_timer_handler +p' > /sys/kernel/debug/dynamic_debug/control
```

#### Module Statistics

Check module statistics:
```bash
# Via sysfs (if implemented)
cat /sys/module/amdgpu_pmu/stats/*

# Via kernel messages
dmesg | grep "pmu_stub.*statistics"
```

## Security Considerations

### Kernel Module Security

- **Root Privileges**: Module loading requires root access
- **Module Signing**: DKMS automatically signs modules for Secure Boot
- **Input Validation**: All user inputs are validated
- **Memory Safety**: Proper bounds checking and memory management

### User Permissions

- **perf Access**: Standard perf permissions apply
- **Capability Requirements**: CAP_PERFMON for system-wide monitoring
- **Process Isolation**: Events isolated per process by default

## Performance Impact

### Overhead Analysis

- **Timer Frequency**: 10-100Hz (configurable)
- **Memory Usage**: ~4KB per module instance
- **CPU Impact**: <0.1% on modern systems
- **Lock Contention**: Minimal due to per-CPU design

### Optimization Opportunities

- **Per-CPU Counters**: Reduce lock contention
- **Hardware Timestamps**: Use TSC for better accuracy
- **Batch Updates**: Group counter updates
- **Lazy Allocation**: Allocate resources on demand

This design document provides a comprehensive overview of the PMU stub implementation, serving as both documentation and a guide for future enhancements targeting real GPU performance monitoring hardware.