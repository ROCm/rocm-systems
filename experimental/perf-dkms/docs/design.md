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
perf-dkms/
├── CMakeLists.txt            # Root build configuration
├── dkms.conf                 # DKMS configuration
├── src/
│   ├── CMakeLists.txt       # Kbuild wrapper
│   ├── amdgpu_pmu.h         # Main header file
│   ├── pmu_main.c           # Core PMU implementation
│   ├── pmu_events.c         # Event handling utilities
│   ├── pmu_dimension.h      # Dimension encoding/validation
│   ├── aql_perf.c/h         # AQL session & GPU discovery
│   ├── aql_packet_ops.c     # PM4 packet operations
│   ├── aql_pmu_integration.c # PMU-AQL bridge
│   ├── aql_error_recovery.c # Error handling & recovery
│   ├── aql_queue_manager.c/h # AQL queue lifecycle
│   ├── kfd_ioctl_bridge.c/h # KFD ioctl bridge (vm_mmap)
│   └── aql_c/               # Pure C library (kernel/userspace)
│       ├── aql_queue.c/h    # Ring buffer & IB pool
│       ├── pm4_packets.c    # Low-level PM4 primitives
│       ├── packet_generation.c # PM4 sequence generation
│       ├── counter_registry.c # Counter definitions
│       ├── arch_creator.c   # Architecture factory
│       ├── gfx12_creator.c  # GFX12 implementation
│       ├── gfx9_creator.c   # GFX9 implementation
│       ├── tests/           # Validation tests
│       └── tools/           # Packet tools
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
   cd perf-dkms
   cmake -B build && cmake --build build
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
## Dimension-Aware Performance Monitoring

### Overview

The dimension-aware performance monitoring feature enables users to monitor specific hardware dimensions (XCC, SE, SA, WGP, CU) instead of aggregated GPU-wide counters. This provides fine-grained performance visibility into different parts of the GPU hardware hierarchy.

### Hardware Dimension Hierarchy

AMD GPUs are organized in a hierarchical structure:

```
GPU
 └── XCC (eXtended Compute Core) [0..N]
      └── SE (Shader Engine) [0..3 typically]
           └── SA (Shader Array) [0..1 typically]
                └── WGP (Work Group Processor) [0..3 typically]
                     └── CU (Compute Unit) [0..63 typically]
```

For GFX12 architecture:
- **XCC**: 1 instance (index 0)
- **SE**: 4 instances (index 0-3)  
- **SA**: 2 instances per SE (index 0-1)
- **WGP**: 4 instances per SA (index 0-3)
- **CU**: 64 total (distributed across WGPs)

### config1 Bit Field Layout

Dimension specifications are encoded in the `config1` field of perf_event_attr:

```
Bits   Field         Description
-----  ------------  -----------------------------------------
0-7    xcc           XCC index (0-255)
8-15   se            Shader Engine index (0-255)
16-23  sa            Shader Array index (0-255)
24-31  wgp           Work Group Processor index (0-255)
32-39  cu            Compute Unit index (0-255)
40     aggregate     Aggregate across dimensions (future)
41-63  reserved      Reserved for future use
```

### Default Behavior

When dimensions are not specified (config1=0), the driver operates in **broadcast mode**: it reads all hardware instances and aggregates (sums) the results.

When specific dimensions are provided, unspecified sub-dimensions default to 0. For example:
- `se=2` is equivalent to `se=2,sa=0,wgp=0,cu=0` - targets SE 2, SA 0, WGP 0
- `se=2,sa=1` is equivalent to `se=2,sa=1,wgp=0,cu=0` - targets SE 2, SA 1, WGP 0

This allows hierarchical targeting of specific hardware units without requiring users to specify all levels of the hierarchy.

### Named Parameter Syntax

Users can specify dimensions using named parameters that perf translates to config1:

```bash
# Monitor specific shader engine
perf stat -e amdgpu_pmu/sq_waves,se=2/ -a sleep 1

# Monitor specific shader engine and shader array
perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1/ -a sleep 1

# Monitor full hierarchy path
perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 1

# Raw config1 encoding (SE=2 is bit 8-15, so 0x0200)
perf stat -e amdgpu_pmu/sq_waves,config1=0x0200/ -a sleep 1
```

### Counter Dimension Support

Not all counters support all dimensions. Counter support is defined in the counter registry:

| Counter Type | Supported Dimensions | Example Counters |
|-------------|---------------------|-----------------|
| Global      | None (DIM_NONE)     | grbm_count, grbm_busy |
| Per-SE      | SE, SA, WGP         | sq_waves, sq_insts_valu |
| Per-SA      | SE, SA              | gl2c_hit, gl2c_miss |
| Per-WGP     | SE, SA, WGP         | ta_busy, ta_total_wavefronts |

The driver validates that requested dimensions are supported by the counter before allowing measurement.

### Implementation Architecture

#### 1. Format Attribute Exposure

The driver exposes format attributes via sysfs that perf uses to parse parameters:

```
/sys/bus/event_source/devices/amdgpu_pmu/format/
├── config         (config:0-63)     - Counter ID
├── config1        (config1:0-63)    - Raw dimension encoding
├── xcc            (config1:0-7)     - XCC index
├── se             (config1:8-15)    - SE index
├── sa             (config1:16-23)   - SA index
├── wgp            (config1:24-31)   - WGP index
├── cu             (config1:32-39)   - CU index
└── aggregate      (config1:40)      - Aggregate flag
```

When a user specifies `se=2`, perf reads the `se` format file, sees it maps to `config1:8-15`, and encodes the value 2 into those bits of event->attr.config1.

#### 2. Dimension Extraction and Validation

The `pmu_dimension.h` header provides helper functions:

```c
/* Extract dimensions from config1 field */
static inline void pmu_extract_dimensions(u64 config1,
                                          struct pmu_dimension_coords *dims)
{
    dims->xcc = (config1 >> PMU_DIM_XCC_SHIFT) & PMU_DIM_XCC_MASK;
    dims->se = (config1 >> PMU_DIM_SE_SHIFT) & PMU_DIM_SE_MASK;
    dims->sa = (config1 >> PMU_DIM_SA_SHIFT) & PMU_DIM_SA_MASK;
    dims->wgp = (config1 >> PMU_DIM_WGP_SHIFT) & PMU_DIM_WGP_MASK;
    dims->cu = (config1 >> PMU_DIM_CU_SHIFT) & PMU_DIM_CU_MASK;
    dims->aggregate = (config1 >> PMU_DIM_AGGREGATE_SHIFT) & 1;

    /* Mark as valid if any dimension or flag is non-zero */
    dims->valid = (config1 != 0);
}

/* Validate dimensions against hardware limits */
static inline bool pmu_validate_dimensions(
    const struct pmu_dimension_coords *dims,
    const struct pmu_dimension_limits *limits)
{
    if (!dims->valid)
        return true; /* No dimensions specified is valid */
    
    return dims->xcc <= limits->max_xcc &&
           dims->se <= limits->max_se &&
           dims->sa <= limits->max_sa &&
           dims->wgp <= limits->max_wgp &&
           dims->cu <= limits->max_cu;
}
```

During event initialization (`amdgpu_pmu_event_init`):
1. Extract dimensions from event->attr.config1
2. Validate against hardware limits from GPU architecture
3. Validate counter supports requested dimensions
4. Pass dimension info to AQL layer for measurement setup

#### 3. Counter Allocation

The `aql_counter_try_allocate_dimension()` function allocates hardware counters for dimension-specific monitoring:

```c
counter_reg_info_t* aql_counter_try_allocate_dimension(
    block_info_t *block,
    uint32_t event_id,
    struct perf_event *perf_event,
    const struct pmu_dimension_coords *dims,
    arch_t *arch)
{
    /* Calculate flat index from hierarchical coordinates */
    uint32_t flat_index = encode_dimension_index(
        dims->se, dims->sa, dims->wgp,
        arch->num_sa, arch->num_wgp_per_sa);
    
    /* Allocate counter */
    counter_reg_info_t *reg = aql_counter_try_allocate(
        block, event_id, perf_event);
    
    /* Store dimension index for packet generation */
    reg->allocation.instance_id = flat_index;
    
    return reg;
}
```

The `encode_dimension_index()` helper converts hierarchical SE/SA/WGP coordinates to a flat array index:

```
Formula: index = (se * num_sa * wgp_per_sa) + (sa * wgp_per_sa) + wgp

Example (GFX12: 4 SE × 2 SA × 4 WGP):
  SE=2, SA=1, WGP=3:
  index = (2 × 2 × 4) + (1 × 4) + 3 = 16 + 4 + 3 = 23
```

#### 4. Hardware Configuration (GRBM_GFX_INDEX)

The hardware uses the GRBM_GFX_INDEX register to target specific SE/SA/WGP instances. The packet generation code sets this register before programming counters:

```c
/* In generate_start_packet() */
if (reg_info->allocation.instance_id != 0) {
    /* Decode flat index back to coordinates */
    dimension_coords_t coords = decode_dimension_index(
        reg_info->allocation.instance_id,
        arch->num_sa,
        arch->num_wgp_per_sa);
    
    /* Set GRBM_GFX_INDEX to target specific dimension */
    pm4_set_grbm_index(buffer, 
                       arch->control_regs.grbm_gfx_index,
                       coords.wgp << 2,  /* instance_index */
                       coords.sa,
                       coords.se);
    
    /* Now configure counter - hardware will target specified dimension */
    generate_counter_config(buffer, arch, counter);
} else {
    /* Broadcast mode - target all instances */
    generate_grbm_broadcast(buffer, arch);
    generate_counter_config(buffer, arch, counter);
}
```

The PM4 packet functions (`pm4_set_grbm_index`) write to the GRBM_GFX_INDEX register:

```c
typedef struct {
    uint32_t instance_index : 8;  /* WGP instance (shifted by 2) */
    uint32_t sa_index : 8;         /* Shader Array index */
    uint32_t se_index : 8;         /* Shader Engine index */
    uint32_t reserved : 8;
} pm4_grbm_gfx_index_t;
```

#### 5. Counter Reading

When reading dimension-specific counters, the driver only reads from the targeted dimension instead of iterating through all instances:

```c
/* In generate_read_packet() */
if (reg_info->allocation.instance_id != 0) {
    /* Dimension-specific: read only from targeted instance */
    dimension_coords_t coords = decode_dimension_index(...);
    
    pm4_set_grbm_index(buffer, ..., coords.wgp, coords.sa, coords.se);
    
    /* Copy counter value to memory (single instance) */
    pm4_append_copy_data(buffer, gpu_addr, 
                        reg_info->register_addr_lo, ...);
} else {
    /* Broadcast mode: iterate through all SE/SA/WGP instances */
    for (se = 0; se < num_se; se++) {
        for (sa = 0; sa < num_sa; sa++) {
            for (wgp = 0; wgp < num_wgp; wgp++) {
                pm4_set_grbm_index(...);
                pm4_append_copy_data(...);
            }
        }
    }
}
```

### Usage Examples

#### Basic Dimension Targeting

```bash
# Monitor shader engine 0
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1

# Monitor shader engine 1, shader array 0
perf stat -e amdgpu_pmu/sq_waves,se=1,sa=0/ -a sleep 1

# Monitor full path: SE=2, SA=1, WGP=3
perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 1
```

#### Comparing Dimensions

```bash
# Compare activity across shader engines
perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -e amdgpu_pmu/sq_waves,se=2/ \
          -e amdgpu_pmu/sq_waves,se=3/ \
          -a sleep 5
```

#### Raw config1 Encoding

```bash
# SE=2 (bits 8-15): 0x0200
perf stat -e amdgpu_pmu/sq_waves,config1=0x0200/ -a sleep 1

# SE=2, SA=1 (bits 8-23): 0x00010200  
perf stat -e amdgpu_pmu/sq_waves,config1=0x00010200/ -a sleep 1

# SE=3, SA=1, WGP=2 (bits 8-31): 0x02010300
perf stat -e amdgpu_pmu/sq_waves,config1=0x02010300/ -a sleep 1
```

### Error Handling

The driver provides clear error messages for common mistakes:

```bash
# Invalid dimension (SE=99 exceeds maximum)
$ perf stat -e amdgpu_pmu/sq_waves,se=99/ -a sleep 1
Error: dimension out of range: se=99 (max: 3)

# Unsupported dimension for counter
$ perf stat -e amdgpu_pmu/grbm_count,se=0/ -a sleep 1
Error: counter 'grbm_count' does not support requested dimensions

# Dimension validation happens during event_init, before allocation
```

Dimension limits are determined from GPU architecture at module load time:
```c
/* In amdgpu_pmu_init() */
global_dim_limits.max_xcc = arch->num_xcc - 1;
global_dim_limits.max_se = arch->num_se - 1;
global_dim_limits.max_sa = arch->num_sa - 1;
global_dim_limits.max_wgp = arch->num_wgp_per_sa - 1;
global_dim_limits.max_cu = arch->num_cu - 1;
```

### Current Limitations and Future Work

#### Implemented
- ✅ Single dimension instance monitoring (one SE/SA/WGP at a time)
- ✅ Named parameter syntax via perf format attributes
- ✅ Raw config1 encoding support
- ✅ Dimension validation against hardware limits
- ✅ Counter compatibility checking
- ✅ GRBM_GFX_INDEX targeting for hardware configuration

#### Future Enhancements

**Aggregate Mode** (config1 bit 40):
```bash
# Monitor SE=0 but aggregate across all SA/WGP within that SE
perf stat -e amdgpu_pmu/sq_waves,se=0,aggregate=1/ -a sleep 1
```
Implementation: Allocate multiple counter instances, configure each for different SA/WGP within the SE, sum results.

**Per-CU Monitoring**:
Currently focused on SE/SA/WGP. CU-level monitoring would require additional hardware support and more fine-grained GRBM_GFX_INDEX configuration.

**Dynamic Dimension Discovery**:
Export per-GPU dimension limits via sysfs for runtime discovery:
```
/sys/bus/event_source/devices/amdgpu_pmu/caps/
├── max_xcc
├── max_se
├── max_sa
├── max_wgp
└── max_cu
```

### Performance Considerations

**Overhead**:
- Dimension-specific monitoring has minimal overhead vs. broadcast mode
- Single targeted register write vs. broadcast affects all instances
- Counter read is faster (1 instance vs. SE×SA×WGP instances)

**Scalability**:
- Each dimension-specific event allocates one hardware counter
- Maximum concurrent events limited by available hardware counters per block
- Typical limits: 4-8 counters per block type (SQ, TA, GL2C, etc.)

**Use Cases**:
- **Load Balancing Analysis**: Compare activity across SEs to detect imbalance
- **Hotspot Detection**: Identify which SE/SA is most active  
- **Power Analysis**: Monitor specific regions during power management
- **Debugging**: Isolate issues to specific hardware units

### Testing

See test files:
- `src/aql_c/tests/test_dimension_helpers.c` - Unit tests for dimension encoding/decoding
- `test/load_test.sh` - Module load/unload verification
- `test/perf_test.sh` - Perf integration tests

### References

- AMD GFX12 Architecture Documentation
- Linux Perf Subsystem Documentation
- GRBM_GFX_INDEX Register Specification
- PM4 Packet Format Reference
