# AMD GPU Performance Monitoring - Perf PMU Stub

> **⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE**
>
> This is a research and development project. The module is not ready for general use and may cause system instability.
>
> **Requirements:**
> - GFX12/RDNA4 GPUs only (RX 9070 series)
> - **Patched Linux kernel required** (see Prerequisites section)
>
> Use at your own risk in development/testing environments only.

A Linux kernel module implementing AMD GPU performance counter integration with the perf subsystem. This module provides AQL (Asynchronous Queuing Language) packet generation for GPU performance monitoring on GFX12/RDNA4 architectures.

## Overview

This kernel module bridges AMD GPU hardware performance counters with the Linux perf interface, enabling standard perf tools to monitor GPU performance events. It includes:

- **AQL C Library**: Pure C implementation of PM4 packet generation for GPU counter programming
- **Perf PMU Driver**: Complete Linux perf subsystem integration
- **Hardware Support**: GFX12/RDNA4 GPU architecture support
- **Counter Registry**: Architecture-specific performance event definitions

## Features

- ✅ Linux perf subsystem integration
- ✅ AQL/PM4 packet generation for GPU counter control
- ✅ GFX12/RDNA4 hardware support
- ✅ Multi-GPU architecture support
- ✅ Hardware block counter definitions (SQ, CPC, GL2C, etc.)
- ✅ **Dimension-Specific Monitoring** - Monitor specific SE/SA/WGP hardware units
- ✅ Standard perf tool compatibility
- ✅ DKMS-compatible installation

## Quick Start

### Prerequisites

```bash
# Install build tools and kernel headers
sudo apt update
sudo apt install build-essential linux-headers-$(uname -r)

# Verify perf tools are available
perf --version
```

### Building the Module

```bash
# Clone and enter directory
cd perf-dkms

# Build kernel module and tools
cmake -B build && cmake --build build

# The build produces:
# - build/src/amdgpu_pmu.ko (kernel module)
# - build/src/aql_c/tools/packet_gen_tool (PM4 packet generator)
# - build/src/aql_c/tools/pm4_decoder (PM4 packet decoder)
# - build/src/aql_c/tests/* (validation tests)
```

### Loading the Module

```bash
# Load the module
sudo insmod build/src/amdgpu_pmu.ko

# Or if using .o file (development kernels):
sudo insmod build/src/amdgpu_pmu.o

# Verify module is loaded
lsmod | grep amdgpu_pmu

# Check kernel messages
dmesg | tail -20
```

### Verification

```bash
# Check PMU device registration
ls -la /sys/bus/event_source/devices/ | grep pmu

# List available GPU performance events
perf list | grep amdgpu_pmu

# Or directly check sysfs
ls /sys/bus/event_source/devices/amdgpu_pmu/events/
```

## Testing with Perf

### Basic GPU Counter Test

Test GPU shader wave counting with detailed verbose output:

```bash
# Monitor SQ_WAVES counter with 800ms intervals
perf stat -I 800 -vv -C 0 -a --no-inherit -e amdgpu_pmu/sq_waves/ sleep 4
```

**Command breakdown:**
- `-I 800`: Print counter deltas every 800ms
- `-vv`: Very verbose output (shows counter reads and updates)
- `-C 0`: Monitor CPU 0 (or use `-a` for all CPUs)
- `-a`: System-wide monitoring
- `--no-inherit`: Don't inherit counters to child processes
- `-e amdgpu_pmu/sq_waves/`: Monitor the SQ_WAVES GPU counter

### Available Hardware Counters

The module exposes GPU hardware performance counters through the perf interface:

```bash
# Shader Processor (SQ) counters
perf stat -e amdgpu_pmu/sq_waves/ <command>           # Active waves
perf stat -e amdgpu_pmu/sq_busy_cycles/ <command>     # SQ busy cycles
perf stat -e amdgpu_pmu/sq_instructions/ <command>    # Instructions executed

# L2 Cache (GL2C) counters
perf stat -e amdgpu_pmu/gl2c_hit/ <command>           # L2 cache hits
perf stat -e amdgpu_pmu/gl2c_miss/ <command>          # L2 cache misses

# Command Processor (CPC) counters
perf stat -e amdgpu_pmu/cpc_busy/ <command>           # CP busy cycles
perf stat -e amdgpu_pmu/cpc_stall/ <command>          # CP stall cycles
```

### Multi-Event Monitoring

```bash
# Monitor multiple GPU counters simultaneously
perf stat -e amdgpu_pmu/sq_waves/,pmu_stub/gl2c_hit/,pmu_stub/gl2c_miss/ sleep 5

# System-wide GPU monitoring
sudo perf stat -a -e amdgpu_pmu/sq_waves/ sleep 10

# Per-process GPU monitoring (when supported)
perf stat -e amdgpu_pmu/sq_waves/ ./gpu_workload
```

### Advanced Perf Usage

```bash
# Record GPU events with sampling
sudo perf record -e amdgpu_pmu/sq_waves/ -a sleep 5
sudo perf report

# Monitor with custom intervals
perf stat -I 1000 -e amdgpu_pmu/sq_waves/ sleep 10

# Export to JSON
perf stat -j -e amdgpu_pmu/sq_waves/ sleep 2
```

## Dimension-Specific Monitoring

The amdgpu_pmu driver supports monitoring specific GPU hardware dimensions (Shader Engines, Shader Arrays, Work Group Processors) for fine-grained performance analysis.

### Quick Examples

```bash
# Monitor specific Shader Engine (SE)
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1

# Monitor SE and Shader Array (SA)
perf stat -e amdgpu_pmu/sq_waves,se=1,sa=0/ -a sleep 1

# Monitor full dimension hierarchy
perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 1

# Compare activity across all Shader Engines
perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -e amdgpu_pmu/sq_waves,se=2/ \
          -e amdgpu_pmu/sq_waves,se=3/ \
          -a ./my_gpu_app
```

### Hardware Dimensions

AMD GPUs are organized hierarchically:

- **SE** (Shader Engine): Top-level parallel units, typically 4 per GPU
- **SA** (Shader Array): Subdivisions within SE, typically 2 per SE
- **WGP** (Work Group Processor): Compute units, typically 4 per SA
- **CU** (Compute Unit): Individual SIMD units

For GFX12: 4 SE × 2 SA × 4 WGP = 32 unique addressable locations

### Available Parameters

- `se=N` - Shader Engine index (0-3 typical)
- `sa=N` - Shader Array index (0-1 typical)
- `wgp=N` - Work Group Processor index (0-3 typical)
- `cu=N` - Compute Unit index (0-63 typical)
- `xcc=N` - XCC index (usually 0)

### Supported Counters

Not all counters support dimension targeting:

- **SQ Counters** (SE/SA/WGP): `sq_waves`, `sq_insts_valu`, `sq_busy_cycles`
- **TA Counters** (SE/SA/WGP): `ta_busy`, `ta_total_wavefronts`
- **GL2C Counters** (SE/SA): `gl2c_hit`, `gl2c_miss`
- **Global Counters** (no dimensions): `grbm_count`, `grbm_busy`

### Use Cases

**Load Balancing Analysis**: Check if work is evenly distributed across SEs
```bash
perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -e amdgpu_pmu/sq_waves,se=2/ \
          -e amdgpu_pmu/sq_waves,se=3/ \
          -a ./workload
```

**Hotspot Detection**: Identify which hardware unit is most active
```bash
perf record -e amdgpu_pmu/sq_busy_cycles,se=0/ \
            -e amdgpu_pmu/sq_busy_cycles,se=1/ \
            -a sleep 10
```

**Memory Analysis**: Monitor cache behavior per Shader Array
```bash
perf stat -e amdgpu_pmu/gl2c_hit,se=0,sa=0/ \
          -e amdgpu_pmu/gl2c_miss,se=0,sa=0/ \
          -a ./memory_test
```

### Documentation

For detailed information, see:
- **User Guide**: `docs/user_guide_dimensions.md` - Complete usage guide with examples
- **Design Doc**: `docs/design.md` - Technical implementation details
- **Tests**: `test/load_test.sh`, `test/perf_test.sh` - Integration tests

## Architecture Support

### Supported GPUs

- **GFX12 (RDNA4)**: ✅ **Currently Supported**
  - Radeon RX 9070 series
  - All hardware blocks (SQ, CPC, CPF, GL1, GL2, CHA, CHC, etc.)

- **GFX9 (Vega/MI)**: ✅ **Supported**
  - MI100, MI200, Vega series
  - SQ, GL2C, TA, GRBM blocks

- **GFX10/11**: ❌ **Not Currently Supported**
  - Framework exists but event mappings incomplete

### Hardware Blocks

The module provides counters for these GPU hardware blocks:

| Block | Description | Example Counters |
|-------|-------------|-----------------|
| SQ | Shader Processor (Sequencer) | waves, instructions, busy_cycles |
| GL2C | Graphics L2 Cache | hit, miss, requests |
| GL1A/C | Graphics L1 Cache | hit, miss, latency |
| CPC | Command Processor Compute | busy, stall, packets |
| CPF | Command Processor Fetcher | fetch_stall, me_busy |
| CHA/CHC | Cache Hierarchy Arbiters | requests, conflicts |
| TCC | Texture Cache Controller | hit, miss, write |
| TCP | Texture Cache Processor | read, write, stall |

## Development

### Building Components

```bash
# Build only kernel module
cmake -B build -DBUILD_KERNEL_MODULE=ON -DBUILD_USERSPACE_TOOLS=OFF -DBUILD_TESTS=OFF && cmake --build build --target kernel_module

# Build only userspace tools
cmake -B build -DBUILD_KERNEL_MODULE=OFF -DBUILD_USERSPACE_TOOLS=ON -DBUILD_TESTS=OFF && cmake --build build

# Build only tests
cmake -B build -DBUILD_KERNEL_MODULE=OFF -DBUILD_USERSPACE_TOOLS=OFF -DBUILD_TESTS=ON && cmake --build build

# Clean everything
rm -rf build
```

### Running Tests

```bash
# Build and run tests
cmake -B build -DBUILD_TESTS=ON && cmake --build build

# Run PM4 packet validation
build/src/aql_c/tests/test_pm4

# Run counter registry tests
build/src/aql_c/tests/test_counter_registry

# Run packet generation tests
build/src/aql_c/tests/test_packet_generation
```

### Debug Output

```bash
# Enable kernel debug messages
echo 'module amdgpu_pmu +p' | sudo tee /sys/kernel/debug/dynamic_debug/control

# View trace output
sudo cat /sys/kernel/debug/tracing/trace

# Check kernel log
sudo dmesg -w | grep -i pmu
```

### Tools

**PM4 Packet Generator:**
```bash
build/src/aql_c/tools/packet_gen_tool <gpu_arch> <block> <event_id>
```

**PM4 Decoder:**
```bash
build/src/aql_c/tools/pm4_decoder <hex_packet_data>
```

## File Structure

```
perf-dkms/
├── src/
│   ├── pmu_main.c              # Perf PMU driver
│   ├── pmu_events.c            # Event definitions
│   ├── pmu_dimension.h         # Dimension encoding/validation
│   ├── aql_perf.c/h            # AQL session & GPU discovery
│   ├── aql_packet_ops.c        # PM4 packet operations
│   ├── aql_pmu_integration.c   # PMU-AQL bridge
│   ├── aql_error_recovery.c    # Error handling & recovery
│   ├── aql_queue_manager.c/h   # AQL queue lifecycle
│   ├── kfd_ioctl_bridge.c/h    # KFD ioctl bridge (vm_mmap)
│   └── aql_c/                  # AQL C library (kernel/userspace)
│       ├── packet_generation.c # PM4 packet generation
│       ├── pm4_packets.c       # Low-level PM4 primitives
│       ├── counter_registry.c  # Counter definitions
│       ├── arch_creator.c      # Architecture factory
│       ├── gfx12_creator.c     # GFX12 implementation
│       ├── gfx12_events.c      # GFX12 event definitions
│       ├── gfx9_creator.c      # GFX9 implementation
│       ├── gfx9_events.c       # GFX9 event definitions
│       ├── aql_queue.c/h       # Ring buffer & IB pool
│       ├── tests/              # Validation tests
│       └── tools/              # Packet tools
├── docs/                       # Documentation
├── test/                       # Integration tests
└── README.md                   # This file
```

## How It Works

### Architecture Overview

```
┌─────────────┐
│  perf tool  │  (userspace)
└──────┬──────┘
       │ syscall
┌──────▼──────────────────────┐
│   Linux Perf Subsystem      │
└──────┬──────────────────────┘
       │ PMU callbacks
┌──────▼──────────────────────┐
│   amdgpu_pmu (this module)    │  (kernel)
│  ┌─────────────────────┐    │
│  │ AQL Packet Gen (C)  │    │
│  │ - PM4 packets       │    │
│  │ - Counter registry  │    │
│  │ - Arch detection    │    │
│  └─────────────────────┘    │
└──────┬──────────────────────┘
       │ PM4 commands
┌──────▼──────────────────────┐
│   AMD GPU Hardware          │
│   (GFX12/RDNA4)             │
└─────────────────────────────┘
```

### Event Flow

1. **User runs perf**: `perf stat -e amdgpu_pmu/sq_waves/ <cmd>`
2. **Perf subsystem**: Calls PMU driver `add()` callback
3. **PMU driver**: Creates AQL measurement session
4. **AQL library**: Generates PM4 START packet
5. **Hardware**: GPU enables SQ_WAVES counter
6. **Monitoring**: Counter accumulates during execution
7. **Read**: PMU driver generates PM4 READ packet
8. **Hardware**: GPU returns counter value
9. **Perf**: Displays counter to user

## Troubleshooting

### Module won't load

```bash
# Check kernel version compatibility
uname -r
ls /lib/modules/$(uname -r)/build

# Check for errors
sudo dmesg | grep -i error
```

### No events visible

```bash
# Verify sysfs registration
ls /sys/bus/event_source/devices/amdgpu_pmu/

# Check event files
cat /sys/bus/event_source/devices/amdgpu_pmu/events/*
```

### Perf can't find events

```bash
# Refresh perf cache
sudo rm -rf ~/.debug/
perf list | grep amdgpu_pmu

# Check permissions
sudo chmod -R 755 /sys/bus/event_source/devices/amdgpu_pmu/
```

### Build failures

```bash
# Ensure kernel headers match running kernel
sudo apt install linux-headers-$(uname -r)

# Clean and rebuild
rm -rf build && cmake -B build && cmake --build build
```

## Unloading

```bash
# Remove module
sudo rmmod amdgpu_pmu

# Verify removal
lsmod | grep amdgpu_pmu

# Check for errors
dmesg | tail -20
```

## Limitations

### Known Bugs

**⚠️ Critical Issues:**

1. **First Dimension Only**: The module currently only supports returning counter values for the first hardware dimension (e.g., first shader engine, first compute unit). Multi-dimensional counter reads are not yet fully implemented.

2. **Memory Leaks on Stop**: The `stop` operation may not properly clean up resources in certain scenarios, leading to memory leaks. This occurs when:
   - Multiple events are stopped in quick succession
   - System shutdown interrupts the cleanup sequence
   - GPU state transitions happen during stop operations

3. **GFX12 Only**: **This module currently only supports GFX12/RDNA4 GPUs**. Attempting to use on GFX9/10/11 architectures will fail. The architecture detection code exists but event mappings and register definitions are incomplete for older generations.

### Current Restrictions

1. **Hardware Integration**: Uses real GPU hardware access via:
   - KFD (Kernel Fusion Driver) integration
   - AMDGPU driver coordination
   - GPU memory allocation for PM4 buffers

2. **Architecture Support**:
   - **GFX12/RDNA4: Full support** (RX 9070 series)
   - **GFX9/Vega/MI: Supported** (MI100, MI200, Vega series)
   - **GFX10/11: Not supported** (framework only, missing event definitions)

4. **Kernel Compatibility**:
   - Development kernels may produce `.o` instead of `.ko` files
   - DKMS requires stable kernel versions

### Future Work

- [ ] Complete GFX9/10/11 support
- [ ] Interrupt-driven counter collection
- [ ] SPM (Streaming Performance Monitor) support
- [ ] Multi-GPU enumeration improvements

## References

### AMD Documentation
- [AMD GPU Performance Counters](https://www.amd.com)
- PM4 Packet Format Specifications
- GFX12/RDNA4 Architecture Documentation

### Linux Kernel
- [Perf Subsystem Documentation](https://www.kernel.org/doc/html/latest/admin-guide/perf-security.html)
- [PMU Driver Development](https://www.kernel.org/doc/Documentation/core-api/pmu.rst)

### Related Projects
- AQLProfile (C++ implementation)
- ROCm Profiler SDK
- ROCProfiler

## License

See LICENSE file for details.

## Contributing

This is a research/development module. Contributions welcome for:
- Additional GPU architecture support
- Hardware integration improvements
- Performance optimizations
- Documentation enhancements

## Contact

For issues and questions, please file a GitHub issue.
