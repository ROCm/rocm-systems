# AMD GPU Performance Monitoring - Perf PMU Stub

> **⚠️ EXPERIMENTAL - NOT FOR PRODUCTION USE**
>
> This is a research and development project. The module is not ready for general use and may cause system instability.
>
> **Requirements:**
> - GFX12/RDNA3 GPUs only (MI300, RX 7000 series)
> - **Patched Linux kernel required** (see Prerequisites section)
>
> Use at your own risk in development/testing environments only.

A Linux kernel module implementing AMD GPU performance counter integration with the perf subsystem. This module provides AQL (Asynchronous Queuing Language) packet generation for GPU performance monitoring on GFX12/RDNA3 architectures.

## Overview

This kernel module bridges AMD GPU hardware performance counters with the Linux perf interface, enabling standard perf tools to monitor GPU performance events. It includes:

- **AQL C Library**: Pure C implementation of PM4 packet generation for GPU counter programming
- **Perf PMU Driver**: Complete Linux perf subsystem integration
- **Hardware Support**: GFX12/RDNA3 GPU architecture support
- **Counter Registry**: Architecture-specific performance event definitions

## Features

- ✅ Linux perf subsystem integration
- ✅ AQL/PM4 packet generation for GPU counter control
- ✅ GFX12/RDNA3 hardware support
- ✅ Multi-GPU architecture support
- ✅ Hardware block counter definitions (SQ, CPC, GL2C, etc.)
- ✅ Standard perf tool compatibility
- ✅ DKMS-compatible installation

## Quick Start

### Prerequisites

**⚠️ IMPORTANT: Linux Kernel Patch Required**

This module requires a patched Linux kernel to function properly. The `linux-kernel.patch` file must be applied to your kernel source before building this module.

**Tested kernel base commit:** `e6b9dce0aeeb91dfc0974ab87f02454e24566182`

```bash
# Apply kernel patch
cd /path/to/linux/kernel/source
git checkout e6b9dce0aeeb91dfc0974ab87f02454e24566182  # Or use your current kernel version
patch -p1 < /path/to/perf-dkms/linux-kernel.patch

# Rebuild and install kernel
make -j$(nproc)
sudo make modules_install
sudo make install

# Reboot into patched kernel
sudo reboot
```

**After kernel is patched and running:**

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
cd perf-pmu-stub

# Build kernel module and tools
make

# The build produces:
# - src/amdgpu_pmu.ko (kernel module)
# - src/aql_c/tools/packet_gen_tool (PM4 packet generator)
# - src/aql_c/tools/pm4_decoder (PM4 packet decoder)
# - src/aql_c/tests/* (validation tests)
```

### Loading the Module

```bash
# Load the module
sudo insmod src/amdgpu_pmu.ko

# Or if using .o file (development kernels):
sudo insmod src/amdgpu_pmu.o

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

## Architecture Support

### Supported GPUs

- **GFX12 (RDNA3)**: ✅ **Currently Supported**
  - MI300 series
  - Radeon RX 7000 series
  - All hardware blocks (SQ, CPC, CPF, GL1, GL2, CHA, CHC, etc.)

- **GFX9/10/11**: ❌ **Not Currently Supported**
  - Framework exists but event mappings incomplete
  - Will fail on non-GFX12 hardware

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
cd src && make

# Build only userspace tools
cd src/aql_c/tools && make

# Build only tests
cd src/aql_c/tests && make

# Clean everything
make clean
```

### Running Tests

```bash
# Run PM4 packet validation
cd src/aql_c/tests
./test_pm4

# Run counter registry tests
./test_counter_registry

# Run packet generation tests
./test_packet_generation
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
cd src/aql_c/tools
./packet_gen_tool <gpu_arch> <block> <event_id>
```

**PM4 Decoder:**
```bash
./pm4_decoder <hex_packet_data>
```

## File Structure

```
perf-pmu-stub/
├── src/
│   ├── pmu_main.c              # Perf PMU driver
│   ├── pmu_events.c            # Event definitions
│   ├── aql_perf.c              # AQL performance integration
│   ├── aql_packet_ops.c        # PM4 packet operations
│   ├── aql_pmu_integration.c   # PMU-AQL bridge
│   └── aql_c/                  # AQL C library
│       ├── packet_generation.c # PM4 packet generation
│       ├── counter_registry.c  # Counter definitions
│       ├── arch_creator.c      # Architecture detection
│       ├── gfx12_creator.c     # GFX12 implementation
│       ├── gfx12_events.c      # GFX12 event definitions
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
│   (GFX12/RDNA3)             │
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
make clean && make
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

3. **GFX12 Only**: **This module currently only supports GFX12/RDNA3 GPUs**. Attempting to use on GFX9/10/11 architectures will fail. The architecture detection code exists but event mappings and register definitions are incomplete for older generations.

### Current Restrictions

1. **Kernel Patch Required**: This module requires the `linux-kernel.patch` to be applied to your Linux kernel source. The module will not function without the patched kernel. Tested on base commit `e6b9dce0aeeb91dfc0974ab87f02454e24566182`.

2. **Hardware Integration**: Currently uses simulation; real GPU hardware access requires:
   - KFD (Kernel Fusion Driver) integration
   - AMDGPU driver coordination
   - GPU memory allocation for PM4 buffers

3. **Architecture Support**:
   - **GFX12/RDNA3: Full support** (MI300, RX 7000 series)
   - **GFX9/10/11: Not supported** (framework only, missing event definitions)

4. **Kernel Compatibility**:
   - Development kernels may produce `.o` instead of `.ko` files
   - DKMS requires stable kernel versions

### Future Work

- [ ] Real GPU hardware integration via KFD
- [ ] AMDGPU driver coordination
- [ ] Complete GFX9/10/11 support
- [ ] Interrupt-driven counter collection
- [ ] SPM (Streaming Performance Monitor) support
- [ ] Multi-GPU enumeration improvements

## References

### AMD Documentation
- [AMD GPU Performance Counters](https://www.amd.com)
- PM4 Packet Format Specifications
- GFX12/RDNA3 Architecture Documentation

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
