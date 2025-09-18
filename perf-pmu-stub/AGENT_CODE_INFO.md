# AGENT_CODE_INFO.md - perf-pmu-stub

## Directory Purpose
This directory contains a Linux kernel module that implements a skeleton PMU (Performance Monitoring Unit) driver for the Linux perf subsystem, integrated with an AQL (Asynchronous Queue Language) C library for AMD GPU performance monitoring. The module provides a framework for monitoring GPU performance counters through the standard Linux perf interface with timer-based counter simulation.

## Key Components

### Core Module Structure
- **perf-pmu-stub/** - Root directory for the kernel module project
- **src/** - Main source code directory containing all implementation files
- **src/aql_c/** - C implementation of AQL packet generation library (ported from C++)
- **Makefile** - Top-level build configuration

### Build System
- Uses standard Linux kernel module build infrastructure (kbuild)
- Targets kernel version 6.17.0-rc4-rocm-gdb (custom ROCm kernel)
- Supports standard `make` targets: all, clean, install

## Public APIs

### Module Interface
The module registers with the Linux kernel as `pmu_stub` and provides:
- Standard PMU driver interface through Linux perf subsystem
- sysfs attributes for event format and available events
- Timer-based counter simulation for testing and development
- AQL packet generation for AMD GPU counter programming

### Module Parameters
- `debug_enable` (bool): Enable debug output (default: false)
- `timer_period_ms` (int): Timer period in milliseconds for counter simulation (default: 100)

## Design Patterns

### Architecture Abstraction
The module implements a multi-architecture support pattern:
- Function pointer tables for architecture-specific operations
- Support for GFX9, GFX10, GFX11, and GFX12 GPU architectures
- Runtime architecture detection and operation dispatch

### Resource Management
- Event slot allocation using bitmap for tracking up to 64 concurrent events
- Thread-safe operations using spinlocks for event management
- High-resolution timer for simulating counter updates
- Automatic cleanup on module unload

### Command Generation
- Builder pattern for PM4 command packet generation
- Command buffer abstraction for accumulating GPU commands
- Architecture-specific packet formats handled transparently

## Dependencies

### External Dependencies
- Linux kernel headers (6.17.0-rc4-rocm-gdb or compatible)
- Linux perf subsystem (perf_event.h, device.h)
- Kernel timer infrastructure (hrtimer.h)
- Standard kernel modules (slab.h, spinlock.h, bitmap.h)

### Internal Dependencies
- **pmu_main.c**: Core PMU driver implementation
- **pmu_events.c**: Event handling and counter management utilities
- **pmu_stub.h**: Core data structures and function prototypes
- **aql_c/ library**: Complete AQL packet generation library

## Data Flow

1. **Initialization Flow**:
   - Module load → PMU registration → Timer setup → sysfs attribute creation

2. **Event Configuration Flow**:
   - User space perf tool → PMU event_init → Event validation → Event slot allocation

3. **Data Collection Flow**:
   - Timer interrupt → Counter simulation updates → Event count updates → perf buffer

4. **AQL Packet Generation Flow**:
   - Counter configuration → Architecture detection → PM4 command generation → AQL packet creation

## Integration Points

### Linux Perf Subsystem
- Registers as a PMU driver with perf_event interface (`perf_pmu_register`)
- Provides event format and available events through sysfs attributes
- Implements standard PMU callbacks (event_init, add, del, start, stop, read)
- Supports standard perf tools (perf stat, perf record)

### AQL C Library Integration
- Provides GPU architecture detection and abstraction
- Generates PM4 command packets for counter programming
- Supports multiple AMD GPU architectures (GFX9, GFX10, GFX11, GFX12)
- Implements AQLProfile v2-compatible interface

### Timer-Based Simulation
- Uses high-resolution timer for counter updates (default 100ms)
- Simulates realistic counter increments (cycles, instructions, cache_misses, bandwidth)
- Provides testing infrastructure without requiring real GPU hardware

## Critical Business Logic

### Performance Counter Management
- Supports up to 64 concurrent events using bitmap allocation
- Implements event lifecycle: init → add → start → stop → del
- Provides simulated counters for testing (cycles, instructions, cache_misses, bandwidth)
- Uses atomic counters for thread-safe updates

### Timer Management
- Starts timer when first event is added
- Stops timer when last event is removed
- Configurable period via `timer_period_ms` module parameter
- Updates all active events on each timer expiration

### Architecture Detection and Operations
- Parses GPU architecture strings (e.g., "gfx942", "gfx1101")
- Provides function pointer tables for architecture-specific operations
- Validates counter availability and register addresses per architecture
- Supports CONFIG and UCONFIG register spaces

### Error Handling
- Comprehensive error codes for all AQL operations
- Validates event configurations and register addresses
- Graceful handling of unsupported features (returns appropriate error codes)
- Debug logging controlled by module parameter

## Module States

### Initialization States
1. **Unloaded**: Module not in kernel
2. **Loading**: Module initialization in progress (pmu_stub_init)
3. **Active**: PMU registered and operational
4. **Unloading**: Cleanup in progress (pmu_stub_exit)

### Event States
1. **Uninitialized**: Event slot available
2. **Initialized**: Event configured but not added to PMU
3. **Added**: Event added to PMU, slot allocated
4. **Active**: Event running and accumulating counts
5. **Stopped**: Event paused but still allocated

## Security Considerations
- Uses standard Linux perf subsystem permissions (CAP_PERFMON or CAP_SYS_ADMIN)
- Validates all user-provided event configurations against supported ranges
- Proper bounds checking for all buffer operations and memory access
- Thread-safe operations using appropriate locking mechanisms

## Known Limitations
- Module build issues with kernel 6.17.0-rc4-rocm-gdb (__modfinal stage)
- Currently uses simulated counters - no real hardware integration
- Limited to 64 concurrent events due to static allocation
- Performance counter programming is prepared but not executed on real hardware

## Testing Infrastructure
- Timer-based counter simulation for development and testing
- Comprehensive event lifecycle testing through standard perf tools
- Debug output for troubleshooting packet generation and event management
- Sysfs attributes for runtime monitoring and configuration

## Future Extensions
The module architecture supports:
- Real hardware counter programming (AQL packets ready for GPU submission)
- Additional GPU architectures through new architecture operation tables
- Extended counter types and events beyond the current four basic types
- Integration with KFD for actual GPU command submission
- Performance counter multiplexing for handling more than 64 concurrent events