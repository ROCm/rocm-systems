# User Guide: Dimension-Aware Performance Monitoring

This guide explains how to use the dimension-aware performance monitoring features of the amdgpu_pmu driver to monitor specific parts of your AMD GPU hardware.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Understanding GPU Hardware Dimensions](#understanding-gpu-hardware-dimensions)
3. [Event Syntax](#event-syntax)
4. [Common Use Cases](#common-use-cases)
5. [Counter Reference](#counter-reference)
6. [Troubleshooting](#troubleshooting)

## Quick Start

### Prerequisites

- Linux system with AMD GPU (GFX12 or compatible)
- amdgpu_pmu kernel module loaded
- perf tool installed

### Basic Examples

```bash
# Monitor shader engine 0
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1

# Monitor specific shader array within SE 1
perf stat -e amdgpu_pmu/sq_waves,se=1,sa=0/ -a sleep 1

# Monitor full dimension path
perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 1
```

### Verify Installation

```bash
# Check module is loaded
lsmod | grep amdgpu_pmu

# List available events
perf list amdgpu_pmu

# Check format attributes
ls /sys/bus/event_source/devices/amdgpu_pmu/format/
```

## Understanding GPU Hardware Dimensions

### Hardware Hierarchy

AMD GPUs are organized in a hierarchical structure. Understanding this hierarchy helps you target the right hardware component:

```
GPU (1)
 │
 └─── XCC - eXtended Compute Core (1)
       │
       └─── SE - Shader Engine (4 typical)
             │
             └─── SA - Shader Array (2 per SE)
                   │
                   └─── WGP - Work Group Processor (4 per SA)
                         │
                         └─── CU - Compute Unit (distributed)
```

### Example: GFX12 Architecture

A typical GFX12 GPU has:
- **1 XCC** (index 0)
- **4 Shader Engines** (SE 0, 1, 2, 3)
- **2 Shader Arrays per SE** (SA 0, 1)
- **4 WGPs per SA** (WGP 0, 1, 2, 3)
- **64 Compute Units total**

Total addressable units: 4 SE × 2 SA × 4 WGP = 32 unique SE/SA/WGP combinations

### What Each Level Represents

**Shader Engine (SE)**:
- Top-level parallel compute unit
- Independent rasterizers and geometry processors
- Typically 4 SEs on modern GPUs
- Workloads are distributed across SEs for parallelism

**Shader Array (SA)**:
- Subdivision within a Shader Engine
- Contains multiple WGPs
- Usually 2 SAs per SE
- Shares L1 cache within SA

**Work Group Processor (WGP)**:
- Basic unit of compute scheduling
- Contains multiple Compute Units
- Work groups (wavefronts) are scheduled to WGPs
- Usually 4 WGPs per SA

**Compute Unit (CU)**:
- Individual SIMD execution unit
- Executes shader instructions
- Multiple CUs per WGP

### Finding Your GPU's Dimensions

```bash
# Method 1: Check kernel log
dmesg | grep "dimension limits"

# Method 2: Using rocminfo (if ROCm installed)
rocminfo | grep -E "Shader Engine|Compute Unit"

# Method 3: Try monitoring and check for errors
# If you get "dimension out of range", you exceeded the limit
perf stat -e amdgpu_pmu/sq_waves,se=10/ -a sleep 1
# Error indicates SE must be < 10
```

## Event Syntax

### Named Parameter Syntax

The recommended way to specify dimensions is using named parameters:

```bash
# Single dimension
-e amdgpu_pmu/sq_waves,se=N/

# Multiple dimensions
-e amdgpu_pmu/sq_waves,se=N,sa=M/
-e amdgpu_pmu/sq_waves,se=N,sa=M,wgp=P/

# All parameters
-e amdgpu_pmu/COUNTER,xcc=X,se=N,sa=M,wgp=P,cu=C/
```

**Available Parameters**:
- `xcc=N` - XCC index (usually 0)
- `se=N` - Shader Engine index (0-3 typical)
- `sa=N` - Shader Array index (0-1 typical)
- `wgp=N` - Work Group Processor index (0-3 typical)
- `cu=N` - Compute Unit index (0-63 typical)

### Raw config1 Encoding

Advanced users can use raw config1 values:

```bash
# SE=1 (bits 8-15): 0x0100
perf stat -e amdgpu_pmu/sq_waves,config1=0x0100/ -a sleep 1

# SE=2, SA=1: 0x00010200
perf stat -e amdgpu_pmu/sq_waves,config1=0x00010200/ -a sleep 1
```

**config1 Bit Layout**:
```
Bits 0-7:    xcc
Bits 8-15:   se
Bits 16-23:  sa
Bits 24-31:  wgp
Bits 32-39:  cu
Bit 40:      aggregate (future)
```

**Default Behavior**:
When dimensions are not specified (no se=/sa=/wgp= parameters), the driver operates in broadcast mode: it reads all hardware instances and aggregates the results.

When you specify dimensions, unspecified sub-dimensions default to 0:
- `se=2` means `se=2,sa=0,wgp=0,cu=0`
- `se=2,sa=1` means `se=2,sa=1,wgp=0,cu=0`

### Combining Multiple Events

Monitor multiple dimensions simultaneously:

```bash
# Compare all 4 shader engines
perf stat \
  -e amdgpu_pmu/sq_waves,se=0/ \
  -e amdgpu_pmu/sq_waves,se=1/ \
  -e amdgpu_pmu/sq_waves,se=2/ \
  -e amdgpu_pmu/sq_waves,se=3/ \
  -a sleep 5

# Monitor different counters on different SEs
perf stat \
  -e amdgpu_pmu/sq_waves,se=0/ \
  -e amdgpu_pmu/sq_insts_valu,se=0/ \
  -e amdgpu_pmu/ta_busy,se=0/ \
  -a ./my_gpu_app
```

## Common Use Cases

### Use Case 1: Load Balancing Analysis

Check if GPU workload is balanced across shader engines:

```bash
#!/bin/bash
# measure_balance.sh

echo "Measuring SE load balance during GPU workload..."

perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -e amdgpu_pmu/sq_waves,se=2/ \
          -e amdgpu_pmu/sq_waves,se=3/ \
          -a ./my_gpu_workload

# Ideal: All SEs show similar counts
# Problem: One SE much higher than others (imbalanced)
```

**Interpreting Results**:
```
 1,234,567  amdgpu_pmu/sq_waves,se=0/
 1,245,890  amdgpu_pmu/sq_waves,se=1/
 1,238,234  amdgpu_pmu/sq_waves,se=2/
 1,241,098  amdgpu_pmu/sq_waves,se=3/
```
✅ Good: Values are similar (±5%) - well balanced

```
 2,456,789  amdgpu_pmu/sq_waves,se=0/
   234,567  amdgpu_pmu/sq_waves,se=1/
   245,890  amdgpu_pmu/sq_waves,se=2/
   238,234  amdgpu_pmu/sq_waves,se=3/
```
⚠️ Problem: SE0 is doing 10x more work - investigate workload distribution

### Use Case 2: Hotspot Detection

Find which shader engine is most active:

```bash
# Continuously monitor and identify hotspot
perf record -e amdgpu_pmu/sq_waves,se=0/ \
            -e amdgpu_pmu/sq_waves,se=1/ \
            -e amdgpu_pmu/sq_waves,se=2/ \
            -e amdgpu_pmu/sq_waves,se=3/ \
            -a sleep 10

perf report --stdio
```

### Use Case 3: Memory Hierarchy Analysis

Monitor cache behavior at different levels:

```bash
# L2 cache hits/misses per Shader Array
perf stat \
  -e amdgpu_pmu/gl2c_hit,se=0,sa=0/ \
  -e amdgpu_pmu/gl2c_miss,se=0,sa=0/ \
  -e amdgpu_pmu/gl2c_hit,se=0,sa=1/ \
  -e amdgpu_pmu/gl2c_miss,se=0,sa=1/ \
  -a ./memory_intensive_app

# Calculate hit rate per SA
```

### Use Case 4: Comparing Workload Patterns

Run different GPU applications and compare their dimension usage:

```bash
# Workload A
perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -a ./workload_a > results_a.txt

# Workload B
perf stat -e amdgpu_pmu/sq_waves,se=0/ \
          -e amdgpu_pmu/sq_waves,se=1/ \
          -a ./workload_b > results_b.txt

# Compare results
diff results_a.txt results_b.txt
```

### Use Case 5: Debugging Compute Kernels

Isolate performance issues to specific hardware units:

```bash
# Monitor specific WGP where kernel is running
# (You need to know which WGP your kernel uses - from profiler)

perf stat \
  -e amdgpu_pmu/sq_waves,se=1,sa=0,wgp=2/ \
  -e amdgpu_pmu/sq_insts_valu,se=1,sa=0,wgp=2/ \
  -e amdgpu_pmu/sq_insts_salu,se=1,sa=0,wgp=2/ \
  -a ./kernel_test
```

## Counter Reference

### Which Counters Support Which Dimensions?

| Counter Group | Supported Dimensions | Example Counters |
|--------------|---------------------|------------------|
| **Global** | None | `grbm_count`, `grbm_busy`, `grbm_gui_active` |
| **SQ (Sequencer)** | SE, SA, WGP | `sq_waves`, `sq_insts_valu`, `sq_insts_salu`, `sq_busy_cycles` |
| **TA (Texture Addressing)** | SE, SA, WGP | `ta_busy`, `ta_total_wavefronts` |
| **GL2C (L2 Cache)** | SE, SA | `gl2c_hit`, `gl2c_miss`, `gl2c_req` |

### Discovering Counter Support

Trying to use dimensions on a global counter will result in an error:

```bash
$ perf stat -e amdgpu_pmu/grbm_count,se=0/ -a sleep 1
Error: counter 'grbm_count' does not support requested dimensions
```

### Most Useful Dimension-Aware Counters

**SQ Counters** (Shader Sequencer):
```bash
# Wave occupancy per SE
-e amdgpu_pmu/sq_waves,se=N/

# VALU instruction throughput
-e amdgpu_pmu/sq_insts_valu,se=N,sa=M/

# SALU instruction throughput
-e amdgpu_pmu/sq_insts_salu,se=N,sa=M/

# Busy cycles (utilization)
-e amdgpu_pmu/sq_busy_cycles,se=N/
```

**TA Counters** (Texture Unit):
```bash
# Texture unit busy
-e amdgpu_pmu/ta_busy,se=N,sa=M,wgp=P/

# Wavefront count
-e amdgpu_pmu/ta_total_wavefronts,se=N,sa=M/
```

**GL2C Counters** (L2 Cache):
```bash
# Cache hit rate per SA
-e amdgpu_pmu/gl2c_hit,se=N,sa=M/
-e amdgpu_pmu/gl2c_miss,se=N,sa=M/
-e amdgpu_pmu/gl2c_req,se=N,sa=M/
```

## Troubleshooting

### Error: "dimension out of range"

```bash
$ perf stat -e amdgpu_pmu/sq_waves,se=5/ -a sleep 1
Error: dimension out of range: se=5 (max: 3)
```

**Solution**: Your GPU only has SEs 0-3. Use a valid SE index.

### Error: "counter does not support requested dimensions"

```bash
$ perf stat -e amdgpu_pmu/grbm_count,se=0/ -a sleep 1
Error: counter 'grbm_count' does not support requested dimensions
```

**Solution**: This is a global counter. Remove dimension parameters or use a different counter that supports dimensions (like `sq_waves`).

### No Events Counted (All Zeros)

```bash
$ perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1

Performance counter stats for 'system wide':

         0      amdgpu_pmu/sq_waves,se=0/

   1.000567890 seconds time elapsed
```

**Possible Causes**:
1. No GPU activity during measurement period
2. Try with an actual GPU workload
3. SE index correct but no work scheduled to that SE

**Solution**:
```bash
# Run with actual GPU workload
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a rocm-smi --showuse &
sleep 2
killall rocm-smi

# Or use a compute application
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a ./my_gpu_app
```

### Module Not Loaded

```bash
$ perf list amdgpu_pmu
List of pre-defined events (to be used in -e):
(empty)
```

**Solution**:
```bash
# Load the module
sudo modprobe amdgpu_pmu

# Verify
lsmod | grep amdgpu_pmu
perf list amdgpu_pmu
```

### Format Attributes Missing

```bash
$ ls /sys/bus/event_source/devices/amdgpu_pmu/format/
ls: cannot access '/sys/bus/event_source/devices/amdgpu_pmu/format/': No such file or directory
```

**Solution**:
1. Module not loaded properly
2. Check dmesg for errors: `dmesg | tail -50 | grep amdgpu_pmu`
3. Reload module: `sudo rmmod amdgpu_pmu && sudo modprobe amdgpu_pmu`

### Permission Denied

```bash
$ perf stat -e amdgpu_pmu/sq_waves,se=0/ sleep 1
Error: Permission denied
```

**Solution**:
```bash
# Either run as root
sudo perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1

# Or use -a flag for system-wide monitoring (requires CAP_PERFMON or root)
perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 1
```

### Checking Dimension Limits

To find your GPU's maximum dimension indices:

```bash
# Check kernel messages at module load
dmesg | grep "amdgpu_pmu.*dimension limits"

# Example output:
# amdgpu_pmu: Dimension limits: XCC=0 SE=3 SA=1 WGP=3 CU=63
```

This means valid ranges are:
- XCC: 0
- SE: 0-3
- SA: 0-1
- WGP: 0-3
- CU: 0-63

## Advanced Topics

### Scripting and Automation

Create scripts to automate dimension monitoring:

```bash
#!/bin/bash
# monitor_all_dimensions.sh
# Monitor all SE/SA combinations

NUM_SE=4
NUM_SA=2

for se in $(seq 0 $((NUM_SE-1))); do
    for sa in $(seq 0 $((NUM_SA-1))); do
        echo "Monitoring SE=$se SA=$sa"
        perf stat -e amdgpu_pmu/sq_waves,se=$se,sa=$sa/ \
                  -a -o results_se${se}_sa${sa}.log sleep 5 &
    done
done

wait
echo "All measurements complete"
```

### Continuous Monitoring

Use perf record for continuous sampling:

```bash
# Record events continuously
perf record -e amdgpu_pmu/sq_waves,se=0/ \
            -e amdgpu_pmu/sq_waves,se=1/ \
            -a sleep 60

# Analyze recording
perf report
perf script
```

### Combining with CPU Events

Monitor GPU dimensions alongside CPU performance:

```bash
perf stat \
  -e amdgpu_pmu/sq_waves,se=0/ \
  -e cpu-cycles \
  -e instructions \
  -a ./gpu_cpu_app
```

## Best Practices

1. **Start Broad, Then Narrow**: Begin with SE-level monitoring, then drill down to SA/WGP if needed

2. **Compare Against Broadcast**: Always measure broadcast mode (no dimensions) first to establish baseline

3. **Use Consistent Duration**: Keep measurement periods consistent for valid comparisons

4. **Account for Noise**: Run multiple iterations and average results for statistical significance

5. **Understand Your Workload**: Know which SEs your application uses before targeting specific dimensions

6. **Check GPU Topology**: Verify dimension limits for your specific GPU before scripting

## Getting Help

- Check kernel logs: `dmesg | grep amdgpu_pmu`
- Review module parameters: `modinfo amdgpu_pmu`
- Test scripts provided: `test/load_test.sh`, `test/perf_test.sh`
- Report issues with full error messages and GPU model

## Summary

Dimension-aware monitoring enables fine-grained GPU performance analysis by targeting specific hardware units. Use named parameters (`se=N,sa=M`) for readability, validate dimension support for each counter, and compare results across dimensions to identify load imbalances and performance bottlenecks.
