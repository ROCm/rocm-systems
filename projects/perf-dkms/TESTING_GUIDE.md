# amdgpu_pmu Kernel Module - Testing Guide

This guide provides commands to load, test, and verify the dimension-aware PMU kernel module.

## Prerequisites

- AMD GPU hardware (MI300, MI200, or RDNA3 series)
- Root/sudo access
- Linux perf tools installed
- Kernel headers matching running kernel

## Module Information

**Module file:** `/home/ben/rocm-systems/worktrees/options/projects/perf-dkms/src/amdgpu_pmu.ko`
**Module name:** `amdgpu_pmu`
**Size:** 3.1 MB
**Dependencies:** `amdgpu` driver

---

## Step 1: Verify AMD GPU is Present

```bash
# Check for AMD GPU
lspci | grep -i amd | grep -i vga

# Check amdgpu driver is loaded
lsmod | grep amdgpu

# Get GPU information
cat /sys/class/drm/card0/device/product_name
cat /sys/class/drm/card0/device/gpu_busy_percent
```

**Expected:** AMD GPU detected and amdgpu driver loaded

---

## Step 2: Unload Old Module (if loaded)

```bash
# Check if module is already loaded
lsmod | grep amdgpu_pmu

# Unload if present
sudo rmmod amdgpu_pmu

# Verify it's unloaded
lsmod | grep amdgpu_pmu  # Should return nothing
```

---

## Step 3: Load the Module

```bash
# Load the module
sudo insmod /home/ben/rocm-systems/worktrees/options/projects/perf-dkms/src/amdgpu_pmu.ko

# Verify module loaded successfully
lsmod | grep amdgpu_pmu

# Check module info
modinfo /home/ben/rocm-systems/worktrees/options/projects/perf-dkms/src/amdgpu_pmu.ko
```

**Expected output from lsmod:**
```
amdgpu_pmu             3145728  0
```

---

## Step 4: Check Kernel Logs

```bash
# View module initialization messages
sudo dmesg | tail -50

# Look for amdgpu_pmu specific messages
sudo dmesg | grep -i amdgpu_pmu

# Check for dimension limit messages (should show SE/SA/WGP/CU counts)
sudo dmesg | grep -i "dimension limits"
```

**Expected messages:**
```
[  XXX] amdgpu_pmu: Module loaded
[  XXX] amdgpu_pmu: Dimension limits: xcc=X se=X sa=X wgp=X cu=X
[  XXX] amdgpu_pmu: PMU registered: amdgpu_pmu
```

---

## Step 5: Verify Sysfs Interface

```bash
# Check PMU device exists
ls -l /sys/bus/event_source/devices/amdgpu_pmu

# List all format attributes
ls -l /sys/bus/event_source/devices/amdgpu_pmu/format/

# Check format attributes content
cat /sys/bus/event_source/devices/amdgpu_pmu/format/config
cat /sys/bus/event_source/devices/amdgpu_pmu/format/config1
cat /sys/bus/event_source/devices/amdgpu_pmu/format/se
cat /sys/bus/event_source/devices/amdgpu_pmu/format/sa
cat /sys/bus/event_source/devices/amdgpu_pmu/format/wgp
cat /sys/bus/event_source/devices/amdgpu_pmu/format/xcc
cat /sys/bus/event_source/devices/amdgpu_pmu/format/cu
cat /sys/bus/event_source/devices/amdgpu_pmu/format/aggregate
```

**Expected format output:**
```
config:0-63         (from config file)
config1:0-63        (from config1 file)
config1:8-15        (from se file)
config1:16-23       (from sa file)
config1:24-31       (from wgp file)
config1:0-7         (from xcc file)
config1:32-39       (from cu file)
config1:40          (from aggregate file)
```

---

## Step 6: List Available Events

```bash
# List all amdgpu_pmu events
perf list | grep amdgpu_pmu

# Get detailed event list
perf list amdgpu_pmu

# Count available events
perf list amdgpu_pmu | grep "amdgpu_pmu/" | wc -l
```

**Expected:** Should show multiple events like:
- `amdgpu_pmu/sq_waves/`
- `amdgpu_pmu/gl2c_hit/`
- `amdgpu_pmu/gl2c_miss/`
- etc.

---

## Step 7: Test Basic Event Monitoring (No Dimensions)

```bash
# Monitor a simple counter (aggregated across all dimensions)
sudo perf stat -e amdgpu_pmu/sq_waves/ -a sleep 2

# Monitor multiple counters
sudo perf stat -e amdgpu_pmu/sq_waves/,amdgpu_pmu/gl2c_hit/,amdgpu_pmu/gl2c_miss/ -a sleep 2

# Monitor during a GPU workload (if available)
sudo perf stat -e amdgpu_pmu/sq_waves/ -a -- <your_gpu_application>
```

**Expected output:**
```
 Performance counter stats for 'system wide':

         12,345,678      amdgpu_pmu/sq_waves/

       2.001234567 seconds time elapsed
```

---

## Step 8: Test Dimension-Specific Monitoring

### Test 8.1: Shader Engine (SE) Specific

```bash
# Monitor SE 0 only
sudo perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 2

# Monitor SE 1 only
sudo perf stat -e amdgpu_pmu/sq_waves,se=1/ -a sleep 2

# Monitor SE 2 only
sudo perf stat -e amdgpu_pmu/sq_waves,se=2/ -a sleep 2

# Monitor SE 3 only
sudo perf stat -e amdgpu_pmu/sq_waves,se=3/ -a sleep 2
```

**Expected:** Different values for different shader engines (if workload is unbalanced)

### Test 8.2: SE + SA Specific

```bash
# Monitor SE 0, SA 0
sudo perf stat -e amdgpu_pmu/gl2c_hit,se=0,sa=0/ -a sleep 2

# Monitor SE 0, SA 1
sudo perf stat -e amdgpu_pmu/gl2c_hit,se=0,sa=1/ -a sleep 2

# Monitor SE 2, SA 1
sudo perf stat -e amdgpu_pmu/gl2c_hit,se=2,sa=1/ -a sleep 2
```

**Expected:** L2 cache counters per shader array

### Test 8.3: Full Hierarchy (SE + SA + WGP)

```bash
# Monitor SE 2, SA 1, WGP 3
sudo perf stat -e amdgpu_pmu/sq_waves,se=2,sa=1,wgp=3/ -a sleep 2

# Monitor SE 0, SA 0, WGP 0
sudo perf stat -e amdgpu_pmu/sq_waves,se=0,sa=0,wgp=0/ -a sleep 2
```

**Expected:** Fine-grained per-WGP counter values

### Test 8.4: Raw config1 Encoding

```bash
# SE=2 using raw config1 (SE is bits 8-15, so 0x0200)
sudo perf stat -e amdgpu_pmu/sq_waves,config1=0x0200/ -a sleep 2

# SE=2, SA=1 (SE=0x02 at bits 8-15, SA=0x01 at bits 16-23 = 0x010200)
sudo perf stat -e amdgpu_pmu/sq_waves,config1=0x010200/ -a sleep 2

# SE=2, SA=1, WGP=3 (WGP=0x03 at bits 24-31 = 0x03010200)
sudo perf stat -e amdgpu_pmu/sq_waves,config1=0x03010200/ -a sleep 2
```

**Expected:** Same results as named parameter syntax

---

## Step 9: Test Error Cases

### Test 9.1: Invalid Dimension Values

```bash
# Try to monitor invalid SE (should fail with error)
sudo perf stat -e amdgpu_pmu/sq_waves,se=99/ -a sleep 2

# Try to monitor invalid SA
sudo perf stat -e amdgpu_pmu/sq_waves,se=0,sa=99/ -a sleep 2

# Try to monitor invalid WGP
sudo perf stat -e amdgpu_pmu/sq_waves,se=0,sa=0,wgp=99/ -a sleep 2
```

**Expected:** Error message about dimension out of range

### Test 9.2: Unsupported Dimensions for Counter

```bash
# Try to use SE dimension on global GRBM counter (should fail)
sudo perf stat -e amdgpu_pmu/grbm_count,se=2/ -a sleep 2

# Try to use WGP on GL2C counter (GL2C only supports SE/SA)
sudo perf stat -e amdgpu_pmu/gl2c_hit,se=0,sa=0,wgp=2/ -a sleep 2
```

**Expected:** Error message about unsupported dimensions for this counter

---

## Step 10: Compare Dimension-Specific vs Aggregated

```bash
# Run comparison test
# Monitor all SEs separately and compare to aggregate

# Aggregate (sum of all)
sudo perf stat -e amdgpu_pmu/sq_waves/ -a sleep 5 2>&1 | tee aggregate.txt

# Individual SEs
sudo perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 5 2>&1 | tee se0.txt
sudo perf stat -e amdgpu_pmu/sq_waves,se=1/ -a sleep 5 2>&1 | tee se1.txt
sudo perf stat -e amdgpu_pmu/sq_waves,se=2/ -a sleep 5 2>&1 | tee se2.txt
sudo perf stat -e amdgpu_pmu/sq_waves,se=3/ -a sleep 5 2>&1 | tee se3.txt

# Manually sum SE values and compare to aggregate
# (SE0 + SE1 + SE2 + SE3) should approximately equal aggregate value
```

**Expected:** Aggregate value ≈ sum of individual SE values

---

## Step 11: Monitor Multiple Dimensions Simultaneously

```bash
# Monitor multiple SEs at the same time
sudo perf stat -e amdgpu_pmu/sq_waves,se=0/ \
               -e amdgpu_pmu/sq_waves,se=1/ \
               -e amdgpu_pmu/sq_waves,se=2/ \
               -e amdgpu_pmu/sq_waves,se=3/ \
               -a sleep 5

# Monitor different counters on same SE
sudo perf stat -e amdgpu_pmu/sq_waves,se=2/ \
               -e amdgpu_pmu/sq_insts_valu,se=2/ \
               -e amdgpu_pmu/gl2c_hit,se=2,sa=0/ \
               -e amdgpu_pmu/gl2c_miss,se=2,sa=0/ \
               -a sleep 5
```

**Expected:** Multiple counter values displayed

---

## Step 12: Check Kernel Logs After Testing

```bash
# View recent kernel messages related to dimension monitoring
sudo dmesg | grep -i "dimension" | tail -20

# Check for any errors or warnings
sudo dmesg | grep -E "(ERROR|WARN)" | grep amdgpu_pmu | tail -20

# View aggregation messages
sudo dmesg | grep "Aggregated" | tail -10

# View dimension-specific read messages
sudo dmesg | grep "dimension-specific read" | tail -10
```

**Expected:** Log messages showing dimension-specific reads and aggregations

---

## Step 13: Unload Module

```bash
# Unload the module
sudo rmmod amdgpu_pmu

# Verify it's unloaded
lsmod | grep amdgpu_pmu  # Should return nothing

# Check unload messages
sudo dmesg | tail -10
```

---

## Step 14: Run Automated Test Scripts

```bash
# Run format attribute test
cd /home/ben/rocm-systems/worktrees/options/projects/perf-dkms/test
sudo ./format_test.sh

# Run dimension test
sudo ./dimension_test.sh

# Run GPU workload test (if GPU workload available)
sudo ./gpu_workload_test.sh
```

---

## Troubleshooting

### Module Fails to Load

```bash
# Check for missing dependencies
modprobe --dry-run amdgpu_pmu

# Check kernel ring buffer for errors
sudo dmesg | grep -i error | tail -20

# Verify amdgpu driver is loaded first
lsmod | grep amdgpu

# Check for kernel version mismatch
uname -r
modinfo amdgpu_pmu | grep vermagic
```

### No Events Show Up

```bash
# Verify sysfs interface exists
ls -l /sys/bus/event_source/devices/

# Check perf is finding the PMU
perf list | grep pmu

# Verify event directory exists
ls -l /sys/bus/event_source/devices/amdgpu_pmu/events/
```

### Perf Command Fails

```bash
# Run with verbose output
sudo perf stat -vvv -e amdgpu_pmu/sq_waves/ -a sleep 1

# Check permissions
sudo perf stat -e amdgpu_pmu/sq_waves/ -a sleep 1

# Verify event syntax
perf list amdgpu_pmu | grep sq_waves
```

### Dimension Values Always Zero

```bash
# Check if GPU is idle
cat /sys/class/drm/card0/device/gpu_busy_percent

# Run with active GPU workload
# (counters will be near zero if GPU is idle)

# Check counter is supported by hardware
sudo dmesg | grep "counter.*not supported"
```

---

## Expected Behavior Summary

### ✅ Module loads successfully without errors
### ✅ Sysfs interface created at `/sys/bus/event_source/devices/amdgpu_pmu/`
### ✅ Format attributes expose dimension parameters (se, sa, wgp, etc.)
### ✅ Events show up in `perf list`
### ✅ Basic monitoring works (aggregate mode)
### ✅ Dimension-specific monitoring works (se=X, sa=Y, etc.)
### ✅ Invalid dimensions are rejected with clear errors
### ✅ Unsupported dimensions for counters are rejected
### ✅ Aggregate value ≈ sum of dimension-specific values
### ✅ Kernel logs show dimension-aware operation

---

## Performance Analysis Use Cases

### Use Case 1: Load Balancing Analysis

```bash
# Check if workload is balanced across shader engines
for se in 0 1 2 3; do
    echo "SE $se:"
    sudo perf stat -e amdgpu_pmu/sq_waves,se=$se/ -a sleep 5 2>&1 | grep sq_waves
done
```

### Use Case 2: Hotspot Detection

```bash
# Find which shader array has most L2 cache misses
for se in 0 1 2 3; do
    for sa in 0 1; do
        echo "SE $se SA $sa:"
        sudo perf stat -e amdgpu_pmu/gl2c_miss,se=$se,sa=$sa/ -a sleep 5 2>&1 | grep gl2c_miss
    done
done
```

### Use Case 3: Memory Subsystem Analysis

```bash
# Analyze L2 cache hit rate per shader engine
sudo perf stat -e amdgpu_pmu/gl2c_hit,se=2,sa=0/ \
               -e amdgpu_pmu/gl2c_miss,se=2,sa=0/ \
               -a <gpu_workload>

# Calculate hit rate: hits / (hits + misses) * 100%
```

---

## Quick Test Command Sequence

```bash
# Quick test sequence (copy and paste)
cd /home/ben/rocm-systems/worktrees/options/projects/perf-dkms/src

# Load module
sudo insmod amdgpu_pmu.ko

# Verify load
lsmod | grep amdgpu_pmu

# Check sysfs
ls -l /sys/bus/event_source/devices/amdgpu_pmu/format/

# List events
perf list amdgpu_pmu | head -20

# Test aggregate
sudo perf stat -e amdgpu_pmu/sq_waves/ -a sleep 2

# Test dimension-specific
sudo perf stat -e amdgpu_pmu/sq_waves,se=0/ -a sleep 2

# Check logs
sudo dmesg | grep amdgpu_pmu | tail -20

# Unload
sudo rmmod amdgpu_pmu
```

---

## Notes

- All dimension indices are 0-based (SE: 0-3, SA: 0-1, WGP: 0-3)
- Some counters may require active GPU workload to show non-zero values
- GRBM counters don't support dimensions (they are global)
- Default behavior: unspecified dimensions default to 0 (e.g., `se=2` means `se=2,sa=0,wgp=0`)
- Aggregate mode: no dimensions specified means sum across all instances
- Root/sudo required for most perf commands

---

For additional information, see:
- `/home/ben/rocm-systems/worktrees/options/projects/perf-dkms/docs/design.md`
- `/home/ben/rocm-systems/worktrees/options/projects/perf-dkms/docs/user_guide_dimensions.md`
- `/home/ben/rocm-systems/worktrees/options/projects/perf-dkms/README.md`
