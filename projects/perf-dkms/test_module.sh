#!/bin/bash
# Test script for loading and verifying the PMU stub module

set -e

echo "=== PMU Stub Module Test ==="
echo

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo "This script must be run with sudo"
    echo "Usage: sudo ./test_module.sh"
    exit 1
fi

# Module path
MODULE_PATH="src/amdgpu_pmu.ko"

# Check if module exists
if [ ! -f "$MODULE_PATH" ]; then
    echo "Error: Module not found at $MODULE_PATH"
    echo "Please build the module first with: make"
    exit 1
fi

echo "1. Loading module with debug enabled..."
insmod $MODULE_PATH debug_enable=1 timer_period_ms=100
echo "   ✓ Module loaded successfully"
echo

echo "2. Verifying module is loaded..."
if lsmod | grep -q amdgpu_pmu; then
    echo "   ✓ Module found in lsmod:"
    lsmod | grep amdgpu_pmu
else
    echo "   ✗ Module not found in lsmod"
    exit 1
fi
echo

echo "3. Checking kernel logs..."
dmesg | tail -20 | grep -i amdgpu_pmu || true
echo

echo "4. Checking sysfs interface..."
if [ -d "/sys/bus/event_source/devices/pmu_stub" ]; then
    echo "   ✓ Sysfs interface created at /sys/bus/event_source/devices/pmu_stub"
    echo "   Contents:"
    ls -la /sys/bus/event_source/devices/amdgpu_pmu/
else
    echo "   ✗ Sysfs interface not found"
fi
echo

echo "5. Checking perf integration..."
echo "   Perf PMU list (filtered for pmu_stub):"
perf list | grep -i amdgpu_pmu || echo "   Note: pmu_stub events not visible in perf list"
echo

echo "6. Testing perf stat with module..."
echo "   Running: perf stat -e amdgpu_pmu/cycles/ sleep 0.1"
perf stat -e amdgpu_pmu/cycles/ sleep 0.1 2>&1 || echo "   Note: Direct perf usage may require additional setup"
echo

echo "7. Module information:"
modinfo $MODULE_PATH
echo

echo "8. Unloading module..."
rmmod amdgpu_pmu
echo "   ✓ Module unloaded successfully"
echo

echo "9. Verifying module is unloaded..."
if ! lsmod | grep -q amdgpu_pmu; then
    echo "   ✓ Module successfully removed from kernel"
else
    echo "   ✗ Module still loaded"
    exit 1
fi
echo

echo "=== Test Complete ==="
echo "All basic module operations completed successfully!"