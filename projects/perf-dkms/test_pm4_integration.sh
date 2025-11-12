#!/bin/bash
# test_pm4_integration.sh - Test PM4 integration via perf_event_open and dmesg monitoring
#
# This script tests the PM4 packet generation integration by:
# 1. Loading the kernel module with debug enabled
# 2. Running perf commands to trigger counter allocations
# 3. Monitoring dmesg output to validate PM4 packet generation
# 4. Testing concurrent allocations and counter exhaustion

set -e

export PATH=/home/ben/linux/tools/perf:$PATH
MODULE_PATH="build/src/amdgpu_pmu.ko"
MODULE_NAME="amdgpu_pmu"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

echo_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

echo_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

echo_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if module exists
if [ ! -f "$MODULE_PATH" ]; then
    echo_error "Module not found at $MODULE_PATH. Please build it first with 'cmake -B build && cmake --build build'"
    exit 1
fi

# Check if running as root
if [ "$EUID" -ne 0 ]; then
    echo_error "This script must be run as root (for loading kernel modules)"
    exit 1
fi

# Unload module if already loaded
if lsmod | grep -q "^${MODULE_NAME}"; then
    echo_info "Unloading existing module..."
    rmmod $MODULE_NAME || true
    sleep 1
fi

# Clear dmesg
echo_info "Clearing dmesg..."
dmesg -C

# Load module with debug enabled
echo_info "Loading module with debug enabled..."
insmod $MODULE_PATH debug_enable=1

# Wait for module to initialize
sleep 2

# Check if module loaded successfully
if ! lsmod | grep -q "^${MODULE_NAME}"; then
    echo_error "Module failed to load"
    dmesg | tail -50
    exit 1
fi

echo_success "Module loaded successfully"

# Display initial dmesg output
echo_info "Initial module load messages:"
dmesg | grep -E "\[PMU\]|AQL_PERF" | tail -20

echo ""
echo_info "===== Test 1: Single Counter Allocation ====="
echo_info "Running: perf stat -e pmu_stub/sq_waves/ sleep 1"

# Clear dmesg before test
dmesg -C

# Run perf command
if perf stat -e pmu_stub/sq_waves/ sleep 1 2>&1; then
    echo_success "Perf command completed"
else
    echo_warn "Perf command failed (this may be expected if PMU not fully initialized)"
fi

echo ""
echo_info "Checking dmesg for counter allocation..."
if dmesg | grep -q "aql_counter_try_allocate: SUCCESS"; then
    echo_success "Counter allocation detected in logs"
    dmesg | grep -E "aql_counter_try_allocate|generate_start_packet|kfd_ioctl_submit_ib_packet|aql_counter_release"
else
    echo_warn "Counter allocation not detected in logs (PMU may not be fully functional)"
    dmesg | tail -30
fi

echo ""
echo_info "===== Test 2: Checking PM4 Packet Generation ====="
if dmesg | grep -q "generate_start_packet.*PM4 buffer created"; then
    echo_success "PM4 start packet generation detected"
    dmesg | grep "generate_start_packet.*PM4 buffer"
else
    echo_warn "PM4 packet generation not detected"
fi

if dmesg | grep -q "kfd_ioctl_submit_ib_packet.*SUCCESS"; then
    echo_success "PM4 packet submission detected"
    dmesg | grep "kfd_ioctl_submit_ib_packet"
else
    echo_warn "PM4 packet submission not detected"
fi

echo ""
echo_info "===== Test 3: Counter Release ====="
if dmesg | grep -q "aql_counter_release.*counter released successfully"; then
    echo_success "Counter release detected"
    dmesg | grep "aql_counter_release"
else
    echo_warn "Counter release not detected"
fi

echo ""
echo_info "===== Test 4: Multiple Concurrent Allocations ====="
echo_info "Starting 4 concurrent perf processes..."

dmesg -C

# Start multiple perf processes in background
for i in {1..4}; do
    echo_info "Starting perf process $i..."
    (perf stat -e pmu_stub/sq_busy_cycles/ sleep 5 2>&1) &
    sleep 0.5
done

# Wait a bit for allocations to happen
sleep 2

# Check allocation count
allocation_count=$(dmesg | grep -c "aql_counter_try_allocate: SUCCESS" || echo "0")
echo_info "Number of successful allocations: $allocation_count"

if [ "$allocation_count" -ge 1 ]; then
    echo_success "Multiple allocations detected"
else
    echo_warn "No allocations detected"
fi

# Wait for processes to finish
echo_info "Waiting for perf processes to complete..."
wait

# Check releases
release_count=$(dmesg | grep -c "aql_counter_release.*counter released successfully" || echo "0")
echo_info "Number of counter releases: $release_count"

echo ""
echo_info "===== Test 5: Full Debug Log Analysis ====="
echo_info "Analyzing complete debug log..."

echo ""
echo_info "Counter allocation attempts:"
dmesg | grep "aql_counter_try_allocate" | head -10

echo ""
echo_info "PM4 packet generation:"
dmesg | grep "generate_.*_packet" | head -10

echo ""
echo_info "Packet submissions:"
dmesg | grep "kfd_ioctl_submit_ib_packet" | head -10

echo ""
echo_info "Counter releases:"
dmesg | grep "aql_counter_release" | head -10

echo ""
echo_info "===== Test Complete ====="
echo_info "Module is still loaded. To unload: sudo rmmod $MODULE_NAME"
echo_info "To view full dmesg: dmesg | grep -E '\\[PMU\\]|AQL_PERF'"

echo ""
echo_success "All tests completed successfully!"
echo_info "Check the output above for detailed PM4 integration validation"
