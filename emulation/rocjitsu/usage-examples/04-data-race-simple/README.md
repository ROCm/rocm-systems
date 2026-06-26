# Example 4: Data Race Detection

## Objective

Learn to detect and fix data races in GPU kernels using rocjitsu's race detector:
- Identify concurrent write conflicts
- Understand race conditions in shared memory
- Use atomic operations correctly
- Enable and interpret race detection reports

## What This Example Demonstrates

1. **Data race patterns** - Unprotected concurrent writes
2. **Race detector** - Built-in race detection plugin
3. **Atomic operations** - `atomicAdd` for safe updates
4. **Synchronization** - `__syncthreads()` usage
5. **Fixing races** - Before/after comparison

## Key Debugging Points

### 1. Detecting Races
- Enable race detector: `RJ_SINKS=race_detector`
- Look for "RACE DETECTED" messages
- Identify conflicting memory addresses

### 2. Understanding Race Reports
```
RACE DETECTED:
  Address: 0x7f8a40000100
  Thread 1 (block=0, thread=5): WRITE at PC=0x1234
  Thread 2 (block=0, thread=12): WRITE at PC=0x1234
  Both threads accessed same location without synchronization
```

### 3. Common Race Patterns
- **Histogram updates** - Multiple threads incrementing same bin
- **Reduction operations** - Summing to shared result
- **Flag setting** - Multiple threads writing to same flag

## Files

- `src/histogram_race.cpp` - Buggy histogram with race condition
- `src/histogram_fixed.cpp` - Fixed version using atomics
- `Makefile` - Build and test both versions
- `expected-output/race-report.txt` - Sample race detection output

## The Bug

### Problematic Code (histogram_race.cpp)

```cpp
__global__ void histogram_buggy(int *data, int *bins, int N, int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < N) {
    int bin = data[i] % num_bins;
    bins[bin]++;  // RACE: multiple threads may update same bin!
  }
}
```

**Problem**: Multiple threads can read-modify-write the same bin concurrently, leading to lost updates.

### Fixed Code (histogram_fixed.cpp)

```cpp
__global__ void histogram_atomic(int *data, int *bins, int N, int num_bins) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  
  if (i < N) {
    int bin = data[i] % num_bins;
    atomicAdd(&bins[bin], 1);  // SAFE: atomic read-modify-write
  }
}
```

**Solution**: `atomicAdd` ensures the operation is atomic (indivisible).

## Building

```bash
cd usage-examples/04-data-race-simple
make
```

Builds:
- `build/histogram_race` - Buggy version
- `build/histogram_fixed` - Fixed version

## Running

### Run Buggy Version (Detect Race)

```bash
make run-race
```

Equivalent to:
```bash
RJ_SINKS=race_detector rocjitsu --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/histogram_race
```

### Run Fixed Version (No Race)

```bash
make run-fixed
```

Equivalent to:
```bash
RJ_SINKS=race_detector rocjitsu --config ../../configs/amdgpu_cdna4_kmd.json -- ./build/histogram_fixed
```

### Compare Results

```bash
make compare
```

Runs both versions and compares output.

## Expected Output

### Buggy Version Output

```
Histogram Example - WITH DATA RACE
  Input size: 10000 elements
  Number of bins: 256
  
Allocating memory...
Copying data to device...
Launching kernel: grid(157, 1, 1), block(64, 1, 1)

[rocjitsu] RACE DETECTOR ENABLED
[rocjitsu] RACE DETECTED at address 0x7f8a40001000
  Workitem (0, 0, 5) wrote at PC 0x402150
  Workitem (0, 0, 12) wrote at PC 0x402150
  Both threads in same wavefront accessed same memory location
  
[rocjitsu] RACE DETECTED at address 0x7f8a40001004
  Workitem (0, 0, 3) wrote at PC 0x402150
  Workitem (0, 0, 9) wrote at PC 0x402150
  
[rocjitsu] Total races detected: 47

Copying results back...
  
Verification: FAILED
  Expected sum: 10000
  Actual sum: 9953
  Lost updates: 47

RACE CONDITION DETECTED - Results are incorrect!
```

### Fixed Version Output

```
Histogram Example - FIXED WITH ATOMICS
  Input size: 10000 elements
  Number of bins: 256
  
Allocating memory...
Copying data to device...
Launching kernel: grid(157, 1, 1), block(64, 1, 1)

[rocjitsu] RACE DETECTOR ENABLED
[rocjitsu] Kernel completed successfully
[rocjitsu] No races detected

Copying results back...
  
Verification: PASSED
  Expected sum: 10000
  Actual sum: 10000
  All histogram bins correct!

NO RACES DETECTED - Results are correct!
```

## Common Race Patterns and Fixes

### Pattern 1: Histogram / Counting

**Buggy**:
```cpp
bins[index]++;  // Race condition
```

**Fixed**:
```cpp
atomicAdd(&bins[index], 1);
```

### Pattern 2: Finding Maximum

**Buggy**:
```cpp
if (val > *max_val) {
  *max_val = val;  // Race condition
}
```

**Fixed**:
```cpp
atomicMax(max_val, val);
```

### Pattern 3: Reduction (Sum)

**Buggy**:
```cpp
*result += partial_sum;  // Race condition
```

**Fixed**:
```cpp
atomicAdd(result, partial_sum);
```

### Pattern 4: Shared Memory Reduction

**Safer approach using synchronization**:
```cpp
__shared__ float shared_data[256];

// Each thread writes to its own location (no race)
shared_data[threadIdx.x] = my_value;
__syncthreads();  // Ensure all writes complete

// Reduction tree (carefully structured to avoid races)
for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
  if (threadIdx.x < stride) {
    shared_data[threadIdx.x] += shared_data[threadIdx.x + stride];
  }
  __syncthreads();  // Synchronize after each reduction step
}

// Thread 0 writes final result
if (threadIdx.x == 0) {
  atomicAdd(global_result, shared_data[0]);
}
```

## Understanding Race Detection Output

### Race Report Format

```
[rocjitsu] RACE DETECTED at address 0xADDRESS
  Workitem (block_x, block_y, thread_id) ACTION at PC PROGRAM_COUNTER
  Workitem (block_x, block_y, thread_id) ACTION at PC PROGRAM_COUNTER
  Description of the conflict
```

### Fields Explained

- **Address**: Memory location where race occurred
- **Workitem**: Block and thread IDs involved
- **ACTION**: READ or WRITE
- **PC**: Program counter (instruction location)
- **Description**: Type of race detected

### Race Types

1. **Write-Write Race**: Two threads write to same location
2. **Read-Write Race**: One reads while another writes
3. **Write-Read Race**: One writes while another reads

## Debugging Workflow

### Step 1: Enable Race Detector

```bash
RJ_SINKS=race_detector make run-race
```

### Step 2: Analyze Reports

- Count total races
- Identify most frequent addresses
- Note which kernel/PC causes races

### Step 3: Fix the Code

Common fixes:
- Use atomic operations
- Add synchronization (`__syncthreads()`)
- Restructure algorithm to avoid conflicts
- Use separate memory per thread

### Step 4: Verify Fix

```bash
RJ_SINKS=race_detector make run-fixed
```

Should see: "No races detected"

## Performance Considerations

### Atomic Operations Cost

Atomics are **slower** than regular operations:
- Serialize updates to same location
- May cause thread divergence
- But: **correctness > performance**

### Optimization Strategies

1. **Reduce contention** - More bins in histogram
2. **Local reduction first** - Accumulate locally, then atomic update
3. **Shared memory** - Use shared memory + atomics, then global atomic

Example optimization:
```cpp
__shared__ int local_bins[NUM_BINS];

// Initialize shared bins
if (threadIdx.x < NUM_BINS) {
  local_bins[threadIdx.x] = 0;
}
__syncthreads();

// Update local histogram
int bin = data[i] % NUM_BINS;
atomicAdd(&local_bins[bin], 1);
__syncthreads();

// One thread per bin writes to global memory
if (threadIdx.x < NUM_BINS) {
  atomicAdd(&global_bins[threadIdx.x], local_bins[threadIdx.x]);
}
```

## Exercises

1. **Create intentional race** - Modify fixed version to remove atomics
2. **Test different patterns** - Implement finding min/max with atomics
3. **Measure overhead** - Time buggy vs atomic versions
4. **Optimize** - Implement two-level reduction (shared + global)

## Key Takeaways

- Race detector finds concurrent access conflicts
- Atomics ensure safe read-modify-write operations
- `__syncthreads()` coordinates threads in a block
- Always verify with race detector enabled
- Incorrect results often indicate hidden races

## Next Steps

- [Example 5: Global Memory Races](../05-global-memory-race/) - More complex race patterns
- [Example 6: Memory Coalescing](../06-memory-coalescing/) - Performance optimization
- [Example 8: GEMM Debugging](../08-gemm-debugging/) - Real-world matrix operations