# CommandPool Refactoring Analysis: Moving from Global Singleton to Per-Stream Pools

## Executive Summary

This document analyzes the feasibility and impact of moving the `CommandPool` from a global static singleton instance to per-stream instances within `HostQueue`. This change aims to eliminate contention bottlenecks in multithreaded applications with many concurrent streams.

## Current Architecture

### CommandPool Implementation

The `CommandPool` is currently implemented as a **static singleton** with the following characteristics:

- **Location**: `rocclr/platform/command.cpp` (lines 338-408)
- **Access Pattern**: `CommandPool::instance()` returns a static singleton
- **Thread Safety**: Protected by a single `std::mutex mutex_`
- **Storage**: Ring buffer with 64 entries (`q_size_ = 64`)
- **Memory Management**: 
  - Allocates aligned memory using `std::aligned_alloc(maxAlignment_, maxSize_)`
  - Reuses deallocated command memory when pool is not full
  - Frees memory when pool is full

### Command Types Using CommandPool

The following command types use the pool via custom `operator new` and `release()` methods:

1. **ReadMemoryCommand** - `operator new()` and `release()` (lines 684-707)
2. **WriteMemoryCommand** - `operator new()` and `release()` (lines 714-733)
3. **FillMemoryCommand** - `operator new()` and `release()` (lines 744-765)
4. **CopyMemoryCommand** - `operator new()` and `release()` (lines 799-820)
5. **CopyMemoryP2PCommand** - `operator new()` and `release()` (lines 1003-1024)
6. **Marker** - `operator new()` and `release()` (lines 1027-1048)

### Current Allocation Flow

```cpp
// Command creation
void* ReadMemoryCommand::operator new(size_t size) {
  void* ptr = CommandPool::instance().allocate();  // Global singleton access
  // ...
  return ptr;
}

// Command destruction
uint ReadMemoryCommand::release() {
  uint newCount = referenceCount_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (newCount == 0) {
    if (terminate()) {
      CommandPool::instance().deallocate(this);  // Global singleton access
      return 0;
    }
  }
  return newCount;
}
```

### Problem: Contention Bottleneck

In a multithreaded application with many streams:
- **All threads** compete for the same global `CommandPool::instance()` mutex
- High-frequency command allocation/deallocation creates lock contention
- Performance degrades as the number of concurrent streams increases
- The single mutex serializes all command pool operations across all streams

## Proposed Solution: Per-Stream CommandPool

### Architecture Changes

1. **Move CommandPool into HostQueue**
   - Each `HostQueue` instance owns its own `CommandPool`
   - Eliminates cross-stream contention
   - Commands allocated from a stream's pool are returned to the same pool

2. **Update Command Allocation**
   - Commands already have a `queue_` pointer (set in constructor)
   - Commands can access their queue's pool via `queue()->commandPool()`
   - No need to pass additional parameters

3. **Lifecycle Management**
   - Pool created when `HostQueue` is constructed
   - Pool destroyed when `HostQueue` is destroyed
   - No global cleanup needed

### Implementation Plan

#### Step 1: Move CommandPool Class Definition
- Keep `CommandPool` class definition in `command.cpp` (or move to header if needed)
- Remove static `instance()` method
- Make it a regular class (no singleton pattern)

#### Step 2: Add CommandPool to HostQueue
```cpp
// In commandqueue.hpp
class HostQueue : public CommandQueue {
  // ... existing members ...
private:
  CommandPool commandPool_;  // Per-queue command pool
};
```

#### Step 3: Update Command Allocation Methods
```cpp
// Before:
void* ReadMemoryCommand::operator new(size_t size) {
  void* ptr = CommandPool::instance().allocate();
  // ...
}

// After:
void* ReadMemoryCommand::operator new(size_t size, HostQueue& queue) {
  void* ptr = queue.commandPool().allocate();
  // ...
}
```

**Challenge**: `operator new` is called with `new ReadMemoryCommand(...)`, but we need access to the queue. The queue is passed to the constructor, not `operator new`.

**Solution**: Commands already store `queue_` pointer. We can use a two-phase approach:
- Phase 1: Allocate memory (may need temporary global pool or direct allocation)
- Phase 2: After construction, move to queue's pool (not practical)

**Better Solution**: Use placement new or modify allocation pattern:
- Option A: Allocate from queue's pool before construction
- Option B: Store pool reference in command and deallocate to correct pool
- Option C: Use a thread-local or queue-specific allocation mechanism

#### Step 4: Update Command Deallocation
```cpp
// Before:
uint ReadMemoryCommand::release() {
  // ...
  CommandPool::instance().deallocate(this);
  // ...
}

// After:
uint ReadMemoryCommand::release() {
  // ...
  queue_->commandPool().deallocate(this);  // Use queue's pool
  // ...
}
```

**Note**: This is straightforward since `queue_` is already available in the command.

### Detailed Implementation Strategy

#### Option 1: Two-Phase Allocation (Recommended)

Since `operator new` is called before the constructor, we need a way to get the queue reference. However, commands are always created with a queue parameter:

```cpp
// Current pattern:
ReadMemoryCommand* cmd = new ReadMemoryCommand(queue, ...);

// The queue is available at call site!
```

**Solution**: Use a placement-new-like pattern or thread-local storage:

1. **Thread-Local Queue Context** (Simpler but less clean):
   ```cpp
   thread_local HostQueue* g_currentQueue = nullptr;
   
   void* ReadMemoryCommand::operator new(size_t size) {
     HostQueue* queue = g_currentQueue;
     if (queue) {
       return queue->commandPool().allocate();
     }
     // Fallback to direct allocation
     return std::aligned_alloc(alignof(ReadMemoryCommand), size);
   }
   ```

2. **Queue Parameter in operator new** (Requires syntax change):
   ```cpp
   // This would require: new(queue) ReadMemoryCommand(...)
   // Not standard C++ placement new syntax
   ```

3. **Allocate from Queue Before Construction** (Most practical):
   ```cpp
   // At call sites, allocate from queue first:
   void* mem = queue.commandPool().allocate();
   ReadMemoryCommand* cmd = new(mem) ReadMemoryCommand(queue, ...);
   ```
   This requires changing all call sites (80+ locations).

#### Option 2: Deferred Pool Assignment (Hybrid Approach)

1. Allocate commands using a temporary mechanism (direct allocation or small per-thread pool)
2. After construction, commands have `queue_` pointer
3. On deallocation, return to the correct queue's pool
4. **Problem**: Can't reuse memory from different queues efficiently

#### Option 3: Queue-Scoped Allocation Helper (Recommended)

Create a helper that wraps command creation:

```cpp
template<typename CmdType, typename... Args>
CmdType* createCommand(HostQueue& queue, Args&&... args) {
  void* mem = queue.commandPool().allocate();
  return new(mem) CmdType(queue, std::forward<Args>(args)...);
}

// Usage:
auto cmd = createCommand<ReadMemoryCommand>(queue, CL_COMMAND_READ_BUFFER, ...);
```

This requires updating all 80+ call sites but provides clean semantics.

### Code Changes Required

#### Files to Modify

1. **rocclr/platform/commandqueue.hpp**
   - Add `CommandPool commandPool_;` member to `HostQueue`
   - Add `CommandPool& commandPool()` accessor method

2. **rocclr/platform/commandqueue.cpp**
   - Initialize `commandPool_` in `HostQueue` constructor
   - Implement `commandPool()` accessor

3. **rocclr/platform/command.cpp**
   - Remove `CommandPool::instance()` static method
   - Update all 6 command types' `operator new()` methods
   - Update all 6 command types' `release()` methods to use `queue_->commandPool()`

4. **All command creation sites** (80+ locations):
   - `hipamd/src/hip_memory.cpp`
   - `hipamd/src/hip_stream.cpp`
   - `hipamd/src/hip_event.cpp`
   - `rocclr/platform/commandqueue.cpp`
   - `opencl/amdocl/cl_execute.cpp`
   - `opencl/amdocl/cl_memobj.cpp`
   - And others...

### Benefits

1. **Eliminates Contention**: Each stream has its own pool, no cross-stream locking
2. **Better Locality**: Commands allocated from a stream are reused by the same stream
3. **Scalability**: Performance scales with number of streams (no global bottleneck)
4. **Memory Efficiency**: Per-stream pools can be sized appropriately
5. **Thread Safety**: Each pool only accessed by its stream's thread (mostly)

### Challenges and Considerations

#### Challenge 1: operator new() Timing
- `operator new()` is called before constructor
- Queue reference not available in `operator new()` signature
- **Solution**: Use helper function or thread-local context

#### Challenge 2: Cross-Queue Command References
- Commands may reference events from other queues
- Commands are destroyed when reference count reaches zero
- **Impact**: Low - commands are typically destroyed by their owning queue

#### Challenge 3: Memory Pool Sizing
- Current: 64 entries shared across all streams
- Per-stream: 64 entries per stream
- **Memory Impact**: N streams × 64 entries × maxSize_ bytes
- **Mitigation**: Could make pool size configurable or smaller per-stream

#### Challenge 4: Thread Safety Within Queue
- Commands may be allocated/deallocated from different threads
- `HostQueue::append()` may be called from any thread
- **Solution**: CommandPool mutex still needed, but contention is per-stream only

#### Challenge 5: Backward Compatibility
- Need to ensure no regression in single-stream scenarios
- Performance should be equal or better

### Testing Considerations

1. **Single Stream**: Verify no performance regression
2. **Multiple Streams**: Measure contention reduction
3. **High Concurrency**: Test with many concurrent streams
4. **Memory Leaks**: Ensure pools are properly cleaned up
5. **Command Lifecycle**: Verify commands are correctly returned to pools

### Migration Strategy

1. **Phase 1**: Implement per-queue pools alongside global pool (feature flag)
2. **Phase 2**: Update command allocation to use queue pools
3. **Phase 3**: Update all call sites to use new allocation pattern
4. **Phase 4**: Remove global pool after validation
5. **Phase 5**: Performance testing and optimization

### Alternative: Thread-Local Pool

Instead of per-queue pools, consider thread-local pools:
- Simpler implementation (no queue parameter needed)
- Still reduces contention (per-thread instead of global)
- **Drawback**: Threads may service multiple queues, less optimal locality

### Recommendation

**Proceed with per-queue CommandPool implementation** using Option 3 (Queue-Scoped Allocation Helper):

1. **High Impact**: Eliminates major contention bottleneck
2. **Manageable Complexity**: Clear ownership model (queue owns pool)
3. **Good Locality**: Commands reused within same stream
4. **Incremental Migration**: Can be done with feature flags

The main effort is updating ~80 call sites to use the allocation helper, but this provides the cleanest semantics and best performance.

## Implementation Example

### Step 1: Modify CommandPool Class

```cpp
// In command.cpp - Remove singleton pattern
class CommandPool {
public:
  CommandPool() {
    static_assert(((q_size_ & (q_size_ - 1)) == 0) && "q_size must be power of 2");
  }
  
  // Remove: static CommandPool& instance();
  
  template <typename CmdType>
  void deallocate(CmdType *ptr) {
    // ... existing implementation ...
  }
  
  void *allocate() {
    // ... existing implementation ...
  }
  
  // ... rest of implementation unchanged ...
};
```

### Step 2: Add CommandPool to HostQueue

```cpp
// In commandqueue.hpp
class HostQueue : public CommandQueue {
  // ... existing members ...
  
public:
  // Accessor for command pool
  CommandPool& commandPool() { return commandPool_; }
  const CommandPool& commandPool() const { return commandPool_; }

private:
  CommandPool commandPool_;  // Per-queue command pool
  // ... rest of members ...
};
```

```cpp
// In commandqueue.cpp - Initialize in constructor
HostQueue::HostQueue(Context& context, Device& device, ...)
    : CommandQueue(...),
      commandPool_(),  // Initialize pool
      // ... other initializations ...
{
  // ... existing constructor code ...
}
```

### Step 3: Create Allocation Helper

```cpp
// In command.hpp or a new command_utils.hpp
namespace amd {

// Helper function to create commands using queue's pool
template<typename CmdType, typename... Args>
CmdType* createCommand(HostQueue& queue, Args&&... args) {
  void* mem = queue.commandPool().allocate();
  if (mem == nullptr) {
    return nullptr;
  }
  return new(mem) CmdType(queue, std::forward<Args>(args)...);
}

} // namespace amd
```

### Step 4: Update Command Deallocation

```cpp
// In command.cpp - Update all 6 command types
uint ReadMemoryCommand::release() {
  uint newCount = referenceCount_.fetch_sub(1, std::memory_order_acq_rel) - 1;
  if (newCount == 0) {
    if (terminate()) {
      // Use queue's pool instead of global singleton
      queue_->commandPool().deallocate(this);
      return 0;
    }
  }
  return newCount;
}

// Repeat for: WriteMemoryCommand, FillMemoryCommand, 
//            CopyMemoryCommand, CopyMemoryP2PCommand, Marker
```

### Step 5: Update Command Allocation (Remove operator new)

Since we're using placement new via the helper, we can either:
- Keep `operator new` as fallback (for compatibility)
- Remove it entirely (cleaner, but requires all call sites updated)

**Option A: Keep as fallback**
```cpp
void* ReadMemoryCommand::operator new(size_t size) {
  // Fallback: direct allocation if helper not used
  return std::aligned_alloc(alignof(ReadMemoryCommand), size);
}
```

**Option B: Remove operator new** (preferred after migration)

### Step 6: Update Call Sites

```cpp
// Before:
amd::ReadMemoryCommand* cmd = new amd::ReadMemoryCommand(
    *pStream, CL_COMMAND_READ_BUFFER, waitList, ...);

// After:
amd::ReadMemoryCommand* cmd = amd::createCommand<amd::ReadMemoryCommand>(
    *pStream, CL_COMMAND_READ_BUFFER, waitList, ...);
```

### Migration Example: hip_memory.cpp

```cpp
// Current code (line 587):
command = new amd::ReadMemoryCommand(*pStream, CL_COMMAND_READ_BUFFER, waitList,
                                     *srcBuffer, origin, size, dst, rowPitch, slicePitch);

// Migrated code:
command = amd::createCommand<amd::ReadMemoryCommand>(*pStream, CL_COMMAND_READ_BUFFER, 
                                                      waitList, *srcBuffer, origin, size, 
                                                      dst, rowPitch, slicePitch);
```

## Performance Impact Estimate

### Current Bottleneck
- **Single mutex** protecting global pool
- **N threads** contending for same lock
- Lock hold time: ~100-500ns per allocation/deallocation
- **Contention cost**: O(N) threads × lock overhead

### After Refactoring
- **N mutexes** (one per stream)
- **1 thread** per stream typically (or small number)
- Lock hold time: Same (~100-500ns)
- **Contention cost**: O(1) per stream

### Expected Improvement
- **Single stream**: No change (or slight improvement from better locality)
- **Multiple streams**: Near-linear scaling with number of streams
- **High concurrency (16+ streams)**: 10-100x improvement in allocation throughput

## Risk Assessment

### Low Risk
- ✅ Command deallocation (already has queue pointer)
- ✅ Pool initialization/destruction (RAII in HostQueue)
- ✅ Memory management (same algorithm, just per-queue)

### Medium Risk
- ⚠️ Call site updates (80+ locations, but mechanical)
- ⚠️ Testing coverage (need multi-stream scenarios)
- ⚠️ Backward compatibility during migration

### Mitigation Strategies
1. **Feature flag**: Enable per-queue pools behind flag
2. **Gradual migration**: Update call sites incrementally
3. **Fallback mechanism**: Keep global pool as fallback initially
4. **Comprehensive testing**: Multi-stream stress tests

## Conclusion

Moving CommandPool from a global singleton to per-queue instances is **highly recommended**:

- **Solves real performance problem** in multithreaded applications
- **Clear implementation path** with manageable complexity
- **Significant scalability improvement** expected
- **Low risk** with proper testing and gradual migration

The main implementation effort is mechanical (updating call sites), and the architectural change is sound and well-scoped.

