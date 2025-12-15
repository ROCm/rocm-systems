# PC Sampling API Redesign: Record Version Selection

## Current Situation

The PC sampling API (`pc_sampling.h:172`) currently has:
- 7 record types: v0 (invalid/error) and v1-v6 (valid samples with different architectures)
- Single configuration call per agent: `rocprofiler_configure_pc_sampling_service()`
- Users specify one `record_version` parameter
- **Problem**: No way to opt-in/out of receiving invalid samples (v0 records)

### Future Requirements
- Need to support partial samples (some fields valid, some invalid)
- Users should control which record types they receive

## Colleague's Proposal: Buffer Tracing Pattern

In `buffer_tracing.h`, the API allows **multiple calls** to `rocprofiler_configure_buffer_tracing_service()` on the same context with different `kind` values:

```c
// Example from samples/api_buffered_tracing/client.cpp:436-459
for(auto itr : {ROCPROFILER_BUFFER_TRACING_HSA_CORE_API,
                ROCPROFILER_BUFFER_TRACING_HSA_AMD_EXT_API}) {
    rocprofiler_configure_buffer_tracing_service(
        client_ctx, itr, nullptr, 0, client_buffer);
}

rocprofiler_configure_buffer_tracing_service(
    client_ctx, ROCPROFILER_BUFFER_TRACING_HIP_RUNTIME_API, nullptr, 0, client_buffer);

rocprofiler_configure_buffer_tracing_service(
    client_ctx, ROCPROFILER_BUFFER_TRACING_KERNEL_DISPATCH, nullptr, 0, client_buffer);
```

### Applying This to PC Sampling

Call `rocprofiler_configure_pc_sampling_service()` multiple times per agent:
- First call with v1 (or any valid version) "locks" the format
- Optional second call with v0 to enable invalid samples
- Subsequent calls with other versions are rejected (except v0)

## Analysis: Why This Doesn't Fit Well

### 1. Semantic Mismatch
- **Buffer tracing**: Each `kind` is a *fundamentally different service*
  - `HIP_RUNTIME_API` ≠ `HSA_CORE_API` ≠ `KERNEL_DISPATCH`
  - They produce different data, go to different buffers, serve different purposes
- **PC Sampling**: Record versions are just *different formats of the same data*
  - v1, v2, v3, etc. are all PC samples with different field layouts
  - All go to the same buffer, serve the same purpose

### 2. Version Selection Ambiguity
If user calls:
```c
rocprofiler_configure_pc_sampling_service(..., VERSION_1, ...);
rocprofiler_configure_pc_sampling_service(..., VERSION_3, ...);
rocprofiler_configure_pc_sampling_service(..., VERSION_5, ...);
```

Which version wins? Biggest? Smallest? First? Last?

This confusion doesn't exist in buffer tracing because each `kind` is independent.

### 3. v0 is Special
- v0 isn't really a "record version" - it's an error indicator
- Treating it like other versions (v1-v6) is conceptually inconsistent
- v0 records contain no meaningful data except "sampling failed"

### 4. User Experience
- Not intuitive: "Why do I call configure twice for one service?"
- Error-prone: Easy to forget the v0 call if you want invalid samples
- Documentation burden: Need to explain the multi-call semantics

## Alternative Solutions

### Option 1: Explicit Boolean Flags (Simplest)

**Use the existing `flags` parameter at line 180:**

```c
// Define new flag enum
typedef enum {
    ROCPROFILER_PC_SAMPLING_CONFIG_NONE = 0,
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES = (1 << 0),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES = (1 << 1),  // future
} rocprofiler_pc_sampling_config_flags_t;

// Usage
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
    sizeof(rocprofiler_pc_sampling_record_v1_t),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES
);
```

#### Pros
- ✅ Simple, clear, explicit
- ✅ Uses existing `flags` parameter (already in the API!)
- ✅ Backward compatible (flags=0 = no invalid samples)
- ✅ Easy to extend: just add new flag bits for partial samples
- ✅ Single configuration call matches user mental model
- ✅ No ambiguity about version selection

#### Cons
- Need to define new flag constants (minor)

---

### Option 2: Configuration Struct (Most Extensible)

```c
typedef struct {
    rocprofiler_pc_sampling_record_version_t version;
    uint32_t flags;  // INCLUDE_INVALID, INCLUDE_PARTIAL, etc.
} rocprofiler_pc_sampling_record_config_t;

rocprofiler_status_t
rocprofiler_configure_pc_sampling_service(
    rocprofiler_context_id_t            context_id,
    rocprofiler_agent_id_t              agent_id,
    rocprofiler_pc_sampling_method_t    method,
    rocprofiler_pc_sampling_unit_t      unit,
    uint64_t                            interval,
    rocprofiler_buffer_id_t             buffer_id,
    const rocprofiler_pc_sampling_record_config_t* config
) ROCPROFILER_API;
```

#### Pros
- ✅ Very extensible for future options
- ✅ Groups related configuration together
- ✅ Clean API design

#### Cons
- ❌ Breaking API change (but you're redesigning anyway)
- ❌ More boilerplate code for users
- ❌ Need to handle struct versioning

---

### Option 3: Always Include v0 Records (Controversial)

Always deliver v0 records alongside the requested version. Add validity field to all records:

```c
typedef struct {
    uint64_t size;
    uint8_t  validity;  // 0=invalid, 1=valid, 2=partial (future)
    // ... rest of fields
} rocprofiler_pc_sampling_record_v1_t;
```

#### Pros
- ✅ Simplest API - no configuration needed
- ✅ Users naturally handle invalids in buffer callback
- ✅ Consistent record structure

#### Cons
- ❌ Forces users to receive records they might not want
- ❌ Wastes buffer space for users who don't care about errors
- ❌ Performance overhead of delivering unwanted data

---

### Option 4: Multiple Configure Calls (Colleague's Proposal)

```c
// Call 1: Lock in valid record format
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1, sizeof(...), 0);

// Call 2: Enable invalid samples
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0, sizeof(...), 0);
```

#### Pros
- ✅ Mirrors existing buffer tracing pattern (team familiarity)

#### Cons
- ❌ Semantic mismatch (versions ≠ independent services)
- ❌ Version selection ambiguity
- ❌ v0 treated like a version (conceptually odd)
- ❌ Not intuitive for users
- ❌ Error-prone
- ❌ Documentation complexity

## Recommendation: Option 1 (Explicit Flags)

**Use the existing `flags` parameter with new flag constants.**

### Rationale

1. **Already in the API**: The `flags` parameter exists at `pc_sampling.h:180` but is currently unused
   ```c
   int flags) ROCPROFILER_API;
   ```

2. **Clear semantics**: One configuration call with all options specified
   ```c
   rocprofiler_configure_pc_sampling_service(
       context_id, agent_id, method, unit, interval, buffer_id,
       ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
       sizeof(rocprofiler_pc_sampling_record_v1_t),
       ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES
   );
   ```

3. **Future-proof**: Easy to extend for partial samples
   ```c
   flags = ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES |
           ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES;
   ```

4. **Backward compatible**: Default `flags=0` means "valid samples only"

5. **Matches user expectations**: "Configure once with all my requirements"

6. **No ambiguity**: Version selection is explicit, flags control what types you want

### Implementation

```c
// Add to pc_sampling.h around line 186

/**
 * @brief Configuration flags for PC sampling record delivery
 */
typedef enum ROCPROFILER_SDK_EXPERIMENTAL rocprofiler_pc_sampling_config_flags_t
{
    ROCPROFILER_PC_SAMPLING_CONFIG_NONE = 0,

    /// Include invalid/error records (v0) in addition to valid samples
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES = (1 << 0),

    /// Include partially valid records (future)
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES = (1 << 1),

    ROCPROFILER_PC_SAMPLING_CONFIG_LAST
} rocprofiler_pc_sampling_config_flags_t;
```

Update documentation at line 152:
```c
/**
 * @param [in] flags Configuration flags controlling record delivery:
 *   - 0 (or ROCPROFILER_PC_SAMPLING_CONFIG_NONE): Deliver only valid samples
 *   - ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES: Also deliver v0
 *     records when sampling errors occur
 *   - ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES: (future) Also
 *     deliver records with some fields invalid
 */
```

## Discussion Points

1. Do we agree that multiple configure calls don't fit the semantic model?
2. Is the flags approach clear enough for users?
3. Should we consider Option 2 (struct) for more extensibility?
4. How do we want to handle the transition if we change the API?
5. Should invalid samples be included by default (flags=0) or opt-in?

## Files Referenced

- `projects/rocprofiler-sdk/source/include/rocprofiler-sdk/pc_sampling.h` (lines 50-180)
- `projects/rocprofiler-sdk/source/include/rocprofiler-sdk/buffer_tracing.h` (lines 585-609)
- `projects/rocprofiler-sdk/samples/api_buffered_tracing/client.cpp` (lines 436-459)
