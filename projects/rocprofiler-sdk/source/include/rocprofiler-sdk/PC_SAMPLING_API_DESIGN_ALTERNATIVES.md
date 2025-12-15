# PC Sampling API Design: Invalid Sample Delivery

## Problem Statement

Users need a way to:
1. Select which record version format they want for valid PC samples
2. Optionally receive invalid/error samples
3. (Future) Optionally receive partially valid samples

How should the API allow users to control this?

---

## Approach 1: Flags (RECOMMENDED)

**File:** `pc_sampling_flags.h`

```c
typedef enum {
    ROCPROFILER_PC_SAMPLING_CONFIG_NONE = 0,
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES = (1 << 0),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES = (1 << 1),
} rocprofiler_pc_sampling_config_flags_t;

rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES);
```

### Pros
- **Simple and clear**: Single function call does everything
- **Less error-prone**: User can't mess up ordering
- **Consistent with other APIs**: Flags are a common pattern
- **Self-documenting**: `INCLUDE_INVALID_SAMPLES` clearly states intent
- **Future-proof**: Easy to add new flags like `INCLUDE_PARTIAL_SAMPLES`
- **Better UX**: Configure once and done

### Cons
- Adds a new enum type to the API surface

### Usage Example
```c
// Valid samples only
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_1,
    sizeof(rocprofiler_pc_sampling_record_v1_t),
    ROCPROFILER_PC_SAMPLING_CONFIG_NONE);

// Valid + invalid samples
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES);

// Future: Valid + invalid + partial
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t),
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_INVALID_SAMPLES |
    ROCPROFILER_PC_SAMPLING_CONFIG_INCLUDE_PARTIAL_SAMPLES);
```

---

## Approach 2: Multiple Configuration Calls

**File:** `pc_sampling_multiple_conf_calls.h`

```c
// Step 1: Lock the valid record format to v2
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t),
    0);

// Step 2: Enable invalid sample delivery
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0,
    sizeof(rocprofiler_pc_sampling_record_v0_t),
    0);
```

### Locking Semantics
- First call with non-v0 version LOCKS the valid record format
- Subsequent calls with different non-v0 versions are REJECTED
- Additional calls with VERSION_0 enable invalid sample delivery
- Order matters: must lock valid format before enabling v0

### Pros
- Similar pattern to `rocprofiler_configure_buffer_tracing_service`
- No new enum needed

### Cons
- **Complex semantics**: "Lock valid version first, then enable v0" is not obvious
- **Error-prone**: Easy to get the ordering wrong
- **Ambiguity with v0-only**: Needs special handling
- **Documentation burden**: Needs 5 examples to explain all cases
- **Confusing for users**: "Why do I call configure twice for the same thing?"
- **Idempotency complexity**: Different behavior for v0 vs non-v0 calls
- **Not analogous to buffer tracing**: Buffer tracing enables different *services* (HIP, HSA, etc.), but here you're just configuring delivery options for the *same* data

### Key Differences from Buffer Tracing
- **Buffer tracing**: Each "kind" is a different service (HIP API, HSA API, etc.)
- **PC Sampling**: Each "version" is a different FORMAT of the SAME data
- **Buffer tracing**: Multiple kinds can coexist naturally
- **PC Sampling**: Multiple versions are mutually exclusive (except v0)

---

## Approach 3: Version Array/Set

```c
rocprofiler_pc_sampling_record_version_t versions[] = {
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,  // Valid samples
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_0   // Invalid samples
};

rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    versions, 2);  // array + count
```

### Pros
- Very explicit about what you get

### Cons
- Can only pick ONE valid version anyway, so array is misleading
- User might think they can get v1 AND v2 simultaneously (impossible)
- No clear way to validate at compile time

---

## Approach 4: Separate Invalid Sample Control Function

```c
// Step 1: Configure the main service (valid samples only)
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t));

// Step 2: Optionally enable invalid sample delivery
rocprofiler_enable_pc_sampling_invalid_records(context_id, agent_id);

// Future
rocprofiler_enable_pc_sampling_partial_records(context_id, agent_id);
```

### Pros
- Very clear separation of concerns
- Optional feature feels optional
- Easy to discover in API docs
- Can disable later if needed (add disable function)

### Cons
- More functions in API surface
- Still requires two calls
- What if user calls enable before configure?
- Needs state validation

---

## Approach 5: Separate Buffers

```c
rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval,
    buffer_valid,   // valid samples
    buffer_invalid, // invalid samples (or NULL to discard)
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t));
```

### Pros
- Clean separation of valid vs invalid streams
- User can handle them with different callbacks
- `NULL buffer_invalid` = discard (very clear)
- Performance: no need to check record type in hot path

### Cons
- Requires managing two buffers
- More complex for simple use cases
- What about partially valid samples? Need a third buffer?
- Buffer lifecycle management complexity

---

## Approach 6: Policy Enum Instead of Flags

```c
typedef enum {
    ROCPROFILER_PC_SAMPLING_POLICY_VALID_ONLY,
    ROCPROFILER_PC_SAMPLING_POLICY_VALID_AND_INVALID,
    ROCPROFILER_PC_SAMPLING_POLICY_VALID_AND_PARTIAL,      // future
    ROCPROFILER_PC_SAMPLING_POLICY_ALL_RECORDS,            // future
} rocprofiler_pc_sampling_policy_t;

rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    sizeof(rocprofiler_pc_sampling_record_v2_t),
    ROCPROFILER_PC_SAMPLING_POLICY_VALID_AND_INVALID);
```

### Pros
- Very explicit and readable
- Can't combine incompatible options
- Clear documentation for each policy

### Cons
- Enum explosion if you add more record types
  - valid+invalid
  - valid+partial
  - invalid+partial
  - all three
- Less flexible than flags for future combinations
- Need new enum value for each combination

---

## Approach 7: Configuration Struct (Builder Pattern)

```c
typedef struct {
    rocprofiler_pc_sampling_record_version_t valid_version;
    size_t                                   valid_size;
    bool                                     include_invalid;
    bool                                     include_partial;
} rocprofiler_pc_sampling_config_t;

rocprofiler_pc_sampling_config_t config = {
    .valid_version = ROCPROFILER_PC_SAMPLING_RECORD_VERSION_2,
    .valid_size = sizeof(rocprofiler_pc_sampling_record_v2_t),
    .include_invalid = true,
    .include_partial = false
};

rocprofiler_configure_pc_sampling_service(
    context_id, agent_id, method, unit, interval, buffer_id,
    &config);
```

### Pros
- Very clear what each field does
- Easy to extend with new options
- Named parameters effectively
- Can add versioning to struct itself

### Cons
- More boilerplate code
- ABI concerns if struct changes (need versioning strategy)
- Overkill for simple on/off toggles
- Pointer lifetime management

---

## Recommendation Ranking

1. **Approach 1: Flags** ⭐ **RECOMMENDED**
   - Best balance of simplicity and flexibility
   - Industry standard pattern
   - Easy to use, hard to misuse

2. **Approach 4: Separate Function**
   - Clean but requires two calls
   - Good if invalid samples are truly rare/optional

3. **Approach 6: Policy Enum**
   - Simple but less flexible long-term
   - Combinatorial explosion risk

4. **Approach 7: Config Struct**
   - Future-proof but heavy
   - Better for complex configuration with many parameters

5. **Approach 5: Separate Buffers**
   - Interesting for performance but complex
   - Might be worth considering for advanced use cases

6. **Approach 2: Multiple Calls**
   - Too complex and error-prone
   - Confusing semantics

7. **Approach 3: Version Array**
   - Conceptually confusing
   - Doesn't match the actual constraints

## Decision

**Use Approach 1 (Flags)** as implemented in `pc_sampling_flags.h`.

The flags approach provides the best developer experience with clear, simple semantics that are hard to misuse and easy to extend.
