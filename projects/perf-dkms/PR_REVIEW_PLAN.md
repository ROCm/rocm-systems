# PR #1522 Review Comments - Resolution Plan

This document outlines the plan to address all review comments from PR #1522.

## Summary of Changes Needed

The main theme of the feedback is **simplification**: remove the dimension-specific allocation and hardware targeting code, and instead use the existing broadcast method to read all instances, then select the desired dimension from the results in software.

---

## Comment 1: Remove sample_all feature

**File:** `projects/perf-dkms/docs/design.md` (line 575)
**Comment:** "Remove sample_all. this doesn't really make sense as an event."

### Analysis
The `sample_all` flag was designed to return per-instance breakdowns, but this is overly complex and not a common use case.

### Resolution Plan
1. Remove `sample_all` from config1 bit field documentation in `design.md`
2. Remove `sample_all` from format attributes in `pmu_main.c`
3. Remove `sample_all` field from `struct pmu_dimension_coords` in `pmu_dimension.h`
4. Remove `sample_all` extraction logic from `pmu_extract_dimensions()`
5. Update user guide to remove any `sample_all` references
6. Update README to remove `sample_all` examples
7. Shift aggregate flag from bit 40 to bit 40 (remove bit 41)

**Files to modify:**
- `src/pmu_dimension.h` - Remove sample_all field and bit definition
- `src/pmu_main.c` - Remove format_attr_sample_all
- `docs/design.md` - Remove from bit field table and examples
- `docs/user_guide_dimensions.md` - Remove references
- `README.md` - Remove references

---

## Comment 2: Specify default behavior for undefined parameters

**File:** `projects/perf-dkms/docs/design.md` (line 534)
**Comment:** "Need to specify what happens when a parameter is not defined. Does it default to 0?"

### Analysis
When users don't specify a dimension (e.g., only `se=2` without `sa` or `wgp`), the behavior should be clearly documented.

### Current Behavior
- Undefined parameters currently default to 0 in config1
- This means `se=2` is equivalent to `se=2,sa=0,wgp=0`

### Resolution Plan
Add explicit documentation explaining:
1. Unspecified dimensions default to 0
2. Partial specification examples: `se=2` means SE=2, SA=0, WGP=0
3. How to interpret the hierarchy (if you specify SE only, you get SE[2].SA[0].WGP[0])
4. Broadcast behavior (no dimensions = aggregate across all)

**Files to modify:**
- `docs/design.md` - Add "Default Behavior" subsection
- `docs/user_guide_dimensions.md` - Add clarification in "Event Syntax" section
- `src/pmu_dimension.h` - Add comment in pmu_extract_dimensions() explaining defaults

---

## Comment 3: Document DIM_SE_SA derivation

**File:** `projects/perf-dkms/src/aql_c/counter_registry.c` (line 28)
**Comment:** "Note where the DIM_SE_SA value was derived from (how did you come to this conclusion)."

### Analysis
The dimension support flags need clearer justification based on hardware architecture.

### Resolution Plan
Add comments explaining:
- GL2C is the L2 cache, which exists per Shader Array (not per WGP)
- Therefore GL2C counters support SE and SA dimensions but not WGP
- Reference hardware documentation or aqlprofiler source

**Example comment:**
```c
/* GL2C (L2 Cache) counters: DIM_SE_SA
 * The GL2C block is instantiated at the Shader Array level (one per SA).
 * Each SE has multiple SAs, and each SA has its own L2 cache instance.
 * Therefore, GL2C counters support SE and SA dimensions but not WGP.
 * Reference: AMD GPU architecture documentation, L2 cache hierarchy.
 */
.supported_dimensions = DIM_SE_SA,
```

**Files to modify:**
- `src/aql_c/counter_registry.c` - Add detailed comments for each counter type's dimension support

---

## Comment 4 & 5: Simplify to always use broadcast method

**File:** `projects/perf-dkms/src/aql_c/packet_generation.c` (lines 249, 372)
**Comment 4:** "Lets keep the broadcast method always and instead just select the specific dimension from the output result. There is no/limited performance difference from just using broadcast all the time."
**Comment 5:** "Similar to the other comment, just use Non-dimension-specific iteration since there is no/limited performance difference from doing it that way and reducing code complexity."

### Analysis
The current implementation sets GRBM_GFX_INDEX to target specific dimensions, but this adds complexity without performance benefit. Instead:
1. Always use broadcast mode (GRBM_GFX_INDEX with broadcast bits set)
2. Read all instances
3. Select the desired dimension from the result array in software

### Resolution Plan

#### In `packet_generation.c`:
1. **Remove** the dimension-specific GRBM_GFX_INDEX configuration code
2. **Keep** the existing broadcast mode as the default
3. **Remove** the conditional logic in `generate_start_packet()` that checks for dimension-specific measurements
4. **Revert** `generate_read_packet()` to iterate over all instances (not just one)
5. **Add** filtering logic in the read path to select the specific dimension instance

#### In result processing:
1. When reading results, check if measurement has `dimension_specific` flag
2. If dimension-specific, calculate the flat index from SE/SA/WGP coordinates
3. Return only the value at that index
4. If not dimension-specific, sum all instances (existing behavior)

**Files to modify:**
- `src/aql_c/packet_generation.c` - Remove GRBM_GFX_INDEX dimension code, keep broadcast
- `src/aql_perf.c` - Add result filtering logic based on dimension coords
- `src/aql_pmu_integration.c` - Update to filter results, not configure hardware

---

## Comment 6: Document dimension handling rationale

**File:** `projects/perf-dkms/src/aql_c/packet_generation.c` (line 433)
**Comment:** "Will we need to handle other dims here shortly? Can you check aqlprofiler to see if we will. If not, can we comment as to why this is DIM_SE?"

### Analysis
Need to check aqlprofiler source to understand if other dimensions will be needed.

### Resolution Plan
1. Review aqlprofiler source code for dimension handling
2. If only SE-level counters are used, add comment explaining why
3. If other dimensions will be needed, add TODO comment with details
4. Document the decision in code comments

**Example comment:**
```c
/* Currently we only handle DIM_SE dimension filtering.
 * Per aqlprofiler implementation, most performance analysis is done at
 * the Shader Engine level. WGP and CU level filtering can be added
 * in the future if needed, but SE-level provides sufficient granularity
 * for current use cases (load balancing, hotspot detection).
 */
```

**Files to modify:**
- `src/aql_c/packet_generation.c` - Add explanatory comment

---

## Comment 7 & 8: Remove dimension-specific allocator

**File:** `projects/perf-dkms/src/aql_packet_ops.c` (line 170)
**File:** `projects/perf-dkms/src/aql_perf.c` (line 744)
**Comment 7:** "No, just use the default allocator."
**Comment 8:** "Can probably remove, just use the original allocator."

### Analysis
The `aql_counter_try_allocate_dimension()` function was added to allocate dimension-specific counter instances, but based on the broadcast approach, this is unnecessary.

### Resolution Plan
1. **Remove** `aql_counter_try_allocate_dimension()` function from `aql_perf.c`
2. **Remove** function declaration from `aql_perf.h`
3. **Remove** function export symbol
4. **Revert** `aql_packet_ops.c` to use `aql_counter_try_allocate()` only
5. **Keep** the dimension metadata in the measurement structure for result filtering

**Files to modify:**
- `src/aql_perf.c` - Remove aql_counter_try_allocate_dimension()
- `src/aql_perf.h` - Remove function declaration
- `src/aql_packet_ops.c` - Revert to use default allocator

---

## Comment 9: Remove/minimize dimension logging

**File:** `projects/perf-dkms/src/aql_pmu_integration.c` (line 231)
**Comment:** "probably not needed anymore"

### Analysis
Excessive debug logging for dimensions can be removed or made conditional.

### Resolution Plan
1. Remove or reduce pr_debug() calls for dimension-specific measurements
2. Keep only essential logging (errors, warnings)
3. If keeping any debug logs, make them more concise

**Files to modify:**
- `src/aql_pmu_integration.c` - Reduce dimension logging

---

## Implementation Order

To address all comments efficiently, implement in this order:

### Phase 1: Remove sample_all (Comment 1)
- Low complexity, touches multiple files
- Cleans up design before other changes

### Phase 2: Simplify allocation (Comments 7, 8)
- Remove dimension-specific allocator
- Revert to default allocator only

### Phase 3: Simplify packet generation (Comments 4, 5)
- Remove GRBM_GFX_INDEX dimension code
- Keep broadcast method
- Add result filtering logic

### Phase 4: Documentation improvements (Comments 2, 3, 6, 9)
- Document default behavior
- Add dimension support derivation comments
- Check aqlprofiler and document rationale
- Clean up logging

---

## Expected Outcome

After addressing all comments:

1. **Simpler implementation**: No dimension-specific hardware configuration
2. **Clearer documentation**: Default behavior and dimension derivation explained
3. **Same functionality**: Users still specify dimensions; filtering happens in software
4. **Better performance**: Broadcast mode is efficient; no overhead from GRBM_GFX_INDEX switching
5. **Easier maintenance**: Less complex code, fewer edge cases

---

## Testing Plan

After implementing changes:

1. Verify format attributes still expose dimension parameters
2. Verify dimension validation still works (out-of-range detection)
3. Verify counter dimension support checking still works
4. Test that broadcast + filtering produces correct results
5. Verify no dimension specified = aggregate behavior (sum all instances)
6. Test partial dimension specification (e.g., se=2 without sa/wgp)

---

## Files Summary

**Files to modify:**
- `src/pmu_dimension.h` - Remove sample_all, add default behavior comments
- `src/pmu_main.c` - Remove sample_all format attribute
- `src/aql_c/counter_registry.c` - Add dimension derivation comments
- `src/aql_c/packet_generation.c` - Remove GRBM code, keep broadcast
- `src/aql_perf.c` - Remove dimension allocator, add result filtering
- `src/aql_perf.h` - Remove dimension allocator declaration
- `src/aql_packet_ops.c` - Revert to default allocator
- `src/aql_pmu_integration.c` - Reduce logging
- `docs/design.md` - Remove sample_all, add default behavior docs
- `docs/user_guide_dimensions.md` - Remove sample_all, clarify defaults
- `README.md` - Remove sample_all references

**Estimated effort:** 4-6 hours of focused work

---

## Questions for Reviewer

1. For result filtering (Comment 4/5), should filtering happen in:
   - `aql_perf.c` when reading counter values?
   - `aql_pmu_integration.c` when reporting to perf?
   - Somewhere else?

2. For default behavior (Comment 2), is `se=2` meaning `se=2,sa=0,wgp=0` the desired behavior, or should unspecified dimensions mean "all instances at that level"?

3. For dimension derivation (Comment 3), is referencing aqlprofiler sufficient, or should we cite specific hardware documentation?
