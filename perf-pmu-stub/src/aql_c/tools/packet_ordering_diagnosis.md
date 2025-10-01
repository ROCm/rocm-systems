# Diagnosis: START Packet Differences for SQ Event 1

## Summary

The START packets for SQ event 1 differ between original and new implementations in **packet ordering** and **register offset calculation**. There are 12 packets in original vs 11 in new, with one critical bug and one missing GRBM broadcast.

## Side-by-Side Comparison

```
Line  ORIGINAL                      NEW                         MEANING
----  ----------------------------  --------------------------  ----------------------------------
  1   EVENT_WRITE (CS_FLUSH)        EVENT_WRITE (CS_FLUSH)      Initial flush
  2   SET_UCONFIG_REG 0x200         SET_UCONFIG_REG 0x200       GRBM broadcast mode
  3   SET_UCONFIG_REG 0x1808        SET_UCONFIG_REG 0x1808      Disable perfmon
  4   SET_UCONFIG_REG 0x19e2        SET_UCONFIG_REG 0x19e2      SQ_PERFCOUNTER_CTRL2
  5*  SET_UCONFIG_REG 0x200         SET_UCONFIG_REG 0x19c0      GRBM vs SQ_PERFCOUNTER0_SELECT
  6*  SET_UCONFIG_REG 0x19c0        SET_UCONFIG_REG 0x19e0      SQ_PERFCOUNTER0_SELECT vs CTRL
  7*  SET_UCONFIG_REG 0x19e0        SET_UCONFIG_REG 0x200       SQ_PERFCOUNTER_CTRL vs GRBM
  8*  SET_UCONFIG_REG 0x200         SET_SH_REG 0xd60b (BUG!)    GRBM vs COMPUTE_PERFCOUNT (wrong!)
  9*  SET_SH_REG 0x20b              SET_UCONFIG_REG 0x1808      COMPUTE_PERFCOUNT vs perfmon
 10   SET_UCONFIG_REG 0x1808        SET_UCONFIG_REG 0x1808      Perfmon disable
 11*  SET_UCONFIG_REG 0x1808        EVENT_WRITE (CS_FLUSH)      Perfmon enable vs flush
 12*  EVENT_WRITE (CS_FLUSH)        (missing)                   Final flush
```

## Register Mapping

| Offset | Absolute | Register Name            | Purpose                          |
|--------|----------|--------------------------|----------------------------------|
| 0x200  | 0xc200   | GRBM_GFX_INDEX           | GPU topology selection           |
| 0x1808 | 0xd808   | CP_PERFMON_CNTL          | Performance monitor control      |
| 0x19e2 | 0xd9e2   | SQ_PERFCOUNTER_CTRL2     | SQ counter force enable + vmid   |
| 0x19c0 | 0xd9c0   | SQ_PERFCOUNTER0_SELECT   | Event selection for counter 0    |
| 0x19e0 | 0xd9e0   | SQ_PERFCOUNTER_CTRL      | SQ counter shader stage enables  |
| 0x20b  | 0x2e0b   | COMPUTE_PERFCOUNT_ENABLE | Compute performance counter flag |
| 0xd60b | 0x1020b  | (INVALID - BUG!)         | Wrong offset due to double sub   |

## Root Causes

### 1. **CRITICAL BUG: Double Subtraction in SET_SH_REG** (Line 8/9)

**Location:** `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/pm4_packets.c:193`

**Problem:**
```c
// pm4_packets.c line 193
offset = (uint16_t)((reg_offset - PERSISTENT_SPACE_START) & 0xFFFF);
```

This function subtracts `PERSISTENT_SPACE_START` from `reg_offset`, but the caller in `packet_generation.c` lines 179-180 already does this subtraction:

```c
// packet_generation.c lines 178-182
ret = pm4_append_write_sh_reg(buffer,
                              arch->control_regs.compute_perfcount_enable -
                                  arch->control_regs.persistent_space_start,  // <-- Already subtracted!
                              0x1, /* enable */
                              0, 0);
```

**Calculation:**
1. `packet_generation.c` passes: `11787 - 11264 = 523 (0x20b)` ✓
2. `pm4_append_write_sh_reg` subtracts again: `523 - 11264 = -10741 = 0xd60b` ✗
3. Cast to uint16_t gives `0xd60b` which is **WRONG**

**Expected behavior:** Original code must pass the ABSOLUTE address (0x2e0b) to `pm4_append_write_sh_reg`, which then calculates the offset (0x20b).

**Impact:** CRITICAL - This causes the wrong register to be written, potentially disabling compute performance counting.

---

### 2. **Missing GRBM Broadcast Before Counter Configuration** (Line 5)

**Location:** `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/packet_generation.c:166`

**Problem:** The original implementation has a GRBM broadcast (0x200) between SQ_PERFCOUNTER_CTRL2 and the counter configuration, but the new implementation goes directly from CTRL2 to counter select.

**Original sequence:**
```
SET_UCONFIG_REG 0x19e2    # SQ_PERFCOUNTER_CTRL2 (force_en=1, vmid_en=0xFFFF)
SET_UCONFIG_REG 0x200     # GRBM broadcast ← MISSING IN NEW
SET_UCONFIG_REG 0x19c0    # SQ_PERFCOUNTER0_SELECT
SET_UCONFIG_REG 0x19e0    # SQ_PERFCOUNTER_CTRL
```

**New sequence:**
```
SET_UCONFIG_REG 0x19e2    # SQ_PERFCOUNTER_CTRL2
SET_UCONFIG_REG 0x19c0    # SQ_PERFCOUNTER0_SELECT ← No GRBM broadcast first!
SET_UCONFIG_REG 0x19e0    # SQ_PERFCOUNTER_CTRL
```

**Why it matters:** The GRBM broadcast ensures the following register writes apply to all GPU instances. Without it, the counter configuration might only apply to instance 0.

**Impact:** FUNCTIONAL - Counter configuration may not be broadcast correctly across GPU topology.

---

### 3. **Packet Count Difference** (12 vs 11)

The original has one extra packet due to the missing GRBM broadcast in the new implementation.

- **Original:** 12 packets (includes extra GRBM broadcast)
- **New:** 11 packets (missing GRBM broadcast before counter config)

---

## Code Analysis: generate_start_packet() Flow

```c
// From packet_generation.c:generate_start_packet()

/* 1. CS partial flush */                    → Packet 1: EVENT_WRITE
/* 2. GRBM broadcast mode */                 → Packet 2: SET_UCONFIG_REG 0x200
/* 3. Disable perfmon initially */           → Packet 3: SET_UCONFIG_REG 0x1808

/* 4. Enable SQ control for SQ counters */
if (has_sq_counters) {
    /* SQ_PERFCOUNTER_CTRL2 */               → Packet 4: SET_UCONFIG_REG 0x19e2
}

/* ⚠️ MISSING: GRBM broadcast here! */       → Should be Packet 5: SET_UCONFIG_REG 0x200

/* 5. Configure each counter */
for (each counter) {
    generate_counter_config(...)             → Packets 5-6: SET_UCONFIG_REG 0x19c0, 0x19e0
}

/* 6. GRBM broadcast again */                → Packet 7: SET_UCONFIG_REG 0x200

/* 7. Enable compute perfcount */
ret = pm4_append_write_sh_reg(buffer,
    arch->control_regs.compute_perfcount_enable -  // ⚠️ BUG: Double subtraction!
        arch->control_regs.persistent_space_start,
    0x1, 0, 0);                              → Packet 8: SET_SH_REG 0xd60b (WRONG!)

/* 8. Enable perfmon (disable first) */      → Packets 9-10: SET_UCONFIG_REG 0x1808 (x2)
/* 9. Final CS partial flush */              → Packet 11: EVENT_WRITE
```

## Recommended Fixes

### Fix 1: Remove Double Subtraction in pm4_append_write_sh_reg Caller

**File:** `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/packet_generation.c:178-182`

**Change:**
```c
// BEFORE (lines 178-182)
ret = pm4_append_write_sh_reg(buffer,
                              arch->control_regs.compute_perfcount_enable -
                                  arch->control_regs.persistent_space_start,  // ← Remove this subtraction
                              0x1, /* enable */
                              0, 0);

// AFTER
ret = pm4_append_write_sh_reg(buffer,
                              arch->control_regs.compute_perfcount_enable,  // ← Pass absolute address
                              0x1, /* enable */
                              0, 0);
```

**Rationale:** The function `pm4_append_write_sh_reg` already handles the subtraction of `PERSISTENT_SPACE_START` internally, so the caller should pass the absolute register address.

---

### Fix 2: Add GRBM Broadcast Before Counter Configuration

**File:** `/home/ben/rocm-systems/perf-pmu-stub/src/aql_c/packet_generation.c:154-170`

**Change:**
```c
// BEFORE (lines 154-170)
if (has_sq_counters) {
    uint32_t sq_ctrl2_value =
        (1U << 0) | (0xFFFFU << 1); /* force_en | vmid_en */
    ret = pm4_append_set_uconfig_reg(
        buffer,
        arch->control_regs.sq_perfcounter_ctrl2,
        sq_ctrl2_value);
    if (ret < 0)
      return ret;
}

/* 5. Configure each counter */
for (size_t i = 0; i < collection->counter_count; i++) {
    ret = generate_counter_config(buffer, arch, &collection->counters[i]);
    if (ret < 0)
      return ret;
}

// AFTER
if (has_sq_counters) {
    uint32_t sq_ctrl2_value =
        (1U << 0) | (0xFFFFU << 1); /* force_en | vmid_en */
    ret = pm4_append_set_uconfig_reg(
        buffer,
        arch->control_regs.sq_perfcounter_ctrl2,
        sq_ctrl2_value);
    if (ret < 0)
      return ret;

    /* GRBM broadcast before counter config */       // ← ADD THIS
    ret = generate_grbm_broadcast(buffer, arch);     // ← ADD THIS
    if (ret < 0)                                     // ← ADD THIS
      return ret;                                    // ← ADD THIS
}

/* 5. Configure each counter */
for (size_t i = 0; i < collection->counter_count; i++) {
    ret = generate_counter_config(buffer, arch, &collection->counters[i]);
    if (ret < 0)
      return ret;
}
```

**Rationale:** The original implementation broadcasts to all GPU instances before configuring counters. This ensures the counter configuration is applied uniformly across the GPU topology.

---

## Functional Significance

### Critical Issues
1. **SET_SH_REG wrong offset (0xd60b vs 0x20b)**: Will write to the wrong register, potentially breaking compute performance counting entirely.

### Important Issues
2. **Missing GRBM broadcast before counter config**: May cause counter configuration to only apply to GPU instance 0, resulting in incorrect or incomplete performance data from other SEs/SAs/WGPs.

### Cosmetic Issues
None - both differences are functionally significant.

## Testing Recommendations

After applying fixes:
1. Regenerate packets with `packet_gen_tool`
2. Compare with `pm4_decoder` to verify:
   - SET_SH_REG shows `0x20b` (not `0xd60b`)
   - GRBM broadcast (0x200) appears before counter config (0x19c0, 0x19e0)
   - Total packet count matches original (12 packets)
3. Verify absolute addresses in decoder output match original
