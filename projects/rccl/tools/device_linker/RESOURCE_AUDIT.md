# Resource computation and consistency audit

This document is a checklist to verify that scratch/private, LDS/shared, SGPR, and VGPR are (1) read correctly from each input, (2) reduced to a correct max, (3) written consistently to .KDs, ABS symbols, and .note in the final image, and (4) encoded correctly in the output bytes.

## 1. Input values (per object)

### 1.1 Where we read

- **Source:** `parseKernel()` in `device_linker.cpp` (around 1239–1310).
- **Per-kernel .note:** We read one `.note` per **input** ELF (each specialized kernel .o has one kernel and one .note).
- **findInt():** For each key we look for the key string in the note, then read the **next byte(s)** as msgpack:
  - `b <= 0x7f` → fixint, value = b
  - `0xcc` → uint8, value = next byte
  - `0xcd` → uint16, value = next 2 bytes (big-endian)
  - No support for uint32/uint64 or other encodings; those return 0.

**Keys read:**

| Key | Stored in | Meaning |
|-----|-----------|--------|
| `.vgpr_count` | `info.vgpr` | VGPR count |
| `.sgpr_count` | `info.sgpr` | SGPR count |
| `.group_segment_fixed_size` | `info.lds` | LDS size (bytes) |
| `.private_segment_fixed_size` | `info.stack` | Scratch/private size (bytes) |

**Things to verify:**

- [ ] **Msgpack encoding:** Are compiler-generated notes ever using uint32/uint64 for these fields? If so, findInt returns 0 and we under-report.
- [ ] **First occurrence:** `view.find(key)` returns the **first** occurrence. If the note has multiple kernels (see below), we only read the **first** kernel’s values per object. For single-kernel .o files this is fine; for the dispatcher we never use this path for the max (see 2.1).
- [ ] **Dispatcher not in reduction:** The **dispatcher** ELF is **never** passed to `parseKernel()` and **never** added to `kernels_`. So the max is taken only over **specialized** kernel inputs. The dispatcher has 18 kernels (3 Generic + 3 Debug + 12 oneRank) with their own .note metadata; those values are **not** included in `max_vgpr_`, `max_sgpr_`, `max_lds_`, or `max_stack_`. If any dispatcher kernel needs more resources than any single specialized kernel, the max will be **too low**.

## 2. Max reduction

### 2.1 Who is included

- **Included:** Every entry in `kernels_`, i.e. one per **specialized** input .o file (each parsed via `parseKernel(inputs[i])`).
- **Not included:** The dispatcher’s 18 kernels. Their resource requirements are not folded into the max.

**Code:** `collectSections()` (around 1942–1959):

```cpp
for (const auto& k : kernels_) {
    max_vgpr_ = std::max(max_vgpr_, k.vgpr);
    max_sgpr_ = std::max(max_sgpr_, k.sgpr);
    max_lds_ = std::max(max_lds_, k.lds);
    max_stack_ = std::max(max_stack_, k.stack);
}
int requiredLDS = calculateRequiredLDS();
if (requiredLDS > max_lds_) { max_lds_ = requiredLDS; }
```

**Things to verify:**

- [ ] **Include dispatcher in max:** Should we parse the dispatcher’s .note (e.g. once per kernel or once with a “worst case” per-kernel read) and merge those values into the max? Otherwise Generic/Debug/oneRank resource needs are ignored.
- [ ] **requiredLDS:** `calculateRequiredLDS()` (around 1905) is the RCCL host-side LDS formula. We set `max_lds_ = max(max_lds_, requiredLDS)`. Confirm the formula matches host and that it is the right ceiling for all launch configurations.

### 2.2 Semantics: count vs “next free”

- **Note:** `.vgpr_count` / `.sgpr_count` in the note may mean “number used” or “first free index” (next_free). The AMDGPU KD uses **granulated** values derived from a “next free” style value.
- **Our formula:** We treat the note value as a linear count and compute:
  - `vgpr_g = (max_vgpr_ + 3) / 4 - 1`
  - `sgpr_g = (max_sgpr_ + 7) / 8 - 1`
- If the note stores “first free index” (e.g. 48 = 48 VGPRs used), the same number can be used in the granulated formula. If the note stores something else (e.g. total allocation in a different unit), the encoding could be wrong. Confirm ABI and note semantics.

## 3. Consistency: .KDs, ABS symbols, .note

### 3.1 Kernel descriptors (.KD)

- **Patched:** First 6 KDs (Generic + Debug at offsets 0, 64, 128, 192, 256, 320); also each specialized KD in `specialized_kd_offsets_`.
- **Fields written:** `patchKD()` (around 3024):
  - Offset 0x00: `max_lds_` (4 bytes)
  - Offset 0x04: `max_stack_` (4 bytes)
  - Offset 0x30 (RSRC1): `vgpr_g`, `sgpr_g` with `vgpr_g = (max_vgpr_+3)/4 - 1`, `sgpr_g = (max_sgpr_+7)/8 - 1`, then `(vgpr_g & 0x3F) | ((sgpr_g & 0xF) << 6)` (and other bits preserved).
- **Truncation:** VGPR field is 6 bits (0x3F). Max representable is 63 granulated → 256 VGPRs. If `max_vgpr_` > 255 we under-allocate (e.g. 280 → 69 → 69 & 0x3F = 5 → 24 VGPRs).

### 3.2 ABS symbols

- **Patched:** In .symtab, ABS symbols (st_shndx == SHN_ABS) with suffixes:
  - `.num_vgpr` → st_value = `max_vgpr_`
  - `.numbered_sgpr` → st_value = `max_sgpr_`
- **Not patched:** `.private_seg_size` is intentionally left at 0 (comment: “must match KD (0)” — but we actually patch the KD with `max_stack_`, so this may be inconsistent if max_stack_ != 0).
- **Not patched:** No ABS symbol for LDS/group_segment is patched here. If such symbols exist, they would be inconsistent with the KD.

### 3.3 .note

- **Patched in `patchNote()`:** (around 2558–2684)
  - `private_segment_fixed_size` → `max_stack_`
  - `.vgpr_count` → `max_vgpr_`
  - `.sgpr_count` → `max_sgpr_`
- **Not patched:** `group_segment_fixed_size` in the .note is **never** updated. The KD has `max_lds_` (e.g. 32832); the note keeps the dispatcher’s per-kernel value (e.g. 4976). This is a known mismatch and a candidate for loader rejection.

**Checklist:**

- [ ] Max values used for reduction are correct (and include dispatcher if required).
- [ ] All .KDs that are launched use the same max values (already so for the 6 we patch; oneRank KDs are not patched for LDS/stack/RSRC1).
- [ ] ABS symbols: .num_vgpr and .numbered_sgpr match max; decide whether .private_seg_size should match `max_stack_` when non-zero.
- [ ] .note: add patching of `group_segment_fixed_size` to `max_lds_` (with msgpack size/expansion handled) so note and KD agree.

## 4. Bits written to the image

### 4.1 RSRC1 (KD offset 0x30)

- **Formula:** `rsrc1 = (rsrc1 & ~0x3FF) | (vgpr_g & 0x3F) | ((sgpr_g & 0xF) << 6)`.
- **VGPR:** 6 bits, granulated. Valid range 0–63 → 4–256 VGPRs (granulated value g → (g+1)*4).
- **SGPR:** 4 bits, granulated. Valid range 0–15 → 8–128 SGPRs (granulated value g → (g+1)*8).
- If `max_sgpr_` > 128, we truncate (e.g. 112 → sgpr_g = 12 → 104 SGPRs; 106 → sgpr_g = 12 → 104, so we’d under-allocate for 106).

### 4.2 LDS / stack in KD

- **Offset 0x00:** uint32 LDS (bytes). We write `max_lds_` as 4 bytes little-endian.
- **Offset 0x04:** uint32 stack (bytes). We write `max_stack_` as 4 bytes little-endian.

### 4.3 .note msgpack

- **expandIntField:** Can expand fixint → uint8 (0xcc) or uint16 (0xcd). Values > 65535 are not fully supported for expansion.
- **private_segment_fixed_size:** We search for 0xbb + ".private_segment_fixed_size" and patch the following value byte(s); we can expand to 0xcd + 2 bytes if needed (max 65535).

## 5. Summary of actions

1. **Inputs:** Confirm findInt handles all msgpack encodings used by the compiler for the four keys; confirm whether the dispatcher’s kernels must be included in the max.
2. **Max:** Include dispatcher kernel resource values in the reduction (e.g. parse dispatcher .note per kernel or extract per-kernel metadata and merge).
3. **.note:** Patch `group_segment_fixed_size` in the .note to `max_lds_` so it matches the KD.
4. **RSRC1:** Resolve truncation when max_vgpr_ > 255 (and max_sgpr_ > 128 if applicable): either cap at ABI max, or ensure the note and code never exceed what we can represent.
5. **ABS:** Revisit .private_seg_size and any LDS-related ABS symbols so they match the chosen max values.

This audit should be done carefully; misinterpretation of “count” vs “next free” or of which objects contribute to the max can explain wrong values and loader failures.
