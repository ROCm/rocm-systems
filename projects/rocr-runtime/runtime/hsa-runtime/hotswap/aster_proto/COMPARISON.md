# waveasm vs Aster: Prototype Comparison

Same input: `vecadd` kernel compiled for gfx942, 1216-byte `.text` section.

## Pipeline Results

| Metric                 | waveasm                    | Aster (amdgcn dialect)      |
|------------------------|----------------------------|-----------------------------|
| Instructions lifted    | 292 / 292 (100%)           | 23 / 23 (100%)             |
| Unsupported mnemonics  | 0 (all via `waveasm.raw`)  | 0                           |
| HSACO generated        | Yes                        | Yes                         |
| GPU execution          | PASS (1024 elements)       | PASS (1024 elements)        |
| Cross-ISA (gfx1250→942)| PASS                       | Not attempted               |
| Fallback mechanism     | `waveasm.raw` (opaque)     | None needed                 |

## IR Quality

### waveasm

- Every instruction maps to a `waveasm` op, with unknown instructions as `waveasm.raw`
- SSA construction is linear (no CFG awareness)
- Physical registers baked into types (`!waveasm.pvreg<v0>`)
- No structured wait counter modeling
- Assembly emission is ad-hoc (mnemonic prefix matching)

### Aster

- Instruction-class ops (`amdgcn.vop2`, `amdgcn.load`, `amdgcn.store`, `amdgcn.cmpi`)
- True SSA with `amdgcn.alloca` + `amdgcn.make_register_range`
- Typed register system (`!amdgcn.sgpr<3>`, `!amdgcn.vgpr<[0 : 2]>`, `!amdgcn.vcc<0>`)
- Structured wait counters (`amdgcn.sopp.s_waitcnt` with named fields)
- Read/write tokens for memory ordering (`!amdgcn.read_token<constant>`)
- Multi-block CFG with `amdgcn.cbranch` / `amdgcn.branch` for control flow
- Assembly emission via `translateModule` with full kernel metadata generation

## Control Flow Lifting

The original binary uses exec-mask manipulation for `if (i < N)` bounds checking:
```
v_cmp_gt_i32_e32 vcc, N, threadId    ; per-lane compare → VCC
s_and_saveexec_b64 s[2:3], vcc       ; save EXEC, EXEC &= VCC
s_cbranch_execz <skip>               ; skip body if EXEC == 0
```

The Aster lifter converts this to VCC-based branching:
```
cmpi v_cmp_gt_i32 %vcc, N, threadId  ; CmpIOp → VCC
cbranch s_cbranch_vccz %vcc ^exit fallthrough(^body)  ; CBranchOp
```

This is semantically correct when the grid dispatch is aligned to wavefront size
(all threads have valid indices). The exec-mask pattern is recognized and converted
to Aster's native multi-block control flow.

## Key Differences

### waveasm advantage: Coverage breadth
waveasm handles arbitrary instructions because `waveasm.raw` acts as an escape
hatch. This means any kernel can round-trip, even if some instructions are opaque
to analysis passes.

### Aster advantage: Soundness
Every op in the IR has well-defined semantics that passes can analyze and
transform. No opaque fallbacks. The exec-mask pattern is converted to structured
control flow rather than being preserved as opaque bytes.

### Aster advantage: Metadata
Aster's `translateModule` generates complete `.amdhsa_kernel` metadata
(register counts, float modes, SGPRs) from the IR. The waveasm approach requires
manually splicing this metadata from a template.

### Aster advantage: Register ranges
First-class support for register ranges (`!amdgcn.sgpr<[0 : 2]>`) and
`make_register_range`, correctly modeling 64-bit addresses and multi-dword loads.

### Aster advantage: Control flow structure
Multi-block CFG with `cbranch`/`branch` ops enables future analysis and
optimization passes. The waveasm approach keeps everything in a single flat
basic block.

## Known Limitations

1. **Exec-mask predication**: The VCC-based branch conversion assumes the grid
   dispatch is aligned to wavefront size. For unaligned dispatches, per-lane
   exec-mask predication would be needed but is not modeled.

2. **Hidden kernel args**: The pipeline must inject HIP implicit argument
   metadata (hidden_group_size_x etc.) into the assembly because Aster's
   kernel descriptor only covers explicit arguments.

3. **Assembly post-processing**: The `.amdhsa_user_sgpr_private_segment_buffer`
   directive must be removed for GFX942 with architected flat scratch.

## Recommendation

For the hotswap binary translation use case:

1. **Aster is the better foundation** for principled binary translation. Its typed
   register system, structured control flow, and metadata generation provide the
   analysis infrastructure needed for cross-ISA translation.

2. **waveasm remains useful** for quick-and-dirty round-tripping of arbitrary
   kernels where coverage is more important than IR analysis capability.

3. **Next steps**: Extend the Aster lifter to handle more instruction families
   and kernel patterns. Add cross-ISA support (gfx1250 → gfx942).
