# How the ROCr kernel trampoline is reached (no branch instruction involved)

## Context

MI4xx (gfx1250/gfx1251/gfx12-5-generic) requires two workaround instructions
(`GLOBAL_WB` / `V_NOP`, per DEGFXMI400-13246 and DEGFXMI400-13174) to run at
the start of every kernel to avoid shader hangs. The current BKC workaround
is a ROCr code object loader "trampoline": the loader appends a small stub
per kernel that runs the two quirk instructions and then jumps into the
kernel's real code.

Question investigated: when disassembling a code object snapshot, the
trampoline shows up in its own unnamed ELF section, but there is no visible
branch instruction *from* the kernel's symbol *to* the trampoline. How is the
trampoline actually reached?

## Reproduction

```
rocprofv3 --pc-sampling-beta-enabled --pc-sampling-method stochastic \
  --pc-sampling-unit cycles --pc-sampling-interval 1048576 \
  --output-format csv json -- bin/transpose 2 100
```

This dumps per-code-object snapshots, e.g.
`build/heliosr-2b805-d7-3/125720_gfx1250_code_object_id_2.out`, which contains
the `_Z9transposePKiPiii` kernel. Disassembled with `disassemble.sh --raw -t gfx1250`.

## Answer: the redirection is a kernel-descriptor field rewrite, not a branch

The GPU command processor does not "branch" to a kernel's code at dispatch
time. It reads the wavefront's initial PC out of the kernel descriptor
(`.kd`, a 64-byte struct) at a fixed field: `kernel_code_entry_byte_offset`
(an 8-byte signed value at `.kd + 0x10`), computed as:

```
entry_pc = address_of_kd + kernel_code_entry_byte_offset
```

The ROCr trampoline loader patches this one field so it points at the
appended trampoline instead of the kernel's real start address. No
instruction bytes in `.text` are touched, and nothing branches into the
trampoline — dispatch just starts execution there directly. That's why
grepping the disassembly for a branch into the trampoline finds nothing.

### Verified evidence

`_Z9transposePKiPiii.kd` lives at `0x880`. Raw bytes:

```
0880: 00100000 00000000 18010000 00000000
0890: 80370000 00000000 00000000 00000000
```

Field layout (kernel descriptor v3/v4/v5):

| offset | field | value |
|---|---|---|
| `+0x00` | `group_segment_fixed_size` | `0x1000` |
| `+0x08` | `kernarg_size` | `0x118` |
| `+0x10` | `kernel_code_entry_byte_offset` | `0x3780` |

```
entry_pc = 0x880 + 0x3780 = 0x4000
```

`0x4000` is exactly the address of the unnamed trampoline section — not
`0x1900`, where the `_Z9transposePKiPiii` symbol/code actually lives. Normally
(no trampoline) this field would be `0x1080`, giving `0x880 + 0x1080 = 0x1900`.
The loader only had to change this one field to redirect the wavefront.

The unnamed section itself (`objdump` prints `Disassembly of section :`) is a
loader-appended PROGBITS/AX section with an empty `sh_name` — section index
17 in this ELF, 4096-byte aligned, sized `0x100` (one 256-byte trampoline
slot per kernel in the code object).

### The trampoline's jump back into the kernel is a computed jump, also not a symbolic branch

```
0000000000004000 <>:
    global_wb                                    // quirk #1 (VMEM-clause workaround)
    v_nop                                        // quirk #2 (PERM_PK16 workaround)
    s_get_pc_i64 s[14:15]        // 000000004010; PC of *next* insn = 0x4014
    s_add_co_u32  s14, s14, 0xffffd8ec            // 0x4014 + (-0x2714)
    s_add_co_ci_u32 s15, s15, -1
    s_set_pc_i64 s[14:15]                         // jump to computed target
```

```
0x4014 + 0xffffd8ec (sign-extended, i.e. -0x2714) = 0x1900
```

`0x1900` is exactly the transpose kernel's real entry point. The trampoline
computes its own PC (`s_get_pc_i64`), adds a fixed baked-in displacement,
and jumps via `s_set_pc_i64` — an indirect jump to a computed address, not a
fixed-immediate `s_branch`/`s_call`. `llvm-objdump` has no operand to
annotate with a `<symbol+offset>` cross-reference in this case, which is the
other reason static disassembly shows no visible edge between the trampoline
and the kernel.

## Summary

Both hops in "dispatch → trampoline → real kernel" are invisible to a naive
disassembly scan for branches:

1. **Dispatch → trampoline**: redirected by rewriting `kernel_code_entry_byte_offset`
   in the kernel descriptor (metadata read by hardware at dispatch, not an
   instruction).
2. **Trampoline → real kernel**: a computed indirect jump (`s_get_pc_i64` +
   fixed immediate arithmetic + `s_set_pc_i64`), not a fixed-target branch.

This is consistent with the ROCr trampoline PR
([rocm-systems#7342](https://github.com/ROCm/rocm-systems/pull/7342)) design
described in the ticket: patch each kernel's start address in its code
object metadata rather than mutating kernel code, specifically so the real
kernel entry point never needs to be passed through a USER_DATA register.

## How to reproduce / find the `.kd` yourself

File used above: `build/heliosr-2b805-d7-3/125720_gfx1250_code_object_id_2.out`
(one of the per-code-object snapshots dumped by `rocprofv3` for the
`transpose` app run in the Reproduction section).

**1. Find the `.kd` symbol and its address:**

```
readelf -sW <code_object.out> | grep -E "FUNC|\.kd"
```

Example output:
```
     1: 0000000000001900   288 FUNC    GLOBAL PROTECTED    8 _Z9transposePKiPiii
     2: 0000000000000880    64 OBJECT  GLOBAL PROTECTED    6 _Z9transposePKiPiii.kd
```
`FUNC` = the kernel's actual code address. `OBJECT ... .kd` = the 64-byte
kernel descriptor's address — note it entirely from the code, not the
disassembly.

**2. Hex-dump the kernel descriptor's raw bytes** (it lives in `.rodata`,
so it never shows up in a disassembly listing):

```
readelf -x .rodata <code_object.out>
```

Example output (for `.kd` at `0x880`):
```
0x00000880 00100000 00000000 18010000 00000000 ................
0x00000890 80370000 00000000 00000000 00000000 .7..............
```
Field at `.kd + 0x10` (`80 37 00 00 00 00 00 00`, little-endian) is
`kernel_code_entry_byte_offset = 0x3780`. Compute
`entry_pc = kd_address + kernel_code_entry_byte_offset` to get the true
dispatch entry point (`0x880 + 0x3780 = 0x4000` here — the trampoline, not
the kernel's own code at `0x1900`).

**3. Disassemble the code object to see both the kernel code and the
trampoline:**

```
./disassemble.sh <code_object.out> --raw -t gfx1250 -o dump.s
```

- Kernel body will be labeled `<kernel_name>` (e.g. `<_Z9transposePKiPiii>`
  at `0x1900`).
- The trampoline will appear as an unnamed section (`Disassembly of section
  :`) at whatever address `entry_pc` computed to (e.g. `0x4000`), containing
  `global_wb` / `v_nop` followed by `s_get_pc_i64` / `s_add_co_u32` /
  `s_add_co_ci_u32` / `s_set_pc_i64` that jumps back into the real kernel.

To confirm the trampoline's own back-jump target, take its `s_get_pc_i64`
result (address of the *next* instruction after `s_get_pc_i64`) and add the
signed immediate from the `s_add_co_u32`/`s_add_co_ci_u32` pair — the result
should equal the kernel's `FUNC` address from step 1.
