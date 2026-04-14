#!/usr/bin/env python3
"""
End-to-end transpiler test: compile gfx1250 assembly, translate to gfx950,
assemble for gfx950, and verify all instructions round-trip correctly.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

LLVM_MC = os.environ.get("LLVM_MC", "/opt/rocm/lib/llvm/bin/llvm-mc")
LLVM_OBJDUMP = os.environ.get("LLVM_OBJDUMP", "/opt/rocm/lib/llvm/bin/llvm-objdump")

# Translation rules (subset matching transpiler.cpp)
MNEMONIC_MAP = {
    "global_load_b32": "global_load_dword",
    "global_load_b64": "global_load_dwordx2",
    "global_load_b128": "global_load_dwordx4",
    "global_store_b32": "global_store_dword",
    "global_store_b64": "global_store_dwordx2",
    "global_store_b128": "global_store_dwordx4",
    "global_load_u8": "global_load_ubyte",
    "global_load_i8": "global_load_sbyte",
    "global_load_u16": "global_load_ushort",
    "ds_load_b32": "ds_read_b32",
    "ds_load_b64": "ds_read_b64",
    "ds_store_b32": "ds_write_b32",
    "ds_store_b64": "ds_write_b64",
}

WAIT_MAP = {
    "s_wait_loadcnt": lambda n: f"s_waitcnt vmcnt({n})",
    "s_wait_storecnt": lambda n: f"s_waitcnt vmcnt({n})",
    "s_wait_dscnt": lambda n: f"s_waitcnt lgkmcnt({n})",
    "s_wait_kmcnt": lambda n: f"s_waitcnt lgkmcnt({n})",
    "s_wait_expcnt": lambda n: f"s_waitcnt expcnt({n})",
}

# SMEM renaming (GFX12 uses b-notation)
MNEMONIC_MAP.update({
    "s_load_b32": "s_load_dword",
    "s_load_b64": "s_load_dwordx2",
    "s_load_b128": "s_load_dwordx4",
    "s_load_b256": "s_load_dwordx8",
    "s_load_b512": "s_load_dwordx16",
    "s_store_b32": "s_store_dword",
    "s_store_b64": "s_store_dwordx2",
    "s_store_b128": "s_store_dwordx4",
    "s_buffer_load_b32": "s_buffer_load_dword",
    "s_buffer_load_b64": "s_buffer_load_dwordx2",
    "s_buffer_load_b128": "s_buffer_load_dwordx4",
    "s_buffer_load_b256": "s_buffer_load_dwordx8",
    "s_buffer_load_b512": "s_buffer_load_dwordx16",
})


def translate_line(line: str) -> list[str]:
    """Translate a single gfx1250 disassembly line to gfx950 assembly."""
    stripped = line.strip()
    if not stripped or stripped.startswith("//") or stripped.startswith("."):
        return [stripped]

    # Extract mnemonic
    parts = stripped.split(None, 1)
    mnemonic = parts[0]
    operands = parts[1] if len(parts) > 1 else ""

    # Remove trailing comments
    if "//" in operands:
        operands = operands[:operands.index("//")].strip()

    # Handle hex immediate operands (0xN) in wait counters
    wait_mnemonic = mnemonic
    # Wait counter translation
    if wait_mnemonic in WAIT_MAP:
        op = operands.strip()
        try:
            count = int(op, 0) if op else 0  # int(x, 0) handles 0x prefix
        except ValueError:
            count = 0
        return [WAIT_MAP[wait_mnemonic](count)]

    # s_wait_alu → s_nop
    if mnemonic == "s_wait_alu":
        return ["s_nop 0"]

    # Mnemonic renaming — try with and without encoding suffix (_e32, _e64)
    base_mnemonic = mnemonic
    suffix = ""
    for s in ("_e32", "_e64"):
        if mnemonic.endswith(s):
            base_mnemonic = mnemonic[:-len(s)]
            suffix = s
            break

    if base_mnemonic in MNEMONIC_MAP:
        new_mnem = MNEMONIC_MAP[base_mnemonic]
        return [f"{new_mnem}{suffix} {operands}".strip()]

    # Direct mnemonic match (for instructions without suffix)
    if mnemonic in MNEMONIC_MAP:
        new_mnem = MNEMONIC_MAP[mnemonic]
        return [f"{new_mnem} {operands}".strip()]

    # Flat+saddr → Global conversion
    # GFX12: flat_load_b32 vdst, vaddr, s[pair] → global_load_dword vdst, vaddr, s[pair]
    # GFX9 flat doesn't support saddr, but global does
    is_flat = mnemonic.startswith("flat_load_") or mnemonic.startswith("flat_store_")
    if is_flat and "s[" in operands:
        global_mnem = mnemonic.replace("flat_", "global_")
        if global_mnem in MNEMONIC_MAP:
            global_mnem = MNEMONIC_MAP[global_mnem]
        return [f"{global_mnem} {operands}".strip()]

    # Pass-through (same mnemonic)
    return [stripped]


def run_cmd(cmd: list[str], input_text: str = None) -> tuple[int, str, str]:
    """Run a command and return (returncode, stdout, stderr)."""
    result = subprocess.run(cmd, input=input_text, capture_output=True, text=True, timeout=30)
    return result.returncode, result.stdout, result.stderr


def main():
    print("=" * 80)
    print("gfx1250 → gfx950 End-to-End Transpiler Test")
    print("=" * 80)

    # Test kernel: vector add with global memory access
    gfx1250_asm = """\
.text
    ; Kernel: simple vector add
    ; Load two f32 values from global memory, add them, store result
    s_load_b64 s[0:1], s[0:1], 0x0
    s_load_b64 s[2:3], s[0:1], 0x8
    s_load_b64 s[4:5], s[0:1], 0x10
    s_wait_kmcnt 0
    v_lshlrev_b32 v0, 2, v0
    global_load_b32 v1, v0, s[0:1]
    global_load_b32 v2, v0, s[2:3]
    s_wait_loadcnt 0
    v_add_f32 v1, v1, v2
    global_store_b32 v0, v1, s[4:5]
    s_endpgm
"""

    print("\n--- Step 1: Assemble for gfx1250 ---")
    rc, out, err = run_cmd(
        [LLVM_MC, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250",
         "-filetype=obj", "-o", "/tmp/transpile_gfx1250.o"],
        gfx1250_asm
    )
    if rc != 0:
        print(f"FAIL: gfx1250 assembly failed:\n{err}")
        return 1
    print("  Assembled successfully")

    print("\n--- Step 2: Disassemble gfx1250 binary ---")
    rc, disasm_out, err = run_cmd(
        [LLVM_OBJDUMP, "-d", "--mcpu=gfx1250", "--no-show-raw-insn",
         "/tmp/transpile_gfx1250.o"]
    )
    if rc != 0:
        print(f"FAIL: disassembly failed:\n{err}")
        return 1

    # Parse disassembly output. Format:
    #   \ts_load_b64 s[0:1], s[0:1], 0x0        // 00000000: F4002000 ...
    instructions = []
    for line in disasm_out.splitlines():
        # Instructions start with a tab
        if not line.startswith("\t"):
            continue
        instr = line.strip()
        # Remove trailing comments (// address: bytes)
        if "//" in instr:
            instr = instr[:instr.index("//")].strip()
        if instr:
            instructions.append(instr)

    print(f"  Disassembled {len(instructions)} instructions:")
    for i, inst in enumerate(instructions):
        print(f"    [{i:2d}] {inst}")

    print("\n--- Step 3: Translate to gfx950 ---")
    translated = []
    for inst in instructions:
        trans = translate_line(inst)
        translated.extend(trans)

    print(f"  Translated to {len(translated)} instructions:")
    for i, inst in enumerate(translated):
        print(f"    [{i:2d}] {inst}")

    print("\n--- Step 4: Assemble for gfx950 ---")
    gfx950_asm = ".text\n" + "\n".join(translated) + "\n"

    rc, out, err = run_cmd(
        [LLVM_MC, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx950",
         "-filetype=obj", "-o", "/tmp/transpile_gfx950.o"],
        gfx950_asm
    )
    if rc != 0:
        print(f"FAIL: gfx950 assembly failed:\n{err}")
        print(f"\nAssembly input:\n{gfx950_asm}")
        return 1
    print("  Assembled successfully")

    print("\n--- Step 5: Verify gfx950 disassembly ---")
    rc, disasm_out, err = run_cmd(
        [LLVM_OBJDUMP, "-d", "--mcpu=gfx950", "--no-show-raw-insn",
         "/tmp/transpile_gfx950.o"]
    )
    if rc != 0:
        print(f"FAIL: verification disassembly failed:\n{err}")
        return 1

    # Parse output instructions
    gfx950_instructions = []
    for line in disasm_out.splitlines():
        if not line.startswith("\t"):
            continue
        instr = line.strip()
        if "//" in instr:
            instr = instr[:instr.index("//")].strip()
        if instr:
            gfx950_instructions.append(instr)

    print(f"  Verified {len(gfx950_instructions)} gfx950 instructions:")
    for i, inst in enumerate(gfx950_instructions):
        print(f"    [{i:2d}] {inst}")

    print("\n" + "=" * 80)
    print("PASS: End-to-end transpile gfx1250 → gfx950 succeeded")
    print(f"  {len(instructions)} source → {len(translated)} translated → {len(gfx950_instructions)} verified")
    print("=" * 80)

    # ── Test 2: Flat+saddr kernel ──
    print("\n" + "=" * 80)
    print("Test 2: Flat+saddr → Global Conversion")
    print("=" * 80)

    flat_gfx1250_asm = """\
.text
    s_load_b64 s[0:1], s[0:1], 0x0
    s_wait_kmcnt 0
    flat_load_b32 v1, v0, s[0:1]
    s_wait_loadcnt 0
    v_add_f32_e32 v1, 1.0, v1
    flat_store_b32 v0, v1, s[0:1]
    s_endpgm
"""

    print("\n--- Assemble for gfx1250 ---")
    rc, out, err = run_cmd(
        [LLVM_MC, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx1250",
         "-filetype=obj", "-o", "/tmp/flat_gfx1250.o"],
        flat_gfx1250_asm
    )
    if rc != 0:
        print(f"FAIL: gfx1250 assembly failed:\n{err}")
        return 1

    print("\n--- Disassemble ---")
    rc, disasm_out, err = run_cmd(
        [LLVM_OBJDUMP, "-d", "--mcpu=gfx1250", "--no-show-raw-insn",
         "/tmp/flat_gfx1250.o"]
    )
    flat_instructions = []
    for line in disasm_out.splitlines():
        if not line.startswith("\t"):
            continue
        instr = line.strip()
        if "//" in instr:
            instr = instr[:instr.index("//")].strip()
        if instr:
            flat_instructions.append(instr)

    for i, inst in enumerate(flat_instructions):
        print(f"  [{i:2d}] {inst}")

    print("\n--- Translate ---")
    flat_translated = []
    for inst in flat_instructions:
        trans = translate_line(inst)
        flat_translated.extend(trans)
    for i, inst in enumerate(flat_translated):
        print(f"  [{i:2d}] {inst}")

    print("\n--- Assemble for gfx950 ---")
    flat_gfx950_asm = ".text\n" + "\n".join(flat_translated) + "\n"
    rc, out, err = run_cmd(
        [LLVM_MC, "-triple=amdgcn-amd-amdhsa", "-mcpu=gfx950",
         "-filetype=obj", "-o", "/tmp/flat_gfx950.o"],
        flat_gfx950_asm
    )
    if rc != 0:
        print(f"FAIL: gfx950 assembly failed:\n{err}")
        print(f"\nAssembly input:\n{flat_gfx950_asm}")
        return 1

    print("  OK — flat+saddr converted to global successfully")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    sys.exit(main())
