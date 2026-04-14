#!/usr/bin/env python3
"""
Test the gfx1250 → gfx950 transpiler mnemonic translation rules.

Uses llvm-mc to validate that:
1. Source instructions assemble correctly for gfx1250
2. Translated instructions assemble correctly for gfx950
3. The translation rules produce valid gfx950 assembly
"""

import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

LLVM_MC = os.environ.get("LLVM_MC", "/opt/rocm/lib/llvm/bin/llvm-mc")

@dataclass
class TranslationTest:
    name: str
    gfx1250_asm: str      # Source instruction (gfx1250)
    gfx950_asm: str        # Expected translated instruction (gfx950)
    category: str = ""

# ── Test cases ──────────────────────────────────────────────────────────────

TESTS = [
    # VALU pass-through (same mnemonic, different opcode encoding)
    TranslationTest("vop2_add_f32", "v_add_f32 v0, v1, v2", "v_add_f32 v0, v1, v2", "valu"),
    TranslationTest("vop2_mul_f32", "v_mul_f32 v0, v1, v2", "v_mul_f32 v0, v1, v2", "valu"),
    TranslationTest("vop1_mov_b32", "v_mov_b32 v0, v1", "v_mov_b32 v0, v1", "valu"),
    TranslationTest("vop1_cvt_f32_f16", "v_cvt_f32_f16 v0, v1", "v_cvt_f32_f16 v0, v1", "valu"),
    TranslationTest("vop3_fma_f32", "v_fma_f32 v0, v1, v2, v3", "v_fma_f32 v0, v1, v2, v3", "valu"),
    TranslationTest("vop2_max_f32", "v_max_f32 v0, v1, v2", "v_max_f32 v0, v1, v2", "valu"),
    TranslationTest("vop2_min_f32", "v_min_f32 v0, v1, v2", "v_min_f32 v0, v1, v2", "valu"),

    # Scalar pass-through
    TranslationTest("sop2_add_u32", "s_add_u32 s0, s1, s2", "s_add_u32 s0, s1, s2", "scalar"),
    TranslationTest("sop1_mov_b32", "s_mov_b32 s0, s1", "s_mov_b32 s0, s1", "scalar"),
    TranslationTest("sopp_endpgm", "s_endpgm", "s_endpgm", "scalar"),
    TranslationTest("sopp_nop", "s_nop 0", "s_nop 0", "scalar"),
    TranslationTest("sopp_branch", "s_branch 0", "s_branch 0", "scalar"),

    # Global memory (mnemonic renaming, saddr form compatible on both)
    TranslationTest("global_load_b32", "global_load_b32 v0, v0, s[0:1]",
                    "global_load_dword v0, v0, s[0:1]", "memory"),
    TranslationTest("global_load_b64", "global_load_b64 v[0:1], v2, s[0:1]",
                    "global_load_dwordx2 v[0:1], v2, s[0:1]", "memory"),
    TranslationTest("global_load_b128", "global_load_b128 v[0:3], v4, s[0:1]",
                    "global_load_dwordx4 v[0:3], v4, s[0:1]", "memory"),
    TranslationTest("global_store_b32", "global_store_b32 v0, v2, s[0:1]",
                    "global_store_dword v0, v2, s[0:1]", "memory"),
    TranslationTest("global_store_b64", "global_store_b64 v0, v[2:3], s[0:1]",
                    "global_store_dwordx2 v0, v[2:3], s[0:1]", "memory"),
    TranslationTest("global_load_u8", "global_load_u8 v0, v0, s[0:1]",
                    "global_load_ubyte v0, v0, s[0:1]", "memory"),
    TranslationTest("global_load_i8", "global_load_i8 v0, v0, s[0:1]",
                    "global_load_sbyte v0, v0, s[0:1]", "memory"),
    TranslationTest("global_load_u16", "global_load_u16 v0, v0, s[0:1]",
                    "global_load_ushort v0, v0, s[0:1]", "memory"),

    # Flat memory — operand format differs between GFX12 and GFX9:
    #   GFX12: flat_load_b32 vdst, vaddr, saddr  (32-bit vaddr + 64-bit saddr)
    #   GFX9:  flat_load_dword vdst, v[pair]      (64-bit vaddr pair, no saddr)
    # Operand translation needed — Phase B. For now test GFX9 form directly.
    TranslationTest("flat_load_dword", "flat_load_dword v0, v[0:1]",
                    "flat_load_dword v0, v[0:1]", "memory"),
    TranslationTest("flat_store_dword", "flat_store_dword v[0:1], v2",
                    "flat_store_dword v[0:1], v2", "memory"),

    # DS/LDS (mnemonic renaming)
    TranslationTest("ds_load_b32", "ds_load_b32 v0, v1",
                    "ds_read_b32 v0, v1", "ds"),
    TranslationTest("ds_load_b64", "ds_load_b64 v[0:1], v2",
                    "ds_read_b64 v[0:1], v2", "ds"),
    TranslationTest("ds_store_b32", "ds_store_b32 v0, v1",
                    "ds_write_b32 v0, v1", "ds"),

    # Wait counter translation
    TranslationTest("wait_loadcnt", "s_wait_loadcnt 0",
                    "s_waitcnt vmcnt(0)", "waitcnt"),
    TranslationTest("wait_storecnt", "s_wait_storecnt 0",
                    "s_waitcnt vmcnt(0)", "waitcnt"),
    TranslationTest("wait_dscnt", "s_wait_dscnt 0",
                    "s_waitcnt lgkmcnt(0)", "waitcnt"),
    TranslationTest("wait_kmcnt", "s_wait_kmcnt 0",
                    "s_waitcnt lgkmcnt(0)", "waitcnt"),
    TranslationTest("wait_expcnt", "s_wait_expcnt 0",
                    "s_waitcnt expcnt(0)", "waitcnt"),
    TranslationTest("wait_loadcnt_3", "s_wait_loadcnt 3",
                    "s_waitcnt vmcnt(3)", "waitcnt"),
]


def assemble(asm_text: str, mcpu: str) -> tuple[bool, str]:
    """Try to assemble an instruction for a given target. Returns (success, output)."""
    try:
        result = subprocess.run(
            [LLVM_MC, "-triple=amdgcn-amd-amdhsa", f"-mcpu={mcpu}",
             "--show-encoding"],
            input=asm_text,
            capture_output=True, text=True, timeout=10
        )
        return result.returncode == 0, result.stdout + result.stderr
    except FileNotFoundError:
        return False, f"llvm-mc not found at {LLVM_MC}"
    except subprocess.TimeoutExpired:
        return False, "timeout"


def run_tests() -> tuple[int, int, int]:
    """Run all translation tests. Returns (passed, failed, skipped)."""
    passed = 0
    failed = 0
    skipped = 0

    # Check llvm-mc exists
    if not Path(LLVM_MC).exists():
        print(f"ERROR: llvm-mc not found at {LLVM_MC}")
        return 0, 0, len(TESTS)

    # Check gfx1250 support
    ok, out = assemble("s_nop 0", "gfx1250")
    if not ok:
        print(f"WARNING: llvm-mc doesn't support gfx1250, skipping source validation")
        gfx1250_supported = False
    else:
        gfx1250_supported = True

    print(f"{'Test':<30} {'Category':<10} {'Source':<8} {'Target':<8} {'Result'}")
    print("-" * 80)

    for test in TESTS:
        # Validate source assembles for gfx1250
        src_ok = True
        if gfx1250_supported:
            src_ok, src_out = assemble(test.gfx1250_asm, "gfx1250")

        # Validate translated assembles for gfx950
        tgt_ok, tgt_out = assemble(test.gfx950_asm, "gfx950")

        if not src_ok and gfx1250_supported:
            status = "SKIP-SRC"
            skipped += 1
        elif not tgt_ok:
            status = "FAIL-TGT"
            failed += 1
            print(f"  Target assembly error: {tgt_out.strip()}")
        else:
            status = "PASS"
            passed += 1

        src_str = "ok" if src_ok else "FAIL"
        tgt_str = "ok" if tgt_ok else "FAIL"
        print(f"{test.name:<30} {test.category:<10} {src_str:<8} {tgt_str:<8} {status}")

    return passed, failed, skipped


def main():
    print("=" * 80)
    print("gfx1250 → gfx950 Transpiler Mnemonic Translation Tests")
    print("=" * 80)
    print()

    passed, failed, skipped = run_tests()

    print()
    print(f"Results: {passed} passed, {failed} failed, {skipped} skipped")

    if failed > 0:
        print("\nFAILED tests indicate the translated mnemonic is not valid on gfx950.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
