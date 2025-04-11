/*
 * Copyright (C) 2021 Advanced Micro Devices, Inc. All Rights Reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 */

#include "ShaderStore.hpp"

/**
 * KFDASMTest List
 */

const std::vector<const char*> ShaderList = {
    NoopIsa,
    CopyDwordIsa,
    InfiniteLoopIsa,
    AtomicIncIsa,
    ScratchCopyDwordIsa,
    PollMemoryIsa,
    CopyOnSignalIsa,
    PollAndCopyIsa,
    WriteFlagAndValueIsa,
    WriteAndSignalIsa,
    LoopIsa,
    PersistentIterateIsa,
    ReadMemoryIsa,
    GwsInitIsa,
    GwsAtomicIncreaseIsa,
    CheckCuMaskIsa
};

/**
 * Macros
 */

#define SHADER_START ".text\n"

/* Macros for portable v_add_co_u32, v_add_co_ci_u32,
 * and v_cmp_lt_u32.
 */
#define SHADER_MACROS_U32 \
    "   .text\n"\
    "   .macro V_ADD_CO_U32 vdst, src0, vsrc1\n"\
    "       .if (.amdgcn.gfx_generation_number >= 10)\n"\
    "           v_add_co_u32        \\vdst, vcc_lo, \\src0, \\vsrc1\n"\
    "       .elseif (.amdgcn.gfx_generation_number >= 9)\n"\
    "           v_add_co_u32        \\vdst, vcc, \\src0, \\vsrc1\n"\
    "       .else\n"\
    "           v_add_u32           \\vdst, vcc, \\src0, \\vsrc1\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro V_ADD_CO_CI_U32 vdst, src0, vsrc1\n"\
    "       .if (.amdgcn.gfx_generation_number >= 10)\n"\
    "           v_add_co_ci_u32     \\vdst, vcc_lo, \\src0, \\vsrc1, vcc_lo\n"\
    "       .elseif (.amdgcn.gfx_generation_number >= 9)\n"\
    "           v_addc_co_u32       \\vdst, vcc, \\src0, \\vsrc1, vcc\n"\
    "       .else\n"\
    "           v_addc_u32          \\vdst, vcc, \\src0, \\vsrc1, vcc\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro V_CMP_LT_U32 src0, vsrc1\n"\
    "       .if (.amdgcn.gfx_generation_number >= 10)\n"\
    "           v_cmp_lt_u32        vcc_lo, \\src0, \\vsrc1\n"\
    "       .else\n"\
    "           v_cmp_lt_u32        vcc, \\src0, \\vsrc1\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro V_CMP_EQ_U32 src0, vsrc1\n"\
    "       .if (.amdgcn.gfx_generation_number >= 10)\n"\
    "           v_cmp_eq_u32        vcc_lo, \\src0, \\vsrc1\n"\
    "       .else\n"\
    "           v_cmp_eq_u32        vcc, \\src0, \\vsrc1\n"\
    "       .endif\n"\
    "   .endm\n"

/* Macros for portable flat load/store/atomic instructions.
 *
 * gc943 (gfx94x) deprecates glc/slc in favour of nt/sc1/sc0.
 * The below macros when used will always use the nt sc1 sc0
 * modifiers for gfx94x, but also take in arg0 arg1 to specify
 * (for non-gfx94x): glc, slc, or glc slc.
 */
#define SHADER_MACROS_FLAT \
    "   .macro FLAT_LOAD_DWORD_NSS vdst, vaddr arg0 arg1\n"\
    "       .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)\n"\
    "           flat_load_dword \\vdst, \\vaddr nt sc1 sc0\n"\
    "       .else\n"\
    "           flat_load_dword \\vdst, \\vaddr \\arg0 \\arg1\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro FLAT_LOAD_DWORDX2_NSS vdst, vaddr arg0 arg1\n"\
    "       .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)\n"\
    "           flat_load_dwordx2 \\vdst, \\vaddr nt sc1 sc0\n"\
    "       .else\n"\
    "           flat_load_dwordx2 \\vdst, \\vaddr \\arg0 \\arg1\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro FLAT_STORE_DWORD_NSS vaddr, vsrc arg0 arg1\n"\
    "       .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)\n"\
    "           flat_store_dword \\vaddr, \\vsrc nt sc1 sc0\n"\
    "       .else\n"\
    "           flat_store_dword \\vaddr, \\vsrc \\arg0 \\arg1\n"\
    "       .endif\n"\
    "   .endm\n"\
    "   .macro FLAT_ATOMIC_ADD_NSS vdst, vaddr, vsrc arg0 arg1\n"\
    "       .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)\n"\
    "           flat_atomic_add \\vdst, \\vaddr, \\vsrc nt sc1 sc0\n"\
    "       .else\n"\
    "           flat_atomic_add \\vdst, \\vaddr, \\vsrc \\arg0 \\arg1\n"\
    "       .endif\n"\
    "   .endm\n"

/**
 * Common
 */

const char *NoopIsa =
    SHADER_START
    R"(
        s_endpgm
)";

const char *CopyDwordIsa =
    SHADER_START
    SHADER_MACROS_FLAT
    R"(
        v_mov_b32 v0, s0
        v_mov_b32 v1, s1
        v_mov_b32 v2, s2
        v_mov_b32 v3, s3
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORD_NSS v4, v[0:1] scope:SCOPE_SYS
            s_wait_loadcnt 0
            FLAT_STORE_DWORD_NSS v[2:3], v4 scope:SCOPE_SYS
        .else
            FLAT_LOAD_DWORD_NSS v4, v[0:1] glc slc
            s_waitcnt 0
            FLAT_STORE_DWORD_NSS v[2:3], v4 glc slc
        .endif
        s_endpgm
)";

const char *InfiniteLoopIsa =
    SHADER_START
    R"(
        .text
        LOOP:
        s_nop 0x10
        s_branch LOOP
        s_endpgm
)";

const char *AtomicIncIsa =
    SHADER_START
    SHADER_MACROS_FLAT
    R"(
        v_mov_b32 v0, s0
        v_mov_b32 v1, s1
        .if (.amdgcn.gfx_generation_number >= 12)
            v_mov_b32 v2, 1
            FLAT_ATOMIC_ADD_NSS v3, v[0:1], v2 scope:SCOPE_SYS th:TH_ATOMIC_RETURN
        .elseif (.amdgcn.gfx_generation_number >= 8)
            v_mov_b32 v2, 1
            FLAT_ATOMIC_ADD_NSS v3, v[0:1], v2 glc slc
        .else
            v_mov_b32 v2, -1
            flat_atomic_inc v3, v[0:1], v2 glc slc
        .endif
        s_endpgm
)";

/**
 * KFDMemoryTest
 */

const char *ScratchCopyDwordIsa =
    SHADER_START
    SHADER_MACROS_FLAT
    R"(
        // Copy the parameters from scalar registers to vector registers
        .if (.amdgcn.gfx_generation_number >= 9)
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            v_mov_b32 v2, s2
            v_mov_b32 v3, s3
        .else
            v_mov_b32_e32 v0, s0
            v_mov_b32_e32 v1, s1
            v_mov_b32_e32 v2, s2
            v_mov_b32_e32 v3, s3
        .endif

        // Setup the scratch parameters. This assumes a single 16-reg block
        .if (.amdgcn.gfx_generation_number >= 12)
            s_setreg_b32 hwreg(HW_REG_SCRATCH_BASE_LO), s4
            s_setreg_b32 hwreg(HW_REG_SCRATCH_BASE_HI), s5
        .elseif (.amdgcn.gfx_generation_number >= 10)
            s_setreg_b32 hwreg(HW_REG_FLAT_SCR_LO), s4
            s_setreg_b32 hwreg(HW_REG_FLAT_SCR_HI), s5
        .elseif (.amdgcn.gfx_generation_number == 9)
            s_mov_b32 flat_scratch_lo, s4
            s_mov_b32 flat_scratch_hi, s5
        .else
            s_mov_b32 flat_scratch_lo, 8
            s_mov_b32 flat_scratch_hi, 0
        .endif

        // Copy a dword between the passed addresses
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORD_NSS v4, v[0:1] scope:SCOPE_SYS
            s_wait_loadcnt 0
            FLAT_STORE_DWORD_NSS v[2:3], v4 scope:SCOPE_SYS
        .else
            FLAT_LOAD_DWORD_NSS v4, v[0:1] slc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            FLAT_STORE_DWORD_NSS v[2:3], v4 slc
        .endif

        s_endpgm
)";

/* Continuously poll src buffer and check buffer value
 * After src buffer is filled with specific value (0x5678,
 * by host program), fill dst buffer with specific
 * value(0x5678) and quit
 */
const char *PollMemoryIsa =
    SHADER_START
    R"(
        // Assume src address in s0, s1, and dst address in s2, s3
        s_movk_i32 s18, 0x5678
        .if (.amdgcn.gfx_generation_number >= 10)
            v_mov_b32 v0, s2
            v_mov_b32 v1, s3
            v_mov_b32 v2, 0x5678
        .endif
        LOOP:
        .if (.amdgcn.gfx_generation_number >= 12)
            s_load_dword s16, s[0:1], 0x0 scope:SCOPE_SYS
        .else
            s_load_dword s16, s[0:1], 0x0 glc
        .endif
        s_cmp_eq_i32 s16, s18
        s_cbranch_scc0   LOOP
        .if (.amdgcn.gfx_generation_number >= 12)
            flat_store_dword v[0:1], v2 scope:SCOPE_SYS
        .elseif (.amdgcn.gfx_generation_number >= 10)
            flat_store_dword v[0:1], v2 slc
        .else
            s_store_dword s18, s[2:3], 0x0 glc
        .endif
        s_endpgm
)";

/* Similar to PollMemoryIsa except that the buffer
 * polled can be Non-coherant memory. SCC system-level
 * cache coherence is not supported in scalar (smem) path.
 * Use vmem operations with scc
 */
const char *PollNCMemoryIsa =
    SHADER_START
    R"(
        // Assume src address in s0, s1, and dst address in s2, s3
        v_mov_b32 v6, 0x5678
        v_mov_b32 v0, s0
        v_mov_b32 v1, s1
        LOOP:
        flat_load_dword v4, v[0:1] scc
        v_cmp_eq_u32 vcc, v4, v6
        s_cbranch_vccz   LOOP
        v_mov_b32 v0, s2
        v_mov_b32 v1, s3
        flat_store_dword v[0:1], v6 scc
        s_endpgm
)";

/* Input: A buffer of at least 3 dwords.
 * DW0: used as a signal. 0xcafe means it is signaled
 * DW1: Input buffer for device to read.
 * DW2: Output buffer for device to write.
 * Once receive signal, device will copy DW1 to DW2
 * This shader continously poll the signal buffer,
 * Once signal buffer is signaled, it copies input buffer
 * to output buffer
 */
const char *CopyOnSignalIsa =
    SHADER_START
    R"(
        // Assume input buffer in s0, s1
        .if (.amdgcn.gfx_generation_number >= 10)
            s_add_u32 s2, s0, 0x8
            s_addc_u32 s3, s1, 0x0
            s_mov_b32 s18, 0xcafe
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            v_mov_b32 v4, s2
            v_mov_b32 v5, s3
        .else
            s_mov_b32 s18, 0xcafe
        .endif

        .if (.amdgcn.gfx_generation_number >= 12)

            POLLSIGNAL:
            s_load_dword s16, s[0:1], 0x0 scope:SCOPE_SYS
            s_cmp_eq_i32 s16, s18
            s_cbranch_scc0   POLLSIGNAL

            s_load_dword s17, s[0:1], 0x4 scope:SCOPE_SYS
            s_wait_kmcnt 0

            v_mov_b32 v2, s17
            flat_store_dword v[4:5], v2 scope:SCOPE_SYS
        .else

            POLLSIGNAL:
            s_load_dword s16, s[0:1], 0x0 glc
            s_cmp_eq_i32 s16, s18
            s_cbranch_scc0   POLLSIGNAL

            s_load_dword s17, s[0:1], 0x4 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            .if (.amdgcn.gfx_generation_number >= 10)
                v_mov_b32 v2, s17
                flat_store_dword v[4:5], v2 glc
            .else
                s_store_dword s17, s[0:1], 0x8 glc
            .endif
            s_waitcnt vmcnt(0) & lgkmcnt(0)

        .endif
        s_endpgm
)";

/* Continuously poll the flag at src buffer
 * After the flag of s[0:1] is 1 filled,
 * copy the value from s[0:1]+4 to dst buffer
 *
 * Note: Only works on GFX9 (only used in
 *       aldebaran tests)
 */
const char *PollAndCopyIsa =
    SHADER_START
    SHADER_MACROS_FLAT
    R"(
        // Assume src buffer in s[0:1] and dst buffer in s[2:3]
        // Path for Aldebaran, Aqua Vanjaram
        .if (.amdgcn.gfx_generation_number == 9 && (.amdgcn.gfx_generation_minor >= 4 || .amdgcn.gfx_generation_stepping == 10))
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            v_mov_b32 v18, 0x1
            LOOP0:
            FLAT_LOAD_DWORD_NSS v16, v[0:1] glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            v_cmp_eq_i32 vcc, v16, v18
            s_cbranch_vccz LOOP0
            .if (.amdgcn.gfx_generation_minor >= 4)
                buffer_inv sc1 sc0
            .else
                buffer_invl2
            .endif
            s_load_dword s17, s[0:1], 0x4 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            s_store_dword s17, s[2:3], 0x0 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            buffer_wbl2
        .elseif (.amdgcn.gfx_generation_number == 9)
            s_movk_i32 s18, 0x1
            LOOP1:
            s_load_dword s16, s[0:1], 0x0 glc
            s_cmp_eq_i32 s16, s18
            s_cbranch_scc0 LOOP1
            s_load_dword s17, s[0:1], 0x4 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            s_store_dword s17, s[2:3], 0x0 glc
        .endif
        s_waitcnt vmcnt(0) & lgkmcnt(0)
        s_endpgm
)";

/* Input0: A buffer of at least 2 dwords.
 * DW0: used as a signal. Write 0x1 to signal
 * DW1: Write the value from 2nd input buffer
 *      for other device to read.
 * Input1: A buffer of at least 2 dwords.
 * DW0: used as the value to be written.
 *
 * Note: Only works on Aldebaran and Aqua Vanjaram
 */
const char *WriteFlagAndValueIsa =
    SHADER_START
    SHADER_MACROS_FLAT
    R"(
        // Assume two inputs buffer in s[0:1] and s[2:3]
        .if (.amdgcn.gfx_generation_number == 9 && (.amdgcn.gfx_generation_minor >= 4 || .amdgcn.gfx_generation_stepping == 10))
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            s_load_dword s18, s[2:3], 0x0 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            s_store_dword s18, s[0:1], 0x4 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            buffer_wbl2
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            v_mov_b32 v16, 0x1
            FLAT_STORE_DWORD_NSS v[0:1], v16 glc
        .endif
        s_endpgm
)";

/* Input0: A buffer of at least 2 dwords.
 * DW0: used as a signal. Write 0xcafe to signal
 * DW1: Write to this buffer for other device to read.
 * Input1: mmio base address
 */
const char *WriteAndSignalIsa =
    SHADER_START
    R"(
        // Assume input buffer in s0, s1
        .if (.amdgcn.gfx_generation_number >= 10)
            s_add_u32 s4, s0, 0x4
            s_addc_u32 s5, s1, 0x0
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            v_mov_b32 v2, s2
            v_mov_b32 v3, s3
            v_mov_b32 v4, s4
            v_mov_b32 v5, s5
            .if (.amdgcn.gfx_generation_number >= 12)
                v_mov_b32 v18, 0xbeef
                flat_store_dword v[4:5], v18 scope:SCOPE_SYS
                v_mov_b32 v18, 0x1
                flat_store_dword v[2:3], v18 scope:SCOPE_SYS
                v_mov_b32 v18, 0xcafe
                flat_store_dword v[0:1], v18 scope:SCOPE_SYS
            .else
                v_mov_b32 v18, 0xbeef
                flat_store_dword v[4:5], v18 glc
                v_mov_b32 v18, 0x1
                flat_store_dword v[2:3], v18 glc
                v_mov_b32 v18, 0xcafe
                flat_store_dword v[0:1], v18 glc
            .endif
        .else
            s_mov_b32 s18, 0xbeef
            s_store_dword s18, s[0:1], 0x4 glc
            s_mov_b32 s18, 0x1
            s_store_dword s18, s[2:3], 0 glc
            s_mov_b32 s18, 0xcafe
            s_store_dword s18, s[0:1], 0x0 glc
        .endif
        s_endpgm
)";

/* Input:
 * s[0:1], A buffer of at least 64 * 6 bytes
 *
 * Store the value 0x77 at the 5 addresses 0x40,
 * 0x80, ..., 0x140 in the buffer
 *
 * Aqua Vanjaram only
 */
const char *FlushBufferForAcquireReleaseIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
            s_mov_b32 s11, 0x77
            s_mov_b32 s12, 0x0
            // Store some data on 5 different cache lines
            s_store_dword s12, s[0:1], 0x0 glc
            s_store_dword s11, s[0:1], 0x40 glc
            s_store_dword s11, s[0:1], 0x80 glc
            s_store_dword s11, s[0:1], 0xc0 glc
            s_store_dword s11, s[0:1], 0x100 glc
            s_store_dword s11, s[0:1], 0x140 glc
            s_waitcnt lgkmcnt(0)
        .endif
        s_endpgm
)";

/* Input:
 * s[0:1], A buffer of at least 64 * 6 bytes,
 * shared with the acquiring shader
 *
 * Store the values 1 - 5 at the 5 addresses 0x40,
 * 0x80, ..., 0x140 in the buffer, then signal
 * the flag at address 0x0 in the buffer.
 *
 * Uses vector stores
 *
 * Aqua Vanjaram only
 */
const char *WriteReleaseVectorIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
            v_mov_b32 v11, 0x1
            v_mov_b32 v12, 0x2
            v_mov_b32 v13, 0x3
            v_mov_b32 v14, 0x4
            v_mov_b32 v15, 0x5
            v_mov_b32 v21, 0x40
            v_mov_b32 v22, 0x80
            v_mov_b32 v23, 0xc0
            v_mov_b32 v24, 0x100
            v_mov_b32 v25, 0x140
            // Store some data on 5 different cache lines
            global_store_dword v21, v11, s[0:1]
            global_store_dword v22, v12, s[0:1]
            global_store_dword v23, v13, s[0:1]
            global_store_dword v24, v14, s[0:1]
            global_store_dword v25, v15, s[0:1] nt sc1 sc0
            s_waitcnt vmcnt(0)
            // Write-Release
            s_mov_b32 s16, 0x1
            buffer_wbl2 sc1 sc0
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            s_store_dword s16, s[0:1], 0x0 glc
        .endif
        s_endpgm
)";

/* Input:
 * s[0:1], A buffer of at least 64 * 6 bytes,
 * shared with the acquiring shader
 *
 * Store the values 6 - 10 at the 5 addresses 0x40,
 * 0x80, ..., 0x140 in the buffer, then signal
 * the flag at address 0x0 in the buffer.
 *
 * Uses scalar stores
 *
 * Aqua Vanjaram only
 */
const char *WriteReleaseScalarIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
            s_mov_b32 s11, 0x6
            s_mov_b32 s12, 0x7
            s_mov_b32 s13, 0x8
            s_mov_b32 s14, 0x9
            s_mov_b32 s15, 0xa
            // Store some data on 5 different cache lines
            s_store_dword s11, s[0:1], 0x40
            s_store_dword s12, s[0:1], 0x80
            s_store_dword s13, s[0:1], 0xc0
            s_store_dword s14, s[0:1], 0x100
            s_store_dword s15, s[0:1], 0x140 glc
            s_waitcnt lgkmcnt(0)
            // Write-Release
            s_dcache_wb // WB Scalar L1 cache
            s_mov_b32 s16, 0x1
            buffer_wbl2 sc1 sc0
            s_waitcnt vmcnt(0) & lgkmcnt(0)
            s_store_dword s16, s[0:1], 0x0 glc
            s_waitcnt lgkmcnt(0)
        .endif
        s_endpgm
)";

/* Input:
 * s[0:1], A buffer of at least 64 * 6 bytes,
 * shared with the releasing shader
 * s[2:3], A buffer of at least 64 * 6 bytes,
 * accessible by the CPU, used for output
 *
 * Polls the flag at address 0x0 in the shared buffer.
 * When the signal is received, read the values
 * at the 5 addresses 0x40, 0x80, ... 0x140,
 * and store them at the same locations in
 * the output buffer
 *
 * Uses vector loads
 *
 * Aqua Vanjaram only
 */
const char *ReadAcquireVectorIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
            // Read-Acquire
            s_mov_b32 s18, 0x1
            LOOP:
            s_load_dword s17, s[0:1], 0x0 glc
            s_waitcnt lgkmcnt(0)
            s_cmp_eq_i32 s17, s18
            s_cbranch_scc0 LOOP
            buffer_inv sc1 sc0
            // Load data
            v_mov_b32 v21, 0x40
            v_mov_b32 v22, 0x80
            v_mov_b32 v23, 0xc0
            v_mov_b32 v24, 0x100
            v_mov_b32 v25, 0x140
            global_load_dword v11, v21, s[0:1]
            global_load_dword v12, v22, s[0:1]
            global_load_dword v13, v23, s[0:1]
            global_load_dword v14, v24, s[0:1]
            global_load_dword v15, v25, s[0:1]
            s_waitcnt vmcnt(0)
            // Store data for output
            v_mov_b32 v21, 0x40
            v_mov_b32 v22, 0x80
            v_mov_b32 v23, 0xc0
            v_mov_b32 v24, 0x100
            v_mov_b32 v25, 0x140
            global_store_dword v21, v11, s[2:3] nt sc1 sc0
            global_store_dword v22, v12, s[2:3] nt sc1 sc0
            global_store_dword v23, v13, s[2:3] nt sc1 sc0
            global_store_dword v24, v14, s[2:3] nt sc1 sc0
            global_store_dword v25, v15, s[2:3] nt sc1 sc0
            s_waitcnt vmcnt(0)
        .endif
        s_endpgm
)";

/* Input:
 * s[0:1], A buffer of at least 64 * 6 bytes,
 * shared with the releasing shader
 * s[2:3], A buffer of at least 64 * 6 bytes,
 * accessible by the CPU, used for output
 *
 * Polls the flag at address 0x0 in the shared buffer.
 * When the signal is received, read the values
 * at the 5 addresses 0x40, 0x80, ... 0x140,
 * and store them at the same locations in
 * the output buffer
 *
 * Uses scalar loads
 *
 * Aqua Vanjaram only
 */
const char *ReadAcquireScalarIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number == 9 && .amdgcn.gfx_generation_minor >= 4)
            // Read-Acquire
            s_mov_b32 s18, 0x1
            LOOP:
            s_load_dword s17, s[0:1], 0x0 glc
            s_waitcnt lgkmcnt(0)
            s_cmp_eq_i32 s17, s18
            s_cbranch_scc0 LOOP
            buffer_inv sc1 sc0
            // Load data
            s_load_dword s21, s[0:1], 0x40
            s_load_dword s22, s[0:1], 0x80
            s_load_dword s23, s[0:1], 0xc0
            s_load_dword s24, s[0:1], 0x100
            s_load_dword s25, s[0:1], 0x140
            s_waitcnt lgkmcnt(0)
            // Store data for output
            s_store_dword s21, s[2:3], 0x40 glc
            s_store_dword s22, s[2:3], 0x80 glc
            s_store_dword s23, s[2:3], 0xc0 glc
            s_store_dword s24, s[2:3], 0x100 glc
            s_store_dword s25, s[2:3], 0x140 glc
            s_waitcnt lgkmcnt(0)
        .endif
        s_endpgm
)";

/**
 * KFDQMTest
 */

/* A simple isa loop program with dense mathematic operations
 * s1 controls the number iterations of the loop
 * This shader can be used by GFX8, GFX9 and GFX10
 */
const char *LoopIsa =
    SHADER_START
    R"(
        s_movk_i32    s0, 0x0008
        s_movk_i32    s1, 0x00ff
        s_mov_b32     s4, 0
        s_mov_b32     s5, 0
        s_mov_b32     s6, 0
        s_mov_b32     s7, 0
        s_mov_b32     s12, 0
        s_mov_b32     s13, 0
        s_mov_b32     s14, 0
        s_mov_b32     s15, 0
        v_mov_b32     v0, 0
        v_mov_b32     v1, 0
        v_mov_b32     v2, 0
        v_mov_b32     v3, 0
        v_mov_b32     v4, 0
        v_mov_b32     v5, 0
        v_mov_b32     v6, 0
        v_mov_b32     v7, 0
        v_mov_b32     v8, 0
        v_mov_b32     v9, 0
        v_mov_b32     v10, 0
        v_mov_b32     v11, 0
        v_mov_b32     v12, 0
        v_mov_b32     v13, 0
        v_mov_b32     v14, 0
        v_mov_b32     v15, 0
        v_mov_b32     v16, 0
        LOOP:
        s_mov_b32     s8, s4
        s_mov_b32     s9, s1
        s_mov_b32     s10, s6
        s_mov_b32     s11, s7
        s_cmp_le_i32  s1, s0
        s_cbranch_scc1  END_OF_PGM
        v_add_f32     v0, 2.0, v0
        v_cvt_f32_i32 v17, s1
        .if (.amdgcn.gfx_generation_number >= 12)
            s_wait_dscnt     0
            s_wait_kmcnt     0
        .else
            s_waitcnt lgkmcnt(0)
        .endif
        v_add_f32     v18, s8, v17
        v_add_f32     v19, s9, v17
        v_add_f32     v20, s10, v17
        v_add_f32     v21, s11, v17
        v_add_f32     v22, s12, v17
        v_add_f32     v23, s13, v17
        v_add_f32     v24, s14, v17
        v_add_f32     v17, s15, v17
        v_log_f32     v25, v18
        v_mul_f32     v25, v22, v25
        v_exp_f32     v25, v25
        v_log_f32     v26, v19
        v_mul_f32     v26, v23, v26
        v_exp_f32     v26, v26
        v_log_f32     v27, v20
        v_mul_f32     v27, v24, v27
        v_exp_f32     v27, v27
        v_log_f32     v28, v21
        v_mul_f32     v28, v17, v28
        v_exp_f32     v28, v28
        v_add_f32     v5, v5, v25
        v_add_f32     v6, v6, v26
        v_add_f32     v7, v7, v27
        v_add_f32     v8, v8, v28
        v_mul_f32     v18, 0x3fb8aa3b, v18
        v_exp_f32     v18, v18
        v_mul_f32     v19, 0x3fb8aa3b, v19
        v_exp_f32     v19, v19
        v_mul_f32     v20, 0x3fb8aa3b, v20
        v_exp_f32     v20, v20
        v_mul_f32     v21, 0x3fb8aa3b, v21
        v_exp_f32     v21, v21
        v_add_f32     v9, v9, v18
        v_add_f32     v10, v10, v19
        v_add_f32     v11, v11, v20
        v_add_f32     v12, v12, v21
        v_sqrt_f32    v18, v22
        v_sqrt_f32    v19, v23
        v_sqrt_f32    v20, v24
        v_sqrt_f32    v21, v17
        v_add_f32     v13, v13, v18
        v_add_f32     v14, v14, v19
        v_add_f32     v15, v15, v20
        v_add_f32     v16, v16, v21
        v_rsq_f32     v18, v22
        v_rsq_f32     v19, v23
        v_rsq_f32     v20, v24
        v_rsq_f32     v17, v17
        v_add_f32     v1, v1, v18
        v_add_f32     v2, v2, v19
        v_add_f32     v3, v3, v20
        v_add_f32     v4, v4, v17
        s_add_u32     s0, s0, 1
        s_branch      LOOP
        END_OF_PGM:
        s_endpgm
)";


/**
 * KFDCWSRTest
 */

/* Initial state:
 *   s[0:1] - input buffer base address
 *   s[2:3] - output buffer base address
 *   s4 - workgroup id
 *   v0 - workitem id
 * Registers:
 *   v0 - calculated workitem = v0 + s4 * NUM_THREADS_X, which is s4
 *   v[4:5] - corresponding output buf address: s[2:3] + v0 * 4
 *   v6 - register storing known-value output for mangle testing
 *   v7 - counter
 */
const char *PersistentIterateIsa =
    SHADER_START
    SHADER_MACROS_U32
    SHADER_MACROS_FLAT
    R"(
        // Compute address of output buffer
        .if (.amdgcn.gfx_generation_number >= 12)
            v_mov_b32               v0, ttmp9   // use workgroup id as index
        .else
            v_mov_b32               v0, s4      // use workgroup id as index
        .endif
        v_lshlrev_b32           v0, 2, v0       // v0 *= 4
        V_ADD_CO_U32            v4, s2, v0      // v[4:5] = s[2:3] + v0 * 4
        v_mov_b32               v5, s3          // v[4:5] = s[2:3] + v0 * 4
        V_ADD_CO_CI_U32         v5, v5, 0       // v[4:5] = s[2:3] + v0 * 4

        // Store known-value output in register
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORD_NSS     v6, v[4:5] scope:SCOPE_SYS
            s_wait_loadcnt 0                        // wait for memory reads to finish
        .else
            FLAT_LOAD_DWORD_NSS     v6, v[4:5] glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)         // wait for memory reads to finish
        .endif

        // Initialize counter
        v_mov_b32               v7, 0

        LOOP:
        flat_store_dword        v[4:5], v6      // store known-val in output
        V_ADD_CO_U32            v7, 1, v7       // increment counter

        .if (.amdgcn.gfx_generation_number >= 12)
            s_load_dword            s6, s[0:1], 0 scope:SCOPE_SYS
            s_wait_loadcnt 0                        // wait for memory reads to finish
        .else
            s_load_dword            s6, s[0:1], 0 glc
            s_waitcnt vmcnt(0) & lgkmcnt(0)         // wait for memory reads to finish
        .endif
        s_cmp_eq_i32            s6, 0x12345678  // compare input buf to stopval
        s_cbranch_scc1          L_QUIT          // branch if notified to quit by host

        s_branch LOOP

        L_QUIT:
        s_endpgm
)";

/**
 * KFDEvictTest
 */

/* Shader to read local buffers using multiple wavefronts in parallel
 * until address buffer is filled with specific value 0x5678 by host program,
 * then each wavefront fills value 0x5678 at corresponding result buffer and quit
 *
 * Initial state:
 *   s[0:1]   - address buffer base address
 *   s[2:3]   - result buffer base address
 *   s4/ttmp9 - workgroup id (in s4 pre-GFX12, in ttmp9 on GFX12)
 *   v0       - workitem id, always 0 because NUM_THREADS_X(number of threads) in workgroup set to 1
 * Registers:
 *   v0 - calculated workitem id, v0 = v0 + s4 * NUM_THREADS_X
 *   v[2:3] - address of corresponding local buf address offset: s[0:1] + v0 * 8
 *   v[4:5] - corresponding output buf address: s[2:3] + v0 * 4
 *   v[6:7] - local buf address used for read test
 *   v11 - size of local buffer in MB
 */
const char *ReadMemoryIsa =
    SHADER_START
    SHADER_MACROS_U32
    SHADER_MACROS_FLAT
    R"(
        // Compute address of corresponding output buffer
        .if (.amdgcn.gfx_generation_number >= 12)
            v_mov_b32           v0, ttmp9       // use workgroup id as index
        .else
            v_mov_b32           v0, s4          // use workgroup id as index
        .endif
        v_lshlrev_b32           v0, 2, v0       // v0 *= 4
        V_ADD_CO_U32            v4, s2, v0      // v[4:5] = s[2:3] + v0 * 4
        v_mov_b32               v5, s3          // v[4:5] = s[2:3] + v0 * 4
        V_ADD_CO_CI_U32         v5, v5, 0       // v[4:5] = s[2:3] + v0 * 4

        // Compute input buffer offset used to store corresponding local buffer address
        v_lshlrev_b32           v0, 1, v0       // v0 *= 8
        V_ADD_CO_U32            v2, s0, v0      // v[2:3] = s[0:1] + v0 * 8
        v_mov_b32               v3, s1          // v[2:3] = s[0:1] + v0 * 8
        V_ADD_CO_CI_U32         v3, v3, 0       // v[2:3] = s[0:1] + v0 * 8

        // Load local buffer size from output buffer
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORD_NSS     v11, v[4:5] scope:SCOPE_DEV
        .else
            FLAT_LOAD_DWORD_NSS     v11, v[4:5] slc
        .endif

        // Load 64bit local buffer address stored at v[2:3] to v[6:7]
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORDX2_NSS   v[6:7], v[2:3] scope:SCOPE_DEV
            s_wait_loadcnt 0
        .else
            FLAT_LOAD_DWORDX2_NSS   v[6:7], v[2:3] slc
            s_waitcnt vmcnt(0) & lgkmcnt(0)         // wait for memory reads to finish
        .endif
        v_mov_b32               v8, 0x5678
        s_movk_i32              s8, 0x5678
        L_REPEAT:
        .if (.amdgcn.gfx_generation_number >= 12)
            s_load_dword        s16, s[0:1], 0x0 scope:SCOPE_SYS
            s_wait_kmcnt        0                      // wait for memory reads to finish
        .else
            s_load_dword        s16, s[0:1], 0x0 glc
            s_waitcnt           vmcnt(0) & lgkmcnt(0)  // wait for memory reads to finish
        .endif
        s_cmp_eq_i32            s16, s8
        s_cbranch_scc1          L_QUIT          // if notified to quit by host

        // Loop read local buffer starting at v[6:7]
        // every 4k page only read once
        v_mov_b32               v9, 0
        v_mov_b32               v10, 0x1000     // 4k page
        v_mov_b32               v12, v6
        v_mov_b32               v13, v7
        L_LOOP_READ:
        .if (.amdgcn.gfx_generation_number >= 12)
            FLAT_LOAD_DWORDX2_NSS   v[14:15], v[12:13] scope:SCOPE_DEV
        .else
            FLAT_LOAD_DWORDX2_NSS   v[14:15], v[12:13] slc
        .endif
        V_ADD_CO_U32            v9, v9, v10
        V_ADD_CO_U32            v12, v12, v10
        V_ADD_CO_CI_U32         v13, v13, 0
        V_CMP_LT_U32            v9, v11
        s_cbranch_vccnz         L_LOOP_READ
        s_branch                L_REPEAT
        L_QUIT:
        flat_store_dword        v[4:5], v8
        .if (.amdgcn.gfx_generation_number >= 12)
            s_wait_storecnt     0
        .else
            s_waitcnt vmcnt(0) & lgkmcnt(0)         // wait for memory writes to finish
        .endif
        s_endpgm
)";

/**
 * KFDGWSTest
 */

/* Shader to initialize gws counter to 1 */
const char *GwsInitIsa =
    SHADER_START
    R"(
        .if (.amdgcn.gfx_generation_number >= 12)
        .else
            s_mov_b32 m0, 0
            s_nop 0
            s_load_dword s16, s[0:1], 0x0 glc
            s_waitcnt 0
            v_mov_b32 v0, s16
            s_waitcnt 0
            ds_gws_init v0 offset:0 gds
            s_waitcnt 0
            s_endpgm
        .endif
)";

/* Atomically increase a value in memory
 * This is expected to be executed from
 * multiple work groups simultaneously.
 * GWS semaphore is used to guarantee
 * the operation is atomic.
 */
const char *GwsAtomicIncreaseIsa =
    SHADER_START
    R"(
        // Assume src address in s0, s1
        .if (.amdgcn.gfx_generation_number >= 12)
        .elseif (.amdgcn.gfx_generation_number >= 10)
            s_mov_b32 m0, 0
            s_mov_b32 exec_lo, 0x1
            v_mov_b32 v0, s0
            v_mov_b32 v1, s1
            ds_gws_sema_p offset:0 gds
            s_waitcnt 0
            flat_load_dword v2, v[0:1] glc dlc
            s_waitcnt 0
            v_add_nc_u32 v2, v2, 1
            flat_store_dword v[0:1], v2
            s_waitcnt_vscnt null, 0
            ds_gws_sema_v offset:0 gds
        .else
            s_mov_b32 m0, 0
            s_nop 0
            ds_gws_sema_p offset:0 gds
            s_waitcnt 0
            s_load_dword s16, s[0:1], 0x0 glc
            s_waitcnt 0
            s_add_u32 s16, s16, 1
            s_store_dword s16, s[0:1], 0x0 glc
            s_waitcnt lgkmcnt(0)
            ds_gws_sema_v offset:0 gds
        .endif
        s_waitcnt 0
        s_endpgm
)";


/*
 * Shader used by ExtendedCuMasking test case to check if CU mask is used correctly.
 *
 * Shader will write to output buffer the (SE, SA, WGP) used by the wave.
 * The test program will then analyse the data.
 *
 * Inputs
 * ------
 * s[2:3]  : output buffer base address
 * s4/ttmp9: workgroup id (s4 for pre-GFX12, ttmp9 for GFX12)
 *
 * Output
 * ------
 * Store HW_ID1 content in output buffer at index corresponding to workgroup id.
 *
 */
const char *CheckCuMaskIsa =
    SHADER_START
    SHADER_MACROS_U32
    SHADER_MACROS_FLAT
    R"(
        // Get workgroup id
        .if (.amdgcn.gfx_generation_number >= 12)
            v_mov_b32    v0, ttmp9
        .else
            v_mov_b32    v0, s4
        .endif

        // Address of output buffer element: v[4:5] = s[2:3] + v0 * 4
        v_lshlrev_b32    v6, 2, v0
        V_ADD_CO_U32     v4, s2, v6
        v_mov_b32        v5, s3
        V_ADD_CO_CI_U32  v5, v5, 0

        // Store HW_ID1 content
        .if (.amdgcn.gfx_generation_number >= 12)
            s_getreg_b32     s6, hwreg(HW_REG_HW_ID1)
        .else
            s_getreg_b32     s6, hwreg(HW_REG_HW_ID)
        .endif
        v_mov_b32        v1, s6
        flat_store_dword v[4:5], v1

        s_endpgm
)";


const char *JumpToTrapIsa =
    SHADER_START
    SHADER_MACROS_U32
    R"(
        /*copy the parameters from scalar registers to vector registers*/
        v_mov_b32 v4, 0
        v_mov_b32 v0, s0
        v_mov_b32 v1, s1
        s_trap 1
        EXIT_LOOP:
        V_CMP_EQ_U32 v4, 0
        s_cbranch_vccnz EXIT_LOOP
        flat_store_dword v[0:1], v4
        s_waitcnt vmcnt(0)&lgkmcnt(0)
        s_endpgm
)";

const char *TrapHandlerIsa =
    SHADER_START
    R"(
        CHECK_VMFAULT:
        /*if trap jumped to by vmfault, restore skip m0 signalling*/
        .if (.amdgcn.gfx_generation_number < 12)
            s_getreg_b32 ttmp14, hwreg(HW_REG_TRAPSTS)
            s_and_b32 ttmp2, ttmp14, 0x800
        .else
            s_getreg_b32 ttmp14, hwreg(HW_REG_EXCP_FLAG_PRIV)
            s_and_b32 ttmp2, ttmp14, 0x10
        .endif
        s_cbranch_scc1 RESTORE_AND_EXIT
        /*check for address watch event and record pc check point delta*/
        .if (.amdgcn.gfx_generation_number < 12)
            s_and_b32 ttmp2, ttmp14, 0x7080
        .else
            s_and_b32 ttmp2, ttmp14, 0xf
        .endif
        s_cbranch_scc0 GET_DOORBELL
        v_mov_b32 v5, v4 // capture watch checkpoint
        v_mov_b32 v6, ttmp14 // capture watch trapsts
        s_branch RESTORE_AND_EXIT
        GET_DOORBELL:
        .if .amdgcn.gfx_generation_number < 11
            s_mov_b32 ttmp2, exec_lo
            s_mov_b32 ttmp3, exec_hi
            s_mov_b32 exec_lo, 0x80000000
            s_sendmsg 10
            WAIT_SENDMSG:
            /*wait until msb is cleared (i.e. doorbell fetched)*/
            s_nop 7
            s_bitcmp0_b32 exec_lo, 0x1F
            s_cbranch_scc0 WAIT_SENDMSG
            /* restore exec */
            s_mov_b32 exec_hi, ttmp3
            s_and_b32 exec_lo, exec_lo, 0xfff
            s_mov_b32 ttmp3, exec_lo
            s_mov_b32 exec_lo, ttmp2
        .else
            s_sendmsg_rtn_b32 ttmp3, sendmsg(MSG_RTN_GET_DOORBELL)
            s_waitcnt lgkmcnt(0)
            s_and_b32 ttmp3, ttmp3, 0x3ff
        .endif
        s_mov_b32 ttmp2, m0
        s_or_b32 ttmp3, ttmp3, 0x800
        /* set m0, send interrupt and restore m0 and exit trap*/
        s_mov_b32 m0, ttmp3
        s_nop 0x0
        s_sendmsg sendmsg(MSG_INTERRUPT)
        s_waitcnt lgkmcnt(0)
        s_mov_b32 m0, ttmp2
        v_mov_b32 v4, ttmp1
        .if (.amdgcn.gfx_generation_number >= 12)
            s_getreg_b32 ttmp14, hwreg(HW_REG_EXCP_FLAG_PRIV)
            s_and_b32 ttmp2, ttmp14, 0x300
            s_cbranch_scc1 RESTORE_AND_EXIT
         .elseif (.amdgcn.gfx_generation_number == 11)
            s_getreg_b32 ttmp14, hwreg(HW_REG_TRAPSTS)
            s_and_b32 ttmp2, ttmp14, 0x30000
            s_cbranch_scc1 RESTORE_AND_EXIT
         .else
            s_getreg_b32 ttmp14, hwreg(HW_REG_TRAPSTS)
            s_and_b32 ttmp2, ttmp14, 0x1800000
            s_cbranch_scc1 RESTORE_AND_EXIT
         .endif
        /* restore and increment program counter to skip shader trap jump*/
        s_add_u32 ttmp0, ttmp0, 4
        s_addc_u32 ttmp1, ttmp1, 0
        s_and_b32 ttmp1, ttmp1, 0xffff
        RESTORE_AND_EXIT:
        /* restore SQ_WAVE_IB_STS */
        s_lshr_b32 ttmp2, ttmp11, (26 - 15)
        s_and_b32 ttmp2, ttmp2, (0x8000 | 0x1F0000)
        s_setreg_b32 hwreg(HW_REG_IB_STS), ttmp2
        /* restore SQ_WAVE_STATUS */
        s_and_b64 exec, exec, exec
        s_and_b64 vcc, vcc, vcc
        s_setreg_b32 hwreg(HW_REG_STATUS), ttmp12
        s_rfe_b64 [ttmp0, ttmp1]
)";

#define WATCH_START SHADER_START SHADER_MACROS_U32\
    "v_mov_b32 v0, s0\n"\
    "v_mov_b32 v1, s1\n"\
    "v_mov_b32 v2, s2\n"\
    "v_mov_b32 v3, s3\n"\
    "flat_load_dword v4, v[2:3]\n"\
    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n"\
    "v_mov_b32 v5, 0\n"\
    "v_mov_b32 v6, 0\n"

#define WATCH_END "\n"\
    "v_mov_b32 v4, 2\n"\
    "LOOP:\n"\
    "V_CMP_EQ_U32 v6, 0\n"\
    "s_cbranch_vccnz LOOP\n"\
    "V_ADD_CO_U32 v6, v6, v5\n"\
    "flat_store_dword v[2:3], v6\n"\
    "s_waitcnt vmcnt(0) & lgkmcnt(0)\n"\
    "s_endpgm\n"

const char *WatchReadIsa =
    WATCH_START
    "flat_load_dword v7, v[0:1]"
    WATCH_END;

const char *WatchWriteIsa =
    WATCH_START
    "flat_store_dword v[0:1], v4"
    WATCH_END;
