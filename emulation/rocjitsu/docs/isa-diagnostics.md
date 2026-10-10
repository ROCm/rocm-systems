# ISA diagnostics

RocJITsu checks for these explicitly undefined ISA conditions:

| Check | Targets | Legal cases kept silent |
| --- | --- | --- |
| Negative combined offset on non-buffer scalar memory | CDNA1–4, RDNA1–4 (including RDNA3.5) | A negative immediate compensated by the unsigned register offset, including unaligned offsets; CDNA5 ordinary loads |
| Negative scalar-buffer load immediate | RDNA4, CDNA5 | Nonnegative immediates, including loads beyond the buffer bounds that return zero |
| MFMA broadcast group larger than the instruction's block count | CDNA3, CDNA4 | All supported integer/float broadcast sizes; F64 ignores CBSZ; F8/F6/F4 format selectors that reuse the CBSZ field |
| Nonempty, partial EXEC on global transpose loads | RDNA4, CDNA5 | Full or empty effective EXEC, including wave32 with unused upper EXEC bits |
| VOPD source-bank/scalar-input limits | RDNA3, RDNA3.5, RDNA4, CDNA5 | GFX12 same-register/same-width sharing and MOV/MOV routing; unused sources; unqualified cross-operation dependencies and older EXEC/SCC budgets |
| `NEG[2]` or `NEG_HI` on IU DOT/WMMA | RDNA3, RDNA3.5, RDNA4, CDNA5 | `NEG[1:0]` signedness, floating-point modifiers, unqualified SWMMAC forms and CDNA5 DOT8 |
| Misaligned normal SGPR data in `s_mov_b64` and scalar loads, or scalar-buffer descriptors | RDNA4, CDNA5 | Constants, special selectors, NULL destinations; other operand roles and ISA families |
| Zero or non-power-of-two scalar group width on permlane BCAST/UP/DOWN/XOR | CDNA5 | Larger powers of two, empty EXEC, unqualified VGPR widths and IDX_GEN |

Each `isa-undefined` warning identifies the CU, workgroup, wave and PC. Execution
continues with the simulator's existing behavior. Reports are capped at 16 per CU.
Static operand restrictions are decoded once and share one fast-path flag test.
They are reported on the issuing thread, including when a helper executes the
instruction. Other checks use operands already calculated for execution; they
do not scan register allocations, track lane conflicts or maintain an access
history.

These small checks run unconditionally, independently of memory-wait checking,
in both RocJITsu and Mirage. They have no CLI or configuration switch.
A warning on a defined instruction is a checker bug and should be reported.

Combined offsets are checked before alignment masks are applied. CDNA1–4 and
RDNA1–3.5 scalar-buffer operations are excluded: their separate manual sections do not
establish the ordinary scalar-memory rule for buffer offsets.

VOPD checks apply only to nonempty wave32 execution. They account for FMAMK's
SRC2 port, the older accumulator-port rules, and explicit VOPD3 sources. The
bank check includes both registers of F64 sources and ignores scalar sources.
Scalar input counting includes CNDMASK's implicit VCC read on RDNA3/3.5/4 and
deduplicates explicit VCC_LO reads. CDNA5 implicit VCC and RDNA3/3.5 EXEC/SCC
budgets remain unqualified. GFX12 identical SRC2 sharing stays silent because
LLVM permits it despite the manual's broader parity rule. Alignment checks require
an even SGPR for 64-bit data and a multiple of four for wider data or a buffer
descriptor. They only cover normal SGPR tuples wholly within their register
region; they do not infer lane-mask widths from generic operand metadata.

Cross-operation VOPD dependencies remain unchecked. RDNA4's broad prohibition
conflicts with LLVM's deliberate single-cycle pairing, while CDNA5's restriction
depends on product-specific multi-cycle execution. Neither supports an
unconditional warning with the available qualification.

This is a qualified set of checks, not a complete ISA validator. An access beyond
an allocation is not sufficient evidence of undefined behavior: the manuals
define zero-filled LDS reads, discarded LDS writes, register-source substitution
and discarded register destinations in many cases. The diagnostics do not flag
those cases or same-address LDS/global stores. They also do not infer HIP/C++
data races from machine instructions. Source-register rules differ between
instruction classes and sometimes conflict with general descriptions, so a
generic allocation-bound warning would not meet this checker's standard.

The checks follow the scalar-memory addressing sections across the supported
families, and the MFMA broadcast sections of
the [CDNA3](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-mi300-cdna3-instruction-set-architecture.pdf)
and [CDNA4](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
manuals, scalar-memory addressing and VOPD/VOP3P restrictions in the
[RDNA3 manual](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna3-shader-instruction-set-architecture-feb-2023_0.pdf),
and the scalar-memory, SGPR-alignment, VOPD/VOP3P, global load-transpose and
permlane sections of the
[RDNA4](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna4-instruction-set-architecture.pdf)
and [CDNA5](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna5-instruction-set-architecture.pdf)
manuals. CDNA5 explicitly permits negative combined offsets on ordinary scalar
loads; its buffer-load restriction still applies even when a register offset
would compensate for a negative immediate.

RDNA3/3.5 negative scalar-buffer offsets specify a MEMVIOL rather than undefined
behavior. That fault is outside this checker; its absence from the warnings does
not imply the access succeeds on hardware.
