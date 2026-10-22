# Instruction coverage

The two gfx12 Wave32 instructions support loose and empirical bit-exact numeric
results with default controls. FP16 FMA also supports FP-status reporting
(mode 3); WMMA has no FP-status output. Both accept `RJ_ARITH_FP16_OVFL`: it saturates finite FMA overflow while
preserving OVERFLOW | INEXACT status, and has no effect on WMMA’s FP32 result.
Other guest-mode overrides and instruction modifiers are not yet supported.
