# Instruction coverage

RDNA3 and RDNA4 support FP16/BF16 WMMA with FP32 output and FP16/FP32 FMA,
in both wave sizes. All support loose and empirical exact numeric results.
Only RDNA4 FP16 FMA supports FP-status reporting. WMMA has no status output.
Instruction modifiers and guest controls are not yet supported.
