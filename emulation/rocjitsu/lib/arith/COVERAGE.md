# Instruction coverage

The two gfx12 Wave32 instructions support loose and empirical bit-exact numeric
results with default controls. FP16 FMA status reporting is not yet implemented;
mode 3 is rejected before outputs are changed. WMMA has no FP-status output.
