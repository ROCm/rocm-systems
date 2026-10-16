# Instruction coverage

The gfx12 Wave32/Wave64 FP16/BF16 WMMA instructions with FP32 outputs and FP16/FP32
FMA support loose and empirical exact numeric results with default controls.
FP16 FMA supports FP status; FP32 FMA rejects mode 3 without changing outputs.
WMMA has no FP-status output.
