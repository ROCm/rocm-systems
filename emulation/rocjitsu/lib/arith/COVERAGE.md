# Instruction coverage

The current implementation provides gfx12 Wave32 FP16 WMMA with FP32 output
and FP16 FMA, in loose semantics with default instruction and guest controls.
Strict exact semantics are rejected without modifying outputs.
