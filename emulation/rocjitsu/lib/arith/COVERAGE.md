# Instruction coverage

RDNA3 and RDNA4 FP16/BF16 WMMA with FP32 output and FP16/FP32 FMA support both
wave sizes, loose and empirical exact arithmetic, instruction modifiers, and
their documented guest FP controls. FP16 FMA reports status for the supported
gfx12 policy; FP32 and gfx11 FMA status requests are unsupported. Compatibility
resolves to each instruction’s inherited policy. AVX2, AVX512, and native BF16 implementations are enabled for supported CPUs.
