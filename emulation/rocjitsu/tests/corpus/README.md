# RocJITsu corpus policy

`gfx1250_b0_a0_semantic_tests.json` selects semantic programs whose instruction
forms have implemented runtime translations. The companion
`gfx1250_b0_a0_semantic_rewrites.json` pins the exact positive offline rewrite
count for every selected executable.

Four source-coverage programs are intentionally outside this qualification
until their translations are implemented:

- `barrier_signal_isfirst_test`
- `fp8_e5m3_pack_test`
- `wmma_split_f16_test`
- `wmma_split_fp4_test`

The offline translator currently rejects those forms as unsupported. They are
not silently accepted or treated as passing translations.

## gfx1201 simulator exclusions

`fpsan_global_load_tr_gfx12_w64_test` is excluded because its wave64
`global_load_tr` path currently attempts to write a lane outside the gfx1201
simulator wavefront. This is a simulator modeling gap, not a change introduced
by the corpus SDK nightly.

## Sanitizer simulator coverage

The Clang and GCC ASan+UBSan lanes run `fpsan_build_canary`, `fpsan_cast_test`,
and `fpsan_core_test` on every simulated target. These runtime cases exercise
the instrumented launcher and interpreter while keeping the five-target matrix
within its CI job budget. The release lane continues to run the complete
corpus.
