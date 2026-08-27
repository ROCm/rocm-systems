# Decoder fuzzing

This directory keeps the coverage-guided decoder loop independent of LLVM. The
hot path supports every target in rocjitsu's statically composed registry and
accepts an exact 16-byte little-endian instruction window. Sixteen bytes cover
the largest supported encoding while letting the decoder report the actual 4,
8, 12, or 16-byte instruction length. The registry contains AMDGPU targets
only. Canonical target IDs and their registered aliases are accepted, including
`gfx950`, `gfx1201`, and `gfx1250`.

The fuzz executable links the model-only AMDGPU registry. It therefore carries
the generated decoder and disassembler code for every AMDGPU target without
linking the execution backends, VM, simdojo, or DBT implementation.

LLVM comparison is a separate offline process. `decode_differential.py` invokes
a caller-selected `llvm-mc`, compares only the first instruction in each
window, and writes structured mismatch records. LLVM is therefore neither a
configure-time nor a link-time dependency.

## Build and test

From this directory's project root (`emulation/rocjitsu`):

```sh
cmake -S . -B build-decode -GNinja \
  -DBUILD_TESTING=ON
ninja -C build-decode rj_decode_fuzz rj_decode_fuzz_test
ctest --test-dir build-decode -j4 \
  -R '^Decode(Fuzz(Core|Targets)Test\.|DifferentialUnitTest$|SeedExtractorUnitTest$|FuzzCliTest$)' \
  --output-on-failure
```

Generate model-derived seeds:

```sh
build-decode/fuzz/decode/rj_decode_fuzz \
  --emit-seeds decode-seeds --target gfx1201
```

The destination must be absent or empty so stale non-window files cannot enter
the AFL++ corpus.

Object files can supply additional seeds without putting ELF parsing in the
fuzz loop. The extractor samples aligned `.text` windows deterministically and
records their source offsets:

```sh
python3 fuzz/decode/prepare_decode_seeds.py \
  --input /path/to/object-corpus --output object-seeds \
  --llvm-objcopy /path/to/llvm-objcopy
```

Inputs must be unbundled AMDGPU ELF code objects. Other ELF machine types and
host executables containing bundled device code are skipped.

The extractor writes provenance beside the seed directory as
`object-seeds.manifest.json`, keeping the AFL++ input directory binary-only.
Generated test encodings currently provide at most two populated DWORDs for
most targets; use object-derived seeds to seed broader 12- and 16-byte windows.

## AFL++ with sanitizers

Configure the entire model with the AFL++ compiler wrappers so coverage reaches
the generated decoder code:

```sh
AFL_ROOT=/path/to/AFLplusplus
cmake -S . -B build-decode-afl -GNinja \
  -DCMAKE_C_COMPILER="$AFL_ROOT/afl-clang-fast" \
  -DCMAKE_CXX_COMPILER="$AFL_ROOT/afl-clang-fast++" \
  -DBUILD_TESTING=ON \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON
ninja -C build-decode-afl rj_decode_fuzz

"$AFL_ROOT/afl-fuzz" -i decode-seeds -o decode-findings \
  -g 16 -G 16 -m none -- \
  build-decode-afl/fuzz/decode/rj_decode_fuzz --afl --target gfx1250
```

Decoder failure is an ordinary rejection. Exceptions from unrelated failures, signals, sanitizer
findings, and decoder invariant failures remain crashes for AFL++ to retain.

## Offline LLVM comparison

Compare an AFL++ queue or a single minimized crash using a pinned tool path:

```sh
python3 fuzz/decode/decode_differential.py \
  --decoder build-decode-afl/fuzz/decode/rj_decode_fuzz \
  --llvm-mc /path/to/llvm-mc \
  --target gfx1201 \
  --corpus decode-findings/default/queue \
  --output decode-mismatches.jsonl \
  --llvm-revision LLVM_SOURCE_COMMIT
```

The comparator checks acceptance, consumed byte count, mnemonic, numeric
tokens, and the full ordered textual form. It normalizes case, whitespace, and
equivalent decimal/hexadecimal integer spelling only. Register ranges,
mnemonic suffixes, operand order, floating-point spelling, punctuation, and
modifier order remain significant. Every mismatch contains both raw and
normalized text, tool revisions, and replay commands. A decoder or `llvm-mc`
process failure is recorded as its own mismatch category without stopping the
rest of the corpus. Failures seen only while probing shorter instruction
prefixes are retained separately as `llvm_prefix_tool_failure` records.
Expect even generated seeds to produce a high mismatch rate, potentially around
half the corpus. Common noise includes encoding suffixes, tied implicit
operands, default `op_sel_hi` modifiers, and true16 register-half spelling.

Raw mutated encodings are not proof that an instruction is executable. Leave
the default `--input-qualification unqualified` for AFL++ queues and use
`generated-unqualified` for generated opcode seeds. Generated seeds can still
contain unusable zero-filled operand fields. Use `manual-qualified` only after
checking the complete instruction. Textual, operand, or acceptance differences
from unqualified inputs are investigation leads; qualify them against the
public ISA manual before treating either decoder as incorrect. Crashes,
sanitizer findings, and invariant violations do not require that semantic
qualification.
