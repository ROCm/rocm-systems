# DBT branch work summary

This branch was rewritten from `37dc74766b3f080a44be29ebe66bd13ef7b45c69`
into reviewable commits. The original running logs were intentionally omitted;
this file keeps the durable summary of what was implemented and where to review
it in the rewritten history.

## Rewritten history

- `dbt: add cursor text relocation foundation`
  - Replaces local code-cave patching with cursor-based text emission.
  - Remaps direct and recovered indirect branch targets after text growth.
  - Adds long-branch support for out-of-range relocated transfers.

- `hooks: add HSA profiling timestamp support`
  - Records dispatch start/end timestamps in the VM signal path.
  - Forwards AMD profiling APIs through guest-to-host agent mapping.
  - Fixes HIP/MIOpen paths that expect event timing to work under DBT.

- `dbt: implement virtual LDS sidecar dispatch`
  - Emits normal and virtual-LDS sidecar descriptors.
  - Encodes virtual-LDS metadata in the translated code object.
  - Rewrites dispatch packets to use backing buffers when LDS exceeds CDNA3.

- `dbt: add cdna4 to cdna3 semantic hardening`
  - Adds BF16, dot2, MFMA, permlane, bitop3, and scratch/liveness fixes.
  - Regenerates CDNA4 read/write operand facts used by liveness.
  - Adds BF16 pack and workload-shaped semantic regression coverage.

- `dbt: harden virtual LDS fallback coverage`
  - Keeps virtual LDS as a runtime fallback instead of the default path.
  - Gates fallback on total dispatch group size.
  - Covers sidecar lookup, b128 reads, dword MUBUF-to-LDS, AccVGPR sources,
    packet-boundary cases, and buffer lifetime fencing.
  - Adds HipKittens BF16 matmul fixtures as virtual-LDS workload coverage.

- `hooks: add DBT usability enhancements`
  - Adds persistent translated-code-object cache support.
  - Adds skipped-kernel loadability, diagnostics, and optional code-object dumps.
  - Keeps Qwen activation/generation/SDPA probes as focused repro tools.

- `docs: document DBT branch work`
  - Adds this summary and the text relocation design note.

## Validation themes

- Translation unit tests cover relocated text layout, branch fixups, descriptor
  updates, CDNA4-to-CDNA3 semantic expansions, virtual-LDS metadata, and
  fallback policy boundaries.
- HIP tests cover large-LDS copy shapes and HipKittens BF16 matmul fixtures.
- Qwen and ResNet repro/probe tooling remains available for workload triage, but
  the raw investigation logs were removed from the rewritten history.

## Workload reproduction

Use `repro_qwen_resnet.md` for the full `sharkmi300x2` sync, ROCm SDK
environment, build commands, and inline Python checks. The short setup on the
remote host is:

```bash
source ~/resnet_torch/.env/bin/activate
cd /tmp/rocjitsu-users-Groverkss-resnet_check_prs

root=$(rocm-sdk path --root)
bin=$(rocm-sdk path --bin)
cmake_prefix=$(rocm-sdk path --cmake)
export ROCM_PATH="$root"
export PATH="$bin:$PATH"
export LD_LIBRARY_PATH="$root/lib:${LD_LIBRARY_PATH:-}"

cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_PREFIX_PATH="$cmake_prefix" \
  -DHSA_RUNTIME64="$root/lib/libhsa-runtime64.so" \
  -DAMDCLANG="$(which amdclang++)"
cmake --build build --target rocjitsu_bin rocjitsu_kmd_shim rocjitsu_hooks -j"$(nproc)"
```

To reproduce the ResNet50 FP16 acceptance path, run sections 5 and 6 of
`repro_qwen_resnet.md`. The native and DBT runs save
`/tmp/resnet-fp16-native-output.pt` and `/tmp/resnet-fp16-dbt-output.pt`; the
expected DBT signal is a finite `(1, 1000)` output with `argmax 21`. The current
known-good comparison has matching top-5 class IDs `[21, 22, 92, 127, 23]` and
output drift around `max=0.0068`, `mean=0.0008`.

The final 128-token Qwen generation legs used the shark-side
`/home/kunwar/resnet_pytorch/qwen_suite.py` wrapper from the
`~/resnet_pytorch/.env` environment, cache-enabled BF16, default attention, the
suite's default prompt set, and CPU/forward/batch checks disabled. At the time
of the matrix run the default prompt set was six short prompts plus two longer
diagnostic prompts:

```bash
source ~/resnet_pytorch/.env/bin/activate

out=/tmp/rocjitsu-qwen-results-rerun
mkdir -p "$out"

while read -r slug model; do
  cache=/tmp/rocjitsu-qwen-cache-t128-${slug}
  native_json=$out/${slug}-t128-native.json
  dbt_json=$out/${slug}-t128-dbt.json

  ROCR_VISIBLE_DEVICES=0 \
    python /home/kunwar/resnet_pytorch/qwen_suite.py run \
      --model "$model" --tokens 128 --use-cache \
      --torch-dtype bfloat16 --attn-implementation default \
      --no-cpu-reference --no-forward-check --no-batch-check \
      --warmup-runs 0 --measure-runs 1 \
      --json-out "$native_json" \
      2>&1 | tee "$out/${slug}-t128-native.log"

  ROCJITSU_DBT_CACHE_DIR="$cache" ROCJITSU_DBT_DUMP_ALL=0 ROCR_VISIBLE_DEVICES=0 \
    build/tools/rocjitsu/rocjitsu \
      --config configs/guest_gfx950_on_gfx942.json -- \
      python /home/kunwar/resnet_pytorch/qwen_suite.py run \
        --model "$model" --tokens 128 --use-cache \
        --torch-dtype bfloat16 --attn-implementation default \
        --no-cpu-reference --no-forward-check --no-batch-check \
        --warmup-runs 0 --measure-runs 1 \
        --json-out "$dbt_json" \
        2>&1 | tee "$out/${slug}-t128-dbt.log"

  python /home/kunwar/resnet_pytorch/qwen_suite.py compare \
    --native "$native_json" --dbt "$dbt_json" \
    > "$out/${slug}-t128-compare.txt" || true
done <<'EOF'
qwen25-05b Qwen/Qwen2.5-0.5B
qwen35-9b Qwen/Qwen3.5-9B
qwen35-27b Qwen/Qwen3.5-27B
EOF
```

The saved focused single-prompt 128-token artifacts on `sharkmi300x2` are also
available as historical run outputs:

- `Qwen/Qwen2.5-0.5B`: native
  `/tmp/rocjitsu-qwen-results/qwen25-05b-t128-native-env.json`, DBT
  `/tmp/rocjitsu-qwen-results/qwen25-05b-t128-dbt-blaslt-host-default.json`,
  compare `/tmp/rocjitsu-qwen-results/qwen25-05b-t128-blaslt-host-default-compare.log`
  (`PASS`; DBT median `9.261s`, `13.8 tok/s`).
- `Qwen/Qwen3.5-9B`: native
  `/tmp/rocjitsu-qwen-results/qwen35-9b-t128-native-env.json`, DBT
  `/tmp/rocjitsu-qwen-results/qwen35-9b-t128-dbt-host-default.json`, compare
  `/tmp/rocjitsu-qwen-results/qwen35-9b-t128-host-default-compare.log`
  (`PASS`; DBT median `15.893s`, `8.1 tok/s`).
- `Qwen/Qwen3.5-27B`: native
  `/tmp/rocjitsu-qwen-results/qwen35-27b-t128-native-env.json`, DBT
  `/tmp/rocjitsu-qwen-results/qwen35-27b-t128-dbt-host-default.json`, compare
  `/tmp/rocjitsu-qwen-results/qwen35-27b-t128-host-default-compare.log`
  (`PASS`; DBT median `21.640s`, `5.9 tok/s`).

There is also one earlier intermediate Qwen2.5-0.5B 128-token artifact at
`47283a0fa3`: `/tmp/qwen-suite-qwen25-t128-47283-native.json`,
`/tmp/qwen-suite-qwen25-t128-47283-dbt.json`, and
`/tmp/qwen-suite-qwen25-t128-47283-compare.txt`. That run completed on both
native and DBT; strict generated-text comparison failed, but the forward logits
argmax still matched (`15695`).
