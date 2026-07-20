# ConSan fuzz targets

Configure a separate Clang build so sanitizer and libFuzzer flags never alter
the normal developer build:

```sh
cmake -S emulation/rocjitsu -B /tmp/rocjitsu-consan-fuzz -G Ninja \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ \
  -DRJ_ENABLE_ASAN=ON -DRJ_ENABLE_UBSAN=ON \
  -DRJ_BUILD_FUZZERS=ON -DLTO=OFF
ninja -C /tmp/rocjitsu-consan-fuzz -j4 \
  consan_transform_fuzz consan_placement_fuzz
```

`consan_transform_fuzz` is a build umbrella for four binaries:
`consan_transform_supercollider_fuzz`,
`consan_transform_record_replay_fuzz`,
`consan_transform_inline_shadow_fuzz`, and
`consan_transform_sampled_fuzz`. Each sends every input through one public
ConSan profile and asserts the same transactional install contract as the
loader. Run the full budget against every binary; one libFuzzer callback is one
public transform deadline. Seed all four with the same retained AMDGPU code
objects and prior malformed inputs. Run large production objects as fixed
replay inputs; use a compact valid gfx1201 object with a real instrumentable
site plus minimized malformed regressions as the evolving mutation corpus.
This preserves exact large-object coverage without allowing near-duplicate
half-megabyte variants to consume the entire event budget.
`consan_placement_fuzz` interprets bounded records as overlapping inline,
local-cave, and appended-cave placement requests.

Example reproducible event-budget campaign:

```sh
mkdir -p /tmp/consan-transform-replay /tmp/consan-transform-mutation \
  /tmp/consan-placement-corpus \
  /tmp/consan-transform-artifacts /tmp/consan-placement-artifacts
cp retained-production-code-objects/*.hsaco /tmp/consan-transform-replay/
cp compact-padded-lds-store.hsaco minimized-regressions/* \
  /tmp/consan-transform-mutation/
for profile in supercollider record_replay inline_shadow sampled; do
  mkdir -p "/tmp/consan-transform-artifacts/${profile}"
  /tmp/rocjitsu-consan-fuzz/tests/consan_transform_${profile}_fuzz \
    /tmp/consan-transform-replay/* -runs=1 -timeout=10
  /tmp/rocjitsu-consan-fuzz/tests/consan_transform_${profile}_fuzz \
    /tmp/consan-transform-mutation -runs=10000 -max_len=1048576 -timeout=10 \
    -artifact_prefix="/tmp/consan-transform-artifacts/${profile}/"
done
/tmp/rocjitsu-consan-fuzz/tests/consan_placement_fuzz \
  /tmp/consan-placement-corpus -runs=100000 -max_len=4096 -timeout=10 \
  -artifact_prefix=/tmp/consan-placement-artifacts/
```

Always retain the source revision, CMake cache, command line, corpus hashes,
final libFuzzer statistics, and any crash artifact. A campaign is evidence for
the exact event budget and corpus only, not proof that malformed inputs are
exhausted.
