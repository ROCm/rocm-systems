# HotSwap test fixtures

## gfx1250_min.hsaco

A minimal gfx1250 code object (empty kernel) used by `hotswap_test` to exercise
the real parse -> ISA-derivation -> rewrite path. It carries a valid
`NT_AMDGPU_METADATA` note (`amdhsa.target: amdgcn-amd-amdhsa--gfx1250`,
`.gfx1250_revision: B0`), which is what `amd_comgr_get_data_isa_name` reads.

Regenerate with the ROCm clang:

```bash
echo 'kernel void k(){}' > k.cl
clang -target amdgcn-amd-amdhsa -mcpu=gfx1250 -mcode-object-version=5 \
      -nogpulib -x cl -cl-std=CL2.0 -O2 k.cl -o gfx1250_min.hsaco
```
