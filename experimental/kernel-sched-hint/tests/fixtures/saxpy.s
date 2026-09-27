.text
.globl saxpy
.p2align 8
.type saxpy,@function
saxpy:
  s_waitcnt lgkmcnt(0)
  v_lshlrev_b32_e32 v0, 2, v0
  global_load_dword v1, v0, s[0:1]
  global_load_dword v2, v0, s[2:3]
  s_waitcnt vmcnt(0)
  v_fma_f32 v1, v1, s4, v2
  global_store_dword v0, v1, s[2:3]
  s_endpgm

.amdgpu_metadata
---
amdhsa.kernels:
  - .name: saxpy
    .sgpr_count: 8
    .vgpr_count: 4
    .wavefront_size: 64
    .group_segment_fixed_size: 0
    .private_segment_fixed_size: 0
    .kernarg_segment_size: 24
    .kind: normal
    .args:
      - .name: x
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .access: read_only
      - .name: y
        .size: 8
        .value_kind: global_buffer
        .address_space: global
        .access: read_write
      - .name: a
        .size: 4
        .value_kind: by_value
      - .size: 8
        .value_kind: hidden_global_offset_x
...
.end_amdgpu_metadata
