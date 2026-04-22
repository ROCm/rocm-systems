// Source for tdm_smoke_gfx1250.hsaco — gfx1250 TDM (Tensor Data
// Mover) regression fixture for the `VIMAGE` TENSOR cross-target
// lift.
//
// This kernel uses `tensor_load_to_lds` to copy a 1-D region of N
// uint32 elements from global memory into LDS, then writes the
// LDS contents back to global via plain stores. Dispatched on the
// source ISA this runs on the gfx1250 TENSORcnt unit; dispatched
// on gfx942 it runs through the Salmon-linked emulation helper
// (`runtime/tdm.hip` -> `salmon_tdm_load_to_lds`).
//
// Two things use this fixture:
//
//   1. `tests/tdm_gpu_test.cpp::TdmGpu.CrossTargetCorpus` — raises
//      the kernel to gfx942, asserts the raised IR references
//      `salmon_tdm_load_to_lds`, and smoke-tests that the merged
//      module survives codegen (`hipModuleLoadData`).
//
//   2. Ad-hoc correctness harness (see `~/tdm_artifacts/` on the
//      author's machine or reproduce per the block comment below):
//      host builds a D# pointing at a device-resident input buffer,
//      runs the raised gfx942 binary, and compares out == in. A
//      byte-for-byte match proves the emulation actually walked the
//      descriptor, issued the MUBUF loads, and staged the data into
//      LDS on real gfx942 hardware.
//
// D# layout the host populates (pointer-to-struct as the first two
// kernargs):
//   g0[0] = count=1                          = 0x0000_0001
//   g0[1] = lds_byte_addr                    = 0
//   g0[2] = global_addr low32
//   g0[3] = global_addr high (bits 24:0)
//   g1[0] = data_size_log2 = 2               = 0x0002_0000
//   g1[1] = tensor_dim0 low16 << 16          = N << 16
//   g1[2] = 0
//   g1[3] = tile_dim0 << 16                  = N << 16
//   g1[4] = 0
//   g1[5] = tensor_dim0_stride low32         = 1
//   g1[6..7] = 0
//
// Build / regenerate:
//
//   DD=projects/rocr-runtime/runtime/hsa-runtime/hotswap/test_data/gfx1250
//   hipcc --offload-arch=gfx1250 --genco \
//     "$DD/tdm_smoke_kernel.hip" \
//     -o "$DD/tdm_smoke_gfx1250.hsaco.bundle"
//   /opt/rocm-7.2.1/lib/llvm/bin/clang-offload-bundler --type=o \
//     --targets=hipv4-amdgcn-amd-amdhsa--gfx1250 \
//     --input="$DD/tdm_smoke_gfx1250.hsaco.bundle" \
//     --output="$DD/tdm_smoke_gfx1250.hsaco" --unbundle
//   rm -f "$DD/tdm_smoke_gfx1250.hsaco.bundle"
//
//   # ROCm 7.2.1 clang and upstream LLVM-23 disagree on where the
//   # 0x7C NULL sentinel sits in the VIMAGE TENSOR 12-byte encoding
//   # (byte 4 = `vdata` vs. byte 7 = `vaddr4`). Salmon links against
//   # an LLVM-23 build tree, so we post-patch the emitted encoding
//   # so its bundled disassembler accepts it. This is a binary
//   # no-op at execution time on gfx1250 hardware: the instruction
//   # has no vdata output and both encodings read the same D# SGPRs.
//   # Same workaround the `lit_tests/tensor_load_to_lds/` fixture
//   # applies by hand-encoding from scratch.
//   python3 - <<'PY'
//   with open("$DD/tdm_smoke_gfx1250.hsaco", "r+b") as f:
//       data = bytearray(f.read())
//   # Opcode dword of tensor_load_to_lds is `01 xx 71 d0` (LE);
//   # scan the .text section and move the NULL sentinel from the
//   # vdata slot (byte+4) into the vaddr4 slot (byte+7).
//   for i in range(len(data) - 12):
//       if data[i] == 0x01 and data[i+2] == 0x71 and data[i+3] == 0xd0:
//           if data[i+7] == 0x00:
//               data[i+7] = 0x7c
//   with open("$DD/tdm_smoke_gfx1250.hsaco", "wb") as f:
//       f.write(bytes(data))
//   PY
//
//   # Verify the result disassembles as tensor_load_to_lds:
//   llvm-objdump -d --mcpu=gfx1250 "$DD/tdm_smoke_gfx1250.hsaco" \
//     | grep tensor_load_to_lds

#include <hip/hip_runtime.h>
#include <cstdint>

typedef int v4i __attribute__((ext_vector_type(4)));
typedef int v8i __attribute__((ext_vector_type(8)));

extern "C" __global__ __launch_bounds__(64) void
tdm_smoke_kernel(const v4i *__restrict__ g0_ptr,
                 const v8i *__restrict__ g1_ptr,
                 uint32_t *__restrict__ out) {
  __shared__ uint32_t lds[64];
  int tid = threadIdx.x;

  // Pre-fill LDS with a sentinel so an accidentally-skipped TDM
  // emulation shows 0xdeadbeef in the output (distinguishable
  // from a zero-fill on HW OOB rules).
  if (tid < 64) lds[tid] = 0xdeadbeefu;
  __syncthreads();

  v4i g0 = *g0_ptr;
  v8i g1 = *g1_ptr;
  v4i zero4 = {0, 0, 0, 0};
  // ROCm 7.2.1 / AMD clang 22 signature (no reserved `grp4_reserved`
  // argument). Upstream LLVM 23 adds a <8 x i32> at argument 4; when
  // this fixture is regenerated against a newer toolchain, add
  //   v8i zero8 = {0,0,0,0,0,0,0,0};
  //   ... g0, g1, zero4, zero4, zero8, 0);
  __builtin_amdgcn_tensor_load_to_lds(g0, g1, zero4, zero4, 0);
  __builtin_amdgcn_s_wait_tensorcnt(0);
  __syncthreads();

  out[tid] = lds[tid];
}
