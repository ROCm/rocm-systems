#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

typedef _Float16 half_t;
typedef half_t v4h __attribute__((ext_vector_type(4)));
typedef float v4f __attribute__((ext_vector_type(4)));
typedef float v16f __attribute__((ext_vector_type(16)));

// ============================================================================
// Minimal MFMA 16x16 GEMM: one wavefront computes one 16x16 output tile.
// C (f32, 16x16) += A (f16, 16xK) * B (f16, Kx16), K iterated in steps of 16.
//
// Grid: (ceil(N/16), ceil(M/16), 1) workgroups of 64 threads each.
//
// v_mfma_f32_16x16x16_f16 lane mapping (gfx942):
//   64 threads = 4 groups of 16
//   group = tid / 16, lane = tid % 16
//   A: srcA[i] = A[lane][group*4 + i]     (row=lane, K_col=group*4+i)
//   B: srcB[i] = B[group*4 + i][lane]     (K_row=group*4+i, col=lane)
//   C: accum[i] = C[lane][group*4 + i]    (row=lane, col=group*4+i)
// ============================================================================
extern "C" __global__ void mfma_gemm_16x16(
    float* __restrict__ C,
    const half_t* __restrict__ A,
    const half_t* __restrict__ B,
    int M, int N, int K)
{
  int tile_row = blockIdx.y * 16;
  int tile_col = blockIdx.x * 16;
  int tid = threadIdx.x; // 0..63

  int lane = tid % 16;
  int group = tid / 16; // 0..3

  v4f acc = {0.0f, 0.0f, 0.0f, 0.0f};

  for (int kk = 0; kk < K; kk += 16) {
    // A: row = tile_row + lane, cols in K = kk + group*4 + {0,1,2,3}
    int a_row = tile_row + lane;
    int a_base_k = kk + group * 4;
    v4h a_frag;
    if (a_row < M && a_base_k + 3 < K) {
      a_frag[0] = A[a_row * K + a_base_k + 0];
      a_frag[1] = A[a_row * K + a_base_k + 1];
      a_frag[2] = A[a_row * K + a_base_k + 2];
      a_frag[3] = A[a_row * K + a_base_k + 3];
    } else {
      a_frag = (v4h){(half_t)0, (half_t)0, (half_t)0, (half_t)0};
    }

    // B: rows in K = kk + group*4 + {0,1,2,3}, col = tile_col + lane
    int b_base_k = kk + group * 4;
    int b_col = tile_col + lane;
    v4h b_frag;
    if (b_base_k + 3 < K && b_col < N) {
      b_frag[0] = B[(b_base_k + 0) * N + b_col];
      b_frag[1] = B[(b_base_k + 1) * N + b_col];
      b_frag[2] = B[(b_base_k + 2) * N + b_col];
      b_frag[3] = B[(b_base_k + 3) * N + b_col];
    } else {
      b_frag = (v4h){(half_t)0, (half_t)0, (half_t)0, (half_t)0};
    }

    acc = __builtin_amdgcn_mfma_f32_16x16x16f16(a_frag, b_frag, acc, 0, 0, 0);
  }

  // Store: row = tile_row + lane, cols = tile_col + group*4 + {0,1,2,3}
  int out_row = tile_row + lane;
  int out_base_col = tile_col + group * 4;
  if (out_row < M) {
    if (out_base_col + 0 < N) C[out_row * N + out_base_col + 0] = acc[0];
    if (out_base_col + 1 < N) C[out_row * N + out_base_col + 1] = acc[1];
    if (out_base_col + 2 < N) C[out_row * N + out_base_col + 2] = acc[2];
    if (out_base_col + 3 < N) C[out_row * N + out_base_col + 3] = acc[3];
  }
}
