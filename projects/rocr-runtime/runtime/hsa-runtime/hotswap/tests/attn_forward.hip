#include <hip/hip_runtime.h>
// Fused single-head scaled dot-product attention.
// grid=(M,1,1), block=(32,1,1), shared_mem=(N+32)*sizeof(float)
//
// smem layout: [0..N-1] = dot products S, [N..N+31] = reduction scratch
// Constrains N to fit in shared memory: N*4 + 32*4 bytes total.
//
// Phase 1: S[n] = dot(Q[m,:], K[n,:]) / sqrt(D)  for n in [0,N)
// Phase 2: softmax S in-place (max-subtract, exp, normalize) using smem[N..]
// Phase 3: O[m,d] = sum_n S[n] * V[n,d]  for d = tid stride 32
extern "C" __global__ void attn_forward(
    float* __restrict__ O,
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    int M, int N, int D)
{
  extern __shared__ float smem[];  // N + 32 floats
  float* S    = smem;          // dot products [0..N-1]
  float* scratch = smem + N;   // reduction scratch [N..N+31]

  int m   = blockIdx.x;
  int tid = threadIdx.x;
  if (m >= M) return;

  const float* Qm = Q + m * D;
  float scale = 1.0f / sqrtf((float)D);

  // Phase 1: dot products Q[m,:] · K[n,:] / sqrt(D)
  for (int n = 0; n < N; n++) {
    const float* Kn = K + n * D;
    float dot = 0.0f;
    for (int d = tid; d < D; d += blockDim.x)
      dot += Qm[d] * Kn[d];
    // Warp reduce dot via scratch
    scratch[tid] = dot;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
      if (tid < s)
        scratch[tid] += scratch[tid + s];
      __syncthreads();
    }
    if (tid == 0)
      S[n] = scratch[0] * scale;
    __syncthreads();
  }

  // Phase 2: softmax over S
  // max reduction
  float thread_max = -1e30f;
  for (int n = tid; n < N; n += blockDim.x)
    thread_max = fmaxf(thread_max, S[n]);
  scratch[tid] = thread_max;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      scratch[tid] = fmaxf(scratch[tid], scratch[tid + s]);
    __syncthreads();
  }
  float s_max = scratch[0];
  __syncthreads();

  // exp and sum
  float thread_sum = 0.0f;
  for (int n = tid; n < N; n += blockDim.x) {
    S[n] = __expf(S[n] - s_max);
    thread_sum += S[n];
  }
  scratch[tid] = thread_sum;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s)
      scratch[tid] += scratch[tid + s];
    __syncthreads();
  }
  float inv_sum = 1.0f / scratch[0];
  __syncthreads();

  for (int n = tid; n < N; n += blockDim.x)
    S[n] *= inv_sum;
  __syncthreads();

  // Phase 3: O[m,d] = sum_n P[n] * V[n,d]
  float* Om = O + m * D;
  for (int d = tid; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int n = 0; n < N; n++)
      acc += S[n] * V[n * D + d];
    Om[d] = acc;
  }
}
