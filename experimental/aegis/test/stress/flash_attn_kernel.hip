#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cfloat>

// Standalone flash attention kernel modeled after llama.cpp's fattn-mma.
// Designed to stress AegisBit with:
//   - Large .text section (tiled Q*K^T -> softmax -> *V fused in one kernel)
//   - Heavy VMEM traffic (Q/K/V loads from global, output stores)
//   - LDS for tile staging (Q, K, V tiles loaded through shared memory)
//   - High register pressure (accumulator tiles + softmax running stats)
//   - Complex control flow (K/V tile loop, online softmax renormalization)
//
// Parameters:
//   HEAD_DIM = 128 (Llama-style)
//   BLOCK_M  = 64  (query tile rows)
//   BLOCK_N  = 64  (KV tile cols)
//   Threads  = 256

#define HEAD_DIM  128
#define BLOCK_M   64
#define BLOCK_N   64
#define BLOCK_K   HEAD_DIM
#define NTHREADS  256
#define NWARPS    (NTHREADS / 64)

// Each thread handles BLOCK_M/NWARPS rows of Q, accumulating over K/V tiles.
// With 4 warps and 64 rows, each warp handles 16 rows.
#define ROWS_PER_WARP  (BLOCK_M / NWARPS)

// LDS layout: Q tile + K tile + V tile
// Q: [BLOCK_M][HEAD_DIM] floats
// K: [BLOCK_N][HEAD_DIM] floats
// V: [BLOCK_N][HEAD_DIM] floats
#define LDS_Q_SIZE  (BLOCK_M * HEAD_DIM)
#define LDS_K_SIZE  (BLOCK_N * HEAD_DIM)
#define LDS_V_SIZE  (BLOCK_N * HEAD_DIM)
#define LDS_TOTAL   (LDS_Q_SIZE + LDS_K_SIZE + LDS_V_SIZE)

__global__ __attribute__((amdgpu_flat_work_group_size(NTHREADS, NTHREADS)))
void flash_attn_fwd(const float* __restrict__ Q,   // [batch, num_heads, seq_len, head_dim]
                    const float* __restrict__ K,   // [batch, num_heads, seq_len, head_dim]
                    const float* __restrict__ V,   // [batch, num_heads, seq_len, head_dim]
                    float* __restrict__ O,         // [batch, num_heads, seq_len, head_dim]
                    int seq_len, int num_heads, float scale) {
    // Block assignment: blockIdx.x -> query tile, blockIdx.y -> batch*head
    int q_tile_idx = blockIdx.x;
    int bh = blockIdx.y;
    int batch_idx = bh / num_heads;
    int head_idx  = bh % num_heads;

    int q_start = q_tile_idx * BLOCK_M;

    // Per-head pointer offsets
    int head_offset = ((batch_idx * num_heads) + head_idx) * seq_len * HEAD_DIM;
    const float* q_ptr = Q + head_offset + q_start * HEAD_DIM;
    const float* k_base = K + head_offset;
    const float* v_base = V + head_offset;
    float* o_ptr = O + head_offset + q_start * HEAD_DIM;

    int tid = threadIdx.x;
    int warp_id = tid / 64;
    int lane_id = tid % 64;

    // LDS allocation
    __shared__ float smem[LDS_TOTAL];
    float* s_Q = smem;
    float* s_K = smem + LDS_Q_SIZE;
    float* s_V = smem + LDS_Q_SIZE + LDS_K_SIZE;

    // ---- Load Q tile into LDS ----
    // Each thread loads multiple elements: BLOCK_M * HEAD_DIM / NTHREADS elements
    #pragma unroll 4
    for (int i = tid; i < BLOCK_M * HEAD_DIM; i += NTHREADS) {
        int row = i / HEAD_DIM;
        int col = i % HEAD_DIM;
        int global_row = q_start + row;
        if (global_row < seq_len)
            s_Q[row * HEAD_DIM + col] = q_ptr[row * HEAD_DIM + col] * scale;
        else
            s_Q[row * HEAD_DIM + col] = 0.0f;
    }
    __syncthreads();

    // Per-thread accumulators for output: each thread handles a subset of rows x head_dim
    // Thread layout: each warp handles ROWS_PER_WARP rows, lane handles HEAD_DIM/64 cols
    int my_row_start = warp_id * ROWS_PER_WARP;
    int my_col_start = lane_id * 2;  // each lane handles 2 columns

    // Online softmax: running max and sum-of-exp per row
    float row_max[ROWS_PER_WARP];
    float row_sum[ROWS_PER_WARP];
    float acc[ROWS_PER_WARP][2];  // output accumulator, 2 cols per lane

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        row_max[r] = -FLT_MAX;
        row_sum[r] = 0.0f;
        acc[r][0] = 0.0f;
        acc[r][1] = 0.0f;
    }

    // ---- KV tile loop (outer loop over sequence) ----
    for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_N) {

        // Load K tile into LDS
        #pragma unroll 4
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += NTHREADS) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = kv_start + row;
            if (global_row < seq_len)
                s_K[row * HEAD_DIM + col] = k_base[global_row * HEAD_DIM + col];
            else
                s_K[row * HEAD_DIM + col] = 0.0f;
        }

        // Load V tile into LDS
        #pragma unroll 4
        for (int i = tid; i < BLOCK_N * HEAD_DIM; i += NTHREADS) {
            int row = i / HEAD_DIM;
            int col = i % HEAD_DIM;
            int global_row = kv_start + row;
            if (global_row < seq_len)
                s_V[row * HEAD_DIM + col] = v_base[global_row * HEAD_DIM + col];
            else
                s_V[row * HEAD_DIM + col] = 0.0f;
        }
        __syncthreads();

        // ---- Compute S = Q * K^T for this tile ----
        // Each thread computes dot products for its assigned rows against all BLOCK_N key rows.
        // This is the heavy part: ROWS_PER_WARP * BLOCK_N dot products, each over HEAD_DIM.
        float s_vals[ROWS_PER_WARP][BLOCK_N];

        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            int qr = my_row_start + r;
            #pragma unroll
            for (int kc = 0; kc < BLOCK_N; kc++) {
                float dot = 0.0f;
                #pragma unroll 8
                for (int d = 0; d < HEAD_DIM; d++) {
                    dot += s_Q[qr * HEAD_DIM + d] * s_K[kc * HEAD_DIM + d];
                }
                s_vals[r][kc] = dot;
            }
        }

        // ---- Online softmax update ----
        // For each row, find new max, rescale old accumulator, compute exp, accumulate
        #pragma unroll
        for (int r = 0; r < ROWS_PER_WARP; r++) {
            // Find max of this tile's scores
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int kc = 0; kc < BLOCK_N; kc++) {
                int global_kv = kv_start + kc;
                float val = (global_kv < seq_len) ? s_vals[r][kc] : -FLT_MAX;
                if (val > tile_max) tile_max = val;
            }

            float old_max = row_max[r];
            float new_max = (tile_max > old_max) ? tile_max : old_max;

            // Rescale old accumulator
            float rescale = expf(old_max - new_max);
            row_sum[r] *= rescale;
            acc[r][0] *= rescale;
            acc[r][1] *= rescale;

            // Compute exp(s - new_max) and accumulate V
            float tile_sum = 0.0f;
            #pragma unroll
            for (int kc = 0; kc < BLOCK_N; kc++) {
                int global_kv = kv_start + kc;
                float p = (global_kv < seq_len) ? expf(s_vals[r][kc] - new_max) : 0.0f;
                tile_sum += p;

                // Weighted V accumulation (2 output columns per lane)
                if (my_col_start < HEAD_DIM)
                    acc[r][0] += p * s_V[kc * HEAD_DIM + my_col_start];
                if (my_col_start + 1 < HEAD_DIM)
                    acc[r][1] += p * s_V[kc * HEAD_DIM + my_col_start + 1];
            }

            row_sum[r] += tile_sum;
            row_max[r] = new_max;
        }

        __syncthreads();
    }

    // ---- Finalize: divide by sum and write output ----
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; r++) {
        int global_row = q_start + my_row_start + r;
        if (global_row >= seq_len) continue;

        float inv_sum = (row_sum[r] > 0.0f) ? (1.0f / row_sum[r]) : 0.0f;

        if (my_col_start < HEAD_DIM)
            o_ptr[(my_row_start + r) * HEAD_DIM + my_col_start] = acc[r][0] * inv_sum;
        if (my_col_start + 1 < HEAD_DIM)
            o_ptr[(my_row_start + r) * HEAD_DIM + my_col_start + 1] = acc[r][1] * inv_sum;
    }
}

// Correctness check: compute reference attention for a single query row
void reference_attention_row(const float* Q, const float* K, const float* V,
                             float* out, int seq_len, float scale, int row) {
    float scores[8192];
    float max_s = -FLT_MAX;
    for (int j = 0; j < seq_len; j++) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; d++)
            dot += Q[row * HEAD_DIM + d] * K[j * HEAD_DIM + d];
        scores[j] = dot * scale;
        if (scores[j] > max_s) max_s = scores[j];
    }
    float sum = 0.0f;
    for (int j = 0; j < seq_len; j++) {
        scores[j] = expf(scores[j] - max_s);
        sum += scores[j];
    }
    for (int d = 0; d < HEAD_DIM; d++) {
        float val = 0.0f;
        for (int j = 0; j < seq_len; j++)
            val += scores[j] * V[j * HEAD_DIM + d];
        out[d] = val / sum;
    }
}

int main() {
    // Small enough to verify correctness, big enough to be a real workload
    int batch = 1;
    int num_heads = 1;
    int seq_len = 512;
    float scale = 1.0f / sqrtf((float)HEAD_DIM);

    size_t qkv_bytes = (size_t)batch * num_heads * seq_len * HEAD_DIM * sizeof(float);

    // Host allocations
    float* hQ = (float*)malloc(qkv_bytes);
    float* hK = (float*)malloc(qkv_bytes);
    float* hV = (float*)malloc(qkv_bytes);
    float* hO = (float*)malloc(qkv_bytes);

    srand(42);
    for (size_t i = 0; i < (size_t)batch * num_heads * seq_len * HEAD_DIM; i++) {
        hQ[i] = ((float)(rand() % 200) - 100.0f) / 1000.0f;
        hK[i] = ((float)(rand() % 200) - 100.0f) / 1000.0f;
        hV[i] = ((float)(rand() % 200) - 100.0f) / 1000.0f;
    }

    // Device allocations
    float *dQ, *dK, *dV, *dO;
    hipMalloc(&dQ, qkv_bytes);
    hipMalloc(&dK, qkv_bytes);
    hipMalloc(&dV, qkv_bytes);
    hipMalloc(&dO, qkv_bytes);
    hipMemset(dO, 0, qkv_bytes);

    hipMemcpy(dQ, hQ, qkv_bytes, hipMemcpyHostToDevice);
    hipMemcpy(dK, hK, qkv_bytes, hipMemcpyHostToDevice);
    hipMemcpy(dV, hV, qkv_bytes, hipMemcpyHostToDevice);

    // Launch
    int num_q_tiles = (seq_len + BLOCK_M - 1) / BLOCK_M;
    dim3 grid(num_q_tiles, batch * num_heads);
    dim3 block(NTHREADS);

    printf("Launching flash_attn_fwd: seq=%d, heads=%d, head_dim=%d, grid=(%d,%d), block=%d\n",
           seq_len, num_heads, HEAD_DIM, num_q_tiles, batch * num_heads, NTHREADS);

    flash_attn_fwd<<<grid, block>>>(dQ, dK, dV, dO, seq_len, num_heads, scale);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("FAIL: GPU error: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(hO, dO, qkv_bytes, hipMemcpyDeviceToHost);

    // Correctness check: compare a few rows against CPU reference
    float ref_row[HEAD_DIM];
    int rows_to_check[] = {0, 1, seq_len / 2, seq_len - 1};
    float max_err = 0.0f;
    for (int ri = 0; ri < 4; ri++) {
        int row = rows_to_check[ri];
        reference_attention_row(hQ, hK, hV, ref_row, seq_len, scale, row);
        for (int d = 0; d < HEAD_DIM; d++) {
            float diff = fabsf(hO[row * HEAD_DIM + d] - ref_row[d]);
            float denom = fabsf(ref_row[d]) + 1e-6f;
            float rel = diff / denom;
            if (rel > max_err) max_err = rel;
        }
    }

    printf("CHECKSUM: max_relative_error=%.6f\n", max_err);
    if (max_err < 0.05f) {
        printf("PASS\n");
    } else {
        printf("FAIL: max relative error %.6f exceeds tolerance 0.05\n", max_err);
    }

    hipFree(dQ); hipFree(dK); hipFree(dV); hipFree(dO);
    free(hQ); free(hK); free(hV); free(hO);
    return (max_err < 0.05f) ? 0 : 1;
}
