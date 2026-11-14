/******************************************************************************
 * Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
 *
 * SPDX-License-Identifier: MIT
 *****************************************************************************/

/**
 * @file tile_ck_gemm.cc
 * @brief Distributed GEMM using rocSHMEM Tile API + AMD Composable Kernels (CK Tile)
 *
 * Demonstrates using the rocSHMEM tile broadcast API together with CK Tile's
 * GemmKernel for high-performance local matrix multiplication.
 *
 * Algorithm: 2D SUMMA (Scalable Universal Matrix Multiply Algorithm)
 *
 *   Problem: C(M×N) = A(M×K) × B(K×N) distributed across a Pr×Pc PE grid.
 *
 *   Distribution:
 *     A(M×K)     : row-block partitioned; PE(pr,pc) owns A[ pr*M_local : (pr+1)*M_local , 0:K ]
 *     B(K×N)     : col-block partitioned; PE(pr,pc) owns B[ 0:K , pc*N_local : (pc+1)*N_local ]
 *     C(M×N)     : 2D block partitioned; PE(pr,pc) owns C[ pr*M_local:(pr+1)*M_local ,
 *                                                           pc*N_local:(pc+1)*N_local ]
 *
 *   SUMMA loop (Pc steps):
 *     for step = 0 .. Pc-1:
 *       1. PE(pr, step) broadcasts its A slab [M_local × K_panel] along its row team
 *       2. PE(step, pc) broadcasts its B slab [K_panel × N_local] along its column team
 *       3. All PEs compute: C_local += A_panel × B_panel  (via CK Tile GEMM)
 *
 *   where K_panel = K / Pc  (each SUMMA step covers K_panel columns of K).
 *
 * Identified API gaps (see inline NOTE comments):
 *   1. No non-blocking tile broadcast (rocshmem_tile_broadcast_nbi_wg absent) prevents
 *      overlapping communication with computation.
 *   2. Tile reductions only support float/double/integer types; fp16/bf16/fp8 are missing.
 *   3. Tile API supports 1D and 2D tensors only; batched/3D GEMM cannot be expressed.
 *
 * Build requirements:
 *   - ROCm with HIP
 *   - rocSHMEM
 *   - AMD Composable Kernels (composablekernel, linked via CMake)
 *   - GPU target: gfx94x, gfx95x, gfx90a, or gfx11x/gfx12x
 *
 * Usage:
 *   mpirun -n <P^2> ./tile_ck_gemm [M] [N] [K]
 *   where P is an integer (number of PEs = P^2, e.g. 1, 4, 9, 16).
 *   Defaults: M=256, N=256, K=128
 */

#include <hip/hip_runtime.h>
#include <rocshmem/rocshmem.hpp>
#ifdef __HIP_DEVICE_COMPILE__
#include <rocshmem/rocshmem_device.hpp>
#endif

// CK Tile headers
#include "ck_tile/core.hpp"
#include "ck_tile/host/kernel_launch.hpp"
#include "ck_tile/ops/epilogue.hpp"
#include "ck_tile/ops/gemm.hpp"

#include <cmath>
#include <iostream>
#include <vector>

#include "util.h"

using namespace rocshmem;

/******************************************************************************
 * TENSOR TYPES — Compatible with rocSHMEM Tile API duck-type contract
 *
 * The Tile API requires tensor types to expose:
 *   element_type   : scalar element type alias
 *   ndim           : static constexpr int dimensionality
 *   data_handle()  : returns raw pointer to first element
 *   stride(int)    : returns stride in elements for dimension dim
 *****************************************************************************/

template <typename T>
struct Tensor2D {
    using element_type = T;
    static constexpr int ndim = 2;

    T*  data;
    int rows;
    int cols;
    int row_stride;
    int col_stride;

    __host__ __device__ Tensor2D(T* data_, int rows_, int cols_,
                                  int row_stride_ = -1, int col_stride_ = 1)
        : data(data_), rows(rows_), cols(cols_), col_stride(col_stride_)
    {
        row_stride = (row_stride_ == -1) ? cols : row_stride_;
    }

    __host__ __device__ T*  data_handle() const { return data; }
    __host__ __device__ int stride(int dim) const { return (dim == 0) ? row_stride : col_stride; }

    __host__ __device__ T&       operator()(int i, int j)       { return data[i * row_stride + j * col_stride]; }
    __host__ __device__ const T& operator()(int i, int j) const { return data[i * row_stride + j * col_stride]; }
};

struct Coord2D {
    int row, col;
    __host__ __device__ Coord2D(int r, int c) : row(r), col(c) {}
    __host__ __device__ int get(int dim) const { return (dim == 0) ? row : col; }
};

/******************************************************************************
 * CK TILE KERNEL CHAIN — fp32 GEMM using BASIC_V1 pipeline
 *
 * Tile sizes for fp32 on gfx94x/gfx95x (non-WMMA path, mirroring BasicInvoker):
 *   BlockTile : M=128, N=128, K=64
 *   BlockWarps: M_Warp=4, N_Warp=4, K_Warp=1  → 16 warps × 64 threads = 1024 threads/block
 *   WarpTile  : M=16, N=16, K=16  (MFMA 16×16×16 instruction)
 *
 * NOTE: gap — CK Tile GEMM requires K to be a multiple of K_Tile (64 here). The example
 * enforces this in main() by requiring K to be divisible by (Pc * 64).
 *****************************************************************************/

using ADataType   = float;
using BDataType   = float;
using AccDataType = float;
using CDataType   = float;
using ALayout     = ck_tile::tensor_layout::gemm::RowMajor;
using BLayout     = ck_tile::tensor_layout::gemm::RowMajor;
using CLayout     = ck_tile::tensor_layout::gemm::RowMajor;

using CodegenGemmShape =
    ck_tile::TileGemmShape<ck_tile::sequence<128, 128, 64>,   // BlockTile (M, N, K)
                           ck_tile::sequence<4, 4, 1>,         // BlockWarps
                           ck_tile::sequence<16, 16, 16>>;     // WarpTile

using TilePartitioner = ck_tile::GemmTile1DPartitioner<CodegenGemmShape>;

using GemmTraits = ck_tile::TileGemmTraits</*kPadM=*/false, /*kPadN=*/false, /*kPadK=*/false,
                                           ALayout, BLayout, CLayout>;

using PipelineProblem =
    ck_tile::GemmPipelineProblem<ADataType, BDataType, AccDataType,
                                 CodegenGemmShape, GemmTraits>;

using GemmPipeline = ck_tile::GemmPipelineAGmemBGmemCRegV1<PipelineProblem>;

using GemmEpilogueProblem =
    ck_tile::CShuffleEpilogueProblem<ADataType,
                                     BDataType,
                                     ck_tile::tuple<>,        // no extra D tensors
                                     AccDataType,
                                     CDataType,
                                     ck_tile::tuple<>,        // no D layouts
                                     CLayout,
                                     ck_tile::element_wise::PassThrough,
                                     TilePartitioner::MPerBlock,
                                     TilePartitioner::NPerBlock,
                                     /*MWave=*/4,
                                     /*NWave=*/4,
                                     /*MPerXdl=*/16,
                                     /*NPerXdl=*/16,
                                     /*KPerXdl=*/16,
                                     /*isCTransposed=*/PipelineProblem::TransposeC>;

using GemmEpilogue = ck_tile::CShuffleEpilogue<GemmEpilogueProblem>;
using GemmKernel   = ck_tile::GemmKernel<TilePartitioner, GemmPipeline, GemmEpilogue>;

/******************************************************************************
 * DEVICE KERNELS
 *****************************************************************************/

/**
 * SUMMA broadcast kernel — runs two workgroups concurrently:
 *   WG 0: broadcasts this PE's A panel along the row team (if this PE is the root column)
 *   WG 1: broadcasts this PE's B panel along the column team (if this PE is the root row)
 *
 * After this kernel, all PEs in the same row have a copy of root_col_pe's A panel in
 * d_A_panel, and all PEs in the same column have a copy of root_row_pe's B panel in
 * d_B_panel.
 *
 * @param a_panel   Symmetric buffer receiving the broadcast A panel (M_local × K_panel)
 * @param b_panel   Symmetric buffer receiving the broadcast B panel (K_panel × N_local)
 * @param a_src     Source A data on the root column PE (M_local × K_panel submatrix of A_local)
 * @param b_src     Source B data on the root row PE (K_panel × N_local submatrix of B_local)
 * @param row_team  Team of all PEs in the same process-grid row (for A broadcast)
 * @param col_team  Team of all PEs in the same process-grid column (for B broadcast)
 * @param pc        Column index of this PE in the process grid
 * @param pr        Row index of this PE in the process grid
 * @param step      Current SUMMA step (0..Pc-1); root for A is col=step, root for B is row=step
 */
// summa_broadcast_kernel launches 2 workgroups; status[0] holds WG0's return
// value (A broadcast), status[1] holds WG1's return value (B broadcast).
// The host checks both before proceeding to GEMM.
__global__ void summa_broadcast_kernel(Tensor2D<float> a_panel,
                                        Tensor2D<float> b_panel,
                                        Tensor2D<float> a_src,
                                        Tensor2D<float> b_src,
                                        rocshmem_team_t row_team,
                                        rocshmem_team_t col_team,
                                        int pc, int pr, int step,
                                        int* status)
{
    __shared__ rocshmem_ctx_t ctx;
    rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);

    Coord2D start(0, 0);
    int rc;

    if (blockIdx.x == 0) {
        // Broadcast A panel along this PE's row team.
        // Root is the PE at column index `step` in this row.
        //
        // NOTE: gap — A non-blocking broadcast (rocshmem_tile_broadcast_nbi_wg) would allow
        // WG0 to post the A broadcast and immediately return while WG1 posts the B broadcast,
        // then both could be waited on together. Currently we must block until complete.
        Coord2D boundary(a_panel.rows, a_panel.cols);
        rc = rocshmem_ctx_tile_broadcast_wg(ctx, row_team, a_panel, a_src,
                                             start, boundary, /*pe_root=*/step, 0);
        if (threadIdx.x == 0) status[0] = rc;
    } else {
        // Broadcast B panel along this PE's column team.
        // Root is the PE at row index `step` in this column.
        Coord2D boundary(b_panel.rows, b_panel.cols);
        rc = rocshmem_ctx_tile_broadcast_wg(ctx, col_team, b_panel, b_src,
                                             start, boundary, /*pe_root=*/step, 0);
        if (threadIdx.x == 0) status[1] = rc;
    }

    rocshmem_wg_ctx_destroy(&ctx);
}

/**
 * Accumulate C_local += scratch element-wise.
 * Used after each SUMMA step to accumulate partial GEMM results.
 * Each thread handles one element.
 */
__global__ void accumulate_kernel(float* __restrict__ c_local,
                                   const float* __restrict__ scratch,
                                   int n_elems)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_elems)
        c_local[idx] += scratch[idx];
}

/**
 * Workgroup-collective integer sum-reduce across all PEs in ROCSHMEM_TEAM_WORLD.
 * dst[0] = sum of src[0] across all PEs.
 */
__global__ void int_sum_reduce_kernel(int* dst, int* src)
{
    __shared__ rocshmem_ctx_t ctx;
    rocshmem_wg_ctx_create(ROCSHMEM_CTX_WG_PRIVATE, &ctx);
    rocshmem_ctx_int_sum_reduce_wg(ctx, ROCSHMEM_TEAM_WORLD, dst, src, 1);
    rocshmem_wg_ctx_destroy(&ctx);
}

/******************************************************************************
 * HOST HELPERS
 *****************************************************************************/

/**
 * Launch the CK Tile GEMM kernel: out = A × B
 *
 * Uses CK Tile's make_kernel + launch_kernel so the kernel is wrapped in the
 * kentry trampoline that CK requires for proper occupancy hints.
 *
 * @param d_A      Device pointer to A (M×K, row-major, stride_A = K)
 * @param d_B      Device pointer to B (K×N, row-major, stride_B = N)
 * @param d_C      Device pointer to output C (M×N, row-major, stride_C = N)
 * @param M, N, K  Matrix dimensions
 */
static void run_ck_gemm(const float* d_A, const float* d_B, float* d_C,
                         int M, int N, int K)
{
    ck_tile::GemmHostArgs args;
    args.a_ptr    = d_A;
    args.b_ptr    = d_B;
    args.e_ptr    = d_C;
    args.k_batch  = 1;
    args.M        = M;
    args.N        = N;
    args.K        = K;
    args.stride_A = K;
    args.stride_B = N;
    args.stride_E = N;

    auto kargs = GemmKernel::MakeKernelArgs(args);
    if (!GemmKernel::IsSupportedArgument(kargs)) {
        std::cerr << "CK Tile GEMM: unsupported argument"
                  << " M=" << M << " N=" << N << " K=" << K << "\n";
        return;
    }

    const dim3 grids  = GemmKernel::GridSize(M, N, /*k_batch=*/1);
    const dim3 blocks = GemmKernel::BlockSize();

    // CK Tile kernels must be launched through make_kernel so that the kentry
    // trampoline sets the correct block-per-CU occupancy hint.
    ck_tile::launch_kernel(ck_tile::stream_config{},
        ck_tile::make_kernel</*kBlockPerCu=*/1>(GemmKernel{}, grids, blocks, 0, kargs));
    // Check for kernel launch errors immediately.
    CHECK_HIP(hipGetLastError());
}

/** CPU reference GEMM: C = A × B */
static void cpu_gemm(const std::vector<float>& A, const std::vector<float>& B,
                     std::vector<float>& C, int M, int N, int K)
{
    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++) {
            float acc = 0.f;
            for (int k = 0; k < K; k++)
                acc += A[i * K + k] * B[k * N + j];
            C[i * N + j] = acc;
        }
}

/******************************************************************************
 * MAIN
 *****************************************************************************/

int main(int argc, char* argv[])
{
    // rocSHMEM must be initialized before any HIP calls that use symmetric memory.
    // Set device from launcher local rank first.
    int local_rank = get_launcher_local_rank();
    if (local_rank >= 0)
        CHECK_HIP(hipSetDevice(local_rank));

    rocshmem_init();
    int my_pe = rocshmem_my_pe();
    int n_pes = rocshmem_n_pes();

    // Require a perfect-square number of PEs for the 2D process grid.
    // Using Pr = Pc = P ensures the SUMMA loop root indices (0..Pc-1) are
    // always valid within both the row team (size Pc) and column team (size Pr).
    int P  = static_cast<int>(std::round(std::sqrt(static_cast<double>(n_pes))));
    if (P * P != n_pes) {
        if (my_pe == 0)
            std::cerr << "n_pes (" << n_pes << ") must be a perfect square\n";
        rocshmem_finalize();
        return 1;
    }
    int Pr = P;
    int Pc = P;

    // This PE's coordinates in the Pr×Pc process grid.
    int pr = my_pe / Pc;
    int pc = my_pe % Pc;

    // Matrix dimensions (defaults produce tile-aligned values for Pc=1 and Pc=2).
    int M = (argc > 1) ? std::atoi(argv[1]) : 256;
    int N = (argc > 2) ? std::atoi(argv[2]) : 256;
    int K = (argc > 3) ? std::atoi(argv[3]) : 128;

    // Each SUMMA step processes K_panel = K/Pc columns of A and rows of B.
    // K must be divisible by Pc. K_panel must be a multiple of the CK Tile K_Tile (64).
    if (K % Pc != 0 || (K / Pc) % 64 != 0) {
        if (my_pe == 0)
            std::cerr << "K (" << K << ") must be divisible by Pc*64 = " << Pc * 64 << "\n";
        rocshmem_finalize();
        return 1;
    }
    if (M % Pr != 0 || N % Pc != 0) {
        if (my_pe == 0)
            std::cerr << "M must be divisible by Pr=" << Pr
                      << " and N by Pc=" << Pc << "\n";
        rocshmem_finalize();
        return 1;
    }

    int M_local = M / Pr;   // rows owned by this PE
    int N_local = N / Pc;   // cols owned by this PE
    int K_panel = K / Pc;   // columns of K handled per SUMMA step

    if (my_pe == 0) {
        std::cout << "2D SUMMA distributed GEMM  (CK Tile + rocSHMEM tile broadcast)\n";
        std::cout << "  C(" << M << "×" << N << ") = A(" << M << "×" << K
                  << ") × B(" << K << "×" << N << ")\n";
        std::cout << "  Process grid: " << Pr << "×" << Pc << " PEs\n";
        std::cout << "  Per-PE partition: A(" << M_local << "×" << K
                  << "), B(" << K << "×" << N_local
                  << "), C(" << M_local << "×" << N_local << ")\n";
        std::cout << "  SUMMA steps: " << Pc << ", K_panel=" << K_panel << "\n\n";
    }

    // -------------------------------------------------------------------------
    // Create 2D process-grid teams.
    //   xrange = Pc → row team has Pc members (same pr, varying pc)
    //   xaxis_team = row team, yaxis_team = column team
    //
    // rocshmem_team_split_2d signature:
    //   (parent, xrange, xconfig, xmask, &xteam, yconfig, ymask, &yteam)
    // -------------------------------------------------------------------------
    rocshmem_team_t row_team, col_team;
    int rc = rocshmem_team_split_2d(ROCSHMEM_TEAM_WORLD, Pc,
                                    nullptr, 0, &row_team,
                                    nullptr, 0, &col_team);
    if (rc != 0) {
        std::cerr << "PE " << my_pe << ": rocshmem_team_split_2d failed\n";
        rocshmem_finalize();
        return 1;
    }

    // -------------------------------------------------------------------------
    // Allocate symmetric memory.
    //
    // d_A_local : PE's row slab of A — M_local × K  (full K, all Pc panels concatenated)
    // d_B_local : PE's col slab of B — K × N_local  (full K)
    // d_C_local : PE's 2D block of C — M_local × N_local  (accumulated result)
    // d_A_panel : broadcast buffer for current SUMMA step A panel — M_local × K_panel
    // d_B_panel : broadcast buffer for current SUMMA step B panel — K_panel × N_local
    // d_scratch : GEMM output staging (CK Tile writes here; we then add to d_C_local)
    // -------------------------------------------------------------------------
    auto* d_A_local = static_cast<float*>(rocshmem_malloc(M_local * K         * sizeof(float)));
    auto* d_B_local = static_cast<float*>(rocshmem_malloc(K       * N_local   * sizeof(float)));
    auto* d_C_local = static_cast<float*>(rocshmem_malloc(M_local * N_local   * sizeof(float)));
    auto* d_A_panel = static_cast<float*>(rocshmem_malloc(M_local * K_panel   * sizeof(float)));
    auto* d_B_panel = static_cast<float*>(rocshmem_malloc(K_panel * N_local   * sizeof(float)));
    // d_scratch does not need to be on the symmetric heap (not communicated directly).
    float* d_scratch = nullptr;
    CHECK_HIP(hipMalloc(&d_scratch, M_local * N_local * sizeof(float)));

    // Status buffer for the broadcast kernel: status[0]=A broadcast rc, status[1]=B broadcast rc.
    int* d_bcast_status = nullptr;
    CHECK_HIP(hipMalloc(&d_bcast_status, 2 * sizeof(int)));

    if (!d_A_local || !d_B_local || !d_C_local || !d_A_panel || !d_B_panel) {
        std::cerr << "PE " << my_pe << ": rocshmem_malloc failed\n";
        // rocshmem_global_exit terminates all PEs symmetrically, preventing
        // other PEs from hanging in later collectives while this one exits.
        rocshmem_global_exit(1);
    }

    // -------------------------------------------------------------------------
    // Initialize matrices on the host, copy to device.
    //
    // A_local (PE pr,pc): rows [pr*M_local .. (pr+1)*M_local) of global A.
    //   Global A[i][k] = (i+1) + 0.1*(k+1)
    //   Local  A[i][k] with i offset: (pr*M_local + i + 1) + 0.1*(k+1)
    //
    // B_local (PE pr,pc): cols [pc*N_local .. (pc+1)*N_local) of global B.
    //   Global B[k][j] = 0.01*(k+1) + 0.001*(j+1)
    //   Local  B[k][j] with j offset: 0.01*(k+1) + 0.001*(pc*N_local + j + 1)
    // -------------------------------------------------------------------------
    std::vector<float> h_A_local(M_local * K);
    std::vector<float> h_B_local(K * N_local);

    for (int i = 0; i < M_local; i++)
        for (int k = 0; k < K; k++)
            h_A_local[i * K + k] = static_cast<float>(pr * M_local + i + 1)
                                  + 0.1f * static_cast<float>(k + 1);

    for (int k = 0; k < K; k++)
        for (int j = 0; j < N_local; j++)
            h_B_local[k * N_local + j] = 0.01f * static_cast<float>(k + 1)
                                        + 0.001f * static_cast<float>(pc * N_local + j + 1);

    CHECK_HIP(hipMemcpy(d_A_local, h_A_local.data(), M_local * K       * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemcpy(d_B_local, h_B_local.data(), K       * N_local * sizeof(float),
                        hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_C_local, 0, M_local * N_local * sizeof(float)));
    CHECK_HIP(hipMemset(d_scratch, 0, M_local * N_local * sizeof(float)));

    // All PEs must have their data in place before any SUMMA step begins.
    rocshmem_barrier_all();

    // -------------------------------------------------------------------------
    // SUMMA loop
    //
    // At step `s`:
    //   a_src = slice of d_A_local for columns [s*K_panel .. (s+1)*K_panel)
    //           i.e. d_A_local + s * K_panel  (row-major, row_stride=K)
    //   b_src = slice of d_B_local for rows [s*K_panel .. (s+1)*K_panel)
    //           i.e. d_B_local + s * K_panel * N_local  (row-major, row_stride=N_local)
    //
    // NOTE: gap — The two broadcasts (A along row team, B along column team) are logically
    // independent and could be issued non-blocking to overlap with each other and with the
    // subsequent GEMM. A future rocshmem_tile_broadcast_nbi_wg + collective fence would
    // enable this. Currently each broadcast blocks the workgroup until complete.
    // -------------------------------------------------------------------------
    for (int step = 0; step < Pc; step++) {
        // Build source tensor views pointing into the K_panel-wide slice for this step.
        // Both views share the original backing allocation (no data copy on host).
        float* a_slice_ptr = d_A_local + step * K_panel;   // offset into row-major A_local
        float* b_slice_ptr = d_B_local + step * K_panel * N_local;

        // a_src has shape M_local × K_panel but lives in a stride-K array — strided tensor.
        Tensor2D<float> a_src(a_slice_ptr, M_local, K_panel, /*row_stride=*/K);
        // b_src has shape K_panel × N_local, contiguous.
        Tensor2D<float> b_src(b_slice_ptr, K_panel, N_local);

        // a_panel and b_panel are contiguous broadcast-receive buffers.
        Tensor2D<float> a_panel(d_A_panel, M_local, K_panel);
        Tensor2D<float> b_panel(d_B_panel, K_panel, N_local);

        // Broadcast A panel along the row team, B panel along the column team.
        // WG 0 handles A (root = PE at column `step` in this row).
        // WG 1 handles B (root = PE at row `step` in this column).
        // Both root and non-root PEs receive the tile in their panel buffers.
        CHECK_HIP(hipMemset(d_bcast_status, 0, 2 * sizeof(int)));
        hipLaunchKernelGGL(summa_broadcast_kernel, /*gridDim=*/2, /*blockDim=*/256, 0, 0,
                           a_panel, b_panel, a_src, b_src,
                           row_team, col_team, pc, pr, step, d_bcast_status);
        CHECK_HIP(hipStreamSynchronize(0));

        // Check broadcast return values before proceeding to GEMM.
        // On unsupported backends (e.g. reverse-offload) the broadcast returns
        // ROCSHMEM_ERROR; continuing into GEMM with uninitialized panels would
        // produce silent numerical garbage.
        int h_bcast_status[2] = {0, 0};
        CHECK_HIP(hipMemcpy(h_bcast_status, d_bcast_status, 2 * sizeof(int),
                            hipMemcpyDeviceToHost));
        if (h_bcast_status[0] != ROCSHMEM_SUCCESS ||
            h_bcast_status[1] != ROCSHMEM_SUCCESS) {
            std::cerr << "PE (" << pr << "," << pc << ") step " << step
                      << ": tile_broadcast_wg failed"
                      << " (A rc=" << h_bcast_status[0]
                      << ", B rc=" << h_bcast_status[1] << ")\n";
            rocshmem_global_exit(1);
        }

        // Local GEMM: scratch = A_panel × B_panel  (M_local × N_local)
        // CK Tile writes to d_scratch so we can accumulate across SUMMA steps.
        CHECK_HIP(hipMemset(d_scratch, 0, M_local * N_local * sizeof(float)));
        run_ck_gemm(d_A_panel, d_B_panel, d_scratch, M_local, N_local, K_panel);
        CHECK_HIP(hipStreamSynchronize(0));

        // C_local += scratch
        int n_elems = M_local * N_local;
        int threads  = 256;
        int blocks   = (n_elems + threads - 1) / threads;
        hipLaunchKernelGGL(accumulate_kernel, blocks, threads, 0, 0,
                           d_C_local, d_scratch, n_elems);
        CHECK_HIP(hipStreamSynchronize(0));

        // Barrier ensures all PEs complete this SUMMA step before the next broadcast.
        rocshmem_barrier_all();
    }

    // -------------------------------------------------------------------------
    // Verification
    //
    // Build the full global A (M×K) and B (K×N) on PE 0 using rocshmem_tile_get_wg,
    // then compute the CPU reference and compare each PE's C_local block.
    //
    // NOTE: gap — There is no rocshmem_tile_get_wg variant that takes an offset into
    // a destination larger matrix; the start_coord/boundary pair addresses the *source*
    // tensor only. The destination must match the tile size exactly. For the gather below
    // we work around this by staging into an intermediate buffer per PE.
    // -------------------------------------------------------------------------
    CHECK_HIP(hipDeviceSynchronize());
    rocshmem_barrier_all();

    // Bring this PE's C_local back to host.
    std::vector<float> h_C_local(M_local * N_local);
    CHECK_HIP(hipMemcpy(h_C_local.data(), d_C_local,
                        M_local * N_local * sizeof(float), hipMemcpyDeviceToHost));

    // Build CPU reference for this PE's block: C_ref[i,j] = sum_k A[pr*M_local+i, k] * B[k, pc*N_local+j]
    // Reconstruct full-row A_local from host copy (already available) and full B_local from h_B_local.
    // For simplicity, PE 0 gathers and verifies; other PEs send their result to PE 0.
    // We do a simple reference computation locally on each PE using its own slabs.

    // A_local covers all K columns; B_local covers N_local columns.
    // C_block[i,j] = sum over all k of A_local[i,k] * B_local[k,j]  (B_local is already the right slab).
    std::vector<float> h_C_ref(M_local * N_local, 0.f);
    cpu_gemm(h_A_local, h_B_local, h_C_ref, M_local, N_local, K);

    // Element-wise atol + rtol * |ref| comparison.
    // fp32 accumulation error grows with K; the absolute floor handles near-zero
    // references where a pure relative check would be too strict.
    constexpr float rtol = 1e-4f;
    constexpr float atol = 1e-4f;
    bool local_pass = true;
    float max_rel_err = 0.f;
    for (int idx = 0; idx < M_local * N_local; idx++) {
        float ref = h_C_ref[idx];
        float err = std::fabs(h_C_local[idx] - ref);
        float tol = atol + rtol * std::fabs(ref);
        if (err > tol) {
            local_pass = false;
            max_rel_err = std::max(max_rel_err, err / (std::fabs(ref) + 1e-8f));
        }
    }

    if (!local_pass) {
        std::cerr << "PE (" << pr << "," << pc << "): FAILED — max relative error="
                  << max_rel_err << " (rtol=" << rtol << ", atol=" << atol << ")\n";
    }

    // Reduce pass/fail count across all PEs.
    // rocSHMEM scalar reductions are device-side only, so we launch a one-thread kernel.
    // Allocate symmetrically; treat either null as a global fatal error so that
    // no PE skips the barrier/reduce and leaves others hanging.
    int local_flag  = local_pass ? 1 : 0;
    int global_flag = 0;
    int* d_flag_src = static_cast<int*>(rocshmem_malloc(sizeof(int)));
    int* d_flag_dst = static_cast<int*>(rocshmem_malloc(sizeof(int)));
    if (!d_flag_src || !d_flag_dst) {
        std::cerr << "PE " << my_pe << ": rocshmem_malloc failed for reduction flags\n";
        rocshmem_global_exit(1);
    }
    CHECK_HIP(hipMemcpy(d_flag_src, &local_flag, sizeof(int), hipMemcpyHostToDevice));
    CHECK_HIP(hipMemset(d_flag_dst, 0, sizeof(int)));
    rocshmem_barrier_all();
    // One workgroup of 64 threads for the wg-collective reduce.
    hipLaunchKernelGGL(int_sum_reduce_kernel, 1, 64, 0, 0, d_flag_dst, d_flag_src);
    CHECK_HIP(hipStreamSynchronize(0));
    rocshmem_barrier_all();
    CHECK_HIP(hipMemcpy(&global_flag, d_flag_dst, sizeof(int), hipMemcpyDeviceToHost));
    rocshmem_free(d_flag_src);
    rocshmem_free(d_flag_dst);

    if (my_pe == 0) {
        bool all_pass = (global_flag == n_pes);
        std::cout << (all_pass ? "PASSED" : "FAILED")
                  << " — global pass count: " << global_flag << "/" << n_pes << "\n";

        // -----------------------------------------------------------------------
        // API gap summary
        // -----------------------------------------------------------------------
        std::cout << "\n--- rocSHMEM Tile API gaps identified by this example ---\n";
        std::cout << "1. No non-blocking tile broadcast (rocshmem_tile_broadcast_nbi_wg):\n"
                  << "   A and B broadcasts in each SUMMA step are sequential; non-blocking\n"
                  << "   variants + a collective fence would allow them to be pipelined.\n\n";
        std::cout << "2. Tile reductions support only float/double/integer element types:\n"
                  << "   fp16, bf16, fp8 are not listed in ROCSHMEM_TILE_ELEMENT_TYPE.\n"
                  << "   A CK Tile GEMM example in fp16 could not use tile sum-reduce.\n\n";
        std::cout << "3. Tile API is limited to 1D and 2D tensors:\n"
                  << "   Batched GEMM (A: batch×M×K) would need 3D tile addressing.\n";
    }

    // -------------------------------------------------------------------------
    // Cleanup
    // -------------------------------------------------------------------------
    rocshmem_team_destroy(row_team);
    rocshmem_team_destroy(col_team);

    rocshmem_free(d_A_local);
    rocshmem_free(d_B_local);
    rocshmem_free(d_C_local);
    rocshmem_free(d_A_panel);
    rocshmem_free(d_B_panel);
    CHECK_HIP(hipFree(d_scratch));
    CHECK_HIP(hipFree(d_bcast_status));

    rocshmem_finalize();
    return (global_flag == n_pes) ? 0 : 1;
}
