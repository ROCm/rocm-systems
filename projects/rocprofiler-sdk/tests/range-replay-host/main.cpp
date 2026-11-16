// MIT License
//
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

// A time-stepping solver run inside one host-replay range.
//
// Each time step mixes everything packet replay cannot re-execute: work on two streams joined by
// events, an FFT / spectral filter / inverse FFT, a 5-point stencil, Jacobi sweeps, a
// device-to-device copy, a memset, a device-to-host copy of a residual that the host reduces, and
// host-computed boundary data copied back to the device for the next step. Optionally a HIP graph
// launch (RR_HOST_GRAPH=1) and a scratch allocation made and freed inside every step
// (RR_HOST_SCRATCH=1).
//
// The range is opened with ROCPROFILER_RANGE_REPLAY_HOST_BEGIN, so each extra pass re-runs this
// host code from the restored entry state. The checks: every pass ends with the same device
// checksum and the same host state, the host state registered with
// rocprofiler_range_replay_host_track_memory reflects one execution of the range rather than one
// per pass, and the number of passes is the one expected (RR_HOST_EXPECT_PASSES).

#include "range.hpp"

#include <rocprofiler-sdk/experimental/range_replay.h>

#include <hip/hip_runtime.h>

#include <dlfcn.h>
#include <setjmp.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define HIP_CHECK(call)                                                                            \
    do                                                                                             \
    {                                                                                              \
        hipError_t _err = (call);                                                                  \
        if(_err != hipSuccess)                                                                     \
        {                                                                                          \
            fprintf(                                                                               \
                stderr, "HIP error '%s' at %s:%d\n", hipGetErrorString(_err), __FILE__, __LINE__); \
            return EXIT_FAILURE;                                                                   \
        }                                                                                          \
    } while(0)

namespace
{
constexpr int   kCells       = kN * kN;
constexpr int   kResidBlock  = 256;
constexpr int   kResidBlocks = kCells / kResidBlock;
constexpr float kAlpha       = 0.2f;
constexpr float kPi          = 3.14159265358979f;

__device__ unsigned
bit_reverse(unsigned value, int bits)
{
    unsigned out = 0;
    for(int i = 0; i < bits; ++i)
    {
        out = (out << 1) | (value & 1U);
        value >>= 1;
    }
    return out;
}

__global__ void
rr_host_pack(const float* u, float2* spec)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kCells) spec[i] = make_float2(u[i], 0.0f);
}

// One radix-2 FFT of length kN per block (one grid row per block), in shared memory.
__global__ void
rr_host_fft_rows(float2* spec, int inverse)
{
    __shared__ float2 s[kN];
    float2*           row = spec + blockIdx.x * kN;
    const int         t   = threadIdx.x;

    s[bit_reverse(t, kLogN)] = row[t];
    __syncthreads();

    for(int len = 2; len <= kN; len <<= 1)
    {
        const int half = len >> 1;
        if(t < kN / 2)
        {
            const int   k     = t % half;
            const int   i     = (t / half) * len + k;
            const int   j     = i + half;
            const float angle = (inverse != 0 ? 2.0f : -2.0f) * kPi * k / len;
            float       sn    = 0.0f;
            float       cs    = 0.0f;
            sincosf(angle, &sn, &cs);
            const float2 a = s[i];
            const float2 b = s[j];
            const float2 w = make_float2(b.x * cs - b.y * sn, b.x * sn + b.y * cs);
            s[i]           = make_float2(a.x + w.x, a.y + w.y);
            s[j]           = make_float2(a.x - w.x, a.y - w.y);
        }
        __syncthreads();
    }
    row[t] = s[t];
}

__global__ void
rr_host_spectral_filter(float2* spec, int step)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i >= kCells) return;
    const int   k    = i % kN;
    const int   freq = k < kN - k ? k : kN - k;
    const float damp = 1.0f / (1.0f + 0.01f * freq * freq * (1 + step % 3));
    spec[i]          = make_float2(spec[i].x * damp, spec[i].y * damp);
}

__global__ void
rr_host_unpack(const float2* spec, float* rhs)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kCells) rhs[i] = spec[i].x / kN;
}

// 5-point average; the top edge reads the host-computed boundary row instead of a neighbour.
__global__ void
rr_host_stencil(const float* in, const float* boundary, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= kN || y >= kN) return;
    const float c   = in[y * kN + x];
    const float n   = y > 0 ? in[(y - 1) * kN + x] : boundary[x];
    const float s   = y < kN - 1 ? in[(y + 1) * kN + x] : 0.0f;
    const float w   = x > 0 ? in[y * kN + x - 1] : 0.0f;
    const float e   = x < kN - 1 ? in[y * kN + x + 1] : 0.0f;
    out[y * kN + x] = 0.2f * (c + n + s + w + e);
}

// One Jacobi sweep for (1 + 4a) u - a * sum(neighbours) = rhs, zero outside the grid.
__global__ void
rr_host_jacobi(const float* u, const float* rhs, float* out)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if(x >= kN || y >= kN) return;
    const float n   = y > 0 ? u[(y - 1) * kN + x] : 0.0f;
    const float s   = y < kN - 1 ? u[(y + 1) * kN + x] : 0.0f;
    const float w   = x > 0 ? u[y * kN + x - 1] : 0.0f;
    const float e   = x < kN - 1 ? u[y * kN + x + 1] : 0.0f;
    out[y * kN + x] = (rhs[y * kN + x] + kAlpha * (n + s + w + e)) / (1.0f + 4.0f * kAlpha);
}

// Per-block sum of squared residuals, reduced in a fixed order so it is bitwise repeatable.
__global__ void
rr_host_residual(const float* u, const float* rhs, float* partials)
{
    __shared__ float s[kResidBlock];
    const int        i = blockIdx.x * blockDim.x + threadIdx.x;
    const int        x = i % kN;
    const int        y = i / kN;

    const float n  = y > 0 ? u[(y - 1) * kN + x] : 0.0f;
    const float so = y < kN - 1 ? u[(y + 1) * kN + x] : 0.0f;
    const float w  = x > 0 ? u[y * kN + x - 1] : 0.0f;
    const float e  = x < kN - 1 ? u[y * kN + x + 1] : 0.0f;
    const float r  = rhs[i] - ((1.0f + 4.0f * kAlpha) * u[i] - kAlpha * (n + so + w + e));

    s[threadIdx.x] = r * r;
    __syncthreads();
    for(int stride = kResidBlock / 2; stride > 0; stride >>= 1)
    {
        if(static_cast<int>(threadIdx.x) < stride) s[threadIdx.x] += s[threadIdx.x + stride];
        __syncthreads();
    }
    if(threadIdx.x == 0) partials[blockIdx.x] = s[0];
}

__global__ void
rr_graph_relax(float* u)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kCells) u[i] *= 0.999f;
}

__global__ void
rr_init_field(float* u, float* boundary)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < kCells) u[i] = static_cast<float>((i * 37) % 101) / 101.0f;
    if(i < kN) boundary[i] = 0.0f;
}

// Host state the range reads and writes. Registered with the SDK, so it is rewound with device
// memory before every extra pass.
struct host_state_t
{
    double   energy       = 0.0;
    uint64_t steps        = 0;
    float    boundary[kN] = {};
};

host_state_t g_host = {};

// Per-pass results. Deliberately not registered: they must survive the rewinds to be compared.
double   g_checksum[kMaxPasses] = {};
double   g_energy[kMaxPasses]   = {};
uint64_t g_steps[kMaxPasses]    = {};
int      g_passes_run           = 0;
int      g_pass0_log_lines      = 0;

struct resources_t
{
    hipStream_t                                       a          = nullptr;
    hipStream_t                                       b          = nullptr;
    hipEvent_t                                        fft_done   = nullptr;
    hipEvent_t                                        halo_done  = nullptr;
    hipGraphExec_t                                    graph_exec = nullptr;
    float*                                            u          = nullptr;
    float*                                            tmp        = nullptr;
    float*                                            rhs        = nullptr;
    float2*                                           spec       = nullptr;
    float*                                            boundary   = nullptr;
    float*                                            partials   = nullptr;
    float*                                            h_partials = nullptr;  // pinned
    float*                                            h_boundary = nullptr;  // pinned
    float*                                            h_field    = nullptr;  // pinned
    bool                                              scratch    = false;
    decltype(&rocprofiler_range_replay_get_host_pass) get_pass   = nullptr;
};

uint64_t
current_pass(const resources_t& r)
{
    auto pass = uint64_t{0};
    if(r.get_pass != nullptr) r.get_pass(&pass);
    return pass;
}

int
field_checksum(const resources_t& r, double& out)
{
    HIP_CHECK(hipMemcpyAsync(r.h_field, r.u, kCells * sizeof(float), hipMemcpyDeviceToHost, r.a));
    HIP_CHECK(hipStreamSynchronize(r.a));
    out = 0.0;
    for(int i = 0; i < kCells; ++i)
        out += static_cast<double>(r.h_field[i]) * (1 + i % 7);
    return EXIT_SUCCESS;
}

// One time step. Everything it creates is destroyed before it returns, so nothing with a
// destructor is live when end() jumps.
int
time_step(const resources_t& r, int step)
{
    const dim3 grid2d{kN / 16, kN / 16};
    const dim3 block2d{16, 16};
    const int  blocks1d = (kCells + 255) / 256;

    // Stream A: spectral smoothing of the field into the right-hand side.
    rr_host_pack<<<blocks1d, 256, 0, r.a>>>(r.u, r.spec);
    rr_host_fft_rows<<<kN, kN, 0, r.a>>>(r.spec, 0);
    rr_host_spectral_filter<<<blocks1d, 256, 0, r.a>>>(r.spec, step);
    rr_host_fft_rows<<<kN, kN, 0, r.a>>>(r.spec, 1);
    rr_host_unpack<<<blocks1d, 256, 0, r.a>>>(r.spec, r.rhs);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(r.fft_done, r.a));

    // Stream B, after A: copy the field aside and apply the stencil with the host's boundary row.
    HIP_CHECK(hipStreamWaitEvent(r.b, r.fft_done, 0));
    HIP_CHECK(hipMemcpyAsync(r.tmp, r.u, kCells * sizeof(float), hipMemcpyDeviceToDevice, r.b));
    rr_host_stencil<<<grid2d, block2d, 0, r.b>>>(r.tmp, r.boundary, r.u);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipEventRecord(r.halo_done, r.b));

    // Stream A, after B: Jacobi sweeps, ping-ponging between u and tmp (an even count ends in u).
    HIP_CHECK(hipStreamWaitEvent(r.a, r.halo_done, 0));
    for(int sweep = 0; sweep < kJacobiSweeps; ++sweep)
    {
        const bool to_tmp = (sweep % 2) == 0;
        rr_host_jacobi<<<grid2d, block2d, 0, r.a>>>(
            to_tmp ? r.u : r.tmp, r.rhs, to_tmp ? r.tmp : r.u);
    }
    HIP_CHECK(hipGetLastError());

    if(r.graph_exec != nullptr) HIP_CHECK(hipGraphLaunch(r.graph_exec, r.a));

    // Residual into a partials buffer -- the long-lived one, or a scratch allocation made and freed
    // inside the step -- then to the host.
    float* partials = r.partials;
    if(r.scratch) HIP_CHECK(hipMalloc(&partials, kResidBlocks * sizeof(float)));
    HIP_CHECK(hipMemsetAsync(partials, 0, kResidBlocks * sizeof(float), r.a));
    rr_host_residual<<<kResidBlocks, kResidBlock, 0, r.a>>>(r.u, r.rhs, partials);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipMemcpyAsync(
        r.h_partials, partials, kResidBlocks * sizeof(float), hipMemcpyDeviceToHost, r.a));
    HIP_CHECK(hipStreamSynchronize(r.a));
    if(r.scratch) HIP_CHECK(hipFree(partials));

    // Host call: reduce the residual, advance host state, and compute the next boundary row from
    // it. The device consumes this on the next step, which a packet recording cannot reproduce.
    auto resid = 0.0;
    for(int i = 0; i < kResidBlocks; ++i)
        resid += r.h_partials[i];
    g_host.energy += resid;
    g_host.steps += 1;
    for(int k = 0; k < kN; ++k)
    {
        g_host.boundary[k] = static_cast<float>(0.5 * std::sin(0.1 * k + step) / (1.0 + resid));
        r.h_boundary[k]    = g_host.boundary[k];
    }
    HIP_CHECK(
        hipMemcpyAsync(r.boundary, r.h_boundary, kN * sizeof(float), hipMemcpyHostToDevice, r.b));
    HIP_CHECK(hipStreamSynchronize(r.b));

    // Side effects that must happen once go through the pass query.
    if(current_pass(r) == 0)
    {
        printf("[app] step %d residual=%.9e\n", step, resid);
        ++g_pass0_log_lines;
    }
    return EXIT_SUCCESS;
}

// The whole range body: the time-stepping loop plus the per-pass bookkeeping.
int
range_body(const resources_t& r)
{
    for(int step = 0; step < kTimeSteps; ++step)
        if(time_step(r, step) != EXIT_SUCCESS) return EXIT_FAILURE;

    auto checksum = 0.0;
    if(field_checksum(r, checksum) != EXIT_SUCCESS) return EXIT_FAILURE;

    const auto pass = current_pass(r);
    if(pass < static_cast<uint64_t>(kMaxPasses))
    {
        g_checksum[pass] = checksum;
        g_energy[pass]   = g_host.energy;
        g_steps[pass]    = g_host.steps;
    }
    ++g_passes_run;

    HIP_CHECK(hipDeviceSynchronize());
    return EXIT_SUCCESS;
}

int
env_int(const char* name, int fallback)
{
    const char* value = getenv(name);
    return (value != nullptr && *value != '\0') ? atoi(value) : fallback;
}

int
build_graph(resources_t& r)
{
    float* graph_scratch = nullptr;
    HIP_CHECK(hipMalloc(&graph_scratch, kN * sizeof(float)));

    hipGraph_t graph = nullptr;
    HIP_CHECK(hipStreamBeginCapture(r.a, hipStreamCaptureModeGlobal));
    rr_graph_relax<<<(kCells + 255) / 256, 256, 0, r.a>>>(r.u);
    HIP_CHECK(hipMemsetAsync(graph_scratch, 0, kN * sizeof(float), r.a));
    HIP_CHECK(hipStreamEndCapture(r.a, &graph));
    HIP_CHECK(hipGraphInstantiate(&r.graph_exec, graph, nullptr, nullptr, 0));
    HIP_CHECK(hipGraphDestroy(graph));
    return EXIT_SUCCESS;
}
}  // namespace

int
main()
{
    auto* begin_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_begin)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_begin"));
    auto* end_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_end)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_end"));
    auto* jmpbuf_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_jmpbuf)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_jmpbuf"));
    auto* track_fn = reinterpret_cast<decltype(&rocprofiler_range_replay_host_track_memory)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_host_track_memory"));

    auto r     = resources_t{};
    r.get_pass = reinterpret_cast<decltype(&rocprofiler_range_replay_get_host_pass)>(
        dlsym(RTLD_DEFAULT, "rocprofiler_range_replay_get_host_pass"));

    if(begin_fn == nullptr || end_fn == nullptr || jmpbuf_fn == nullptr || track_fn == nullptr ||
       r.get_pass == nullptr)
    {
        fprintf(stderr, "[app] FAIL: the range replay host API is not in this process\n");
        return EXIT_FAILURE;
    }

    const bool arm           = env_int("RR_HOST_APP_ARM", 1) != 0;
    const int  expect_passes = env_int("RR_HOST_EXPECT_PASSES", 4);
    r.scratch                = env_int("RR_HOST_SCRATCH", 0) != 0;
    const bool use_graph     = env_int("RR_HOST_GRAPH", 0) != 0;

    HIP_CHECK(hipStreamCreateWithFlags(&r.a, hipStreamNonBlocking));
    HIP_CHECK(hipStreamCreateWithFlags(&r.b, hipStreamNonBlocking));
    HIP_CHECK(hipEventCreateWithFlags(&r.fft_done, hipEventDisableTiming));
    HIP_CHECK(hipEventCreateWithFlags(&r.halo_done, hipEventDisableTiming));
    HIP_CHECK(hipMalloc(&r.u, kCells * sizeof(float)));
    HIP_CHECK(hipMalloc(&r.tmp, kCells * sizeof(float)));
    HIP_CHECK(hipMalloc(&r.rhs, kCells * sizeof(float)));
    HIP_CHECK(hipMalloc(&r.spec, kCells * sizeof(float2)));
    HIP_CHECK(hipMalloc(&r.boundary, kN * sizeof(float)));
    HIP_CHECK(hipMalloc(&r.partials, kResidBlocks * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&r.h_partials, kResidBlocks * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_boundary, kN * sizeof(float), hipHostMallocDefault));
    HIP_CHECK(hipHostMalloc(&r.h_field, kCells * sizeof(float), hipHostMallocDefault));
    if(use_graph && build_graph(r) != EXIT_SUCCESS) return EXIT_FAILURE;

    // Warm-up step outside the range: HIP loads each module on the first launch of one of its
    // kernels and sets up its copy and memset paths lazily, and neither should happen inside the
    // range for the first time.
    rr_init_field<<<(kCells + 255) / 256, 256, 0, r.a>>>(r.u, r.boundary);
    HIP_CHECK(hipGetLastError());
    if(time_step(r, 0) != EXIT_SUCCESS) return EXIT_FAILURE;
    {
        float* warm = nullptr;
        HIP_CHECK(hipMalloc(&warm, kResidBlocks * sizeof(float)));
        HIP_CHECK(hipFree(warm));
    }

    // The range's real starting state.
    rr_init_field<<<(kCells + 255) / 256, 256, 0, r.a>>>(r.u, r.boundary);
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());
    g_host            = host_state_t{};
    g_pass0_log_lines = 0;

    if(track_fn(&g_host, sizeof(g_host)) != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: rocprofiler_range_replay_host_track_memory failed\n");
        return EXIT_FAILURE;
    }

    // Written after the sigsetjmp and read after a siglongjmp, so volatile.
    volatile auto begin_status = ROCPROFILER_STATUS_SUCCESS;
    if(arm)
    {
        // ROCPROFILER_RANGE_REPLAY_HOST_BEGIN, spelled out because the SDK is resolved via dlsym.
        if(sigsetjmp(*static_cast<sigjmp_buf*>(jmpbuf_fn()), 1) == 0)
            begin_status = begin_fn(kRangeId);
    }
    else
    {
        begin_status = begin_fn(kRangeId);
    }
    if(begin_status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: begin returned status %d\n", static_cast<int>(begin_status));
        return EXIT_FAILURE;
    }

    if(range_body(r) != EXIT_SUCCESS) return EXIT_FAILURE;

    if(const auto status = end_fn(); status != ROCPROFILER_STATUS_SUCCESS)
    {
        fprintf(stderr, "[app] FAIL: end returned status %d\n", static_cast<int>(status));
        return EXIT_FAILURE;
    }

    // From here on the range is closed and runs exactly once.
    auto final_checksum = 0.0;
    if(field_checksum(r, final_checksum) != EXIT_SUCCESS) return EXIT_FAILURE;

    printf("[app] passes=%d checksum=%.17g energy=%.17g steps=%lu log_lines=%d\n",
           g_passes_run,
           g_checksum[0],
           g_energy[0],
           static_cast<unsigned long>(g_host.steps),
           g_pass0_log_lines);

    auto ok = true;
    if(g_passes_run != expect_passes)
    {
        fprintf(stderr, "[app] FAIL: ran %d passes, expected %d\n", g_passes_run, expect_passes);
        ok = false;
    }
    for(int pass = 1; pass < g_passes_run && pass < kMaxPasses; ++pass)
    {
        if(std::memcmp(&g_checksum[pass], &g_checksum[0], sizeof(double)) != 0 ||
           std::memcmp(&g_energy[pass], &g_energy[0], sizeof(double)) != 0 ||
           g_steps[pass] != g_steps[0])
        {
            fprintf(stderr,
                    "[app] FAIL: pass %d ended with checksum=%.17g energy=%.17g steps=%lu, pass 0 "
                    "with checksum=%.17g energy=%.17g steps=%lu\n",
                    pass,
                    g_checksum[pass],
                    g_energy[pass],
                    static_cast<unsigned long>(g_steps[pass]),
                    g_checksum[0],
                    g_energy[0],
                    static_cast<unsigned long>(g_steps[0]));
            ok = false;
        }
    }
    if(g_host.steps != static_cast<uint64_t>(kTimeSteps))
    {
        fprintf(stderr,
                "[app] FAIL: host state counted %lu steps; it was not rewound between passes\n",
                static_cast<unsigned long>(g_host.steps));
        ok = false;
    }
    if(std::memcmp(&final_checksum, &g_checksum[0], sizeof(double)) != 0)
    {
        fprintf(stderr,
                "[app] FAIL: the device state after the range (%.17g) is not the state one "
                "execution of the range produces (%.17g)\n",
                final_checksum,
                g_checksum[0]);
        ok = false;
    }
    if(g_pass0_log_lines != kTimeSteps)
    {
        fprintf(stderr,
                "[app] FAIL: %d once-only log lines, expected %d\n",
                g_pass0_log_lines,
                kTimeSteps);
        ok = false;
    }

    if(r.graph_exec != nullptr) HIP_CHECK(hipGraphExecDestroy(r.graph_exec));
    HIP_CHECK(hipHostFree(r.h_field));
    HIP_CHECK(hipHostFree(r.h_boundary));
    HIP_CHECK(hipHostFree(r.h_partials));
    HIP_CHECK(hipFree(r.partials));
    HIP_CHECK(hipFree(r.boundary));
    HIP_CHECK(hipFree(r.spec));
    HIP_CHECK(hipFree(r.rhs));
    HIP_CHECK(hipFree(r.tmp));
    HIP_CHECK(hipFree(r.u));
    HIP_CHECK(hipEventDestroy(r.halo_done));
    HIP_CHECK(hipEventDestroy(r.fft_done));
    HIP_CHECK(hipStreamDestroy(r.b));
    HIP_CHECK(hipStreamDestroy(r.a));

    return ok ? EXIT_SUCCESS : EXIT_FAILURE;
}
