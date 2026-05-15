// test_stride2_then_stride1.cpp
// Launches stride2 then stride1 in sequence to test if stride2 contaminates
// the GPU state for stride1.
//
// Also tests the sub-offset scenario: stride1 input pointer is at offset 180224
// within a larger allocation (matching the capture's layout).

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <vector>


#define CHECK(x) do { \
    hipError_t _r = (x); \
    if (_r != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", _r, hipGetErrorString(_r), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

static void* load_hsaco(const char* path, size_t* sz_out) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Failed to open: %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f); rewind(f);
    void* code = malloc(sz);
    fread(code, 1, sz, f);
    fclose(f);
    *sz_out = sz;
    return code;
}

// Build and launch stride1 kernel with given pointers
static hipError_t launch_stride1(hipFunction_t func,
                                   void* d_in, void* d_wei, void* d_out) {
    uint8_t kbuf[256] = {0};
    uint32_t u32; uint64_t u64;

    // Scalars (matching capture: C=64, H=56, W=56, K=64, n_groups=96, flags=0x4600)
    u32 = 1;      memcpy(kbuf+0,   &u32, 4);  // BATCHSIZE
    u32 = 64;     memcpy(kbuf+4,   &u32, 4);  // C
    u32 = 56;     memcpy(kbuf+8,   &u32, 4);  // H
    u32 = 56;     memcpy(kbuf+12,  &u32, 4);  // W
    u32 = 64;     memcpy(kbuf+16,  &u32, 4);  // K
    u32 = 96;     memcpy(kbuf+20,  &u32, 4);  // n_groups
    u32 = 0x4600; memcpy(kbuf+24,  &u32, 4);  // flags
    u32 = 0;      memcpy(kbuf+28,  &u32, 4);  // reserved

    u64 = (uint64_t)d_in;  memcpy(kbuf+32, &u64, 8);  // in
    u64 = (uint64_t)d_wei; memcpy(kbuf+40, &u64, 8);  // wei
    u64 = (uint64_t)d_out; memcpy(kbuf+48, &u64, 8);  // out
    u64 = 0;               memcpy(kbuf+56, &u64, 8);  // rsv (null)

    u32 = 3;      memcpy(kbuf+64,  &u32, 4);  // R
    u32 = 3;      memcpy(kbuf+68,  &u32, 4);  // S
    u32 = 1;      memcpy(kbuf+72,  &u32, 4);  // pad_H
    u32 = 1;      memcpy(kbuf+76,  &u32, 4);  // pad_W
    u32 = 56;     memcpy(kbuf+80,  &u32, 4);  // out_H
    u32 = 56;     memcpy(kbuf+84,  &u32, 4);  // out_W
    u64 = 0;      memcpy(kbuf+88,  &u64, 8);  // bias_addr (null)
    u32 = 0;      memcpy(kbuf+96,  &u32, 4);  // alpha
    u32 = 0;      memcpy(kbuf+100, &u32, 4);  // beta
    // args 21-24 (offsets) = 0 (already zero)
    u32 = 200704; memcpy(kbuf+136, &u32, 4);  // d_N_stride = 64*56*56
    u32 = 3136;   memcpy(kbuf+140, &u32, 4);  // d_C_stride = 56*56
    u32 = 56;     memcpy(kbuf+144, &u32, 4);  // d_H_stride
    u32 = 1;      memcpy(kbuf+148, &u32, 4);  // d_W_stride
    u32 = 576;    memcpy(kbuf+152, &u32, 4);  // f_N_stride = 64*3*3
    u32 = 9;      memcpy(kbuf+156, &u32, 4);  // f_C_stride
    u32 = 3;      memcpy(kbuf+160, &u32, 4);  // f_R_stride
    u32 = 1;      memcpy(kbuf+164, &u32, 4);  // f_S_stride
    u32 = 200704; memcpy(kbuf+168, &u32, 4);  // o_N_stride
    u32 = 3136;   memcpy(kbuf+172, &u32, 4);  // o_K_stride
    u32 = 56;     memcpy(kbuf+176, &u32, 4);  // o_H_stride
    u32 = 1;      memcpy(kbuf+180, &u32, 4);  // o_W_stride
    u32 = 1;      memcpy(kbuf+184, &u32, 4);  // G
    u32 = 200704; memcpy(kbuf+188, &u32, 4);  // d_G_stride
    u32 = 36864;  memcpy(kbuf+192, &u32, 4);  // f_G_stride = 64*9*64
    u32 = 200704; memcpy(kbuf+196, &u32, 4);  // o_G_stride
    kbuf[200] = 0;                              // activation_mode

    size_t kbuf_sz = 256;
    void* extra[5] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kbuf_sz,
        HIP_LAUNCH_PARAM_END
    };

    hipError_t r = hipModuleLaunchKernel(func, 24576,1,1, 256,1,1, 0, nullptr, nullptr, extra);
    fprintf(stderr, "  launch=%d ", (int)r);
    hipDeviceSynchronize();
    hipError_t last = hipGetLastError();
    fprintf(stderr, "last_err=%d\n", (int)last);
    return last;
}

// Build and launch stride2 kernel with given pointers
static hipError_t launch_stride2(hipFunction_t func,
                                  void* d_in, void* d_wei, void* d_out) {
    uint8_t kbuf[256] = {0};
    uint32_t u32; uint64_t u64;

    // Stride2: C=3, H=224, W=224, K=64, n_groups=96
    u32 = 1;      memcpy(kbuf+0,   &u32, 4);  // BATCHSIZE
    u32 = 3;      memcpy(kbuf+4,   &u32, 4);  // C
    u32 = 224;    memcpy(kbuf+8,   &u32, 4);  // H
    u32 = 224;    memcpy(kbuf+12,  &u32, 4);  // W
    u32 = 64;     memcpy(kbuf+16,  &u32, 4);  // K
    u32 = 96;     memcpy(kbuf+20,  &u32, 4);  // n_groups
    u32 = 0x4600; memcpy(kbuf+24,  &u32, 4);  // flags
    u32 = 0;      memcpy(kbuf+28,  &u32, 4);  // reserved

    u64 = (uint64_t)d_in;  memcpy(kbuf+32, &u64, 8);  // in
    u64 = (uint64_t)d_wei; memcpy(kbuf+40, &u64, 8);  // wei
    u64 = (uint64_t)d_out; memcpy(kbuf+48, &u64, 8);  // out
    u64 = 0;               memcpy(kbuf+56, &u64, 8);  // rsv (null)

    u32 = 7;      memcpy(kbuf+64,  &u32, 4);  // R
    u32 = 7;      memcpy(kbuf+68,  &u32, 4);  // S
    u32 = 3;      memcpy(kbuf+72,  &u32, 4);  // pad_H
    u32 = 3;      memcpy(kbuf+76,  &u32, 4);  // pad_W
    u32 = 112;    memcpy(kbuf+80,  &u32, 4);  // out_H
    u32 = 112;    memcpy(kbuf+84,  &u32, 4);  // out_W
    u64 = 0;      memcpy(kbuf+88,  &u64, 8);  // bias_addr
    u32 = 0;      memcpy(kbuf+96,  &u32, 4);  // alpha
    u32 = 0;      memcpy(kbuf+100, &u32, 4);  // beta
    u32 = 150528; memcpy(kbuf+136, &u32, 4);  // d_N_stride = 3*224*224
    u32 = 50176;  memcpy(kbuf+140, &u32, 4);  // d_C_stride = 224*224
    u32 = 224;    memcpy(kbuf+144, &u32, 4);  // d_H_stride
    u32 = 1;      memcpy(kbuf+148, &u32, 4);  // d_W_stride
    u32 = 147;    memcpy(kbuf+152, &u32, 4);  // f_N_stride = 3*7*7
    u32 = 49;     memcpy(kbuf+156, &u32, 4);  // f_C_stride = 7*7
    u32 = 7;      memcpy(kbuf+160, &u32, 4);  // f_R_stride
    u32 = 1;      memcpy(kbuf+164, &u32, 4);  // f_S_stride
    u32 = 802816; memcpy(kbuf+168, &u32, 4);  // o_N_stride = 64*112*112
    u32 = 12544;  memcpy(kbuf+172, &u32, 4);  // o_K_stride = 112*112
    u32 = 112;    memcpy(kbuf+176, &u32, 4);  // o_H_stride
    u32 = 1;      memcpy(kbuf+180, &u32, 4);  // o_W_stride
    u32 = 1;      memcpy(kbuf+184, &u32, 4);  // G
    u32 = 150528; memcpy(kbuf+188, &u32, 4);  // d_G_stride
    u32 = 9408;   memcpy(kbuf+192, &u32, 4);  // f_G_stride = 3*7*7*64 /G = 9408? (147*64=9408)
    u32 = 802816; memcpy(kbuf+196, &u32, 4);  // o_G_stride
    kbuf[200] = 0;

    size_t kbuf_sz = 256;
    void* extra[5] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf,
        HIP_LAUNCH_PARAM_BUFFER_SIZE,    &kbuf_sz,
        HIP_LAUNCH_PARAM_END
    };

    hipError_t r = hipModuleLaunchKernel(func, 24576,1,1, 256,1,1, 0, nullptr, nullptr, extra);
    fprintf(stderr, "  launch=%d ", (int)r);
    hipDeviceSynchronize();
    hipError_t last = hipGetLastError();
    fprintf(stderr, "last_err=%d\n", (int)last);
    return last;
}

int main() {
    const char* hsaco2_path =
        "C:/MIGraphX/test/build2/resnet50_capture2.hrr/code_objects/fbcc2b2fa8ff8cb73449ba7ef0d43155.hsaco";
    const char* hsaco1_path =
        "C:/MIGraphX/test/build2/resnet50_capture2.hrr/code_objects/61d0789c47bcaf3bcbe94d85c9b3afa7.hsaco";

    // Load both modules
    size_t sz2, sz1;
    void* code2 = load_hsaco(hsaco2_path, &sz2);
    void* code1 = load_hsaco(hsaco1_path, &sz1);

    hipModule_t mod2, mod1;
    CHECK(hipModuleLoadData(&mod2, code2));
    CHECK(hipModuleLoadData(&mod1, code1));

    hipFunction_t func2, func1;
    CHECK(hipModuleGetFunction(&func2, mod2, "miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride2"));
    CHECK(hipModuleGetFunction(&func1, mod1, "miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride1"));
    fprintf(stderr, "Both kernels loaded.\n");


    // -----------------------------------------------------------------------
    // Test 1: stride1 alone (baseline)
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Test 1: stride1 ALONE ===\n");
    {
        void *d_in, *d_wei, *d_out;
        CHECK(hipMalloc(&d_in,  802816));
        CHECK(hipMalloc(&d_wei, 147456));
        CHECK(hipMalloc(&d_out, 802816));
        CHECK(hipMemset(d_in, 0, 802816));
        CHECK(hipMemset(d_wei, 0, 147456));
        CHECK(hipMemset(d_out, 0, 802816));
        CHECK(hipDeviceSynchronize());

        // Test with several n_groups and grid configurations
        for (uint32_t ng : {96u, 64u, 32u, 16u, 8u, 1u}) {
            for (uint32_t gx : {24576u, 12544u, 3136u, 784u}) {
                uint8_t kbuf2[256] = {0};
                uint32_t u32v; uint64_t u64v;
                u32v = 1;      memcpy(kbuf2+0,  &u32v, 4);
                u32v = 64;     memcpy(kbuf2+4,  &u32v, 4);
                u32v = 56;     memcpy(kbuf2+8,  &u32v, 4);
                u32v = 56;     memcpy(kbuf2+12, &u32v, 4);
                u32v = 64;     memcpy(kbuf2+16, &u32v, 4);
                u32v = ng;     memcpy(kbuf2+20, &u32v, 4);
                u32v = 0x4600; memcpy(kbuf2+24, &u32v, 4);
                u32v = 0;      memcpy(kbuf2+28, &u32v, 4);
                u64v = (uint64_t)d_in;  memcpy(kbuf2+32, &u64v, 8);
                u64v = (uint64_t)d_wei; memcpy(kbuf2+40, &u64v, 8);
                u64v = (uint64_t)d_out; memcpy(kbuf2+48, &u64v, 8);
                u64v = 0;               memcpy(kbuf2+56, &u64v, 8);
                u32v = 3;      memcpy(kbuf2+64,  &u32v, 4);
                u32v = 3;      memcpy(kbuf2+68,  &u32v, 4);
                u32v = 1;      memcpy(kbuf2+72,  &u32v, 4);
                u32v = 1;      memcpy(kbuf2+76,  &u32v, 4);
                u32v = 56;     memcpy(kbuf2+80,  &u32v, 4);
                u32v = 56;     memcpy(kbuf2+84,  &u32v, 4);
                // offsets 88-128 = 0
                u32v = 200704; memcpy(kbuf2+136, &u32v, 4);
                u32v = 3136;   memcpy(kbuf2+140, &u32v, 4);
                u32v = 56;     memcpy(kbuf2+144, &u32v, 4);
                u32v = 1;      memcpy(kbuf2+148, &u32v, 4);
                u32v = 576;    memcpy(kbuf2+152, &u32v, 4);
                u32v = 9;      memcpy(kbuf2+156, &u32v, 4);
                u32v = 3;      memcpy(kbuf2+160, &u32v, 4);
                u32v = 1;      memcpy(kbuf2+164, &u32v, 4);
                u32v = 200704; memcpy(kbuf2+168, &u32v, 4);
                u32v = 3136;   memcpy(kbuf2+172, &u32v, 4);
                u32v = 56;     memcpy(kbuf2+176, &u32v, 4);
                u32v = 1;      memcpy(kbuf2+180, &u32v, 4);
                u32v = 1;      memcpy(kbuf2+184, &u32v, 4);
                u32v = 200704; memcpy(kbuf2+188, &u32v, 4);
                u32v = 36864;  memcpy(kbuf2+192, &u32v, 4);
                u32v = 200704; memcpy(kbuf2+196, &u32v, 4);
                kbuf2[200] = 0;

                size_t kb2sz = 256;
                void* ex2[5] = {HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf2, HIP_LAUNCH_PARAM_BUFFER_SIZE, &kb2sz, HIP_LAUNCH_PARAM_END};
                hipModuleLaunchKernel(func1, gx,1,1, 256,1,1, 0, nullptr, nullptr, ex2);
                hipError_t s1 = hipDeviceSynchronize();
                hipError_t s1e = hipGetLastError();
                // Issue a probe command then second sync to catch WDDM deferred faults
                void* dp2; hipMalloc(&dp2, 4); hipMemset(dp2, 0, 4); hipFree(dp2);
                hipError_t e1 = hipDeviceSynchronize();
                hipError_t e2 = hipGetLastError();
                bool ok = (s1==hipSuccess && s1e==hipSuccess && e1==hipSuccess && e2==hipSuccess);
                fprintf(stderr, "  n_groups=%u grid=%u: s1=%d s1e=%d s2=%d s2e=%d %s\n",
                        ng, gx, (int)s1, (int)s1e, (int)e1, (int)e2,
                        ok ? "PASS" : "FAIL");
            }
        }

        hipFree(d_in); hipFree(d_wei); hipFree(d_out);
    }

    // -----------------------------------------------------------------------
    // Test 2: stride2 then stride1 (shared input alloc simulating capture)
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Test 2: stride2 THEN stride1 (shared large alloc) ===\n");
    {
        // Simulate capture: one large alloc, stride2 in at offset 0, stride1 in at offset 180224
        const size_t LARGE_SZ = 2 * 1024 * 1024;
        const size_t STRIDE1_IN_OFFSET = 180224;

        void *d_large, *d_wei2, *d_out2, *d_wei1, *d_out1;
        CHECK(hipMalloc(&d_large, LARGE_SZ));
        CHECK(hipMalloc(&d_wei2,  9408 * 4));  // 3*7*7*64*4 = 37632 bytes
        CHECK(hipMalloc(&d_out2,  802816));
        CHECK(hipMalloc(&d_wei1,  147456));
        CHECK(hipMalloc(&d_out1,  802816));

        // Zero all
        CHECK(hipMemset(d_large, 0, LARGE_SZ));
        CHECK(hipMemset(d_wei2,  0, 9408 * 4));
        CHECK(hipMemset(d_out2,  0, 802816));
        CHECK(hipMemset(d_wei1,  0, 147456));
        CHECK(hipMemset(d_out1,  0, 802816));
        CHECK(hipDeviceSynchronize());

        // stride2: in=d_large+0, wei=d_wei2, out=d_out2
        void* d_in2 = d_large;
        fprintf(stderr, "  Launching stride2: in=%p wei=%p out=%p\n", d_in2, d_wei2, d_out2);
        hipError_t r2 = launch_stride2(func2, d_in2, d_wei2, d_out2);
        fprintf(stderr, "  stride2: %s\n", r2 == hipSuccess ? "PASS" : "FAIL");

        // stride1: in=d_large+180224, wei=d_wei1, out=d_out1
        void* d_in1 = reinterpret_cast<uint8_t*>(d_large) + STRIDE1_IN_OFFSET;
        fprintf(stderr, "  Launching stride1: in=%p wei=%p out=%p\n", d_in1, d_wei1, d_out1);
        hipError_t r1 = launch_stride1(func1, d_in1, d_wei1, d_out1);
        fprintf(stderr, "  stride1: %s\n", r1 == hipSuccess ? "PASS" : "FAIL");

        fprintf(stderr, "Test 2 overall: %s\n",
                (r2 == hipSuccess && r1 == hipSuccess) ? "PASS" : "FAIL");

        hipFree(d_large); hipFree(d_wei2); hipFree(d_out2); hipFree(d_wei1); hipFree(d_out1);
    }

    // -----------------------------------------------------------------------
    // Test 3: stride2 then stride1 (both with fresh separate allocs, zeroed)
    // -----------------------------------------------------------------------
    fprintf(stderr, "\n=== Test 3: stride2 THEN stride1 (separate allocs) ===\n");
    {
        void *d_in2, *d_wei2, *d_out2;
        void *d_in1, *d_wei1, *d_out1;
        CHECK(hipMalloc(&d_in2,  150528 * 4));
        CHECK(hipMalloc(&d_wei2, 9408 * 4));
        CHECK(hipMalloc(&d_out2, 802816));
        CHECK(hipMalloc(&d_in1,  802816));
        CHECK(hipMalloc(&d_wei1, 147456));
        CHECK(hipMalloc(&d_out1, 802816));
        CHECK(hipMemset(d_in2, 0, 150528 * 4));
        CHECK(hipMemset(d_wei2, 0, 9408 * 4));
        CHECK(hipMemset(d_out2, 0, 802816));
        CHECK(hipMemset(d_in1, 0, 802816));
        CHECK(hipMemset(d_wei1, 0, 147456));
        CHECK(hipMemset(d_out1, 0, 802816));
        CHECK(hipDeviceSynchronize());

        fprintf(stderr, "  Launching stride2: in=%p wei=%p out=%p\n", d_in2, d_wei2, d_out2);
        hipError_t r2 = launch_stride2(func2, d_in2, d_wei2, d_out2);
        fprintf(stderr, "  stride2: %s\n", r2 == hipSuccess ? "PASS" : "FAIL");

        fprintf(stderr, "  Launching stride1: in=%p wei=%p out=%p\n", d_in1, d_wei1, d_out1);
        hipError_t r1 = launch_stride1(func1, d_in1, d_wei1, d_out1);
        fprintf(stderr, "  stride1: %s\n", r1 == hipSuccess ? "PASS" : "FAIL");

        fprintf(stderr, "Test 3 overall: %s\n",
                (r2 == hipSuccess && r1 == hipSuccess) ? "PASS" : "FAIL");

        hipFree(d_in2); hipFree(d_wei2); hipFree(d_out2);
        hipFree(d_in1); hipFree(d_wei1); hipFree(d_out1);
    }

    hipModuleUnload(mod1);
    hipModuleUnload(mod2);
    free(code1); free(code2);
    return 0;
}
