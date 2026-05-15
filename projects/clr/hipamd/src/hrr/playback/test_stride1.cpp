#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { \
    hipError_t r = (x); \
    if (r != hipSuccess) { \
        fprintf(stderr, "HIP error %d (%s) at %s:%d\n", r, hipGetErrorString(r), __FILE__, __LINE__); \
        exit(1); \
    } \
} while(0)

int main(int argc, char** argv) {
    // Load stride1 HSACO — default to the patched version for testing the bug fix
    const char* hsaco_path = (argc > 1) ? argv[1] :
        "C:/tmp/stride1_patched.hsaco";
    // Original (buggy): C:/MIGraphX/test/build2/resnet50_capture2.hrr/code_objects/61d0789c47bcaf3bcbe94d85c9b3afa7.hsaco
    // Patched (fixed):  C:/tmp/stride1_patched.hsaco
    FILE* f = fopen(hsaco_path, "rb");
    if (!f) {
        fprintf(stderr, "Failed to open HSACO: %s\n", hsaco_path);
        return 1;
    }
    fprintf(stderr, "Loading HSACO: %s\n", hsaco_path);
    if (!f) { fprintf(stderr, "Failed to open HSACO\n"); return 1; }
    fseek(f, 0, SEEK_END);
    size_t sz = ftell(f);
    rewind(f);
    void* code = malloc(sz);
    fread(code, 1, sz, f);
    fclose(f);
    
    hipModule_t mod;
    CHECK(hipModuleLoadData(&mod, code));
    
    hipFunction_t func;
    CHECK(hipModuleGetFunction(&func, mod, "miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride1"));
    
    printf("Module and function loaded OK\n");
    
    // Allocate buffers with large padding on both sides to simulate the
    // original application's larger allocation (stride1 in_ptr was at
    // offset 0x2c4000 within a large GPU allocation in the capture).
    // Without padding, the kernel may page-fault accessing memory just
    // outside the exact buffer bounds during its inner loop.
    const size_t PAD = 4 * 1024 * 1024;  // 4 MB padding each side
    size_t in_sz  = 802816;  // 64*56*56*4
    size_t w_sz   = 147456;  // from H2D capture
    size_t out_sz = 802816;  // 64*56*56*4

    void *d_in_base, *d_w_base, *d_out_base;
    CHECK(hipMalloc(&d_in_base,  PAD + in_sz  + PAD));
    CHECK(hipMalloc(&d_w_base,   PAD + w_sz   + PAD));
    CHECK(hipMalloc(&d_out_base, PAD + out_sz + PAD));

    // Zero initialize the full padded allocations
    CHECK(hipMemset(d_in_base,  0, PAD + in_sz  + PAD));
    CHECK(hipMemset(d_w_base,   0, PAD + w_sz   + PAD));
    CHECK(hipMemset(d_out_base, 0, PAD + out_sz + PAD));

    // Point the kernel args into the middle of the padded allocations
    void *d_in  = (uint8_t*)d_in_base  + PAD;
    void *d_w   = (uint8_t*)d_w_base   + PAD;
    void *d_out = (uint8_t*)d_out_base + PAD;
    fprintf(stderr, "[test] in=%p wei=%p out=%p (each padded ±4MB)\n", d_in, d_w, d_out);
    
    // Build kernarg buffer (248 bytes, zero-initialized)
    uint8_t kbuf[256] = {0};
    
    // Set visible args at their offsets (from HSACO metadata)
    uint32_t u32;
    uint64_t u64;
    
    // arg[0] BATCHSIZE = 1
    u32 = 1; memcpy(kbuf + 0, &u32, 4);
    // arg[1] C = 64
    u32 = 64; memcpy(kbuf + 4, &u32, 4);
    // arg[2] H = 56
    u32 = 56; memcpy(kbuf + 8, &u32, 4);
    // arg[3] W = 56
    u32 = 56; memcpy(kbuf + 12, &u32, 4);
    // arg[4] K = 64
    u32 = 64; memcpy(kbuf + 16, &u32, 4);
    // arg[5] n_groups = 96
    u32 = 96; memcpy(kbuf + 20, &u32, 4);
    // arg[6] flags = 17920 = 0x4600
    u32 = 0x4600; memcpy(kbuf + 24, &u32, 4);
    // arg[7] reserved = 0
    u32 = 0; memcpy(kbuf + 28, &u32, 4);
    // arg[8] in = d_in
    u64 = (uint64_t)d_in; memcpy(kbuf + 32, &u64, 8);
    // arg[9] weights = d_w
    u64 = (uint64_t)d_w; memcpy(kbuf + 40, &u64, 8);
    // arg[10] out = d_out
    u64 = (uint64_t)d_out; memcpy(kbuf + 48, &u64, 8);
    // arg[11] rsv_ptr = null
    u64 = 0; memcpy(kbuf + 56, &u64, 8);
    // arg[12] R = 3
    u32 = 3; memcpy(kbuf + 64, &u32, 4);
    // arg[13] S = 3
    u32 = 3; memcpy(kbuf + 68, &u32, 4);
    // arg[14] pad_h = 1
    u32 = 1; memcpy(kbuf + 72, &u32, 4);
    // arg[15] pad_w = 1
    u32 = 1; memcpy(kbuf + 76, &u32, 4);
    // arg[16] out_h = 56
    u32 = 56; memcpy(kbuf + 80, &u32, 4);
    // arg[17] out_w = 56
    u32 = 56; memcpy(kbuf + 84, &u32, 4);
    // arg[18] bias_addr = null
    u64 = 0; memcpy(kbuf + 88, &u64, 8);
    // arg[19] alpha = 0
    u32 = 0; memcpy(kbuf + 96, &u32, 4);
    // arg[20] beta = 0
    u32 = 0; memcpy(kbuf + 100, &u32, 4);
    // args 21-24 (offsets) = 0
    // arg[25] d_N_stride = 200704
    u32 = 200704; memcpy(kbuf + 136, &u32, 4);
    // arg[26] d_C_stride = 3136
    u32 = 3136; memcpy(kbuf + 140, &u32, 4);
    // arg[27] d_H_stride = 56
    u32 = 56; memcpy(kbuf + 144, &u32, 4);
    // arg[28] d_W_stride = 1
    u32 = 1; memcpy(kbuf + 148, &u32, 4);
    // arg[29] f_N_stride = 576 (3*3*64)
    u32 = 576; memcpy(kbuf + 152, &u32, 4);
    // arg[30] f_C_stride = 9
    u32 = 9; memcpy(kbuf + 156, &u32, 4);
    // arg[31] f_R_stride = 3
    u32 = 3; memcpy(kbuf + 160, &u32, 4);
    // arg[32] f_S_stride = 1
    u32 = 1; memcpy(kbuf + 164, &u32, 4);
    // arg[33] o_N_stride = 200704
    u32 = 200704; memcpy(kbuf + 168, &u32, 4);
    // arg[34] o_K_stride = 3136
    u32 = 3136; memcpy(kbuf + 172, &u32, 4);
    // arg[35] o_H_stride = 56
    u32 = 56; memcpy(kbuf + 176, &u32, 4);
    // arg[36] o_W_stride = 1
    u32 = 1; memcpy(kbuf + 180, &u32, 4);
    // arg[37] G = 1
    u32 = 1; memcpy(kbuf + 184, &u32, 4);
    // arg[38] d_G_stride = 200704
    u32 = 200704; memcpy(kbuf + 188, &u32, 4);
    // arg[39] f_G_stride = 36864
    u32 = 36864; memcpy(kbuf + 192, &u32, 4);
    // arg[40] o_G_stride = 200704
    u32 = 200704; memcpy(kbuf + 196, &u32, 4);
    // arg[41] activation_mode = 0
    kbuf[200] = 0;
    
    size_t kbuf_sz = 256;
    void* extra[5] = {
        HIP_LAUNCH_PARAM_BUFFER_POINTER, kbuf,
        HIP_LAUNCH_PARAM_BUFFER_SIZE, &kbuf_sz,
        HIP_LAUNCH_PARAM_END
    };
    
    // Try a small grid first (1 block) to see if a tiny launch faults too
    uint32_t grid_x = 24576;
    if (argc > 2) grid_x = (uint32_t)atoi(argv[2]);
    printf("Launching stride1 kernel (grid=%u)...\n", grid_x);
    hipError_t r = hipModuleLaunchKernel(func, grid_x, 1, 1, 256, 1, 1, 0, nullptr, nullptr, extra);
    printf("Launch returned: %d (%s)\n", r, hipGetErrorString(r));
    
    hipError_t sync_r = hipDeviceSynchronize();
    hipError_t last_r = hipGetLastError();
    printf("Sync1: %d, LastError1: %d\n", sync_r, last_r);

    // Second sync to catch WDDM deferred GPU faults
    void* d_probe; hipMalloc(&d_probe, 4);
    hipError_t pr = hipMemset(d_probe, 0, 4);
    hipError_t sync2 = hipDeviceSynchronize();
    hipError_t last2 = hipGetLastError();
    printf("Probe memset: %d, Sync2: %d, LastError2: %d\n", (int)pr, (int)sync2, (int)last2);
    hipFree(d_probe);

    bool ok = (sync_r == hipSuccess && last_r == hipSuccess && sync2 == hipSuccess && last2 == hipSuccess);
    if (ok) {
        printf("SUCCESS!\n");
    } else {
        printf("FAILED with GPU fault (deferred=%d)\n",
               sync_r == hipSuccess && sync2 != hipSuccess);
    }
    
    hipFree(d_in_base); hipFree(d_w_base); hipFree(d_out_base);
    hipModuleUnload(mod);
    free(code);
    return (sync_r != hipSuccess) ? 1 : 0;
}
