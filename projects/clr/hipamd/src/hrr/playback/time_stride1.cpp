// test_large_buf.cpp - test grid=585 with extra-large buffers
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { hipError_t r=(x); if(r!=hipSuccess){fprintf(stderr,"HIP error %d (%s) at %d\n",r,hipGetErrorString(r),__LINE__);exit(1);} } while(0)

int main(int argc, char** argv) {
    unsigned grid = (argc > 1) ? (unsigned)atoi(argv[1]) : 585;
    // Extra multiplier for buffer sizes
    unsigned mult = (argc > 2) ? (unsigned)atoi(argv[2]) : 1;

    const char* hsaco = "C:/MIGraphX/test/build2/resnet50_capture2.hrr/code_objects/61d0789c47bcaf3bcbe94d85c9b3afa7.hsaco";
    FILE* f = fopen(hsaco,"rb"); if(!f){fprintf(stderr,"No file\n");return 1;}
    fseek(f,0,SEEK_END); size_t sz=ftell(f); rewind(f);
    void* code=malloc(sz); fread(code,1,sz,f); fclose(f);
    
    hipModule_t mod; CHECK(hipModuleLoadData(&mod, code));
    hipFunction_t func;
    CHECK(hipModuleGetFunction(&func,mod,"miopenSp3AsmConv_v30_3_1_gfx11_fp32_f2x3_stride1"));

    // in_sz = BATCHSIZE * C * H * W * 4 (float)
    // For H=56, W=56, C=64, BATCHSIZE=1: 802816 bytes
    // Kernel may access beyond this for certain tile indices
    // Try with 256x larger buffer to cover all possible tile accesses
    size_t in_sz  = 802816 * mult;
    size_t w_sz   = 147456 * mult;
    size_t out_sz = 802816 * mult;
    
    const size_t PAD=4*1024*1024;
    void *di,*dw,*dout;
    CHECK(hipMalloc(&di,  PAD+in_sz +PAD)); CHECK(hipMemset(di,  0,PAD+in_sz +PAD));
    CHECK(hipMalloc(&dw,  PAD+w_sz  +PAD)); CHECK(hipMemset(dw,  0,PAD+w_sz  +PAD));
    CHECK(hipMalloc(&dout,PAD+out_sz+PAD)); CHECK(hipMemset(dout,0,PAD+out_sz+PAD));
    CHECK(hipDeviceSynchronize());
    void *d_in=(uint8_t*)di+PAD, *d_w=(uint8_t*)dw+PAD, *d_out=(uint8_t*)dout+PAD;
    fprintf(stderr,"[test] grid=%u mult=%u  in=%p wei=%p out=%p\n",grid,mult,d_in,d_w,d_out);

    uint8_t kbuf[256]={0};
    uint32_t u32; uint64_t u64;
    u32=1;memcpy(kbuf+0,&u32,4); u32=64;memcpy(kbuf+4,&u32,4);
    u32=56;memcpy(kbuf+8,&u32,4); u32=56;memcpy(kbuf+12,&u32,4);
    u32=64;memcpy(kbuf+16,&u32,4); u32=96;memcpy(kbuf+20,&u32,4);
    u32=0x4600;memcpy(kbuf+24,&u32,4); u32=0;memcpy(kbuf+28,&u32,4);
    u64=(uint64_t)d_in;memcpy(kbuf+32,&u64,8);
    u64=(uint64_t)d_w;memcpy(kbuf+40,&u64,8);
    u64=(uint64_t)d_out;memcpy(kbuf+48,&u64,8);
    u64=0;memcpy(kbuf+56,&u64,8);
    u32=3;memcpy(kbuf+64,&u32,4); u32=3;memcpy(kbuf+68,&u32,4);
    u32=1;memcpy(kbuf+72,&u32,4); u32=1;memcpy(kbuf+76,&u32,4);
    u32=56;memcpy(kbuf+80,&u32,4); u32=56;memcpy(kbuf+84,&u32,4);
    u64=0;memcpy(kbuf+88,&u64,8); u32=0;memcpy(kbuf+96,&u32,4);
    u32=0;memcpy(kbuf+100,&u32,4);
    // offsets 104-135 = 0 (args 21-24 are the "offset" args)
    u32=200704;memcpy(kbuf+136,&u32,4); u32=3136;memcpy(kbuf+140,&u32,4);
    u32=56;memcpy(kbuf+144,&u32,4); u32=1;memcpy(kbuf+148,&u32,4);
    u32=576;memcpy(kbuf+152,&u32,4); u32=9;memcpy(kbuf+156,&u32,4);
    u32=3;memcpy(kbuf+160,&u32,4); u32=1;memcpy(kbuf+164,&u32,4);
    u32=200704;memcpy(kbuf+168,&u32,4); u32=3136;memcpy(kbuf+172,&u32,4);
    u32=56;memcpy(kbuf+176,&u32,4); u32=1;memcpy(kbuf+180,&u32,4);
    u32=1;memcpy(kbuf+184,&u32,4); u32=200704;memcpy(kbuf+188,&u32,4);
    u32=36864;memcpy(kbuf+192,&u32,4); u32=200704;memcpy(kbuf+196,&u32,4);

    size_t ksz=256;
    void* extra[5]={HIP_LAUNCH_PARAM_BUFFER_POINTER,kbuf,HIP_LAUNCH_PARAM_BUFFER_SIZE,&ksz,HIP_LAUNCH_PARAM_END};

    hipError_t r=hipModuleLaunchKernel(func,grid,1,1,256,1,1,0,nullptr,nullptr,extra);
    hipError_t sync_r=hipDeviceSynchronize();
    hipError_t last_r=hipGetLastError();
    
    // Deferred fault check
    void* dp; hipMalloc(&dp,4);
    hipError_t pr=hipMemset(dp,0,4);
    hipError_t s2=hipDeviceSynchronize();
    hipError_t l2=hipGetLastError();
    hipFree(dp);

    bool ok=(r==hipSuccess&&sync_r==hipSuccess&&last_r==hipSuccess&&s2==hipSuccess&&l2==hipSuccess);
    printf("grid=%u mult=%u: launch=%d sync=%d last=%d probe=%d sync2=%d last2=%d  %s\n",
        grid,mult,r,sync_r,last_r,pr,s2,l2,ok?"TRUE_OK":"FAIL");
    
    hipFree(di); hipFree(dw); hipFree(dout);
    hipModuleUnload(mod); free(code);
    return ok?0:1;
}
