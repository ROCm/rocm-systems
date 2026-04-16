#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Force maximum VGPR pressure to trigger scratch spill path.
// Uses 256+ float accumulators live simultaneously, all fed from VMEM loads.
// Target: exhaust all VGPRs, forcing aegisbit into scratch_store/scratch_load
// spill of v253/v254/v255 for its trampoline VGPRs.
//
// KNOWN ISSUE: Partial instrumentation — 215/257 sites (83.7%).
//   Kernel uses 344 VGPRs (256 regular + 88 accum), 104 SGPRs.
//   SGPR overflow triggers zero-SGPR mode. 9336-byte kernel grows to
//   128KB+ with trampolines, exceeding s_branch ±128KB range for the
//   last 42 sites.

#define LIVE_BLOCK(arr, base, a0, a1, a2, a3, a4, a5, a6, a7) \
    a0 += arr[tid + (base +  0) * stride]; \
    a1 += arr[tid + (base +  1) * stride]; \
    a2 += arr[tid + (base +  2) * stride]; \
    a3 += arr[tid + (base +  3) * stride]; \
    a4 += arr[tid + (base +  4) * stride]; \
    a5 += arr[tid + (base +  5) * stride]; \
    a6 += arr[tid + (base +  6) * stride]; \
    a7 += arr[tid + (base +  7) * stride];

__global__ __attribute__((amdgpu_flat_work_group_size(64, 64)))
__attribute__((amdgpu_waves_per_eu(1, 1)))
void max_vgpr_pressure(const float* __restrict__ A,
                       const float* __restrict__ B,
                       float* __restrict__ out,
                       int stride) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    // 128 live float accumulators from array A
    float a00=0,a01=0,a02=0,a03=0,a04=0,a05=0,a06=0,a07=0;
    float a08=0,a09=0,a10=0,a11=0,a12=0,a13=0,a14=0,a15=0;
    float a16=0,a17=0,a18=0,a19=0,a20=0,a21=0,a22=0,a23=0;
    float a24=0,a25=0,a26=0,a27=0,a28=0,a29=0,a30=0,a31=0;
    float a32=0,a33=0,a34=0,a35=0,a36=0,a37=0,a38=0,a39=0;
    float a40=0,a41=0,a42=0,a43=0,a44=0,a45=0,a46=0,a47=0;
    float a48=0,a49=0,a50=0,a51=0,a52=0,a53=0,a54=0,a55=0;
    float a56=0,a57=0,a58=0,a59=0,a60=0,a61=0,a62=0,a63=0;
    float a64=0,a65=0,a66=0,a67=0,a68=0,a69=0,a70=0,a71=0;
    float a72=0,a73=0,a74=0,a75=0,a76=0,a77=0,a78=0,a79=0;
    float a80=0,a81=0,a82=0,a83=0,a84=0,a85=0,a86=0,a87=0;
    float a88=0,a89=0,a90=0,a91=0,a92=0,a93=0,a94=0,a95=0;
    float a96=0,a97=0,a98=0,a99=0,aa0=0,aa1=0,aa2=0,aa3=0;
    float aa4=0,aa5=0,aa6=0,aa7=0,aa8=0,aa9=0,ab0=0,ab1=0;
    float ab2=0,ab3=0,ab4=0,ab5=0,ab6=0,ab7=0,ab8=0,ab9=0;
    float ac0=0,ac1=0,ac2=0,ac3=0,ac4=0,ac5=0,ac6=0,ac7=0;

    // 128 live float accumulators from array B
    float b00=0,b01=0,b02=0,b03=0,b04=0,b05=0,b06=0,b07=0;
    float b08=0,b09=0,b10=0,b11=0,b12=0,b13=0,b14=0,b15=0;
    float b16=0,b17=0,b18=0,b19=0,b20=0,b21=0,b22=0,b23=0;
    float b24=0,b25=0,b26=0,b27=0,b28=0,b29=0,b30=0,b31=0;
    float b32=0,b33=0,b34=0,b35=0,b36=0,b37=0,b38=0,b39=0;
    float b40=0,b41=0,b42=0,b43=0,b44=0,b45=0,b46=0,b47=0;
    float b48=0,b49=0,b50=0,b51=0,b52=0,b53=0,b54=0,b55=0;
    float b56=0,b57=0,b58=0,b59=0,b60=0,b61=0,b62=0,b63=0;
    float b64=0,b65=0,b66=0,b67=0,b68=0,b69=0,b70=0,b71=0;
    float b72=0,b73=0,b74=0,b75=0,b76=0,b77=0,b78=0,b79=0;
    float b80=0,b81=0,b82=0,b83=0,b84=0,b85=0,b86=0,b87=0;
    float b88=0,b89=0,b90=0,b91=0,b92=0,b93=0,b94=0,b95=0;
    float b96=0,b97=0,b98=0,b99=0,ba0=0,ba1=0,ba2=0,ba3=0;
    float ba4=0,ba5=0,ba6=0,ba7=0,ba8=0,ba9=0,bb0=0,bb1=0;
    float bb2=0,bb3=0,bb4=0,bb5=0,bb6=0,bb7=0,bb8=0,bb9=0;
    float bc0=0,bc1=0,bc2=0,bc3=0,bc4=0,bc5=0,bc6=0,bc7=0;

    // Load pass 1: fill all 128 A-accumulators (128 VMEM load sites)
    LIVE_BLOCK(A,  0, a00,a01,a02,a03,a04,a05,a06,a07)
    LIVE_BLOCK(A,  8, a08,a09,a10,a11,a12,a13,a14,a15)
    LIVE_BLOCK(A, 16, a16,a17,a18,a19,a20,a21,a22,a23)
    LIVE_BLOCK(A, 24, a24,a25,a26,a27,a28,a29,a30,a31)
    LIVE_BLOCK(A, 32, a32,a33,a34,a35,a36,a37,a38,a39)
    LIVE_BLOCK(A, 40, a40,a41,a42,a43,a44,a45,a46,a47)
    LIVE_BLOCK(A, 48, a48,a49,a50,a51,a52,a53,a54,a55)
    LIVE_BLOCK(A, 56, a56,a57,a58,a59,a60,a61,a62,a63)
    LIVE_BLOCK(A, 64, a64,a65,a66,a67,a68,a69,a70,a71)
    LIVE_BLOCK(A, 72, a72,a73,a74,a75,a76,a77,a78,a79)
    LIVE_BLOCK(A, 80, a80,a81,a82,a83,a84,a85,a86,a87)
    LIVE_BLOCK(A, 88, a88,a89,a90,a91,a92,a93,a94,a95)
    LIVE_BLOCK(A, 96, a96,a97,a98,a99,aa0,aa1,aa2,aa3)
    LIVE_BLOCK(A,104, aa4,aa5,aa6,aa7,aa8,aa9,ab0,ab1)
    LIVE_BLOCK(A,112, ab2,ab3,ab4,ab5,ab6,ab7,ab8,ab9)
    LIVE_BLOCK(A,120, ac0,ac1,ac2,ac3,ac4,ac5,ac6,ac7)

    // Load pass 2: fill all 128 B-accumulators (128 VMEM load sites)
    LIVE_BLOCK(B,  0, b00,b01,b02,b03,b04,b05,b06,b07)
    LIVE_BLOCK(B,  8, b08,b09,b10,b11,b12,b13,b14,b15)
    LIVE_BLOCK(B, 16, b16,b17,b18,b19,b20,b21,b22,b23)
    LIVE_BLOCK(B, 24, b24,b25,b26,b27,b28,b29,b30,b31)
    LIVE_BLOCK(B, 32, b32,b33,b34,b35,b36,b37,b38,b39)
    LIVE_BLOCK(B, 40, b40,b41,b42,b43,b44,b45,b46,b47)
    LIVE_BLOCK(B, 48, b48,b49,b50,b51,b52,b53,b54,b55)
    LIVE_BLOCK(B, 56, b56,b57,b58,b59,b60,b61,b62,b63)
    LIVE_BLOCK(B, 64, b64,b65,b66,b67,b68,b69,b70,b71)
    LIVE_BLOCK(B, 72, b72,b73,b74,b75,b76,b77,b78,b79)
    LIVE_BLOCK(B, 80, b80,b81,b82,b83,b84,b85,b86,b87)
    LIVE_BLOCK(B, 88, b88,b89,b90,b91,b92,b93,b94,b95)
    LIVE_BLOCK(B, 96, b96,b97,b98,b99,ba0,ba1,ba2,ba3)
    LIVE_BLOCK(B,104, ba4,ba5,ba6,ba7,ba8,ba9,bb0,bb1)
    LIVE_BLOCK(B,112, bb2,bb3,bb4,bb5,bb6,bb7,bb8,bb9)
    LIVE_BLOCK(B,120, bc0,bc1,bc2,bc3,bc4,bc5,bc6,bc7)

    // Cross-multiply to keep all 256 values live simultaneously
    float r0  = a00*b00 + a01*b01 + a02*b02 + a03*b03;
    float r1  = a04*b04 + a05*b05 + a06*b06 + a07*b07;
    float r2  = a08*b08 + a09*b09 + a10*b10 + a11*b11;
    float r3  = a12*b12 + a13*b13 + a14*b14 + a15*b15;
    float r4  = a16*b16 + a17*b17 + a18*b18 + a19*b19;
    float r5  = a20*b20 + a21*b21 + a22*b22 + a23*b23;
    float r6  = a24*b24 + a25*b25 + a26*b26 + a27*b27;
    float r7  = a28*b28 + a29*b29 + a30*b30 + a31*b31;
    float r8  = a32*b32 + a33*b33 + a34*b34 + a35*b35;
    float r9  = a36*b36 + a37*b37 + a38*b38 + a39*b39;
    float r10 = a40*b40 + a41*b41 + a42*b42 + a43*b43;
    float r11 = a44*b44 + a45*b45 + a46*b46 + a47*b47;
    float r12 = a48*b48 + a49*b49 + a50*b50 + a51*b51;
    float r13 = a52*b52 + a53*b53 + a54*b54 + a55*b55;
    float r14 = a56*b56 + a57*b57 + a58*b58 + a59*b59;
    float r15 = a60*b60 + a61*b61 + a62*b62 + a63*b63;
    float r16 = a64*b64 + a65*b65 + a66*b66 + a67*b67;
    float r17 = a68*b68 + a69*b69 + a70*b70 + a71*b71;
    float r18 = a72*b72 + a73*b73 + a74*b74 + a75*b75;
    float r19 = a76*b76 + a77*b77 + a78*b78 + a79*b79;
    float r20 = a80*b80 + a81*b81 + a82*b82 + a83*b83;
    float r21 = a84*b84 + a85*b85 + a86*b86 + a87*b87;
    float r22 = a88*b88 + a89*b89 + a90*b90 + a91*b91;
    float r23 = a92*b92 + a93*b93 + a94*b94 + a95*b95;
    float r24 = a96*b96 + a97*b97 + a98*b98 + a99*b99;
    float r25 = aa0*ba0 + aa1*ba1 + aa2*ba2 + aa3*ba3;
    float r26 = aa4*ba4 + aa5*ba5 + aa6*ba6 + aa7*ba7;
    float r27 = aa8*ba8 + aa9*ba9 + ab0*bb0 + ab1*bb1;
    float r28 = ab2*bb2 + ab3*bb3 + ab4*bb4 + ab5*bb5;
    float r29 = ab6*bb6 + ab7*bb7 + ab8*bb8 + ab9*bb9;
    float r30 = ac0*bc0 + ac1*bc1 + ac2*bc2 + ac3*bc3;
    float r31 = ac4*bc4 + ac5*bc5 + ac6*bc6 + ac7*bc7;

    float total = r0+r1+r2+r3+r4+r5+r6+r7+r8+r9+r10+r11+r12+r13+r14+r15
                 +r16+r17+r18+r19+r20+r21+r22+r23+r24+r25+r26+r27+r28+r29+r30+r31;

    out[tid] = total;
}

int main() {
    const int N = 64 * 256;
    const int stride = N;
    const int total_elems = N * 128;

    float *hA = (float*)malloc(total_elems * sizeof(float));
    float *hB = (float*)malloc(total_elems * sizeof(float));
    float *hOut = (float*)malloc(N * sizeof(float));

    srand(42);
    for (int i = 0; i < total_elems; i++) {
        hA[i] = (float)(rand() % 100) / 100.0f;
        hB[i] = (float)(rand() % 100) / 100.0f;
    }

    float *dA, *dB, *dOut;
    hipMalloc(&dA, total_elems * sizeof(float));
    hipMalloc(&dB, total_elems * sizeof(float));
    hipMalloc(&dOut, N * sizeof(float));

    hipMemcpy(dA, hA, total_elems * sizeof(float), hipMemcpyHostToDevice);
    hipMemcpy(dB, hB, total_elems * sizeof(float), hipMemcpyHostToDevice);

    max_vgpr_pressure<<<N/64, 64>>>(dA, dB, dOut, stride);
    hipError_t err = hipDeviceSynchronize();
    if (err != hipSuccess) {
        printf("FAIL: GPU error: %s\n", hipGetErrorString(err));
        return 1;
    }

    hipMemcpy(hOut, dOut, N * sizeof(float), hipMemcpyDeviceToHost);

    int nonzero = 0;
    for (int i = 0; i < N; i++) {
        if (hOut[i] != 0.0f) nonzero++;
    }

    printf("CHECKSUM: nonzero=%d/%d first=%.6f last=%.6f\n",
           nonzero, N, hOut[0], hOut[N-1]);
    printf("PASS\n");

    hipFree(dA); hipFree(dB); hipFree(dOut);
    free(hA); free(hB); free(hOut);
    return 0;
}
