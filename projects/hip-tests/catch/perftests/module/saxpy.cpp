/*
Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:
The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include "saxpy.hpp"

__global__ void saxpy_kernel_0(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_1(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_2(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_3(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_4(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_5(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_6(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_7(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_8(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_9(const float a, const float *d_x, float *d_y,
                               const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_10(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_11(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_12(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_13(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_14(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_15(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_16(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_17(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_18(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_19(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_20(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_21(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_22(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_23(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_24(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_25(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_26(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_27(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_28(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_29(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_30(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_31(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_32(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_33(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_34(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_35(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_36(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_37(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_38(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_39(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_40(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_41(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_42(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_43(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_44(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_45(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_46(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_47(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_48(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_49(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_50(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_51(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_52(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_53(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_54(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_55(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_56(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_57(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_58(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_59(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_60(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_61(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_62(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_63(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_64(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_65(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_66(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_67(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_68(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_69(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_70(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_71(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_72(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_73(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_74(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_75(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_76(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_77(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_78(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_79(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_80(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_81(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_82(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_83(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_84(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_85(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_86(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_87(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_88(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_89(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_90(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_91(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_92(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_93(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_94(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_95(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_96(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_97(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_98(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_99(const float a, const float *d_x, float *d_y,
                                const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_100(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_101(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_102(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_103(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_104(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_105(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_106(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_107(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_108(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_109(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_110(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_111(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_112(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_113(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_114(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_115(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_116(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_117(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_118(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_119(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_120(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_121(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_122(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_123(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_124(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_125(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_126(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_127(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_128(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_129(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_130(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_131(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_132(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_133(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_134(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_135(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_136(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_137(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_138(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_139(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_140(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_141(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_142(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_143(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_144(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_145(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_146(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_147(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_148(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_149(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_150(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_151(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_152(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_153(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_154(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_155(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_156(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_157(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_158(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_159(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_160(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_161(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_162(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_163(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_164(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_165(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_166(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_167(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_168(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_169(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_170(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_171(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_172(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_173(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_174(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_175(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_176(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_177(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_178(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_179(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_180(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_181(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_182(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_183(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_184(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_185(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_186(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_187(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_188(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_189(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_190(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_191(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_192(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_193(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_194(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_195(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_196(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_197(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_198(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_199(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_200(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_201(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_202(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_203(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_204(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_205(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_206(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_207(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_208(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_209(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_210(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_211(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_212(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_213(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_214(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_215(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_216(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_217(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_218(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_219(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_220(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_221(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_222(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_223(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_224(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_225(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_226(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_227(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_228(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_229(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_230(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_231(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_232(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_233(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_234(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_235(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_236(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_237(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_238(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_239(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_240(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_241(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_242(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_243(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_244(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_245(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_246(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_247(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_248(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_249(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_250(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_251(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_252(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_253(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_254(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_255(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_256(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_257(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_258(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_259(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_260(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_261(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_262(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_263(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_264(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_265(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_266(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_267(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_268(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_269(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_270(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_271(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_272(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_273(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_274(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_275(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_276(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_277(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_278(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_279(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_280(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_281(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_282(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_283(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_284(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_285(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_286(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_287(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_288(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_289(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_290(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_291(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_292(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_293(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_294(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_295(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_296(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_297(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_298(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_299(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_300(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_301(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_302(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_303(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_304(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_305(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_306(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_307(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_308(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_309(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_310(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_311(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_312(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_313(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_314(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_315(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_316(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_317(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_318(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_319(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_320(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_321(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_322(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_323(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_324(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_325(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_326(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_327(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_328(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_329(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_330(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_331(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_332(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_333(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_334(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_335(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_336(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_337(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_338(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_339(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_340(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_341(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_342(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_343(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_344(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_345(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_346(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_347(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_348(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_349(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_350(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_351(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_352(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_353(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_354(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_355(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_356(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_357(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_358(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_359(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_360(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_361(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_362(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_363(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_364(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_365(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_366(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_367(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_368(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_369(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_370(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_371(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_372(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_373(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_374(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_375(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_376(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_377(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_378(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_379(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_380(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_381(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_382(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_383(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_384(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_385(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_386(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_387(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_388(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_389(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_390(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_391(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_392(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_393(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_394(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_395(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_396(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_397(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_398(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_399(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_400(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_401(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_402(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_403(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_404(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_405(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_406(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_407(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_408(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_409(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_410(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_411(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_412(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_413(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_414(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_415(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_416(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_417(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_418(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_419(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_420(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_421(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_422(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_423(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_424(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_425(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_426(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_427(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_428(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_429(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_430(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_431(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_432(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_433(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_434(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_435(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_436(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_437(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_438(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_439(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_440(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_441(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_442(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_443(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_444(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_445(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_446(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_447(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_448(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_449(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_450(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_451(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_452(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_453(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_454(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_455(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_456(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_457(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_458(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_459(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_460(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_461(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_462(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_463(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_464(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_465(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_466(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_467(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_468(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_469(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_470(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_471(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_472(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_473(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_474(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_475(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_476(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_477(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_478(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_479(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_480(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_481(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_482(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_483(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_484(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_485(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_486(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_487(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_488(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_489(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_490(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_491(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_492(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_493(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_494(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_495(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_496(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_497(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_498(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_499(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_500(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_501(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_502(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_503(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_504(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_505(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_506(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_507(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_508(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_509(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_510(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}

__global__ void saxpy_kernel_511(const float a, const float *d_x, float *d_y,
                                 const unsigned int size) {
  const unsigned int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

  if (global_idx < size) {
    d_y[global_idx] = a * d_x[global_idx] + d_y[global_idx];
  }
}
