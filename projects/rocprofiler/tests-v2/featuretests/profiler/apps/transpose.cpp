/**********************************************************************
Copyright ©2023 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

   Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
   Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <iostream>

#define TILE_DIM 16
#define BLOCK_ROWS 16

__global__ void transposeNaive(float *odata, float *idata, int width, int height) {
  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;

  int index_in = xIndex + width * yIndex;
  int index_out = yIndex + height * xIndex;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i] = idata[index_in + i * width];
  }
}

__global__ void transposeCoalesced(float *odata, float *idata, int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

__global__ void transposeNoBankConflicts(float *odata, float *idata, int width, int height) {
  __shared__ float tile[TILE_DIM][TILE_DIM + 1];

  int xIndex = blockIdx.x * TILE_DIM + threadIdx.x;
  int yIndex = blockIdx.y * TILE_DIM + threadIdx.y;
  int index_in = xIndex + (yIndex)*width;

  xIndex = blockIdx.y * TILE_DIM + threadIdx.x;
  yIndex = blockIdx.x * TILE_DIM + threadIdx.y;
  int index_out = xIndex + (yIndex)*height;

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    tile[threadIdx.y + i][threadIdx.x] = idata[index_in + i * width];
  }

  __syncthreads();

  for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
    odata[index_out + i * height] = tile[threadIdx.x][threadIdx.y + i];
  }
}

/**
 * fillRandom
 * fill array with random values
 */
__global__ void fillRandom(float* arrayPtr, int size) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    uint32_t seed = 12345678;
    while (id < size) {
        seed = (seed<<3) ^ id;
        arrayPtr[id] = seed;
        id += blockDim.x * gridDim.x;
    }
}

int main(int argc, char * argv[])
{
    float* input = nullptr;
    float* output = nullptr;
    size_t width = 1024;
    size_t height = 1024;

    // Set input data to matrix A and matrix B
    hipMalloc((void**)&input, width * height * sizeof(float));
    hipMalloc((void**)&output, width * height * sizeof(float));

    fillRandom<<<256, 256>>>(input, width*height);
    hipDeviceSynchronize();

    dim3 block(width/TILE_DIM,height/TILE_DIM);
    dim3 threads(TILE_DIM,TILE_DIM);

    transposeNoBankConflicts<<<block, threads>>>(
        output,
        input,
        width,
        height
    );

    hipDeviceSynchronize();

    hipFree(input);
    hipFree(output);

    return 0;
}