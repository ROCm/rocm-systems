/**********************************************************************
Copyright ©2015 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

/**
 * The kernel has two implementation of convolution. 
 * 1. Non-Separable Convolution 
 * 2. Separable Convolution
*/


/**
 * NonSeparableConvolution
 * is where each pixel of the output image
 * is the weighted sum of the neighbourhood pixels of the input image
 * The neighbourhood is defined by the dimensions of the mask and 
 * weight of each neighbour is defined by the mask itself.
 * @param input  Padded Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param output Output matrix after performing convolution
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 * @param nExWidth		  Size of padded input width
 */

__kernel void simpleNonSeparableConvolution(__global  uint  * input,
											__global  float  * mask,
											__global  int  * output,
											const     uint2  inputDimensions,
											const     uint2  maskDimensions,
											const     uint	 nExWidth)
{
    uint tid   = get_global_id(0);
    
    uint width  = inputDimensions.x;
    uint height = inputDimensions.y;
    
    uint x      = tid%width;
    uint y      = tid/width;
    
    uint maskWidth  = maskDimensions.x;
    uint maskHeight = maskDimensions.y;

    if(x >= width || y >= height)
		return;

    /*
     * initializing weighted sum value
     */
    float sumFX = 0.0f;
	int m = 0, n = 0;
  
    //performing weighted sum within the mask boundaries
    for(uint j = y ; j < (y + maskHeight); ++j, m++)    
    {
		n = 0;
		for(uint i = x; i < (x + maskWidth); ++i, n++)
		{ 
            uint maskIndex = m * maskWidth  + n;
            uint index     = j * nExWidth + i;
            
            sumFX += ((float)input[index] * mask[maskIndex]);
        }
	}

    sumFX += 0.5f;
	output[tid] = (int)sumFX;
}




/**
 * SeparableConvolution 
 * is product of 2 one-dimensional convolution.
 * A 2-dimensional convolution operation is separated into 2 one one-dimensional convolution.
 * SeparableConvolution is implemented in two passes.
 * The first pass is called Row-wise convolution.
 * And second pass is called Column-wise convolution.
 */

 /**
 * First Pass - Row-wise convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param rowFilter rowFilter vector using which row-wise convolution was to be performed
 * @param tmpOutput Output matrix after performing first pass convolution
 * @param inputDimensions dimensions of the input matrix
 * @param filterSize  length of row filter vector
 * @param exInputDimensions	  dimensions of padded input
 */
 __kernel void simpleSeparableConvolutionPass1(__global  uint  * input,
											   __global  float  * rowFilter,
											   __global  float  * tmpOutput,
											   const     uint2  inputDimensions,
											   const     uint  filterSize,
											   const	 uint2  exInputDimensions)
{
	int i = 0, cnt = 0;
	
    uint width  = inputDimensions.x;
    uint height = inputDimensions.y;
    
	uint tid    = get_global_id(0);
    uint x      = tid%width;
    uint y      = tid/width;
   
   if(x >= width || y >= (height+filterSize-1))
		return;

	/*
     * initializing weighted sum value
    */
    float sum = 0.0f;

    for(uint i = x; i < (x + filterSize); ++i) {
        sum = mad((float)input[y * exInputDimensions.x + i], rowFilter[cnt++], sum);        
    }

    /* Transposed save */
	tmpOutput[x * exInputDimensions.y + y] = sum;
}

/**
 * Second Pass - Column-wise convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param colFilter colFilter vector using which column-wise convolution was to be performed
 * @param Output Output matrix after performing second pass convolution
 * @param inputDimensions dimensions of the input matrix
 * @param filterSize  length of col filter vector
 * @param exInputDimensions	  dimensions of padded input
 */
 __kernel void simpleSeparableConvolutionPass2(__global  float  * input,
											   __global  float  * colFilter,
											   __global  int  * output,
											   const     uint2  inputDimensions,
											   const     uint  filterSize,
											   const     uint2  exInputDimensions)
{
	int i = 0, cnt = 0;
	
    uint width  = inputDimensions.x;
    uint height = inputDimensions.y;
    
	uint tid    = get_global_id(0);
	uint x      = tid%height;
    uint y      = tid/height;
   
	if(y >= width || x >= height)
		return;

    /*
     * initializing wighted sum value
    */
    float sum = 0.0f;

	for(uint i = x; i < (x + filterSize); ++i) {
        sum = mad(input[y * exInputDimensions.y + i], colFilter[cnt++], sum);        
    }

    /* Tranposed save */
	sum += 0.5f;
	output[x * width + y] = (int)sum;
}
