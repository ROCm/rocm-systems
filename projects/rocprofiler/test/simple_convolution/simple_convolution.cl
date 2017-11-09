/******************************************************************************

Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list
of conditions and the following disclaimer.

Redistributions in binary form must reproduce the above copyright notice, this
list of conditions and the following disclaimer in the documentation and/or
other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
OF THE POSSIBILITY OF SUCH DAMAGE.

********************************************************************************/

/**
 * SimpleConvolution is where each pixel of the output image
 * is the weighted sum of the neighborhood pixels of the input image
 * The neighborhood is defined by the dimensions of the mask and 
 * weight of each neighbor is defined by the mask itself.
 * @param output Output matrix after performing convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 */
__kernel void SimpleConvolution(__global  uint  * output,
                                __global  uint  * input,
                                __global  float  * mask,
                                const     uint2  inputDimensions,
                                const     uint2  maskDimensions) {

  uint tid   = get_global_id(0);

  uint width  = inputDimensions.x;
  uint height = inputDimensions.y;

  uint x      = tid%width;
  uint y      = tid/width;

  uint maskWidth  = maskDimensions.x;
  uint maskHeight = maskDimensions.y;

  uint vstep = (maskWidth  -1)/2;
  uint hstep = (maskHeight -1)/2;

  // find the left, right, top and bottom indices such that
  // the indices do not go beyond image boundaires
  uint left    = (x           <  vstep) ? 0         : (x - vstep);
  uint right   = ((x + vstep) >= width) ? width - 1 : (x + vstep); 
  uint top     = (y           <  hstep) ? 0         : (y - hstep);
  uint bottom  = ((y + hstep) >= height)? height - 1: (y + hstep); 

  // initializing wighted sum value
  float sumFX = 0;

  for(uint i = left; i <= right; ++i) {
    for(uint j = top ; j <= bottom; ++j) {
      // performing wighted sum within the mask boundaries
      uint maskIndex = (j - (y - hstep)) * maskWidth  + (i - (x - vstep));
      uint index     = j                 * width      + i;
      sumFX += ((float)input[index] * mask[maskIndex]);
    }
  }

  // To round to the nearest integer
  sumFX += 0.5f;
  output[tid] = (uint)sumFX;
}
