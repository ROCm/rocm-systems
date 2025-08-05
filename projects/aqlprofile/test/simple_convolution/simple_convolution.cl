

/**
 * simple_convolution is where each pixel of the output image
 * is the weighted sum of the neighborhood pixels of the input image
 * The neighborhood is defined by the dimensions of the mask and
 * weight of each neighbor is defined by the mask itself.
 * @param output Output matrix after performing convolution
 * @param input  Input  matrix on which convolution is to be performed
 * @param mask   mask matrix using which convolution was to be performed
 * @param inputDimensions dimensions of the input matrix
 * @param maskDimensions  dimensions of the mask matrix
 */
__kernel void simple_convolution(__global  uint  * output,
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
    for(uint j = top; j <= bottom; ++j) {
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
