/*
 * =============================================================================
 *   ROC Runtime Conformance Release License
 * =============================================================================
 * The University of Illinois/NCSA
 * Open Source License (NCSA)
 *
 * Copyright (c) 2017, Advanced Micro Devices, Inc.
 * All rights reserved.
 *
 * Developed by:
 *
 *                 AMD Research and AMD ROC Software Development
 *
 *                 Advanced Micro Devices, Inc.
 *
 *                 www.amd.com
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to
 * deal with the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 *  - Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimers.
 *  - Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimers in
 *    the documentation and/or other materials provided with the distribution.
 *  - Neither the names of <Name of Development Group, Name of Institution>,
 *    nor the names of its contributors may be used to endorse or promote
 *    products derived from this Software without specific prior written
 *    permission.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
 * THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
 * DEALINGS WITH THE SOFTWARE.
 *
 */

/**
 * One instance of this kernel call is a thread.
 * Each thread finds out the segment in which it should look for the element.
 * After that, it checks if the element is between the lower bound and upper
 * bound of its segment. If yes, then this segment becomes the total
 * searchspace for the next pass.
 *
 * To achieve this, it writes the lower bound and upper bound to the output
 * array. In case the element at the left end (lower bound) matches the element
 * we are looking for, that is marked in the output and we no longer need to
 * look any further.
 */
 
__kernel void
binarySearch(__global uint4 * outputArray,
             __const __global uint2  * sortedArray,
             const   unsigned int findMe) {
  unsigned int tid = get_global_id(0);

  // Then we find the elements  for this thread
  uint2 element = sortedArray[tid];


  // If the element to be found does not lie between
  // them, then nothing left to do in this thread
  if((element.x > findMe) || (element.y < findMe)) {
    return;
  } else {
    // However, if the element does lie between the lower
    // and upper bounds of this thread's searchspace
    // we need to narrow down the search further in this
    // search space 
    // The search space for this thread is marked in the
    // output as being the total search space for the next pass
    outputArray[0].x = tid;
    outputArray[0].w = 1;
  }
}


__kernel void
binarySearch_mulkeys(__global int *keys,
                     __global uint *input,
                     const unsigned int numKeys,
                     __global int *output) {

  int gid = get_global_id(0);
  int lBound = gid * 256;
  int uBound = lBound + 255;

  for(int i = 0; i < numKeys; i++) {
    if(keys[i] >= input[lBound] && keys[i] <= input[uBound])
      output[i]=lBound;
  }

}


__kernel void
binarySearch_mulkeysConcurrent(__global uint *keys,
                               __global uint *input,
                               const unsigned int inputSize, // num. of inputs
                               const unsigned int numSubdivisions,
                               __global int *output) {

  int lBound = (get_global_id(0) % numSubdivisions) * (inputSize / numSubdivisions);
  int uBound = lBound + inputSize / numSubdivisions;
  int myKey = keys[get_global_id(0) / numSubdivisions];
  int mid;

  while(uBound >= lBound) {
    mid = (lBound + uBound) / 2;
    if(input[mid] == myKey) {
      output[get_global_id(0) / numSubdivisions] = mid;
      return;
    } else if(input[mid] > myKey) {
      uBound = mid - 1;
    } else {
      lBound = mid + 1;
    }
  }
}
