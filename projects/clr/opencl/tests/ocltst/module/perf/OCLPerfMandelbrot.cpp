/* Copyright (c) 2010 - 2021 Advanced Micro Devices, Inc.

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
 FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 THE SOFTWARE. */

#include "OCLPerfMandelbrot.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

#include "CL/cl.h"
#include "Timer.h"

// Quiet pesky warnings
#ifdef WIN_OS
#define SNPRINTF sprintf_s
#else
#define SNPRINTF snprintf
#endif

typedef struct {
  double x;
  double y;
  double width;
} coordRec;

coordRec coords[] = {
    {0.0, 0.0, 4.0},                                     // Whole set
    {0.0, 0.0, 0.00001},                                 // All black
    {-0.0180789661868, 0.6424294066162, 0.00003824140},  // Hit detail
};

static unsigned int numCoords = sizeof(coords) / sizeof(coordRec);

static const char* float_mandel =
    "__kernel void mandelbrot(__global uint *out, uint width, float xPos, "
    "float yPos, float xStep, float yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % width;\n"
    "    int j = tid / width;\n"
    "    float x0 = (float)(xPos + xStep*i);\n"
    "    float y0 = (float)(yPos + yStep*j);\n"
    "\n"
    "    float x = x0;\n"
    "    float y = y0;\n"
    "\n"
    "    uint iter = 0;\n"
    "    float tmp;\n"
    "    for (iter = 0; (x*x + y*y <= 4.0f) && (iter < maxIter); iter++)\n"
    "    {\n"
    "        tmp = x;\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "    }\n"
    "    out[tid] = iter;\n"
    "}\n";

static const char* float_mandel_vec =
    "__kernel void mandelbrot(__global uint *out, uint width, float xPos, "
    "float yPos, float xStep, float yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % (width/4);\n"
    "    int j = tid / (width/4);\n"
    "    int4 veci = (int4)(4*i, 4*i+1, 4*i+2, 4*i+3);\n"
    "    int4 vecj = (int4)(j, j, j, j);\n"
    "    float4 x0;\n"
    "    x0.s0 = (float)(xPos + xStep*veci.s0);\n"
    "    x0.s1 = (float)(xPos + xStep*veci.s1);\n"
    "    x0.s2 = (float)(xPos + xStep*veci.s2);\n"
    "    x0.s3 = (float)(xPos + xStep*veci.s3);\n"
    "    float4 y0;\n"
    "    y0.s0 = (float)(yPos + yStep*vecj.s0);\n"
    "    y0.s1 = (float)(yPos + yStep*vecj.s1);\n"
    "    y0.s2 = (float)(yPos + yStep*vecj.s2);\n"
    "    y0.s3 = (float)(yPos + yStep*vecj.s3);\n"
    "\n"
    "    float4 x = x0;\n"
    "    float4 y = y0;\n"
    "\n"
    "    uint iter = 0;\n"
    "    float4 tmp;\n"
    "    int4 stay;\n"
    "    int4 ccount = 0;\n"
    "    float4 savx = x;\n"
    "    float4 savy = y;\n"
    "    stay = (x*x+y*y) <= (float4)(4.0f, 4.0f, 4.0f, 4.0f);\n"
    "    for (iter = 0; (stay.s0 | stay.s1 | stay.s2 | stay.s3) && (iter < "
    "maxIter); iter+=16)\n"
    "    {\n"
    "        x = savx;\n"
    "        y = savy;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        stay = (x*x+y*y) <= (float4)(4.0f, 4.0f, 4.0f, 4.0f);\n"
    "        savx = select(savx,x,stay);\n"
    "        savy = select(savy,y,stay);\n"
    "        ccount -= stay*16;\n"
    "    }\n"
    "    // Handle remainder\n"
    "    if (!(stay.s0 & stay.s1 & stay.s2 & stay.s3))\n"
    "    {\n"
    "        iter = 16;\n"
    "        do\n"
    "        {\n"
    "            x = savx;\n"
    "            y = savy;\n"
    "            // More efficient to use scalar ops here: Why?\n"
    "            stay.s0 = ((x.s0*x.s0+y.s0*y.s0) <= 4.0f) && (ccount.s0 < "
    "maxIter);\n"
    "            stay.s1 = ((x.s1*x.s1+y.s1*y.s1) <= 4.0f) && (ccount.s1 < "
    "maxIter);\n"
    "            stay.s2 = ((x.s2*x.s2+y.s2*y.s2) <= 4.0f) && (ccount.s2 < "
    "maxIter);\n"
    "            stay.s3 = ((x.s3*x.s3+y.s3*y.s3) <= 4.0f) && (ccount.s3 < "
    "maxIter);\n"
    "		     tmp = x;\n"
    "            x = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "            y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "            ccount += stay;\n"
    "            iter--;\n"
    "            savx.s0 = (stay.s0 ? x.s0 : savx.s0);\n"
    "            savx.s1 = (stay.s1 ? x.s1 : savx.s1);\n"
    "            savx.s2 = (stay.s2 ? x.s2 : savx.s2);\n"
    "            savx.s3 = (stay.s3 ? x.s3 : savx.s3);\n"
    "            savy.s0 = (stay.s0 ? y.s0 : savy.s0);\n"
    "            savy.s1 = (stay.s1 ? y.s1 : savy.s1);\n"
    "            savy.s2 = (stay.s2 ? y.s2 : savy.s2);\n"
    "            savy.s3 = (stay.s3 ? y.s3 : savy.s3);\n"
    "        } while ((stay.s0 | stay.s1 | stay.s2 | stay.s3) && iter);\n"
    "    }\n"
    "    __global uint4 *vecOut = (__global uint4 *)out;\n"
    "    vecOut[tid] = convert_uint4(ccount);\n"
    "}\n";

static const char* float_mandel_unroll =
    "__kernel void mandelbrot(__global uint *out, uint width, float xPos, "
    "float yPos, float xStep, float yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % width;\n"
    "    int j = tid / width;\n"
    "    float x0 = (float)(xPos + xStep*(float)i);\n"
    "    float y0 = (float)(yPos + yStep*(float)j);\n"
    "\n"
    "    float x = x0;\n"
    "    float y = y0;\n"
    "\n"
    "#define FAST\n"
    "    uint iter = 0;\n"
    "    float tmp;\n"
    "    int stay;\n"
    "    int ccount = 0;\n"
    "    stay = (x*x+y*y) <= 4.0;\n"
    "    float savx = x;\n"
    "    float savy = y;\n"
    "#ifdef FAST\n"
    "    for (iter = 0; (iter < maxIter); iter+=16)\n"
    "#else\n"
    "    for (iter = 0; stay && (iter < maxIter); iter+=16)\n"
    "#endif\n"
    "    {\n"
    "        x = savx;\n"
    "        y = savy;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        stay = (x*x+y*y) <= 4.0;\n"
    "        savx = select(savx,x,stay);\n"
    "        savy = select(savy,y,stay);\n"
    "        ccount += stay*16;\n"
    "#ifdef FAST\n"
    "        if (!stay)\n"
    "            break;\n"
    "#endif\n"
    "    }\n"
    "    // Handle remainder\n"
    "    if (!stay)\n"
    "    {\n"
    "        iter = 16;\n"
    "        do\n"
    "        {\n"
    "            x = savx;\n"
    "            y = savy;\n"
    "            stay = ((x*x+y*y) <= 4.0) && (ccount < maxIter);\n"
    "            tmp = x;\n"
    "            x = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "            y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "            ccount += stay;\n"
    "            iter--;\n"
    "            savx = select(savx,x,stay);\n"
    "            savy = select(savy,y,stay);\n"
    "         } while (stay && iter);\n"
    "    }\n"
    "    out[tid] = (uint)ccount;\n"
    "}\n";

static const char* double_mandel =
    "#ifdef USE_CL_AMD_FP64\n"
    "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
    "#endif\n"
    "#ifdef USE_CL_KHR_FP64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#endif\n"
    "__kernel void mandelbrot(__global uint *out, uint width, double xPos, "
    "double yPos, double xStep, double yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % width;\n"
    "    int j = tid / width;\n"
    "    double x0 = (double)(xPos + xStep*i);\n"
    "    double y0 = (double)(yPos + yStep*j);\n"
    "\n"
    "    double x = x0;\n"
    "    double y = y0;\n"
    "\n"
    "    uint iter = 0;\n"
    "    double tmp;\n"
    "    for (iter = 0; (x*x + y*y <= 4.0) && (iter < maxIter); iter++)\n"
    "    {\n"
    "        tmp = x;\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "    }\n"
    "    out[tid] = iter;\n"
    "}\n";

static const char* double_mandel_unroll =
    "#ifdef USE_CL_AMD_FP64\n"
    "#pragma OPENCL EXTENSION cl_amd_fp64 : enable\n"
    "#endif\n"
    "#ifdef USE_CL_KHR_FP64\n"
    "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
    "#endif\n"
    "__kernel void mandelbrot(__global uint *out, uint width, double xPos, "
    "double yPos, double xStep, double yStep, uint maxIter)\n"
    "{\n"
    "    int tid = get_global_id(0);\n"
    "    int i = tid % width;\n"
    "    int j = tid / width;\n"
    "    double x0 = (double)(xPos + xStep*(double)i);\n"
    "    double y0 = (double)(yPos + yStep*(double)j);\n"
    "\n"
    "    double x = x0;\n"
    "    double y = y0;\n"
    "\n"
    "#define FAST\n"
    "    uint iter = 0;\n"
    "    double tmp;\n"
    "    int stay;\n"
    "    int ccount = 0;\n"
    "    stay = (x*x+y*y) <= 4.0;\n"
    "    double savx = x;\n"
    "    double savy = y;\n"
    "#ifdef FAST\n"
    "    for (iter = 0; (iter < maxIter); iter+=16)\n"
    "#else\n"
    "    for (iter = 0; stay && (iter < maxIter); iter+=16)\n"
    "#endif\n"
    "    {\n"
    "        x = savx;\n"
    "        y = savy;\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        // Two iterations\n"
    "        tmp = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "        y = MUL_ADD_INS(2.0f*x,y,y0);\n"
    "        x = MUL_ADD_INS(-y,y,MUL_ADD_INS(tmp,tmp,x0));\n"
    "        y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "\n"
    "        stay = (x*x+y*y) <= 4.0;\n"
    "        savx = (stay ? x : savx);//select(savx,x,stay);\n"
    "        savy = (stay ? y : savy);//select(savy,y,stay);\n"
    "        ccount += stay*16;\n"
    "#ifdef FAST\n"
    "        if (!stay)\n"
    "            break;\n"
    "#endif\n"
    "    }\n"
    "    // Handle remainder\n"
    "    if (!stay)\n"
    "    {\n"
    "        iter = 16;\n"
    "        do\n"
    "        {\n"
    "            x = savx;\n"
    "            y = savy;\n"
    "            stay = ((x*x+y*y) <= 4.0) && (ccount < maxIter);\n"
    "            tmp = x;\n"
    "            x = MUL_ADD_INS(-y,y,MUL_ADD_INS(x,x,x0));\n"
    "            y = MUL_ADD_INS(2.0f*tmp,y,y0);\n"
    "            ccount += stay;\n"
    "            iter--;\n"
    "            savx = (stay ? x : savx);//select(savx,x,stay);\n"
    "            savy = (stay ? y : savy);//select(savy,y,stay);\n"
    "         } while (stay && iter);\n"
    "    }\n"
    "    out[tid] = (uint)ccount;\n"
    "}\n";

static const unsigned int FMA_EXPECTEDVALUES_INDEX = 15;

// Expected results for each kernel run at each coord
unsigned long long expectedIters[] = {
    203277748ull, 2147483648ull, 120254651ull, 203277748ull, 2147483648ull, 120254651ull,
    203277748ull, 2147483648ull, 120254651ull, 203315114ull, 2147483648ull, 120042599ull,
    203315114ull, 2147483648ull, 120042599ull, 203280620ull, 2147483648ull, 120485704ull,
    203280620ull, 2147483648ull, 120485704ull, 203280620ull, 2147483648ull, 120485704ull,
    203315114ull, 2147483648ull, 120042599ull, 203315114ull, 2147483648ull, 120042599ull};

// nvidia supports CL_KHR_FP64, so they get better results for doubles.  Not
// sure why we differ in floats though
unsigned long long expectedItersNV[] = {
    203277748ull,  2147483648ull, 120254651ull,  203277748ull,  2147483648ull,
    120254651ull,  203277748ull,  2147483648ull, 120254651ull,  203315226ull,
    2147483648ull, 120091921ull,  203315226ull,  2147483648ull, 120091921ull,  // end of mad
    203280620ull,  2147483648ull, 120485704ull,  203280620ull,  2147483648ull,
    120485704ull,  203280620ull,  2147483648ull, 120485704ull,  203315114ull,
    2147483648ull, 120042599ull,  203315114ull,  2147483648ull, 120042599ull};

const char* shaderStr[] = {"        float_mad", " float_vector_mad", " float_unroll_mad",
                           "       double_mad", "double_unroll_mad", "        float_fma",
                           " float_vector_fma", " float_unroll_fma", "       double_fma",
                           "double_unroll_fma"};

OCLPerfMandelbrot::OCLPerfMandelbrot() { _numSubTests = 10 * numCoords; }

OCLPerfMandelbrot::~OCLPerfMandelbrot() {}

void OCLPerfMandelbrot::setData(cl_mem buffer, unsigned int val) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_WRITE, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_ * width_; i++) data[i] = val;
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

void OCLPerfMandelbrot::checkData(cl_mem buffer) {
  unsigned int* data = (unsigned int*)_wrapper->clEnqueueMapBuffer(
      cmd_queue_, buffer, true, CL_MAP_READ, 0, bufSize_, 0, NULL, NULL, &error_);
  for (unsigned int i = 0; i < width_ * width_; i++) {
    totalIters += data[i];
  }
  error_ = _wrapper->clEnqueueUnmapMemObject(cmd_queue_, buffer, data, 0, NULL, NULL);
}

static void CL_CALLBACK notify_callback(const char* errinfo, const void* private_info, size_t cb,
                                        void* user_data) {}

void OCLPerfMandelbrot::open(unsigned int test, char* units, double& conversion,
                             unsigned int deviceId) {
  cl_uint numPlatforms;
  cl_platform_id platform = NULL;
  cl_uint num_devices = 0;
  cl_device_id* devices = NULL;
  device = NULL;
  _crcword = 0;
  conversion = 1.0f;
  _deviceId = deviceId;
  _openTest = test;
  skip = false;
  totalIters = 0;
  isAMD = false;

  context_ = 0;
  cmd_queue_ = 0;
  program_ = 0;
  kernel_ = 0;
  outBuffer_ = 0;

  // Maximum iteration count
  // NOTE: Some kernels are unrolled 16 times, so make sure maxIter is divisible
  // by 16 NOTE: Can increase to get better peak performance numbers, but be
  // sure not to TDR slow ASICs!
  unsigned int maxIter = 32768;

  // NOTE: Width needs to be divisible by 4 because the float_mandel_vec kernel
  // processes 4 pixels at once NOTE: Can increase to get better peak
  // performance numbers, but be sure not to TDR slow ASICs!
  width_ = 256;

  // We compute a square domain
  bufSize_ = width_ * width_ * sizeof(cl_uint);

  error_ = _wrapper->clGetPlatformIDs(0, NULL, &numPlatforms);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
  if (0 < numPlatforms) {
    cl_platform_id* platforms = new cl_platform_id[numPlatforms];
    error_ = _wrapper->clGetPlatformIDs(numPlatforms, platforms, NULL);
    CHECK_RESULT(error_ != CL_SUCCESS, "clGetPlatformIDs failed");
    // Get last for default
#if 0
        platform = platforms[numPlatforms-1];
        for (unsigned i = 0; i < numPlatforms; ++i) {
#endif
    char pbuf[100];
    error_ = _wrapper->clGetPlatformInfo(platforms[_platformIndex], CL_PLATFORM_VENDOR,
                                         sizeof(pbuf), pbuf, NULL);
#if 0
            if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
                platform = platforms[i];
                break;
            }
#endif
    num_devices = 0;
    /* Get the number of requested devices */
    error_ = _wrapper->clGetDeviceIDs(platforms[_platformIndex], type_, 0, NULL, &num_devices);
    // Runtime returns an error when no GPU devices are present instead of just
    // returning 0 devices
    // CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");
    // Choose platform with GPU devices
    if (num_devices > 0) {
      if (!strcmp(pbuf, "Advanced Micro Devices, Inc.")) {
        isAMD = true;
      }
      platform = platforms[_platformIndex];
    }
#if 0
        }
#endif
    delete platforms;
  }
  /*
   * If we could find our platform, use it. If not, die as we need the AMD
   * platform for these extensions.
   */
  CHECK_RESULT(platform == 0, "Couldn't find platform with GPU devices, cannot proceed");

  devices = (cl_device_id*)malloc(num_devices * sizeof(cl_device_id));
  CHECK_RESULT(devices == 0, "no devices");

  /* Get the requested device */
  error_ = _wrapper->clGetDeviceIDs(platform, type_, num_devices, devices, NULL);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceIDs failed");

  CHECK_RESULT(_deviceId >= num_devices, "Requested deviceID not available");
  device = devices[_deviceId];

  context_ = _wrapper->clCreateContext(NULL, 1, &device, notify_callback, NULL, &error_);
  CHECK_RESULT(context_ == 0, "clCreateContext failed");

  char charbuf[1024];
  size_t retsize;
  error_ = _wrapper->clGetDeviceInfo(device, CL_DEVICE_EXTENSIONS, 1024, charbuf, &retsize);
  CHECK_RESULT(error_ != CL_SUCCESS, "clGetDeviceInfo failed");

  doubleSupport = false;

  char* p = strstr(charbuf, "cl_amd_fp64");
  char* p2 = strstr(charbuf, "cl_khr_fp64");

  if (p || p2)
    doubleSupport = true;
  else
    doubleSupport = false;

  cmd_queue_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue_ == 0, "clCreateCommandQueue failed");

  outBuffer_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer) failed");

  const char* tmp;
  shaderIdx = _openTest / numCoords;
  if ((doubleSupport != true) &&
      ((shaderIdx == 3) || (shaderIdx == 4) || (shaderIdx == 8) || (shaderIdx == 9))) {
    // We don't support doubles, so skip those tests
    skip = true;
    _perfInfo = 0.0f;
    return;
  }

  if (shaderIdx == 0 || shaderIdx == 5) {
    tmp = float_mandel;
  } else if (shaderIdx == 1 || shaderIdx == 6) {
    tmp = float_mandel_vec;
  } else if (shaderIdx == 2 || shaderIdx == 7) {
    tmp = float_mandel_unroll;
  } else if (shaderIdx == 3 || shaderIdx == 8) {
    tmp = double_mandel;
  } else {
    tmp = double_mandel_unroll;
  }
  std::string curr(tmp);
  std::string searchString("MUL_ADD_INS");
  std::string replaceString;
  if (shaderIdx < 5) {
    replaceString = "mad";
  } else {
    replaceString = "fma";
  }

  std::string::size_type pos = 0;
  while ((pos = curr.find(searchString, pos)) != std::string::npos) {
    curr.replace(pos, searchString.size(), replaceString);
    pos++;
  }

  tmp = curr.c_str();

  program_ = _wrapper->clCreateProgramWithSource(context_, 1, (const char**)&tmp, NULL, &error_);
  CHECK_RESULT(program_ == 0, "clCreateProgramWithSource failed");

  const char* buildOps = NULL;
  if (p)
    buildOps = "-DUSE_CL_AMD_FP64";
  else if (p2)
    buildOps = "-DUSE_CL_KHR_FP64";
  error_ = _wrapper->clBuildProgram(program_, 1, &device, buildOps, NULL, NULL);

  if (error_ != CL_SUCCESS) {
    cl_int intError;
    char log[16384];
    intError = _wrapper->clGetProgramBuildInfo(program_, device, CL_PROGRAM_BUILD_LOG,
                                               16384 * sizeof(char), log, NULL);
    printf("Build error -> %s\n", log);

    CHECK_RESULT(0, "clBuildProgram failed");
  }
  kernel_ = _wrapper->clCreateKernel(program_, "mandelbrot", &error_);
  CHECK_RESULT(kernel_ == 0, "clCreateKernel failed");

  coordIdx = _openTest % numCoords;
  if ((shaderIdx == 0) || (shaderIdx == 1) || (shaderIdx == 2) || (shaderIdx == 5) ||
      (shaderIdx == 6) || (shaderIdx == 7)) {
    float xStep = (float)(coords[coordIdx].width / (double)width_);
    float yStep = (float)(-coords[coordIdx].width / (double)width_);
    float xPos = (float)(coords[coordIdx].x - 0.5 * coords[coordIdx].width);
    float yPos = (float)(coords[coordIdx].y + 0.5 * coords[coordIdx].width);
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_uint), (void*)&width_);
    error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_float), (void*)&xPos);
    error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_float), (void*)&yPos);
    error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_float), (void*)&xStep);
    error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_float), (void*)&yStep);
    error_ = _wrapper->clSetKernelArg(kernel_, 6, sizeof(cl_uint), (void*)&maxIter);
  } else {
    double xStep = coords[coordIdx].width / (double)width_;
    double yStep = -coords[coordIdx].width / (double)width_;
    double xPos = coords[coordIdx].x - 0.5 * coords[coordIdx].width;
    double yPos = coords[coordIdx].y + 0.5 * coords[coordIdx].width;
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
    error_ = _wrapper->clSetKernelArg(kernel_, 1, sizeof(cl_uint), (void*)&width_);
    error_ = _wrapper->clSetKernelArg(kernel_, 2, sizeof(cl_double), (void*)&xPos);
    error_ = _wrapper->clSetKernelArg(kernel_, 3, sizeof(cl_double), (void*)&yPos);
    error_ = _wrapper->clSetKernelArg(kernel_, 4, sizeof(cl_double), (void*)&xStep);
    error_ = _wrapper->clSetKernelArg(kernel_, 5, sizeof(cl_double), (void*)&yStep);
    error_ = _wrapper->clSetKernelArg(kernel_, 6, sizeof(cl_uint), (void*)&maxIter);
  }
  setData(outBuffer_, 0xdeadbeef);
}

void OCLPerfMandelbrot::run(void) {
  if (skip) return;
  int global = width_ * width_;
  // We handle 4 pixels per thread
  if ((shaderIdx == 1) || (shaderIdx == 6)) global >>= 2;
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  // Warm-up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  double totalTime = 0.0;

  for (unsigned int k = 0; k < numLoops; k++) {
    CPerfCounter timer;

    timer.Reset();
    timer.Start();
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    _wrapper->clFinish(cmd_queue_);

    timer.Stop();
    double sec = timer.GetElapsedTime();
    totalTime += sec;
  }

  checkData(outBuffer_);
  // Compute GFLOPS.  There are 7 FLOPs per iteration
  double perf = ((double)totalIters * 7 * (double)(1e-09)) / (totalTime / (double)numLoops);

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " %s (GFLOPS) ", shaderStr[shaderIdx]);
  testDescString = buf;
  // Dump iteration count
  // printf(" totalIter = %lld\n", totalIters);
  if (isAMD && (type_ == CL_DEVICE_TYPE_GPU)) {
    CHECK_RESULT((totalIters != expectedIters[_openTest]) &&
                     (totalIters != expectedIters[(_openTest < FMA_EXPECTEDVALUES_INDEX
                                                       ? _openTest + FMA_EXPECTEDVALUES_INDEX
                                                       : _openTest)]),
                 "Incorrect iteration count detected!");
  } else {
    CHECK_RESULT(totalIters != expectedItersNV[_openTest], "Incorrect iteration count detected!");
  }
}

unsigned int OCLPerfMandelbrot::close(void) {
  if (outBuffer_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer_) failed");
  }
  if (kernel_) {
    error_ = _wrapper->clReleaseKernel(kernel_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseKernel failed");
  }
  if (program_) {
    error_ = _wrapper->clReleaseProgram(program_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseProgram failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  if (context_) {
    error_ = _wrapper->clReleaseContext(context_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseContext failed");
  }

  return _crcword;
}

OCLPerfAsyncMandelbrot::OCLPerfAsyncMandelbrot() {}

OCLPerfAsyncMandelbrot::~OCLPerfAsyncMandelbrot() {}

void OCLPerfAsyncMandelbrot::open(unsigned int test, char* units, double& conversion,
                                  unsigned int deviceId) {
  // Create common items first
  OCLPerfMandelbrot::open(test, units, conversion, deviceId);

  // Create resources for async test
  cmd_queue2_ = _wrapper->clCreateCommandQueue(context_, device, 0, NULL);
  CHECK_RESULT(cmd_queue2_ == 0, "clCreateCommandQueue failed");

  outBuffer2_ = _wrapper->clCreateBuffer(context_, 0, bufSize_, NULL, &error_);
  CHECK_RESULT(outBuffer_ == 0, "clCreateBuffer(outBuffer2) failed");
}

void OCLPerfAsyncMandelbrot::run(void) {
  if (skip) return;
  int global = width_ * width_;
  // We handle 4 pixels per thread
  if ((shaderIdx == 1) || (shaderIdx == 6)) global >>= 2;
  int local = 64;

  size_t global_work_size[1] = {(size_t)global};
  size_t local_work_size[1] = {(size_t)local};

  // Warm-up
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue_);

  error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer2_);
  error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue2_, kernel_, 1, NULL,
                                            (const size_t*)global_work_size,
                                            (const size_t*)local_work_size, 0, NULL, NULL);

  CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
  _wrapper->clFinish(cmd_queue2_);

  double totalTime = 0.0;

  for (unsigned int k = 0; k < numLoops; k++) {
    CPerfCounter timer;

    timer.Reset();
    timer.Start();
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer_);
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    error_ = _wrapper->clSetKernelArg(kernel_, 0, sizeof(cl_mem), (void*)&outBuffer2_);
    error_ = _wrapper->clEnqueueNDRangeKernel(cmd_queue2_, kernel_, 1, NULL,
                                              (const size_t*)global_work_size,
                                              (const size_t*)local_work_size, 0, NULL, NULL);

    CHECK_RESULT(error_, "clEnqueueNDRangeKernel failed");
    _wrapper->clFlush(cmd_queue_);
    _wrapper->clFlush(cmd_queue2_);
    _wrapper->clFinish(cmd_queue_);
    _wrapper->clFinish(cmd_queue2_);

    timer.Stop();
    double sec = timer.GetElapsedTime();
    totalTime += sec;
  }

  checkData(outBuffer_);
  checkData(outBuffer2_);
  // Compute GFLOPS.  There are 7 FLOPs per iteration
  double perf = ((double)(totalIters * 7) * (double)(1e-09)) / (totalTime / (double)numLoops);

  _perfInfo = (float)perf;
  char buf[256];
  SNPRINTF(buf, sizeof(buf), " async %s (GFLOPS) ", shaderStr[shaderIdx]);
  testDescString = buf;
  // Dump iteration count
  // printf(" totalIter = %lld\n", totalIters);
  if (isAMD && (type_ == CL_DEVICE_TYPE_GPU)) {
    CHECK_RESULT((totalIters != 2 * expectedIters[_openTest]) &&
                     (totalIters != 2 * expectedIters[(_openTest < FMA_EXPECTEDVALUES_INDEX
                                                           ? _openTest + FMA_EXPECTEDVALUES_INDEX
                                                           : _openTest)]),
                 "Incorrect iteration count detected!");
  } else {
    CHECK_RESULT(totalIters != 2 * expectedItersNV[_openTest],
                 "Incorrect iteration count detected!");
  }
}

unsigned int OCLPerfAsyncMandelbrot::close(void) {
  _wrapper->clFinish(cmd_queue_);
  _wrapper->clFinish(cmd_queue2_);

  // Clean up async test items
  if (outBuffer2_) {
    error_ = _wrapper->clReleaseMemObject(outBuffer2_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseMemObject(outBuffer2_) failed");
  }
  if (cmd_queue_) {
    error_ = _wrapper->clReleaseCommandQueue(cmd_queue2_);
    CHECK_RESULT_NO_RETURN(error_ != CL_SUCCESS, "clReleaseCommandQueue failed");
  }
  // Clean up the rest
  return OCLPerfMandelbrot::close();
}
