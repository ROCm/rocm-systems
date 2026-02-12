/* Copyright (c) 2026 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE. */

#include "hip_greenctx.hpp"

namespace hip {

hipError_t hipGreenCtxCreate(hipGreenCtx_t* ctx, hipDevResourceDesc_t desc, int device,
                             unsigned int flags) {
  HIP_INIT_API(hipGreenCtxCreate, ctx, desc, device, flags);

  if (ctx != nullptr) {
    *ctx = nullptr;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGreenCtxDestroy(hipGreenCtx_t ctx) {
  HIP_INIT_API(hipGreenCtxDestroy, ctx);

  HIP_RETURN(hipSuccess);
}

hipError_t hipGreenCtxStreamCreate(hipStream_t* stream, hipGreenCtx_t greenctx, unsigned int flags,
                                   int priority) {
  HIP_INIT_API(hipGreenCtxStreamCreate, stream, greenctx, flags, priority);

  if (stream != nullptr) {
    *stream = nullptr;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipStreamGetGreenCtx(hipStream_t hStream, hipGreenCtx_t* greenCtx) {
  HIP_INIT_API(hipStreamGetGreenCtx, hStream, greenCtx);

  if (greenCtx != nullptr) {
    *greenCtx = nullptr;
  }

  HIP_RETURN(hipSuccess);
}

hipError_t hipGreenCtxRecordEvent(hipGreenCtx_t greenCtx, hipEvent_t event) {
  HIP_INIT_API(hipGreenCtxRecordEvent, greenCtx, event);

  HIP_RETURN(hipSuccess);
}

hipError_t hipGreenCtxWaitEvent(hipGreenCtx_t greenCtx, hipEvent_t event) {
  HIP_INIT_API(hipGreenCtxWaitEvent, greenCtx, event);

  HIP_RETURN(hipSuccess);
}

hipError_t hipCtxFromGreenCtx(hipCtx_t* ctx, hipGreenCtx_t greenCtx) {
  HIP_INIT_API(hipCtxFromGreenCtx, ctx, greenCtx);

  if (ctx != nullptr) {
    *ctx = nullptr;
  }

  HIP_RETURN(hipSuccess);
}
}