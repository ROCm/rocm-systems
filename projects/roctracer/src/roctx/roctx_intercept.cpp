/*
Copyright (c) 2018 Advanced Micro Devices, Inc. All rights reserved.

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

#include "inc/roctx.h"
#include "inc/roctracer_roctx.h"
#include "util/logger.h"

#define PUBLIC_API __attribute__((visibility("default")))

///////////////////////////////////////////////////////////////////////////////////////////////////
// Library implementation
//
namespace roctx {

// callbacks table
cb_table_t cb_table;

}  // namespace roctx

///////////////////////////////////////////////////////////////////////////////////////////////////
// Public library methods
//
extern "C" {

PUBLIC_API bool RegisterApiCallback(uint32_t op, void* callback, void* arg) {
  return roctx::cb_table.set(op, reinterpret_cast<activity_rtapi_callback_t>(callback), arg);
}

PUBLIC_API bool RemoveApiCallback(uint32_t op) { return roctx::cb_table.set(op, NULL, NULL); }

}  // extern "C"
