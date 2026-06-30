// MIT License
//
// Copyright (c) 2017-2026 Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef TEST_CTRL_HSA_INIT_H_
#define TEST_CTRL_HSA_INIT_H_

#include <hsa/hsa_ven_amd_aqlprofile.h>

namespace ctrl_test
{
// The migrated hsa_rsrc_factory normally receives the HSA API tables from the
// rocprofiler-sdk intercept via hsa_rsrc_factory_init(). When running the ctrl
// test as a standalone executable there is no intercept, and the rocprofiler
// dlsym fallback is unavailable (its symbols are not linked into this binary),
// so we populate the core/amd-ext tables directly from the HSA runtime we link
// against and feed them to the factory ourselves. Must be called before
// TestHsa::HsaInstantiate().
void
InitHsaTables();

// Returns a v1 aqlprofile API table populated with the in-tree (statically
// linked) hsa_ven_amd_aqlprofile_* implementation. Using this instead of the
// factory's HSA-extension table makes the ctrl test exercise the migrated
// in-tree aqlprofile rather than the installed system shared library.
const hsa_ven_amd_aqlprofile_pfn_t*
InTreeAqlProfileApi();

}  // namespace ctrl_test

#endif  // TEST_CTRL_HSA_INIT_H_
