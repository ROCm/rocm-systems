// MIT License
//
// Copyright (c) 2023-2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "lib/rocprofiler-sdk/aql/intercept.hpp"

#include "lib/rocprofiler-sdk/hsa/hsa.hpp"

namespace rocprofiler
{
namespace aql
{
std::shared_ptr<const Intercept>
Intercept::create(const std::function<void(HsaApiTable&)>& mod_cb)
{
    return std::make_shared<const Intercept>(mod_cb);
}

Intercept::Intercept(const std::function<void(HsaApiTable&)>& mod_cb)
: _original(rocprofiler::hsa::get_table())
, _modified(rocprofiler::hsa::get_table())
{
    mod_cb(_modified);
};

}  // namespace aql
}  // namespace rocprofiler
