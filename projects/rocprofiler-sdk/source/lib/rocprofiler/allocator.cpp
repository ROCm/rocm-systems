// MIT License
//
// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
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

#include "lib/rocprofiler/allocator.hpp"
#include "lib/rocprofiler/registration.hpp"

#include <glog/logging.h>

#include <mutex>

namespace rocprofiler
{
namespace common
{
namespace memory
{
void
deleter<allocator::static_data>::operator()() const
{
    // if fully initialized and not yet finalized
    if(registration::get_init_status() > 0 && registration ::get_fini_status() == 0)
    {
        static auto _once = std::atomic_flag{};
        if(!_once.test_and_set()) registration::finalize();
        // above returns false for only first invocation
    }
}
}  // namespace memory
}  // namespace common
}  // namespace rocprofiler
