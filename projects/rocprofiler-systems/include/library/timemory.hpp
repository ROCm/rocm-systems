// MIT License
//
// Copyright (c) 2022 Advanced Micro Devices, Inc. All Rights Reserved.
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

#pragma once

#include "library/components/fwd.hpp"
#include "library/defines.hpp"

#include <timemory/api.hpp>
#include <timemory/backends/mpi.hpp>
#include <timemory/backends/process.hpp>
#include <timemory/backends/threading.hpp>
#include <timemory/components.hpp>
#include <timemory/components/gotcha/mpip.hpp>
#include <timemory/config.hpp>
#include <timemory/environment.hpp>
#include <timemory/manager.hpp>
#include <timemory/mpl.hpp>
#include <timemory/operations.hpp>
#include <timemory/runtime.hpp>
#include <timemory/settings.hpp>
#include <timemory/storage.hpp>
#include <timemory/variadic.hpp>

namespace omnitrace
{
namespace audit     = tim::audit;      // NOLINT
namespace comp      = tim::component;  // NOLINT
namespace quirk     = tim::quirk;      // NOLINT
namespace threading = tim::threading;  // NOLINT
namespace scope     = tim::scope;      // NOLINT
namespace dmp       = tim::dmp;        // NOLINT
namespace process   = tim::process;    // NOLINT
namespace units     = tim::units;      // NOLINT
namespace trait     = tim::trait;      // NOLINT

// same sort of functionality as python's " ".join([...])
#if !defined(JOIN)
#    define JOIN(...) tim::mpl::apply<std::string>::join(__VA_ARGS__)
#endif
}  // namespace omnitrace
