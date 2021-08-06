// MIT License
//
// Copyright (c) 2020, The Regents of the University of California,
// through Lawrence Berkeley National Laboratory (subject to receipt of any
// required approvals from the U.S. Dept. of Energy).  All rights reserved.
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

#include "timemory/api/macros.hpp"
#include "timemory/macros/os.hpp"
#include "timemory/mpl/concepts.hpp"

#include <type_traits>

//
// General APIs
//
TIMEMORY_DEFINE_NS_API(project, none)      // dummy type
TIMEMORY_DEFINE_NS_API(project, timemory)  // provided by timemory API exclusively
TIMEMORY_DEFINE_NS_API(project, python)    // provided by timemory python interface
TIMEMORY_DEFINE_NS_API(project, kokkosp)   // kokkos profiling API
//
// Device APIs
//
TIMEMORY_DECLARE_NS_API(device, cpu)  // collects data on CPU
TIMEMORY_DECLARE_NS_API(device, gpu)  // collects data on GPU
//
// Category APIs
//
TIMEMORY_DEFINE_NS_API(category, debugger)          // provided debugging utilities
TIMEMORY_DEFINE_NS_API(category, decorator)         // decorates external profiler
TIMEMORY_DEFINE_NS_API(category, external)          // relies on external package
TIMEMORY_DEFINE_NS_API(category, io)                // collects I/O data
TIMEMORY_DEFINE_NS_API(category, logger)            // logs generic data or messages
TIMEMORY_DEFINE_NS_API(category, hardware_counter)  // collects HW counter data
TIMEMORY_DEFINE_NS_API(category, memory)            // collects memory data
TIMEMORY_DEFINE_NS_API(category, resource_usage)    // collects resource usage data
TIMEMORY_DEFINE_NS_API(category, timing)            // collects timing data
TIMEMORY_DEFINE_NS_API(category, visualization)     // related to viz (currently unused)
//
// External Third-party library APIs
//
TIMEMORY_DEFINE_NS_API(tpls, allinea)
TIMEMORY_DEFINE_NS_API(tpls, caliper)
TIMEMORY_DEFINE_NS_API(tpls, craypat)
TIMEMORY_DEFINE_NS_API(tpls, gotcha)
TIMEMORY_DEFINE_NS_API(tpls, gperftools)
TIMEMORY_DEFINE_NS_API(tpls, intel)
TIMEMORY_DEFINE_NS_API(tpls, likwid)
TIMEMORY_DEFINE_NS_API(tpls, nvidia)
TIMEMORY_DEFINE_NS_API(tpls, openmp)
TIMEMORY_DEFINE_NS_API(tpls, rocm)
TIMEMORY_DEFINE_NS_API(tpls, papi)
TIMEMORY_DEFINE_NS_API(tpls, tau)
//
// OS-specific APIs
//
TIMEMORY_DEFINE_NS_API(os, agnostic)
TIMEMORY_DEFINE_NS_API(os, supports_unix)
TIMEMORY_DEFINE_NS_API(os, supports_linux)
TIMEMORY_DEFINE_NS_API(os, supports_darwin)
TIMEMORY_DEFINE_NS_API(os, supports_windows)
//
//--------------------------------------------------------------------------------------//
//
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, project::timemory, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, project::python, true_type)
//
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::debugger, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::decorator, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::external, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::io, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::logger, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::hardware_counter,
                                 true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::resource_usage,
                                 true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::timing, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, category::visualization,
                                 true_type)
//
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::allinea, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::caliper, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::craypat, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::gotcha, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::gperftools, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::intel, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::likwid, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::nvidia, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::openmp, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::papi, true_type)
TIMEMORY_DEFINE_CONCRETE_CONCEPT(is_runtime_configurable, tpls::tau, true_type)
//
namespace tim
{
namespace api
{
using native_tag = project::timemory;
}
//
namespace trait
{
//
#if !defined(TIMEMORY_UNIX)
template <>
struct is_available<os::supports_unix> : false_type
{};
#endif
//
#if !defined(TIMEMORY_LINUX)
template <>
struct is_available<os::supports_linux> : false_type
{};
#endif
//
#if !defined(TIMEMORY_MACOS)
template <>
struct is_available<os::supports_darwin> : false_type
{};
#endif
//
#if !defined(TIMEMORY_WINDOWS)
template <>
struct is_available<os::supports_windows> : false_type
{};
#endif
//
}  // namespace trait
//
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
namespace tim
{
namespace cereal
{
class JSONInputArchive;
class XMLInputArchive;
class XMLOutputArchive;
}  // namespace cereal
}  // namespace tim
//
//--------------------------------------------------------------------------------------//
//
//                              Default pre-processor settings
//
//--------------------------------------------------------------------------------------//
//

#if !defined(TIMEMORY_DEFAULT_API)
#    define TIMEMORY_DEFAULT_API ::tim::project::timemory
#endif

#if !defined(TIMEMORY_API)
#    define TIMEMORY_API TIMEMORY_DEFAULT_API
#endif

#if defined(DISABLE_TIMEMORY) || defined(TIMEMORY_DISABLED)
#    if !defined(TIMEMORY_DEFAULT_AVAILABLE)
#        define TIMEMORY_DEFAULT_AVAILABLE false_type
#    endif
#else
#    if !defined(TIMEMORY_DEFAULT_AVAILABLE)
#        define TIMEMORY_DEFAULT_AVAILABLE true_type
#    endif
#endif

#if !defined(TIMEMORY_DEFAULT_STATISTICS_TYPE)
#    if defined(TIMEMORY_USE_STATISTICS)
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE true_type
#    else
#        define TIMEMORY_DEFAULT_STATISTICS_TYPE false_type
#    endif
#endif

#if !defined(TIMEMORY_DEFAULT_PLOTTING)
#    define TIMEMORY_DEFAULT_PLOTTING false
#endif

#if !defined(TIMEMORY_DEFAULT_ENABLED)
#    define TIMEMORY_DEFAULT_ENABLED true
#endif

#if !defined(TIMEMORY_PYTHON_PLOTTER)
#    define TIMEMORY_PYTHON_PLOTTER "python"
#endif

#if !defined(TIMEMORY_DEFAULT_INPUT_ARCHIVE)
#    define TIMEMORY_DEFAULT_INPUT_ARCHIVE cereal::JSONInputArchive
#endif

#if !defined(TIMEMORY_DEFAULT_OUTPUT_ARCHIVE)
#    define TIMEMORY_DEFAULT_OUTPUT_ARCHIVE ::tim::type_list<>
#endif

#if !defined(TIMEMORY_INPUT_ARCHIVE)
#    define TIMEMORY_INPUT_ARCHIVE TIMEMORY_DEFAULT_INPUT_ARCHIVE
#endif

#if !defined(TIMEMORY_OUTPUT_ARCHIVE)
#    define TIMEMORY_OUTPUT_ARCHIVE TIMEMORY_DEFAULT_OUTPUT_ARCHIVE
#endif
