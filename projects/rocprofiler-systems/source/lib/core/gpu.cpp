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

#if !defined(OMNITRACE_USE_ROCM_SMI)
#    define OMNITRACE_USE_ROCM_SMI 0
#endif

#if !defined(OMNITRACE_USE_HIP)
#    define OMNITRACE_USE_HIP 0
#endif

#if OMNITRACE_USE_HIP > 0
#    if !defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_USE_HIP 1
#    endif
#endif

#include "gpu.hpp"
#include "debug.hpp"
#include "defines.hpp"

#include <timemory/manager.hpp>

#if OMNITRACE_USE_ROCM_SMI > 0
#    include <rocm_smi/rocm_smi.h>
#endif

#if OMNITRACE_USE_HIP > 0
#    include <hip/hip_runtime.h>
#    include <hip/hip_runtime_api.h>
#    include <timemory/components/hip/backends.hpp>

#    if !defined(OMNITRACE_HIP_RUNTIME_CALL)
#        define OMNITRACE_HIP_RUNTIME_CALL(err)                                          \
            {                                                                            \
                if(err != ::tim::hip::success_v && (int) err != 0)                       \
                {                                                                        \
                    OMNITRACE_THROW(                                                     \
                        "[%s:%d] Warning! HIP API call failed with code %i :: %s\n",     \
                        __FILE__, __LINE__, (int) err, hipGetErrorString(err));          \
                }                                                                        \
            }
#    endif
#endif

namespace omnitrace
{
namespace gpu
{
namespace
{
namespace scope = ::tim::scope;

#if OMNITRACE_USE_ROCM_SMI > 0
#    define OMNITRACE_ROCM_SMI_CALL(ERROR_CODE)                                          \
        ::omnitrace::gpu::check_rsmi_error(ERROR_CODE, __FILE__, __LINE__)

void
check_rsmi_error(rsmi_status_t _code, const char* _file, int _line)
{
    if(_code == RSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = rsmi_status_string(_code, &_msg);
    if(_err != RSMI_STATUS_SUCCESS)
        OMNITRACE_THROW("rsmi_status_string failed. No error message available. "
                        "Error code %i originated at %s:%i\n",
                        static_cast<int>(_code), _file, _line);
    OMNITRACE_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                    _msg);
}

bool
rsmi_init()
{
    auto _rsmi_init = []() {
        try
        {
            OMNITRACE_ROCM_SMI_CALL(::rsmi_init(0));
        } catch(std::exception& _e)
        {
            OMNITRACE_BASIC_VERBOSE(1, "Exception thrown initializing rocm-smi: %s\n",
                                    _e.what());
            return false;
        }
        return true;
    }();

    return _rsmi_init;
}
#endif
}  // namespace

int
hip_device_count()
{
#if OMNITRACE_USE_HIP > 0
    return ::tim::hip::device_count();
#else
    return 0;
#endif
}

int
rsmi_device_count()
{
#if OMNITRACE_USE_ROCM_SMI > 0
    if(!rsmi_init()) return 0;

    static auto _num_devices = []() {
        uint32_t _v = 0;
        try
        {
            OMNITRACE_ROCM_SMI_CALL(rsmi_num_monitor_devices(&_v));
        } catch(std::exception& _e)
        {
            OMNITRACE_BASIC_VERBOSE(
                1, "Exception thrown getting the rocm-smi devices: %s\n", _e.what());
        }
        return _v;
    }();

    return _num_devices;
#else
    return 0;
#endif
}

int
device_count()
{
#if OMNITRACE_USE_ROCM_SMI > 0
    // store as static since calls after rsmi_shutdown will return zero
    return rsmi_device_count();
#elif OMNITRACE_USE_HIP > 0
    return ::tim::hip::device_count();
#else
    return 0;
#endif
}

template <typename ArchiveT>
void
add_hip_device_metadata(ArchiveT& ar)
{
#if OMNITRACE_USE_HIP > 0
    namespace cereal = tim::cereal;
    using cereal::make_nvp;
    using intvec_t = std::vector<int>;

    int        _device_count     = 0;
    int        _current_device   = 0;
    hipError_t _device_count_err = hipGetDeviceCount(&_device_count);

    if(_device_count_err != hipSuccess) return;

    hipError_t _current_device_err = hipGetDevice(&_current_device);

    scope::destructor _dtor{ [_current_device, _current_device_err]() {
        if(_current_device_err == hipSuccess)
        {
            OMNITRACE_HIP_RUNTIME_CALL(hipSetDevice(_current_device));
        }
    } };

    if(_current_device_err != hipSuccess || _device_count == 0) return;

#    define OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(NAME)                                    \
        ar(make_nvp(#NAME, _device_prop.NAME));

#    define OMNITRACE_SERIALIZE_HIP_DEVICE_PROP_ARRAY(NAME, ...)                         \
        ar(make_nvp(NAME, __VA_ARGS__));

    ar.setNextName("hip_device_properties");
    ar.startNode();
    ar.makeArray();

    scope::destructor _prop_dtor{ [&ar]() { ar.finishNode(); } };
    for(int dev = 0; dev < _device_count; ++dev)
    {
        auto _device_prop     = hipDeviceProp_t{};
        int  _driver_version  = 0;
        int  _runtime_version = 0;
        OMNITRACE_HIP_RUNTIME_CALL(hipSetDevice(dev));
        OMNITRACE_HIP_RUNTIME_CALL(hipGetDeviceProperties(&_device_prop, dev));
        OMNITRACE_HIP_RUNTIME_CALL(hipDriverGetVersion(&_driver_version));
        OMNITRACE_HIP_RUNTIME_CALL(hipRuntimeGetVersion(&_runtime_version));

        ar.startNode();
        ar(make_nvp("name", std::string{ _device_prop.name }));
        ar(make_nvp("driver_version", _driver_version));
        ar(make_nvp("runtime_version", _runtime_version));
        ar(make_nvp("capability.major_version", _device_prop.major));
        ar(make_nvp("capability.minor_version", _device_prop.minor));

        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(totalGlobalMem)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(totalConstMem)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(clockRate)

#    if OMNITRACE_HIP_VERSION >= 5000
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(memoryClockRate)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(memoryBusWidth)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(l2CacheSize)
#    endif

        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(sharedMemPerBlock)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(regsPerBlock)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(warpSize)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(multiProcessorCount)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerMultiProcessor)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerBlock)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP_ARRAY(
            "maxThreadsDim",
            intvec_t{ _device_prop.maxThreadsDim[0], _device_prop.maxThreadsDim[1],
                      _device_prop.maxThreadsDim[2] })
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP_ARRAY("maxGridSize",
                                                  intvec_t{ _device_prop.maxGridSize[0],
                                                            _device_prop.maxGridSize[1],
                                                            _device_prop.maxGridSize[2] })
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(memPitch)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(textureAlignment)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(kernelExecTimeoutEnabled)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(integrated)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(canMapHostMemory)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(ECCEnabled)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(cooperativeLaunch)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceLaunch)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(pciDomainID)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(pciBusID)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(pciDeviceID)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(computeMode)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(computeMode)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(gcnArch)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(gcnArchName)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(isMultiGpuBoard)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(clockInstructionRate)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccess)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccessUsesHostPageTables)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(directManagedMemAccessFromHost)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(concurrentManagedAccess)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(concurrentKernels)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(maxSharedMemoryPerMultiProcessor)
        OMNITRACE_SERIALIZE_HIP_DEVICE_PROP(asicRevision)

        const char* _compute_mode_descr[] = {
            "Default (multiple host threads can use ::hipSetDevice() with device "
            "simultaneously)",
            "Exclusive (only one host thread in one process is able to use "
            "::hipSetDevice() with this device)",
            "Prohibited (no host thread can use ::hipSetDevice() with this device)",
            "Exclusive Process (many threads in one process is able to use "
            "::hipSetDevice() with this device)",
            "Unknown",
            nullptr
        };
        ar(make_nvp("computeModeDescription",
                    std::string{ _compute_mode_descr[_device_prop.computeMode] }));
        ar.finishNode();
    }
#else
    (void) ar;
#endif
}

void
add_hip_device_metadata()
{
    if(device_count() == 0) return;

    OMNITRACE_METADATA([](auto& ar) {
        try
        {
            add_hip_device_metadata(ar);
        } catch(std::runtime_error& _e)
        {
            OMNITRACE_VERBOSE(2, "%s\n", _e.what());
        }
    });
}
}  // namespace gpu
}  // namespace omnitrace
