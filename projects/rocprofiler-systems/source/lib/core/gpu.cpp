// MIT License
//
// Copyright (c) 2022-2024 Advanced Micro Devices, Inc. All Rights Reserved.
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

#include "common/defines.h"

#if !defined(ROCPROFSYS_USE_ROCM_SMI)
#    define ROCPROFSYS_USE_ROCM_SMI 0
#endif

#if !defined(ROCPROFSYS_USE_HIP)
#    define ROCPROFSYS_USE_HIP 0
#endif

#include "core/hip_runtime.hpp"

#if ROCPROFSYS_USE_HIP > 0
#    if !defined(TIMEMORY_USE_HIP)
#        define TIMEMORY_USE_HIP 1
#    endif
#endif

#include "debug.hpp"
#include "defines.hpp"
#include "gpu.hpp"

#include <timemory/manager.hpp>

#if ROCPROFSYS_USE_ROCM_SMI > 0
#    include <rocm_smi/rocm_smi.h>
#endif

#if ROCPROFSYS_USE_HIP > 0
#    include <timemory/components/hip/backends.hpp>

#    if !defined(ROCPROFSYS_HIP_RUNTIME_CALL)
#        define ROCPROFSYS_HIP_RUNTIME_CALL(err)                                         \
            {                                                                            \
                if(err != ::tim::hip::success_v && (int) err != 0)                       \
                {                                                                        \
                    ROCPROFSYS_THROW(                                                    \
                        "[%s:%d] Warning! HIP API call failed with code %i :: %s\n",     \
                        __FILE__, __LINE__, (int) err, hipGetErrorString(err));          \
                }                                                                        \
            }
#    endif
#endif

namespace rocprofsys
{
namespace gpu
{
namespace
{
namespace scope = ::tim::scope;

#if ROCPROFSYS_USE_ROCM_SMI > 0
#    define ROCPROFSYS_ROCM_SMI_CALL(ERROR_CODE)                                         \
        ::rocprofsys::gpu::check_rsmi_error(ERROR_CODE, __FILE__, __LINE__)

void
check_rsmi_error(rsmi_status_t _code, const char* _file, int _line)
{
    if(_code == RSMI_STATUS_SUCCESS) return;
    const char* _msg = nullptr;
    auto        _err = rsmi_status_string(_code, &_msg);
    if(_err != RSMI_STATUS_SUCCESS)
        ROCPROFSYS_THROW("rsmi_status_string failed. No error message available. "
                         "Error code %i originated at %s:%i\n",
                         static_cast<int>(_code), _file, _line);
    ROCPROFSYS_THROW("[%s:%i] Error code %i :: %s", _file, _line, static_cast<int>(_code),
                     _msg);
}

bool
rsmi_init()
{
    auto _rsmi_init = []() {
        try
        {
            ROCPROFSYS_ROCM_SMI_CALL(::rsmi_init(0));
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(1, "Exception thrown initializing rocm-smi: %s\n",
                                     _e.what());
            return false;
        }
        return true;
    }();

    return _rsmi_init;
}
#endif

#if ROCPROFSYS_HIP_VERSION >= 60000
template <typename ArchiveT, typename ArgT,
          std::enable_if_t<!std::is_pointer<ArgT>::value, int> = 0>
void
device_prop_serialize(ArchiveT& archive, const char* name, const ArgT& arg)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;
    archive(make_nvp(name, arg));
}

template <typename ArchiveT, typename ArgT, size_t N>
void
device_prop_serialize(ArchiveT& archive, const char* name, ArgT arg[N])
{
    if constexpr(!std::is_same<ArgT, char>::value &&
                 !std::is_same<ArgT, const char>::value)
    {
        namespace cereal = tim::cereal;
        using cereal::make_nvp;
        auto data = std::array<int, N>{};
        for(size_t i = 0; i < N; ++i)
            data[i] = arg[i];
        archive(make_nvp(name, data));
    }
    else
    {
        device_prop_serialize(archive, name, std::string{ arg });
    }
}

template <typename ArchiveT>
void
device_prop_serialize(ArchiveT& archive, const char* name, hipUUID_t arg)
{
    constexpr auto N = sizeof(arg.bytes);
    namespace cereal = tim::cereal;
    using cereal::make_nvp;
    auto data = std::array<char, N + 1>{};
    data.fill('\0');
    for(size_t i = 0; i < N; ++i)
        data[i] = arg.bytes[i];
    auto str_v = std::string_view{ data.data() };
    auto str   = std::string{ str_v }.substr(0, str_v.find('\0'));
    archive(make_nvp(name, str));
}

template <typename ArchiveT>
void
device_prop_serialize(ArchiveT& archive, const char* name, hipDeviceArch_t arg)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;

#    define ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(NAME)                                   \
        {                                                                                \
            auto val = arg.NAME;                                                         \
            archive(make_nvp(#NAME, val));                                               \
        }

    archive.setNextName(name);
    archive.startNode();
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasGlobalInt32Atomics)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasGlobalFloatAtomicExch)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasSharedInt32Atomics)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasSharedFloatAtomicExch)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasFloatAtomicAdd)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasGlobalInt64Atomics)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasSharedInt64Atomics)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasDoubles)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasWarpVote)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasWarpBallot)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasWarpShuffle)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasFunnelShift)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasThreadFenceSystem)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasSyncThreadsExt)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasSurfaceFuncs)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(has3dGrid)
    ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH(hasDynamicParallelism)
    archive.finishNode();

#    undef ROCPROFSYS_SERIALIZE_HIP_DEVICE_ARCH
}
#endif
}  // namespace

int
hip_device_count()
{
#if ROCPROFSYS_USE_HIP > 0
    return ::tim::hip::device_count();
#else
    return 0;
#endif
}

int
rsmi_device_count()
{
#if ROCPROFSYS_USE_ROCM_SMI > 0
    if(!rsmi_init()) return 0;

    static auto _num_devices = []() {
        uint32_t _v = 0;
        try
        {
            ROCPROFSYS_ROCM_SMI_CALL(rsmi_num_monitor_devices(&_v));
        } catch(std::exception& _e)
        {
            ROCPROFSYS_BASIC_VERBOSE(
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
#if ROCPROFSYS_USE_ROCM_SMI > 0
    // store as static since calls after rsmi_shutdown will return zero
    return rsmi_device_count();
#elif ROCPROFSYS_USE_HIP > 0
    return ::tim::hip::device_count();
#else
    return 0;
#endif
}

template <typename ArchiveT>
void
add_hip_device_metadata(ArchiveT& ar)
{
    namespace cereal = tim::cereal;
    using cereal::make_nvp;

#if ROCPROFSYS_USE_HIP > 0
    int        _device_count     = 0;
    int        _current_device   = 0;
    hipError_t _device_count_err = hipGetDeviceCount(&_device_count);

    if(_device_count_err != hipSuccess) return;

    hipError_t _current_device_err = hipGetDevice(&_current_device);

    scope::destructor _dtor{ [_current_device, _current_device_err]() {
        if(_current_device_err == hipSuccess)
        {
            ROCPROFSYS_HIP_RUNTIME_CALL(hipSetDevice(_current_device));
        }
    } };

    if(_current_device_err != hipSuccess || _device_count == 0) return;

    ar.setNextName("hip_device_properties");
    ar.startNode();
    ar.makeArray();

    scope::destructor _prop_dtor{ [&ar]() { ar.finishNode(); } };
    for(int dev = 0; dev < _device_count; ++dev)
    {
        auto _device_prop     = hipDeviceProp_t{};
        int  _driver_version  = 0;
        int  _runtime_version = 0;
        ROCPROFSYS_HIP_RUNTIME_CALL(hipSetDevice(dev));
        ROCPROFSYS_HIP_RUNTIME_CALL(hipGetDeviceProperties(&_device_prop, dev));
        ROCPROFSYS_HIP_RUNTIME_CALL(hipDriverGetVersion(&_driver_version));
        ROCPROFSYS_HIP_RUNTIME_CALL(hipRuntimeGetVersion(&_runtime_version));

        ar.startNode();

#    if ROCPROFSYS_HIP_VERSION < 60000
        using intvec_t = std::vector<int>;

#        define ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(NAME)                               \
            ar(make_nvp(#NAME, _device_prop.NAME));

#        define ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP_ARRAY(NAME, ...)                    \
            ar(make_nvp(NAME, __VA_ARGS__));

        ar(make_nvp("name", std::string{ _device_prop.name }));
        ar(make_nvp("driver_version", _driver_version));
        ar(make_nvp("runtime_version", _runtime_version));
        ar(make_nvp("capability.major_version", _device_prop.major));
        ar(make_nvp("capability.minor_version", _device_prop.minor));

        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(totalGlobalMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(totalConstMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(clockRate)

#        if ROCPROFSYS_HIP_VERSION >= 50000
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryClockRate)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryBusWidth)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(l2CacheSize)
#        endif

        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(sharedMemPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(regsPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(warpSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(multiProcessorCount)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerMultiProcessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP_ARRAY(
            "maxThreadsDim",
            intvec_t{ _device_prop.maxThreadsDim[0], _device_prop.maxThreadsDim[1],
                      _device_prop.maxThreadsDim[2] })
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP_ARRAY(
            "maxGridSize",
            intvec_t{ _device_prop.maxGridSize[0], _device_prop.maxGridSize[1],
                      _device_prop.maxGridSize[2] })
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memPitch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(textureAlignment)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(kernelExecTimeoutEnabled)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(integrated)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(canMapHostMemory)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(ECCEnabled)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeLaunch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceLaunch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciDomainID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciBusID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciDeviceID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(computeMode)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gcnArch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gcnArchName)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(isMultiGpuBoard)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(clockInstructionRate)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccess)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccessUsesHostPageTables)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(directManagedMemAccessFromHost)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(concurrentManagedAccess)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(concurrentKernels)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSharedMemoryPerMultiProcessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(asicRevision)
#    else
#        define ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(NAME)                               \
            device_prop_serialize(ar, #NAME, _device_prop.NAME);

        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(name)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(uuid)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(luid)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(luidDeviceNodeMask)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(totalGlobalMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(sharedMemPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(regsPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(warpSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memPitch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxThreadsDim)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxGridSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(clockRate)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(totalConstMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(major)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(minor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(textureAlignment)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(texturePitchAlignment)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(deviceOverlap)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(multiProcessorCount)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(kernelExecTimeoutEnabled)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(integrated)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(canMapHostMemory)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(computeMode)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture1D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture1DMipmap)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture1DLinear)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture2D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture2DMipmap)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture2DLinear)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture2DGather)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture3D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture3DAlt)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTextureCubemap)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture1DLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTexture2DLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxTextureCubemapLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurface1D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurface2D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurface3D)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurface1DLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurface2DLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurfaceCubemap)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSurfaceCubemapLayered)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(surfaceAlignment)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(concurrentKernels)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(ECCEnabled)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciBusID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciDeviceID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pciDomainID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(tccDriver)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(asyncEngineCount)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(unifiedAddressing)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryClockRate)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryBusWidth)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(l2CacheSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(persistingL2CacheMaxSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxThreadsPerMultiProcessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(streamPrioritiesSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(globalL1CacheSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(localL1CacheSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(sharedMemPerMultiprocessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(regsPerMultiprocessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(managedMemory)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(isMultiGpuBoard)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(multiGpuBoardGroupID)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(hostNativeAtomicSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(singleToDoublePrecisionPerfRatio)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccess)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(concurrentManagedAccess)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(computePreemptionSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(canUseHostPointerForRegisteredMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeLaunch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceLaunch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(sharedMemPerBlockOptin)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(pageableMemoryAccessUsesHostPageTables)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(directManagedMemAccessFromHost)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxBlocksPerMultiProcessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(accessPolicyMaxWindowSize)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(reservedSharedMemPerBlock)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(hostRegisterSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(sparseHipArraySupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(hostRegisterReadOnlySupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(timelineSemaphoreInteropSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryPoolsSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gpuDirectRDMASupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gpuDirectRDMAFlushWritesOptions)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gpuDirectRDMAWritesOrdering)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(memoryPoolSupportedHandleTypes)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(deferredMappingHipArraySupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(ipcEventSupported)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(clusterLaunch)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(unifiedFunctionPointers)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(gcnArchName)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(maxSharedMemoryPerMultiProcessor)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(clockInstructionRate)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(arch)
        // ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(hdpMemFlushCntl)
        // ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(hdpRegFlushCntl)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceUnmatchedFunc)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceUnmatchedGridDim)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceUnmatchedBlockDim)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(cooperativeMultiDeviceUnmatchedSharedMem)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(isLargeBar)
        ROCPROFSYS_SERIALIZE_HIP_DEVICE_PROP(asicRevision)
#    endif

        const auto _compute_mode_descr = std::array<const char*, 6>{
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

        auto _compute_mode = std::min<int>(_device_prop.computeMode, 5);
        ar(make_nvp("computeModeDescription",
                    std::string{ _compute_mode_descr.at(_compute_mode) }));

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

    ROCPROFSYS_METADATA([](auto& ar) {
        try
        {
            add_hip_device_metadata(ar);
        } catch(std::runtime_error& _e)
        {
            ROCPROFSYS_VERBOSE(2, "%s\n", _e.what());
        }
    });
}
}  // namespace gpu
}  // namespace rocprofsys
