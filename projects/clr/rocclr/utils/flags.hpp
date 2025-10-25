/* Copyright (c) 2009 - 2025 Advanced Micro Devices, Inc.

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

#ifndef FLAGS_HPP_
#define FLAGS_HPP_

// clang-format off
#define RUNTIME_FLAGS(debug,release,release_on_stg)                           \
                                                                              \
release(int, AMD_LOG_LEVEL, 0,                                                \
        "The default log level")                                              \
release(uint, AMD_LOG_MASK, 0X7FFFFFFF,                                       \
        "The mask to enable specific kinds of logs")                          \
release(cstring, AMD_LOG_LEVEL_FILE, "",                                      \
        "Set output file for AMD_LOG_LEVEL, Default is stderr")               \
release(size_t, AMD_LOG_LEVEL_SIZE, 2048,                                     \
        "The max size of AMD_LOG generated in MB if printed to a file")       \
debug(uint, DEBUG_GPU_FLAGS, 0,                                               \
        "The debug options for GPU device")                                   \
release(size_t, CQ_THREAD_STACK_SIZE, 256*Ki, /* @todo: that much! */         \
        "The default command queue thread stack size")                        \
release(int, GPU_MAX_WORKGROUP_SIZE, 0,                                       \
        "Maximum number of workitems in a workgroup for GPU, 0 -use default") \
debug(bool, CPU_MEMORY_GUARD_PAGES, false,                                    \
        "Use guard pages for CPU memory")                                     \
debug(size_t, CPU_MEMORY_GUARD_PAGE_SIZE, 64,                                 \
        "Size in KB of CPU memory guard page")                                \
debug(size_t, CPU_MEMORY_ALIGNMENT_SIZE, 256,                                 \
        "Size in bytes for the default alignment for guarded memory on CPU")  \
debug(size_t, PARAMETERS_MIN_ALIGNMENT, NATIVE_ALIGNMENT_SIZE,                \
        "Minimum alignment required for the abstract parameters stack")       \
debug(size_t, MEMOBJ_BASE_ADDR_ALIGN, 4*Ki,                                   \
        "Alignment of the base address of any allocate memory object")        \
release(uint, ROC_HMM_FLAGS, 0,                                               \
        "ROCm HMM configuration flags")                                       \
release(cstring, GPU_DEVICE_ORDINAL, "",                                      \
        "Select the device ordinal (comma seperated list of available devices)") \
release(bool, REMOTE_ALLOC, false,                                            \
        "Use remote memory for the global heap allocation")                   \
release(uint, GPU_CP_DMA_COPY_SIZE, 1,                                        \
        "Set maximum size of CP DMA copy in KiB")                             \
release(uint, GPU_MAX_HEAP_SIZE, 100,                                         \
        "Set maximum size of the GPU heap to % of board memory")              \
release(uint, GPU_STAGING_BUFFER_SIZE, 4,                                     \
        "Size of the GPU staging buffer in MiB")                              \
release(bool, GPU_DUMP_BLIT_KERNELS, false,                                   \
        "Dump the kernels for blit manager")                                  \
release(uint, GPU_BLIT_ENGINE_TYPE, 0x0,                                      \
        "Blit engine type: 0 - Default, 1 - Host, 2 - CAL, 3 - Kernel")       \
release(bool, GPU_FLUSH_ON_EXECUTION, false,                                  \
        "Submit commands to HW on every operation. 0 - Disable, 1 - Enable")  \
release(bool, CL_KHR_FP64, true,                                              \
        "Enable/Disable support for double precision")                        \
release(cstring, AMD_OCL_BUILD_OPTIONS, 0,                                    \
        "Set clBuildProgram() and clCompileProgram()'s options (override)")   \
release(cstring, AMD_OCL_BUILD_OPTIONS_APPEND, 0,                             \
        "Append clBuildProgram() and clCompileProgram()'s options")           \
release(cstring, AMD_OCL_LINK_OPTIONS, 0,                                     \
        "Set clLinkProgram()'s options (override)")                           \
release(cstring, AMD_OCL_LINK_OPTIONS_APPEND, 0,                              \
        "Append clLinkProgram()'s options")                                   \
debug(cstring, AMD_OCL_SUBST_OBJFILE, 0,                                      \
        "Specify binary substitution config file for OpenCL")                 \
release(size_t, GPU_PINNED_XFER_SIZE, 32,                                     \
        "The pinned buffer size for pinning in read/write transfers in MiB")  \
release(size_t, GPU_PINNED_MIN_XFER_SIZE, 128,                                \
        "The minimal buffer size for pinned read/write transfers in MiB")     \
release(size_t, GPU_RESOURCE_CACHE_SIZE, 64,                                  \
        "The resource cache size in MB")                                      \
release(size_t, GPU_MAX_SUBALLOC_SIZE, 4096,                                  \
        "The maximum size accepted for suballocations in KB")                 \
release(size_t, GPU_NUM_MEM_DEPENDENCY, 256,                                  \
        "Number of memory objects for dependency tracking")                   \
release(size_t, GPU_XFER_BUFFER_SIZE, 0,                                      \
        "Transfer buffer size for image copy optimization in KB")             \
release(bool, GPU_IMAGE_DMA, true,                                            \
        "Enable DRM DMA for image transfers")                                 \
release(uint, GPU_SINGLE_ALLOC_PERCENT, 100,                                  \
        "Maximum size of a single allocation as percentage of total")         \
release(uint, GPU_NUM_COMPUTE_RINGS, 2,                                       \
        "GPU number of compute rings. 0 - disabled, 1 , 2,.. - the number of compute rings") \
release(bool, AMD_OCL_WAIT_COMMAND, false,                                    \
        "1 = Enable a wait for every submitted command")                      \
release(uint, GPU_PRINT_CHILD_KERNEL, 0,                                      \
        "Prints the specified number of the child kernels")                   \
release(bool, GPU_USE_DEVICE_QUEUE, false,                                    \
        "Use a dedicated device queue for the actual submissions")            \
release(bool, AMD_THREAD_TRACE_ENABLE, true,                                  \
        "Enable thread trace extension")                                      \
release(uint, OPENCL_VERSION, 200,                                            \
        "Force GPU opencl version")                                           \
release(bool, HSA_LOCAL_MEMORY_ENABLE, true,                                  \
        "Enable HSA device local memory usage")                               \
release(uint, HSA_KERNARG_POOL_SIZE, 1024 * 1024,                             \
        "Kernarg pool size")                                                  \
release(bool, GPU_MIPMAP, true,                                               \
        "Enables GPU mipmap extension")                                       \
release(uint, GPU_ENABLE_PAL, 2,                                              \
        "Enables PAL backend. 0 - ROC, 1 - PAL, 2 - ROC or PAL")              \
release(bool, DISABLE_DEFERRED_ALLOC, false,                                  \
        "Disables deferred memory allocation on device")                      \
release(int, AMD_GPU_FORCE_SINGLE_FP_DENORM, -1,                              \
        "Force denorm for single precision: -1 - don't force, 0 - disable, 1 - enable") \
release(uint, OCL_SET_SVM_SIZE, 4*16384,                                      \
        "set SVM space size for discrete GPU")                                \
release(uint, GPU_WAVES_PER_SIMD, 0,                                          \
        "Force the number of waves per SIMD (1-10)")                          \
release(bool, OCL_STUB_PROGRAMS, false,                                       \
        "1 = Enables OCL programs stubing")                                   \
release(bool, GPU_ANALYZE_HANG, false,                                        \
        "1 = Enables GPU hang analysis")                                      \
release(uint, GPU_MAX_REMOTE_MEM_SIZE, 2,                                     \
        "Maximum size (in Ki) that allows device memory substitution with system") \
release(bool, GPU_ADD_HBCC_SIZE, false,                                        \
        "Add HBCC size to the reported device memory")                        \
release(bool, PAL_DISABLE_SDMA, false,                                        \
        "1 = Disable SDMA for PAL")                                           \
release(uint, PAL_RGP_DISP_COUNT, 10000,                                      \
        "The number of dispatches for RGP capture with SQTT")                 \
release(uint, PAL_MALL_POLICY, 0,                                             \
        "Controls the behaviour of allocations with respect to the MALL"      \
        "0 = MALL policy is decided by KMD"                                   \
        "1 = Allocations are never put through the MALL"                      \
        "2 = Allocations will always be put through the MALL")                \
release(bool, GPU_ENABLE_WAVE32_MODE, true,                                   \
        "Enables Wave32 compilation in HW if available")                      \
release(bool, GPU_ENABLE_LC, true,                                            \
        "Enables LC path")                                                    \
release(bool, GPU_ENABLE_HW_P2P, false,                                       \
        "Enables HW P2P path")                                                \
release(bool, GPU_ENABLE_COOP_GROUPS, true,                                   \
         "Enables cooperative group launch")                                  \
release(uint, GPU_MAX_COMMAND_BUFFERS, 8,                                     \
         "The maximum number of command buffers allocated per queue")         \
release(uint, GPU_MAX_HW_QUEUES, 4,                                           \
         "The maximum number of HW queues allocated per device")              \
release(bool, GPU_IMAGE_BUFFER_WAR, true,                                     \
        "Enables image buffer workaround")                                    \
release(cstring, HIP_VISIBLE_DEVICES, "",                                     \
        "Only devices whose index is present in the sequence are visible to HIP")  \
release(cstring, CUDA_VISIBLE_DEVICES, "",                                    \
        "Only devices whose index is present in the sequence are visible to CUDA") \
release(bool, GPU_ENABLE_WGP_MODE, true,                                      \
        "Enables WGP Mode in HW if available")                                \
release(bool, GPU_DUMP_CODE_OBJECT, false,                                    \
        "Enable dump code object")                                            \
release(uint, GPU_MAX_USWC_ALLOC_SIZE, 2048,                                  \
        "Set a limit in Mb on the maximum USWC allocation size"               \
        "-1 = No limit")                                                      \
release(uint, AMD_SERIALIZE_KERNEL, 0,                                        \
        "Serialize kernel enqueue, 0x1 = Wait for completion before enqueue"  \
        "0x2 = Wait for completion after enqueue 0x3 = both")                 \
release(uint, AMD_SERIALIZE_COPY, 0,                                          \
        "Serialize copies, 0x1 = Wait for completion before enqueue"          \
        "0x2 = Wait for completion after enqueue 0x3 = both")                 \
release(uint, HIP_LAUNCH_BLOCKING, 0,                                         \
        "Serialize kernel enqueue 0x1 = Wait for completion after enqueue,"   \
        "same as AMD_SERIALIZE_KERNEL=2")                                     \
release(bool, PAL_ALWAYS_RESIDENT, false,                                     \
        "Force memory resources to become resident at allocation time")       \
release(uint, HIP_HOST_COHERENT, 0,                                           \
        "Coherent memory in hipHostMalloc, 0x1 = memory is coherent with host"\
        "0x0 = memory is not coherent between host and GPU")                  \
release(uint, AMD_OPT_FLUSH, 1,                                               \
        "Kernel flush option , 0x0 = Use system-scope fence operations."      \
        "0x1 = Use device-scope fence operations when possible.")             \
release(bool, AMD_DIRECT_DISPATCH, false,                                     \
        "Enable direct kernel dispatch.")                                     \
release(uint, HIP_HIDDEN_FREE_MEM, 0,                                         \
        "Reserve free mem reporting in Mb"                                    \
        "0 = Disable")                                                        \
release(size_t, GPU_FORCE_BLIT_COPY_SIZE, 16,                                 \
        "Use Blit until this size(in KB) for copies")                         \
release(uint, ROC_ACTIVE_WAIT_TIMEOUT, 0,                                     \
        "Forces active wait of GPU interrup for the timeout(us)")             \
release(bool, ROC_ENABLE_LARGE_BAR, true,                                     \
        "Enable Large Bar if supported by the device")                        \
release(bool, ROC_CPU_WAIT_FOR_SIGNAL, true,                                  \
        "Enable CPU wait for dependent HSA signals.")                         \
release(bool, ROC_SYSTEM_SCOPE_SIGNAL, true,                                  \
        "Enable system scope for signals (uses interrupts).")                 \
release(bool, GPU_FORCE_QUEUE_PROFILING, false,                               \
        "Force command queue profiling by default")                           \
release(bool, HIP_MEM_POOL_SUPPORT, true,                                     \
        "Enables memory pool support in HIP")                                 \
release(bool, HIP_MEM_POOL_USE_VM, true,                                      \
        "Enables memory pool support in HIP")                                 \
release(bool, DEBUG_HIP_MEM_POOL_VMHEAP, true,                                \
        "Enables virtual memory for memory pools")                            \
release(bool, PAL_HIP_IPC_FLAG, true,                                         \
        "Enable interprocess flag for device allocation in PAL HIP")          \
release(uint, PAL_FORCE_ASIC_REVISION, 0,                                     \
        "Force a specific asic revision for all devices")                     \
release(bool, PAL_EMBED_KERNEL_MD, false,                                     \
        "Enables writing kernel metadata into command buffers.")              \
release(cstring, ROC_GLOBAL_CU_MASK, "",                                      \
        "Sets a global CU mask (entered as hex value) for all queues,"        \
        "Each active bit represents using one CU (e.g., 0xf enables only 4 CUs)") \
release(size_t, PAL_PREPINNED_MEMORY_SIZE, 64,                                \
        "Size in KBytes of prepinned memory")                                 \
release(bool, AMD_CPU_AFFINITY, false,                                        \
        "Reset CPU affinity of any runtime threads")                          \
release(bool, ROC_USE_FGS_KERNARG, true,                                      \
        "Use fine grain kernel args segment for supported asics")             \
release(uint, ROC_P2P_SDMA_SIZE, 1024,                                        \
        "The minimum size in KB for P2P transfer with SDMA")                  \
release(uint, ROC_AQL_QUEUE_SIZE, 16384,                                      \
        "AQL queue size in AQL packets")                                      \
release(uint, ROC_SIGNAL_POOL_SIZE, 64,                                       \
        "Initial size of HSA signal pool")                                    \
release(uint, DEBUG_CLR_LIMIT_BLIT_WG, 16,                                    \
        "Limit the number of workgroups in blit operations")                  \
release(bool, DEBUG_CLR_BLIT_KERNARG_OPT, false,                              \
        "Enable blit kernel arguments optimization")                          \
release(bool, ROC_SKIP_KERNEL_ARG_COPY, false,                                \
        "If true, then runtime can skip kernel arg copy")                     \
release(bool, GPU_STREAMOPS_CP_WAIT, false,                                   \
        "Force the stream wait memory operation to wait on CP.")              \
release(bool, HIPRTC_USE_RUNTIME_UNBUNDLER, false,                            \
        "Set this to true to force runtime unbundler in hiprtc.")             \
release(size_t, HIP_INITIAL_DM_SIZE, 8 * Mi,                                  \
        "Set initial heap size for device malloc.")                           \
release(bool, HIP_FORCE_DEV_KERNARG, true,                                    \
         "Force device mem for kernel args.")                                 \
release(bool, DEBUG_CLR_GRAPH_PACKET_CAPTURE, true,                           \
         "Enable/Disable graph packet capturing")                             \
release(bool, GPU_DEBUG_ENABLE, false,                                        \
        "Enables collection of extra info for debugger at some perf cost")    \
release(cstring, HIPRTC_COMPILE_OPTIONS_APPEND, "",                           \
        "Set compile options needed for hiprtc compilation")                  \
release(cstring, HIPRTC_LINK_OPTIONS_APPEND, "",                              \
        "Set link options needed for hiprtc compilation")                     \
release(bool, HIP_VMEM_MANAGE_SUPPORT, true,                                  \
        "Virtual Memory Management Support")                                  \
release(bool, DEBUG_HIP_GRAPH_DOT_PRINT, false,                               \
         "Enable/Disable graph debug dot print dump")                         \
release(bool, DEBUG_HIP_FORCE_ASYNC_QUEUE, false,                             \
        "Forces grpahs into async queue mode. DEBUG_HIP_FORCE_GRAPH_QUEUES must be 1") \
release(uint, DEBUG_HIP_FORCE_GRAPH_QUEUES, 4,                                \
        "Forces the number of streams for the graph parallel execution")      \
release(uint, DEBUG_HIP_GRAPH_BATCH_SIZE, 256,                                \
        "Number of graph nodes to batch at a time")                           \
release(uint, DEBUG_HIP_BLOCK_SYNC, 50,                                       \
        "Blocks synchronization on CPU until the callback processing is done")\
release(uint, DEBUG_CLR_MAX_BATCH_SIZE, 1000,                                 \
        "Forces the callback to clean-up CPU submission queue")               \
release(bool, DEBUG_CLR_SYSMEM_POOL, false,                                   \
        "Use sysmem pool implementation in runtime for amd commands")         \
release(bool, DEBUG_HIP_KERNARG_COPY_OPT, true,                               \
        "Enable/Disable multiple kern arg copies")                            \
release(bool, DEBUG_CLR_KERNARG_HDP_FLUSH_WA, false,                          \
        "Toggle kernel arg copy workaround")                                  \
release(bool, DEBUG_HIP_DYNAMIC_QUEUES, false,                                \
        "Forces dynamic queue management")                                    \
release(uint, HIP_SKIP_ABORT_ON_GPU_ERROR, true,                              \
        "Set this to true, to avoid host side abort for GPU errors")          \
release(bool, HIP_FORCE_SPIRV_CODEOBJECT, false,                              \
        "Force use of SPIRV instead of device specific code object.")         \
release(uint, DEBUG_CLR_BATCH_CPU_SYNC_SIZE, 8,                               \
        "Forces the minimum batch size for CPU sync")  // clang-format on

namespace amd {

extern bool IS_HIP;

extern bool IS_LEGACY;

//! \addtogroup Utils
//  @{

struct Flag {
  enum Type {
    Tinvalid = 0,
    Tbool,    //!< A boolean type flag (true, false).
    Tint,     //!< An integer type flag (signed).
    Tuint,    //!< An integer type flag (unsigned).
    Tsize_t,  //!< A size_t type flag.
    Tcstring  //!< A string type flag.
  };

#define DEFINE_FLAG_NAME(type, name, value, help) k##name,
  enum Name { RUNTIME_FLAGS(DEFINE_FLAG_NAME, DEFINE_FLAG_NAME, DEFINE_FLAG_NAME) numFlags_ };
#undef DEFINE_FLAG_NAME

#define CAN_SET(type, name, v, h) static const bool cannotSet##name = false;
#define CANNOT_SET(type, name, v, h) static const bool cannotSet##name = true;

#ifdef DEBUG
  RUNTIME_FLAGS(CAN_SET, CAN_SET, CAN_SET)
#else   // !DEBUG
  RUNTIME_FLAGS(CANNOT_SET, CAN_SET, CANNOT_SET)
#endif  // !DEBUG

#undef CAN_SET
#undef CANNOT_SET

 private:
  static Flag flags_[];

 public:
  static char* envstr_;
  const char* name_;
  const void* value_;
  Type type_;
  bool isDefault_;

 public:
  static bool init();

  static void tearDown();

  bool setValue(const char* value);

  static bool isDefault(Name name) { return flags_[name].isDefault_; }
};

#define flagIsDefault(name) (amd::Flag::cannotSet##name || amd::Flag::isDefault(amd::Flag::k##name))

#define setIfNotDefault(var, opt, other)                                                           \
  if (!flagIsDefault(opt))                                                                         \
    var = (opt);                                                                                   \
  else                                                                                             \
    var = (other);

//  @}

}  // namespace amd

#ifdef _WIN32
#define EXPORT_FLAG extern "C" __declspec(dllexport)
#else  // !_WIN32
#ifdef BUILD_STATIC_LIBS
#define EXPORT_FLAG extern
#else
#define EXPORT_FLAG extern "C"
#endif
namespace amd::flags {
#endif  // !_WIN32

#define DECLARE_RELEASE_FLAG(type, name, value, help) EXPORT_FLAG type name;
#ifdef DEBUG
#define DECLARE_DEBUG_FLAG(type, name, value, help) EXPORT_FLAG type name;
#else  // !DEBUG
#define DECLARE_DEBUG_FLAG(type, name, value, help) const type name = value;
#endif  // !DEBUG

RUNTIME_FLAGS(DECLARE_DEBUG_FLAG, DECLARE_RELEASE_FLAG, DECLARE_DEBUG_FLAG);

#undef DECLARE_DEBUG_FLAG
#undef DECLARE_RELEASE_FLAG
#ifndef _WIN32
}
using namespace amd::flags;
#endif  // !_WIN32
#endif  /*FLAGS_HPP_*/
