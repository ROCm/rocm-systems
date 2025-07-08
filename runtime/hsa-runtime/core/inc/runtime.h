////////////////////////////////////////////////////////////////////////////////
//
// The University of Illinois/NCSA
// Open Source License (NCSA)
//
// Copyright (c) 2014-2025, Advanced Micro Devices, Inc. All rights reserved.
//
// Developed by:
//
//                 AMD Research and AMD HSA Software Development
//
//                 Advanced Micro Devices, Inc.
//
//                 www.amd.com
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to
// deal with the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
//  - Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimers.
//  - Redistributions in binary form must reproduce the above copyright
//    notice, this list of conditions and the following disclaimers in
//    the documentation and/or other materials provided with the distribution.
//  - Neither the names of Advanced Micro Devices, Inc,
//    nor the names of its contributors may be used to endorse or promote
//    products derived from this Software without specific prior written
//    permission.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
// THE CONTRIBUTORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS WITH THE SOFTWARE.
//
////////////////////////////////////////////////////////////////////////////////

// HSA runtime C++ interface file.

#ifndef HSA_RUNTME_CORE_INC_RUNTIME_H_
#define HSA_RUNTME_CORE_INC_RUNTIME_H_

#include <vector>
#include <map>
#include <memory>
#include <tuple>
#include <utility>
#include <thread>
#include <sys/un.h>

#if defined(__linux__)
#include <xf86drm.h>
#include <amdgpu.h>
#endif

#include "core/inc/hsa_ext_interface.h"
#include "core/inc/hsa_internal.h"
#include "core/inc/hsa_ext_amd_impl.h"

#include "core/inc/agent.h"
#include "core/inc/driver.h"
#include "core/inc/exceptions.h"
#include "core/inc/interrupt_signal.h"
#include "core/inc/memory_region.h"
#include "core/inc/signal.h"
#include "core/inc/svm_profiler.h"
#include "core/inc/thunk_loader.h"
#include "core/util/flag.h"
#include "core/util/locks.h"
#include "core/util/os.h"
#include "core/util/utils.h"

#include "core/inc/amd_loader_context.hpp"
#include "core/inc/amd_hsa_code.hpp"

#if defined(__clang__)
#if __has_feature(address_sanitizer)
#define SANITIZER_AMDGPU 1
#endif
#endif

//---------------------------------------------------------------------------//
//    Constants                                                              //
//---------------------------------------------------------------------------//

#define HSA_ARGUMENT_ALIGN_BYTES 16
#define HSA_QUEUE_ALIGN_BYTES 64
#define HSA_PACKET_ALIGN_BYTES 64
#define HSA_MAX_DEP_SIGNALS 5

//Avoids include
namespace rocr {
namespace AMD {
  class MemoryRegion;
} // namespace amd

namespace core {
extern bool g_use_interrupt_wait;
extern bool g_use_mwaitx;

/// @brief  Runtime class provides the following functions:
/// - open and close connection to kernel driver.
/// - load supported extension library (image and finalizer).
/// - load tools library.
/// - expose supported agents.
/// - allocate and free memory.
/// - memory copy and fill.
/// - grant access to memory (dgpu memory pool extension).
/// - maintain loader state.
/// - monitor asynchronous event from agent.
class Runtime {
 friend class AMD::MemoryRegion;
 public:
  /// @brief Structure to describe connectivity between agents.
  struct LinkInfo {
    LinkInfo() : num_hop(0), rec_sdma_eng_id_mask(0), info{0} {}

    uint32_t num_hop;
    uint32_t rec_sdma_eng_id_mask;
    hsa_amd_memory_pool_link_info_t info;
  };

  struct KfdVersion_t {
    HsaVersionInfo version;
    bool supports_exception_debugging;
    bool supports_event_age;
    bool supports_core_dump;
  };

  /// @brief Open connection to kernel driver and increment reference count.
  static hsa_status_t Acquire();

  /// @brief Decrement reference count and close connection to kernel driver.
  static hsa_status_t Release();

  /// @brief Checks if connection to kernel driver is opened.
  /// @retval True if the connection to kernel driver is opened.
  static bool IsOpen();

  // @brief Callback handler for HW Exceptions.
  static bool HwExceptionHandler(hsa_signal_value_t val, void* arg);

  // @brief Callback handler for VM fault access.
  static bool VMFaultHandler(hsa_signal_value_t val, void* arg);

  // @brief Print known allocations near ptr.
  static void PrintMemoryMapNear(void* ptr);

  /// @brief Singleton object of the runtime.
  static Runtime* runtime_singleton_;

  /// @brief Insert agent into agent list ::agents_.
  /// @param [in] agent Pointer to the agent object.
  void RegisterAgent(Agent* agent, bool Enabled);

  /// @brief Insert agent into the driver list.
  /// @param [in] driver Unique pointer to the driver object.
  void RegisterDriver(std::unique_ptr<Driver> driver);

  /// @brief Delete all agent objects from ::agents_.
  void DestroyAgents();

  /// @brief Close and delete all agent driver objects from ::agent_drivers_.
  void DestroyDrivers();

  /// @brief Set the number of links connecting the agents in the platform.
  void SetLinkCount(size_t num_link);

  /// @brief Register link information connecting @p node_id_from and @p
  /// node_id_to.
  /// @param [in] node_id_from Node id of the source node.
  /// @param [in] node_id_to Node id of the destination node.
  /// @param [in] link_info The link information between source and destination
  /// nodes.
  void RegisterLinkInfo(uint32_t node_id_from, uint32_t node_id_to,
                        uint32_t num_hop, uint32_t rec_sdma_eng_id_mask,
                        hsa_amd_memory_pool_link_info_t& link_info);

  /// @brief Query link information between two nodes.
  /// @param [in] node_id_from Node id of the source node.
  /// @param [in] node_id_to Node id of the destination node.
  /// @retval The link information between source and destination nodes.
  const LinkInfo GetLinkInfo(uint32_t node_id_from, uint32_t node_id_to);

  /// @brief Invoke the user provided call back for each agent in the agent
  /// list.
  ///
  /// @param [in] callback User provided callback function.
  /// @param [in] data User provided pointer as input for @p callback.
  ///
  /// @retval ::HSA_STATUS_SUCCESS if the callback function for each traversed
  /// agent returns ::HSA_STATUS_SUCCESS.
  hsa_status_t IterateAgent(hsa_status_t (*callback)(hsa_agent_t agent,
                                                     void* data),
                            void* data);

  /// @brief Allocate memory on a particular region.
  ///
  /// @param [in] region Pointer to region object.
  /// @param [in] size Allocation size in bytes.
  /// @param [in] alloc_flags Modifiers to pass to MemoryRegion allocator.
  /// @param [out] address Pointer to store the allocation result.
  ///
  /// @retval ::HSA_STATUS_SUCCESS If allocation is successful.
  hsa_status_t AllocateMemory(const MemoryRegion* region, size_t size,
                              MemoryRegion::AllocateFlags alloc_flags,
                              void** address, int agent_node_id = 0);

  /// @brief Free memory previously allocated with AllocateMemory.
  ///
  /// @param [in] ptr Address of the memory to be freed.
  ///
  /// @retval ::HSA_STATUS_ERROR If @p ptr is not the address of previous
  /// allocation via ::core::Runtime::AllocateMemory
  /// @retval ::HSA_STATUS_SUCCESS if @p ptr is successfully released.
  hsa_status_t FreeMemory(void* ptr);

  hsa_status_t RegisterReleaseNotifier(void* ptr, hsa_amd_deallocation_callback_t callback,
                                       void* user_data);

  hsa_status_t DeregisterReleaseNotifier(void* ptr, hsa_amd_deallocation_callback_t callback);

  /// @brief Blocking memory copy from src to dst.
  ///
  /// @param [in] dst Memory address of the destination.
  /// @param [in] src Memory address of the source.
  /// @param [in] size Copy size in bytes.
  ///
  /// @retval ::HSA_STATUS_SUCCESS if memory copy is successful and completed.
  hsa_status_t CopyMemory(void* dst, const void* src, size_t size);

  /// @brief Non-blocking memory copy from src to dst.
  ///
  /// @details The memory copy will be performed after all signals in
  /// @p dep_signals have value of 0. On completion @p completion_signal
  /// will be decremented.
  ///
  /// @param [in] dst Memory address of the destination.
  /// @param [in] dst_agent Agent object associated with the destination. This
  /// agent should be able to access the destination and source.
  /// @param [in] src Memory address of the source.
  /// @param [in] src_agent Agent object associated with the source. This
  /// agent should be able to access the destination and source.
  /// @param [in] size Copy size in bytes.
  /// @param [in] dep_signals Array of signal dependency.
  /// @param [in] completion_signal Completion signal object.
  ///
  /// @retval ::HSA_STATUS_SUCCESS if copy command has been submitted
  /// successfully to the agent DMA queue.
  hsa_status_t CopyMemory(void* dst, core::Agent* dst_agent, const void* src,
                          core::Agent* src_agent, size_t size,
                          std::vector<core::Signal*>& dep_signals, core::Signal& completion_signal);

  /// @brief Non-blocking memory copy from src to dst on engine_id.
  ///
  /// @details All semantics and params are dentical to CopyMemory
  ///  with the exception of engine_id.
  ///
  /// @param [in] engine_id Target engine to copy on.
  ///
  /// @param [in] force_copy_on_sdma By default, a blit kernel copy is used
  /// when dst_agent == src_agent.  Setting this to true will force the copy
  /// over SDMA1.
  ///
  /// @retval ::HSA_STATUS_SUCCESS if copy command has been submitted
  /// successfully to the agent DMA queue.
  hsa_status_t CopyMemoryOnEngine(void* dst, core::Agent* dst_agent, const void* src,
                          core::Agent* src_agent, size_t size,
                          std::vector<core::Signal*>& dep_signals, core::Signal& completion_signal,
                          hsa_amd_sdma_engine_id_t  engine_id, bool force_copy_on_sdma);

  /// @brief Return SDMA availability status for copy direction
  ///
  /// @param [in] dst_agent Destination agent.
  /// @param [in] src_agent Source agent.
  /// @param [out] engine_ids_mask Mask of engine_ids.
  ///
  /// @retval HSA_STATUS_SUCCESS DMA engines are available
  /// @retval HSA_STATUS_ERROR_OUT_OF_RESOURCES DMA engines are not available
  hsa_status_t CopyMemoryStatus(core::Agent* dst_agent, core::Agent* src_agent,
                                uint32_t *engine_ids_mask);

  /// @brief Get preferred SDMA engine for the copy direction
  ///
  /// @param [in] dst_agent Destination agent.
  /// @param [in] src_agent Source agent.
  /// @param [out] recommended_ids_mask Mask of recommended_ids.
  ///
  /// @retval HSA_STATUS_SUCCESS For mask returned
  hsa_status_t GetPreferredEngine(core::Agent* dst_agent, core::Agent* src_agent,
                                  uint32_t* recommended_ids_mask);

  /// @brief Fill the first @p count of uint32_t in ptr with value.
  ///
  /// @param [in] ptr Memory address to be filled.
  /// @param [in] value The value/pattern that will be used to set @p ptr.
  /// @param [in] count Number of uint32_t element to be set.
  ///
  /// @retval ::HSA_STATUS_SUCCESS if memory fill is successful and completed.
  hsa_status_t FillMemory(void* ptr, uint32_t value, size_t count);

  /// @brief Set agents as the whitelist to access ptr.
  ///
  /// @param [in] num_agents The number of agent handles in @p agents array.
  /// @param [in] agents Agent handle array.
  /// @param [in] ptr Pointer of memory previously allocated via
  /// core::Runtime::AllocateMemory.
  ///
  /// @retval ::HSA_STATUS_SUCCESS The whitelist has been configured
  /// successfully and all agents in the @p agents could start accessing @p ptr.
  hsa_status_t AllowAccess(uint32_t num_agents, const hsa_agent_t* agents,
                           const void* ptr);

  /// @brief Query system information.
  ///
  /// @param [in] attribute System info attribute to query.
  /// @param [out] value Pointer to store the attribute value.
  ///
  /// @retval HSA_STATUS_SUCCESS The attribute is valid and the @p value is
  /// set.
  hsa_status_t GetSystemInfo(hsa_system_info_t attribute, void* value);

  /// @brief Register a callback function @p handler that is associated with
  /// @p signal to asynchronous event monitor thread.
  ///
  /// @param [in] signal Signal handle associated with @p handler.
  /// @param [in] cond The condition to execute the @p handler.
  /// @param [in] value The value to compare with @p signal value. If the
  /// comparison satisfy @p cond, the @p handler will be called.
  /// @param [in] arg Pointer to the argument that will be provided to @p
  /// handler.
  ///
  /// @retval ::HSA_STATUS_SUCCESS Registration is successful.
  hsa_status_t SetAsyncSignalHandler(hsa_signal_t signal,
                                     hsa_signal_condition_t cond,
                                     hsa_signal_value_t value,
                                     hsa_amd_signal_handler handler, void* arg);

  hsa_status_t InteropMap(uint32_t num_agents, Agent** agents,
                          int interop_handle, uint32_t flags, size_t* size,
                          void** ptr, size_t* metadata_size,
                          const void** metadata);

  hsa_status_t InteropUnmap(void* ptr);

  struct PtrInfoBlockData {
    void* base;
    size_t length;
    core::Agent* agentOwner;
  };

  hsa_status_t PtrInfo(const void* ptr, hsa_amd_pointer_info_t* info, void* (*alloc)(size_t),
                       uint32_t* num_agents_accessible, hsa_agent_t** accessible,
                       PtrInfoBlockData* block_info = nullptr);

  hsa_status_t SetPtrInfoData(const void* ptr, void* userptr);

  hsa_status_t IPCCreate(void* ptr, size_t len, hsa_amd_ipc_memory_t* handle);

  hsa_status_t IPCAttach(const hsa_amd_ipc_memory_t* handle, size_t len, uint32_t num_agents,
                         Agent** mapping_agents, void** mapped_ptr);

  hsa_status_t IPCDetach(void* ptr);

  hsa_status_t SetSvmAttrib(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* attribute_list,
                            size_t attribute_count);

  hsa_status_t GetSvmAttrib(void* ptr, size_t size, hsa_amd_svm_attribute_pair_t* attribute_list,
                            size_t attribute_count);

  hsa_status_t SvmPrefetch(void* ptr, size_t size, hsa_agent_t agent, uint32_t num_dep_signals,
                           const hsa_signal_t* dep_signals, hsa_signal_t completion_signal);

  hsa_status_t DmaBufExport(const void* ptr, size_t size, int* dmabuf,
                                            uint64_t* offset, uint64_t flags);

  hsa_status_t DmaBufClose(int dmabuf);

  hsa_status_t VMemoryAddressReserve(void** ptr, size_t size, uint64_t address, uint64_t alignment, uint64_t flags);

  hsa_status_t VMemoryAddressFree(void* ptr, size_t size);

  hsa_status_t VMemoryHandleCreate(const MemoryRegion* region, size_t size,
                                   MemoryRegion::AllocateFlags alloc_flags,
                                   uint64_t flags, hsa_amd_vmem_alloc_handle_t* memoryHandle);

  hsa_status_t VMemoryHandleRelease(hsa_amd_vmem_alloc_handle_t memoryHandle);

  hsa_status_t VMemoryHandleMap(void* va, size_t size, size_t in_offset,
                                hsa_amd_vmem_alloc_handle_t memoryHandle, uint64_t flags);

  hsa_status_t VMemoryHandleUnmap(void* va, size_t size);

  hsa_status_t VMemorySetAccess(void* va, size_t size, const hsa_amd_memory_access_desc_t* desc,
                                size_t desc_cnt);

  hsa_status_t VMemoryGetAccess(const void* va, hsa_access_permission_t* perms,
                                hsa_agent_t agent_handle);

  hsa_status_t VMemoryExportShareableHandle(int* dmabuf_fd,
                                            const hsa_amd_vmem_alloc_handle_t handle,
                                            const uint64_t flags);

  hsa_status_t VMemoryImportShareableHandle(const int dmabuf_fd,
                                            hsa_amd_vmem_alloc_handle_t* handle);

  hsa_status_t VMemoryRetainAllocHandle(hsa_amd_vmem_alloc_handle_t* memoryHandle, void* addr);

  hsa_status_t VMemoryGetAllocPropertiesFromHandle(const hsa_amd_vmem_alloc_handle_t memoryHandle,
                                                   const core::MemoryRegion** mem_region,
                                                   hsa_amd_memory_type_t* type);

  hsa_status_t EnableLogging(uint8_t* flags, void* file);

  const std::vector<Agent*>& cpu_agents() { return cpu_agents_; }

  const std::vector<Agent*>& gpu_agents() { return gpu_agents_; }

  const std::vector<Agent *> &aie_agents() { return aie_agents_; }

  const std::vector<Agent*>& disabled_gpu_agents() { return disabled_gpu_agents_; }

  const std::vector<uint32_t>& gpu_ids() { return gpu_ids_; }

  Agent* agent_by_gpuid(uint32_t gpuid) { return agents_by_gpuid_[gpuid]; }

  Agent* region_gpu() { return region_gpu_; }

  const std::vector<const MemoryRegion*>& system_regions_fine() const {
    return system_regions_fine_;
  }

  const std::vector<const MemoryRegion*>& system_regions_coarse() const {
    return system_regions_coarse_;
  }

  amd::hsa::loader::Loader* loader() { return loader_; }

  amd::LoaderContext* loader_context() { return &loader_context_; }

  amd::hsa::code::AmdHsaCodeManager* code_manager() { return &code_manager_; }

  std::function<void*(size_t size, size_t align, MemoryRegion::AllocateFlags flags, int agent_node_id)>&
  system_allocator() {
    return system_allocator_;
  }

  std::function<void(void*)>& system_deallocator() {
    return system_deallocator_;
  }

  const Flag& flag() const { return flag_; }

  const ThunkLoader* thunkLoader() const { return thunkLoader_; }

  ExtensionEntryPoints extensions_;

  hsa_status_t SetCustomSystemEventHandler(hsa_amd_system_event_callback_t callback,
                                           void* data);

  hsa_status_t SetInternalQueueCreateNotifier(hsa_amd_runtime_queue_notifier callback,
                                              void* user_data);

  void InternalQueueCreateNotify(const hsa_queue_t* queue, hsa_agent_t agent);

  SharedSignalPool_t* GetSharedSignalPool() { return &SharedSignalPool; }

  InterruptSignal::EventPool* GetEventPool() { return &EventPool; }

  uint64_t sys_clock_freq() const { return sys_clock_freq_; }

  void KfdVersion(const HsaVersionInfo& version) {
    kfd_version.version = version;
    if (version.KernelInterfaceMajorVersion == 1 &&
      version.KernelInterfaceMinorVersion >= 14)
      kfd_version.supports_event_age = true;
  }

  void KfdVersion(bool exception_debugging, bool core_dump) {
    kfd_version.supports_exception_debugging = exception_debugging;
    kfd_version.supports_core_dump = core_dump;
  }

  KfdVersion_t KfdVersion() const { return kfd_version; }

  bool VirtualMemApiSupported() const { return virtual_mem_api_supported_; }
  bool XnackEnabled() const { return xnack_enabled_; }
  void XnackEnabled(bool enable) { xnack_enabled_ = enable; }

  Driver &AgentDriver(DriverType drv_type) {
    auto is_drv_type = [&](const std::unique_ptr<Driver> &d) {
      return d->kernel_driver_type_ == drv_type;
    };

    auto driver(std::find_if(agent_drivers_.begin(), agent_drivers_.end(),
                             is_drv_type));

    if (driver == agent_drivers_.end()) {
      throw AMD::hsa_exception(HSA_STATUS_ERROR_INVALID_ARGUMENT,
                               "Invalid agent device type, no driver found.");
    }

    return **driver;
  }

  std::vector<std::unique_ptr<Driver>>& AgentDrivers() { return agent_drivers_; }

  static bool IsGPUDriver(DriverType driver_type) {
    return driver_type == core::DriverType::KFD
#ifdef HSAKMT_VIRTIO_ENABLED
        || driver_type == core::DriverType::KFD_VIRTIO
#endif
        ;
  }

 protected:
  static void AsyncEventsLoop(void*);
  static void AsyncIPCSockServerConnLoop(void*);

  struct AllocationRegion {
    AllocationRegion()
        : region(NULL),
          size(0),
          size_requested(0),
          alloc_flags(core::MemoryRegion::AllocateNoFlags),
          user_ptr(nullptr),
          ldrm_bo(NULL) {}
    AllocationRegion(const MemoryRegion* region_arg, size_t size_arg, size_t size_requested,
                     MemoryRegion::AllocateFlags alloc_flags)
        : region(region_arg),
          size(size_arg),
          size_requested(size_requested),
          alloc_flags(alloc_flags),
          user_ptr(nullptr),
          ldrm_bo(NULL) {}

    struct notifier_t {
      void* ptr;
      AMD::callback_t<hsa_amd_deallocation_callback_t> callback;
      void* user_data;
    };

    const MemoryRegion* region;
    size_t size;           /* actual size = align_up(size_requested, granularity) */
    size_t size_requested; /* size requested by user */
    MemoryRegion::AllocateFlags alloc_flags;
    void* user_ptr;
    std::unique_ptr<std::vector<notifier_t>> notifiers;
    amdgpu_bo_handle ldrm_bo;
  };

  struct AsyncEventsControl {
    AsyncEventsControl() : async_events_thread_(NULL) {}
    void Shutdown();

    hsa_signal_t wake;
    os::Thread async_events_thread_;
    HybridMutex lock;
    bool exit;
  };

  struct AsyncEvents {
    void PushBack(hsa_signal_t signal, hsa_signal_condition_t cond,
                  hsa_signal_value_t value, hsa_amd_signal_handler handler,
                  void* arg);

    void CopyIndex(size_t dst, size_t src);

    size_t Size();

    void PopBack();

    void Clear();

    std::vector<hsa_signal_t> signal_;
    std::vector<hsa_signal_condition_t> cond_;
    std::vector<hsa_signal_value_t> value_;
    std::vector<hsa_amd_signal_handler> handler_;
    std::vector<HsaEvent*> hsa_events_; //!< A list of HSA events for KFD wait
    std::vector<uint64_t> age_;         //!< The age list for KFD wait
    std::vector<void*> arg_;
  };

  struct PrefetchRange;
  typedef std::map<uintptr_t, PrefetchRange> prefetch_map_t;

  struct PrefetchOp {
    void* base;
    size_t size;
    uint32_t node_id;
    int remaining_deps;
    hsa_signal_t completion;
    std::vector<hsa_signal_t> dep_signals;
    prefetch_map_t::iterator prefetch_map_entry;
  };

  struct PrefetchRange {
    PrefetchRange() {}
    PrefetchRange(size_t Bytes, PrefetchOp* Op) : bytes(Bytes), op(Op) {}
    size_t bytes;
    PrefetchOp* op;
    prefetch_map_t::iterator prev;
    prefetch_map_t::iterator next;
  };

  // Will be created before any user could call hsa_init but also could be
  // destroyed before incorrectly written programs call hsa_shutdown.
  static __forceinline KernelMutex& bootstrap_lock() {
    // This allocation is meant to last until the last thread has exited.
    // It is intentionally not freed.
    static KernelMutex* bootstrap_lock_ = new KernelMutex;
    return *bootstrap_lock_;
  }
  Runtime();

  Runtime(const Runtime&);

  Runtime& operator=(const Runtime&);

  ~Runtime() {}

  /// @brief Open connection to kernel driver.
  hsa_status_t Load();

  /// @brief Close connection to kernel driver and cleanup resources.
  void Unload();

  /// @brief Dynamically load extension libraries (images, finalizer) and
  /// call OnLoad method on each loaded library.
  void LoadExtensions();

  /// @brief Call OnUnload method on each extension library then close it.
  void UnloadExtensions();

  /// @brief Dynamically load tool libraries and call OnUnload method on each
  /// loaded library.
  void LoadTools();

  /// @brief Call OnUnload method of each tool library.
  void UnloadTools();

  /// @brief Close tool libraries.
  void CloseTools();

  // @brief Binds Error handlers to this node.
  void BindErrorHandlers();

  // @brief Acquire snapshot of system event handlers.
  // Returns a copy to avoid holding a lock during callbacks.
  std::vector<std::pair<AMD::callback_t<hsa_amd_system_event_callback_t>, void*>>
  GetSystemEventHandlers();

  /// @brief Get the index of ::link_matrix_.
  /// @param [in] node_id_from Node id of the source node.
  /// @param [in] node_id_to Node id of the destination node.
  /// @retval Index in ::link_matrix_.
  uint32_t GetIndexLinkInfo(uint32_t node_id_from, uint32_t node_id_to);

  /// @brief Get most recently issued SVM prefetch agent for the range in question.
  Agent* GetSVMPrefetchAgent(void* ptr, size_t size);

  /// @brief Get the highest used node id.
  uint32_t max_node_id() const { return agents_by_node_.rbegin()->first; }

  // Mutex object to protect multithreaded access to ::allocation_map_.
  // Also ensures atomicity of pointer info queries by interlocking
  // KFD map/unmap, register/unregister, and access to hsaKmtQueryPointerInfo
  // registered & mapped arrays.
  KernelSharedMutex memory_lock_;

  // Array containing driver interfaces for compatible agent kernel-mode
  // drivers. Currently supports AIE agents.
  std::vector<std::unique_ptr<Driver>> agent_drivers_;

  // Array containing tools library handles.
  std::vector<os::LibHandle> tool_libs_;

  // Agent list containing all CPU agents in the platform.
  std::vector<Agent*> cpu_agents_;

  // Agent list containing all compatible GPU agents in the platform.
  std::vector<Agent*> gpu_agents_;

  // Agent list containing all compatible AIE agents in the platform.
  std::vector<Agent *> aie_agents_;

  // Agent list containing incompletely initialized GPU agents not to be used by the process.
  std::vector<Agent*> disabled_gpu_agents_;

  // Agent map containing all agents indexed by their KFD node IDs.
  std::map<uint32_t, std::vector<Agent*> > agents_by_node_;

  // Agent map containing all agents indexed by their KFD gpuid.
  std::map<uint32_t, Agent*> agents_by_gpuid_;

  // Agent list containing all compatible gpu agent ids in the platform.
  std::vector<uint32_t> gpu_ids_;

  // List of all fine grain system memory region in the platform.
  std::vector<const MemoryRegion*> system_regions_fine_;

  // List of all coarse grain system memory region in the platform.
  std::vector<const MemoryRegion*> system_regions_coarse_;

  // Matrix of IO link.
  std::vector<LinkInfo> link_matrix_;

  // Loader instance.
  amd::hsa::loader::Loader* loader_;

  // Loader context.
  amd::LoaderContext loader_context_;

  // Code object manager.
  amd::hsa::code::AmdHsaCodeManager code_manager_;

  // Contains the region, address, and size of previously allocated memory.
  std::map<const void*, AllocationRegion> allocation_map_;

  // Pending prefetch containers.
  KernelMutex prefetch_lock_;
  prefetch_map_t prefetch_map_;

  // Allocator using ::system_region_
  std::function<void*(size_t size, size_t align, MemoryRegion::AllocateFlags flags, int agent_node_id)> system_allocator_;

  // Deallocator using ::system_region_
  std::function<void(void*)> system_deallocator_;

  // Deprecated HSA Region API GPU (for legacy APU support only)
  Agent* region_gpu_;

  struct AsyncEventsInfo {
    AsyncEventsControl control;
    AsyncEvents events;
    AsyncEvents new_events;
    bool monitor_exceptions;
  };

  struct AsyncEventsInfo asyncSignals_;
  struct AsyncEventsInfo asyncExceptions_;

  // System clock frequency.
  uint64_t sys_clock_freq_;

  // Number of Numa Nodes
  size_t num_nodes_;

  // @brief AMD HSA event to monitor for virtual memory access fault.
  HsaEvent* vm_fault_event_;

  // @brief HSA signal to contain the VM fault event.
  Signal* vm_fault_signal_;

  // @brief AMD HSA event to monitor for HW exceptions.
  HsaEvent* hw_exception_event_;

  // @brief HSA signal to contain the HW exceptionevent.
  Signal* hw_exception_signal_;

  // Custom system event handlers.
  std::vector<std::pair<AMD::callback_t<hsa_amd_system_event_callback_t>, void*>>
      system_event_handlers_;

  // System event handler lock
  KernelMutex system_event_lock_;

  // Internal queue creation notifier
  AMD::callback_t<hsa_amd_runtime_queue_notifier> internal_queue_create_notifier_;

  void* internal_queue_create_notifier_user_data_;

  // Holds reference count to runtime object.
  std::atomic<uint32_t> ref_count_;

  // Track environment variables.
  Flag flag_;

  ThunkLoader* thunkLoader_;

  // Pools memory for SharedSignal (Signal ABI blocks)
  SharedSignalPool_t SharedSignalPool;

  // Pools KFD Events for InterruptSignal
  InterruptSignal::EventPool EventPool;

  // Kfd version
  KfdVersion_t kfd_version;

  std::unique_ptr<AMD::SvmProfileControl> svm_profile_;

  // IPC DMA buf unix domain socket server dmabuf FD passing
  int ipc_sock_server_fd_;
  std::map<uint64_t, int> ipc_sock_server_conns_;
  KernelMutex ipc_sock_server_lock_;

 private:
  void CheckVirtualMemApiSupport();
  int GetAmdgpuDeviceArgs(Agent *agent, ShareableHandle handle, int *drm_fd,
                          uint64_t *cpu_addr);

  bool virtual_mem_api_supported_;
  bool xnack_enabled_;

  typedef void* ThunkHandle;

  struct AddressHandle {
    AddressHandle() : os_addr(nullptr), size(0), use_count(0), registered(false) {}
    AddressHandle(void* addr, size_t _size, bool _registered) : os_addr(addr), size(_size), use_count(0), registered(_registered) {}

    // Address returned by OS. May be different from user address when adjusted for alignment
    void *os_addr;
    size_t size;
    int use_count;
    bool registered;
  };
  std::map<const void*, AddressHandle> reserved_address_map_;  // Indexed by VA

  struct MemoryHandle {
    MemoryHandle() : region(NULL), size(0), ref_count(0), thunk_handle(NULL), alloc_flag(0) {}
    MemoryHandle(const MemoryRegion* region, size_t size, uint64_t flags_unused,
                 ThunkHandle thunk_handle, MemoryRegion::AllocateFlags alloc_flag)
        : region(region),
          size(size),
          ref_count(1),
          use_count(0),
          thunk_handle(thunk_handle),
          alloc_flag(alloc_flag) {}

    static __forceinline hsa_amd_vmem_alloc_handle_t Convert(void* handle) {
      hsa_amd_vmem_alloc_handle_t ret_handle = {
          static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle))};
      return ret_handle;
    }

    __forceinline core::Agent* agentOwner() const { return region->owner(); }

    const MemoryRegion* region;
    size_t size;
    int ref_count;
    int use_count;
    ThunkHandle thunk_handle;  // handle returned by hsaKmtAllocMemory(NoAddress = 1)
    MemoryRegion::AllocateFlags alloc_flag;
  };
  std::map<ThunkHandle, MemoryHandle> memory_handle_map_;

  struct MappedHandle;
  struct MappedHandleAllowedAgent {
    MappedHandleAllowedAgent(MappedHandle* _mappedHandle, Agent* targetAgent, void* va, size_t size,
                             hsa_access_permission_t perms);
    ~MappedHandleAllowedAgent();

    hsa_status_t RemoveAccess();
    hsa_status_t EnableAccess(hsa_access_permission_t perms);

    void* va;
    size_t size;
    Agent* targetAgent;
    hsa_access_permission_t permissions;
    MappedHandle* mappedHandle;
    ShareableHandle shareable_handle;
  };

  struct MappedHandle {
    MappedHandle(MemoryHandle *mem_handle, AddressHandle *address_handle,
                 uint64_t offset, size_t size, int drm_fd, void *drm_cpu_addr,
                 hsa_access_permission_t perm, ShareableHandle shareable_handle)
        : mem_handle(mem_handle), address_handle(address_handle),
          offset(offset), size(size), drm_fd(drm_fd),
          drm_cpu_addr(drm_cpu_addr), shareable_handle(shareable_handle) {}

    __forceinline core::Agent* agentOwner() const { return mem_handle->region->owner(); }

    MemoryHandle* mem_handle;
    AddressHandle* address_handle;
    uint64_t offset;
    size_t size;
    int drm_fd;
    void* drm_cpu_addr;  // CPU Buffer address
    ShareableHandle shareable_handle;
    std::map<Agent*, MappedHandleAllowedAgent> allowed_agents;
  };
  std::map<const void*, MappedHandle> mapped_handle_map_;  // Indexed by VA

  hsa_status_t VMemoryMapAllowAccess(const void *va,
                                     hsa_access_permission_t perm,
                                     const hsa_agent_t *agents,
                                     size_t num_agents);
  hsa_status_t
  VMemorySetAccessPerHandle(void *va, MappedHandle &MappedHandle,
                            const hsa_amd_memory_access_desc_t *desc,
                            const size_t desc_cnt);

  void InitIPCDmaBufSupport();
  bool ipc_dmabuf_supported_;
  int  IPCClientImport(uint32_t conn_handle, uint64_t dmabuf_fd_handle,
                       amdgpu_bo_import_result *res,
                       unsigned int numNodes, HSAuint32 *nodes,
                       void **importAddress, HSAuint64 *importSize);
};

}  // namespace core
}  // namespace rocr
#endif  // header guard
