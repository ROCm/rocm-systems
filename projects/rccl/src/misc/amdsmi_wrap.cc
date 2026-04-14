// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "amdsmi_wrap.h"
#include "alt_rsmi.h"
#include "core.h"
#include "utils.h"
#include <cstdio>
#include <vector>
#include <cstring>
#include <mutex>
#include <atomic>

static int is_wsl2 = -1;

#define AMDSMICHECK(cmd) do {                \
  amdsmi_status_t ret = cmd;                 \
  if( ret != AMDSMI_STATUS_SUCCESS ) {       \
    const char *err;                         \
    pfn_amdsmi_status_code_to_string(ret, &err);         \
    ERROR("AMD SMI failure: %s at line: %d in file: %s", err, __LINE__, __FILE__);    \
    return ncclInternalError;                \
  }                                          \
} while(false)

#define ARSMICHECK(cmd) do {         \
  int ret = cmd;                     \
  if( ret != 0 ) {                   \
    ERROR("ARSMI failure: %d", ret); \
    return ncclInternalError;        \
  }                                  \
} while(false)

#define AMDSMITRY(name, ...) do { \
  if (!AMDSMI_DIRECT && pfn_##name == nullptr) \
    return ncclInternalError; /* missing symbol is not a warned error */ \
  amdsmi_status_t ret = pfn_##name(__VA_ARGS__); \
  if( ret != AMDSMI_STATUS_SUCCESS ) {       \
    const char *err;                         \
    pfn_amdsmi_status_code_to_string(ret, &err); \
    ERROR("AMD SMI failure: %s at line: %d in file: %s", err, __LINE__, __FILE__);    \
    return ncclInternalError;                \
  }                                          \
} while(0)

RCCL_PARAM(UseAmdSmiLib, "USE_AMD_SMI_LIB", 0); // Opt-in environment variable for enabling using amd_smi_lib instead of internal code


#include <dlfcn.h>
#define RCCL_AMDSMI_FN(name, rettype, arglist) rettype(*pfn_##name)arglist = nullptr;


namespace {
  // Core AMD SMI functions
  RCCL_AMDSMI_FN(amdsmi_init, amdsmi_status_t, (uint64_t init_flags))
  RCCL_AMDSMI_FN(amdsmi_shut_down, amdsmi_status_t, ())
  RCCL_AMDSMI_FN(amdsmi_status_code_to_string, amdsmi_status_t, (amdsmi_status_t status, const char **status_string))
  RCCL_AMDSMI_FN(amdsmi_get_lib_version, amdsmi_status_t, (amdsmi_version_t *version))
  // Socket and processor handle functions
  RCCL_AMDSMI_FN(amdsmi_get_socket_handles, amdsmi_status_t, (uint32_t *socket_count, amdsmi_socket_handle* socket_handles))
  RCCL_AMDSMI_FN(amdsmi_get_processor_handles, amdsmi_status_t, (amdsmi_socket_handle socket_handle, uint32_t *processor_count, amdsmi_processor_handle* processor_handles))
  RCCL_AMDSMI_FN(amdsmi_get_processor_type, amdsmi_status_t, (amdsmi_processor_handle processor_handle, processor_type_t* processor_type))
  RCCL_AMDSMI_FN(amdsmi_get_processor_handle_from_bdf, amdsmi_status_t, (amdsmi_bdf_t bdf, amdsmi_processor_handle* processor_handle))
  // GPU enumeration and BDF functions
  RCCL_AMDSMI_FN(amdsmi_get_gpu_enumeration_info, amdsmi_status_t, (amdsmi_processor_handle processor_handle, amdsmi_enumeration_info_t *info))
  RCCL_AMDSMI_FN(amdsmi_get_gpu_bdf_id, amdsmi_status_t, (amdsmi_processor_handle processor_handle, uint64_t *bdfid))
  // Topology functions
  RCCL_AMDSMI_FN(amdsmi_topo_get_link_type, amdsmi_status_t, (amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t *hops, amdsmi_link_type_t *type))
  RCCL_AMDSMI_FN(amdsmi_topo_get_link_weight, amdsmi_status_t, (amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t *weight))
  RCCL_AMDSMI_FN(amdsmi_get_minmax_bandwidth_between_processors, amdsmi_status_t, (amdsmi_processor_handle processor_handle_src, amdsmi_processor_handle processor_handle_dst, uint64_t *min_bandwidth, uint64_t *max_bandwidth))
  // UALoE Fabric support
  RCCL_AMDSMI_FN(amdsmi_get_gpu_fabric_info, amdsmi_status_t, (amdsmi_processor_handle processor_handle, amdsmi_fabric_info_t* info))
  // UALoE Fabric Telemetry support
  RCCL_AMDSMI_FN(amdsmi_alloc_fabric_telemetry, amdsmi_status_t, (amdsmi_processor_handle processor_handle, uint32_t category_mask, amdsmi_fabric_telemetry_t **telemetry))
  RCCL_AMDSMI_FN(amdsmi_get_fabric_telemetry_data, amdsmi_status_t, (amdsmi_processor_handle processor_handle, amdsmi_fabric_telemetry_t *telemetry))
  RCCL_AMDSMI_FN(amdsmi_free_fabric_telemetry, amdsmi_status_t, (amdsmi_processor_handle processor_handle, amdsmi_fabric_telemetry_t *telemetry))
  RCCL_AMDSMI_FN(amdsmi_fabric_telem_id_to_string, const char*, (uint64_t telem_id))
  // Firmware info
  RCCL_AMDSMI_FN(amdsmi_get_fw_info, amdsmi_status_t, (amdsmi_processor_handle processor_handle, amdsmi_fw_info_t *info))
}

/*************************************************************************
 * UALoE Fabric Support - Global State
 * We cache fabric info
 * for all devices during initialization.
 ************************************************************************/

int amdsmiFabricDeviceCount = 0;
struct amdsmiFabricDeviceInfo amdsmiFabricDevices[amdsmiFabricMaxDevices];

namespace {
  std::mutex fabricLock;  // Thread safety for fabric operations
  bool fabricInitialized = false;
  thread_local bool threadFabricInitialized = false;
  ncclResult_t fabricInitResult = ncclSuccess;
  ncclResult_t amdSmiInitResult = ncclSuccess;
  std::atomic<bool> amdSmiInitCalled{false};  // Track if amd_smi_init has been called
}

/*************************************************************************
 * Helper: Get processor handle for a device index
 ************************************************************************/
static ncclResult_t getProcessorHandle(uint32_t deviceIndex, amdsmi_processor_handle* procHandle) {
  if (!rcclParamUseAmdSmiLib()) {
    return ncclSystemError;  // Fabric only supported with amd_smi_lib
  }

  uint32_t socket_count = 0;
  AMDSMITRY(amdsmi_get_socket_handles, &socket_count, nullptr);
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  AMDSMITRY(amdsmi_get_socket_handles, &socket_count, sockets.data());

  for (auto& socket : sockets) {
    uint32_t processor_handle_count = 0;
    AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, nullptr);
    std::vector<amdsmi_processor_handle> processor_handles(processor_handle_count);
    AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, processor_handles.data());

    for (auto& proc : processor_handles) {
      processor_type_t type;
      AMDSMITRY(amdsmi_get_processor_type, proc, &type);
      if (type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        amdsmi_enumeration_info_t info;
        AMDSMITRY(amdsmi_get_gpu_enumeration_info, proc, &info);
        if (info.hip_id == deviceIndex) {
          *procHandle = proc;
          return ncclSuccess;
        }
      }
    }
  }
  return ncclInternalError;
}

static bool amd_smi_FabricFunctionsLoaded() {
  return (pfn_amdsmi_get_gpu_fabric_info != nullptr);
}

/*************************************************************************
 * Existing AMD SMI Wrapper Functions
 ************************************************************************/

ncclResult_t amd_smi_init() {
  // Ensure we only initialize once
  if (amdSmiInitCalled.exchange(true)) return amdSmiInitResult;

  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE) == -1)
    __atomic_store_n(&is_wsl2, (access("/dev/dxg", F_OK) == -1) ? 0 : 1, __ATOMIC_RELEASE);
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    INFO(NCCL_INIT, "Not using amdsmi_lib due to WSL2 environment detected.");
    return ncclSuccess;
  }

  if (rcclParamUseAmdSmiLib()) {
    if (pfn_amdsmi_init == nullptr) {
      static void *libhandle = dlopen("libamd_smi.so", RTLD_NOW);
      if (libhandle == nullptr) {
        WARN("Failed to open libamd_smi.so");
        amdSmiInitResult = ncclInternalError;
        return ncclInternalError;
      }

      struct Symbol { void **ppfn; char const *name; };
      std::initializer_list<Symbol> symbols = {
        {(void**)&pfn_amdsmi_init, "amdsmi_init"},
        {(void**)&pfn_amdsmi_shut_down, "amdsmi_shut_down"},
        {(void**)&pfn_amdsmi_status_code_to_string, "amdsmi_status_code_to_string"},
        {(void**)&pfn_amdsmi_get_lib_version, "amdsmi_get_lib_version"},
        {(void**)&pfn_amdsmi_get_socket_handles, "amdsmi_get_socket_handles"},
        {(void**)&pfn_amdsmi_get_processor_handles, "amdsmi_get_processor_handles"},
        {(void**)&pfn_amdsmi_get_processor_type, "amdsmi_get_processor_type"},
        {(void**)&pfn_amdsmi_get_processor_handle_from_bdf, "amdsmi_get_processor_handle_from_bdf"},
        {(void**)&pfn_amdsmi_get_gpu_enumeration_info, "amdsmi_get_gpu_enumeration_info"},
        {(void**)&pfn_amdsmi_get_gpu_bdf_id, "amdsmi_get_gpu_bdf_id"},
        {(void**)&pfn_amdsmi_topo_get_link_type, "amdsmi_topo_get_link_type"},
        {(void**)&pfn_amdsmi_topo_get_link_weight, "amdsmi_topo_get_link_weight"},
        {(void**)&pfn_amdsmi_get_minmax_bandwidth_between_processors, "amdsmi_get_minmax_bandwidth_between_processors"},
        // UALoE support
        {(void**)&pfn_amdsmi_get_gpu_fabric_info, "amdsmi_get_gpu_fabric_info"},
        {(void**)&pfn_amdsmi_fabric_telem_id_to_string, "amdsmi_fabric_telem_id_to_string"},
        // UALoE Telemetry support
        {(void**)&pfn_amdsmi_alloc_fabric_telemetry, "amdsmi_alloc_fabric_telemetry"},
        {(void**)&pfn_amdsmi_get_fabric_telemetry_data, "amdsmi_get_fabric_telemetry_data"},
        {(void**)&pfn_amdsmi_free_fabric_telemetry, "amdsmi_free_fabric_telemetry"},
        {(void**)&pfn_amdsmi_get_fw_info, "amdsmi_get_fw_info"},
      };
      for(Symbol sym: symbols) {
        *sym.ppfn = dlsym(libhandle, sym.name);
      }
    }

    // initialize amd-smi for AMD GPUs
    AMDSMITRY(amdsmi_init, AMDSMI_INIT_AMD_GPUS);

    // get amd-smi version
    amdsmi_version_t version;
    AMDSMITRY(amdsmi_get_lib_version, &version);
    INFO(NCCL_INIT, "amdsmi_lib: version %d.%d.%d.%s", version.major, version.minor, version.release, version.build);
  } else {
#ifdef HIP_FABRIC_API
    WARN("RCCL_USE_AMD_SMI_LIB not set, but HIP_FABRIC_API is defined. Fabric support is only available through AMD SMI. Rerun with RCCL_USE_AMD_SMI_LIB=1 to enable AMD SMI and UALoE fabric support.");
#endif
    // initialize alternate rsmi
    ARSMICHECK(ARSMI_init());
    INFO(NCCL_INIT, "initialized internal alternative rsmi functionality");
  }
  return ncclSuccess;
}

ncclResult_t amd_smi_shutdown() {
  // Only shutdown if we actually initialized with amd_smi_lib
  if (!rcclParamUseAmdSmiLib() || pfn_amdsmi_shut_down == nullptr) {
    return ncclSuccess;
  }
  amdSmiInitCalled.store(false);
  amdSmiInitResult = ncclSuccess;

  AMDSMITRY(amdsmi_shut_down);
  return ncclSuccess;
}

ncclResult_t amd_smi_getNumDevice(uint32_t* num_devs) {
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE))
    CUDACHECK(cudaGetDeviceCount((int *)num_devs));
  else {
    if (rcclParamUseAmdSmiLib()) {
      // rsmi_num_monitor_devices is deprecated

      // with amd-smi, first get list of socket handles,
      // then get number of processor handles in said sockets,
      // and then query no. of gpus in said processor handles
      uint32_t socket_count = 0;
      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, nullptr);
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, sockets.data());

      uint32_t total_gpus = 0;
      for (auto& socket : sockets) {
        uint32_t num_gpus_per_socket = 0;
        AMDSMITRY(amdsmi_get_processor_handles, socket, &num_gpus_per_socket, nullptr);
        std::vector<amdsmi_processor_handle> processor_handles(num_gpus_per_socket);
        AMDSMITRY(amdsmi_get_processor_handles, socket, &num_gpus_per_socket, processor_handles.data());
        total_gpus += num_gpus_per_socket;
      }
      *num_devs = total_gpus;
    } else {
      ARSMICHECK(ARSMI_get_num_devices(num_devs));
    }
  }
  return ncclSuccess;
}

ncclResult_t amd_smi_getDevicePciBusIdString(uint32_t deviceIndex, char* busId, size_t len) {
  uint64_t id = 0;
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    CUDACHECK(cudaDeviceGetPCIBusId(busId, len, deviceIndex));
  } else {
    /** amd-smi's bus ID format
     *  | Name        | Field   |
     *  ------------- | ------- |
     *  | Domain      | [63:16] |
     *  | Bus         | [15: 8] |
     *  | Device      | [ 7: 3] |
     *  | Function    | [ 2: 0] |
     **/
    if (rcclParamUseAmdSmiLib()) {
      // rsmi_dev_pci_id_get is deprecated

      /// with amd-smi, first get list of socket handles,
      // then get number of processor handles in said sockets,
      // and then query the BDF for GPU matching deviceIndex in said processor handles
      uint32_t socket_count = 0;
      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, nullptr);
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, sockets.data());
      bool found = false;
      id = 0;
      for (auto& socket : sockets) {
        uint32_t processor_handle_count = 0;
        AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, nullptr);
        std::vector<amdsmi_processor_handle> processor_handles(processor_handle_count);
        AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, processor_handles.data());

        // this does not work?
        // AMDSMICHECK(amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_GPU, nullptr, &num_gpus_per_socket));
        // workaround
        for (auto& proc : processor_handles) {
          processor_type_t type;

          AMDSMITRY(amdsmi_get_processor_type, proc, &type);
          if(type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
            amdsmi_enumeration_info_t info;
            AMDSMITRY(amdsmi_get_gpu_enumeration_info, proc, &info);
            if(info.hip_id == deviceIndex) {
              AMDSMITRY(amdsmi_get_gpu_bdf_id, proc, &id);
              found = true;
              break;
            }
          }
        }
        if (found) break;
      }
      if (!found) {
        ERROR("amdsmi_lib: device index %u not found", deviceIndex);
        return ncclInternalError;
      }
      // borrowing NCCL's format from utils.cc:int64ToBusId
      // !! To be reconciled after discussion with amdsmi team !!
      snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id) >> 32, (id & 0xff00) >> 8, (id & 0xf8) >> 3, (id & 0x7));
      // snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
    } else {
    // rocm-smi format
      ARSMICHECK(ARSMI_dev_pci_id_get(deviceIndex, &id));
      // ARSMI uses the same BDF packing as rocm_smi.
      // Keep this formatting identical to rocm_smi_wrap to avoid
      // generating inconsistent PCI IDs in topology XML.
      snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id) >> 32, (id & 0xff00) >> 8, (id & 0xf8) >> 3, (id & 0x7));
    }
  }
  return ncclSuccess;
}


ncclResult_t amd_smi_getDeviceIndexByPciBusId(const char* pciBusId, uint32_t* deviceIndex) {
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    CUDACHECK(hipDeviceGetByPCIBusId((int *)deviceIndex, pciBusId));
    return ncclSuccess;
  } else {
    int64_t busid;

    busIdToInt64(pciBusId, &busid);
    /** convert to amd-smi's bus ID format
     *  | Name        | Field   |
     *  ------------- | ------- |
     *  | Domain      | [63:16] |
     *  | Bus         | [15: 8] |
     *  | Device      | [ 7: 3] |
     *  | Function    | [ 2: 0] |
     **/

    // instead of getting device count and then comparing the busid to each GPUs BDF

    // with amd-smi, we can use amdsmi_get_processor_handle_from_bdf,
    // and then query the enumeration info for that processor_handle
    if (rcclParamUseAmdSmiLib()) {
      amdsmi_processor_handle processor_handle = 0;

      amdsmi_bdf_t bdf = {};
      // This is the format that matches amd-smi BDF
      // bdf.function_number = (busid & 0x7);
      // bdf.device_number = (busid & 0xf8) >> 3;
      // bdf.bus_number = (busid & 0xff00) >> 8;
      // bdf.domain_number = (busid & 0xffffffffffff0000) >> 16;

      // However, it is incompatible with the format enforced by NCCL in utils.cc:int64ToBusId
      // !! To be reconciled after discussion with amdsmi team !!
      bdf.function_number = (busid & 0xf);
      bdf.device_number = (busid & 0xff) >> 4;
      bdf.bus_number = (busid & 0xff000) >> 12;
      bdf.domain_number = busid >> 20;

      AMDSMITRY(amdsmi_get_processor_handle_from_bdf, bdf, &processor_handle);

      processor_type_t type;
      AMDSMITRY(amdsmi_get_processor_type, processor_handle, &type);
      if(type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        amdsmi_enumeration_info_t info;
        AMDSMITRY(amdsmi_get_gpu_enumeration_info, processor_handle, &info);
        *deviceIndex = info.hip_id;
        return ncclSuccess;
      }

      ERROR("amdsmi_lib: %s device index not found", pciBusId);
    } else {
      uint32_t i, num_devs = 0;
      busid = ((busid&0xffff00000L)<<12)+((busid&0xff000L)>>4)+((busid&0xff0L)>>1)+(busid&0x7L);

      ARSMICHECK(ARSMI_get_num_devices(&num_devs));
      for (i = 0; i < num_devs; i++) {
        uint64_t bdfid;
        ARSMICHECK(ARSMI_dev_pci_id_get(i, &bdfid));
        if ((int64_t) bdfid == busid) break;
      }
      if (i < num_devs) {
        *deviceIndex = i;
        return ncclSuccess;
      }
      else {
        WARN("ARSMI_lib: %s device index not found", pciBusId);
      }
    }
    return ncclInternalError;
  }
}

ncclResult_t amd_smi_getLinkInfo(int srcIndex, int dstIndex, amdsmi_link_type_t* type, int *hops, int *count) {
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    *type = AMDSMI_LINK_TYPE_PCIE;
    *hops = 1;
    *count = 1;
  } else {
    amdsmi_link_type_t amdsmi_type;
    uint64_t amdsmi_hops, amdsmi_weight ;
    *count = 1;
    *hops = 2;
    // rsmi_minmax_bandwidth_get is replaced by amdsmi_get_minmax_bandwidth_between_processors
    // where the arguments for src and dst change from index to processor_handles

    // with amd-smi, first get list of socket handles,
    // then get number of processor handles in said sockets,
    // then get the prcoessor handle matching the src and dst index,
    // and then use these processor handles for amdsmi hardware topology functions
    if (rcclParamUseAmdSmiLib()) {
      uint32_t socket_count = 0;
      amdsmi_processor_handle src_processor_handle = 0;
      amdsmi_processor_handle dst_processor_handle = 0;
      bool found_src = false, found_dst = false;

      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, nullptr);
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMITRY(amdsmi_get_socket_handles, &socket_count, sockets.data());

      for (auto& socket : sockets) {
        uint32_t processor_handle_count = 0;
        AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, nullptr);
        std::vector<amdsmi_processor_handle> processor_handles(processor_handle_count);
        AMDSMITRY(amdsmi_get_processor_handles, socket, &processor_handle_count, processor_handles.data());

        // this does not work?
        // AMDSMICHECK(amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_GPU, nullptr, &num_gpus_per_socket));
        // workaround
        for (auto& proc : processor_handles) {
          processor_type_t proc_type;
          AMDSMITRY(amdsmi_get_processor_type, proc, &proc_type);
          if(proc_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
            amdsmi_enumeration_info_t info;
            AMDSMITRY(amdsmi_get_gpu_enumeration_info, proc, &info);
            if((int) info.hip_id == srcIndex) {
              src_processor_handle = proc;
              found_src = true;
            }
            if((int) info.hip_id == dstIndex) {
              dst_processor_handle = proc;
              found_dst = true;
            }
          }
        }
        if (found_src && found_dst) break;
      }
      if (!found_src) ERROR("amd-smi could not find processor handle for srcIndex: %d", srcIndex);
      if (!found_dst) ERROR("amd-smi could not find processor handle for dstIndex: %d", dstIndex);
      AMDSMITRY(amdsmi_topo_get_link_type, src_processor_handle, dst_processor_handle, &amdsmi_hops, &amdsmi_type);
      AMDSMITRY(amdsmi_topo_get_link_weight, src_processor_handle, dst_processor_handle, &amdsmi_weight);

      // amd-smi reports weight=0 for XGMI ??
      if (amdsmi_type == AMDSMI_LINK_TYPE_XGMI) {
        uint64_t min_bw = 0, max_bw = 0;
        AMDSMITRY(amdsmi_get_minmax_bandwidth_between_processors, src_processor_handle, dst_processor_handle, &min_bw, &max_bw);
        if (max_bw && min_bw) *count = max_bw/min_bw;
      }

      *type = amdsmi_type;
      *hops = amdsmi_hops;
    } else {
      ARSMI_linkInfo tinfo;
      ARSMICHECK(ARSMI_topo_get_link_info(srcIndex, dstIndex, &tinfo));

      *type  = (amdsmi_link_type_t) tinfo.type;
      if (*type == AMDSMI_LINK_TYPE_XGMI && (tinfo.weight == 15 ||
        tinfo.weight == 41 || tinfo.weight == 13)) {
        *hops = 1;
        if (tinfo.max_bandwidth && tinfo.min_bandwidth)
          *count = tinfo.max_bandwidth/tinfo.min_bandwidth;
      }
    }
  }

  return ncclSuccess;
}

ncclResult_t amd_smi_getFirmwareVersion(uint32_t deviceIndex, uint64_t* fwVersion) {
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    *fwVersion = 0;
    return ncclSuccess;  // Firmware query not supported on WSL2
  }

  if (rcclParamUseAmdSmiLib()) {
    // Use AMD SMI library
    amdsmi_processor_handle procHandle;
    NCCLCHECK(getProcessorHandle(deviceIndex, &procHandle));

    if (pfn_amdsmi_get_fw_info == nullptr) {
      ERROR("amdsmi_get_fw_info symbol not loaded");
      return ncclInternalError;
    }

    amdsmi_fw_info_t info;
    memset(&info, 0, sizeof(info));
    AMDSMITRY(amdsmi_get_fw_info, procHandle, &info);
    if (info.num_fw_info == 0) {
      *fwVersion = 0;
      return ncclSuccess;
    }
    *fwVersion = info.fw_info_list[0].fw_version;
  } else {
    // Use ARSMI fallback - avoid direct rsmi calls that can conflict with libamd_smi.so
    // ARSMI doesn't have a firmware query API, so return 0
    *fwVersion = 0;
  }
  return ncclSuccess;
}

/*************************************************************************
 * UALoE Fabric Wrapper Functions Implementation
 *
 * These functions provide access to AMD's UALoE fabric for scale-up
 * networking, similar to how nvmlwrap.cc provides MNNVL support for NVIDIA.
 ************************************************************************/

ncclResult_t amd_smi_ensureFabricInitialized() {
  // Optimization to avoid repeatedly grabbing the lock when we only want to
  // read from the global tables (same pattern as ncclNvmlEnsureInitialized)
  if (threadFabricInitialized) return fabricInitResult;
  threadFabricInitialized = true;

  std::lock_guard<std::mutex> locked(fabricLock);

  ncclResult_t initRes = amd_smi_init();
  if (initRes != ncclSuccess) {
    fabricInitResult = initRes;
    return fabricInitResult;
  }

  if (fabricInitialized) return fabricInitResult;
  fabricInitialized = true;

  // Fabric requires amd_smi_lib
  if (!rcclParamUseAmdSmiLib()) {
    INFO(NCCL_INIT, "UALoE fabric detection skipped: RCCL_USE_AMD_SMI_LIB not set");
    fabricInitResult = ncclSuccess;
    return fabricInitResult;
  }

  // WSL2 doesn't support fabric
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    INFO(NCCL_INIT, "UALoE fabric detection skipped: WSL2 environment");
    fabricInitResult = ncclSuccess;
    return fabricInitResult;
  }

  // Get device count
  uint32_t numDevs = 0;
  ncclResult_t res = amd_smi_getNumDevice(&numDevs);
  if (res != ncclSuccess) {
    fabricInitResult = res;
    return fabricInitResult;
  }

  if (numDevs > amdsmiFabricMaxDevices) {
    WARN("AMD SMI fabric: device count %u exceeds max %d, truncating",
         numDevs, amdsmiFabricMaxDevices);
    numDevs = amdsmiFabricMaxDevices;
  }

  amdsmiFabricDeviceCount = numDevs;

  // Initialize fabric info for each device
  for (uint32_t d = 0; d < numDevs; d++) {
    struct amdsmiFabricDeviceInfo* devInfo = &amdsmiFabricDevices[d];
    memset(devInfo, 0, sizeof(*devInfo));

    amdsmi_processor_handle procHandle;
    if (getProcessorHandle(d, &procHandle) != ncclSuccess || !amd_smi_FabricFunctionsLoaded()) {
      WARN("AMD SMI fabric: unable to get processor handle or fabric functions not loaded for device %u, skipping fabric detection", d);
      devInfo->fabricSupported = false;
      continue;
    }

    // Query fabric info from AMD SMI
    amdsmi_fabric_info_t fabricInfo;
    memset(&fabricInfo, 0, sizeof(fabricInfo));


    amdsmi_status_t status = pfn_amdsmi_get_gpu_fabric_info(procHandle, &fabricInfo);
    if (status != AMDSMI_STATUS_SUCCESS) {
      devInfo->fabricSupported = false;
      continue;
    }
    // Check fabric info version
    if (fabricInfo.info.version != AMDSMI_FABRIC_INFO_CURRENT_VERSION) {
      WARN("AMD SMI fabric: unexpected fabric info version %u for device %u, expected %u",
         fabricInfo.info.version, d, AMDSMI_FABRIC_INFO_CURRENT_VERSION);
      devInfo->fabricSupported = false;
      continue;
    }

    // Populate cached info from v1 structure
    const amdsmi_fabric_info_v1_t* v1 = &fabricInfo.info.v1;
    devInfo->fabricSupported = (v1->fabric_type == AMDSMI_FABRIC_TYPE_UALOE &&
                               (v1->accel_state == AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_ACTIVE ||
                                v1->accel_state == AMDSMI_FABRIC_ACCELERATOR_VPOD_STATE_READY));
    devInfo->fabricType = v1->fabric_type;
    devInfo->state = v1->accel_state;
    devInfo->acceleratorId = v1->accelerator_id;
    devInfo->bandwidth = v1->bandwidth;
    devInfo->latency = v1->latency;
    memcpy(devInfo->clusterUuid, v1->ppod_id, sizeof(v1->ppod_id));
    devInfo->ppodSize = v1->ppod_size;
    devInfo->cliqueId = v1->vpod_id;
    devInfo->vpodSize = v1->vpod_size;

    if (devInfo->fabricSupported) {
      uint64_t uuidHigh, uuidLow;
      memcpy(&uuidHigh, devInfo->clusterUuid, sizeof(uint64_t));
      memcpy(&uuidLow, devInfo->clusterUuid + sizeof(uint64_t), sizeof(uint64_t));
      INFO(NCCL_INIT, "GPU %d: UALoE fabric detected - accelId=%u bw=%uMb/s lat=%uns vpod=%u/%u uuid=%lx.%lx ppod_size=%u",
         d, devInfo->acceleratorId, devInfo->bandwidth, devInfo->latency,
         devInfo->cliqueId, devInfo->vpodSize, uuidHigh, uuidLow, devInfo->ppodSize);
    }

  }

  fabricInitResult = amd_smi_FabricFunctionsLoaded()? ncclSuccess : ncclInternalError;
  return fabricInitResult;
}

ncclResult_t amd_smi_isFabricSupported(uint32_t deviceIndex, bool* supported) {
  NCCLCHECK(amd_smi_ensureFabricInitialized());

  if (deviceIndex >= (uint32_t)amdsmiFabricDeviceCount) {
    *supported = false;
    return ncclSuccess;
  }

  *supported = amdsmiFabricDevices[deviceIndex].fabricSupported;
  return ncclSuccess;
}

ncclResult_t amd_smi_getFabricDeviceInfo(uint32_t deviceIndex, struct amdsmiFabricDeviceInfo* info) {
  NCCLCHECK(amd_smi_ensureFabricInitialized());

  if (deviceIndex >= (uint32_t)amdsmiFabricDeviceCount) {
    return ncclInvalidArgument;
  }

  *info = amdsmiFabricDevices[deviceIndex];
  return ncclSuccess;
}

ncclResult_t amd_smi_getFabricBandwidth(uint32_t deviceIndex, uint32_t* bandwidthMbps) {
  NCCLCHECK(amd_smi_ensureFabricInitialized());

  if (deviceIndex >= (uint32_t)amdsmiFabricDeviceCount) {
    *bandwidthMbps = 0;
    return ncclSuccess;
  }

  const struct amdsmiFabricDeviceInfo* devInfo = &amdsmiFabricDevices[deviceIndex];
  if (devInfo->fabricSupported) {
    *bandwidthMbps = devInfo->bandwidth;
  } else {
    *bandwidthMbps = 0;  // Indicate fallback to arch-based defaults
  }

  return ncclSuccess;
}

ncclResult_t amd_smi_allocFabricTelemetry(uint32_t deviceIndex,
                                          uint32_t categoryMask,
                                          amdsmi_fabric_telemetry_t** telemetry) {
  if (!rcclParamUseAmdSmiLib()) {
    return ncclSystemError;
  }

  amdsmi_processor_handle procHandle;
  NCCLCHECK(getProcessorHandle(deviceIndex, &procHandle));

  std::lock_guard<std::mutex> locked(fabricLock);
  AMDSMITRY(amdsmi_alloc_fabric_telemetry, procHandle, categoryMask, telemetry);
  return ncclSuccess;
}

ncclResult_t amd_smi_getFabricTelemetryData(uint32_t deviceIndex,
                                            amdsmi_fabric_telemetry_t* telemetry) {
  if (!rcclParamUseAmdSmiLib() || telemetry == nullptr) {
    return ncclSystemError;
  }
  amdsmi_processor_handle procHandle;
  NCCLCHECK(getProcessorHandle(deviceIndex, &procHandle));

  std::lock_guard<std::mutex> locked(fabricLock);
  AMDSMITRY(amdsmi_get_fabric_telemetry_data, procHandle, telemetry);
  return ncclSuccess;
}

ncclResult_t amd_smi_freeFabricTelemetry(uint32_t deviceIndex,
                                         amdsmi_fabric_telemetry_t* telemetry) {
  if (!rcclParamUseAmdSmiLib() || telemetry == nullptr) {
    return ncclSystemError;
  }

  amdsmi_processor_handle procHandle;
  NCCLCHECK(getProcessorHandle(deviceIndex, &procHandle));

  std::lock_guard<std::mutex> locked(fabricLock);
  AMDSMITRY(amdsmi_free_fabric_telemetry, procHandle, telemetry);
  return ncclSuccess;
}

const char* amd_smi_fabricTelemIdToString(uint64_t telemId) {
  if (pfn_amdsmi_fabric_telem_id_to_string == nullptr) {
    return "UNKNOWN";
  }
  return pfn_amdsmi_fabric_telem_id_to_string(telemId);
}
