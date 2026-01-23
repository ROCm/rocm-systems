// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.

#include "amdsmi_wrap.h"
#include "alt_rsmi.h"
#include "core.h"
#include "utils.h"
#include <cstdio>
#include <vector>
#include <cstring>

static int is_wsl2 = -1;

#define AMDSMICHECK(cmd) do {                \
  amdsmi_status_t ret = cmd;                 \
  if( ret != AMDSMI_STATUS_SUCCESS ) {       \
    const char *err;                         \
    amdsmi_status_code_to_string(ret, &err);         \
    ERROR("AMD SMI failure: %s at line: %d in file: %s", err, __LINE__, __FILE__);    \
    return ncclInternalError;                \
  }                                          \
} while(false)

#define ARSMICHECK(cmd) do {         \
  int ret = cmd;                     \
  if( ret != 0 ) {		     \
    ERROR("ARSMI failure: %d", ret);   \
    return ncclInternalError;        \
  }                                  \
} while(false)

RCCL_PARAM(UseAmdSmiLib, "USE_AMD_SMI_LIB", 0); // Opt-in environment variable for enabling using amd_smi_lib instead of internal code

ncclResult_t amd_smi_init() {
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE) == -1)
    __atomic_store_n(&is_wsl2, (access("/dev/dxg", F_OK) == -1) ? 0 : 1, __ATOMIC_RELEASE);
  if (__atomic_load_n(&is_wsl2, __ATOMIC_ACQUIRE)) {
    INFO(NCCL_INIT, "Not using amdsmi_lib due to WSL2 environment detected.");
    return ncclSuccess;
  }

  if (rcclParamUseAmdSmiLib()) {
    // initialize amd-smi for AMD GPUs
    AMDSMICHECK(amdsmi_init(AMDSMI_INIT_AMD_GPUS));

    // get amd-smi version
    amdsmi_version_t version;
    AMDSMICHECK(amdsmi_get_lib_version(&version));
    INFO(NCCL_INIT, "amdsmi_lib: version %d.%d.%d.%s", version.major, version.minor, version.release, version.build);
  } else {
    // initialize alternate rsmi
    ARSMICHECK(ARSMI_init());
    INFO(NCCL_INIT, "initialized internal alternative rsmi functionality");
  }
  return ncclSuccess;
}

ncclResult_t amd_smi_shutdown() {
  AMDSMICHECK(amdsmi_shut_down());
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
      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, nullptr));
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, sockets.data()));

      uint32_t total_gpus = 0;
      for (auto& socket : sockets) {
        uint32_t num_gpus_per_socket = 0;
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &num_gpus_per_socket, nullptr));
        std::vector<amdsmi_processor_handle> processor_handles(num_gpus_per_socket);
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &num_gpus_per_socket, processor_handles.data()));
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
  uint64_t id;
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
      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, nullptr));
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, sockets.data()));

      for (auto& socket : sockets) {
        uint32_t processor_handle_count = 0;
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &processor_handle_count, nullptr));
        std::vector<amdsmi_processor_handle> processor_handles(processor_handle_count);
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &processor_handle_count, processor_handles.data()));

        // this does not work?
        // AMDSMICHECK(amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_GPU, nullptr, &num_gpus_per_socket));

        // workaround
        for (auto& proc : processor_handles) {
          processor_type_t type;
          uint64_t id;

          AMDSMICHECK(amdsmi_get_processor_type(proc, &type));
          if(type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
            amdsmi_enumeration_info_t info;
            AMDSMICHECK(amdsmi_get_gpu_enumeration_info(proc, &info));
            if(info.hip_id == deviceIndex) {
              AMDSMICHECK(amdsmi_get_gpu_bdf_id(proc, &id));
              break;
            }
          }
        }
      }
    } else {
      ARSMICHECK(ARSMI_dev_pci_id_get(deviceIndex, &id));
    }

    // rocm-smi/amd-smi format
    //snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id & 0xffffffff) >> 32, (id & 0xff00) >> 8, (id & 0xf8) >> 3, (id & 0x7));

    // borrowing NCCL's format from utils.cc:int64ToBusId
    // !! To be reconciled after discussion with amdsmi team !!
    snprintf(busId, len, "%04lx:%02lx:%02lx.%01lx", (id) >> 20, (id & 0xff000) >> 12, (id & 0xff0) >> 4, (id & 0xf));
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

      AMDSMICHECK(amdsmi_get_processor_handle_from_bdf(bdf, &processor_handle));
      
      processor_type_t type;
      AMDSMICHECK(amdsmi_get_processor_type(processor_handle, &type));
      if(type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
        amdsmi_enumeration_info_t info;
        AMDSMICHECK(amdsmi_get_gpu_enumeration_info(processor_handle, &info));
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
        if (bdfid == busid) break;
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
    uint64_t amdsmi_hops = 1, amdsmi_weight ;
    *count = 1;

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

      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, nullptr));
      std::vector<amdsmi_socket_handle> sockets(socket_count);
      AMDSMICHECK(amdsmi_get_socket_handles(&socket_count, sockets.data()));

      for (auto& socket : sockets) {
        uint32_t processor_handle_count = 0;
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &processor_handle_count, nullptr));
        std::vector<amdsmi_processor_handle> processor_handles(processor_handle_count);
        AMDSMICHECK(amdsmi_get_processor_handles(socket, &processor_handle_count, processor_handles.data()));

        // this does not work?
        // AMDSMICHECK(amdsmi_get_processor_handles_by_type(socket, AMDSMI_PROCESSOR_TYPE_AMD_GPU, nullptr, &num_gpus_per_socket));

        // workaround
        for (auto& proc : processor_handles) {
          processor_type_t proc_type;
          AMDSMICHECK(amdsmi_get_processor_type(proc, &proc_type));
          if(proc_type == AMDSMI_PROCESSOR_TYPE_AMD_GPU) {
            amdsmi_enumeration_info_t info;
            AMDSMICHECK(amdsmi_get_gpu_enumeration_info(proc, &info));
            if(info.hip_id == srcIndex) {
              src_processor_handle = proc;
              found_src = true;
            }
            if(info.hip_id == dstIndex) {
              dst_processor_handle = proc;
              found_dst = true;
            }
          }
        }
      }
      if (!found_src) ERROR("amd-smi could not find processor handle for srcIndex: %d", srcIndex);
      if (!found_dst) ERROR("amd-smi could not find processor handle for dstIndex: %d", dstIndex);
      AMDSMICHECK(amdsmi_topo_get_link_type(src_processor_handle, dst_processor_handle, &amdsmi_hops, &amdsmi_type));
      AMDSMICHECK(amdsmi_topo_get_link_weight(src_processor_handle, dst_processor_handle, &amdsmi_weight));

      // amd-smi reports weight=0 for XGMI ??
      if (amdsmi_type == AMDSMI_LINK_TYPE_XGMI) {
        uint64_t min_bw = 0, max_bw = 0;
        AMDSMICHECK(amdsmi_get_minmax_bandwidth_between_processors(src_processor_handle, dst_processor_handle, &min_bw, &max_bw));
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
