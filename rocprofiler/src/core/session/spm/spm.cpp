#include "spm.h"
#include "src/core/hsa/hsa_support.h"
#include "src/utils/helper.h"
#include "src/api/rocprofiler_singleton.h"

#include <hsa/hsa.h>

#include <stdlib.h>

#include <bitset>

#define QUEUE_NUM_PACKETS 64
static const size_t CMD_SLOT_SIZE_B = 0x40;
// #define ASSERTM(exp, msg) assert(((void)msg, exp))
#define DEST_BUFFER_MAX 4

namespace {
struct devices_t {
  std::vector<hsa_agent_t> cpu_devices;
  std::vector<hsa_agent_t> gpu_devices;
  std::vector<hsa_agent_t> other_devices;
};
typedef struct {
  uint32_t size;  // size of buffer in bytes
  uint32_t timeout;
  uint32_t len;    // len of streamed data in spm buffer
  void* addr;      // address of spm buffer
  bool data_loss;  // OUT
} spm_buffer_params_t;

typedef struct spm_data_buffer {
  char* addr;
  uint32_t buffSize;
  // spm_data_buffer(char * data, uint32_t len) :
  //    addr{data}, buffSize{len} {}
} spm_data_buffer_t;

std::queue<spm_data_buffer_t> process_queue;
// spm_buffer_params_t spm_buffer_params[3];

std::atomic<bool> is_started;
std::atomic<bool> buffer_read_flag;
std::atomic<uint32_t> spm_buffer_idx;


std::thread thread_buffer_setup;
std::thread thread_spm_data_parse;

// std::atomic<uint32_t> currIndex;
// std::atomic<uint32_t> preIndex;
std::mutex processQueueLock;
// void spmDataParse();
// void spmBufferSetup(hsa_agent_t preferredGpuNode);
// FILE* fd;

// rocprofiler_status_t setSpmDestBuffer(hsa_agent_t preferred_agent, size_t size_in_bytes,
//                                     uint32_t* timeout, uint32_t* size_copied, void* dest,
//                                     bool* is_data_loss) {
//   [[maybe_unused]] hsa_status_t status = HSA_STATUS_SUCCESS;
// #if 0
//   status = rocprofiler::hsa_support::GetAmdExtTable().hsa_amd_spm_set_dest_buffer_fn(
//       preferred_agent, size_in_bytes, timeout, size_copied, dest, is_data_loss);
//   ASSERTM(status == HSA_STATUS_SUCCESS, "ERROR: SPM set buffer failed");
// #endif
//   return ROCPROFILER_STATUS_SUCCESS;
// }

// rocprofiler_status_t SetDestBuffer(hsa_agent_t GPUNode, uint32_t size, uint32_t timeout) {
//   rocprofiler_status_t ret;
//   uint32_t idx = currIndex.load(std::memory_order_acquire);
//   if (size) {
//     // Check if user buffer in using
//     if (spm_buffer_params[idx].addr != NULL) {
//       std::cout << "Buffer in use ." << std::endl;
//       return ROCPROFILER_STATUS_ERROR;
//     }

//     spm_buffer_params[idx].addr = malloc(size);
//     if (spm_buffer_params[idx].addr == NULL) {
//       std::cout << "Malloc(size) Failed." << std::endl;
//       return ROCPROFILER_STATUS_ERROR;
//     }
//   } else {
//     spm_buffer_params[idx].addr = NULL;
//   }

//   spm_buffer_params[idx].timeout = timeout;
//   spm_buffer_params[idx].data_loss = 0;

//   ret = setSpmDestBuffer(GPUNode, spm_buffer_params[idx].size, &spm_buffer_params[idx].timeout,
//                          &spm_buffer_params[idx].len, spm_buffer_params[idx].addr,
//                          &spm_buffer_params[idx].data_loss);
//   if (ret != ROCPROFILER_STATUS_SUCCESS) {
//     std::cout << "Fail to set Dest Buf "
//               << "ret " << ret << std::endl;
//     return ROCPROFILER_STATUS_ERROR;
//   }
//   if (spm_buffer_params[idx].data_loss) std::cout << "Data Loss" << std::endl;
//   if (spm_buffer_params[idx].len) {
//     uint32_t pidx = preIndex.load(std::memory_order_acquire);
//     if (spm_buffer_params[idx].len == spm_buffer_params[pidx].size) {
//       std::cout << "Buffer completely filled with bytes" << spm_buffer_params[idx].len <<
//       std::endl; fd = fopen("SPM_rocprofiler_data.txt", "wb"); size_t retele =
//       fwrite(spm_buffer_params[pidx].addr, 1, spm_buffer_params[idx].len, fd); if (retele <= 0)
//       rocprofiler::warning("SPM Data is wrong!"); fclose(fd);
//     } else {
//       std::cout << "Buffer partially filled with %d bytes" << spm_buffer_params[idx].len
//                 << std::endl;
//     }
//     if (timeout)
//       if (spm_buffer_params[idx].timeout == timeout) std::cout << "Timeout occurred" <<
//       std::endl;
//     ret = ROCPROFILER_STATUS_SUCCESS;
//   } else {
//     std::cout << "Data collection failed" << std::endl;
//     ret = ROCPROFILER_STATUS_SUCCESS;
//   }
//   spm_buffer_params[idx].addr = NULL;
//   return ret;
// }

// void spmBufferSetup(hsa_agent_t GPUNode) {
//   rocprofiler_status_t ret;
//   if (is_started.load(std::memory_order_acquire)) {
//     uint32_t idx = currIndex.load(std::memory_order_acquire);
//     ret = SetDestBuffer(GPUNode, spm_buffer_params[idx].size, spm_buffer_params[idx].timeout);
//     if (ret != ROCPROFILER_STATUS_SUCCESS) {
//       std::cout << "Fail to set Dest Buf 2 "
//                 << "ret " << ret << std::endl;
//       return;
//     }
//     usleep(5 * 1000);
//     // Set blocking dest buff
//     currIndex.store(1, std::memory_order_release);
//     preIndex.store(0, std::memory_order_release);
//     spm_buffer_params[idx].timeout = 1000;

//     ret = SetDestBuffer(GPUNode, spm_buffer_params[idx].size, spm_buffer_params[idx].timeout);
//     if (ret != ROCPROFILER_STATUS_SUCCESS) {
//       std::cout << "Fail to set Dest Buf 1"
//                 << "ret " << ret << std::endl;
//     }
//     usleep(5 * 1000);

//     currIndex.store(0, std::memory_order_release);
//     preIndex.store(1, std::memory_order_release);
//     spm_buffer_params[idx].timeout = 80;
//   }
// }

// void AddSpmRecords(std::vector<uint16_t>& sample) {
//   // Getting timestamps
//   int index = 0;
//   int nSample = 0;
//   int se = 0;
//   uint64_t count = 0;
//   std::vector<uint64_t> timestamp_vec;
//   // Get Buffer
//   rocprofiler::Session* session =
//       rocprofiler::ROCProfiler_Singleton::GetInstance().GetSession(rocprofiler::rocmtool::GetInstance().GetCurrentSessionId());
//   rocprofiler_filter_id_t filter_id = session->GetFilterIdWithKind(ROCPROFILER_SPM_COLLECTION);
//   rocprofiler::Filter* filter = session->GetFilter(filter_id);
//   rocprofiler_buffer_id_t buffer_id = filter->GetBufferId();
//   Memory::GenericBuffer* buffer = session->GetBuffer(buffer_id);
//   // Getting timestamps

//   while (static_cast<size_t>(index) < sample.size()) {
//     int64_t timestamp = sample[index] >> 16 | sample[index + 1];
//     timestamp = timestamp >> 16 | sample[index + 2];
//     timestamp = timestamp >> 16 | sample[index + 3];
//     timestamp_vec.emplace_back(timestamp);
//     index = index + 160;
//   }

//   index = 32;

//   while (static_cast<size_t>(index) < sample.size()) {
//     se = 0;
//     rocprofiler_record_spm_t record = {};
//     record.timestamps = rocprofiler_record_header_timestamp_t{timestamp_vec[nSample]};
//     while (se < 4) {
//       count = 0;
//       while (count < 15) {
//         record.shader_engine_data[se].counters_data[count].value = sample[index];
//         record.shader_engine_data[se].counters_data[count + 15].value = sample[index + 15];

//         count++;
//       }
//       se++;
//     }
//     record.header.id = rocprofiler_record_id_t{rocprofiler::ROCProfiler_Singleton::GetInstance().GetUniqueRecordId()};
//     buffer->AddRecord(record);
//     nSample++;
//     index += 160;
//   }
// }

// void spmDataParse() {
//   std::vector<uint16_t> lines;
//   fd = fopen("SPM_rocprofiler_data.txt", "rb");
//   while (!feof(fd)) {
//     char bytes[2];
//     size_t size = fread(&bytes, 1, 2, fd);
//     if (size) {
//       uint16_t value;
//       memcpy(&value, bytes, 2);
//       lines.push_back(value);
//     }
//   }
//   AddSpmRecords(lines);
// }

hsa_status_t device_cb(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  devices_t* devices = reinterpret_cast<devices_t*>(data);
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) != HSA_STATUS_SUCCESS)
    rocprofiler::fatal("hsa_agent_get_info failed");
  switch (device_type) {
    case HSA_DEVICE_TYPE_CPU:
      devices->cpu_devices.push_back(agent);
      break;
    case HSA_DEVICE_TYPE_GPU:
      devices->gpu_devices.push_back(agent);
      break;
    default:
      devices->other_devices.push_back(agent);
      break;
  }
  return HSA_STATUS_SUCCESS;
}

void get_hsa_agents_list(devices_t* device_list) {
  hsa_status_t status;
  // Enumerate the agents.
  status = hsa_iterate_agents(device_cb, device_list);
  if (status != HSA_STATUS_SUCCESS) rocprofiler::fatal("hsa_iterate_agents failed");
}
uint64_t submitPacket(hsa_queue_t* queue, const void* packet) {
  const uint32_t slot_size_b = CMD_SLOT_SIZE_B;

  // advance command queue
  const uint64_t write_idx =
      rocprofiler::HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_queue_add_write_index_scacq_screl_fn(queue, 1);
  while ((write_idx -
          rocprofiler::HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_queue_load_read_index_relaxed_fn(queue)) >=
         queue->size) {
    sched_yield();  // TODO: remove
  }

  const uint32_t slot_idx = (uint32_t)(write_idx % queue->size);
  uint32_t* queue_slot =
      reinterpret_cast<uint32_t*>((uintptr_t)(queue->base_address) + (slot_idx * slot_size_b));
  const uint32_t* slot_data = reinterpret_cast<const uint32_t*>(packet);

  // Copy buffered commands into the queue slot.
  // Overwrite the AQL invalid header (first dword) last.
  // This prevents the slot from being read until it's fully written.
  memcpy(&queue_slot[1], &slot_data[1], slot_size_b - sizeof(uint32_t));
  std::atomic<uint32_t>* header_atomic_ptr =
      reinterpret_cast<std::atomic<uint32_t>*>(&queue_slot[0]);
  header_atomic_ptr->store(slot_data[0], std::memory_order_release);

  // ringdoor bell
  rocprofiler::HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_store_relaxed_fn(queue->doorbell_signal,
                                                                        write_idx);

  return write_idx;
}

// bool createHsaQueue(hsa_queue_t** queue, hsa_agent_t gpu_agent) {
//   // create a single-producer queue
//   // TODO: check if API args are correct, especially UINT32_MAX
//   hsa_status_t status;
//   status = rocprofiler::hsa_support::GetCoreApiTable().hsa_queue_create_fn(
//       gpu_agent, QUEUE_NUM_PACKETS, HSA_QUEUE_TYPE_SINGLE, nullptr, nullptr, UINT32_MAX,
//       UINT32_MAX, queue);

//   if (status != HSA_STATUS_SUCCESS) rocprofiler::fatal("queue creation failed");

//   return (status == HSA_STATUS_SUCCESS);
// }

hsa_signal_value_t signalWait(const hsa_signal_t& signal, const hsa_signal_value_t& signal_value) {
  const hsa_signal_value_t exp_value = signal_value - 1;
  hsa_signal_value_t ret_value = signal_value;
  while (1) {
    // TODO: The 4th argument mentioning timeout is current set to UINT64_MAX.
    // Probably a maximum wait time should be set. We don't want application to hang because of
    // unlimited wait.
    // TODO2 : try 500000 assuming nanosecond granularity -- must be verified.
    ret_value = rocprofiler::HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_wait_scacquire_fn(
        signal, HSA_SIGNAL_CONDITION_LT, signal_value, UINT64_MAX, HSA_WAIT_STATE_BLOCKED);

    if (ret_value == exp_value) break;
    if (ret_value != signal_value)
      rocprofiler::fatal("Error: signalWait: signal_value(%lu), ret_value(%lu)", signal_value,
                         ret_value);
  }
  return ret_value;
}

}  // namespace
namespace rocprofiler {


spm::SpmCounters::SpmCounters(rocprofiler_buffer_id_t buffer_id, rocprofiler_filter_id_t filter_id,
                              rocprofiler_spm_parameter_t* spmparameter,
                              rocprofiler_session_id_t session_id)
    : buffer_id_(buffer_id),
      filter_id_(filter_id),
      spmparameter_(spmparameter),
      session_id_(session_id) {
  queue_ = nullptr;
  devices_t* device_list_ = new devices_t;
  get_hsa_agents_list(device_list_);
  defaultGpuNode_ = device_list_->gpu_devices[0];
  defaultCpuNode_ = device_list_->cpu_devices[0];
  delete device_list_;

  // create signals
  hsa_status_t status =
      HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_create_fn(1, 0, NULL, &start_signal_);
  if (status != HSA_STATUS_SUCCESS) fatal("start signal creation failed");
  status = HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_signal_create_fn(1, 0, NULL, &stop_signal_);
  if (status != HSA_STATUS_SUCCESS) fatal("start signal creation failed");
  is_started.store(false, std::memory_order_relaxed);
  buffer_read_flag.store(false, std::memory_order_relaxed);
  spm_buffer_idx.store(false, std::memory_order_relaxed);
}

rocprofiler_status_t spm::SpmCounters::startSpm() {
  if (spmparameter_->gpu_agent_id != NULL)
    preferredGpuNode_.handle = spmparameter_->gpu_agent_id->handle;
  else
    // else choose the default node to collect SPM
    preferredGpuNode_ = defaultGpuNode_;
    // hsa_agent_t preferred_cpu_agent = defaultCpuNode_;
    // int counter_count = spmparameter_->counters_count;
    // Packet::packet_t start_packet;
#if 0
  hsa_status_t hsa_status = hsa_support::GetAmdExtTable().hsa_amd_spm_acquire_fn(preferredGpuNode_);
  if (hsa_status == HSA_STATUS_SUCCESS) {
    if (!createHsaQueue(&queue_, preferredGpuNode_))
      std::cout << "Create queue is failed" << std::endl;
    agent_queue_map_.insert(std::make_pair(preferredGpuNode_.handle, queue_));
    // Generate the start and stop packets
    char gpu_name[64];
    hsa_agent_get_info(preferredGpuNode_, HSA_AGENT_INFO_NAME, &gpu_name);
    std::vector<std::string> counter_names;
    for (int i = 0; i < counter_count; i++) {
      counter_names.push_back(std::string(spmparameter_->counters_names[i]));
    }
    profiles_ =
        Packet::InitializeAqlPackets(preferred_cpu_agent, preferredGpuNode_, counter_names, true);
    // Submit the start packet
    start_packet = *(*profiles_)[0].first->start_packet;
    start_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
    start_packet.completion_signal = {};
    start_packet.completion_signal = start_signal_;
    submitPacket(queue_, &start_packet);
    signalWait(start_packet.completion_signal, 1);
    // restore signal to a value of 1
    hsa_signal_store_screlease(start_signal_, 1);
    is_started.exchange(true, std::memory_order_release);
    uint32_t timeout = 10000;
    const uint32_t spm_buffer_size = 0x2000000;
    spm_buffer_params[0].size = spm_buffer_size;
    spm_buffer_params[0].timeout = timeout;
    spm_buffer_params[0].len = 0;
    spm_buffer_params[0].addr = nullptr;
    spm_buffer_params[0].data_loss = false;
    spm_buffer_params[1].size = spm_buffer_size;
    spm_buffer_params[1].timeout = timeout;
    spm_buffer_params[1].len = 0;
    spm_buffer_params[1].addr = nullptr;
    spm_buffer_params[1].data_loss = false;
    spm_buffer_params[2].size = spm_buffer_size;
    spm_buffer_params[2].timeout = timeout;
    spm_buffer_params[2].len = 0;
    spm_buffer_params[2].addr = malloc(spm_buffer_size);
    spm_buffer_params[2].data_loss = false;
    setSpmDestBuffer(preferredGpuNode_, spm_buffer_params[2].size, &timeout,
                     &spm_buffer_params[2].len, &spm_buffer_params[2].addr,
                     &spm_buffer_params[2].data_loss);
    currIndex.store(0, std::memory_order_release);
    preIndex.store(0, std::memory_order_release);
    // thread_buffer_setup = std::thread(spmBufferSetup, preferredGpuNode_);
    // thread_spm_data_parse = std::thread(spmDataParse);
    spmBufferSetup(preferredGpuNode_);
    spmDataParse();
    return ROCPROFILER_STATUS_SUCCESS;
  } else {
    std::cout << "SPM acquire failed\n" << std::endl;
    return ROCPROFILER_STATUS_ERROR;
  }
#endif
  return ROCPROFILER_STATUS_SUCCESS;  // delete this line with if 0
}

rocprofiler_status_t spm::SpmCounters::stopSpm() {
  Packet::packet_t stop_packet;
  // submit the start packet
  is_started.exchange(false, std::memory_order_release);
  // thread_buffer_setup.join();

  // thread_spm_data_parse.join();
  stop_packet = *(*profiles_)[0].first->stop_packet;
  buffer_read_flag.exchange(false, std::memory_order_release);
  stop_packet.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;
  stop_packet.completion_signal = {};
  stop_packet.completion_signal = stop_signal_;
  submitPacket(queue_, &stop_packet);
  signalWait(stop_packet.completion_signal, 1);
  // restore signal to a value of 1
  hsa_signal_store_screlease(stop_signal_, 1);
  hsa_status_t status = HSA_STATUS_SUCCESS;
  if (queue_ != nullptr) {
    status = HSASupport_Singleton::GetInstance().GetCoreApiTable().hsa_queue_destroy_fn(queue_);
    queue_ = nullptr;
  }
  if (status != HSA_STATUS_SUCCESS) rocprofiler::warning("Queue destroy failed");
  return ROCPROFILER_STATUS_SUCCESS;
}


}  // namespace rocprofiler
