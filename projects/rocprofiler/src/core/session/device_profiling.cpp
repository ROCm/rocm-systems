#include "device_profiling.h"
// #include "src/utils/debug.h"

#include <iostream>
#include <sched.h>
#include <atomic>
#include <vector>

#include "src/utils/exception.h"
#include "src/core/hsa/queues/queue.h"
// #include "src/core/counters/rdc/rdc_metrics.h"


#include <exception>
#include <typeinfo>
#include <stdexcept>

#define QUEUE_NUM_PACKETS 64
static const size_t CMD_SLOT_SIZE_B = 0x40;

using namespace rocprofiler;

typedef std::vector<hsa_ven_amd_aqlprofile_info_data_t> pmc_callback_data_t;


static std::atomic<uint64_t> SESSION_COUNTER{1};

uint64_t GenerateUniqueSessionId() {
  return SESSION_COUNTER.fetch_add(1, std::memory_order_release);
}

struct devices_t {
  std::vector<hsa_agent_t> cpu_devices;
  std::vector<hsa_agent_t> gpu_devices;
  std::vector<hsa_agent_t> other_devices;
};


bool createHsaQueue(hsa_queue_t** queue, hsa_agent_t gpu_agent) {
  // create a single-producer queue
  // TODO: check if API args are correct, especially UINT32_MAX
  hsa_status_t status;
  status = hsa_queue_create(gpu_agent, QUEUE_NUM_PACKETS, HSA_QUEUE_TYPE_SINGLE, NULL, NULL,
                            UINT32_MAX, UINT32_MAX, queue);
  if (status != HSA_STATUS_SUCCESS) fatal("Queue creation failed");

  status = hsa_amd_queue_set_priority(*queue, HSA_AMD_QUEUE_PRIORITY_HIGH);
  if (status != HSA_STATUS_SUCCESS) warning("Device Profiling HSA Queue Priority Set Failed");

  return (status == HSA_STATUS_SUCCESS);
}


uint64_t submitPacket(hsa_queue_t* queue, const void* packet) {
  const uint32_t slot_size_b = CMD_SLOT_SIZE_B;

  // advance command queue
  const uint64_t write_idx = hsa_queue_add_write_index_scacq_screl(queue, 1);
  while ((write_idx - hsa_queue_load_read_index_relaxed(queue)) >= queue->size) {
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
  hsa_signal_store_relaxed(queue->doorbell_signal, write_idx);

  return write_idx;
}

// Wait signal
hsa_signal_value_t signalWait(const hsa_signal_t& signal, const hsa_signal_value_t& signal_value) {
  const hsa_signal_value_t exp_value = signal_value - 1;
  hsa_signal_value_t ret_value = signal_value;
  while (1) {
    // TODO: The 4th argument mentioning timeout is current set to UINT64_MAX.
    // Probably a maximum wait time should be set. We don't want application to hang because of
    // unlimited wait.
    // TODO2 : try 500000 assuming nanosecond granularity -- must be verified.
    ret_value = hsa_signal_wait_scacquire(signal, HSA_SIGNAL_CONDITION_LT, signal_value, UINT64_MAX,
                                          HSA_WAIT_STATE_BLOCKED);
    if (ret_value == exp_value) break;
    if (ret_value != signal_value)
      fatal("Error: signalWait: signal_value(%lu), ret_value(%lu)", signal_value, ret_value);
  }
  return ret_value;
}

bool DeviceProfileSession::generatePackets() {
  // char gpu_name[64];
  // hsa_agent_get_info(gpu_agent_, HSA_AGENT_INFO_NAME, gpu_name);

  // Get the PM4 Packets
  // TODO: The below function is wasteful. Doesn't do resource cleanup.
  // write a function that is specific to the needs of this class

  /*
  profiles_ = Packet::initializeAqlPackets(
      cpu_agent_, gpu_agent_, gpu_name, profiling_data_,
      profiling_data_.size());

  if(profiles_->size() > 1)
      std::cout<<"Multiple profiles present!\n";

  profile_ = (*profiles_)[0].second;
  start_packet_ = *(*profiles_)[0].first->start_packet;
  stop_packet_  = *(*profiles_)[0].first->stop_packet;
  read_packet_  = *(*profiles_)[0].first->read_packet;

  counter_map_ = *(*profiles_)[0].first->counter_map; */

  std::map<std::pair<uint32_t, uint32_t>, uint64_t> events_max_block_counters;
  std::map<std::string, std::set<std::string>> metrics_counters;


  metrics::ExtractMetricEvents(profiling_data_, gpu_agent_, metrics_dict_, results_map_,
                               events_list_, results_list_, events_max_block_counters,
                               metrics_counters);

  profile_ = Packet::InitializeDeviceProfilingAqlPackets(cpu_agent_, gpu_agent_, &events_list_[0],
                                                         events_list_.size(), &start_packet_,
                                                         &stop_packet_, &read_packet_);

  start_packet_.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

  start_packet_.completion_signal = {};


  read_packet_.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

  read_packet_.completion_signal = {};

  stop_packet_.header = HSA_PACKET_TYPE_VENDOR_SPECIFIC << HSA_PACKET_HEADER_TYPE;

  stop_packet_.completion_signal = {};

  return true;
}

bool DeviceProfileSession::createQueue() {
  // Ensuring there is only one queue per device
  hsa_queue_t* queue = DeviceProfileSession::getQueue(gpu_agent_);
  if (queue != nullptr) return true;
  if (::createHsaQueue(&queue, gpu_agent_) == false) return false;

  std::lock_guard<std::mutex> lock(agent_queue_map_mutex_);
  agent_queue_map_.insert(std::make_pair(gpu_agent_.handle, queue));
  return true;
}

hsa_queue_t* DeviceProfileSession::getQueue(hsa_agent_t gpu_agent) {
  std::lock_guard<std::mutex> lock(agent_queue_map_mutex_);
  auto it = agent_queue_map_.find(gpu_agent.handle);
  return (it != agent_queue_map_.end()) ? it->second : nullptr;
}

DeviceProfileSession::DeviceProfileSession(std::vector<std::string> profiling_data,
                                           hsa_agent_t cpu_agent, hsa_agent_t gpu_agent,
                                           uint64_t* session_id)
    : profiling_data_(profiling_data), cpu_agent_(cpu_agent), gpu_agent_(gpu_agent) {
  session_id_ = GenerateUniqueSessionId();
  *session_id = session_id_;

  // initialize packets struct
  start_packet_ = {};
  stop_packet_ = {};
  read_packet_ = {};

  profile_ = NULL;

  char gpu_name[64];
  if (hsa_agent_get_info(gpu_agent_, HSA_AGENT_INFO_NAME, gpu_name) != HSA_STATUS_SUCCESS)
    fatal("Agent name query failed");

  HSAAgentInfo agentInfo = (HSASupport_Singleton::GetInstance().GetHSAAgentInfo(gpu_agent_.handle));
  metrics_dict_ = MetricsDict::Create(&agentInfo);

  for (auto& d : profiling_data_) {
    Metric* metric = const_cast<Metric*>(metrics_dict_->Get(d));
    if (metric == NULL) std::cout << d << " not found in metrics_dict\n";
    metrics_list_.push_back(metric);
  }

  createQueue();
  generatePackets();

  // create signals
  hsa_status_t status = hsa_signal_create(1, 0, NULL, &start_signal_);
  if (status != HSA_STATUS_SUCCESS) fatal("start signal creation failed");

  status = hsa_signal_create(1, 0, NULL, &completion_signal_);
  if (status != HSA_STATUS_SUCCESS) fatal("completion signal creation failed");

  status = hsa_signal_create(1, 0, NULL, &stop_signal_);
  if (status != HSA_STATUS_SUCCESS) fatal("stop signal creation failed");
}

DeviceProfileSession::~DeviceProfileSession() {
  // TODO:
  //  delete queue
  //  delete signals
  //  free command buffer/output buffer
}

void DeviceProfileSession::StartSession() {
  // TODO: check if session was already started. Don't allow start twice

  // Set completion signal
  start_packet_.completion_signal = start_signal_;

  // Place the "Start" packet in the Queue
  submitPacket(DeviceProfileSession::getQueue(gpu_agent_), &start_packet_);

  // Wait for the completion signal of the packet
  ::signalWait(start_packet_.completion_signal, 1);

  // restore signal to a value of 1
  hsa_signal_store_screlease(start_signal_, 1);

  // set a variable that this session has started
}

void DeviceProfileSession::PollMetrics(rocprofiler_device_profile_metric_t* data) {
  // TODO: check if session was already started
  // TODO: can't poll if stopped
  // Reset the completion signal value for read packet
  // TODO: clear profile output buffer

  // Set completion signal
  read_packet_.completion_signal = completion_signal_;

  // Place the "Read" packet in the Queue
  ::submitPacket(DeviceProfileSession::getQueue(gpu_agent_), &read_packet_);

  // Wait for the completion signal of the packet
  ::signalWait(read_packet_.completion_signal, 1);

  // Collect counter values for events
  metrics::GetCounterData(profile_, gpu_agent_, results_list_);

  // evaluate metrics based on collected counter values
  metrics::GetMetricsData(results_map_, metrics_list_);

  for (size_t i = 0; i < profiling_data_.size(); i++) {
    auto it = results_map_.find(profiling_data_[i]);
    if (it != results_map_.end()) {
      strcpy(data[i].metric_name, it->first.c_str());
      data[i].value.value = it->second->val_double;
    }
  }

  // restore signal to a value of 1
  hsa_signal_store_screlease(completion_signal_, 1);
}

void DeviceProfileSession::StopSession() {
  // TODO: check if session was already started

  // Set completion signal
  stop_packet_.completion_signal = stop_signal_;

  // Place the "Stop" packet in the Queue
  submitPacket(DeviceProfileSession::getQueue(gpu_agent_), &stop_packet_);

  // Wait for the completion signal of the packet
  // What is the correct value to wait for?
  signalWait(stop_packet_.completion_signal, 1);

  // restore signal to a value of 1
  hsa_signal_store_screlease(stop_signal_, 1);
}

hsa_status_t device_cb(hsa_agent_t agent, void* data) {
  hsa_device_type_t device_type;
  devices_t* devices = reinterpret_cast<devices_t*>(data);
  if (hsa_agent_get_info(agent, HSA_AGENT_INFO_DEVICE, &device_type) != HSA_STATUS_SUCCESS)
    fatal("hsa_agent_get_info failed");
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
  // Enumerate the agents.
  if (hsa_iterate_agents(device_cb, device_list) != HSA_STATUS_SUCCESS)
    fatal("hsa_iterate_agents failed");
}


bool rocprofiler::find_hsa_agent_cpu(uint64_t index, hsa_agent_t* agent) {
  devices_t device_list;
  get_hsa_agents_list(&device_list);

  if (index > device_list.cpu_devices.size()) return false;

  *agent = device_list.cpu_devices[index];
  return true;
}

bool rocprofiler::find_hsa_agent_gpu(uint64_t index, hsa_agent_t* agent) {
  devices_t device_list;
  get_hsa_agents_list(&device_list);

  if (index > device_list.gpu_devices.size()) return false;

  *agent = device_list.gpu_devices[index];
  return true;
}

std::map<uint64_t, hsa_queue_t*> DeviceProfileSession::agent_queue_map_;
std::mutex DeviceProfileSession::agent_queue_map_mutex_;