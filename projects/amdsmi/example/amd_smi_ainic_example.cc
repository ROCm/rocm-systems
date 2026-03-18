#include <amd_smi/amdsmi.h>

#include <cstdlib>
#include <iostream>
#include <vector>

// Show all RDMA key-value pairs for one processor handle.
void show_data_for_one_handle(amdsmi_processor_handle processor_handle,
                              amdsmi_nic_rdma_devices_info_t& info);

// Show all RDMA stats.
void show_stats() {
  uint32_t soc_count = 0;

  // Call amdsmi_get_socket_handles with second parameter (socket_handles)
  // nullptr to get the number of socket handles.
  amdsmi_status_t status = amdsmi_get_socket_handles(&soc_count, nullptr);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_get_socket_handles failed with status " << (int)status << std::endl;
    exit(1);
  }

  if (soc_count == 0)  // Nothing to do.
    return;

  // Reserve a vector for soc_count socket handles.
  std::vector<amdsmi_socket_handle> sockets(soc_count);

  // Get the socket handles.
  status = amdsmi_get_socket_handles(&soc_count, sockets.data());
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_get_socket_handles failed with status " << (int)status << std::endl;
    exit(1);
  }

  // Iterate through all socket handles to find all AI NIC processor
  // handles and update the statistics for each of them.
  for (uint32_t index = 0; index < soc_count; index++) {
    uint32_t processor_count = 0;
    status = amdsmi_get_processor_handles_by_type(sockets[index], AMDSMI_PROCESSOR_TYPE_AMD_NIC,
                                                  nullptr, &processor_count);
    if (status != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "amdsmi_get_processor_handles_by_type failed with status " << (int)status
                << std::endl;
      continue;
    }

    // Reserve a vector for processor_count processor handles.
    std::vector<amdsmi_processor_handle> processor_handles(processor_count);
    status = amdsmi_get_processor_handles_by_type(sockets[index], AMDSMI_PROCESSOR_TYPE_AMD_NIC,
                                                  processor_handles.data(), &processor_count);

    if (status != AMDSMI_STATUS_SUCCESS) {
      std::cerr << "amdsmi_get_processor_handles_by_type failed with status " << (int)status
                << std::endl;
      continue;
    }

    for (uint32_t idx = 0; idx < processor_count; ++idx) {
      amdsmi_status_t status;
      amdsmi_nic_rdma_devices_info_t info;
      status = amdsmi_get_nic_rdma_dev_info(processor_handles[idx], &info);
      if (status != AMDSMI_STATUS_SUCCESS) continue;

      // Show info and stats.
      show_data_for_one_handle(processor_handles[idx], info);
    }
  }
}

void show_data_for_one_handle(amdsmi_processor_handle processor_handle,
                              amdsmi_nic_rdma_devices_info_t& info) {
  for (uint8_t rdma_dev_idx = 0; rdma_dev_idx < info.num_rdma_dev; ++rdma_dev_idx) {
    amdsmi_nic_rdma_dev_info_t dev_info = info.rdma_dev_info[rdma_dev_idx];
    for (uint8_t rdma_port_idx = 0; rdma_port_idx < dev_info.num_rdma_ports; ++rdma_port_idx) {
      amdsmi_nic_rdma_port_info_t port_info = dev_info.rdma_port_info[rdma_port_idx];

      // Call *_statistics the first time to get the number of statistics.
      uint32_t num_stats = 0;
      amdsmi_status_t status;

      status =
          amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_idx, &num_stats, nullptr);
      if (status != AMDSMI_STATUS_SUCCESS) continue;

      // Reserve a vector for stats.
      std::vector<amdsmi_nic_stat_t> stats(num_stats);

      // Call *_statistics the second time to get the statistics.
      status = amdsmi_get_nic_rdma_port_statistics(processor_handle, rdma_port_idx, &num_stats,
                                                   stats.data());
      if (status != AMDSMI_STATUS_SUCCESS) continue;

      std::cout << "name: " << dev_info.rdma_dev << ", netdev: " << port_info.netdev << std::endl;

      // Get all stats and show them.
      for (uint32_t stat_idx{}; stat_idx < num_stats; ++stat_idx) {
        std::cout << "[" << stats[stat_idx].name << ": " << stats[stat_idx].value << "]"
                  << std::endl;
      }
    }
  }
}

int main() {
  amdsmi_status_t status;

  status = amdsmi_init(AMDSMI_INIT_AMD_NICS);
  if (status != AMDSMI_STATUS_SUCCESS) {
    std::cerr << "amdsmi_init failed: " << status << std::endl;
    exit(1);
  }

  std::cout << "amd-smi initialized" << std::endl;

  show_stats();

  amdsmi_shut_down();
  std::cout << "amd-smi shut down" << std::endl;
  return 0;
}
