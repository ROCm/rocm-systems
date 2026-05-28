#include <amd_smi/amdsmi.h>

#include <iomanip>
#include <iostream>
#include <ostream>
#include <sstream>
#include <string_view>
#include <tuple>
#include <vector>

#define CHK_AMDSMI_RET(RET)                                                                \
  {                                                                                        \
    if (RET != AMDSMI_STATUS_SUCCESS) {                                                    \
      const char* err_str;                                                                 \
      std::cout << "AMDSMI call returned " << RET << " at line " << __LINE__ << std::endl; \
      amdsmi_status_code_to_string(RET, &err_str);                                         \
      std::cout << err_str << std::endl;                                                   \
      return RET;                                                                          \
    }                                                                                      \
  }

const char* category_to_string(int category) {
  switch (category) {
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE:
      return "UALOE";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_SWITCH:
      return "SWITCH";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_CRYPTO:
      return "CRYPTO";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_PFC:
      return "PFC";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_NETPORT:
      return "NETPORT";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_UALOE:
      return "DERIVED_UALOE";
    case AMDSMI_FABRIC_TELEMETRY_CATEGORY_DERIVED_NETPORT:
      return "DERIVED_NETPORT";
    default:
      return "UNKNOWN";
  }
}

std::string bdf_to_str(const amdsmi_bdf_t& bdf) {
  std::ostringstream outstream{};
  outstream << std::hex << std::setfill('0') << std::setw(4)
            << static_cast<std::uint32_t>(bdf.bdf.domain_number) << ":" << std::setw(2)
            << static_cast<std::uint32_t>(bdf.bdf.bus_number) << ":" << std::setw(2)
            << static_cast<std::uint32_t>(bdf.bdf.device_number) << "."
            << static_cast<std::uint32_t>(bdf.bdf.function_number);
  return outstream.str();
}

constexpr auto MAX_UUID_ELEMENTS = std::uint16_t(16);
std::string ppod_id_to_str(const std::uint8_t (&id)[MAX_UUID_ELEMENTS]) {
  std::ostringstream outstream{};
  outstream << std::hex << std::setfill('0');
  for (auto idx = std::size_t(0); idx < MAX_UUID_ELEMENTS; ++idx) {
    if ((idx == 4) || (idx == 6) || (idx == 8) || (idx == 10)) {
      outstream << '-';
    }
    outstream << std::setw(2) << static_cast<unsigned>(id[idx]);
  }
  outstream << std::dec;

  return outstream.str();
}

constexpr auto MAX_COLUMNS_PER_GRID = std::uint16_t(8);
void print_array_grid(std::ostream& outstrm, std::string_view line_prefix,
                      std::string_view line_title, const uint32_t* array_data,
                      std::size_t array_count, std::size_t columns) {
  outstrm << line_prefix << "** " << line_title << ":\n";
  for (auto idx = std::size_t(0); idx < array_count; ++idx) {
    if (idx % columns == 0) {
      outstrm << line_prefix << "\t";
    }
    outstrm << array_data[idx];
    const auto is_end_row = static_cast<bool>((idx + 1) % columns == 0);
    const auto is_last = static_cast<bool>(idx + 1 == array_count);
    if (is_end_row || is_last) {
      outstrm << '\n';
    } else {
      outstrm << ' ';
    }
  }
}

int main() {
  amdsmi_status_t ret;

  // Init amdsmi for sockets and devices.
  ret = amdsmi_init(AMDSMI_INIT_AMD_GPUS);
  CHK_AMDSMI_RET(ret);

  // Get all sockets
  uint32_t socket_count = 0;

  // Get the socket count available for the system.
  ret = amdsmi_get_socket_handles(&socket_count, nullptr);
  CHK_AMDSMI_RET(ret);

  // Allocate the memory for the sockets
  std::vector<amdsmi_socket_handle> sockets(socket_count);
  // Get the sockets of the system
  ret = amdsmi_get_socket_handles(&socket_count, &sockets[0]);
  CHK_AMDSMI_RET(ret);

  // For each socket, get identifier and devices
  for (uint32_t i = 0; i < socket_count; i++) {
    // Get Socket info
    char socket_info[128];
    ret = amdsmi_get_socket_info(sockets[i], 128, socket_info);
    CHK_AMDSMI_RET(ret);
    std::cout << "\t**Socket Info: " << socket_info << std::endl;

    // Get the device count available for the socket.
    uint32_t device_count = 0;
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, nullptr);
    CHK_AMDSMI_RET(ret);

    // Allocate the memory for the device handlers on the socket
    std::vector<amdsmi_processor_handle> processor_handles(device_count);
    // Get all devices of the socket
    ret = amdsmi_get_processor_handles(sockets[i], &device_count, &processor_handles[0]);
    CHK_AMDSMI_RET(ret);

    std::cout << "\t**Processor Count: " << device_count << std::endl;

    // For each device of the socket, get Fabric telemetry
    for (uint32_t device_index = 0; device_index < device_count; device_index++) {
      std::cout << "\t\t**Processing Fabric Device: " << device_index << std::endl;

      // Allocate storage for Fabric telemetry
      // Request all categories of telemetry

      uint32_t category_mask = AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_UALOE |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_SWITCH |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_CRYPTO |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_PFC |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_NETPORT |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_UALOE |
                               AMDSMI_FABRIC_TELEMETRY_CATEGORY_MASK_DERIVED_NETPORT;

      amdsmi_fabric_telemetry_t* telemetry = nullptr;
      ret =
          amdsmi_alloc_fabric_telemetry(processor_handles[device_index], category_mask, &telemetry);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "\t\t\tFailed to allocate Fabric telemetry storage: " << ret << std::endl;
        continue;
      }
      std::cout << "\t\t\tSuccessfully allocated Fabric telemetry storage" << std::endl;

      // Get the current telemetry data
      ret = amdsmi_get_fabric_telemetry_data(processor_handles[device_index], telemetry);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "\t\t\tFailed to get Fabric telemetry data: " << ret << std::endl;
        std::ignore = amdsmi_free_fabric_telemetry(processor_handles[device_index], telemetry);
        continue;
      } else {
        std::cout << "\t\t\tSuccessfully retrieved Fabric telemetry data" << std::endl;

        // Display telemetry information
        for (int cat = AMDSMI_FABRIC_TELEMETRY_CATEGORY_UALOE;
             cat < static_cast<int>(AMDSMI_FABRIC_TELEMETRY_CATEGORY_MAX); cat++) {
          auto category = static_cast<amdsmi_fabric_telemetry_category_t>(cat);
          if (telemetry->datasets[cat] != nullptr) {
            auto dataset = telemetry->datasets[cat];
            std::cout << "\t\t\t\tCategory " << category_to_string(category) << ": "
                      << dataset->instance_count << " instances, "
                      << "generation " << dataset->generation_count << std::endl;

            // Display instances
            for (uint32_t inst = 0; inst < dataset->instance_count; inst++) {
              auto instance = &dataset->instances[inst];
              std::cout << "\t\t\t\t\tInstance: " << instance->name.text
                        << " (idx: " << instance->logical_idx << ", items: " << instance->item_count
                        << ")" << std::endl;

              // Display items for this instance
              for (uint32_t item_idx = 0; item_idx < instance->item_count; item_idx++) {
                auto& item = instance->items[item_idx];
                std::cout << "\t\t\t\t\t\tItem " << item_idx << ": "
                          << "ID=0x" << std::hex << item.id << std::dec << "("
                          << amdsmi_fabric_telem_id_to_string(item.id) << ")"
                          << ", Value=" << item.value << std::endl;
              }
            }
          }
        }
      }

      // Free the telemetry storage
      ret = amdsmi_free_fabric_telemetry(processor_handles[device_index], telemetry);
      if (ret != AMDSMI_STATUS_SUCCESS) {
        std::cout << "\t\t\tWarning: Failed to free Fabric telemetry storage: " << ret << std::endl;
      }
    }

    // For each device of the socket, get Fabric info
    std::cout << "\n";
    for (uint32_t device_index = 0; device_index < device_count; device_index++) {
      std::cout << "\t\t**Processing Fabric Info: " << device_index << "\n";
      // Get BDF info
      amdsmi_bdf_t bdf = {};
      ret = amdsmi_get_gpu_device_bdf(processor_handles[device_index], &bdf);
      std::cout << "\t\t\t** BDF: " << bdf_to_str(bdf) << "\n";

      // Get the fabric info (partial sysfs reads; NO_DATA if no UALoE files had content)
      amdsmi_fabric_info_t fabric_info{};
      ret = amdsmi_get_gpu_fabric_info(processor_handles[device_index], &fabric_info);
      if (ret != AMDSMI_STATUS_SUCCESS && ret != AMDSMI_STATUS_NO_DATA &&
          ret != AMDSMI_STATUS_NOT_SUPPORTED) {
        std::cout << "\t\t\tWarning: Failed to get Fabric info: " << ret << "\n";
      } else {
        if (ret == AMDSMI_STATUS_NO_DATA) {
          std::cout << "\t\t\tNote: Fabric sysfs had no usable lines (NO_DATA); "
                       "numeric fields may be sentinels."
                    << "\n";
        } else if (ret == AMDSMI_STATUS_NOT_SUPPORTED) {
          std::cout << "\t\t\tNote: Fabric info is not supported on this device."
                    << "\n";
          continue;
        }
        std::cout << "\t\t\tFabric info data: " << "\n";
        std::cout << "\t\t\t\t** BDF: " << bdf_to_str(fabric_info.bdf) << "\n";
        std::cout << "\t\t\t\t** Fabric Version: " << fabric_info.fabric_info.version << "\n";
        std::cout << "\t\t\t\t** Fabric Type: "
                  << fabric_info.fabric_info.fabric_version.v1.fabric_type << "\n";
        std::cout << "\t\t\t\t** Accelerator ID: "
                  << fabric_info.fabric_info.fabric_version.v1.accelerator_id << "\n";
        std::cout << "\t\t\t\t** Bandwidth: " << fabric_info.fabric_info.fabric_version.v1.bandwidth
                  << "\n";
        std::cout << "\t\t\t\t** Latency: " << fabric_info.fabric_info.fabric_version.v1.latency
                  << "\n";
        std::cout << "\t\t\t\t** PPOD ID: "
                  << ppod_id_to_str(fabric_info.fabric_info.fabric_version.v1.ppod_id) << "\n";
        std::cout << "\t\t\t\t** PPOD Size: " << fabric_info.fabric_info.fabric_version.v1.ppod_size
                  << "\n";
        std::cout << "\t\t\t\t** VPOD ID: " << fabric_info.fabric_info.fabric_version.v1.vpod_id
                  << "\n";
        std::cout << "\t\t\t\t** VPOD Size: " << fabric_info.fabric_info.fabric_version.v1.vpod_size
                  << "\n";
        print_array_grid(std::cout, "\t\t\t\t", "VPOD Active Accelerators",
                         fabric_info.fabric_info.fabric_version.v1.vpod_active_accelerators,
                         AMDSMI_FABRIC_ACTIVE_ACCELERATORS_BITMAP_SIZE, MAX_COLUMNS_PER_GRID);
        print_array_grid(std::cout, "\t\t\t\t", "Local Accelerators",
                         fabric_info.fabric_info.fabric_version.v1.local_accelerators,
                         AMDSMI_FABRIC_MAX_LOCAL_GPUS, MAX_COLUMNS_PER_GRID);
        std::cout << "\t\t\t\t** Address Mode: "
                  << fabric_info.fabric_info.fabric_version.v1.addr_mode << "\n";
        std::cout << "\t\t\t\t** Accelerator State: "
                  << fabric_info.fabric_info.fabric_version.v1.accel_state << "\n";
        std::cout << "\n";
      }
    }
  }
  // Shutdown amdsmi
  ret = amdsmi_shut_down();
  CHK_AMDSMI_RET(ret);

  std::cout << "Fabric telemetry collection completed successfully!" << std::endl;
  return 0;
}
