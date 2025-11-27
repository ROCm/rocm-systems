#include "cuid_device.h"
#include "cuid_util.h"
#include "cuid_cpu.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
#include "cuid_file.h"
#include <dirent.h>
#include <sys/types.h>
#include <unistd.h>
#include <vector>
#include <mutex>
#include <cstring>
#include <fstream>
#include <sstream>
#include <iostream>
#include <openssl/sha.h>

amdcuid_status_t AmdCuidDevice::get_secondary_cuid(amdcuid& id, AMDCUID_HMAC * hmac) const {
    //attempt to find the secondary CUID in file first
    std::string cuid_file_path = "/tmp/cuid";
    CuidFile secondary_file(cuid_file_path, false);
    secondary_file.load();
    std::vector<CuidFileEntry> entries = secondary_file.get_entries();

    amdcuid_device_type_t type = this->type();
    // there's only 1 platform entry, so handle that case first
    if (type == AMDCUID_DEVICE_TYPE_PLATFORM)
    {
        // for platform, just return the first entry found
        CuidFileEntry entry;
        amdcuid_status_t status = secondary_file.find_by_device_type(AMDCUID_DEVICE_TYPE_PLATFORM, entry);
        if (status == AMDCUID_STATUS_SUCCESS) {
            id = entry.secondary_cuid;
            return AMDCUID_STATUS_SUCCESS;
        }
    }
    for (auto& entry : entries) {
        switch (type)
        {
            case AMDCUID_DEVICE_TYPE_GPU:
                // search by render node
                {
                    auto gpu = reinterpret_cast<AmdCuidGpu*>(const_cast<AmdCuidDevice*>(this));
                    if (gpu) {
                        const auto& info = gpu->get_info();
                        if (entry.device_node == info.render_node) {
                            id = entry.secondary_cuid;
                            return AMDCUID_STATUS_SUCCESS;
                        }
                    }
                }
                break;
            case AMDCUID_DEVICE_TYPE_CPU:
                // search by package_core_id
                {
                    auto cpu = reinterpret_cast<AmdCuidCpu*>(const_cast<AmdCuidDevice*>(this));
                    if (cpu) {
                        const auto& info = cpu->get_info();
                        std::string core_id = std::to_string(info.header.fields.cpu.physical_id) + 
                                          ":" + std::to_string(info.header.fields.cpu.core);
                        if (entry.package_core_id == core_id) {
                            id = entry.secondary_cuid;
                            return AMDCUID_STATUS_SUCCESS;
                        }
                    }
                }
                break;
            case AMDCUID_DEVICE_TYPE_NIC:
                // search by device node
                {
                    auto nic = reinterpret_cast<AmdCuidNic*>(const_cast<AmdCuidDevice*>(this));
                    if (nic) {
                        const auto& info = nic->get_info();
                        if (entry.device_node == info.network_interface) {
                            id = entry.secondary_cuid;
                            return AMDCUID_STATUS_SUCCESS;
                        }
                    }
                }
                break;
            default:
                break;
            // Will expand with different devices as we implement them
        }
    }

    // if not found, generate secondary CUID
    if (!hmac) {
        // if we must generate the secondary CUID, then require HMAC
        return AMDCUID_STATUS_INVALID_ARGUMENT;
    }
    if (geteuid() != 0)
    {
        return AMDCUID_STATUS_PERMISSION_DENIED;
    }
    amdcuid primary;
    amdcuid_status_t status = get_primary_cuid(primary);
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }

    status = AmdCuidUtilities::generate_secondary_cuid(&primary, &id, hmac);
    return status;
}
