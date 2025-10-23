#include "cuid_device.h"
#include "cuid_util.h"
#include "cuid_cpu.h"
#include "cuid_gpu.h"
#include "cuid_nic.h"
#include "cuid_platform.h"
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


// TODO: If the primary cuid cannot be obtained directly, then using the CUID files.
amdcuid_status_t CuidDevice::get_secondary_cuid(amdcuid_salt_t salt, amdcuid& id) const {
    amdcuid primary;
    amdcuid_status_t status = get_primary_cuid(primary);
    if (status != AMDCUID_STATUS_SUCCESS) {
        return status;
    }
    id = CuidUtilities::get_secondary_cuid(salt, &primary);
    return AMDCUID_STATUS_SUCCESS;
}
