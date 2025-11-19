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

amdcuid_status_t AmdCuidDevice::get_secondary_cuid(amdcuid& id) const {
    (void)id;
    return AMDCUID_STATUS_UNSUPPORTED;
}
