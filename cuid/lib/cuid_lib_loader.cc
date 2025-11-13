#include "cuid_lib_loader.h"
#include "cuid.h"
#include <iostream>

CuidLibLoader::CuidLibLoader(): libHandler_(nullptr) {
}

amdcuid_status_t CuidLibLoader::load(const char* filename) {
    if (filename == nullptr) {
        return AMDCUID_STATUS_FILE_NOT_FOUND;
    }
    if (libHandler_ || library_loaded_) {
        unload();
    }

    std::lock_guard<std::mutex> guard(library_mutex_);
    // check if already loaded, return success if it is
    // dlopen(filename, RTLD_NOLOAD) == null only IFF library is not loaded
    void* isLibOpen = dlopen(filename, RTLD_NOLOAD);
    if (isLibOpen == nullptr) {
      libHandler_ = dlopen(filename, RTLD_LAZY);
      if (!libHandler_) {
          char* error = dlerror();
          std::cerr << "Fail to open " << filename <<": " << error
                    << std::endl;
          return AMDCUID_STATUS_UNSUPPORTED;
      }
    }
    library_loaded_ = true;

    return AMDCUID_STATUS_SUCCESS;
}

amdcuid_status_t CuidLibLoader::unload() {
        std::lock_guard<std::mutex> guard(library_mutex_);
        if (libHandler_) {
            dlclose(libHandler_);
            libHandler_ = nullptr;
            library_loaded_ = false;
        }
        return AMDCUID_STATUS_SUCCESS;
}

CuidLibLoader::~CuidLibLoader() {
        unload();
}