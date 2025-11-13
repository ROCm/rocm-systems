#ifndef CUID_LIB_LOADER_H
#define CUID_LIB_LOADER_H

#include "cuid.h"
#include <dlfcn.h>
#include <cstring>
#include <iostream>
#include <mutex>

class CuidLibLoader {
    public:
        CuidLibLoader();

        amdcuid_status_t load(const char* filename);

        template<typename T> amdcuid_status_t load_function(T* func_handler,
            const char* func_name);
        template<typename T> amdcuid_status_t load_object(T* obj_handler,
            const char* obj_name);


     amdcuid_status_t unload();

     ~CuidLibLoader();

    private:
        void* libHandler_;
        std::mutex library_mutex_;
        bool library_loaded_ = false;
};

template<typename T> amdcuid_status_t CuidLibLoader::load_function(
            T* func_handler,
            const char* func_name) {
    if (!libHandler_) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    if (!func_handler || !func_name) {
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }

    std::lock_guard<std::mutex> guard(library_mutex_);

    *reinterpret_cast<void**>(func_handler) =
            dlsym(libHandler_, func_name);
    if (func_handler == nullptr) {
        char* error = dlerror();
        std::cout << "CuidLibLoader: Fail to load the symbol "
                    << func_name << ": " << error << std::endl;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
    }

    return AMDCUID_STATUS_SUCCESS;
}

template<typename T> amdcuid_status_t CuidLibLoader::load_object(
            T* obj_handler,
            const char* obj_name) {
    if (!libHandler_) {
        return AMDCUID_STATUS_DEVICE_NOT_FOUND;
    }

    if (!obj_handler || !obj_name) {
        return AMDCUID_STATUS_WRONG_DEVICE_TYPE;
    }

    std::lock_guard<std::mutex> guard(library_mutex_);

    obj_handler =
            reinterpret_cast<T *>(dlsym(libHandler_, obj_name));
    if (obj_handler == nullptr) {
        char* error = dlerror();
        std::cout << "CuidLibLoader: Fail to load the symbol "
                    << obj_name << ": " << error << std::endl;
        return AMDCUID_STATUS_INSUFFICIENT_SIZE;
    }

    return AMDCUID_STATUS_SUCCESS;
}

#endif // CUID_LIB_LOADER_H
