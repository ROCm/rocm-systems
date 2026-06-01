## Copyright (c) Advanced Micro Devices, Inc.
## SPDX-License-Identifier:  MIT

if(NOT TARGET asan_build_dependencies)
    add_library(asan_build_dependencies INTERFACE)
endif()

if(NOT WIN32 AND ENABLE_SANITIZER MATCHES "ASAN")
    set(THREADS_PREFER_PTHREAD_FLAG ON)
    find_package(Threads REQUIRED)
    target_link_libraries(asan_build_dependencies INTERFACE Threads::Threads)
    # ASAN with -shared-libsan requires explicit pthread linkage (rocm-systems#6455)
    target_link_options(asan_build_dependencies INTERFACE -lpthread)
endif()
