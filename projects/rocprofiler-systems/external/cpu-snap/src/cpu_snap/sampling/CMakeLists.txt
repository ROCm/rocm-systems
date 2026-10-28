if(NOT TARGET LibDw::LibDw)
    add_library(LibDw::LibDw INTERFACE IMPORTED)
    set_target_properties(LibDw::LibDw PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${LIBDW_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES       "${LIBDW_LIBRARY}")
endif()

add_library(sampling STATIC
    dl_module_watcher.cpp
    sampler.cpp
    fork_registry.cpp
    sampler_plan.cpp
    streaming_pipeline.cpp
    thread_tracker.cpp
    triggers/perf/backend.cpp
    triggers/signal/backend.cpp
    unwind/dwarf_unwinder.cpp
    unwind/fp_copy_unwinder.cpp)
add_library(cpu_snap::sampling ALIAS sampling)

target_include_directories(sampling PUBLIC
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/src>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# rt: timer_create (signal backend) lives in librt on glibc < 2.34.
target_link_libraries(sampling PUBLIC
    cpu_snap::core
    LibDw::LibDw
    rt)

target_compile_features(sampling PUBLIC cxx_std_20)
# sampler.cpp / signal_backend.cpp bodies use glibc features
# (pthread_setname_np, SIGEV_THREAD_ID + sigev._sigev_un._tid, sigaltstack,
# ucontext gregs / REG_RIP) gated behind _GNU_SOURCE. PRIVATE because the
# public headers no longer require the macro for their declarations.
target_compile_definitions(sampling PRIVATE _GNU_SOURCE)
target_compile_options(sampling PRIVATE
    -Wall -Wextra -Wpedantic -fno-omit-frame-pointer)
# GCC 13 raises -Wmaybe-uninitialized false positives inside libstdc++
# std::variant headers when std::visit is inlined across the 8-arm
# record_variant. Suppress only on GCC; clang does not warn.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(sampling PRIVATE -Wno-maybe-uninitialized)
endif()

if(CPU_SNAP_INSTALL)
    install(TARGETS sampling
        EXPORT  cpu_snapTargets
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
    install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/cpu_snap
        FILES_MATCHING PATTERN "*.hpp")
endif()
