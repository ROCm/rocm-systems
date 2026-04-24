cmake_minimum_required(VERSION 3.10)
project(rocm_lttng_spike CXX)

find_package(PkgConfig REQUIRED)
pkg_check_modules(LTTNG_UST REQUIRED IMPORTED_TARGET lttng-ust>=2.13)

# --- Plain-link spike (Steps 1-7) ----------------------------------------
add_executable(rocm_lttng_spike spike_main.cpp spike_tp.cpp)
target_include_directories(rocm_lttng_spike PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rocm_lttng_spike PRIVATE PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})

# --- Restrictive-link spike (Step 8) -------------------------------------
# Mirrors the HSA-runtime build pattern:
#   -fvisibility=hidden + -flto + --version-script + --exclude-libs,ALL
# This proves the LTTng provider __tracepoint_provider_* and __tracepoint__*
# symbols can be made visible despite the otherwise-restrictive link, by
# whitelisting them in the version script. Phase 4 will copy the working
# pattern from restrictive.map into hsacore.so.def.
add_library(rocm_lttng_spike_so SHARED spike_main.cpp spike_tp.cpp)
target_include_directories(rocm_lttng_spike_so PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rocm_lttng_spike_so PRIVATE PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})
target_compile_options(rocm_lttng_spike_so PRIVATE -fvisibility=hidden -flto)
target_link_options(rocm_lttng_spike_so PRIVATE
    -fvisibility=hidden -flto
    -Wl,--exclude-libs,ALL
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/restrictive.map)
