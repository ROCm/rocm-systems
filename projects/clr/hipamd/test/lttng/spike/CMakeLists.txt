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
#
# Two .so flavors are built so Phase 0 can determine empirically how
# strict hsacore.so.def needs to be in Phase 4:
#
#   librocm_lttng_spike_so.so          -- whitelist-style version script
#                                         (restrictive.map). Exports
#                                         __tracepoint_*, __lttng_*, etc.
#                                         Conservative; matches the
#                                         original plan assumption.
#
#   librocm_lttng_spike_strict_so.so   -- maximally-strict version script
#                                         (restrictive_strict.map). Exports
#                                         NOTHING. Tests whether the LTTng
#                                         provider's __attribute__((ctor))
#                                         alone is sufficient for sessiond
#                                         registration, without any
#                                         dynsym-visible symbols.
#
# If both .so flavors register their provider with sessiond, Phase 4 can
# leave hsacore.so.def untouched -- a major simplification.
add_library(rocm_lttng_spike_so SHARED spike_main.cpp spike_tp.cpp)
target_include_directories(rocm_lttng_spike_so PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rocm_lttng_spike_so PRIVATE PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})
target_compile_options(rocm_lttng_spike_so PRIVATE -fvisibility=hidden -flto)
target_link_options(rocm_lttng_spike_so PRIVATE
    -fvisibility=hidden -flto
    -Wl,--exclude-libs,ALL
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/restrictive.map)

add_library(rocm_lttng_spike_strict_so SHARED spike_main.cpp spike_tp.cpp)
target_include_directories(rocm_lttng_spike_strict_so PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(rocm_lttng_spike_strict_so PRIVATE PkgConfig::LTTNG_UST ${CMAKE_DL_LIBS})
target_compile_options(rocm_lttng_spike_strict_so PRIVATE -fvisibility=hidden -flto)
target_link_options(rocm_lttng_spike_strict_so PRIVATE
    -fvisibility=hidden -flto
    -Wl,--exclude-libs,ALL
    -Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/restrictive_strict.map)
