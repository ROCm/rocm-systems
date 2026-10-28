cmake_minimum_required(VERSION 3.21)
project(cpu_snap
    LANGUAGES CXX
    VERSION   0.1.0
    DESCRIPTION "CPU profiling library suite: cpu_snap core + sampling + processing")

# ---------------------------------------------------------------------------
# Project-wide settings
# ---------------------------------------------------------------------------
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Symlink compile_commands.json into the source root so editor / clangd
# at the project root sees the compile DB without manual setup.
if(CMAKE_EXPORT_COMPILE_COMMANDS AND NOT CMAKE_BINARY_DIR STREQUAL CMAKE_SOURCE_DIR)
    file(CREATE_LINK
        "${CMAKE_BINARY_DIR}/compile_commands.json"
        "${CMAKE_SOURCE_DIR}/compile_commands.json"
        SYMBOLIC)
endif()

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# ---------------------------------------------------------------------------
# Sanitizer build: CPU_SNAP_SANITIZER=none|asan|tsan|ubsan|asan+ubsan
#   ASan/UBSan: catches use-after-free / overflow / undefined behaviour.
#   TSan      : catches data races; use to validate worker-vs-main access.
#   Sanitizers are mutually exclusive except asan+ubsan, which compose.
# ---------------------------------------------------------------------------
set(CPU_SNAP_SANITIZER "none" CACHE STRING
    "Sanitizer mode: none, asan, tsan, ubsan, asan+ubsan")
set_property(CACHE CPU_SNAP_SANITIZER PROPERTY STRINGS
    "none" "asan" "tsan" "ubsan" "asan+ubsan")

if(NOT CPU_SNAP_SANITIZER STREQUAL "none")
    set(_san_flags "")
    if(CPU_SNAP_SANITIZER MATCHES "asan")
        list(APPEND _san_flags -fsanitize=address)
    endif()
    if(CPU_SNAP_SANITIZER MATCHES "tsan")
        list(APPEND _san_flags -fsanitize=thread)
    endif()
    if(CPU_SNAP_SANITIZER MATCHES "ubsan")
        list(APPEND _san_flags -fsanitize=undefined -fno-sanitize-recover=undefined)
    endif()
    list(APPEND _san_flags -fno-omit-frame-pointer -g)
    add_compile_options(${_san_flags})
    add_link_options(${_san_flags})
    message(STATUS "cpu_snap sanitizer: ${CPU_SNAP_SANITIZER}")
endif()

# ---------------------------------------------------------------------------
# Coverage build: CPU_SNAP_COVERAGE=ON enables gcov/clang-coverage
#   instrumentation across all targets. Mutually exclusive with
#   CPU_SNAP_SANITIZER (the runtime libraries collide).
# ---------------------------------------------------------------------------
option(CPU_SNAP_COVERAGE "Enable gcov/clang-coverage instrumentation" OFF)

if(CPU_SNAP_COVERAGE AND NOT CPU_SNAP_SANITIZER STREQUAL "none")
    message(FATAL_ERROR
        "CPU_SNAP_COVERAGE is mutually exclusive with CPU_SNAP_SANITIZER "
        "(got CPU_SNAP_SANITIZER=${CPU_SNAP_SANITIZER}). The runtime "
        "instrumentation collides; pick one.")
endif()

if(CPU_SNAP_COVERAGE)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(STATUS "cpu_snap coverage: ON (${CMAKE_CXX_COMPILER_ID})")
        add_compile_options(--coverage -O0 -g -fno-inline -fno-inline-small-functions
                            -fno-default-inline)
        add_link_options(--coverage)
    else()
        message(FATAL_ERROR
            "CPU_SNAP_COVERAGE=ON requires GCC or Clang "
            "(got ${CMAKE_CXX_COMPILER_ID}).")
    endif()
endif()

# ---------------------------------------------------------------------------
# Component selection. Auto-enable cpu_snap when any consumer is on.
# ---------------------------------------------------------------------------
option(CPU_SNAP_BUILD_CORE    "Shared core: wire records, chunk framer, reader, sample reconstructor" ON)
option(CPU_SNAP_BUILD_SAMPLING   "Sampling component"                             ON)
option(CPU_SNAP_BUILD_PROCESSING "Processing: symbol resolution + timeline shaping" ON)
option(CPU_SNAP_BUILD_EXAMPLES  "Build examples"                                 ON)
option(CPU_SNAP_BUILD_TESTS     "Build unit tests"                               ON)
option(CPU_SNAP_BUILD_BENCH     "Build benchmarks"                               ON)
option(CPU_SNAP_INSTALL         "Install rules + package config"                 ON)

# ---------------------------------------------------------------------------
# HIP example: opt-in, OFF by default. A HIP/ROCm toolchain (hipcc + the
# HIP/HSA runtime libraries) is not present on most build machines, so
# CPU_SNAP_BUILD_EXAMPLES=ON must not silently require it. Turning this ON
# without a usable toolchain only emits a warning and skips the one target;
# it never fails the main configure.
# ---------------------------------------------------------------------------
option(CPU_SNAP_BUILD_HIP_EXAMPLES
    "Build examples/hip_matmul_demo (requires a HIP/ROCm toolchain)" OFF)

if(CPU_SNAP_BUILD_SAMPLING)
    set(CPU_SNAP_BUILD_CORE ON CACHE BOOL "" FORCE)
endif()
if(CPU_SNAP_BUILD_PROCESSING)
    set(CPU_SNAP_BUILD_CORE ON CACHE BOOL "" FORCE)
endif()

find_package(Threads REQUIRED)

# libdw (elfutils) for DWARF stack unwinding. Required only by
# cpu_snap::sampling; cpu_snap core builds without it.
if(CPU_SNAP_BUILD_SAMPLING)
    find_path(LIBDW_INCLUDE_DIR elfutils/libdwfl.h
        PATHS /usr/include /usr/include/x86_64-linux-gnu)
    find_library(LIBDW_LIBRARY NAMES dw)
    if(NOT LIBDW_INCLUDE_DIR OR NOT LIBDW_LIBRARY)
        message(FATAL_ERROR
            "cpu_snap::sampling needs libdw (elfutils). "
            "Install libdw-dev (Debian/Ubuntu) or elfutils-devel (Fedora/RHEL).")
    endif()
    message(STATUS "cpu_snap: libdw found (${LIBDW_LIBRARY})")
endif()

include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

if(CPU_SNAP_BUILD_TESTS)
    enable_testing()
endif()

# ---------------------------------------------------------------------------
# Component subdirectories
# ---------------------------------------------------------------------------
if(CPU_SNAP_BUILD_CORE)
    add_subdirectory(src/cpu_snap/core)
endif()
if(CPU_SNAP_BUILD_SAMPLING)
    add_subdirectory(src/cpu_snap/sampling)
endif()
if(CPU_SNAP_BUILD_PROCESSING)
    add_subdirectory(src/cpu_snap/processing)
endif()

# ---------------------------------------------------------------------------
# Umbrella interface target linking every enabled component
# ---------------------------------------------------------------------------
add_library(cpu_snap_all INTERFACE)
add_library(cpu_snap::all ALIAS cpu_snap_all)
set_target_properties(cpu_snap_all PROPERTIES EXPORT_NAME all)
if(CPU_SNAP_BUILD_CORE)
    target_link_libraries(cpu_snap_all INTERFACE cpu_snap::core)
endif()
if(CPU_SNAP_BUILD_SAMPLING)
    target_link_libraries(cpu_snap_all INTERFACE cpu_snap::sampling)
endif()
if(CPU_SNAP_BUILD_PROCESSING)
    target_link_libraries(cpu_snap_all INTERFACE cpu_snap::processing)
endif()

# ---------------------------------------------------------------------------
# Examples, benchmarks, tests
#   Examples + bench gate per-target on the components they actually use.
# ---------------------------------------------------------------------------
if(CPU_SNAP_BUILD_EXAMPLES)
    add_subdirectory(examples)
    add_subdirectory(tools)
endif()
if(CPU_SNAP_BUILD_BENCH)
    add_subdirectory(bench)
endif()
if(CPU_SNAP_BUILD_TESTS)
    add_subdirectory(tests)
endif()

# ---------------------------------------------------------------------------
# Install + package config for find_package(cpu_snap COMPONENTS ...)
# ---------------------------------------------------------------------------
if(CPU_SNAP_INSTALL)
    install(TARGETS cpu_snap_all EXPORT cpu_snapTargets)

    install(EXPORT cpu_snapTargets
        FILE        cpu_snapTargets.cmake
        NAMESPACE   cpu_snap::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_snap)

    configure_package_config_file(
        ${CMAKE_CURRENT_SOURCE_DIR}/cmake/cpu_snapConfig.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/cpu_snapConfig.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_snap)

    write_basic_package_version_file(
        ${CMAKE_CURRENT_BINARY_DIR}/cpu_snapConfigVersion.cmake
        VERSION       ${PROJECT_VERSION}
        COMPATIBILITY SameMajorVersion)

    install(FILES
        ${CMAKE_CURRENT_BINARY_DIR}/cpu_snapConfig.cmake
        ${CMAKE_CURRENT_BINARY_DIR}/cpu_snapConfigVersion.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/cpu_snap)
endif()

# ---------------------------------------------------------------------------
# clang-format / clang-tidy convenience targets
# ---------------------------------------------------------------------------
file(GLOB_RECURSE CPU_SNAP_FORMAT_SOURCES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/core/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/core/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/sampling/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/sampling/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/processing/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/processing/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/examples/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/bench/*.cpp")

find_program(CLANG_FORMAT_EXE NAMES clang-format-18 clang-format-17 clang-format-16 clang-format-15 clang-format)
if(CLANG_FORMAT_EXE)
    add_custom_target(format
        COMMAND ${CLANG_FORMAT_EXE} -i -style=file ${CPU_SNAP_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running ${CLANG_FORMAT_EXE} on project sources"
        VERBATIM)
    add_custom_target(format-check
        COMMAND ${CLANG_FORMAT_EXE} --dry-run -Werror -style=file ${CPU_SNAP_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Checking format with ${CLANG_FORMAT_EXE}"
        VERBATIM)
else()
    message(STATUS "clang-format not found; 'format' target unavailable")
endif()

find_program(CLANG_TIDY_EXE NAMES clang-tidy-18 clang-tidy-17 clang-tidy-16 clang-tidy-15 clang-tidy)
if(CLANG_TIDY_EXE)
    add_custom_target(tidy
        COMMAND ${CLANG_TIDY_EXE} -p ${CMAKE_BINARY_DIR} ${CPU_SNAP_FORMAT_SOURCES}
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        COMMENT "Running ${CLANG_TIDY_EXE} on project sources"
        VERBATIM)
else()
    message(STATUS "clang-tidy not found; 'tidy' target unavailable")
endif()

# ---------------------------------------------------------------------------
# 'coverage' target: only created when CPU_SNAP_COVERAGE=ON.
# ---------------------------------------------------------------------------
if(CPU_SNAP_COVERAGE)
    find_program(GCOVR_EXE   NAMES gcovr)
    find_program(LCOV_EXE    NAMES lcov)
    find_program(GENHTML_EXE NAMES genhtml)

    set(_cov_root_arg "--root=${CMAKE_CURRENT_SOURCE_DIR}")
    set(_cov_filter_args
        "--filter=${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/core/"
        "--filter=${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/sampling/"
        "--filter=${CMAKE_CURRENT_SOURCE_DIR}/src/cpu_snap/processing/")
    set(_cov_exclude_args
        "--exclude=${CMAKE_CURRENT_SOURCE_DIR}/tests/"
        "--exclude=${CMAKE_CURRENT_SOURCE_DIR}/bench/"
        "--exclude=${CMAKE_CURRENT_SOURCE_DIR}/examples/")

    if(GCOVR_EXE)
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    ${CMAKE_BINARY_DIR}/coverage_html
            COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
            COMMAND ${CMAKE_CTEST_COMMAND}
                    --test-dir ${CMAKE_BINARY_DIR}
                    --output-on-failure -j
            COMMAND ${GCOVR_EXE}
                    ${_cov_root_arg}
                    ${_cov_filter_args}
                    ${_cov_exclude_args}
                    --print-summary
            COMMAND ${CMAKE_COMMAND} -E make_directory
                    ${CMAKE_BINARY_DIR}/coverage_html
            COMMAND ${GCOVR_EXE}
                    ${_cov_root_arg}
                    ${_cov_filter_args}
                    ${_cov_exclude_args}
                    --html-details
                    -o ${CMAKE_BINARY_DIR}/coverage_html/index.html
            DEPENDS ${CMAKE_BINARY_DIR}/CMakeCache.txt
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ctest + gcovr (HTML report: ${CMAKE_BINARY_DIR}/coverage_html/index.html)"
            VERBATIM USES_TERMINAL)
        message(STATUS
            "cpu_snap coverage: 'coverage' target available (gcovr=${GCOVR_EXE})")
    elseif(LCOV_EXE AND GENHTML_EXE)
        add_custom_target(coverage
            COMMAND ${CMAKE_COMMAND} -E rm -rf
                    ${CMAKE_BINARY_DIR}/coverage_html
            COMMAND find ${CMAKE_BINARY_DIR} -name "*.gcda" -delete
            COMMAND ${CMAKE_CTEST_COMMAND}
                    --test-dir ${CMAKE_BINARY_DIR}
                    --output-on-failure -j
            COMMAND ${LCOV_EXE} --directory ${CMAKE_BINARY_DIR}
                                --capture
                                --output-file ${CMAKE_BINARY_DIR}/coverage.info
                                --rc lcov_branch_coverage=1
                                --no-external
            COMMAND ${LCOV_EXE} --remove ${CMAKE_BINARY_DIR}/coverage.info
                                "${CMAKE_CURRENT_SOURCE_DIR}/tests/*"
                                "${CMAKE_CURRENT_SOURCE_DIR}/bench/*"
                                "${CMAKE_CURRENT_SOURCE_DIR}/examples/*"
                                "/usr/include/*"
                                "/usr/lib/*"
                                --output-file ${CMAKE_BINARY_DIR}/coverage.filtered.info
                                --rc lcov_branch_coverage=1
            COMMAND ${LCOV_EXE} --list ${CMAKE_BINARY_DIR}/coverage.filtered.info
            COMMAND ${GENHTML_EXE} ${CMAKE_BINARY_DIR}/coverage.filtered.info
                                   --output-directory ${CMAKE_BINARY_DIR}/coverage_html
                                   --branch-coverage
            DEPENDS ${CMAKE_BINARY_DIR}/CMakeCache.txt
            WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
            COMMENT "Running ctest + lcov (HTML report: ${CMAKE_BINARY_DIR}/coverage_html/index.html)"
            VERBATIM USES_TERMINAL)
        message(STATUS
            "cpu_snap coverage: 'coverage' target available (lcov=${LCOV_EXE})")
    else()
        message(WARNING
            "CPU_SNAP_COVERAGE=ON but neither gcovr nor lcov+genhtml found; "
            "'coverage' target NOT created. Install with:\n"
            "  Debian/Ubuntu: apt install gcovr  (or lcov)\n"
            "  Fedora/RHEL:   dnf install gcovr  (or lcov)")
    endif()
endif()
