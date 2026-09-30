# CMake build for consumers that build addc-base from source (add_subdirectory,
# FetchContent, ExternalProject). meson.build is the primary build: the version
# and soversion are read from it, and tools/check_cmake_parity.py (run in CI)
# fails if this file compiles different sources or ADDC_* defines for any
# product selection. When you change a target in meson.build, mirror it here.

cmake_minimum_required(VERSION 3.20...3.31)

if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/meson.build")
    message(FATAL_ERROR
        "addc-base: meson.build is missing from ${CMAKE_CURRENT_SOURCE_DIR}. "
        "CMake reads the version from it; copy the whole addc-base tree.")
endif()
file(READ "${CMAKE_CURRENT_SOURCE_DIR}/meson.build" _addc_meson_build)
string(REGEX MATCH "version: '([0-9.]+)'" _ "${_addc_meson_build}")
set(_addc_version "${CMAKE_MATCH_1}")
string(REGEX MATCH "soversion: '([0-9]+)'" _ "${_addc_meson_build}")
set(_addc_soversion "${CMAKE_MATCH_1}")
if(NOT _addc_version OR NOT _addc_soversion)
    message(FATAL_ERROR "Could not read version/soversion from meson.build")
endif()
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/meson.build")

project(addc-base VERSION ${_addc_version} LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
    set(ADDC_IS_TOP_LEVEL ON)
else()
    set(ADDC_IS_TOP_LEVEL OFF)
endif()

# ── Options (mirror meson_options.txt) ─────────────────────────────────────────

option(ADDC_ENABLE_LIB "Build and install the Base consumer library" ON)
option(ADDC_ENABLE_CLI "Build the Base command-line tool" ${ADDC_IS_TOP_LEVEL})
option(ADDC_ENABLE_TESTS "Build Base tests and examples" ${ADDC_IS_TOP_LEVEL})
option(ADDC_ENABLE_FUZZING "Build Clang libFuzzer targets (developer/CI only)" OFF)
option(ADDC_ENABLE_INSTALL_HEADERS "Install Base public headers" ON)
set(ADDC_PROJECTS "all" CACHE STRING
    "Hardware projects to compile (all, or a comma/semicolon-separated list of canonical project names)")
# OFF: separate libaddc_lib.so. ON: static archive that a consumer links into
# its own shared library (e.g. libamd_smi.so); consumer's version script or
# visibility then decides what is exported.
option(ADDC_LIB_STATIC "Build addc_lib as a static library" OFF)
# Installing the static library would also require exporting the internal
# archives, so install rules default to OFF for it.
if(ADDC_LIB_STATIC)
    option(ADDC_INSTALL "Generate install rules" OFF)
else()
    option(ADDC_INSTALL "Generate install rules" ON)
endif()
option(ADDC_WERROR "Treat compiler warnings as errors" ${ADDC_IS_TOP_LEVEL})

include(GNUInstallDirs)

# Directory-scoped, so these do not leak into a parent project.
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(ADDC_LIB_STATIC)
    # The archives end up inside the consumer's shared library; export only
    # ADDC_API symbols from them, as --exclude-libs does for the shared build.
    set(CMAKE_C_VISIBILITY_PRESET hidden)
    set(CMAKE_CXX_VISIBILITY_PRESET hidden)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
    if(ADDC_WERROR)
        add_compile_options(-Werror)
    endif()
endif()

# ── Product selection ──────────────────────────────────────────────────────────

set(_addc_all_projects
    firerange genoa mi300a mi300c mi300x mi325x mi350 mi450 turin venice)
string(REPLACE "," ";" _addc_requested "${ADDC_PROJECTS}")
list(LENGTH _addc_requested _addc_requested_count)
if(_addc_requested_count EQUAL 0)
    message(FATAL_ERROR "ADDC_PROJECTS must contain at least one project or all")
endif()
if("all" IN_LIST _addc_requested)
    if(NOT _addc_requested_count EQUAL 1)
        message(FATAL_ERROR "ADDC_PROJECTS=all cannot be combined with project names")
    endif()
    set(_addc_selected ${_addc_all_projects})
else()
    set(_addc_selected "")
    foreach(_p IN LISTS _addc_requested)
        if(NOT _p IN_LIST _addc_all_projects)
            message(FATAL_ERROR "unknown project in ADDC_PROJECTS: ${_p}")
        endif()
        if(_p IN_LIST _addc_selected)
            message(FATAL_ERROR "duplicate project in ADDC_PROJECTS: ${_p}")
        endif()
        list(APPEND _addc_selected ${_p})
    endforeach()
endif()

set(_addc_product_defs "")
foreach(_p IN LISTS _addc_all_projects)
    string(TOUPPER "${_p}" _P)
    if(_p IN_LIST _addc_selected)
        set(ADDC_ENABLE_${_P} ON)
        list(APPEND _addc_product_defs ADDC_HAS_${_P})
    else()
        set(ADDC_ENABLE_${_P} OFF)
    endif()
endforeach()
set(_addc_cpu_project OFF)
if(ADDC_ENABLE_FIRERANGE OR ADDC_ENABLE_GENOA OR ADDC_ENABLE_TURIN OR ADDC_ENABLE_VENICE)
    set(_addc_cpu_project ON)
endif()
set(_addc_mi300_gpu_project OFF)
if(ADDC_ENABLE_MI300A OR ADDC_ENABLE_MI300C OR ADDC_ENABLE_MI300X
        OR ADDC_ENABLE_MI325X OR ADDC_ENABLE_MI350)
    set(_addc_mi300_gpu_project ON)
endif()
list(GET _addc_selected 0 _addc_first_project)
set(_addc_test_defs ${_addc_product_defs} ADDC_TEST_PROJECT="${_addc_first_project}")

# ── Dependencies ───────────────────────────────────────────────────────────────

# Prefer a parent project's target, then an installed package, then the same
# pinned archive as subprojects/nlohmann_json.wrap. For offline builds, set
# FETCHCONTENT_SOURCE_DIR_ADDC_NLOHMANN_JSON to an extracted copy.
if(NOT TARGET nlohmann_json::nlohmann_json)
    find_package(nlohmann_json QUIET)
endif()
if(NOT TARGET nlohmann_json::nlohmann_json)
    include(FetchContent)
    FetchContent_Declare(addc_nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.12.0/include.zip
        URL_HASH SHA256=b8cb0ef2dd7f57f18933997c9934bb1fa962594f701cd5a8d3c2c80541559372
    )
    FetchContent_MakeAvailable(addc_nlohmann_json)
    add_library(nlohmann_json::nlohmann_json INTERFACE IMPORTED GLOBAL)
    target_include_directories(nlohmann_json::nlohmann_json
        INTERFACE "${addc_nlohmann_json_SOURCE_DIR}/single_include")
endif()

set(_addc_include "${PROJECT_SOURCE_DIR}/include")
set(_addc_tier_api "${PROJECT_SOURCE_DIR}/src/tier_api")

# ── Internal libraries ─────────────────────────────────────────────────────────

# Header-only view of Base's supported consumer surface.
add_library(addc_public_headers INTERFACE)
target_include_directories(addc_public_headers INTERFACE
    "$<BUILD_INTERFACE:${_addc_include}>"
    "$<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>"
)
target_link_libraries(addc_public_headers INTERFACE nlohmann_json::nlohmann_json)
target_compile_features(addc_public_headers INTERFACE cxx_std_20)

add_library(addc_common STATIC src/addc-common/base64.cpp)
target_include_directories(addc_common PUBLIC "${PROJECT_SOURCE_DIR}/src")

add_library(addc_product_identity STATIC src/products/product.cpp)
target_include_directories(addc_product_identity PUBLIC "${_addc_tier_api}")
target_compile_definitions(addc_product_identity PRIVATE ${_addc_product_defs})

# addc_cper_parser
set(_srcs
    src/addc-cper-parser/cper-parse.cpp
    src/addc-cper-parser/cper-record.cpp
    src/addc-cper-parser/section_dispatch.cpp
    src/addc-cper-parser/sections/cper-section-ainic.cpp
    src/addc-cper-parser/sections/cper-section-ia32x64.cpp
    src/addc-cper-parser/sections/cper-section-pcie.cpp
)
if(_addc_cpu_project OR _addc_mi300_gpu_project)
    list(APPEND _srcs src/addc-cper-parser/sections/cper-section-amd-epyc-crashdump.cpp)
endif()
if(_addc_cpu_project)
    list(APPEND _srcs
        src/addc-cper-parser/sections/cper-section-amd-epyc-cdd.cpp
        src/addc-cper-parser/sections/cper-section-amd-epyc-pmic.cpp
    )
endif()
if(_addc_cpu_project OR ADDC_ENABLE_MI450)
    list(APPEND _srcs src/addc-cper-parser/sections/cper-section-amd-epyc-mtl.cpp)
endif()
if(_addc_mi300_gpu_project)
    list(APPEND _srcs src/addc-cper-parser/sections/cper-section-amd-mi300-crashdump.cpp)
endif()
if(_addc_mi300_gpu_project OR ADDC_ENABLE_MI450)
    list(APPEND _srcs src/addc-cper-parser/sections/cper-section-amd-mi300-runtime.cpp)
endif()
if(ADDC_ENABLE_MI450)
    list(APPEND _srcs
        src/addc-cper-parser/sections/cper-section-amd-mi450-crashdump.cpp
        src/addc-cper-parser/sections/cper-section-amd-mi450-mtl.cpp
    )
endif()
add_library(addc_cper_parser STATIC ${_srcs})
target_include_directories(addc_cper_parser PUBLIC "${_addc_include}" "${_addc_tier_api}")
target_link_libraries(addc_cper_parser PUBLIC nlohmann_json::nlohmann_json addc_common)
target_compile_definitions(addc_cper_parser PRIVATE ${_addc_product_defs})

# addc_decoder_base
set(_srcs
    src/addc-decoder/boot/boot_afid.cpp
    src/addc-decoder/dbglog/decoder.cpp
    src/addc-decoder/dbglog/venice.cpp
    src/addc-decoder/outbound/msg.cpp
    src/addc-decoder/pcie/decoder.cpp
    src/addc-decoder/wdt/decoder.cpp
)
if(ADDC_ENABLE_VENICE)
    list(APPEND _srcs src/addc-decoder/wdt/venice.cpp)
endif()
if(ADDC_ENABLE_MI450)
    list(APPEND _srcs
        src/addc-decoder/boot_mi450/decoder.cpp
        src/addc-decoder/dbglog/mi450.cpp
    )
endif()
add_library(addc_decoder_base STATIC ${_srcs})
target_include_directories(addc_decoder_base PUBLIC "${_addc_include}" "${_addc_tier_api}")
target_link_libraries(addc_decoder_base PUBLIC
    nlohmann_json::nlohmann_json addc_common addc_product_identity)
target_compile_definitions(addc_decoder_base PRIVATE ${_addc_product_defs})

# addc_mca_base
set(_srcs
    src/addc-mca/decoder.cpp
    src/addc-mca/policy.cpp
)
if(ADDC_ENABLE_VENICE)
    list(APPEND _srcs src/addc-mca/venice/venice.cpp)
endif()
if(ADDC_ENABLE_MI300X OR ADDC_ENABLE_MI325X OR ADDC_ENABLE_MI350)
    list(APPEND _srcs src/addc-mca/mi300x.cpp)
endif()
if(ADDC_ENABLE_MI450)
    list(APPEND _srcs src/addc-mca/mi450.cpp)
endif()
if(ADDC_ENABLE_MI300A)
    list(APPEND _srcs src/addc-mca/mi300a.cpp)
endif()
add_library(addc_mca_base STATIC ${_srcs})
target_include_directories(addc_mca_base PUBLIC "${_addc_include}" "${_addc_tier_api}")
target_link_libraries(addc_mca_base PUBLIC nlohmann_json::nlohmann_json addc_product_identity)
target_compile_definitions(addc_mca_base PRIVATE
    ADDC_VERSION="${PROJECT_VERSION}" ${_addc_product_defs})

# addc_cper_pipeline
set(_srcs
    src/addc-cper-pipeline/detect.cpp
    src/addc-cper-pipeline/pipeline.cpp
    src/addc-cper-pipeline/section_dispatch.cpp
    src/addc-cper-parser/structural.cpp
    src/addc-cper-pipeline/sections/ainic_decoder.cpp
    src/addc-cper-pipeline/sections/ainic_section.cpp
    src/addc-cper-pipeline/sections/epyc_crashdump_section.cpp
    src/addc-cper-pipeline/sections/mtl_section.cpp
    src/addc-cper-pipeline/sections/mi450_mtl_section.cpp
    src/addc-cper-pipeline/sections/cdd_section.cpp
    src/addc-cper-pipeline/sections/pmic_section.cpp
    src/addc-cper-pipeline/sections/gpu_crashdump_section.cpp
    src/addc-cper-pipeline/sections/gpu_mi450_crashdump_section.cpp
    src/addc-cper-pipeline/sections/gpu_mi450_runtime_section.cpp
    src/addc-cper-pipeline/sections/gpu_runtime_section.cpp
    src/addc-cper-pipeline/sections/ia32x64_section.cpp
    src/addc-cper-pipeline/sections/pcie_section.cpp
    src/addc-cper-pipeline/sections/section_conversion.cpp
)
if(_addc_cpu_project)
    list(APPEND _srcs
        src/addc-cper-pipeline/sections/epyc_crashdump_decoder.cpp
        src/addc-cper-pipeline/sections/cdd_decoder.cpp
        src/addc-cper-pipeline/sections/pmic_decoder.cpp
        src/addc-cper-pipeline/sections/ia32x64_decoder.cpp
        src/addc-cper-pipeline/sections/pcie_decoder.cpp
    )
endif()
if(_addc_cpu_project OR ADDC_ENABLE_MI450)
    list(APPEND _srcs src/addc-cper-pipeline/sections/mtl_decoder.cpp)
endif()
if(_addc_mi300_gpu_project)
    list(APPEND _srcs
        src/addc-cper-pipeline/sections/gpu_crashdump_decoder.cpp
        src/addc-cper-pipeline/sections/gpu_runtime_decoder.cpp
    )
endif()
if(ADDC_ENABLE_MI450)
    list(APPEND _srcs
        src/addc-cper-pipeline/sections/gpu_mi450_crashdump_decoder.cpp
        src/addc-cper-pipeline/sections/gpu_mi450_runtime_decoder.cpp
    )
endif()
add_library(addc_cper_pipeline STATIC ${_srcs})
target_include_directories(addc_cper_pipeline PUBLIC "${_addc_include}" "${_addc_tier_api}")
target_link_libraries(addc_cper_pipeline PUBLIC
    nlohmann_json::nlohmann_json addc_common addc_cper_parser addc_mca_base addc_decoder_base)
target_compile_definitions(addc_cper_pipeline PRIVATE ${_addc_product_defs})

# ── Tests ──────────────────────────────────────────────────────────────────────

if(ADDC_ENABLE_TESTS)
    enable_testing()
endif()

# addc_add_test(<name> <test-name> SOURCES ... LIBS ... DEFS ...)
function(addc_add_test exe test_name)
    cmake_parse_arguments(PARSE_ARGV 2 arg "" "" "SOURCES;LIBS;DEFS")
    add_executable(${exe} ${arg_SOURCES})
    target_link_libraries(${exe} PRIVATE ${arg_LIBS})
    target_compile_definitions(${exe} PRIVATE ${arg_DEFS})
    add_test(NAME ${test_name} COMMAND ${exe})
endfunction()

if(ADDC_ENABLE_TESTS)
    string(REPLACE ";" "," _addc_selected_csv "${_addc_selected}")
    addc_add_test(product-selection-test product-selection
        SOURCES tests/product_selection.cpp LIBS addc_mca_base
        DEFS ADDC_EXPECTED_PROJECTS="${_addc_selected_csv}")
    addc_add_test(base64-codec-test base64-codec
        SOURCES tests/base64_codec.cpp LIBS addc_common)
    addc_add_test(base64-fuzz-smoke base64-fuzz-smoke
        SOURCES fuzz/base64_decode_fuzz.cpp LIBS addc_common DEFS ADDC_FUZZ_SMOKE)
    addc_add_test(cper-parser-fuzz-smoke cper-parser-fuzz-smoke
        SOURCES fuzz/cper_parser_fuzz.cpp LIBS addc_cper_parser DEFS ADDC_FUZZ_SMOKE)
    addc_add_test(parser-modes-test parser-modes
        SOURCES tests/parser_modes.cpp LIBS addc_cper_parser DEFS ${_addc_product_defs})
    addc_add_test(parser-boundary-test parser-boundary
        SOURCES tests/parser_boundary.cpp LIBS addc_cper_pipeline)
    addc_add_test(allocation-contract-test allocation-contract
        SOURCES tests/allocation_contract.cpp
        LIBS addc_common addc_cper_parser addc_cper_pipeline)
    if(ADDC_ENABLE_MI300X AND ADDC_ENABLE_MI450)
        addc_add_test(descriptor-platform-smoke descriptor-platform-smoke
            SOURCES tests/descriptor_platform_smoke.cpp LIBS addc_cper_pipeline)
    endif()
    if(_addc_cpu_project)
        addc_add_test(section-dispatch-test section-dispatch
            SOURCES tests/section_dispatch.cpp LIBS addc_cper_pipeline)
    endif()
    addc_add_test(parser-classification-test parser-classification
        SOURCES tests/parser_classification.cpp LIBS addc_cper_parser DEFS ${_addc_product_defs})
    addc_add_test(pipeline-output-contract-test pipeline-output-contract
        SOURCES tests/pipeline_output_contract.cpp LIBS addc_cper_pipeline)
    addc_add_test(bad-page-policy-test bad-page-policy
        SOURCES tests/bad_page_policy.cpp LIBS addc_mca_base)
    addc_add_test(outbound-base-contract-test outbound-base-contract
        SOURCES tests/outbound_base_contract.cpp LIBS addc_decoder_base)
    if(ADDC_ENABLE_VENICE AND ADDC_ENABLE_MI450 AND ADDC_ENABLE_TURIN)
        addc_add_test(decoder-contract-test decoder-contract
            SOURCES tests/decoder_contract.cpp LIBS addc_decoder_base)
    endif()
    if(ADDC_ENABLE_MI450)
        addc_add_test(mca-contract-test mca-contract
            SOURCES tests/mca_contract.cpp LIBS addc_mca_base)
    endif()
endif()

if(ADDC_ENABLE_FUZZING)
    if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        message(FATAL_ERROR "ADDC_ENABLE_FUZZING=ON requires Clang")
    endif()
    add_executable(cper-parser-fuzz fuzz/cper_parser_fuzz.cpp)
    target_link_libraries(cper-parser-fuzz PRIVATE addc_cper_parser)
    target_compile_options(cper-parser-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
    target_link_options(cper-parser-fuzz PRIVATE -fsanitize=fuzzer,address,undefined)
endif()

# ── addc_lib ───────────────────────────────────────────────────────────────────

# Typed report resolution and summary projection, kept independent of the
# shared library so dependent products need not link through a public API edge.
add_library(addc_base_report STATIC src/core/base_report.cpp)
target_include_directories(addc_base_report PUBLIC "${_addc_include}" "${_addc_tier_api}")
target_link_libraries(addc_base_report PUBLIC addc_cper_pipeline addc_public_headers)

if(ADDC_ENABLE_CLI AND NOT ADDC_ENABLE_LIB)
    message(FATAL_ERROR "ADDC_ENABLE_CLI=ON requires ADDC_ENABLE_LIB=ON")
endif()

if(ADDC_INSTALL AND ADDC_LIB_STATIC)
    message(FATAL_ERROR "ADDC_INSTALL=ON is not supported with ADDC_LIB_STATIC=ON")
endif()

if(ADDC_ENABLE_LIB)
    if(ADDC_LIB_STATIC)
        set(_addc_lib_type STATIC)
    else()
        set(_addc_lib_type SHARED)
    endif()
    add_library(addc_lib ${_addc_lib_type}
        src/api/lib.cpp
        src/api/lib_c.cpp
        src/core/runtime.cpp
    )
    add_library(addc::addc_lib ALIAS addc_lib)
    target_include_directories(addc_lib PRIVATE "${_addc_include}" "${_addc_tier_api}")
    # Consumers link only the final facade; the parser/decoder archives stay
    # private. addc_public_headers carries the public include path, C++20
    # (lib.hpp uses std::span) and nlohmann_json (included by lib.hpp).
    target_link_libraries(addc_lib
        PUBLIC addc_public_headers
        PRIVATE addc_base_report addc_cper_parser addc_cper_pipeline
                addc_mca_base addc_decoder_base
    )
    target_compile_definitions(addc_lib PRIVATE
        ADDC_BUILDING_LIB ADDC_VERSION="${PROJECT_VERSION}")
    set_target_properties(addc_lib PROPERTIES
        C_VISIBILITY_PRESET hidden
        CXX_VISIBILITY_PRESET hidden
    )
    if(NOT ADDC_LIB_STATIC)
        set_target_properties(addc_lib PROPERTIES
            VERSION ${PROJECT_VERSION}
            SOVERSION ${_addc_soversion}
        )
    endif()
    if(NOT ADDC_LIB_STATIC AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
        # Keep symbols from static archives local to the shared library.
        target_link_options(addc_lib PRIVATE "LINKER:--exclude-libs,ALL")
    endif()

    if(ADDC_ENABLE_CLI)
        add_executable(addc-tool src/cli/main.cpp)
        target_include_directories(addc-tool PRIVATE "${_addc_tier_api}")
        target_link_libraries(addc-tool PRIVATE addc_lib)
        target_compile_definitions(addc-tool PRIVATE ADDC_VERSION="${PROJECT_VERSION}")
        if(ADDC_ENABLE_TESTS)
            add_test(NAME project-list-native-mca COMMAND addc-tool mca --list-projects)
        endif()
    endif()

    if(ADDC_ENABLE_TESTS)
        find_package(Threads REQUIRED)
        addc_add_test(project-list-c-smoke project-list-c-abi
            SOURCES tests/project_list_c_smoke.c LIBS addc_lib DEFS ${_addc_test_defs})
        # Compares the public facade with the private pipeline; links those
        # test-only dependencies explicitly.
        addc_add_test(cper-api-smoke cper-api-contract
            SOURCES tests/cper_api_smoke.cpp
            LIBS addc_lib addc_cper_parser addc_cper_pipeline addc_mca_base addc_decoder_base
            DEFS ${_addc_test_defs})
        addc_add_test(cper-c-abi-smoke cper-c-abi-contract
            SOURCES tests/cper_c_abi_smoke.c LIBS addc_lib DEFS ${_addc_test_defs})
        addc_add_test(contract-versions-test contract-versions
            SOURCES tests/contract_versions.cpp LIBS addc_lib)
        addc_add_test(context-concurrency-test context-concurrency
            SOURCES tests/context_concurrency.cpp LIBS addc_lib Threads::Threads)
    endif()
endif()

# ── Install ────────────────────────────────────────────────────────────────────

if(ADDC_INSTALL AND ADDC_ENABLE_INSTALL_HEADERS)
    install(FILES
        include/addc/lib.hpp
        include/addc/lib_c.h
        include/addc/export.h
        include/addc/options.hpp
        include/addc/report_summary.hpp
        include/addc/common.hpp
        include/addc/json.hpp
        include/addc/schema_version.hpp
        DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/addc"
    )
endif()

if(ADDC_INSTALL AND ADDC_ENABLE_LIB)
    include(CMakePackageConfigHelpers)

    install(TARGETS addc_lib addc_public_headers EXPORT addc-baseTargets)
    set(_addc_cmake_dir "${CMAKE_INSTALL_LIBDIR}/cmake/addc-base")
    install(EXPORT addc-baseTargets NAMESPACE addc:: DESTINATION "${_addc_cmake_dir}")
    configure_package_config_file(cmake/addc-baseConfig.cmake.in
        "${PROJECT_BINARY_DIR}/addc-baseConfig.cmake"
        INSTALL_DESTINATION "${_addc_cmake_dir}"
    )
    write_basic_package_version_file("${PROJECT_BINARY_DIR}/addc-baseConfigVersion.cmake"
        COMPATIBILITY SameMajorVersion
    )
    install(FILES
        "${PROJECT_BINARY_DIR}/addc-baseConfig.cmake"
        "${PROJECT_BINARY_DIR}/addc-baseConfigVersion.cmake"
        DESTINATION "${_addc_cmake_dir}"
    )

    configure_file(cmake/addc-lib.pc.in "${PROJECT_BINARY_DIR}/addc-lib.pc" @ONLY)
    install(FILES "${PROJECT_BINARY_DIR}/addc-lib.pc"
        DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")
endif()
