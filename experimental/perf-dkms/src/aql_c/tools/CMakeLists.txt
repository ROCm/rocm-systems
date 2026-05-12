# Userspace Tools for Packet Generation and Decoding

# Common include directories
set(AQL_INCLUDE_DIRS
    ${CMAKE_CURRENT_SOURCE_DIR}/..
    ${CMAKE_CURRENT_SOURCE_DIR}/../..
)

# Common compile options for tools
set(TOOLS_COMPILE_OPTIONS -Wall -Wextra -O2)

#================================================================================
# Packet Generation Tool
#================================================================================

set(PACKET_GEN_SOURCES
    packet_gen_tool.c
    ../packet_generation.c
    ../arch_creator.c
    ../gfx12_creator.c
    ../gfx9_creator.c
    ../pm4_packets.c
    ../counter_registry.c
    ../gfx12_events.c
    ../gfx9_events.c
)

add_executable(packet_gen_tool ${PACKET_GEN_SOURCES})

target_include_directories(packet_gen_tool PRIVATE ${AQL_INCLUDE_DIRS})
target_compile_options(packet_gen_tool PRIVATE ${TOOLS_COMPILE_OPTIONS})
target_link_libraries(packet_gen_tool PRIVATE m)

#================================================================================
# PM4 Decoder Tool
#================================================================================

set(PM4_DECODER_SOURCES
    pm4_decoder.c
)

add_executable(pm4_decoder ${PM4_DECODER_SOURCES})

target_include_directories(pm4_decoder PRIVATE ${AQL_INCLUDE_DIRS})
target_compile_options(pm4_decoder PRIVATE ${TOOLS_COMPILE_OPTIONS})

#================================================================================
# AQLProfile GFX9 Capture Tool (requires libhsa-amd-aqlprofile64)
#================================================================================

# Optional roots for dependency discovery.
# - AQLPROFILE_ROOT: local aqlprofile checkout root (source + build dirs)
# - ROCM_PATH: ROCm installation root used for library/header fallback
set(AQLPROFILE_ROOT "$ENV{AQLPROFILE_ROOT}" CACHE PATH
    "Optional root path to local aqlprofile source tree")
set(ROCM_PATH "$ENV{ROCM_PATH}" CACHE PATH
    "ROCm installation root used for dependency discovery")
if(NOT ROCM_PATH)
    set(ROCM_PATH "/opt/rocm")
endif()

set(AQLPROFILE_LIB_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../aqlprofile/build_fresh
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../aqlprofile/build
    ${ROCM_PATH}/lib
)
if(AQLPROFILE_ROOT)
    list(PREPEND AQLPROFILE_LIB_HINTS
        ${AQLPROFILE_ROOT}/build_fresh
        ${AQLPROFILE_ROOT}/build
    )
endif()

set(AQLPROFILE_V2_INCLUDE_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../aqlprofile/src/core/include
    ${ROCM_PATH}/include
)
if(AQLPROFILE_ROOT)
    list(PREPEND AQLPROFILE_V2_INCLUDE_HINTS
        ${AQLPROFILE_ROOT}/src/core/include
    )
endif()

set(AQLPROFILE_VERSION_INCLUDE_HINTS
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../aqlprofile/build/src/core/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../aqlprofile/build_fresh/src/core/include
    ${ROCM_PATH}/include
)
if(AQLPROFILE_ROOT)
    list(PREPEND AQLPROFILE_VERSION_INCLUDE_HINTS
        ${AQLPROFILE_ROOT}/build/src/core/include
        ${AQLPROFILE_ROOT}/build_fresh/src/core/include
    )
endif()

# Only build if AQLProfile headers and library are available
find_library(AQLPROFILE_LIB hsa-amd-aqlprofile64
    HINTS ${AQLPROFILE_LIB_HINTS}
)
find_path(AQLPROFILE_V2_INCLUDE_DIR aqlprofile-sdk/aql_profile_v2.h
    HINTS ${AQLPROFILE_V2_INCLUDE_HINTS}
)
# version.h is generated during aqlprofile build
find_path(AQLPROFILE_VERSION_INCLUDE_DIR aqlprofile-sdk/version.h
    HINTS ${AQLPROFILE_VERSION_INCLUDE_HINTS}
)
find_path(HSA_INCLUDE_DIR hsa/hsa.h
    HINTS ${ROCM_PATH}/include
)

if(AQLPROFILE_LIB AND AQLPROFILE_V2_INCLUDE_DIR AND AQLPROFILE_VERSION_INCLUDE_DIR AND HSA_INCLUDE_DIR)
    enable_language(CXX)
    add_executable(capture_aqlprofile_gfx9 capture_aqlprofile_gfx9.cpp)

    target_include_directories(capture_aqlprofile_gfx9 PRIVATE
        ${HSA_INCLUDE_DIR}
        ${AQLPROFILE_V2_INCLUDE_DIR}
        ${AQLPROFILE_VERSION_INCLUDE_DIR}
    )
    target_compile_options(capture_aqlprofile_gfx9 PRIVATE ${TOOLS_COMPILE_OPTIONS} -std=c++17)
    target_link_libraries(capture_aqlprofile_gfx9 PRIVATE ${AQLPROFILE_LIB})

    message(STATUS "AQLProfile capture tool: enabled")
    message(STATUS "  AQLProfile lib: ${AQLPROFILE_LIB}")
    message(STATUS "  AQLProfile V2 header: ${AQLPROFILE_V2_INCLUDE_DIR}")
    message(STATUS "  HSA include: ${HSA_INCLUDE_DIR}")
else()
    message(STATUS "AQLProfile capture tool: disabled (missing dependencies)")
    if(NOT AQLPROFILE_LIB)
        message(STATUS "  Missing: libhsa-amd-aqlprofile64")
    endif()
    if(NOT AQLPROFILE_V2_INCLUDE_DIR)
        message(STATUS "  Missing: aqlprofile-sdk/aql_profile_v2.h")
    endif()
    if(NOT HSA_INCLUDE_DIR)
        message(STATUS "  Missing: hsa/hsa.h")
    endif()
endif()

#================================================================================
# Installation
#================================================================================

install(TARGETS packet_gen_tool pm4_decoder
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

#================================================================================
# Help target
#================================================================================

add_custom_target(tools-help
    COMMAND ${CMAKE_COMMAND} -E echo "Available tools:"
    COMMAND ${CMAKE_COMMAND} -E echo "  packet_gen_tool - Packet generation tool"
    COMMAND ${CMAKE_COMMAND} -E echo "  pm4_decoder     - PM4 packet decoder"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage examples:"
    COMMAND ${CMAKE_COMMAND} -E echo "  ./packet_gen_tool gfx12 0x1000000 SQ:0:0x42 SQ:1:0x43"
    COMMAND ${CMAKE_COMMAND} -E echo "  echo 'c001790000000200e0000000' | ./pm4_decoder"
    COMMENT "Displaying tools help"
)
