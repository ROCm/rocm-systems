cmake_minimum_required(VERSION 3.10)
project(builder_dump CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -w -g -DDEBUG_TRACE")

# Set paths (adjust these as needed or pass as -DAQLPROFILE_ROOT=... etc.)
set(AQLPROFILE_ROOT ${PROJECT_SOURCE_DIR}/../..)
set(ROCM_PATH /opt/rocm)

include_directories(
    ${AQLPROFILE_ROOT}/src
    ${AQLPROFILE_ROOT}
    ${AQLPROFILE_ROOT}/gfxip/gfx9
    ${AQLPROFILE_ROOT}/gfxip
    ${ROCM_PATH}/include
)

set(FACTORY_SRC_FILES
    ${AQLPROFILE_ROOT}/src/core/pm4_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx9_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx908_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx90a_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx940_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx10_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx11_factory.cpp
    ${AQLPROFILE_ROOT}/src/core/gfx12_factory.cpp
)

set(IP_DISCOVERY_SRC_FILES
    ${AQLPROFILE_ROOT}/src/core/vega20_reg_init.cpp
    ${AQLPROFILE_ROOT}/src/core/navi_reg_init.cpp
    ${AQLPROFILE_ROOT}/src/core/parse_ip_discovery.cpp
    ${AQLPROFILE_ROOT}/src/core/ip_offset_table_init.cpp
)

add_executable(builder_dump
    ${AQLPROFILE_ROOT}/src/util/hsa_rsrc_factory.cpp
    generate_dump/builder_dump.cpp
    ${FACTORY_SRC_FILES}
    ${IP_DISCOVERY_SRC_FILES}
)

target_link_libraries(builder_dump
    ${ROCM_PATH}/lib/libhsa-runtime64.so
)

# Always copy builder_dump to the test binary directory after build
add_custom_command(TARGET builder_dump POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_BINARY_DIR}/pm4-packets-regression-tests
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:builder_dump> ${TEST_BINARY_DIR}/pm4-packets-regression-tests/
    COMMENT "Copying builder_dump to test binary directory"
)