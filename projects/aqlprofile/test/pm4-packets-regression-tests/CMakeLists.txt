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


find_program(PYTHON3_EXECUTABLE python3)
find_program(PYTEST_EXECUTABLE pytest)

set(CPP_EXECUTABLE builder_dump)

if(PYTHON3_EXECUTABLE AND PYTEST_EXECUTABLE)
    add_test(
        NAME pm4-packets-regression
        COMMAND bash -c "${PYTHON3_EXECUTABLE} -u run_regression_tests.py"
    )
    set_tests_properties(pm4-packets-regression PROPERTIES
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        TIMEOUT 30
        LABELS "regression-tests"
    )
endif()


file(MAKE_DIRECTORY ${TEST_BINARY_DIR}/pm4-packets-regression-tests)

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/test_packets.py
               ${TEST_BINARY_DIR}/pm4-packets-regression-tests/test_packets.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/run_regression_tests.py
               ${TEST_BINARY_DIR}/pm4-packets-regression-tests/run_regression_tests.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/conftest.py
               ${TEST_BINARY_DIR}/pm4-packets-regression-tests/conftest.py COPYONLY)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/pytest.ini
               ${TEST_BINARY_DIR}/pm4-packets-regression-tests/pytest.ini COPYONLY)
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/baselines
     DESTINATION ${TEST_BINARY_DIR}/pm4-packets-regression-tests)