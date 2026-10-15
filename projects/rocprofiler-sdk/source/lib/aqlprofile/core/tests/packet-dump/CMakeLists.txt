set(AQLPROFILE_PACKET_GOLDEN_TEST_SOURCES packet_golden_tests.cpp packet_golden_main.cpp
                                          packet_golden_utils.cpp packet_golden_json.cpp)

find_package(Python3 REQUIRED COMPONENTS Interpreter)
set(AQLPROFILE_PACKET_ARGS_OUTPUT)
foreach(AQLPROFILE_PACKET_GFX gfx1201 gfx942)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/packet_args.py --kind pmc
                --gfx ${AQLPROFILE_PACKET_GFX}
        RESULT_VARIABLE AQLPROFILE_PACKET_ARGS_RESULT
        OUTPUT_VARIABLE AQLPROFILE_PACKET_GFX_ARGS
        ERROR_VARIABLE AQLPROFILE_PACKET_ARGS_LOG
        OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_STRIP_TRAILING_WHITESPACE)
    if(NOT AQLPROFILE_PACKET_ARGS_RESULT EQUAL 0)
        message(
            FATAL_ERROR
                "Failed to resolve ${AQLPROFILE_PACKET_GFX} packet arguments:\n${AQLPROFILE_PACKET_GFX_ARGS}${AQLPROFILE_PACKET_ARGS_LOG}"
            )
    endif()
    if(AQLPROFILE_PACKET_ARGS_LOG)
        message(STATUS "[aqlprofile-packet-golden-tests] ${AQLPROFILE_PACKET_ARGS_LOG}")
    endif()
    string(APPEND AQLPROFILE_PACKET_ARGS_OUTPUT " ${AQLPROFILE_PACKET_GFX_ARGS}")
endforeach()
string(STRIP "${AQLPROFILE_PACKET_ARGS_OUTPUT}" AQLPROFILE_PACKET_ARGS_OUTPUT)
separate_arguments(AQLPROFILE_PACKET_ARGS UNIX_COMMAND "${AQLPROFILE_PACKET_ARGS_OUTPUT}")

set_property(
    DIRECTORY
    APPEND
    PROPERTY CMAKE_CONFIGURE_DEPENDS ${CMAKE_CURRENT_LIST_DIR}/packet_args.py
             ${PROJECT_SOURCE_DIR}/source/share/rocprofiler-sdk/config.yaml)

add_executable(aqlprofile-packet-golden-tests)
target_sources(aqlprofile-packet-golden-tests
               PRIVATE ${AQLPROFILE_PACKET_GOLDEN_TEST_SOURCES})

target_link_libraries(
    aqlprofile-packet-golden-tests
    PRIVATE -Wl,--start-group
            rocprofiler-sdk::rocprofiler-sdk-static-library
            rocprofiler-sdk::rocprofiler-sdk-common-library
            rocprofiler-sdk::rocprofiler-sdk-aqlprofile
            -Wl,--end-group
            rocprofiler-sdk::rocprofiler-sdk-amd-comgr
            rocprofiler-sdk::rocprofiler-sdk-drm
            aqlprofile-core-tests-flags
            GTest::gtest
            nlohmann_json::nlohmann_json)

rocprofiler_add_unit_test(
    TARGET aqlprofile-packet-golden-tests
    SOURCES ${AQLPROFILE_PACKET_GOLDEN_TEST_SOURCES}
    TEST_PREFIX "aqlprofile.packet-golden."
    TEST_LIST aqlprofile_packet_golden_TESTS ARGS ${AQLPROFILE_PACKET_ARGS}
    TIMEOUT 30
    LABELS "unittests;aqlprofile"
    DATA "fixtures"
    ENVIRONMENT
        "AQLPROFILE_PACKET_FIXTURE_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/fixtures"
        "AQLPROFILE_PACKET_DUMP_PATH=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/packet-dumps"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
