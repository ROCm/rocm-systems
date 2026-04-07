#
# Decoder test
#
cmake_minimum_required(VERSION 3.16.0)

project(
    att-decoder-tests
    LANGUAGES CXX
    VERSION 0.0.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE RelWithDebInfo)

option(UNIT_TESTS "Build unit tests" ON)
option(BUILD_INTEGRATION_TESTS "Build integration tests" ON)
option(DISABLE_COMGR "Disable dependency on disassembly" OFF)

# Allow parent CMakeLists.txt to disable unit tests via BUILD_UNIT_TESTS
if(DEFINED BUILD_UNIT_TESTS AND NOT BUILD_UNIT_TESTS)
    set(UNIT_TESTS OFF)
endif()

# Standalone build: if test/ is the top-level project, build the library from source/
if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(BUILD_TESTS
        ON
        CACHE BOOL "" FORCE)
    add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../source ${CMAKE_CURRENT_BINARY_DIR}/source)
endif()

if(DISABLE_COMGR)
    set(APPLE True)
endif()

if(NOT APPLE AND BUILD_INTEGRATION_TESTS)
    # Add nlohmann JSON as an external dependency
    include(FetchContent)
    FetchContent_Declare(
        json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.3)
    FetchContent_MakeAvailable(json)

    add_subdirectory(att-tool)
endif()

enable_testing()

# Build unit tests
if(UNIT_TESTS)
    add_subdirectory(unit)
endif()

if(BUILD_INTEGRATION_TESTS)
    configure_file(unpack.sh unpack.sh COPYONLY USE_SOURCE_PERMISSIONS)
    configure_file(diff.sh diff.sh COPYONLY USE_SOURCE_PERMISSIONS)
    file(COPY "control" DESTINATION "${CMAKE_CURRENT_BINARY_DIR}/")

    set(TEST_BIN_DIR ${CMAKE_CURRENT_BINARY_DIR})

    set(GFX10_DATA navi2_fifo)
    set(GFX11_DATA navi3_float strix strix_barrier)
    set(GFX12_DATA navi4_sdata navi4_fifo navi4_float navi4_cwsr navi4_setpc navi4_barriernop)
    set(MI300_DATA
        mi300_shaderdata
        mi300_gemmv2
        mi300_sgemm
        mi300_template_attention
        mi300_perf
        mi300_pcs
        mi300_pc2
        mi300_cwsr)
    set(MI350_DATA mi350_histo0 mi350_scale)

    set(DATA_DIR ${GFX10_DATA} ${GFX11_DATA} ${GFX12_DATA} ${MI300_DATA} ${MI350_DATA})

    set(CSV_PY ${CMAKE_CURRENT_SOURCE_DIR}/csv_test.py)

    set(LIB_PATH ${CMAKE_BINARY_DIR}/lib/librocprof-trace-decoder${CMAKE_SHARED_LIBRARY_SUFFIX})
    set(SANITIZED_LIB_PATH ${CMAKE_BINARY_DIR}/lib/librocprof-trace-decoder-sanitized${CMAKE_SHARED_LIBRARY_SUFFIX})

    add_test(NAME csv-launch COMMAND python3 ${CSV_PY} --lib ${LIB_PATH})
    set_tests_properties(csv-launch PROPERTIES TIMEOUT 5 LABELS "integration-tests" WORKING_DIRECTORY ${TEST_BIN_DIR})

    if(NOT APPLE)
        add_test(NAME att-launch COMMAND $<TARGET_FILE:tool>)
        set_tests_properties(att-launch PROPERTIES TIMEOUT 5 LABELS "integration-tests" WORKING_DIRECTORY
                                                   ${TEST_BIN_DIR})
    endif()

    add_test(NAME decompress COMMAND ${TEST_BIN_DIR}/unpack.sh)
    set_tests_properties(
        decompress
        PROPERTIES TIMEOUT
                   60
                   LABELS
                   "decompress"
                   WORKING_DIRECTORY
                   ${TEST_BIN_DIR}
                   FIXTURES_SETUP
                   unpack_data)

    set(FAIL_EXP "\\.h|\\.cpp|arning|Error|bort|egmentation|xception")

    # Preload the sanitized library so ASan is active from process start. On macOS, the sanitized dylib carries ASan
    # statically (clang). On Linux (GCC), libasan.so must come first in LD_PRELOAD.
    if(CMAKE_SYSTEM_NAME STREQUAL "Darwin")
        set(SANITIZER_ENV "DYLD_INSERT_LIBRARIES=${SANITIZED_LIB_PATH}")
    else()
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -print-file-name=libasan.so
            OUTPUT_VARIABLE ASAN_RUNTIME
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(SANITIZER_ENV "LD_PRELOAD=${ASAN_RUNTIME}:${SANITIZED_LIB_PATH}")
    endif()

    foreach(DATA ${DATA_DIR})
        string(FIND ${DATA} "_cwsr" IS_CWSR)
        string(FIND ${DATA} "mi300_pcs" IS_PCS)
        if(${DATA} STREQUAL "mi350_histo0")
            set(WENV "SKIP_OCCUPANCY=1")
        elseif(${DATA} STREQUAL "navi4_sdata")
            set(WENV "CHECK_SDATA=1")
        elseif(IS_CWSR EQUAL -1 AND IS_PCS EQUAL -1)
            set(WENV "")
        else()
            set(WENV "ATT_SUPPRESS_WARNING=1")
        endif()

        configure_file("data/${DATA}.tar.gz" "data/${DATA}.tar.gz" COPYONLY USE_SOURCE_PERMISSIONS)
        configure_file("diff.py" "diff.py" COPYONLY USE_SOURCE_PERMISSIONS)

        add_test(NAME ${DATA}_csv COMMAND python3 ${CSV_PY} --lib ${LIB_PATH} "${TEST_BIN_DIR}/${DATA}/*.att"
                                          "${TEST_BIN_DIR}/control/${DATA}/*.csv")
        set_tests_properties(${DATA}_csv PROPERTIES WORKING_DIRECTORY ${TEST_BIN_DIR} FIXTURES_REQUIRED unpack_data)

        # Sanitized test: always use csv_test.py with the sanitized library
        add_test(NAME ${DATA}_sanitize COMMAND python3 ${CSV_PY} --lib ${SANITIZED_LIB_PATH}
                                               "${TEST_BIN_DIR}/${DATA}/*.att" "${TEST_BIN_DIR}/control/${DATA}/*.csv")
        set_tests_properties(
            ${DATA}_sanitize
            PROPERTIES WORKING_DIRECTORY
                       ${TEST_BIN_DIR}
                       TIMEOUT
                       300
                       ENVIRONMENT
                       "${SANITIZER_ENV};ASAN_OPTIONS=detect_leaks=0"
                       FIXTURES_REQUIRED
                       unpack_data)

        set(USES_VCMPX FALSE)
        if(${DATA} STREQUAL "mi300_pcs"
           OR ${DATA} STREQUAL "mi300_gemmv2"
           OR ${DATA} STREQUAL "mi300_sgemm"
           OR ${DATA} STREQUAL "mi300_perf")
            set(USES_VCMPX TRUE)
        endif()

        if(NOT APPLE AND NOT USES_VCMPX)
            add_test(NAME ${DATA}_execute COMMAND $<TARGET_FILE:tool> ${TEST_BIN_DIR}/${DATA}/1_results.json)
            set_tests_properties(
                ${DATA}_execute
                PROPERTIES TIMEOUT
                           60
                           LABELS
                           "integration-tests"
                           FAIL_REGULAR_EXPRESSION
                           ${FAIL_EXP}
                           ENVIRONMENT
                           "${WENV}"
                           WORKING_DIRECTORY
                           ${TEST_BIN_DIR}
                           FIXTURES_REQUIRED
                           unpack_data)

            add_test(NAME ${DATA}_validate COMMAND ${TEST_BIN_DIR}/diff.sh ${DATA})
            set_tests_properties(
                ${DATA}_validate
                PROPERTIES TIMEOUT
                           60
                           LABELS
                           "integration-tests"
                           FAIL_REGULAR_EXPRESSION
                           ".+"
                           WORKING_DIRECTORY
                           ${TEST_BIN_DIR}
                           FIXTURES_REQUIRED
                           unpack_data
                           DEPENDS
                           ${DATA}_execute)
        endif()
    endforeach(DATA)
endif()
