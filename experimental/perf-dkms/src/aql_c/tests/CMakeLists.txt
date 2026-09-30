# AQL C Userspace Tests

# Common include directories
set(AQL_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/..)

# Common compile options for tests
set(TEST_COMPILE_OPTIONS -Wall -Wextra -std=c99)
set(TEST_COMPILE_DEFINITIONS USERSPACE_BUILD)

if(ENABLE_DEBUG)
    list(APPEND TEST_COMPILE_DEFINITIONS DEBUG)
endif()

#================================================================================
# Helper function to create test executables
#================================================================================

function(add_aql_test TEST_NAME TEST_SOURCE)
    # Get library sources from ARGN
    set(LIB_SOURCES ${ARGN})

    # Create object library for the test dependencies
    set(LIB_NAME ${TEST_NAME}_lib)
    add_library(${LIB_NAME} OBJECT ${LIB_SOURCES})
    target_include_directories(${LIB_NAME} PRIVATE ${AQL_INCLUDE_DIRS})
    target_compile_options(${LIB_NAME} PRIVATE ${TEST_COMPILE_OPTIONS})
    target_compile_definitions(${LIB_NAME} PRIVATE ${TEST_COMPILE_DEFINITIONS})

    # Create test executable
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    target_include_directories(${TEST_NAME} PRIVATE ${AQL_INCLUDE_DIRS})
    target_compile_options(${TEST_NAME} PRIVATE ${TEST_COMPILE_OPTIONS})
    target_compile_definitions(${TEST_NAME} PRIVATE ${TEST_COMPILE_DEFINITIONS})
    target_link_libraries(${TEST_NAME} PRIVATE ${LIB_NAME})

    # Add to CTest
    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
endfunction()

#================================================================================
# PM4 Packet Tests
#================================================================================

add_aql_test(test_pm4
    test_pm4_packets.c
    ../pm4_packets.c
)

#================================================================================
# Counter Registry Tests
#================================================================================

add_aql_test(test_counter_registry
    test_counter_registry.c
    ../counter_registry.c
    ../gfx12_events.c
    ../gfx9_events.c
    ../arch_creator.c
    ../gfx12_creator.c
    ../gfx9_creator.c
)

#================================================================================
# Packet Generation Tests
#================================================================================

add_aql_test(test_packet_generation
    test_packet_generation.c
    ../packet_generation.c
    ../pm4_packets.c
    ../arch_creator.c
    ../gfx12_creator.c
    ../gfx9_creator.c
    ../gfx12_events.c
    ../gfx9_events.c
    ../counter_registry.c
)

#================================================================================
# AQL Queue Tests
#================================================================================

add_aql_test(test_aql_queue
    test_aql_queue.c
    ../aql_queue.c
)

#================================================================================
# Run all tests target
#================================================================================

add_custom_target(run-tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_pm4 test_counter_registry test_packet_generation test_aql_queue
    COMMENT "Running all AQL tests"
)

#================================================================================
# Individual test run targets
#================================================================================

add_custom_target(run-pm4-test
    COMMAND test_pm4
    DEPENDS test_pm4
    COMMENT "Running PM4 packet tests"
)

add_custom_target(run-counter-test
    COMMAND test_counter_registry
    DEPENDS test_counter_registry
    COMMENT "Running counter registry tests"
)

add_custom_target(run-packet-test
    COMMAND test_packet_generation
    DEPENDS test_packet_generation
    COMMENT "Running packet generation tests"
)
