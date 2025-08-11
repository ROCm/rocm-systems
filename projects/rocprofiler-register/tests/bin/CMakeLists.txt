#
#
#

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Threads REQUIRED)
find_package(
    hip
    HINTS
    ${ROCM_PATH}
    ENV
    ROCM_PATH
    /opt/rocm
    PATHS
    ${ROCM_PATH}
    ENV
    ROCM_PATH
    /opt/rocm)

if(hip_FOUND)
    add_executable(simple-hip)
    target_sources(simple-hip PRIVATE simple-hip.cpp)
    target_compile_options(simple-hip PRIVATE -W -Wall -Wextra -Werror)
    target_link_libraries(simple-hip PRIVATE Threads::Threads hip::host)
    set(PRELOAD_TESTS_DISABLED OFF)
    install(
        TARGETS simple-hip
        DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/tests/bin
        COMPONENT tests)
else()
    add_executable(simple-hip EXCLUDE_FROM_ALL)
    target_sources(simple-hip PRIVATE simple-hip.cpp)
    target_link_libraries(simple-hip PRIVATE Threads::Threads)
    set(PRELOAD_TESTS_DISABLED ON)
endif()

#
# INSTALL
#
set(TEST_DESTDIR "${CMAKE_BINARY_DIR}/tests/install")
set(simple-hip-install-env "DESTDIR=${TEST_DESTDIR}")

add_test(
    NAME test-simple-hip-install
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install --parallel 4
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

set_tests_properties(
    test-simple-hip-install
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-test"
               ENVIRONMENT
               "${simple-hip-install-env}"
               DISABLED
               ${PRELOAD_TESTS_DISABLED}
               FIXTURES_SETUP
               "simple-hip-install")

#
# NORMAL (no profiling library)
#
string(REPLACE "//" "/" TEST_INSTALL_PREFIX "${TEST_DESTDIR}/${CMAKE_INSTALL_PREFIX}")
set(simple-hip-normal-env
    "LD_LIBRARY_PATH=${TEST_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}:${ROCM_PATH}/${CMAKE_INSTALL_LIBDIR}"
    "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}")

add_test(NAME test-simple-hip-normal COMMAND $<TARGET_FILE:simple-hip>)

set_tests_properties(
    test-simple-hip-normal
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-test"
               ENVIRONMENT
               "${simple-hip-normal-env}"
               DEPENDS
               test-simple-hip-install
               DISABLED
               ${PRELOAD_TESTS_DISABLED}
               FIXTURES_SETUP
               "simple-hip-works"
               FIXTURES_REQUIRED
               "simple-hip-install")

#
# PRELOAD
#
set(simple-hip-preload-env
    ${simple-hip-normal-env}
    "LD_PRELOAD=${ROCPROFILER_REGISTER_MEMCHECK_PRELOAD_LIBRARY}:$<TARGET_FILE:generic-tool::generic-tool>"
    )

add_test(NAME test-simple-hip-preload COMMAND $<TARGET_FILE:simple-hip>)

set_tests_properties(
    test-simple-hip-preload
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-test"
               ENVIRONMENT
               "${simple-hip-preload-env}"
               DEPENDS
               test-simple-hip-install
               DISABLED
               ${PRELOAD_TESTS_DISABLED}
               FIXTURES_REQUIRED
               "simple-hip-install;simple-hip-works")
