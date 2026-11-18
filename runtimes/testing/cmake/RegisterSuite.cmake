# SPDX-License-Identifier: MIT

function(
    runtime_test_register_suite
    suite
    family
    executable
    test_name
    requires_runtime
)
    get_property(_suites GLOBAL PROPERTY runtime_test_suites)
    if(suite IN_LIST _suites)
        message(FATAL_ERROR "Duplicate runtime test suite: ${suite}")
    endif()
    set_property(GLOBAL APPEND PROPERTY runtime_test_suites "${suite}")
    set(suite_family "${family}")
    set(suite_executable "${executable}")
    set(suite_test "${test_name}")
    set(suite_requires_runtime "${requires_runtime}")
    configure_file(
        "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/Suite.cmake.in"
        "${runtime_test_prefix}/cmake/suites/${suite}.cmake"
        @ONLY
    )
endfunction()

function(runtime_test_native target family)
    set_target_properties(
        ${target}
        PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${runtime_test_prefix}/bin/${family}"
    )
    target_compile_features(${target} PRIVATE c_std_11)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Werror -Wpedantic)
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _configuration)
    # Build records are generated after compilation, including on no-op builds.
    add_custom_target(
        ${target}_record
        ALL
        COMMAND
            "${CMAKE_COMMAND}" "-DTEST_EXECUTABLE=$<TARGET_FILE:${target}>"
            "-DRECORD=${runtime_test_prefix}/share/rocm-runtime-tests/${family}.txt"
            "-DSOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            "-DEXTRA_INPUT_ROOTS=${PROJECT_SOURCE_DIR}/cmake;${PROJECT_SOURCE_DIR}/CMakeLists.txt;${ARGN}"
            "-DCOMPILER=${CMAKE_C_COMPILER}; ${CMAKE_C_COMPILER_ID} ${CMAKE_C_COMPILER_VERSION}; configuration=${CMAKE_BUILD_TYPE}; flags=${CMAKE_C_FLAGS} ${CMAKE_C_FLAGS_${_configuration}}; link_flags=${CMAKE_EXE_LINKER_FLAGS} ${CMAKE_EXE_LINKER_FLAGS_${_configuration}}"
            -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RecordBuild.cmake"
        DEPENDS ${target}
        VERBATIM
    )
endfunction()
