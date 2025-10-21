#
#   function for
#
include_guard(DIRECTORY)

if(NOT TARGET rocprofiler-systems::rocprofiler-systems-user-library)
    find_package(rocprofiler-systems REQUIRED COMPONENTS user)
endif()

if(NOT coz-profiler_FOUND)
    find_package(coz-profiler QUIET)
endif()

if(NOT TARGET rocprofsys-causal-samples)
    add_custom_target(rocprofsys-causal-samples)
endif()

function(rocprofiler_systems_causal_sample_executable _NAME)
    cmake_parse_arguments(
        CAUSAL
        ""
        ""
        "SOURCES;DEFINITIONS;INCLUDE_DIRECTORIES;LINK_LIBRARIES"
        ${ARGN}
    )

    function(rocprofiler_systems_causal_sample_interface _TARGET)
        if(NOT TARGET ${_TARGET})
            find_package(Threads REQUIRED)
            add_library(${_TARGET} INTERFACE)
            target_link_libraries(${_TARGET} INTERFACE Threads::Threads ${CMAKE_DL_LIBS})
        endif()
    endfunction()

    rocprofiler_systems_causal_sample_interface(rocprofsys-causal-sample-lib-debug)
    rocprofiler_systems_causal_sample_interface(rocprofsys-causal-sample-lib-no-debug)

    target_compile_options(
        rocprofsys-causal-sample-lib-debug
        INTERFACE -g3 -fno-omit-frame-pointer
    )
    target_compile_options(rocprofsys-causal-sample-lib-no-debug INTERFACE -g0)

    add_executable(${_NAME} ${CAUSAL_SOURCES})
    target_compile_definitions(
        ${_NAME}
        PRIVATE USE_COZ=0 USE_OMNI=0 ${CAUSAL_DEFINITIONS}
    )
    target_include_directories(
        ${_NAME}
        PRIVATE ${ROCPROFSYS_SAMPLE_ROOT_DIR}/causal ${CAUSAL_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
        ${_NAME}
        PRIVATE
            ${CAUSAL_LINK_LIBRARIES}
            rocprofiler-systems::rocprofiler-systems-user-library
            rocprofsys-causal-sample-lib-debug
    )

    add_executable(${_NAME}-rocprofsys ${CAUSAL_SOURCES})
    target_compile_definitions(
        ${_NAME}-rocprofsys
        PRIVATE USE_COZ=0 USE_OMNI=1 ${CAUSAL_DEFINITIONS}
    )
    target_include_directories(
        ${_NAME}-rocprofsys
        PRIVATE ${ROCPROFSYS_SAMPLE_ROOT_DIR}/causal ${CAUSAL_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
        ${_NAME}-rocprofsys
        PRIVATE
            ${CAUSAL_LINK_LIBRARIES}
            rocprofiler-systems::rocprofiler-systems-user-library
            rocprofsys-causal-sample-lib-debug
    )

    add_executable(${_NAME}-ndebug ${CAUSAL_SOURCES})
    target_compile_definitions(
        ${_NAME}-ndebug
        PRIVATE USE_COZ=0 USE_OMNI=0 ${CAUSAL_DEFINITIONS}
    )
    target_include_directories(
        ${_NAME}-ndebug
        PRIVATE ${ROCPROFSYS_SAMPLE_ROOT_DIR}/causal ${CAUSAL_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
        ${_NAME}-ndebug
        PRIVATE
            ${CAUSAL_LINK_LIBRARIES}
            rocprofiler-systems::rocprofiler-systems-user-library
            rocprofsys-causal-sample-lib-no-debug
    )

    add_executable(${_NAME}-rocprofsys-ndebug ${CAUSAL_SOURCES})
    target_compile_definitions(
        ${_NAME}-rocprofsys-ndebug
        PRIVATE USE_COZ=0 USE_OMNI=1 ${CAUSAL_DEFINITIONS}
    )
    target_include_directories(
        ${_NAME}-rocprofsys-ndebug
        PRIVATE ${ROCPROFSYS_SAMPLE_ROOT_DIR}/causal ${CAUSAL_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(
        ${_NAME}-rocprofsys-ndebug
        PRIVATE
            ${CAUSAL_LINK_LIBRARIES}
            rocprofiler-systems::rocprofiler-systems-user-library
            rocprofsys-causal-sample-lib-no-debug
    )

    add_dependencies(
        rocprofsys-causal-samples
        ${_NAME}
        ${_NAME}-rocprofsys
        ${_NAME}-ndebug
        ${_NAME}-rocprofsys-ndebug
    )

    if(coz-profiler_FOUND)
        rocprofiler_systems_causal_sample_interface(rocprofsys-causal-sample-lib-coz)
        target_compile_options(
            rocprofsys-causal-sample-lib-coz
            INTERFACE -g3 -gdwarf-3 -fno-omit-frame-pointer
        )

        add_executable(${_NAME}-coz ${CAUSAL_SOURCES})
        target_compile_definitions(
            ${_NAME}-coz
            PRIVATE USE_COZ=1 USE_OMNI=0 ${CAUSAL_DEFINITIONS}
        )
        target_include_directories(
            ${_NAME}-coz
            PRIVATE ${ROCPROFSYS_SAMPLE_ROOT_DIR}/causal ${CAUSAL_INCLUDE_DIRECTORIES}
        )
        target_link_libraries(
            ${_NAME}-coz
            PRIVATE ${CAUSAL_LINK_LIBRARIES} rocprofsys-causal-sample-lib-coz coz::coz
        )

        add_dependencies(rocprofsys-causal-samples ${_NAME}-coz)
    endif()

    if(ROCPROFSYS_INSTALL_SAMPLES)
        install(
            TARGETS ${_NAME} ${_NAME}-rocprofsys ${_NAME}-coz
            DESTINATION bin
            COMPONENT rocprofiler-systems-samples
            OPTIONAL
        )
    endif()
endfunction()
