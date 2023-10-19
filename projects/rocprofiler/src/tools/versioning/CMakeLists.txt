# # TODO(aelwazir): To be added later once rocm-core is updated to be rocprofiler dependency

find_library(ROCM_CORE_LIB rocm-core HINTS ${ROCM_PATH}/lib)

if(ROCM_CORE_LIB)
        add_executable(rocprofiler-version version.cpp)
        target_link_libraries(rocprofiler-version PRIVATE ${ROCM_CORE_LIB})
        set_target_properties(rocprofiler-version PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/libexec/rocprofiler)
        target_include_directories(rocprofiler-version PRIVATE ${ROCM_PATH}/include)

        install(TARGETS rocprofiler-version
                RUNTIME DESTINATION ${CMAKE_INSTALL_LIBEXECDIR}/${PROJECT_NAME})
endif()