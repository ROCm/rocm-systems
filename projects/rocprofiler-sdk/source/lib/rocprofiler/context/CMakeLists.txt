#
# context
#
set(ROCPROFILER_LIB_CONFIG_SOURCES context.cpp domain.cpp)
set(ROCPROFILER_LIB_CONFIG_HEADERS context.hpp domain.hpp allocator.hpp)

target_sources(rocprofiler-library PRIVATE ${ROCPROFILER_LIB_CONFIG_SOURCES}
                                           ${ROCPROFILER_LIB_CONFIG_HEADERS})

# add_executable(rocr-example hsa.cpp rocr.hpp) target_link_libraries(rocr-example PRIVATE
# rocprofiler-v2) target_include_directories( rocr-example PRIVATE ${PROJECT_SOURCE_DIR}
# ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)
# target_compile_definitions( rocr-example PRIVATE AMD_INTERNAL_BUILD PROF_API_IMPL
# HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)
