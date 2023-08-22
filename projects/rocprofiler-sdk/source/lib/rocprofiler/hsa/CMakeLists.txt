set(ROCPROFILER_LIB_HSA_SOURCES hsa.cpp)
set(ROCPROFILER_LIB_HSA_HEADERS hsa.hpp defines.hpp ostream.hpp types.hpp utils.hpp)
target_sources(rocprofiler-library PRIVATE ${ROCPROFILER_LIB_HSA_SOURCES}
                                           ${ROCPROFILER_LIB_HSA_HEADERS})

# add_executable(rocr-example hsa.cpp rocr.hpp) target_link_libraries(rocr-example PRIVATE
# rocprofiler-v2) target_include_directories( rocr-example PRIVATE ${PROJECT_SOURCE_DIR}
# ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)
# target_compile_definitions( rocr-example PRIVATE AMD_INTERNAL_BUILD PROF_API_IMPL
# HIP_PROF_HIP_API_STRING=1 __HIP_PLATFORM_AMD__=1)
