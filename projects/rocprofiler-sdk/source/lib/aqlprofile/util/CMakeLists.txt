#
# CMakeLists.txt for aqlprofile utility source files
#

target_sources(rocprofiler-sdk-aqlprofile PRIVATE hsa_rsrc_factory.cpp)

if(ROCPROFILER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
