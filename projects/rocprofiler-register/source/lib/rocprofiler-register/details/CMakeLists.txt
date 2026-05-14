#
#   builds the rocprofiler-register library
#
set(rocprofiler_register_details_sources dl.cpp environment.cpp logging.cpp
                                         utility.cpp)

set(rocprofiler_register_details_headers environment.hpp dl.hpp filesystem.hpp
                                         logging.hpp utility.hpp)

target_sources(rocprofiler-register PRIVATE ${rocprofiler_register_details_sources}
                                            ${rocprofiler_register_details_headers})

add_subdirectory(platform)

if(ROCPROFILER_REGISTER_BUILD_TESTS)
    add_subdirectory(tests)
endif()
