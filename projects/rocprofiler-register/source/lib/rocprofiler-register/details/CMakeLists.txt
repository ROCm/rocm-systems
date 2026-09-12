#
#   builds the rocprofiler-register library
#
set(ROCPROFILER_REGISTER_SHARED_LIBRARY_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX}")
if(DEFINED CMAKE_SHARED_LIBRARY_PREFIX_CXX)
    set(ROCPROFILER_REGISTER_SHARED_LIBRARY_PREFIX "${CMAKE_SHARED_LIBRARY_PREFIX_CXX}")
endif()

set(ROCPROFILER_REGISTER_SHARED_LIBRARY_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX}")
if(DEFINED CMAKE_SHARED_LIBRARY_SUFFIX_CXX)
    set(ROCPROFILER_REGISTER_SHARED_LIBRARY_SUFFIX "${CMAKE_SHARED_LIBRARY_SUFFIX_CXX}")
endif()

configure_file(library_config.hpp.in ${CMAKE_CURRENT_BINARY_DIR}/library_config.hpp @ONLY)

set(rocprofiler_register_details_sources dl.cpp logging.cpp utility.cpp)

set(rocprofiler_register_details_headers
    ${CMAKE_CURRENT_BINARY_DIR}/library_config.hpp environment.hpp dl.hpp filesystem.hpp
    logging.hpp utility.hpp)

target_sources(rocprofiler-register PRIVATE ${rocprofiler_register_details_sources}
                                            ${rocprofiler_register_details_headers})
