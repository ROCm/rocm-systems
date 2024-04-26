#
#
#   Installation of public C++ headers
#
#
set(ROCPROFILER_CXX_HEADER_FILES name_info.hpp serialization.hpp)

install(
    FILES ${ROCPROFILER_CXX_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/cxx
    COMPONENT development)

add_subdirectory(details)
