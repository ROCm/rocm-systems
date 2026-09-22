#
#
#   Installation of public C++ headers (implementations)
#
#
set(ROCPROFILER_CXX_CONTAINERS_HEADER_FILES c_array.hpp)

install(
    FILES ${ROCPROFILER_CXX_CONTAINERS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/cxx/container
    COMPONENT development)
