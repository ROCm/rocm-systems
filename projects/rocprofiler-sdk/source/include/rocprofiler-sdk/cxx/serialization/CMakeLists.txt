#
#
#   Installation of public C++ headers (implementations)
#
#
set(ROCPROFILER_CXX_SERIALIZATION_HEADER_FILES save.hpp load.hpp)

install(
    FILES ${ROCPROFILER_CXX_SERIALIZATION_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/cxx/serialization
    COMPONENT development)
