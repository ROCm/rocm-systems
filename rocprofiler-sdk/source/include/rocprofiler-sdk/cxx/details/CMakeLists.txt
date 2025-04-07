#
#
#   Installation of public C++ headers (implementations)
#
#
set(ROCPROFILER_CXX_DETAILS_HEADER_FILES tokenize.hpp join.hpp mpl.hpp name_info.hpp)

install(
    FILES ${ROCPROFILER_CXX_DETAILS_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/cxx/details
    COMPONENT development)
