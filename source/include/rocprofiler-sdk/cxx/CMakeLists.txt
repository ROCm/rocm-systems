#
#
# Installation of public C++ headers
#
#
set(ROCPROFILER_CXX_HEADER_FILES enum_string.hpp hash.hpp name_info.hpp operators.hpp
                                 perfetto.hpp utility.hpp serialization.hpp version.hpp)

install(
    FILES ${ROCPROFILER_CXX_HEADER_FILES}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/rocprofiler-sdk/cxx
    COMPONENT development)

add_subdirectory(codeobj)
add_subdirectory(details)
