#
# Win32 LoadLibrary + PSAPI implementation of the platform-abstraction layer.
# Selected only on Windows builds by the parent CMakeLists.txt.
# encoding.hpp provides UTF-8 <-> UTF-16 conversion used by loader.cpp.
#

target_sources(rocprofiler-register
               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/pe_parser.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/encoding.hpp)

# WINDOWS-DIVERGENCE: psapi provides EnumProcessModulesEx and
# GetModuleInformation used in loader.cpp and pe_parser.cpp.
target_link_libraries(rocprofiler-register PRIVATE psapi)
