#
# Linux dlfcn + /proc/<pid>/maps implementation of the platform-abstraction
# layer. Selected only on non-Windows builds by the parent CMakeLists.txt.
#

target_sources(rocprofiler-register
               PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp
                       ${CMAKE_CURRENT_SOURCE_DIR}/pe_parser.cpp)
