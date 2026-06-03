#
# Win32 LoadLibrary + PSAPI implementation of the platform-abstraction layer.
# Selected only on Windows builds by the parent CMakeLists.txt.
# encoding.hpp provides UTF-8 <-> UTF-16 conversion used by loader.cpp.
#
# The implementation is compiled into an OBJECT library so that unit tests
# under details/tests/ can link the platform symbols directly. PE/COFF DLLs
# only expose symbols marked __declspec(dllexport); the internal platform
# helpers are not exported from rocprofiler-register.dll, so the test
# executable cannot resolve them by linking against the import library.
# Linking the OBJECT library's .obj files into the test exe sidesteps the
# DLL export boundary. The same pattern is used on Linux for parity.
#

add_library(
    rocprofiler-register-platform-windows-objects OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pe_parser.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/encoding.hpp)

# Match the include search path and compile options of the consuming SHARED
# target so that rooted includes (e.g. "details/platform/loader.hpp") and
# transitive interface include dirs from fmt/glog/headers/build-flags resolve
# identically when the same TU is compiled into either target.
target_include_directories(
    rocprofiler-register-platform-windows-objects
    PRIVATE $<TARGET_PROPERTY:rocprofiler-register,INCLUDE_DIRECTORIES>)

target_link_libraries(
    rocprofiler-register-platform-windows-objects
    PRIVATE rocprofiler-register::headers rocprofiler-register::build-flags
            fmt::fmt glog::glog)

target_sources(
    rocprofiler-register
    PRIVATE $<TARGET_OBJECTS:rocprofiler-register-platform-windows-objects>)

# WINDOWS-DIVERGENCE: psapi provides EnumProcessModulesEx and
# GetModuleInformation used in loader.cpp and pe_parser.cpp. The OBJECT
# library only compiles; the system import library is required at link time
# on every consuming target (the SHARED rocprofiler-register DLL and the
# details unit-test exe, which adds the link in its own CMakeLists.txt).
target_link_libraries(rocprofiler-register PRIVATE psapi)
