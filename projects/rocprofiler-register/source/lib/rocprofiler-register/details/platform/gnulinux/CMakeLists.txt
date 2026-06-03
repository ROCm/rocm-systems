#
# Linux dlfcn + /proc/<pid>/maps implementation of the platform-abstraction
# layer. Selected only on non-Windows builds by the parent CMakeLists.txt.
#
# The implementation is compiled into an OBJECT library so that unit tests
# under details/tests/ can link the platform symbols directly. On Windows the
# equivalent symbols are not exported from the SHARED rocprofiler-register
# DLL (PE/COFF requires explicit __declspec(dllexport)), so tests cannot
# obtain them via the shared library; the OBJECT-library approach is used on
# both platforms for parity.
#

add_library(
    rocprofiler-register-platform-gnulinux-objects OBJECT
    ${CMAKE_CURRENT_SOURCE_DIR}/loader.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pe_parser.cpp)

# Match the include search path and compile options of the consuming SHARED
# target so that rooted includes (e.g. "details/platform/loader.hpp") and
# transitive interface include dirs from fmt/glog/headers/build-flags resolve
# identically when the same TU is compiled into either target.
target_include_directories(
    rocprofiler-register-platform-gnulinux-objects
    PRIVATE $<TARGET_PROPERTY:rocprofiler-register,INCLUDE_DIRECTORIES>)

target_link_libraries(
    rocprofiler-register-platform-gnulinux-objects
    PRIVATE rocprofiler-register::headers rocprofiler-register::build-flags
            fmt::fmt glog::glog)

target_sources(
    rocprofiler-register
    PRIVATE $<TARGET_OBJECTS:rocprofiler-register-platform-gnulinux-objects>)
