# cmake/RcclFormat.cmake
#
# Chooses the string-formatting back end and resolves fmtlib when it is still
# needed.
#
# C++20 toolchains whose standard library provides a usable <format> let RCCL
# drop fmtlib altogether. Anything else -- C++17, or C++20 paired with a
# standard library predating <format>, such as libstdc++ before GCC 13 -- keeps
# fmtlib. src/include/rccl_format.h makes the same decision in the
# preprocessor, so the two must stay in agreement.
#
# Requires RCCL_CXX_STANDARD (see cmake/RcclCxxStandard.cmake).
#
# Sets in the including scope:
#   RCCL_HAS_STD_FORMAT     TRUE when std::format is used
#   RCCL_FORMAT_LINK_LIBS   libraries to link for formatting (empty for std::format)

include(CheckCXXSourceCompiles)
include(FetchContent)

if(RCCL_CXX_STANDARD GREATER_EQUAL 20)
  # Probe the full feature surface RCCL relies on: dynamic width, fill/align
  # and fixed-precision floats.
  set(CMAKE_REQUIRED_FLAGS "-std=c++${RCCL_CXX_STANDARD}")
  check_cxx_source_compiles("
    #include <format>
    #include <string>
    int main() {
      std::string s = std::format(\"{:<{}}|{:>6}|{:.2f}\", \"a\", 4, \"b\", 1.5);
      return s.empty() ? 1 : 0;
    }
  " RCCL_HAS_STD_FORMAT)
  unset(CMAKE_REQUIRED_FLAGS)
endif()

if(RCCL_HAS_STD_FORMAT)
  message(STATUS "RCCL formatting back end: std::format (fmt not required)")
  set(RCCL_FORMAT_LINK_LIBS "")
else()
  if(RCCL_CXX_STANDARD GREATER_EQUAL 20)
    message(STATUS
      "RCCL formatting back end: fmt (C++${RCCL_CXX_STANDARD} toolchain has no usable <format>)")
  else()
    message(STATUS "RCCL formatting back end: fmt (C++${RCCL_CXX_STANDARD})")
  endif()

  # fmt below 10 does not compile under C++20: its own FMT_STRING/consteval
  # paths in format-inl.h are rejected as non-constant expressions. So a C++20
  # fallback needs at least the 10.x that the FetchContent pin below provides,
  # while C++17 is happy with whatever the system offers.
  if(RCCL_CXX_STANDARD GREATER_EQUAL 20)
    set(_rccl_fmt_min 10.0.0)
  else()
    set(_rccl_fmt_min "")
  endif()

  find_package(fmt ${_rccl_fmt_min} QUIET)
  if(NOT fmt_FOUND)
    set(FMT_INSTALL OFF)
    message(STATUS "fmt not found, fetching from source...")
    FetchContent_Declare(
      fmt
      GIT_REPOSITORY https://github.com/fmtlib/fmt
      GIT_TAG        e69e5f977d458f2650bb346dadf2ad30c5320281 # 10.2.1
    )
    FetchContent_MakeAvailable(fmt)
  else()
    message(STATUS "Using system fmt")
    get_target_property(FMT_INCLUDE_DIRS fmt::fmt-header-only INTERFACE_INCLUDE_DIRECTORIES)
    message(STATUS "fmt include directories: ${FMT_INCLUDE_DIRS}")
  endif()

  set(RCCL_FORMAT_LINK_LIBS fmt::fmt-header-only)
endif()
