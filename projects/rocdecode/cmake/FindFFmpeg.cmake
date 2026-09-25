################################################################################
# Copyright (c) 2023 - 2026 Advanced Micro Devices, Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
################################################################################
################################################################################
# - Try to find ffmpeg libraries (libavcodec, libavformat and libavutil)
# Once done this will define
#
# FFMPEG_FOUND - system has ffmpeg or libav
# FFMPEG_INCLUDE_DIR - the ffmpeg include directory
# FFMPEG_LIBRARIES - Link these to use ffmpeg
################################################################################

set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:/usr/local/lib/pkgconfig")
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(
  FFmpeg
  FOUND_VAR FFMPEG_FOUND
  REQUIRED_VARS
    FFMPEG_LIBRARIES
    FFMPEG_INCLUDE_DIR
    AVCODEC_INCLUDE_DIR
    AVCODEC_LIBRARY
    AVFORMAT_INCLUDE_DIR
    AVFORMAT_LIBRARY
    AVUTIL_INCLUDE_DIR
    AVUTIL_LIBRARY
  VERSION_VAR FFMPEG_VERSION
)

# Every result below is cached, so the shortcut below skips discovery outright
# once FFMPEG_LIBRARIES is set. Drop the cached results when the effective root
# changes so the new one is actually picked up. Both the CMake variable and the
# environment variable are tracked: either is a deliberate, FFmpeg-specific knob
# whose change should trigger re-discovery. PATH-based auto-discovery is not
# tracked -- PATH changes constantly for unrelated reasons, and re-scanning it
# on every reconfigure would be too aggressive for a convenience fallback.
set(_FFMPEG_EFFECTIVE_ROOT "${FFMPEG_ROOT};$ENV{FFMPEG_ROOT}")
if(WIN32 AND NOT "${_FFMPEG_EFFECTIVE_ROOT}" STREQUAL "${_FFMPEG_CACHED_ROOT}")
  unset(AVCODEC_INCLUDE_DIR CACHE)
  unset(AVCODEC_LIBRARY CACHE)
  unset(AVFORMAT_INCLUDE_DIR CACHE)
  unset(AVFORMAT_LIBRARY CACHE)
  unset(AVUTIL_INCLUDE_DIR CACHE)
  unset(AVUTIL_LIBRARY CACHE)
  unset(FFMPEG_INCLUDE_DIR CACHE)
  unset(FFMPEG_LIBRARIES CACHE)
  unset(_FFMPEG_AVCODEC_VERSION CACHE)
  set(_FFMPEG_CACHED_ROOT "${_FFMPEG_EFFECTIVE_ROOT}" CACHE INTERNAL "")
endif()

if(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  set(FFMPEG_FOUND TRUE)
else()
  # find_package_handle_standard_args ran above against whatever was still
  # cached, so FFMPEG_FOUND can be TRUE here -- notably right after the block
  # above invalidated a stale root. Discovery below only ever sets it TRUE, so
  # without this reset a partially populated root would keep that stale TRUE and
  # cache NOTFOUND component paths into FFMPEG_LIBRARIES.
  set(FFMPEG_FOUND FALSE)

  # Reaching this branch means the previous configure did not produce a usable
  # result, so everything here is about to be re-derived -- including the version,
  # which is cached and otherwise only dropped when FFMPEG_ROOT changes. Without
  # this, an FFmpeg upgraded in place under an unchanged root would keep failing
  # the gate below against the version it had when it was first rejected.
  # Windows-only: on Linux pkg_check_modules re-populates this each time, and
  # the header parser below must not override the value pkg-config reports.
  if(WIN32)
    unset(_FFMPEG_AVCODEC_VERSION CACHE)
  endif()

  # use pkg-config to get the directories and then use these values
  # in the FIND_PATH() and FIND_LIBRARY() calls
  if(NOT WIN32)
    find_package(PkgConfig)
    if(PKG_CONFIG_FOUND)
      pkg_check_modules(_FFMPEG_AVCODEC libavcodec)
      pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
      pkg_check_modules(_FFMPEG_AVUTIL libavutil)
    endif()
  endif()

  # Collect hints from FFMPEG_ROOT (CMake var or env var).
  set(_FFMPEG_ROOT_HINTS ${FFMPEG_ROOT} $ENV{FFMPEG_ROOT})
  list(FILTER _FFMPEG_ROOT_HINTS EXCLUDE REGEX "^$")

  # On Windows, if no root was explicitly given, try to locate an FFmpeg
  # installation automatically. Probe PATH for the runtime DLLs and derive
  # the install prefix from their directory (bin/../), then try well-known
  # package-manager and manual install locations.
  if(WIN32 AND "${_FFMPEG_ROOT_HINTS}" STREQUAL "")
    foreach(_dir $ENV{PATH})
      file(GLOB _avcodec_dll "${_dir}/avcodec*.dll")
      file(GLOB _avformat_dll "${_dir}/avformat*.dll")
      file(GLOB _avutil_dll "${_dir}/avutil*.dll")
      if(_avcodec_dll AND _avformat_dll AND _avutil_dll)
        get_filename_component(_FFMPEG_ROOT_FROM_PATH "${_dir}" DIRECTORY)
        list(APPEND _FFMPEG_ROOT_HINTS "${_FFMPEG_ROOT_FROM_PATH}")
      endif()
    endforeach()
    file(GLOB _chocolatey_ffmpeg_roots
      "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg"
      "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg-*")
    list(APPEND _FFMPEG_ROOT_HINTS ${_chocolatey_ffmpeg_roots}
      "$ENV{USERPROFILE}/scoop/apps/ffmpeg/current"
      "$ENV{ProgramFiles}/ffmpeg"
      "C:/ffmpeg"
    )
  endif()

  if(NOT WIN32)
    # Union of all three components' pkg-config dirs plus standard system paths.
    set(_FFMPEG_SEARCH_INCLUDE
      ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
      ${_FFMPEG_AVFORMAT_INCLUDE_DIRS}
      ${_FFMPEG_AVUTIL_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include)
    set(_FFMPEG_SEARCH_LIB
      ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
      ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
      ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib)
  else()
    # Windows resolves headers/libs through _FFMPEG_ROOT_HINTS (FFMPEG_ROOT,
    # $ENV{FFMPEG_ROOT}, or the PATH/common-location probing above).
    set(_FFMPEG_SEARCH_INCLUDE)
    set(_FFMPEG_SEARCH_LIB)
  endif()

  # AVCODEC
  find_path(AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES include include/ffmpeg include/libav ffmpeg libav
    PATHS ${_FFMPEG_SEARCH_INCLUDE}
  )
  mark_as_advanced(AVCODEC_INCLUDE_DIR)
  find_library(AVCODEC_LIBRARY
    NAMES avcodec
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES lib
    PATHS ${_FFMPEG_SEARCH_LIB}
  )
  mark_as_advanced(AVCODEC_LIBRARY)

  # AVFORMAT
  find_path(AVFORMAT_INCLUDE_DIR
    NAMES libavformat/avformat.h
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES include include/ffmpeg include/libav ffmpeg libav
    PATHS ${_FFMPEG_SEARCH_INCLUDE}
  )
  mark_as_advanced(AVFORMAT_INCLUDE_DIR)
  find_library(AVFORMAT_LIBRARY
    NAMES avformat
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES lib
    PATHS ${_FFMPEG_SEARCH_LIB}
  )
  mark_as_advanced(AVFORMAT_LIBRARY)

  # AVUTIL
  find_path(AVUTIL_INCLUDE_DIR
    NAMES libavutil/avutil.h
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES include include/ffmpeg include/libav ffmpeg libav
    PATHS ${_FFMPEG_SEARCH_INCLUDE}
  )
  mark_as_advanced(AVUTIL_INCLUDE_DIR)
  find_library(AVUTIL_LIBRARY
    NAMES avutil
    HINTS ${_FFMPEG_ROOT_HINTS}
    PATH_SUFFIXES lib
    PATHS ${_FFMPEG_SEARCH_LIB}
  )
  mark_as_advanced(AVUTIL_LIBRARY)

  # All six are required. FFMPEG_LIBRARIES below links all three libraries and
  # consumers include all three header directories, so accepting a partial
  # prefix would configure cleanly and then fail later on a literal
  # AVUTIL_LIBRARY-NOTFOUND at link time or AVCODEC_INCLUDE_DIR-NOTFOUND at
  # compile time. find_package_handle_standard_args is called at the top of this
  # file, before discovery runs, so it cannot catch this.
  if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY AND AVUTIL_LIBRARY
     AND AVCODEC_INCLUDE_DIR AND AVFORMAT_INCLUDE_DIR AND AVUTIL_INCLUDE_DIR)
    set(FFMPEG_FOUND TRUE)
  endif()
  
  if(NOT WIN32 AND (_FFMPEG_AVCODEC_VERSION VERSION_LESS 58.18.100 OR _FFMPEG_AVFORMAT_VERSION VERSION_LESS 58.12.100 OR _FFMPEG_AVUTIL_VERSION VERSION_LESS 56.14.100))
    if(FFMPEG_FOUND)
      message("-- ${White}FFMPEG   required min version - 4.0.4 Found:${FFMPEG_VERSION}")
      message("-- ${White}AVCODEC  required min version - 58.18.100 Found:${_FFMPEG_AVCODEC_VERSION}${ColourReset}")
      message("-- ${White}AVFORMAT required min version - 58.12.100 Found:${_FFMPEG_AVFORMAT_VERSION}${ColourReset}")
      message("-- ${White}AVUTIL   required min version - 56.14.100 Found:${_FFMPEG_AVUTIL_VERSION}${ColourReset}")
    endif()
    set(FFMPEG_FOUND FALSE)
    message( "-- ${Yellow}NOTE: FindFFmpeg failed to find -- FFMPEG${ColourReset}" )
  endif()
  
  # When pkg-config is not available (e.g. Windows), parse the version from headers
  if(FFMPEG_FOUND AND NOT _FFMPEG_AVCODEC_VERSION AND AVCODEC_INCLUDE_DIR)
    # LIBAVCODEC_VERSION_MAJOR moved to version_major.h in FFmpeg 5.1; older
    # releases only have version.h. file(STRINGS) is a fatal error on a missing
    # file, so both reads must be guarded by EXISTS.
    if(EXISTS "${AVCODEC_INCLUDE_DIR}/libavcodec/version_major.h")
      file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version_major.h" _avcodec_major_line
           REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+[0-9]+")
    endif()
    if(NOT _avcodec_major_line AND EXISTS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h")
      file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" _avcodec_major_line
           REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+[0-9]+")
    endif()
    # MINOR/MICRO always stay in version.h. They are needed in full: downstream
    # gates compare against 58.134.100 and 60.31.100, so a major-only "60.0.0"
    # would misclassify FFmpeg 6.1 (60.31.102) as pre-6.1.
    if(EXISTS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h")
      file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" _avcodec_minor_line
           REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MINOR[ \t]+[0-9]+")
      file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" _avcodec_micro_line
           REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MICRO[ \t]+[0-9]+")
    endif()
    if(_avcodec_major_line)
      string(REGEX REPLACE ".*LIBAVCODEC_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _avcodec_major "${_avcodec_major_line}")
      set(_avcodec_minor 0)
      set(_avcodec_micro 0)
      if(_avcodec_minor_line)
        string(REGEX REPLACE ".*LIBAVCODEC_VERSION_MINOR[ \t]+([0-9]+).*" "\\1" _avcodec_minor "${_avcodec_minor_line}")
      endif()
      if(_avcodec_micro_line)
        string(REGEX REPLACE ".*LIBAVCODEC_VERSION_MICRO[ \t]+([0-9]+).*" "\\1" _avcodec_micro "${_avcodec_micro_line}")
      endif()
      set(_FFMPEG_AVCODEC_VERSION "${_avcodec_major}.${_avcodec_minor}.${_avcodec_micro}" CACHE INTERNAL "")
    endif()
  endif()

  # The gate above needs pkg-config version data, so on Windows it is applied
  # here instead -- after the headers have been parsed. Only avcodec's version
  # is recoverable that way; avformat/avutil are not checked. An unparseable
  # version is rejected, matching the Linux path, where an unknown version
  # likewise fails the gate.
  if(WIN32 AND FFMPEG_FOUND)
    if(NOT _FFMPEG_AVCODEC_VERSION)
      message("-- ${Yellow}NOTE: FindFFmpeg could not determine the AVCODEC version${ColourReset}")
      set(FFMPEG_FOUND FALSE)
    elseif(_FFMPEG_AVCODEC_VERSION VERSION_LESS 58.18.100)
      message("-- ${White}FFMPEG   required min version - 4.0.4${ColourReset}")
      message("-- ${White}AVCODEC  required min version - 58.18.100 Found:${_FFMPEG_AVCODEC_VERSION}${ColourReset}")
      message("-- ${Yellow}NOTE: FindFFmpeg failed to find -- FFMPEG${ColourReset}")
      set(FFMPEG_FOUND FALSE)
    endif()
  endif()

  if(FFMPEG_FOUND)
    set(FFMPEG_INCLUDE_DIR ${AVFORMAT_INCLUDE_DIR} CACHE INTERNAL "")
    set(FFMPEG_LIBRARIES
      ${AVCODEC_LIBRARY}
      ${AVFORMAT_LIBRARY}
      ${AVUTIL_LIBRARY}
      CACHE INTERNAL ""
    )
  endif()

  if(FFMPEG_FOUND)
    message("-- ${White}Using FFMPEG -- \n\tLibraries:${FFMPEG_LIBRARIES} \n\tIncludes:${FFMPEG_INCLUDE_DIR}${ColourReset}")
    if(WIN32)
      get_filename_component(_FFMPEG_LIB_DIR "${AVCODEC_LIBRARY}" DIRECTORY)
      get_filename_component(_FFMPEG_BIN_DIR "${_FFMPEG_LIB_DIR}/../bin" ABSOLUTE)
      message("-- ${Yellow}NOTE: at run time the FFmpeg DLLs must be on PATH, e.g. add \"${_FFMPEG_BIN_DIR}\" to PATH${ColourReset}")
    endif()
  else()
    if(FFmpeg_FIND_REQUIRED)
      message(FATAL_ERROR "{Red}FindFFmpeg -- libavcodec or libavformat or libavutil NOT FOUND${ColourReset}")
    endif()
  endif()
endif()

