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

if(FFMPEG_LIBRARIES AND FFMPEG_INCLUDE_DIR)
  set(FFMPEG_FOUND TRUE)
else()
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

  # On Windows, if no root was explicitly given, try to locate an FFmpeg/libav
  # install automatically. We probe PATH for the libav runtime libraries
  # (avcodec/avformat/avutil DLLs) and derive the install prefix from their
  # directory (bin/../), then add well-known package-manager / manual install
  # locations. Headers and import libraries are resolved by the
  # find_path/find_library calls below.
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
    # Fallback: well-known package-manager install locations
    file(GLOB _chocolatey_ffmpeg_roots
      "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg"
      "C:/ProgramData/chocolatey/lib/ffmpeg/tools/ffmpeg-*")
    list(APPEND _FFMPEG_ROOT_HINTS ${_chocolatey_ffmpeg_roots}
      "$ENV{USERPROFILE}/scoop/apps/ffmpeg/current"
      "$ENV{ProgramFiles}/ffmpeg"
      "C:/ffmpeg"
    )
  endif()

  if(WIN32)
    set(_FFMPEG_SEARCH_INCLUDE)
    set(_FFMPEG_SEARCH_LIB)
  else()
    set(_FFMPEG_SEARCH_INCLUDE
      ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include)
    set(_FFMPEG_SEARCH_LIB
      ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib)
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

  if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY)
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
    file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version_major.h" _avcodec_major_line
         REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+[0-9]+")
    if(NOT _avcodec_major_line)
      file(STRINGS "${AVCODEC_INCLUDE_DIR}/libavcodec/version.h" _avcodec_major_line
           REGEX "^#define[ \t]+LIBAVCODEC_VERSION_MAJOR[ \t]+[0-9]+")
    endif()
    if(_avcodec_major_line)
      string(REGEX REPLACE ".*LIBAVCODEC_VERSION_MAJOR[ \t]+([0-9]+).*" "\\1" _avcodec_major "${_avcodec_major_line}")
      set(_FFMPEG_AVCODEC_VERSION "${_avcodec_major}.0.0" CACHE INTERNAL "")
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
      # The import libraries above only satisfy link time. At run time the
      # matching avcodec/avformat/avutil DLLs must be resolvable, so the FFmpeg
      # bin directory needs to be on PATH (or the DLLs copied next to the exe).
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
