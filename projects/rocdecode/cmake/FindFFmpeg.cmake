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

# FFMPEG_ROOT is only consulted on Windows, and every result below is cached,
# so a later configure pointing at a different root would keep the first root's
# headers and libraries: the shortcut immediately below skips discovery outright
# once FFMPEG_LIBRARIES is set. Drop the cached results when the root changes so
# the new one is actually picked up. An unset root is a state like any other --
# comparing the value rather than testing DEFINED means clearing the root also
# invalidates, instead of leaving the deleted root's libraries cached.
if(WIN32 AND NOT "${FFMPEG_ROOT}" STREQUAL "${_FFMPEG_CACHED_ROOT}")
  unset(AVCODEC_INCLUDE_DIR CACHE)
  unset(AVCODEC_LIBRARY CACHE)
  unset(AVFORMAT_INCLUDE_DIR CACHE)
  unset(AVFORMAT_LIBRARY CACHE)
  unset(AVUTIL_INCLUDE_DIR CACHE)
  unset(AVUTIL_LIBRARY CACHE)
  unset(FFMPEG_INCLUDE_DIR CACHE)
  unset(FFMPEG_LIBRARIES CACHE)
  unset(_FFMPEG_AVCODEC_VERSION CACHE)
  set(_FFMPEG_CACHED_ROOT "${FFMPEG_ROOT}" CACHE INTERNAL "")
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

  if(NOT WIN32)
    set(_FFMPEG_SYSTEM_INCLUDE /usr/local/include /usr/include /opt/local/include /sw/include)
    set(_FFMPEG_SYSTEM_LIB /usr/local/lib /usr/lib /opt/local/lib /sw/lib)
    # avformat/avutil may live under a different prefix than avcodec, so each
    # component falls back to the others' pkg-config directories -- but its own
    # come first, so a match within its own prefix always wins and headers and
    # libraries cannot be combined across two installations.
    set(_AVCODEC_SEARCH_INCLUDE  ${_FFMPEG_AVCODEC_INCLUDE_DIRS}  ${_FFMPEG_AVFORMAT_INCLUDE_DIRS} ${_FFMPEG_AVUTIL_INCLUDE_DIRS}   ${_FFMPEG_SYSTEM_INCLUDE})
    set(_AVFORMAT_SEARCH_INCLUDE ${_FFMPEG_AVFORMAT_INCLUDE_DIRS} ${_FFMPEG_AVCODEC_INCLUDE_DIRS}  ${_FFMPEG_AVUTIL_INCLUDE_DIRS}   ${_FFMPEG_SYSTEM_INCLUDE})
    set(_AVUTIL_SEARCH_INCLUDE   ${_FFMPEG_AVUTIL_INCLUDE_DIRS}   ${_FFMPEG_AVCODEC_INCLUDE_DIRS}  ${_FFMPEG_AVFORMAT_INCLUDE_DIRS} ${_FFMPEG_SYSTEM_INCLUDE})
    set(_AVCODEC_SEARCH_LIB  ${_FFMPEG_AVCODEC_LIBRARY_DIRS}  ${_FFMPEG_AVFORMAT_LIBRARY_DIRS} ${_FFMPEG_AVUTIL_LIBRARY_DIRS}   ${_FFMPEG_SYSTEM_LIB})
    set(_AVFORMAT_SEARCH_LIB ${_FFMPEG_AVFORMAT_LIBRARY_DIRS} ${_FFMPEG_AVCODEC_LIBRARY_DIRS}  ${_FFMPEG_AVUTIL_LIBRARY_DIRS}   ${_FFMPEG_SYSTEM_LIB})
    set(_AVUTIL_SEARCH_LIB   ${_FFMPEG_AVUTIL_LIBRARY_DIRS}   ${_FFMPEG_AVCODEC_LIBRARY_DIRS}  ${_FFMPEG_AVFORMAT_LIBRARY_DIRS} ${_FFMPEG_SYSTEM_LIB})
  else()
    # On Windows, allow FFMPEG_ROOT to point to a pre-built FFmpeg installation
    # (e.g. -DFFMPEG_ROOT=C:/ffmpeg). All three components come from one prefix.
    if(FFMPEG_ROOT)
      foreach(_comp AVCODEC AVFORMAT AVUTIL)
        set(_${_comp}_SEARCH_INCLUDE ${FFMPEG_ROOT}/include)
        set(_${_comp}_SEARCH_LIB ${FFMPEG_ROOT}/lib)
      endforeach()
      # An explicit root means that installation and no other. Without this, a
      # component missing from the root would be satisfied from CMAKE_PREFIX_PATH
      # or the system paths, silently combining two FFmpeg installations -- and
      # the version gate below only parses the selected avcodec headers, so the
      # ABI mismatch would not surface until link or run time.
      set(_FFMPEG_FIND_OPTS NO_DEFAULT_PATH)
    endif()
  endif()

  # AVCODEC
  find_path(AVCODEC_INCLUDE_DIR
    NAMES libavcodec/avcodec.h
    PATHS ${_AVCODEC_SEARCH_INCLUDE}
    PATH_SUFFIXES ffmpeg libav
    ${_FFMPEG_FIND_OPTS}
  )
  mark_as_advanced(AVCODEC_INCLUDE_DIR)
  find_library(AVCODEC_LIBRARY
    NAMES avcodec
    PATHS ${_AVCODEC_SEARCH_LIB}
    ${_FFMPEG_FIND_OPTS}
  )
  mark_as_advanced(AVCODEC_LIBRARY)

  # AVFORMAT
  find_path(AVFORMAT_INCLUDE_DIR
    NAMES libavformat/avformat.h
    PATHS ${_AVFORMAT_SEARCH_INCLUDE}
    PATH_SUFFIXES ffmpeg libav
    ${_FFMPEG_FIND_OPTS}
  )
  mark_as_advanced(AVFORMAT_INCLUDE_DIR)
  find_library(AVFORMAT_LIBRARY
    NAMES avformat
    PATHS ${_AVFORMAT_SEARCH_LIB}
    ${_FFMPEG_FIND_OPTS}
  )
  mark_as_advanced(AVFORMAT_LIBRARY)

  # AVUTIL
  find_path(AVUTIL_INCLUDE_DIR
    NAMES libavutil/avutil.h
    PATHS ${_AVUTIL_SEARCH_INCLUDE}
    PATH_SUFFIXES ffmpeg libav
    ${_FFMPEG_FIND_OPTS}
  )
  mark_as_advanced(AVUTIL_INCLUDE_DIR)
  find_library(AVUTIL_LIBRARY
    NAMES avutil
    PATHS ${_AVUTIL_SEARCH_LIB}
    ${_FFMPEG_FIND_OPTS}
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
  else()
    if(FFmpeg_FIND_REQUIRED)
      message(FATAL_ERROR "{Red}FindFFmpeg -- libavcodec or libavformat or libavutil NOT FOUND${ColourReset}")
    endif()
  endif()
endif()

