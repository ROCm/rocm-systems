################################################################################
# Copyright (c) 2023 - 2025 Advanced Micro Devices, Inc.
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
  find_package(PkgConfig)
  if(PKG_CONFIG_FOUND)
    pkg_check_modules(_FFMPEG_AVCODEC libavcodec)
    pkg_check_modules(_FFMPEG_AVFORMAT libavformat)
    pkg_check_modules(_FFMPEG_AVUTIL libavutil)
  endif()

  # AVCODEC
  find_path(AVCODEC_INCLUDE_DIR 
    NAMES libavcodec/avcodec.h
    PATHS ${_FFMPEG_AVCODEC_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include
    PATH_SUFFIXES ffmpeg libav
  )
  mark_as_advanced(AVCODEC_INCLUDE_DIR)
  find_library(AVCODEC_LIBRARY
    NAMES avcodec
    PATHS ${_FFMPEG_AVCODEC_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib
  )
  mark_as_advanced(AVCODEC_LIBRARY)

  # AVFORMAT
  find_path(AVFORMAT_INCLUDE_DIR 
    NAMES libavformat/avformat.h
    PATHS ${_FFMPEG_AVFORMAT_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include
    PATH_SUFFIXES ffmpeg libav
  )
  mark_as_advanced(AVFORMAT_INCLUDE_DIR)
  find_library(AVFORMAT_LIBRARY
    NAMES avformat
    PATHS ${_FFMPEG_AVFORMAT_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib
  )
  mark_as_advanced(AVFORMAT_LIBRARY)

  # AVUTIL
  find_path(AVUTIL_INCLUDE_DIR 
    NAMES libavutil/avutil.h
    PATHS ${_FFMPEG_AVUTIL_INCLUDE_DIRS}
      /usr/local/include
      /usr/include
      /opt/local/include
      /sw/include
    PATH_SUFFIXES ffmpeg libav
  )
  mark_as_advanced(AVUTIL_INCLUDE_DIR)
  find_library(AVUTIL_LIBRARY
    NAMES avutil
    PATHS ${_FFMPEG_AVUTIL_LIBRARY_DIRS}
      /usr/local/lib
      /usr/lib
      /opt/local/lib
      /sw/lib
  )
  mark_as_advanced(AVUTIL_LIBRARY)

  if(AVCODEC_LIBRARY AND AVFORMAT_LIBRARY)
    set(FFMPEG_FOUND TRUE)
  endif()
  
  if(_FFMPEG_AVCODEC_VERSION VERSION_LESS 58.18.100 OR _FFMPEG_AVFORMAT_VERSION VERSION_LESS 58.12.100 OR _FFMPEG_AVUTIL_VERSION VERSION_LESS 56.14.100)
    if(FFMPEG_FOUND)
      message("-- ${White}FFMPEG   required min version - 4.0.4 Found:${FFMPEG_VERSION}")
      message("-- ${White}AVCODEC  required min version - 58.18.100 Found:${_FFMPEG_AVCODEC_VERSION}${ColourReset}")
      message("-- ${White}AVFORMAT required min version - 58.12.100 Found:${_FFMPEG_AVFORMAT_VERSION}${ColourReset}")
      message("-- ${White}AVUTIL   required min version - 56.14.100 Found:${_FFMPEG_AVUTIL_VERSION}${ColourReset}")
    endif()
    set(FFMPEG_FOUND FALSE)
    message( "-- ${Yellow}NOTE: FindFFmpeg failed to find -- FFMPEG${ColourReset}" )
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

