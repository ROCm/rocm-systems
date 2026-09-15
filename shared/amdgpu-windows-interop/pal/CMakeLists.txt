##
 #######################################################################################################################
 #
 #  Copyright (c) 2026 Advanced Micro Devices, Inc. All Rights Reserved.
 #
 #  Permission is hereby granted, free of charge, to any person obtaining a copy
 #  of this software and associated documentation files (the "Software"), to deal
 #  in the Software without restriction, including without limitation the rights
 #  to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 #  copies of the Software, and to permit persons to whom the Software is
 #  furnished to do so, subject to the following conditions:
 #
 #  The above copyright notice and this permission notice shall be included in all
 #  copies or substantial portions of the Software.
 #
 #  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 #  IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 #  FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 #  AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 #  LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 #  OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 #  SOFTWARE.
 #
 #######################################################################################################################
cmake_minimum_required(VERSION 3.21)
project(PAL LANGUAGES CXX)
add_library(pal INTERFACE)
set_target_properties(pal PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
    POSITION_INDEPENDENT_CODE TRUE
)
target_compile_features(pal INTERFACE cxx_std_20)
if (NOT PAL_CLIENT_INTERFACE_MAJOR_VERSION EQUAL 995)
    message(WARNING "PAL: PAL_CLIENT_INTERFACE_MAJOR_VERSION ${PAL_CLIENT_INTERFACE_MAJOR_VERSION} not supported !!!")
endif()
if(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(_pal_arch_dir "x86")
else()
    set(_pal_arch_dir "x64")
endif()
set(_pal_lib_dir "${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/${_pal_arch_dir}")
target_link_libraries(pal
    INTERFACE
        ${_pal_lib_dir}/pal.lib
        ${_pal_lib_dir}/palCompilerDeps.lib
        ${_pal_lib_dir}/palUtil.lib
        ${_pal_lib_dir}/pal_uuid.lib
        ${_pal_lib_dir}/cwpack.lib
        ${_pal_lib_dir}/pal_lz4.lib
        ${_pal_lib_dir}/addrlib.lib
        ${_pal_lib_dir}/amdrdf.lib
        ${_pal_lib_dir}/zstd.lib
        ${_pal_lib_dir}/vam.lib
        ${_pal_lib_dir}/UberTraceService.lib
        ${_pal_lib_dir}/DriverUtilsService.lib
        ${_pal_lib_dir}/dd_settings.lib
        ${_pal_lib_dir}/SettingsRpcService2.lib
        ${_pal_lib_dir}/dd_crc32.lib
        ${_pal_lib_dir}/ddApi.lib
        ${_pal_lib_dir}/ddRpcServer.lib
        ${_pal_lib_dir}/ddNet.lib
        ${_pal_lib_dir}/ddRpcShared.lib
        ${_pal_lib_dir}/ddSocket.lib
        ${_pal_lib_dir}/devdriver.lib
        ${_pal_lib_dir}/dd_common.lib
        ${_pal_lib_dir}/ddCommon.lib
        ${_pal_lib_dir}/ddCore.lib
        ${_pal_lib_dir}/dd_libyaml.lib
        ${_pal_lib_dir}/mpack.lib
        ${_pal_lib_dir}/metrohash.lib
        ${_pal_lib_dir}/stb_sprintf.lib
        ${_pal_lib_dir}/ddRpcClient.lib
        ${_pal_lib_dir}/ddEventStreamer.lib
        ${_pal_lib_dir}/ddEventClient.lib
        ${_pal_lib_dir}/ddEventParser.lib
        ${_pal_lib_dir}/ddEventServer.lib
        ${_pal_lib_dir}/ddYaml.lib
        SetupAPI.Lib
)
target_compile_definitions(pal
    INTERFACE
        PAL_CLIENT_INTERFACE_MAJOR_VERSION=995
        GPUOPEN_CLIENT_INTERFACE_MAJOR_VERSION=42
        PAL_BUILD_RDF=1
        PAL_DEVELOPER_BUILD=0
        PAL_KMT_BUILD=1
)
target_include_directories(pal
    INTERFACE
        inc
        inc/core
        inc/gpuUtil
        inc/util
        shared/inc
        shared/devdriver/shared/legacy
        shared/devdriver/shared/legacy/inc
        shared/devdriver/shared/legacy/core/inc
        shared/devdriver/third_party/dd_crc32/inc
        shared/metrohash/src
)
