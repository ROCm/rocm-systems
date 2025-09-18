##
 #######################################################################################################################
 #
 #  Copyright (c) 2025 Advanced Micro Devices, Inc. All Rights Reserved.
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
if (NOT PAL_CLIENT_INTERFACE_MAJOR_VERSION EQUAL 932)
    message(WARNING "PAL: PAL_CLIENT_INTERFACE_MAJOR_VERSION ${PAL_CLIENT_INTERFACE_MAJOR_VERSION} not supported !!!")
endif()
target_link_libraries(pal
    INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/pal.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/palCompilerDeps.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/palUtil.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/pal_uuid.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/cwpack.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/pal_lz4.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/addrlib.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/amdrdf.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/zstd.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/vam.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/UberTraceService.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/DriverUtilsService.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/dd_settings.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/SettingsRpcService2.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddRpcServer.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddNet.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddRpcShared.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddSocket.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/devdriver.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/dd_common.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddCommon.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddCore.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/dd_libyaml.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/mpack.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/metrohash.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/stb_sprintf.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddRpcClient.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddEventStreamer.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddEventClient.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddEventParser.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddEventServer.lib
        ${CMAKE_CURRENT_SOURCE_DIR}/lib/Release/x64/ddYaml.lib
        SetupAPI.Lib
)
target_compile_definitions(pal
    INTERFACE
        PAL_CLIENT_INTERFACE_MAJOR_VERSION=932
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
        shared/devdriver/shared/legacy/inc
        shared/devdriver/third_party/dd_crc32/inc
        shared/metrohash/src
)
