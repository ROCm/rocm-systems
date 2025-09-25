@echo off
setlocal EnableDelayedExpansion
REM This script is for developer convinience only.
REM Combined build script for both ROCR Test Suite and Samples on Windows
REM Builds tests and samples with the real HSA runtime library
REM
REM Usage:
REM   build_all.bat [Debug|Release] [make] [dk_path] - Full build
REM   Examples:
REM     build_all.bat                    - Debug build (cmake + make)
REM     build_all.bat Release            - Release build (cmake + make)
REM     build_all.bat Debug make         - Debug make only
REM     build_all.bat Release make C:\dk - Release make only with custom DK

echo ========================================================================
echo ROCR Test Suite and Samples - Combined Windows Build (Batch Script)
echo ========================================================================
echo.

REM Parse parameters: [Debug|Release] [make] [dk_path]
set BUILD_CONFIG=Debug
set MAKE_ONLY=0
set DK_PARAM=
set PARAM_INDEX=1

REM Process each parameter
for %%i in (%*) do (
    if /i "%%i"=="Debug" (
        set BUILD_CONFIG=Debug
    ) else if /i "%%i"=="Release" (
        set BUILD_CONFIG=Release
    ) else if /i "%%i"=="make" (
        set MAKE_ONLY=1
    ) else (
        if "!DK_PARAM!"=="" set DK_PARAM=%%i
    )
)

if %MAKE_ONLY%==1 (
    echo Make-only mode: Skipping CMake configuration
)
echo Build configuration: %BUILD_CONFIG%
echo.

REM Determine AMD DK directory - check for override parameter, then environment variable
if "%DK_PARAM%"=="" (
    if "%AMD_DK_DIR%"=="" (
        echo ERROR: AMD DK directory not specified
        echo Please set AMD_DK_DIR environment variable or provide path as parameter
        echo Usage: build_all.bat [make] [dk_directory]
        echo Example: build_all.bat C:\tools\dk_github
        echo Example: build_all.bat make C:\tools\dk_github
        exit /b 1
    ) else (
        set DK_DIR=%AMD_DK_DIR%
        echo Using AMD DK directory from environment: %DK_DIR%
    )
) else (
    set DK_DIR=%DK_PARAM%
    echo Using AMD DK directory from command line: %DK_DIR%
)

REM Set MSVC version (can be overridden by environment variable)
if "%MSVC_VERSION%"=="" (
    set MSVC_VERSION=14.31.31107
)

REM Set Windows SDK version (can be overridden by environment variable)
if "%WINSDK_VERSION%"=="" (
    set WINSDK_VERSION=21390
)

REM Set build type for libraries (can be overridden by environment variable)
if "%AMD_BUILD_SHARED_LIBS%"=="" (
    set AMD_BUILD_SHARED_LIBS=ON
)

REM Determine Windows SDK prefix based on version
if %WINSDK_VERSION% GTR 22597 (
    set WINSDK_PREFIX=e
) else (
    set WINSDK_PREFIX=n
)

REM Set up MSVC environment variables
echo Setting up MSVC %MSVC_VERSION% environment...
set INCLUDE=%DK_DIR%\vc\%MSVC_VERSION%\include
set INCLUDE=%INCLUDE%;%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\Include\10.0.%WINSDK_VERSION%.0\ucrt
set INCLUDE=%INCLUDE%;%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\Include\10.0.%WINSDK_VERSION%.0\um
set INCLUDE=%INCLUDE%;%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\Include\10.0.%WINSDK_VERSION%.0\shared

set PATH=%DK_DIR%\vc\%MSVC_VERSION%\bin\Hostx64\x64;%PATH%
set PATH=%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\bin\10.0.%WINSDK_VERSION%.0\x64;%PATH%
set PATH=%DK_DIR%\cmake\cmake-3.21.2-win64-x64\bin;%PATH%

set LIB=%DK_DIR%\vc\%MSVC_VERSION%\lib\x64
set LIB=%LIB%;%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\Lib\10.0.%WINSDK_VERSION%.0\ucrt\x64
set LIB=%LIB%;%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\Lib\10.0.%WINSDK_VERSION%.0\um\x64

set LIBPATH=%DK_DIR%\vc\%MSVC_VERSION%\lib\x64
set VCINSTALLDIR=%DK_DIR%\vc\%MSVC_VERSION%\
set VCToolsInstallDir=%DK_DIR%\vc\%MSVC_VERSION%\
set WindowsSdkDir=%DK_DIR%\ms_sdk\%WINSDK_PREFIX%%WINSDK_VERSION%\10\
set WindowsSDKVersion=10.0.%WINSDK_VERSION%.0\

REM Variable ROCR_ROOT will be used as starting point to find external dependencies (HSA runtime, libraries, compilers)
REM Don't use this variable to locate ROCR source code, etc. CMAKE knows where the source code is.
if "%ROCR_ROOT%"=="" (
    REM Normal builds
    set ROCR_ROOT=%~dp0\..
)

REM Use static to set default extra libraries for static ROCr library. Or set the libraries in this environment variable
if "%AMD_ROCR_EXTRA_LIB%"=="static" (
    set AMD_ROCR_EXTRA_LIB=%ROCR_ROOT%\..\build\native\%BUILD_CONFIG%\x64\rocr_dynamic\runtime\hsa-runtime\oclelf.lib
    set AMD_ROCR_EXTRA_LIB=%AMD_ROCR_EXTRA_LIB%;%ROCR_ROOT%\..\build\native\%BUILD_CONFIG%\x64\rocr\lib\hsakmt.lib
    set AMD_ROCR_EXTRA_LIB=%AMD_ROCR_EXTRA_LIB%;%ROCR_ROOT%\..\build\native\%BUILD_CONFIG%\x64\rocr\lib\hsakmt-staticdrm.lib
    echo Using extra libraries for static ROCR lib: %AMD_ROCR_EXTRA_LIB%
)

set HSA_ROOT=%ROCR_ROOT%\..\install\native\%BUILD_CONFIG%\x64\rocr
set HSA_PATH=%HSA_ROOT%
set HSA_LIBRARY=%HSA_ROOT%\lib\hsa-runtime64.lib

REM Set LLVM/Clang paths (can be overridden by environment variables)
if "%LLVM_ROOT%"=="" (
    set LLVM_ROOT=%ROCR_ROOT%\..\install\native\%BUILD_CONFIG%\x64\lc
)
REM Set up LLVM/Clang environment
echo Setting up LLVM/Clang environment...
echo LLVM Root: %LLVM_ROOT%
set LLVM_BIN=%LLVM_ROOT%\bin
set LLVM_INCLUDE=%LLVM_ROOT%\include
set LLVM_LIB=%LLVM_ROOT%\lib

REM Add LLVM to PATH
set PATH=%LLVM_BIN%;%PATH%

echo Environment setup complete.
echo.
echo Resolved paths:
echo   ROCR_ROOT: %ROCR_ROOT%
echo   LLVM_ROOT: %LLVM_ROOT%
echo   HSA_ROOT: %HSA_ROOT%
echo   HSA_LIBRARY: %HSA_LIBRARY%
echo   AMD_ROCR_EXTRA_LIB: %AMD_ROCR_EXTRA_LIB%
echo   AMD_BUILD_SHARED_LIBS: %AMD_BUILD_SHARED_LIBS%
echo   -DLLVM_DIR="%LLVM_ROOT%"
echo   -DOPENCL_DIR="%LLVM_ROOT%"
echo   -DHSA_PATH="%HSA_PATH%"
echo.

REM Verify tools are available
echo Verifying required tools...
where cl.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cl.exe not found in PATH
    exit /b 1
)
echo Found: cl.exe

where cmake.exe >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: cmake.exe not found in PATH
    exit /b 1
)
echo Found: cmake.exe

REM Check for make.exe at specific path
set MAKE_PATH=%DK_DIR%\bin\make.exe
if not exist "%MAKE_PATH%" (
    echo ERROR: make.exe not found at %MAKE_PATH%
    exit /b 1
)
echo Found: make.exe at %MAKE_PATH%

REM Check for LLVM clang
if not exist "%LLVM_BIN%\clang.exe" (
    echo ERROR: LLVM clang.exe not found at %LLVM_BIN%\clang.exe
    exit /b 1
)
echo Found: LLVM clang.exe at %LLVM_BIN%\clang.exe

echo.
echo ========================================================================
echo Building Test Suite...
echo ========================================================================
echo.

REM Navigate to test_common directory
echo Navigating to test_common directory...
cd suites\test_common

REM Create build directory
echo Creating build directory...
if %MAKE_ONLY%==0 (
    if exist build (
        echo Removing existing build directory...
        rmdir /s /q build
    )
    mkdir build
    cd build

    echo.
    echo Current directory: %CD%
    echo.

    REM Configure CMake for tests
    echo ========================================================================
    echo Running CMake configuration for tests...
    echo ========================================================================
    cmake -G "Unix Makefiles" ^
          -DCMAKE_MAKE_PROGRAM="%MAKE_PATH%" ^
          -DCMAKE_C_COMPILER=cl.exe ^
          -DCMAKE_CXX_COMPILER=cl.exe ^
          -DCMAKE_BUILD_TYPE=%BUILD_CONFIG% ^
          -DBUILD_SHARED_LIBS=%AMD_BUILD_SHARED_LIBS% ^
          -DHSA_PATH="%HSA_PATH%" ^
          -DHSA_LIBRARY="%HSA_LIBRARY%" ^
          -DTARGET_DEVICES="gfx1201;gfx1200" ^
          -DLLVM_DIR="%LLVM_ROOT%" ^
          -DOPENCL_DIR="%LLVM_ROOT%" ^
          -DBITCODE_DIR="%LLVM_ROOT%\amdgcn\bitcode" ^
          -DAMD_ROCR_EXTRA_LIB="%AMD_ROCR_EXTRA_LIB%" ^
          ..

    if %errorlevel% neq 0 (
        echo ERROR: CMake configuration failed for tests
        exit /b 1
    )

    echo.
    echo CMake configuration completed successfully for tests!
    echo.
) else (
    if not exist build (
        echo ERROR: Build directory does not exist. Run without 'make' parameter first.
        exit /b 1
    )
    cd build
    echo Skipping CMake configuration for tests (make-only mode)
    echo.
)

REM Build the tests
echo ========================================================================
echo Building rocrtst with kernels...
echo ========================================================================
echo Using make with 32 parallel build jobs
"%MAKE_PATH%" -j32

if %errorlevel% neq 0 (
    echo ERROR: Test build failed
    exit /b 1
)

echo.
echo ========================================================================
echo Test Build Completed Successfully!
echo ========================================================================
echo.

REM Save test build path
set TEST_BUILD_PATH=%CD%

REM Navigate back to root and build samples
cd ..\..\..\samples

echo.
echo ========================================================================
echo Building Samples...
echo ========================================================================
echo.

REM Create build directory for samples
echo Creating samples build directory...
if %MAKE_ONLY%==0 (
    if exist build (
        echo Removing existing samples build directory...
        rmdir /s /q build
    )
    mkdir build
    cd build

    echo.
    echo Current directory: %CD%
    echo.

    REM Configure CMake for samples
    echo ========================================================================
    echo Running CMake configuration for samples...
    echo ========================================================================
    cmake -G "Unix Makefiles" ^
          -DCMAKE_MAKE_PROGRAM="%MAKE_PATH%" ^
          -DCMAKE_C_COMPILER=cl.exe ^
          -DCMAKE_CXX_COMPILER=cl.exe ^
          -DCMAKE_BUILD_TYPE=%BUILD_CONFIG% ^
          -DBUILD_SHARED_LIBS=%AMD_BUILD_SHARED_LIBS% ^
          -DHSA_PATH="%HSA_PATH%" ^
          -DHSA_LIBRARY="%HSA_LIBRARY%" ^
          -DTARGET_DEVICES="gfx1201;gfx1200" ^
          -DAMD_ROCR_EXTRA_LIB="%AMD_ROCR_EXTRA_LIB%" ^
          ..

    if %errorlevel% neq 0 (
        echo ERROR: CMake configuration failed for samples
        exit /b 1
    )

    echo.
    echo CMake configuration completed successfully for samples!
    echo.
) else (
    if not exist build (
        echo ERROR: Build directory does not exist. Run without 'make' parameter first.
        exit /b 1
    )
    cd build
    echo Skipping CMake configuration for samples (make-only mode)
    echo.
)

REM Build the samples
echo ========================================================================
echo Building samples...
echo ========================================================================
echo Using make with 32 parallel build jobs
"%MAKE_PATH%" -j32

if %errorlevel% neq 0 (
    echo ERROR: Samples build failed
    exit /b 1
)

echo.
echo ========================================================================
echo ALL BUILDS COMPLETED SUCCESSFULLY!
echo ========================================================================

REM Navigate back to original directory
cd ..\..
