@echo off
REM Build script for wddm_lite_test.exe
REM
REM Compiles the wddm_lite libhsakmt backend and test into a single
REM standalone executable. No CMake or WDK required — all D3DKMT
REM types are defined inline, and gdi32.dll is loaded at runtime.
REM
REM Usage:
REM   build.bat
REM
REM Expects VS Build Tools 2022 to be installed.

setlocal

REM Setup MSVC environment
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvarsall.bat" amd64 >nul 2>&1
if errorlevel 1 (
    echo ERROR: Could not find vcvarsall.bat
    echo Install Visual Studio Build Tools 2022
    exit /b 1
)

set SRCDIR=%~dp0..\..\src\wddm_lite
set INCDIR=%~dp0..\..\include

echo Building wddm_lite_test.exe...

cl /nologo /W3 /O2 /EHsc /std:c++17 ^
    /DWIN32 /D_WIN64 /D_WINDOWS /DNOMINMAX ^
    /FI"stdint.h" ^
    /I"%INCDIR%" /I"%SRCDIR%" ^
    "%SRCDIR%\openclose.cpp" ^
    "%SRCDIR%\topology.cpp" ^
    "%SRCDIR%\version.cpp" ^
    "%SRCDIR%\globals.cpp" ^
    "%SRCDIR%\memory.cpp" ^
    "%SRCDIR%\queues.cpp" ^
    "%SRCDIR%\events.cpp" ^
    "%SRCDIR%\gpu_init.cpp" ^
    "%SRCDIR%\stubs.cpp" ^
    "%~dp0main.cpp" ^
    /Fe:"%~dp0wddm_lite_test.exe" ^
    /link /NOLOGO

if errorlevel 1 (
    echo.
    echo BUILD FAILED
    exit /b 1
)

echo.
echo Build succeeded: wddm_lite_test.exe
echo Run: wddm_lite_test.exe
