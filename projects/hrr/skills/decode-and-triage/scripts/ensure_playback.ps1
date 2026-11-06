#!/usr/bin/env pwsh
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$Build = $false
$Help  = $false

$ScriptDir = $PSScriptRoot
$HrrProjectDir = (Resolve-Path (Join-Path $ScriptDir "..\..\..")).Path
# Discovery roots, in order: HIP_PATH, then ROCM_PATH (both when set), else default.
$HipRoots = @()
if ($env:HIP_PATH)  { $HipRoots += $env:HIP_PATH }
if ($env:ROCM_PATH) { $HipRoots += $env:ROCM_PATH }
if ($HipRoots.Count -eq 0) { $HipRoots += "C:\Program Files\AMD\ROCm\6.2" }
$HipRoot = $HipRoots[0]

foreach ($tok in $args) {
    switch -Exact ($tok) {
        "--build" { $Build = $true }
        "--help"  { $Help  = $true }
        "-h"      { $Help  = $true }
        default   { Write-Host "error: unknown arg: $tok" -ForegroundColor Red; exit 1 }
    }
}

if ($Help) {
    @"
usage: ensure_playback.ps1 [--build]
  default: locate an existing hrr-playback.exe (no build)
  --build:  configure + build hrr-playback from projects/hrr when not found

Searched in order:
  HRR_PLAYBACK env, HIP_PATH\bin, ROCM_PATH\bin, PATH.
"@ | Write-Host -ForegroundColor Cyan
    exit 0
}

if ($env:HRR_ENSURE_BUILD -eq "1") { $Build = $true }

function Find-InHipSdk {
    foreach ($root in $HipRoots) {
        foreach ($sub in @("bin", "hip\bin", "tools\bin")) {
            $dir = Join-Path $root $sub
            foreach ($name in @("hrr-playback.exe", "hrr-playback")) {
                $p = Join-Path $dir $name
                if (Test-Path $p -PathType Leaf) { return $p }
            }
        }
    }
    $cmd = Get-Command "hrr-playback.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command "hrr-playback" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Find-InHrrBuild([string]$BuildDir) {
    if (-not $BuildDir) { return $null }
    foreach ($rel in @(
        "playback\hrr-playback.exe",
        "playback\hrr-playback",
        "bin\hrr-playback.exe",
        "bin\hrr-playback"
    )) {
        $p = Join-Path $BuildDir $rel
        if (Test-Path $p -PathType Leaf) { return (Resolve-Path $p).Path }
    }
    $found = Get-ChildItem $BuildDir -Recurse -Filter "hrr-playback.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

if ($env:HRR_PLAYBACK -and (Test-Path $env:HRR_PLAYBACK -PathType Leaf)) {
    Write-Host $env:HRR_PLAYBACK
    exit 0
}

$sdkFound = Find-InHipSdk
if ($sdkFound) { Write-Host $sdkFound; exit 0 }

# ---------------------------------------------------------------------------
# STEP 3 — Build (only when --build requested)
if (-not $Build) {
    Write-Host @"
[ensure_playback] hrr-playback.exe not found.

Choose one of the options below:

  1. Point at an existing binary:
       `$env:HRR_PLAYBACK = 'C:\path\to\hrr-playback.exe'

  2. Install a capture-enabled ROCm/HIP SDK so hrr-playback.exe is under HIP_PATH\bin:
       `$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.2'

  3. Build hrr-playback from projects/hrr:
       .\ensure_playback.ps1 --build

  4. Triage without replay (no binary needed — reads manifest.json only):
       .\triage_archive.ps1 --archive <pid-dir> --no-replay
"@ -ForegroundColor Yellow
    exit 1
}

$CMakeLists = Join-Path $HrrProjectDir "CMakeLists.txt"
if (-not (Test-Path $CMakeLists -PathType Leaf)) {
    Write-Error "projects/hrr CMake project not found at $HrrProjectDir"
    exit 1
}

$BuildOut = if ($env:HRR_BUILD) { $env:HRR_BUILD } else { Join-Path $HrrProjectDir "build-playback" }
New-Item -ItemType Directory -Force -Path $BuildOut | Out-Null

$already = Find-InHrrBuild $BuildOut
if ($already) { Write-Host $already; exit 0 }

Write-Host "[ensure_playback] configuring projects/hrr in $BuildOut" -ForegroundColor DarkGray

# ---- CMake configure -------------------------------------------------------
# ROCM_PATH / HIP_PATH must point to a capture-enabled ROCm/HIP install that
$Prefix = if ($env:CMAKE_PREFIX_PATH) { $env:CMAKE_PREFIX_PATH } else { $HipRoot }
$CMakeArgs = @(
    "-S", $HrrProjectDir,
    "-B", $BuildOut,
    "-DROCM_PATH=$HipRoot",
    "-DCMAKE_PREFIX_PATH=$Prefix",
    "-DHRR_BUILD_PLAYBACK=ON",
    "-DHRR_BUILD_TESTS=OFF",
    "-DCMAKE_BUILD_TYPE=Release"
)

Write-Host "[ensure_playback] cmake $($CMakeArgs -join ' ')" -ForegroundColor DarkGray
cmake @CMakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed (exit $LASTEXITCODE). Review output above."
    exit 1
}

Write-Host "[ensure_playback] cmake --build $BuildOut --target hrr-playback" -ForegroundColor DarkGray
cmake --build $BuildOut --target hrr-playback
if ($LASTEXITCODE -ne 0) {
    Write-Error "build failed (exit $LASTEXITCODE). Review output above."
    exit 1
}

$built = Find-InHrrBuild $BuildOut
if (-not $built) {
    Write-Error "Build completed but hrr-playback.exe not found under $BuildOut"
    exit 1
}

Write-Host "[ensure_playback] built: $built" -ForegroundColor DarkGray
Write-Host $built
exit 0
