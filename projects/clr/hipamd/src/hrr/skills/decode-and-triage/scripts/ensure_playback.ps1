#!/usr/bin/env pwsh
# Locate hrr-playback.exe; optionally build from the colocated CLR tree (--build).
# Prints the absolute path to stdout on success.
#
# Usage:
#   .\ensure_playback.ps1 [--build]
#
# Environment:
#   HRR_PLAYBACK         Explicit binary path (checked first)
#   CLR_BUILD            Existing build dir (hipamd\src\hrr\playback\hrr-playback.exe expected inside)
#   CLR_ROOT             CLR source root (overrides auto-detection)
#   HIP_PATH / ROCM_PATH HIP SDK root for -DHIP_COMMON_DIR / -DROCM_PATH (default: C:\Program Files\AMD\ROCm\6.2)
#   ROCR_INC             Optional ROCR include dir for -DHSA_HEADER_DIR
#   HRR_ROOT             Explicit repo root (overrides git detection)
#   HRR_ENSURE_BUILD=1   Same as passing --build
# ---------------------------------------------------------------------------
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Declare variables — no param() block so $args is always the raw token list
$Build = $false
$Help  = $false

$ScriptDir = $PSScriptRoot
$SkillDir  = Split-Path $ScriptDir -Parent    # decode-and-triage/
$HipRoot   = if ($env:HIP_PATH)    { $env:HIP_PATH }
             elseif ($env:ROCM_PATH) { $env:ROCM_PATH }
             else { "C:\Program Files\AMD\ROCm\6.2" }

# Parse flags from $args directly (no param() block — avoids PS binder interference)
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
  --build:  configure + compile hrr-playback when not found

Searched in order:
  HRR_PLAYBACK env, CLR_BUILD tree, HIP_PATH\bin, ROCM_PATH\bin, PATH,
  build-hrr* / build* dirs adjacent to CLR root.
"@ | Write-Host -ForegroundColor Cyan
    exit 0
}

if ($env:HRR_ENSURE_BUILD -eq "1") { $Build = $true }

# ---------------------------------------------------------------------------
# Find hrr-playback.exe inside a build tree
# ---------------------------------------------------------------------------
function Find-InBuild([string]$BuildDir) {
    if (-not $BuildDir) { return $null }
    foreach ($rel in @(
        "hipamd\src\hrr\playback\hrr-playback.exe",
        "hipamd\src\hrr\playback\hrr-playback",   # no-extension (rare)
        "bin\hrr-playback.exe",
        "bin\hrr-playback"
    )) {
        $p = Join-Path $BuildDir $rel
        if (Test-Path $p -PathType Leaf) { return (Resolve-Path $p).Path }
    }
    # Recursive search as a last resort (slower)
    $found = Get-ChildItem $BuildDir -Recurse -Filter "hrr-playback.exe" -ErrorAction SilentlyContinue |
             Select-Object -First 1
    if ($found) { return $found.FullName }
    return $null
}

# ---------------------------------------------------------------------------
# Locate the CLR source root (hipamd\src\hrr must be present)
# ---------------------------------------------------------------------------
function Resolve-ClrRoot {
    if ($env:CLR_ROOT) {
        $p = $env:CLR_ROOT
        if ((Test-Path (Join-Path $p "CMakeLists.txt")) -and
            (Test-Path (Join-Path $p "hipamd\src\hrr"))) {
            return $p
        }
    }

    # Walk up 5 levels from skill dir (decode-and-triage → skills → hrr → hipamd → clr)
    $candidate = $SkillDir
    for ($i = 0; $i -lt 5; $i++) {
        if (-not $candidate) { break }          # stop before Split-Path "" errors
        $candidate = Split-Path $candidate -Parent
        if (-not $candidate) { break }          # root exhausted (Split-Path "C:\" returns "")
        if ((Test-Path (Join-Path $candidate "CMakeLists.txt")) -and
            (Test-Path (Join-Path $candidate "hipamd\src\hrr"))) {
            return $candidate
        }
    }

    # Try git top-level / projects\clr
    try {
        $repo = & git -C $SkillDir rev-parse --show-toplevel 2>$null
        if ($repo) {
            foreach ($sub in @("projects\clr", "rocm-systems\projects\clr")) {
                $p = Join-Path $repo $sub
                if ((Test-Path (Join-Path $p "CMakeLists.txt")) -and
                    (Test-Path (Join-Path $p "hipamd\src\hrr"))) {
                    return $p
                }
            }
        }
    } catch {}

    return $null
}

# ---------------------------------------------------------------------------
# Probe known HIP SDK paths
# ---------------------------------------------------------------------------
function Find-InHipSdk {
    foreach ($dir in @(
        (Join-Path $HipRoot "bin"),
        (Join-Path $HipRoot "hip\bin"),
        (Join-Path $HipRoot "tools\bin")
    )) {
        foreach ($name in @("hrr-playback.exe", "hrr-playback")) {
            $p = Join-Path $dir $name
            if (Test-Path $p -PathType Leaf) { return $p }
        }
    }
    # PATH fallback
    $cmd = Get-Command "hrr-playback.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command "hrr-playback" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

# ---------------------------------------------------------------------------
# STEP 1 — Honour explicit HRR_PLAYBACK env
# ---------------------------------------------------------------------------
if ($env:HRR_PLAYBACK -and (Test-Path $env:HRR_PLAYBACK -PathType Leaf)) {
    Write-Host $env:HRR_PLAYBACK
    exit 0
}

# ---------------------------------------------------------------------------
# STEP 2 — Look in CLR_BUILD env
# ---------------------------------------------------------------------------
$ClrRoot = Resolve-ClrRoot

if ($env:CLR_BUILD) {
    $found = Find-InBuild $env:CLR_BUILD
    if ($found) { Write-Host $found; exit 0 }
}

# ---------------------------------------------------------------------------
# STEP 3 — Scan build-hrr* / build* dirs adjacent to the CLR root
# ---------------------------------------------------------------------------
if ($ClrRoot) {
    $searchRoot = Split-Path $ClrRoot -Parent
    $dirs = Get-ChildItem $searchRoot -Directory -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^build' } |
            Sort-Object { $_.LastWriteTime } -Descending

    foreach ($d in $dirs) {
        $found = Find-InBuild $d.FullName
        if ($found) { Write-Host $found; exit 0 }
    }

    # Also check inside CLR root itself (e.g. clr\build-win)
    $innerDirs = Get-ChildItem $ClrRoot -Directory -ErrorAction SilentlyContinue |
                 Where-Object { $_.Name -match '^build' } |
                 Sort-Object { $_.LastWriteTime } -Descending

    foreach ($d in $innerDirs) {
        $found = Find-InBuild $d.FullName
        if ($found) { Write-Host $found; exit 0 }
    }
}

# ---------------------------------------------------------------------------
# STEP 4 — HIP SDK install / PATH
# ---------------------------------------------------------------------------
$sdkFound = Find-InHipSdk
if ($sdkFound) { Write-Host $sdkFound; exit 0 }

# ---------------------------------------------------------------------------
# STEP 5 — Build (only when --build requested)
# ---------------------------------------------------------------------------
if (-not $Build) {
    Write-Host @"
[ensure_playback] hrr-playback.exe not found.

This script is running outside the rocm-systems source tree, so --build is not
available here.  Choose one of the options below:

  1. Point at an existing binary:
       `$env:HRR_PLAYBACK = 'C:\path\to\hrr-playback.exe'

  2. Point at a CLR build directory that already contains hrr-playback.exe:
       `$env:CLR_BUILD = 'C:\path\to\build-dir'

  3. Install the HIP SDK and ensure hrr-playback.exe is under HIP_PATH\bin:
       `$env:HIP_PATH = 'C:\Program Files\AMD\ROCm\6.2'

  4. Triage without replay (no binary needed — reads manifest.json only):
       .\triage_archive.ps1 --archive <pid-dir> --no-replay

  5. Build from source: clone rocm-systems, set CLR_ROOT, then re-run with --build:
       git clone https://github.com/ROCm/rocm-systems
       `$env:CLR_ROOT = 'C:\rocm-systems\projects\clr'
       .\ensure_playback.ps1 --build
"@ -ForegroundColor Yellow
    exit 1
}

if (-not $ClrRoot) {
    Write-Host @"
[ensure_playback] Cannot locate the CLR source root for --build.

This script is not inside a rocm-systems checkout, so it cannot find
hipamd\src\hrr\  (required for cmake configure).

Options:
  A. Set CLR_ROOT explicitly:
       `$env:CLR_ROOT = 'C:\path\to\rocm-systems\projects\clr'
       .\ensure_playback.ps1 --build

  B. Clone rocm-systems first:
       git clone https://github.com/ROCm/rocm-systems
       `$env:CLR_ROOT = 'C:\rocm-systems\projects\clr'
       .\ensure_playback.ps1 --build

  C. Skip the build — use an existing binary instead:
       `$env:HRR_PLAYBACK = 'C:\path\to\hrr-playback.exe'

  D. Metadata-only triage (no binary needed):
       .\triage_archive.ps1 --archive <pid-dir> --no-replay
"@ -ForegroundColor Yellow
    exit 1
}

$BuildOut = Join-Path (Split-Path $ClrRoot -Parent) "build-hrr-win"
New-Item -ItemType Directory -Force -Path $BuildOut | Out-Null

Write-Host "[ensure_playback] configuring in $BuildOut" -ForegroundColor DarkGray

# ---- CMake configure -------------------------------------------------------
# ROCM_PATH / HIP_PATH must point to the HIP SDK install that includes
# amdhip64.lib, hip\include, etc.
$CMakeArgs = @(
    "-S", $ClrRoot,
    "-B", $BuildOut,
    "-G", "Ninja",
    "-DHIP_COMMON_DIR=$(Join-Path $HipRoot 'include')",
    "-DROCM_PATH=$HipRoot",
    "-DCLR_BUILD_HIP=ON",
    "-DCLR_BUILD_OCL=OFF",
    "-DHIP_PLATFORM=amd",
    "-DCMAKE_BUILD_TYPE=Release"
)
if ($env:ROCR_INC) { $CMakeArgs += "-DHSA_HEADER_DIR=$($env:ROCR_INC)" }

Write-Host "[ensure_playback] cmake $($CMakeArgs -join ' ')" -ForegroundColor DarkGray
cmake @CMakeArgs
if ($LASTEXITCODE -ne 0) {
    Write-Error "CMake configure failed (exit $LASTEXITCODE). Review output above."
    exit 1
}

# ---- Ninja build -----------------------------------------------------------
$Jobs = if ($env:NUMBER_OF_PROCESSORS) { $env:NUMBER_OF_PROCESSORS } else { "4" }
Write-Host "[ensure_playback] ninja -C $BuildOut hrr-playback -j $Jobs" -ForegroundColor DarkGray
ninja -C $BuildOut hrr-playback -j $Jobs
if ($LASTEXITCODE -ne 0) {
    Write-Error "ninja build failed (exit $LASTEXITCODE). Review output above."
    exit 1
}

# ---- Return path -----------------------------------------------------------
$built = Find-InBuild $BuildOut
if (-not $built) {
    Write-Error "Build completed but hrr-playback.exe not found under $BuildOut"
    exit 1
}

Write-Host "[ensure_playback] built: $built" -ForegroundColor DarkGray
Write-Host $built
exit 0
