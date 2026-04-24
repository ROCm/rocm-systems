# build-bundled-opencode.ps1 - apply perfxpert patches + build opencode on Windows.
#
# This is the native Windows counterpart to build-bundled-opencode.sh. It builds
# the pinned opencode submodule and installs the patched binary into
# perfxpert/_bundled/opencode.exe so perfxpert-code works after pip install.
#
# Usage:
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-bundled-opencode.ps1
#   powershell -NoProfile -ExecutionPolicy Bypass -File scripts/build-bundled-opencode.ps1 --skip-install

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

function Write-Info($Message) {
    Write-Host "build-bundled-opencode: $Message"
}

function Write-Fail($Message, [int]$Code = 2) {
    [Console]::Error.WriteLine("build-bundled-opencode: $Message")
    exit $Code
}

$SkipInstall = $false
foreach ($Arg in $args) {
    switch ($Arg) {
        "--skip-install" { $SkipInstall = $true }
        "-h" {
            Get-Content -Path $PSCommandPath -TotalCount 20
            exit 0
        }
        "--help" {
            Get-Content -Path $PSCommandPath -TotalCount 20
            exit 0
        }
        default {
            Write-Fail "unknown argument: $Arg"
        }
    }
}

function Get-Sha256Lower([string]$Path) {
    return (Get-FileHash -Algorithm SHA256 -Path $Path).Hash.ToLowerInvariant()
}

function Test-GitDiffQuiet([string]$WorkingDirectory) {
    $OldLocation = Get-Location
    try {
        Set-Location -Path $WorkingDirectory
        & git diff --quiet HEAD --
        return ($LASTEXITCODE -eq 0)
    }
    finally {
        Set-Location -Path $OldLocation
    }
}

function Apply-OpencodePatches([string]$PerfXpertRoot, [string]$OpencodeDir) {
    $PatchDir = if ($env:PERFXPERT_PATCH_DIR) { $env:PERFXPERT_PATCH_DIR } else { Join-Path $PerfXpertRoot ".patches" }
    $PatchManifest = if ($env:PERFXPERT_PATCH_MANIFEST) { $env:PERFXPERT_PATCH_MANIFEST } else { Join-Path $PatchDir "SHA256SUMS" }
    $CheckOnly = $env:PERFXPERT_PATCH_CHECK_ONLY

    if (-not (Test-Path -Path $PatchDir -PathType Container)) {
        Write-Fail "patch dir not found: $PatchDir"
    }
    if (-not (Test-Path -Path $PatchManifest -PathType Leaf)) {
        Write-Fail "patch manifest not found: $PatchManifest"
    }

    $Patches = @(Get-ChildItem -Path $PatchDir -Filter "*.patch" -File | Sort-Object Name)
    if ($Patches.Count -eq 0) {
        Write-Info "no patches found in $PatchDir - nothing to do"
        return
    }

    $ManifestEntries = @()
    foreach ($Line in Get-Content -Path $PatchManifest) {
        $Trimmed = $Line.Trim()
        if (-not $Trimmed) {
            continue
        }
        $Parts = $Trimmed -split "\s+", 3
        if ($Parts.Count -lt 2) {
            Write-Fail "malformed patch manifest line: $Line" 1
        }
        $ManifestEntries += [pscustomobject]@{
            Hash = $Parts[0].ToLowerInvariant()
            RelPath = $Parts[1]
            Name = Split-Path -Leaf $Parts[1]
        }
    }
    if ($ManifestEntries.Count -eq 0) {
        Write-Fail "patch manifest is empty: $PatchManifest"
    }

    $PatchNames = @($Patches | ForEach-Object { $_.Name })
    $ManifestNames = @($ManifestEntries | ForEach-Object { $_.Name })
    foreach ($PatchName in $PatchNames) {
        if ($ManifestNames -notcontains $PatchName) {
            Write-Fail "manifest missing entry for $PatchName" 1
        }
    }
    foreach ($Entry in $ManifestEntries) {
        if ($PatchNames -notcontains $Entry.Name) {
            Write-Fail "manifest references missing patch $($Entry.Name)" 1
        }
        $PatchPath = Join-Path $PatchDir $Entry.RelPath
        $Actual = Get-Sha256Lower $PatchPath
        if ($Actual -ne $Entry.Hash) {
            Write-Fail "checksum verification failed for $($Entry.RelPath)" 1
        }
    }
    if ($ManifestEntries.Count -ne $Patches.Count) {
        Write-Fail "manifest/patch set mismatch in $PatchDir" 1
    }

    Write-Info "$($Patches.Count) patches to apply (check_only=$CheckOnly)"
    Write-Info "manifest OK - $PatchManifest"

    if (-not (Test-GitDiffQuiet $OpencodeDir)) {
        Write-Fail "refusing to run: submodule has uncommitted changes. Run: git -C `"$OpencodeDir`" checkout HEAD -- ." 1
    }

    $Applied = @()
    foreach ($Patch in $Patches) {
        Write-Info "checking $($Patch.Name)"
        & git -C $OpencodeDir apply --check $Patch.FullName
        if ($LASTEXITCODE -ne 0) {
            & git -C $OpencodeDir checkout HEAD -- . | Out-Null
            Write-Fail "FAILED --check for $($Patch.Name); already-applied patches: $($Applied -join ', ')" 1
        }
        Write-Info "applying $($Patch.Name)"
        & git -C $OpencodeDir apply $Patch.FullName
        if ($LASTEXITCODE -ne 0) {
            & git -C $OpencodeDir checkout HEAD -- . | Out-Null
            Write-Fail "FAILED apply for $($Patch.Name)" 1
        }
        $Applied += $Patch.Name
    }

    if ($CheckOnly -eq "1" -or $CheckOnly -eq "true") {
        & git -C $OpencodeDir checkout HEAD -- . | Out-Null
        Write-Info "CHECK OK - $($Patches.Count) patch(es) verified (reverted)"
    }
    else {
        Write-Info "OK - $($Patches.Count) patch(es) applied"
    }
}

$ScriptDir = Split-Path -Parent $PSCommandPath
$PerfXpertRoot = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$Submodule = if ($env:PERFXPERT_OPENCODE_DIR) { $env:PERFXPERT_OPENCODE_DIR } else { Join-Path $PerfXpertRoot "opencode" }
$BundledDir = if ($env:PERFXPERT_BUNDLED_DIR) { $env:PERFXPERT_BUNDLED_DIR } else { Join-Path $PerfXpertRoot "perfxpert\_bundled" }
$Output = Join-Path $BundledDir "opencode.exe"

if (-not (Test-Path -Path (Join-Path $Submodule "package.json") -PathType Leaf)) {
    Write-Fail "opencode submodule missing at $Submodule. Run: git submodule update --init --recursive"
}

if (-not (Get-Command bun -ErrorAction SilentlyContinue)) {
    Write-Fail "bun is required to compile opencode. Install bun, then re-run: perfxpert-code install-patches"
}

Write-Info "bun=$((bun --version).Trim()), submodule=$Submodule"

& git -C $Submodule rev-parse --show-toplevel | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Fail "$Submodule is not a git checkout; refusing to build without repo-pinned submodule metadata"
}

if (Test-GitDiffQuiet $Submodule) {
    Write-Info "applying $PerfXpertRoot\.patches\*.patch"
    Apply-OpencodePatches $PerfXpertRoot $Submodule
}
else {
    Write-Info "submodule already patched (dirty tree) - skipping apply"
}

if ((-not (Test-Path -Path (Join-Path $Submodule "bun.lock") -PathType Leaf)) -and (-not (Test-Path -Path (Join-Path $Submodule "bun.lockb") -PathType Leaf))) {
    Write-Fail "missing bun.lock/bun.lockb in $Submodule"
}

$NodeModules = Join-Path $Submodule "node_modules"
if ((-not (Test-Path -Path $NodeModules -PathType Container)) -or $env:PERFXPERT_FORCE_INSTALL -eq "1") {
    Write-Info "running 'bun install --frozen-lockfile --ignore-scripts' at $Submodule"
    Push-Location $Submodule
    try {
        & bun install --frozen-lockfile --ignore-scripts
        if ($LASTEXITCODE -ne 0) {
            [Console]::Error.WriteLine("build-bundled-opencode: frozen lockfile install failed; retrying 'bun install --ignore-scripts'")
            & bun install --ignore-scripts
            if ($LASTEXITCODE -ne 0) {
                Write-Fail "'bun install --ignore-scripts' exited $LASTEXITCODE" $LASTEXITCODE
            }
        }
        Write-Info "running explicit postinstall 'bun run --cwd packages/opencode fix-node-pty'"
        & bun run --cwd packages/opencode fix-node-pty
        if ($LASTEXITCODE -ne 0) {
            Write-Fail "'bun run --cwd packages/opencode fix-node-pty' exited $LASTEXITCODE" $LASTEXITCODE
        }
    }
    finally {
        Pop-Location
    }
}
else {
    Write-Info "node_modules/ present - skipping 'bun install'"
}

$OpencodePackage = Join-Path $Submodule "packages\opencode"
Write-Info "compiling opencode (bun run build --single)"
Push-Location $OpencodePackage
try {
    & bun run build --single
    if ($LASTEXITCODE -ne 0) {
        Write-Fail "'bun run build --single' exited $LASTEXITCODE" $LASTEXITCODE
    }
}
finally {
    Pop-Location
}

$Dist = Join-Path $Submodule "packages\opencode\dist"
$Candidates = @(
    Join-Path $Dist "opencode-windows-x64\bin\opencode.exe",
    Join-Path $Dist "opencode-windows-x64\bin\opencode",
    Join-Path $Dist "opencode-windows-x64-baseline\bin\opencode.exe",
    Join-Path $Dist "opencode-windows-x64-baseline\bin\opencode",
    Join-Path $Dist "opencode-windows-arm64\bin\opencode.exe",
    Join-Path $Dist "opencode-windows-arm64\bin\opencode",
    Join-Path $Dist "opencode\bin\opencode.exe",
    Join-Path $Dist "opencode\bin\opencode"
)

$Built = $null
foreach ($Candidate in $Candidates) {
    if (Test-Path -Path $Candidate -PathType Leaf) {
        $Built = $Candidate
        break
    }
}

if ($null -eq $Built) {
    [Console]::Error.WriteLine("build-bundled-opencode: ERROR - no built Windows binary found under packages/opencode/dist/")
    if (Test-Path -Path $Dist) {
        Get-ChildItem -Path $Dist -Recurse -File -Filter "opencode*" | ForEach-Object { [Console]::Error.WriteLine($_.FullName) }
    }
    exit 3
}

$BuiltInfo = Get-Item -Path $Built
Write-Info "built $Built ($($BuiltInfo.Length) bytes)"

if ($SkipInstall) {
    Write-Info "--skip-install set; not copying to $Output"
    exit 0
}

New-Item -ItemType Directory -Path $BundledDir -Force | Out-Null
Copy-Item -Path $Built -Destination $Output -Force
Write-Info "bundled -> $Output"
Write-Info "sha256=$(Get-Sha256Lower $Output)"
