#!/usr/bin/env pwsh
# HRR triage: optional GPU replay + structured finding.  Windows port of triage_archive.sh.
# Requires: PowerShell 5.1+, Python 3.x, hrr-playback.exe (HIP SDK for Windows or local build).
#
# Usage:
#   .\triage_archive.ps1 --archive <pid-dir> [--replay [native|auto]] [--no-replay]
#                        [-o <path>] [--format markdown|json] [-h]
#
# Environment knobs (mirror the Linux version):
#   HRR_TRIAGE_WORKDIR   Output dir for findings + logs (default: current dir)
#   HRR_PLAYBACK         Explicit path to hrr-playback.exe
#   HIP_PATH / ROCM_PATH HIP SDK root (default: C:\Program Files\AMD\ROCm\6.2)
#   GPU                  Replay GPU ordinal (default: 0)
#   HRR_CONTINUE=1       Proceed after preflight HIP/comgr version mismatch
#   HRR_SKIP_COMPAT=1    Skip manifest preflight entirely
#   HRR_STRICT_VERSION=1 Turn version mismatch into a hard block
#   HRR_STRICT_ARCH=1    Turn arch mismatch into a hard block
#
# NOTE: Docker GPU replay is NOT supported by this script on Windows.
#       --replay docker prints guidance and exits 1.
#       For Docker replay on Windows, use WSL2 + replay_docker.sh from WSL2.
# ---------------------------------------------------------------------------
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Declare variables — no param() block so $args is always the raw token list
$Archive   = ""
$Replay    = ""
$NoReplay  = $false
$Output    = ""
$Format    = "markdown"
$Help      = $false

$ScriptDir  = $PSScriptRoot
$Analyzer   = Join-Path $ScriptDir "analyze_replay_finding.py"
$EnsurePs1  = Join-Path $ScriptDir "ensure_playback.ps1"
$Compat     = Join-Path $ScriptDir "check_replay_compat.py"

# Resolve HIP/ROCm root: HIP_PATH > ROCM_PATH > default install
$HipRoot = if ($env:HIP_PATH)   { $env:HIP_PATH }
           elseif ($env:ROCM_PATH) { $env:ROCM_PATH }
           else { "C:\Program Files\AMD\ROCm\6.2" }

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
function Write-Triage([string]$Msg, [string]$Color = "DarkGray") {
    Write-Host "[triage] $Msg" -ForegroundColor $Color
}

function Show-Usage {
    @"
usage: triage_archive.ps1 --archive <pid-dir> [options]

Options:
  --archive PATH     pid-* HRR archive directory  (required)
  --replay [MODE]    Replay mode: native (default), auto, or docker
                       docker: prints WSL2 guidance and exits 1
  --no-replay        Metadata / --info only (no GPU replay)
  -o PATH            Write finding to PATH
  --format FORMAT    markdown (default) or json
  -h / --help        Show this help

Environment:
  HRR_TRIAGE_WORKDIR   Output dir for findings + logs
  HRR_PLAYBACK         Explicit hrr-playback.exe path
  HIP_PATH / ROCM_PATH HIP SDK root (default: C:\Program Files\AMD\ROCm\6.2)
  GPU                  Replay GPU ordinal (default: 0)
  HRR_CONTINUE=1       Skip confirmation on HIP/comgr mismatch
  HRR_SKIP_COMPAT=1    Skip manifest preflight entirely
"@ | Write-Host -ForegroundColor Cyan
}

# Parse --long-style flags directly from $args (no param() block — avoids PS binder interference)
$i = 0
while ($i -lt $args.Count) {
    $tok = $args[$i]
    switch -Regex ($tok) {
        '^(-h|--help)$'      { $Help     = $true; $i++ }
        '^(--archive|-a)$'   { $Archive  = $args[++$i]; $i++ }
        '^--no-replay$'      { $NoReplay = $true; $i++ }
        '^--replay$'         {
            if (($i + 1 -lt $args.Count) -and $args[$i+1] -notmatch '^-') {
                $Replay = $args[++$i]
            } else {
                $Replay = "native"
            }
            $i++
        }
        '^(-o|--output)$'    { $Output   = $args[++$i]; $i++ }
        '^--format$'         { $Format   = $args[++$i]; $i++ }
        default              { Write-Host "error: unknown arg: $tok" -ForegroundColor Red; exit 1 }
    }
}

if ($Help) { Show-Usage; exit 0 }

# Handle --no-replay flag
if ($NoReplay) { $Replay = "skip" }
# Default replay mode
if (-not $Replay) { $Replay = "auto" }

# Validate archive
if (-not $Archive) {
    Write-Host "error: --archive required" -ForegroundColor Red
    Show-Usage
    exit 1
}
try {
    $Archive = (Resolve-Path $Archive -ErrorAction Stop).Path
} catch {
    Write-Host "error: archive not found: $Archive" -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $Archive -PathType Container)) {
    Write-Host "error: archive is not a directory: $Archive" -ForegroundColor Red
    exit 1
}

$Name    = Split-Path $Archive -Leaf
$Ts      = (Get-Date).ToUniversalTime().ToString("yyyyMMdd'T'HHmmss'Z'")
$Workdir = if ($env:HRR_TRIAGE_WORKDIR) { $env:HRR_TRIAGE_WORKDIR } else { (Get-Location).Path }
New-Item -ItemType Directory -Force -Path $Workdir | Out-Null
$Ext     = if ($Format -eq "json") { ".finding.json" } else { ".finding.md" }
$Finding = if ($Output) { $Output } else { Join-Path $Workdir "${Name}-${Ts}${Ext}" }
$Log     = ""

# ---------------------------------------------------------------------------
# GPU tool discovery
# ---------------------------------------------------------------------------
function Find-RocmExe([string]$BaseName) {
    foreach ($dir in @(
        (Join-Path $HipRoot "bin"),
        (Join-Path $HipRoot "hip\bin"),
        (Join-Path $HipRoot "tools")
    )) {
        foreach ($ext in @(".exe", "")) {
            $p = Join-Path $dir "$BaseName$ext"
            if (Test-Path $p) { return $p }
        }
    }
    $cmd = Get-Command "$BaseName.exe" -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    $cmd = Get-Command $BaseName -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    return $null
}

function Get-BestGpu {
    # Honour explicit override
    if ($env:GPU) { return $env:GPU }

    # Try amd-smi (newer HIP SDK), then rocm-smi
    $smi = Find-RocmExe "amd-smi"
    if (-not $smi) { $smi = Find-RocmExe "rocm-smi" }

    if ($smi) {
        try {
            # amd-smi list --format csv gives GPU indices
            $out = & $smi list 2>&1 | Out-String
            $gpus = [regex]::Matches($out, 'GPU\s*(\d+)') | ForEach-Object { [int]$_.Groups[1].Value }
            if ($gpus.Count -gt 0) {
                $best = $gpus | Sort-Object | Select-Object -First 1
                Write-Triage "GPU $best (first visible via $smi)"
                return "$best"
            }
        } catch {}
    }

    Write-Triage "GPU 0 (default)"
    return "0"
}

# ---------------------------------------------------------------------------
# Replay mode resolution
# ---------------------------------------------------------------------------
function Resolve-ReplayMode {
    if ($Replay -ne "auto") { return $Replay }
    if ($env:HRR_DOCKER_IMAGE) {
        Write-Triage "HRR_DOCKER_IMAGE is set; Docker GPU passthrough is not supported on Windows." "Yellow"
        Write-Triage "Tip: run replay_docker.sh from WSL2 for Docker replay." "Yellow"
    }
    return "native"
}

# ---------------------------------------------------------------------------
# Manifest preflight (check_replay_compat.py)
# ---------------------------------------------------------------------------
function Invoke-ReplayPreflight([string]$Mode, [string]$Gpu) {
    if (-not (Test-Path $Compat)) { return }
    if ($env:HRR_SKIP_COMPAT -eq "1") {
        Write-Triage "skipping preflight (HRR_SKIP_COMPAT=1)"
        return
    }

    $CompatArgs = @($Compat, "--archive", $Archive, "--gpu", $Gpu, "--mode", "host")
    if ($env:HRR_STRICT_VERSION -eq "1") { $CompatArgs += "--strict-version" }
    if ($env:HRR_STRICT_ARCH    -eq "1") { $CompatArgs += "--strict-arch"    }

    Write-Triage "replay preflight (manifest metadata)"
    $rc = 0
    try {
        python @CompatArgs
        $rc = $LASTEXITCODE
    } catch {
        $rc = 1
    }

    if ($rc -eq 2) {
        if ($env:HRR_CONTINUE -eq "1") {
            Write-Triage "continuing despite version mismatch (HRR_CONTINUE=1)" "Yellow"
            return
        }
        $ans = Read-Host "Version mismatch detected. Continue? [y/N]"
        if ($ans -match '^[Yy]$') {
            Write-Triage "continuing after confirmation"
            return
        }
        Write-Host "[triage] aborted at version mismatch." -ForegroundColor Red
        exit 3
    }
    if ($rc -ne 0) { exit $rc }
}

# ---------------------------------------------------------------------------
# Native replay
# ---------------------------------------------------------------------------
function Invoke-NativeReplay([string]$PlaybackExe, [string]$LogPath, [string]$Gpu) {
    Write-Triage "native replay  playback=$PlaybackExe  GPU=$Gpu"

    $env:HIP_VISIBLE_DEVICES             = $Gpu
    $env:HIP_HRR_REPLAY_PROGRESS_SECONDS = if ($env:HIP_HRR_REPLAY_PROGRESS_SECONDS) {
                                               $env:HIP_HRR_REPLAY_PROGRESS_SECONDS
                                           } else { "30" }

    # WHY Start-Process instead of "& exe 2>&1 | Tee-Object":
    #   PowerShell's 2>&1 converts native-process stderr into ErrorRecord objects.
    #   Tee-Object writes them with an "hrr-playback.exe : " prefix and may split
    #   long lines — breaking every regex in analyze_replay_finding.py.
    #   cmd /c avoids that but has its own quoting/exit-code quirks with PowerShell.
    #   Start-Process with file-level redirection bypasses PowerShell's pipeline
    #   entirely: stdout and stderr arrive as plain UTF-8 text, no wrapping, no
    #   prefix, no splitting.
    $tmpOut = [System.IO.Path]::GetTempFileName()
    $tmpErr = [System.IO.Path]::GetTempFileName()
    $rc = -1

    try {
        Write-Triage "log=$LogPath"
        # Strip trailing backslash: "path\" in a quoted Win32 arg is parsed as
        # "path" + escaped-quote, corrupting the path (pid-14372" \events.bin).
        $archArg = $Archive.TrimEnd('\')
        $proc = Start-Process `
            -FilePath               $PlaybackExe `
            -ArgumentList           "`"$archArg`"" `
            -RedirectStandardOutput $tmpOut `
            -RedirectStandardError  $tmpErr `
            -NoNewWindow `
            -Wait `
            -PassThru
        $rc = $proc.ExitCode

        # stderr has [HRR progress] lines; stdout has [HRR] PASS / archive stats.
        # Merge both into the log (stderr first = chronological order).
        $errText = [System.IO.File]::ReadAllText($tmpErr)
        $outText = [System.IO.File]::ReadAllText($tmpOut)
        $combined = $errText + $outText
        [System.IO.File]::WriteAllText($LogPath, $combined, [System.Text.Encoding]::UTF8)

        # Echo to console so progress/result are visible
        if ($errText) { Write-Host $errText.TrimEnd() -ForegroundColor DarkGray }
        if ($outText) { Write-Host $outText.TrimEnd() }

    } finally {
        Remove-Item $tmpOut, $tmpErr -ErrorAction SilentlyContinue
    }

    $logBytes = (Get-Item $LogPath -ErrorAction SilentlyContinue).Length
    Write-Triage "log size=$logBytes bytes"
    Write-Triage "native replay exit=$rc"
    return $rc
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
$Mode = Resolve-ReplayMode
Write-Triage "archive=$Archive  replay=$Mode"

# Docker replay via WSL2.
# Docker Desktop on Windows cannot pass --device=/dev/kfd to a container directly.
# Instead we invoke replay_docker.sh inside the WSL2 Linux kernel where /dev/kfd
# is accessible, translating the Windows archive path to its /mnt/ WSL2 equivalent.
if ($Mode -eq "docker") {
    # Require WSL2
    if (-not (Get-Command wsl -ErrorAction SilentlyContinue)) {
        Write-Host @"
[triage] Docker replay requires WSL2, which is not installed.

Install WSL2 first:
  wsl --install -d Ubuntu-24.04

Then re-run with --replay docker (or set HRR_DOCKER_IMAGE and use --replay auto).
"@ -ForegroundColor Yellow
        exit 1
    }

    # Require HRR_DOCKER_IMAGE
    if (-not $env:HRR_DOCKER_IMAGE) {
        Write-Host "error: set HRR_DOCKER_IMAGE to the capture container image before using --replay docker" -ForegroundColor Red
        exit 1
    }

    # Convert Windows path (C:\foo\bar) -> WSL2 /mnt/c/foo/bar
    function ConvertTo-WslPath([string]$WinPath) {
        $p = $WinPath.TrimEnd('\').Replace('\', '/')
        if ($p -match '^([A-Za-z]):(.*)') {
            return "/mnt/$($Matches[1].ToLower())$($Matches[2])"
        }
        return $p   # already a UNC or relative path — pass through
    }

    $wslArchive  = ConvertTo-WslPath $Archive
    $wslWorkdir  = ConvertTo-WslPath $Workdir
    $wslScript   = ConvertTo-WslPath (Join-Path $ScriptDir "replay_docker.sh")
    $Log         = Join-Path $Workdir "hrr-replay-${Name}-${Ts}.log"
    $wslLog      = ConvertTo-WslPath $Log

    Write-Triage "docker replay via WSL2  image=$($env:HRR_DOCKER_IMAGE)"
    Write-Triage "archive (WSL2)=$wslArchive"
    Write-Triage "log=$Log"

    # Build the bash command that WSL2 will run
    $mountClr = if ($env:HRR_DOCKER_MOUNT_CLR -eq "1") { "HRR_DOCKER_MOUNT_CLR=1 " } else { "" }
    $wslCmd = "HRR_DOCKER_IMAGE='$($env:HRR_DOCKER_IMAGE)' ${mountClr}bash '$wslScript' --archive '$wslArchive' --log '$wslLog'"

    Write-Triage "wsl bash: $wslCmd"
    wsl -- bash -c $wslCmd
    $rc = $LASTEXITCODE
    Write-Triage "docker replay exit=$rc"

    if ($rc -ne 0) {
        Write-Host "[triage] Docker replay failed (exit $rc). Common causes:" -ForegroundColor Yellow
        Write-Host "  - /dev/kfd not accessible in WSL2 (needs AMD GPU driver >= 21.40 + Windows 11 build >= 22000)" -ForegroundColor Yellow
        Write-Host "  - hrr-playback not in image: set HRR_DOCKER_MOUNT_CLR=1 with CLR_BUILD pointing at a Linux build" -ForegroundColor Yellow
        Write-Host "  - Image not pulled: docker pull `$HRR_DOCKER_IMAGE" -ForegroundColor Yellow
    }
    # Fall through to analyzer so we still parse whatever log was written
}

# Locate or build hrr-playback.exe
$HrrPlayback = $env:HRR_PLAYBACK

if ($Mode -ne "skip") {
    if (-not $HrrPlayback -and (Test-Path $EnsurePs1)) {
        Write-Triage "running ensure_playback.ps1 --build"
        try {
            $HrrPlayback = (& pwsh -NoProfile -NonInteractive -File $EnsurePs1 --build 2>&1 |
                            Where-Object { $_ -notmatch '^\[' } |
                            Select-Object -Last 1).Trim()
        } catch {
            Write-Host "error: ensure_playback.ps1 --build failed. Set HRR_PLAYBACK or CLR_BUILD." -ForegroundColor Red
            exit 1
        }
    }

    # Fallback: search HIP SDK bin
    if (-not $HrrPlayback) {
        $HrrPlayback = Find-RocmExe "hrr-playback"
    }

    if (-not $HrrPlayback -or -not (Test-Path $HrrPlayback)) {
        Write-Host @"
error: hrr-playback.exe not found.

Options:
  1. Set HRR_PLAYBACK=C:\path\to\hrr-playback.exe
  2. Set CLR_BUILD=C:\path\to\build-dir  (must contain a built hrr-playback.exe)
  3. Run ensure_playback.ps1 --build   (builds from source; requires CMake + Ninja + HIP SDK)
  4. Install HIP SDK and ensure hrr-playback.exe is in PATH or HIP_PATH\bin
"@ -ForegroundColor Red
        exit 1
    }
}

# GPU selection + preflight
$ReplayGpu = Get-BestGpu
if ($Mode -ne "skip") {
    Invoke-ReplayPreflight -Mode $Mode -Gpu $ReplayGpu
}

# GPU replay
if ($Mode -eq "native") {
    $Log = Join-Path $Workdir "hrr-replay-${Name}-${Ts}.log"
    Invoke-NativeReplay -PlaybackExe $HrrPlayback -LogPath $Log -Gpu $ReplayGpu | Out-Null
}

# Python analyzer
$AnalyzerArgs = @($Analyzer, "--format", $Format, "--archive", $Archive, "-o", $Finding)
if ($HrrPlayback)                        { $AnalyzerArgs += @("--hrr-playback", $HrrPlayback) }
if ($Log) {
    if (Test-Path $Log -PathType Leaf) {
        $logBytes = (Get-Item $Log).Length
        Write-Triage "passing log to analyzer: $Log ($logBytes bytes)"
        $AnalyzerArgs += @("--log", $Log)
    } else {
        Write-Triage "WARNING: log file not found, skipping --log: $Log" "Yellow"
    }
}

Write-Triage "analyzer: python $($AnalyzerArgs -join ' ')"
python @AnalyzerArgs
if ($LASTEXITCODE -ne 0) {
    Write-Host "error: analyze_replay_finding.py failed (exit $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

Write-Triage "finding=$Finding"
Get-Content $Finding
