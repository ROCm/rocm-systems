# Normalise text files to UTF-8/LF, verify, and push them to the target.
#
# The editor on this machine writes UTF-16LE, sometimes with a BOM and sometimes
# without, and PowerShell rewrites LF to CRLF when piping to a native process.
# Bash and hipcc both reject the results, in ways whose error messages point
# nowhere near the encoding. Doing this in one place means no future edit has to
# remember, and the verification pass means a file that cannot be normalised stops
# the push instead of failing confusingly on the target.
#
#   .\sync.ps1                normalise, verify, push
#   .\sync.ps1 -NoPush        normalise and verify only
#
# The target comes from $env:AIRUNTIME28_HOST; override with -Target.
param(
    [switch]$NoPush,
    [string]$Target = $env:AIRUNTIME28_HOST,
    [string]$RemoteDir = "~/airuntime28"
)

$ErrorActionPreference = "Stop"

$extensions = @('.sh', '.cc', '.hip', '.h', '.hpp', '.cpp', '.py', '.md', '.ps1', '.tsv', '.gitattributes')

function Get-TextFiles {
    Get-ChildItem -Recurse -File |
        Where-Object { $_.Extension -in $extensions -and $_.FullName -notmatch '\\(results|isa|build)\\' }
}

# --- normalise --------------------------------------------------------------
$converted = 0
foreach ($f in Get-TextFiles) {
    $bytes = [IO.File]::ReadAllBytes($f.FullName)
    if ($bytes.Length -lt 2) { continue }

    # Two shapes to catch: a UTF-16LE BOM (FF FE), and BOM-less UTF-16LE, which
    # shows up as a zero in the second byte of otherwise-ASCII text.
    $hasBom  = ($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE)
    $isUtf16 = $hasBom -or ($bytes[1] -eq 0)

    $text = if ($isUtf16) {
        [IO.File]::ReadAllText($f.FullName, [Text.Encoding]::Unicode)
    } else {
        [IO.File]::ReadAllText($f.FullName, (New-Object Text.UTF8Encoding $false))
    }
    $normalised = ($text -replace "`r`n", "`n" -replace "`r", "`n").TrimStart([char]0xFEFF)

    if ($isUtf16 -or $normalised -ne $text) {
        [IO.File]::WriteAllText($f.FullName, $normalised, (New-Object Text.UTF8Encoding $false))
        $converted++
    }
}
Write-Host "normalised $converted file(s)"

# --- verify -----------------------------------------------------------------
# Cheap, and it is the difference between catching this here and debugging a
# syntax error on the target.
$bad = @()
foreach ($f in Get-TextFiles) {
    $bytes = [IO.File]::ReadAllBytes($f.FullName)
    if ($bytes.Length -lt 2) { continue }
    $reasons = @()
    if ($bytes -contains 0)    { $reasons += "contains NUL (still UTF-16?)" }
    if ($bytes -contains 13)   { $reasons += "contains CR" }
    if ($bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB) { $reasons += "has a UTF-8 BOM" }
    if ($reasons) { $bad += "  $($f.FullName.Replace("$PWD\", '')): $($reasons -join ', ')" }
}
if ($bad) {
    Write-Host "ENCODING CHECK FAILED - not pushing:" -ForegroundColor Red
    $bad | ForEach-Object { Write-Host $_ -ForegroundColor Red }
    exit 1
}
Write-Host "encoding check passed (UTF-8, LF, no BOM)"

if ($NoPush) { return }

# --- push -------------------------------------------------------------------
if (-not $Target) {
    throw "No target host. Set AIRUNTIME28_HOST (user@host) or pass -Target."
}

$sshopts = @('-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=no',
             '-o', 'UserKnownHostsFile=NUL', '-o', 'LogLevel=ERROR')

& ssh @sshopts $Target "mkdir -p $RemoteDir/src/common $RemoteDir/src/experiments $RemoteDir/build $RemoteDir/isa $RemoteDir/results" 2>&1 |
    Where-Object { $_ -notmatch 'Conductor|Checking authorization|^\s*$' } | Out-Null

foreach ($pair in @(
        @('src\common\*',      "$RemoteDir/src/common/"),
        @('src\experiments\*', "$RemoteDir/src/experiments/"),
        @('remote\*',          "$RemoteDir/"))) {
    if (Test-Path $pair[0]) {
        & scp @sshopts -q $pair[0] "${Target}:$($pair[1])" 2>&1 |
            Where-Object { $_ -notmatch '^\s*$' }
    }
}
Write-Host "pushed to ${Target}:$RemoteDir"
