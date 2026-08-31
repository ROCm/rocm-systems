# Run a bash script on the AIRUNTIME-28 MI450 target and strip the Conductor login banner.
#   .\rsh.ps1 -ScriptFile remote/foo.sh
#   .\rsh.ps1 -Command "hostname; rocminfo | head"
#
# The body is shipped base64-encoded because PowerShell rewrites LF to CRLF when it
# pipes a string into a native process' stdin, which bash then chokes on.
#
# The target defaults to $env:AIRUNTIME28_HOST so the machine name lives in one
# place; override per call with -Target. See REPRODUCE.md for the current host.
param(
    [string]$ScriptFile,
    [string]$Command,
    [string]$Target = $env:AIRUNTIME28_HOST,
    [int]$TimeoutSec = 20
)

$ErrorActionPreference = "Continue"

if (-not $Target) {
    throw "No target host. Set AIRUNTIME28_HOST (user@host) or pass -Target."
}

$sshopts = @(
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=$TimeoutSec",
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=NUL",
    "-o", "LogLevel=ERROR",
    "-o", "ServerAliveInterval=30"
)

# The target prints a large ASCII-art Conductor banner on every login. None of it
# goes to a separate stream, so it has to be filtered out by content.
# Matched against content, so it has to be narrow enough not to eat real output.
# An earlier version filtered any line starting with whitespace and a dot, which
# silently swallowed every printed "  ./script.sh" in a command's own output.
$bannerPattern = '^\s*(\$|\\|\||_|-{4}|`)|Conductor|conductor\.amd\.com|Checking authorization|Reminder:|SSH key is required|denied access to this system|You did not provide|You do not have an active|You are not in the required|other help resources|^\s+\.[-_\\/|`\s]'

if ($ScriptFile) {
    if (-not (Test-Path $ScriptFile)) { throw "Script file not found: $ScriptFile" }
    $body = Get-Content -Raw $ScriptFile
} elseif ($Command) {
    $body = $Command
} else {
    throw "Provide -ScriptFile or -Command"
}

$body = $body -replace "`r", ""
$b64 = [Convert]::ToBase64String([Text.Encoding]::UTF8.GetBytes($body))

ssh @sshopts $Target "echo $b64 | base64 -d | bash" 2>&1 |
    ForEach-Object { $_.ToString() } |
    Where-Object { $_ -notmatch $bannerPattern }
