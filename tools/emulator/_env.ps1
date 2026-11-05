# Shared paths/env for the emulator scripts. Dot-source it: . "$PSScriptRoot\_env.ps1"
# Everything (SDK, AVDs, gradle cache, recordings) stays on F: because C: is almost full.

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
$SdkRoot = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { 'F:\dev\android-sdk' }
$env:ANDROID_HOME = $SdkRoot
$env:ANDROID_SDK_ROOT = $SdkRoot
if (-not $env:ANDROID_USER_HOME) { $env:ANDROID_USER_HOME = Join-Path $SdkRoot '.android' }
$env:ANDROID_AVD_HOME = Join-Path $env:ANDROID_USER_HOME 'avd'
if (-not $env:JAVA_HOME) { $env:JAVA_HOME = 'C:\Program Files\OpenLogic\jdk-21.0.7.6-hotspot' }
if (-not $env:GRADLE_USER_HOME) { $env:GRADLE_USER_HOME = 'F:\dev\gradle\home' }

$Adb = Join-Path $SdkRoot 'platform-tools\adb.exe'
$EmulatorExe = Join-Path $SdkRoot 'emulator\emulator.exe'
$AvdManager = Join-Path $SdkRoot 'cmdline-tools\latest\bin\avdmanager.bat'
$Ffmpeg = 'C:\ProgramData\chocolatey\bin\ffmpeg.exe'

$DefaultAvd = 'pixel9pro_test'
$DefaultPort = 5554
$RunsDir = Join-Path $RepoRoot 'runs\emulator'
$MockPkg = 'com.wazeomatic.mockwaze'
$DetectorPkg = 'com.wazeomatic.detector'

New-Item -ItemType Directory -Force -Path $env:ANDROID_AVD_HOME | Out-Null
New-Item -ItemType Directory -Force -Path $RunsDir | Out-Null

function Get-Serial([int]$Port) { "emulator-$Port" }

# Run adb without PowerShell 5.1 turning stderr lines into errors.
# Usage: Invoke-Adb <serial or ''> <adb args...>. Uses $args (no param block) so adb flags
# like -a / -d / -p are passed through instead of being bound as PowerShell parameters.
function Invoke-Adb {
    $serial = $args[0]
    $all = @()
    if ($serial) { $all += @('-s', "$serial") }
    if ($args.Count -gt 1) { $all += @($args[1..($args.Count - 1)] | ForEach-Object { "$_" }) }
    $old = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try { $out = & $Adb @all 2>&1 | ForEach-Object { "$_" } }
    finally { $ErrorActionPreference = $old }
    $script:AdbExit = $LASTEXITCODE
    return $out
}

function Test-DeviceOnline([string]$Serial) {
    $lines = Invoke-Adb '' devices
    return [bool]($lines | Where-Object { $_ -match "^$([regex]::Escape($Serial))\s+device$" })
}
