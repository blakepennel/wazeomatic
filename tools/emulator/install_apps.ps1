<#
.SYNOPSIS
  Install mock Waze (and the detector APK if it has been built) on the test emulator.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\emulator\install_apps.ps1 -Build
#>
param(
    [int]$Port = 5554,
    [switch]$Build   # run gradlew assembleDebug for mock Waze first
)
. "$PSScriptRoot\_env.ps1"
$serial = Get-Serial $Port
if (-not (Test-DeviceOnline $serial)) { throw "$serial is not online. Run start_emulator.ps1 -Wait first." }

$mockDir = Join-Path $RepoRoot 'android\mockwaze'
$mockApk = Join-Path $mockDir 'app\build\outputs\apk\debug\app-debug.apk'
if ($Build -or -not (Test-Path $mockApk)) {
    Write-Host 'Building mock Waze...'
    Push-Location $mockDir
    try {
        $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'
        & .\gradlew.bat assembleDebug --console=plain 2>&1 | ForEach-Object { "$_" } | Select-Object -Last 5 | Write-Host
        $code = $LASTEXITCODE
        $ErrorActionPreference = $old
        if ($code -ne 0) { throw "mock Waze build failed ($code)" }
    } finally { Pop-Location }
}

$ok = $true
Write-Host "Installing $mockApk"
$out = Invoke-Adb $serial install -r -g $mockApk
$out | Write-Host
if (-not ($out -match 'Success')) { $ok = $false }

$detApks = @(Get-ChildItem -Path (Join-Path $RepoRoot 'android\detector\app\build\outputs\apk\debug') -Filter *.apk -ErrorAction SilentlyContinue)
if ($detApks.Count -gt 0) {
    $apk = ($detApks | Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    Write-Host "Installing detector $apk"
    $out = Invoke-Adb $serial install -r -g $apk
    $out | Write-Host
    if (-not ($out -match 'Success')) { $ok = $false }
} else {
    Write-Host 'Detector APK not built yet (android\detector\app\build\outputs\apk\debug\*.apk), skipping.'
}
if (-not $ok) { exit 1 }
