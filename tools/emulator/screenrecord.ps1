<#
.SYNOPSIS
  Record the emulator screen for N seconds (max 180, Android limit) into runs\emulator\.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\emulator\screenrecord.ps1 -Seconds 20 -Name macro_run
#>
param(
    [int]$Seconds = 20,
    [string]$Name = '',
    [string]$OutDir = '',
    [int]$Port = 5554,
    [string]$Size = '720x1606'   # half of 1280x2856 keeps files small and the encoder happy
)
. "$PSScriptRoot\_env.ps1"
$serial = Get-Serial $Port
if (-not $Name) { $Name = 'rec_' + (Get-Date -Format 'yyyyMMdd_HHmmss') }
if (-not $OutDir) { $OutDir = $RunsDir }
New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
$Seconds = [Math]::Min([Math]::Max($Seconds, 1), 180)
$remote = "/sdcard/$Name.mp4"
Write-Host "Recording $Seconds s..."
Invoke-Adb $serial shell screenrecord --time-limit $Seconds --size $Size $remote | Write-Host
Start-Sleep -Seconds 1
$local = Join-Path $OutDir "$Name.mp4"
Invoke-Adb $serial pull $remote $local | Out-Null
Invoke-Adb $serial shell rm $remote | Out-Null
Write-Host $local
