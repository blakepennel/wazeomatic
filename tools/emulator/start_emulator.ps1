<#
.SYNOPSIS
  Start the pixel9pro_test emulator on port 5554 (serial emulator-5554) in the background.
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File tools\emulator\start_emulator.ps1
  powershell -ExecutionPolicy Bypass -File tools\emulator\start_emulator.ps1 -Headless
  powershell -ExecutionPolicy Bypass -File tools\emulator\start_emulator.ps1 -Webcam   # back cam = webcam0 (OBS)
#>
param(
    [string]$Avd = 'pixel9pro_test',
    [int]$Port = 5554,
    [switch]$Headless,          # -no-window
    [string]$Gpu = 'auto',      # auto | host | swiftshader_indirect | ...
    [switch]$Webcam,            # -camera-back webcam0 for this run only
    [switch]$WipeData,
    [switch]$ColdBoot,          # ignore the quickboot snapshot
    [switch]$Wait               # also run wait_for_boot.ps1
)
. "$PSScriptRoot\_env.ps1"

# Never touch the other agent's emulator (wazeomatic_dev on 5560): we only ever use our own port.
$serial = Get-Serial $Port
if (Test-DeviceOnline $serial) {
    Write-Host "$serial is already running, not starting another one."
} else {
    $emuArgs = @('-avd', $Avd, '-port', "$Port", '-gpu', $Gpu, '-no-boot-anim', '-no-audio', '-netdelay', 'none', '-netspeed', 'full')
    if ($Headless) { $emuArgs += '-no-window' }
    if ($Webcam) { $emuArgs += @('-camera-back', 'webcam0') }
    if ($WipeData) { $emuArgs += '-wipe-data' }
    if ($ColdBoot) { $emuArgs += '-no-snapshot-load' }
    $log = Join-Path $RunsDir "emulator_$Port.log"
    $errLog = Join-Path $RunsDir "emulator_$Port.err.log"
    Write-Host "Starting: emulator $($emuArgs -join ' ')"
    Write-Host "Emulator log: $log"
    $p = Start-Process -FilePath $EmulatorExe -ArgumentList $emuArgs -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $log -RedirectStandardError $errLog
    Write-Host "emulator pid $($p.Id)"
}

if ($Wait) {
    & "$PSScriptRoot\wait_for_boot.ps1" -Port $Port
}
