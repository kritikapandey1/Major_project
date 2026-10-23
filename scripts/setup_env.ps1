# DeepFusionGuard - local (CPU-only) environment setup for Windows 10/11.
# Windows equivalent of scripts/setup_env.sh. Safe to re-run: finished steps are skipped.
#
# Usage, from the repo root in PowerShell:
#   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
# or double-click scripts\setup_env.bat
#
# Installs (via winget, only if missing): Git, FFmpeg, Microsoft Visual C++ runtime, uv.
# Never installs CUDA, ROCm or DirectML: this project runs CPU-only locally.

$ErrorActionPreference = "Stop"
$PyVersion = "3.12"
$VenvDir   = ".venv"
$VenvPy    = Join-Path $VenvDir "Scripts\python.exe"

function Test-Cmd([string]$Name) {
    return [bool](Get-Command $Name -ErrorAction SilentlyContinue)
}

function Update-SessionPath {
    # Pick up PATH changes made by installers without opening a new terminal.
    $machine = [Environment]::GetEnvironmentVariable("Path", "Machine")
    $user    = [Environment]::GetEnvironmentVariable("Path", "User")
    $env:Path = "$machine;$user;$env:USERPROFILE\.local\bin"
}

function Invoke-Checked {
    # Run a native command and stop if it fails (Windows PowerShell 5.1 does not do this itself).
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { throw "'$Exe $($Arguments -join ' ')' failed with exit code $LASTEXITCODE" }
}

function Install-WingetPackage([string]$Id, [string]$What) {
    if (-not (Test-Cmd "winget")) {
        throw "winget not found. Install '$What' manually (or install 'App Installer' from the Microsoft Store), then re-run."
    }
    Write-Host "Installing $What with winget ..."
    Invoke-Checked "winget" @("install", "--id", $Id, "-e", "--source", "winget",
                              "--accept-source-agreements", "--accept-package-agreements")
    Update-SessionPath
}

if (-not (Test-Path "requirements.txt")) {
    throw "requirements.txt not found. Run this script from the repo root (the Major_project folder)."
}

Write-Host "== 1. System tools =="
if (-not (Test-Cmd "git"))    { Install-WingetPackage "Git.Git" "Git" }
if (-not (Test-Cmd "ffmpeg")) { Install-WingetPackage "Gyan.FFmpeg" "FFmpeg" }
Invoke-Checked "git" @("--version")
if (Test-Cmd "ffmpeg") {
    (& ffmpeg -version | Select-Object -First 1)
} else {
    Write-Warning "ffmpeg was installed but is not on PATH in this window yet. Open a new PowerShell and re-run this script."
}

Write-Host "== 2. Microsoft Visual C++ runtime (needed by PyTorch on Windows) =="
$vcDll = Join-Path $env:SystemRoot "System32\vcruntime140_1.dll"
if (Test-Path $vcDll) {
    Write-Host "VC++ runtime found."
} else {
    Install-WingetPackage "Microsoft.VCRedist.2015+.x64" "Microsoft Visual C++ 2015-2022 Redistributable (x64)"
}

Write-Host "== 3. uv (Python + package manager) =="
if (-not (Test-Cmd "uv")) {
    Write-Host "Installing uv with the official installer ..."
    Invoke-RestMethod https://astral.sh/uv/install.ps1 | Invoke-Expression
    Update-SessionPath
}
Invoke-Checked "uv" @("--version")

Write-Host "== 4. Virtual env with Python $PyVersion =="
if (-not (Test-Path $VenvPy)) {
    Invoke-Checked "uv" @("venv", "--python", $PyVersion, $VenvDir)
}
Invoke-Checked $VenvPy @("--version")

Write-Host "== 5. Packages (CPU-only torch) =="
Invoke-Checked "uv" @("pip", "install", "--python", $VenvPy, "-r", "requirements.txt", "--torch-backend", "cpu")

Write-Host "== 6. Environment report =="
& $VenvPy "scripts\env_report.py" "--device" "auto"
exit $LASTEXITCODE
