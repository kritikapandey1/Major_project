@echo off
REM DeepFusionGuard - Windows setup. Double-click this file, or run it from cmd.
REM It runs scripts\setup_env.ps1 from the repo root, bypassing the PowerShell script policy for this run only.
cd /d "%~dp0.."
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup_env.ps1"
pause
