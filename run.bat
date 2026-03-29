@echo off
setlocal

set "ROOT_DIR=%~dp0"
set "BOOTSTRAP_SCRIPT=%ROOT_DIR%scripts\bootstrap_windows.ps1"
set "START_SCRIPT=%ROOT_DIR%scripts\start_webui.ps1"

if not exist "%BOOTSTRAP_SCRIPT%" (
    echo bootstrap_windows.ps1 was not found: %BOOTSTRAP_SCRIPT%
    exit /b 1
)

if not exist "%START_SCRIPT%" (
    echo start_webui.ps1 was not found: %START_SCRIPT%
    exit /b 1
)

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%BOOTSTRAP_SCRIPT%" %*
if errorlevel 1 exit /b %errorlevel%

powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "%START_SCRIPT%" %*
exit /b %errorlevel%
