@echo off
setlocal
set "ROOT_DIR=%~dp0"
call "%ROOT_DIR%run.bat" %*
exit /b %errorlevel%
