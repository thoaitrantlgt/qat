@echo off
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0export_policy.ps1" %*
