@echo off
:: ══════════════════════════════════════════════════════════════════════════════
:: KHIDMETI — shim CMD → PowerShell
:: Toute la logique vit dans khidmeti.ps1 (parité Makefile v14.5).
:: Usage identique :  khidmeti.bat start | stop | check | prod-start | ...
:: ══════════════════════════════════════════════════════════════════════════════
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0khidmeti.ps1" %*
exit /b %ERRORLEVEL%
