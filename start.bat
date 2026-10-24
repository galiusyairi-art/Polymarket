@echo off
REM Windows: double-click to start the bot. The dashboard opens in your browser.
REM To carry on from the server: drag the export ZIP (Settings - System - download) onto this file.
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
if "%~1"=="" (
  python -m polybot
) else (
  python -m polybot --import "%~1"
)
pause
