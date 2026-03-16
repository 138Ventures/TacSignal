@echo off
REM ═══════════════════════════════════════════════════════════════
REM TacSignal — One-Click Data Update (Windows)
REM ═══════════════════════════════════════════════════════════════
REM Double-click this file to run.
REM ═══════════════════════════════════════════════════════════════

cd /d "%~dp0"

echo.
echo ================================================================
echo        TacSignal — Monthly Data Updater
echo ================================================================
echo.

REM ─── Step 1: Check Python ───
where python >nul 2>&1
if %errorlevel% neq 0 (
    where python3 >nul 2>&1
    if %errorlevel% neq 0 (
        echo X Python not found!
        echo.
        echo    Install Python from: https://www.python.org/downloads/
        echo    IMPORTANT: Check "Add Python to PATH" during install!
        echo.
        pause
        exit /b 1
    )
    set PYTHON=python3
) else (
    set PYTHON=python
)

%PYTHON% --version
echo.

REM ─── Step 2: Install packages ───
echo Checking required packages...
%PYTHON% -m pip install yfinance pandas numpy fredapi --quiet 2>nul
echo Done.
echo.

REM ─── Step 3: FRED API Key ───
set "KEY_FILE=%USERPROFILE%\.tacsignal_fred_key"
set "FRED_KEY="

if exist "%KEY_FILE%" (
    set /p FRED_KEY=<"%KEY_FILE%"
    echo Using saved FRED API key.
) else if defined FRED_API_KEY (
    set "FRED_KEY=%FRED_API_KEY%"
    echo Using FRED API key from environment.
) else (
    echo ================================================================
    echo   FRED API Key Setup (one-time only)
    echo ================================================================
    echo.
    echo   You need a free FRED API key for macro data.
    echo   Get one here (takes 30 seconds):
    echo.
    echo   https://fred.stlouisfed.org/docs/api/api_key.html
    echo.
    echo   1. Click 'Request or view your API keys'
    echo   2. Sign in (or create a free account)
    echo   3. Copy the 32-character key
    echo.
    set /p FRED_KEY="  Paste your FRED API key here: "
    echo.

    if defined FRED_KEY (
        echo %FRED_KEY%>"%KEY_FILE%"
        echo   Key saved. Won't ask again.
    ) else (
        echo   No key entered — running WITHOUT FRED data.
    )
)

REM ─── Step 4: Run pipeline ───
echo.
echo ================================================================
echo   Running pipeline...
echo ================================================================
echo.

for /f "tokens=1-3 delims=/ " %%a in ('date /t') do set TODAY=%%c-%%a-%%b
set "OUTPUT_FILE=tacsignal-%TODAY%.json"

if defined FRED_KEY (
    %PYTHON% tacsignal_data_pipeline.py --output "%OUTPUT_FILE%" --fred-key "%FRED_KEY%"
) else (
    %PYTHON% tacsignal_data_pipeline.py --output "%OUTPUT_FILE%"
)

REM ─── Step 5: Done ───
echo.
echo ================================================================
echo   DONE!
echo ================================================================
echo.
echo   Output file: %CD%\%OUTPUT_FILE%
echo.
echo   Next step:
echo   1. Open tactical-signal-v2.html in your browser
echo   2. Click 'Import' (top right)
echo   3. Select %OUTPUT_FILE%
echo.
echo   Opening dashboard...
start "" "tactical-signal-v2.html"
echo.
pause
