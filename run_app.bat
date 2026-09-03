@echo off
setlocal

REM ==========================================================
REM Exercise 1.2 - Intelligent Vehicle Counting
REM Streamlit launcher
REM ==========================================================

cd /d "%~dp0"

title Exercise 1.2 - Downtown Vehicle Counter

echo.
echo ==========================================================
echo   Exercise 1.2 - Downtown Vehicle Counter
echo ==========================================================
echo.
echo Working directory:
echo   %CD%
echo.

REM Make Python output appear immediately in this console.
set PYTHONUNBUFFERED=1

REM ----------------------------------------------------------
REM Verify Python
REM ----------------------------------------------------------

python --version >nul 2>&1

if errorlevel 1 (
    echo ERROR: Python was not found.
    echo.
    echo Activate your Python/Conda environment first,
    echo then run this batch file again.
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------------------------
REM Verify Streamlit
REM ----------------------------------------------------------

python -c "import streamlit" >nul 2>&1

if errorlevel 1 (
    echo ERROR: Streamlit is not installed in the active Python environment.
    echo.
    echo Install it with:
    echo   python -m pip install streamlit
    echo.
    pause
    exit /b 1
)

REM ----------------------------------------------------------
REM Verify application files
REM ----------------------------------------------------------

if not exist "app.py" (
    echo ERROR: app.py was not found in:
    echo   %CD%
    echo.
    pause
    exit /b 1
)

if not exist "vehicle_counting_pipeline.py" (
    echo ERROR: vehicle_counting_pipeline.py was not found in:
    echo   %CD%
    echo.
    pause
    exit /b 1
)

echo Starting Streamlit...
echo.
echo Local URL:
echo   http://localhost:8501
echo.
echo To stop the application, press CTRL+C once and allow
echo Streamlit a few seconds to shut down cleanly.
echo.
echo ==========================================================
echo.

python -m streamlit run app.py --server.port 8501

set STREAMLIT_EXIT_CODE=%ERRORLEVEL%

echo.
echo ==========================================================

if "%STREAMLIT_EXIT_CODE%"=="0" (
    echo Streamlit stopped normally.
) else (
    echo Streamlit exited with code %STREAMLIT_EXIT_CODE%.
)

echo ==========================================================
echo.
echo Press any key to close this window.
pause >nul

endlocal
