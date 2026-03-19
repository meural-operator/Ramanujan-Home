@echo off
setlocal DisableDelayedExpansion
cd /d "%~dp0"

echo ==================================================
echo       Ramanujan@Home - One-Click Compute Node      
echo ==================================================

set "PYTHON_CMD="

:: 1. Check Local AppData Python installation
if exist "%LOCALAPPDATA%\RamanujanPython\python.exe" (
    "%LOCALAPPDATA%\RamanujanPython\python.exe" -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 13 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=%LOCALAPPDATA%\RamanujanPython\python.exe"
)

:: 2. Check system default Python
if "%PYTHON_CMD%"=="" (
    python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor == 13 else 1)" >nul 2>&1
    if not errorlevel 1 set "PYTHON_CMD=python"
)

:: 3. Download and Install if missing
if not "%PYTHON_CMD%"=="" goto :INSTALL_DONE

echo [*] Python 3.13 not found on system.
if not exist "python-installer.exe" (
    echo [*] Downloading Python 3.13.0 directly from python.org...
    powershell -Command "Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.0/python-3.13.0-amd64.exe' -OutFile 'python-installer.exe'"
)

echo [*] Installing isolated Python 3.13 silently to LocalAppData without Admin rights...
start /wait python-installer.exe /quiet InstallAllUsers=0 Include_launcher=0 Include_pip=1 PrependPath=0 TargetDir="%LOCALAPPDATA%\RamanujanPython"

if exist "%LOCALAPPDATA%\RamanujanPython\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\RamanujanPython\python.exe"
    if exist "python-installer.exe" del python-installer.exe
) else (
    echo [!] Python 3.13 isolated installation failed.
    pause
    exit /b 1
)

:INSTALL_DONE
echo [*] Enforcing Python Runtime: "%PYTHON_CMD%"

:: 4. Fast-check and Setup Virtual Environment
if not exist "%USERPROFILE%\.ramanujan_env\Scripts\python.exe" (
    echo [*] First-time standalone setup detected. Bootstrapping AI Environment...
    "%PYTHON_CMD%" setup\autoinstaller.py
    if errorlevel 1 (
        echo [!] Autoinstaller failed. Please check your system Python installation.
        pause
        exit /b 1
    )
)

echo [*] Launching Client Application...
"%USERPROFILE%\.ramanujan_env\Scripts\python.exe" ramanujan_client.py

pause
