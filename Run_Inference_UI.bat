@echo off
setlocal

REM Check if venv_ui folder exists in the root directory
if not exist "%~dp0venv_ui\" (
    REM Print the step to the terminal
    echo Creating virtual environment...

    REM Create a virtual environment named "venv_ui" in the root directory
    python -m venv "%~dp0venv_ui"
)

REM Print the step to the terminal
echo Activating the virtual environment...

REM Activate the virtual environment
call "%~dp0venv_ui\Scripts\activate.bat"

REM Check if the required packages are already installed
echo Checking if required packages are installed...
pip show gradio >nul 2>&1
if %errorlevel% equ 0 (
    echo Required packages are already installed. Skipping requirements installation.
) else (
    REM Print the step to the terminal
    echo Installing requirements...

    REM Install requirements inside the virtual environment
    pip install -r "%~dp0Scripts\requirements_ui.txt"
)

REM Print the step to the terminal
echo Running Python script...

REM Run your Python file here
python "%~dp0Scripts\Run_Inference_UI.py"

REM Print the step to the terminal
echo Deactivating the virtual environment...

REM Deactivate the virtual environment
call "%~dp0venv_ui\Scripts\deactivate.bat"

REM Print the step to the terminal
echo Script execution complete.

endlocal

pause /k
