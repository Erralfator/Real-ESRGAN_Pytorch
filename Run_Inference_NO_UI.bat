@echo off
setlocal

REM Check if venv_no_ui folder exists in the root directory
if not exist "%~dp0venv_no_ui\" (
    REM Print the step to the terminal
    echo Creating virtual environment...

    REM Create a virtual environment named "venv_no_ui" in the root directory
    python -m venv "%~dp0venv_no_ui"
)

REM Print the step to the terminal
echo Activating the virtual environment...

REM Activate the virtual environment
call "%~dp0venv_no_ui\Scripts\activate.bat"

REM Check if the required packages are already installed
echo Checking if required packages are installed...
pip show gradio >nul 2>&1
if %errorlevel% equ 0 (
    echo Required packages are already installed. Skipping requirements installation.
) else (
    REM Print the step to the terminal
    echo Installing requirements...

    REM Install requirements inside the virtual environment
    pip install -r "%~dp0Scripts\requirements_no_ui.txt"
)

REM Print the step to the terminal
echo Running Python script...

REM Ask the user for the image file name or 'ALL'
set "image_name="
set /p "image_name=Enter the filename of the image/video (with extension) in the 'inputs' folder to upscale: "

REM Set the image file path
set "image_path=%~dp0inputs\%image_name%"

REM Ask for the upscale factor
set /p upscale_factor="Enter the upscale factor (2, 4, or 8): "

REM Run the Python script with the file path and upscale factor
python "%~dp0Scripts\Infer_NO_UI.py" --file "%image_path%" --size "%upscale_factor%"


REM Print the step to the terminal
echo Deactivating the virtual environment...

REM Deactivate the virtual environment
call "%~dp0venv_no_ui\Scripts\deactivate.bat"

REM Print the step to the terminal
echo Script execution complete.

endlocal

pause /k
