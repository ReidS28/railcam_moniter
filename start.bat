@echo off

REM Change to the directory where the script and virtual environment are located
cd /d "C:\Users\straw\Desktop\Code\railstream_moniter"

REM Activate the virtual environment
call yolovenv\Scripts\activate.bat

REM Run the Python script
python src\railstream_moniter.py

REM Optional: Pause to see any output before the window closes
pause