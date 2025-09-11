@echo off
setlocal enabledelayedexpansion

:: Get the full git commit hash and store the first 8 characters in a variable
for /f "delims=" %%i in ('git rev-parse HEAD') do (
    set "FULL_HASH=%%i"
    set "SHORT_HASH=!FULL_HASH:~0,8!"
)

:: zip data
powershell Compress-Archive -Path "Antares/input, Pre-Processing/Data, Pre-Processing/Output" -DestinationPath "BAF-Data_EU-system_%SHORT_HASH%.zip"