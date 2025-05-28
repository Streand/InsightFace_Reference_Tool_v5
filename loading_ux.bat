@echo off
setlocal EnableDelayedExpansion

set "TOTAL=20"
echo Loading...
for /L %%i in (1,1,%TOTAL%) do (
    set /a percent=%%i*100/%TOTAL%
    set "bar="
    for /L %%j in (1,1,%%i) do (
        set "bar=!bar!#"
    )
    rem Clear the current line by outputting a carriage return
    <nul set /p="[%bar%] !percent!%% loaded..."
    timeout /t 1 >nul
    rem Return cursor to beginning of the line
    <nul set /p="[0G"
)
echo.
echo Loading complete!
pause