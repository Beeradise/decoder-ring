@echo off
setlocal
REM ring.bat - ensure the Ring Ollama island (127.0.0.1:11436) is up, then run ring_cli.
netstat -ano | findstr /R /C:":11436 .*LISTENING" >nul 2>&1
if not errorlevel 1 goto run
echo [ring] island not listening on 11436 - starting minimized...
start "Ring Ollama Island" /MIN cmd /c ""%~dp0Start Ring Island.bat""
set TRIES=0
:wait
curl -s -o nul --max-time 2 http://127.0.0.1:11436/api/tags >nul 2>&1
if not errorlevel 1 goto run
set /a TRIES+=1
if %TRIES% geq 20 (
    echo [ring] island did not come up within ~20s - check the Start Ring Island window
    exit /b 1
)
REM ~1s sleep via ping (robust: timeout.exe breaks under a Git-Bash PATH / redirected stdin)
ping -n 2 127.0.0.1 >nul
goto wait
:run
python "%~dp0ring_cli.py" %*
