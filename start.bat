@echo off
setlocal
chcp 65001 >nul
cd /d "%~dp0"
if errorlevel 1 goto failed
set "PYTHONUTF8=1"
set "PYTHONUNBUFFERED=1"
if exist ".venv\Scripts\python.exe" goto check_python
where py.exe >nul 2>&1
if errorlevel 1 goto use_python
py -3 -m venv ".venv"
if errorlevel 1 goto failed
goto check_python
:use_python
where python.exe >nul 2>&1
if errorlevel 1 goto python_missing
python -m venv ".venv"
if errorlevel 1 goto failed
goto check_python
:check_python
".venv\Scripts\python.exe" -X utf8 -c "import sys; sys.exit(sys.version_info < (3, 11))"
if errorlevel 1 goto python_missing
".venv\Scripts\python.exe" -X utf8 -c "import importlib.util,sys; names=('flask','waitress','chromadb','jieba','rank_bm25','sentence_transformers','requests','markdown_it','bleach','filelock','onnxruntime','pymysql','cryptography','argon2','pypdf','PIL','docx','openpyxl','xlrd','charset_normalizer'); sys.exit(any(importlib.util.find_spec(name) is None for name in names))" >nul 2>&1
if errorlevel 1 goto install_dependencies
goto run
:install_dependencies
echo Installing project dependencies...
".venv\Scripts\python.exe" -X utf8 -m pip install -r requirements.txt
if errorlevel 1 goto failed
:run
if not exist "config_local.py" goto launch
".venv\Scripts\python.exe" -X utf8 -c "import config_local,sys; sys.exit(not bool(getattr(config_local,'DEEPSEEK_API_KEY','').strip()))" >nul 2>&1
if not errorlevel 1 set "DEEPSEEK_API_KEY="
:launch
echo Starting Lingxing RAG...
".venv\Scripts\python.exe" -X utf8 run.py %*
if errorlevel 1 goto failed
exit /b 0
:python_missing
echo Python 3.11 or newer is required. Please check your Python installation.
goto failed
:failed
echo.
echo Startup failed. See the error above.
if "%LINGXING_NO_PAUSE%"=="1" exit /b 1
pause
exit /b 1
