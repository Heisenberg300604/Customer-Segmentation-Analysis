@echo off
REM One-command setup + run for Windows.
REM   run.bat          set up (first time only) and open the web app
REM   run.bat test     run all tests
REM   run.bat train    retrain the model and regenerate artifacts + docs
REM Needs Python 3.11 or 3.12 from https://www.python.org/downloads/
cd /d "%~dp0"
set PY=.venv\Scripts\python.exe

if not exist %PY% (
  echo ==^> Creating virtual environment
  py -3.11 -m venv .venv 2>nul || py -3.12 -m venv .venv 2>nul || python -m venv .venv
)
%PY% -c "import streamlit, sklearn, plotly" >nul 2>&1
if errorlevel 1 (
  echo ==^> Installing packages ^(first run only^)
  %PY% -m pip install --upgrade pip -q
  %PY% -m pip install -r requirements.txt || exit /b 1
)
if "%1"=="train" ( %PY% -m src.train & exit /b )
if not exist artifacts\kmeans_model.joblib %PY% -m src.train
if "%1"=="test" ( %PY% -m pytest -v & exit /b )
echo ==^> Starting the app at http://localhost:8501
%PY% -m streamlit run app\app.py
