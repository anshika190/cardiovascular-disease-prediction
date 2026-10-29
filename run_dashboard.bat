@echo off
echo ========================================================
echo   Launching Health Analytics & EDA Dashboard
echo ========================================================
echo.
echo Starting Streamlit analytics dashboard...
echo Dashboard will open automatically in your browser at: http://localhost:8501
echo Press Ctrl+C in this terminal to stop the dashboard.
echo.
python -m streamlit run dashboard.py
pause
