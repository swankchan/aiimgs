@echo off
call C:\Users\%USERNAME%\anaconda3\Scripts\activate.bat imgregcon
if errorlevel 1 call C:\ProgramData\anaconda3\Scripts\activate.bat imgregcon
python -m pip install python-dotenv psycopg2-binary
python db_config.py
pause
