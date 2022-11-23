@echo OFF

ECHO "Choose an option .."
ECHO "install to install library (install)"
ECHO "start to run code (y)"


SET /p option=Choose one option-

IF %option%==install python -m venv compV && pause && compV\Scripts\activate && pause && pip install -r requirements.txt && pause && python gui.py
IF %option%==y py gui.py


PAUSE