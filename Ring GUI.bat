@echo off
rem Decoder Ring GUI launcher. The GUI itself owns the island lifecycle (it checks
rem 127.0.0.1:11436 and cold-starts "Start Ring Island.bat" minimized if it is down) - this
rem bat deliberately does NOT pre-start the island: one starter, no bind race.
cd /d "%~dp0"
start "" pythonw gui.py
