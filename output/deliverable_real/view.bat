@echo off
cd /d "%~dp0"
start "" http://127.0.0.1:8767/viewer.html
python -m http.server 8767 --bind 127.0.0.1 --directory "%~dp0."
