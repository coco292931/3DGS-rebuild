@echo off
setlocal
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" 1>nul
if errorlevel 1 ( echo VCVARS_FAILED & exit /b 1 )
echo --- cl.exe ---
where cl
echo --- running toolchain probe ---
python "C:\Users\tonyp\Downloads\3dgs\_env_probe\cuda_toolchain_probe.py"
echo PROBE_EXIT=%errorlevel%
